const std = @import("std");
const log = std.log.scoped(.zignanogpt_eval_bundle);
const mod = @import("module.zig");

/// How a CORE task is scored (`icl_task_type`).
pub const CoreTaskType = enum { multiple_choice, schema, language_modeling };

/// One CORE task from `core.yaml`, with its random baseline from `eval_meta_data.csv`.
pub const CoreTask = struct {
    label: []const u8,
    dataset_uri: []const u8,
    task_type: CoreTaskType,
    num_fewshot: usize,
    continuation_delimiter: []const u8 = " ",
    /// Percent.
    random_baseline: f64 = 0,
};

/// nanochat's CORE eval bundle (`eval_bundle.zip`: `core.yaml`, `eval_meta_data.csv`,
/// `eval_data/*.jsonl`), found in this port's or nanochat's base dir, or
/// downloaded and extracted into `<base>/eval_bundle` on first use.
pub const EvalBundle = struct {
    const Self = @This();

    pub const url = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip";

    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    /// The `eval_bundle` directory.
    dir: []const u8,
    tasks: []CoreTask,

    /// Finds or downloads the bundle, then reads its task list.
    ///
    /// Parameters:
    /// - `allocator`: owns the bundle.
    /// - `io`: file and network access.
    /// - `config`: the base directories.
    /// - `out`: download progress, or null.
    ///
    /// Return: the bundle; download, storage and format errors.
    pub fn open(allocator: std.mem.Allocator, io: std.Io, config: *const mod.Config, out: ?*std.Io.Writer) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const storage = mod.Storage.init(allocator, io);
        for ([_][]const u8{ config.base_dir, config.nanochat_dir }) |root| {
            const dir = try std.fs.path.join(allocator, &.{ root, "eval_bundle" });
            defer allocator.free(dir);
            const yaml = try std.fs.path.join(allocator, &.{ dir, "core.yaml" });
            defer allocator.free(yaml);
            if (try storage.exists(yaml)) return load(allocator, storage, dir);
        }
        const zip_path = try std.fs.path.join(allocator, &.{ config.base_dir, "eval_bundle.zip" });
        defer allocator.free(zip_path);
        if (!try storage.exists(zip_path)) {
            if (out) |w| {
                try w.print("Downloading {s} ...\n", .{url});
                try w.flush();
            }
            var node = try mod.zigstorage.Node.init(allocator, io, .empty, url);
            defer node.deinit();
            const body = try node.read(.all);
            defer allocator.free(body);
            try storage.write(zip_path, body);
        }
        try extract(allocator, io, zip_path, config.base_dir);
        const dir = try std.fs.path.join(allocator, &.{ config.base_dir, "eval_bundle" });
        defer allocator.free(dir);
        return load(allocator, storage, dir);
    }

    /// Unpacks a bundle zip into `root` (members keep their `eval_bundle/` prefix).
    /// `core.yaml` is written last: its presence marks a complete bundle.
    ///
    /// Parameters:
    /// - `allocator`: scratch.
    /// - `io`: file access.
    /// - `zip_path`: the archive.
    /// - `root`: the destination directory.
    ///
    /// Return: nothing; zip and storage errors.
    pub fn extract(allocator: std.mem.Allocator, io: std.Io, zip_path: []const u8, root: []const u8) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const storage = mod.Storage.init(allocator, io);
        const location = try storage.absolute(zip_path);
        defer allocator.free(location);
        var zip = try mod.ZipArchive.open(allocator, io, location);
        defer zip.deinit();
        var marker: ?mod.ZipEntry = null;
        for (zip.entries) |entry| {
            if (std.mem.endsWith(u8, entry.name, "/")) continue;
            if (std.mem.indexOf(u8, entry.name, "..") != null or std.fs.path.isAbsolute(entry.name)) return error.InvalidZip;
            if (std.mem.endsWith(u8, entry.name, "/core.yaml")) {
                marker = entry;
                continue;
            }
            try writeMember(allocator, storage, &zip, entry, root);
        }
        try writeMember(allocator, storage, &zip, marker orelse return error.InvalidBundle, root);
    }

    fn writeMember(allocator: std.mem.Allocator, storage: mod.Storage, zip: *mod.ZipArchive, entry: mod.ZipEntry, root: []const u8) !void {
        const bytes = try zip.read(entry);
        defer allocator.free(bytes);
        const path = try std.fs.path.join(allocator, &.{ root, entry.name });
        defer allocator.free(path);
        try storage.write(path, bytes);
    }

    /// Reads `core.yaml` and the baselines of an extracted bundle.
    ///
    /// Parameters:
    /// - `allocator`: owns the bundle.
    /// - `storage`: file access.
    /// - `dir`: the `eval_bundle` directory.
    ///
    /// Return: the bundle; storage and format errors.
    pub fn load(allocator: std.mem.Allocator, storage: mod.Storage, dir: []const u8) !Self {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const a = arena.allocator();
        // The tasks borrow their strings from the YAML text, kept in the arena.
        const raw = try storage.read(try std.fs.path.join(a, &.{ dir, "core.yaml" }));
        defer allocator.free(raw);
        const tasks = try parseCoreYaml(a, try a.dupe(u8, raw));
        const csv = try storage.read(try std.fs.path.join(a, &.{ dir, "eval_meta_data.csv" }));
        defer allocator.free(csv);
        try applyBaselines(a, csv, tasks);
        return Self{ .allocator = allocator, .arena = arena, .dir = try a.dupe(u8, dir), .tasks = tasks };
    }

    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    /// A task's examples, one JSON object per line.
    ///
    /// Parameters:
    /// - `self`: the bundle.
    /// - `arena`: holds the parsed values.
    /// - `storage`: file access.
    /// - `task`: the task.
    ///
    /// Return: the examples in file order; storage and JSON errors.
    pub fn readExamples(self: *const Self, arena: std.mem.Allocator, storage: mod.Storage, task: CoreTask) ![]std.json.Value {
        const path = try std.fs.path.join(arena, &.{ self.dir, "eval_data", task.dataset_uri });
        const text = try storage.read(path);
        defer storage.allocator.free(text);
        var examples: std.ArrayList(std.json.Value) = .empty;
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;
            try examples.append(arena, try std.json.parseFromSliceLeaky(std.json.Value, arena, trimmed, .{ .allocate = .alloc_always }));
        }
        return examples.items;
    }

    /// The subset of YAML `core.yaml` uses: `icl_tasks:` then `-` items of
    /// `key: value` lines (plain scalars, `[n]` lists, double-quoted strings).
    fn parseCoreYaml(a: std.mem.Allocator, text: []const u8) ![]CoreTask {
        var tasks: std.ArrayList(CoreTask) = .empty;
        const Fields = struct { label: ?[]const u8 = null, uri: ?[]const u8 = null, kind: ?CoreTaskType = null, shots: ?usize = null, delimiter: []const u8 = " " };
        var current: ?Fields = null;
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |raw| {
            const line = std.mem.trim(u8, raw, " \t\r");
            if (line.len == 0 or line[0] == '#' or std.mem.eql(u8, line, "icl_tasks:")) continue;
            if (line[0] == '-') {
                if (current) |f| try tasks.append(a, try finish(f));
                current = .{};
                const rest = std.mem.trim(u8, line[1..], " ");
                if (rest.len == 0) continue;
                return yamlError("inline item");
            }
            const f = if (current) |*c| c else return yamlError("field outside an item");
            const colon = std.mem.indexOfScalar(u8, line, ':') orelse return yamlError("line without a key");
            const key = line[0..colon];
            const value = try scalar(a, std.mem.trim(u8, line[colon + 1 ..], " "));
            if (std.mem.eql(u8, key, "label")) {
                f.label = value;
            } else if (std.mem.eql(u8, key, "dataset_uri")) {
                f.uri = value;
            } else if (std.mem.eql(u8, key, "icl_task_type")) {
                f.kind = std.meta.stringToEnum(CoreTaskType, value) orelse return yamlError("icl_task_type");
            } else if (std.mem.eql(u8, key, "num_fewshot")) {
                const inner = std.mem.trim(u8, value, "[] ");
                const first = std.mem.sliceTo(inner, ',');
                f.shots = std.fmt.parseInt(usize, std.mem.trim(u8, first, " "), 10) catch return yamlError("num_fewshot");
            } else if (std.mem.eql(u8, key, "continuation_delimiter")) {
                f.delimiter = value;
            }
        }
        if (current) |f| try tasks.append(a, try finish(f));
        return tasks.items;
    }

    fn finish(f: anytype) !CoreTask {
        return .{
            .label = f.label orelse return yamlError("task without label"),
            .dataset_uri = f.uri orelse return yamlError("task without dataset_uri"),
            .task_type = f.kind orelse return yamlError("task without icl_task_type"),
            .num_fewshot = f.shots orelse return yamlError("task without num_fewshot"),
            .continuation_delimiter = f.delimiter,
        };
    }

    /// A YAML scalar: double-quoted (with escapes), single-quoted, or plain.
    fn scalar(a: std.mem.Allocator, value: []const u8) ![]const u8 {
        if (value.len >= 2 and value[0] == '\'' and value[value.len - 1] == '\'') return std.mem.replaceOwned(u8, a, value[1 .. value.len - 1], "''", "'");
        if (value.len < 2 or value[0] != '"' or value[value.len - 1] != '"') return value;
        var out: std.ArrayList(u8) = .empty;
        var i: usize = 1;
        while (i < value.len - 1) : (i += 1) {
            const c = value[i];
            if (c != '\\' or i + 1 >= value.len - 1) {
                try out.append(a, c);
                continue;
            }
            i += 1;
            try out.append(a, switch (value[i]) {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '0' => 0,
                '"' => '"',
                '\\' => '\\',
                '/' => '/',
                ' ' => ' ',
                else => return yamlError("unsupported escape"),
            });
        }
        return out.items;
    }

    fn yamlError(what: []const u8) error{InvalidBundle} {
        log.warn("core.yaml: {s}", .{what});
        return error.InvalidBundle;
    }

    /// Sets each task's `Random baseline` from `eval_meta_data.csv` (RFC 4180 quoting).
    fn applyBaselines(a: std.mem.Allocator, csv: []const u8, tasks: []CoreTask) !void {
        var rows: std.ArrayList([]const []const u8) = .empty;
        var fields: std.ArrayList([]const u8) = .empty;
        var field: std.ArrayList(u8) = .empty;
        var quoted = false;
        var i: usize = 0;
        while (i < csv.len) : (i += 1) {
            const c = csv[i];
            if (quoted) {
                if (c == '"') {
                    if (i + 1 < csv.len and csv[i + 1] == '"') {
                        try field.append(a, '"');
                        i += 1;
                    } else quoted = false;
                } else try field.append(a, c);
                continue;
            }
            switch (c) {
                '"' => quoted = true,
                ',' => {
                    try fields.append(a, try field.toOwnedSlice(a));
                },
                '\r' => {},
                '\n' => {
                    try fields.append(a, try field.toOwnedSlice(a));
                    try rows.append(a, try fields.toOwnedSlice(a));
                },
                else => try field.append(a, c),
            }
        }
        if (field.items.len > 0 or fields.items.len > 0) {
            try fields.append(a, try field.toOwnedSlice(a));
            try rows.append(a, try fields.toOwnedSlice(a));
        }
        if (rows.items.len == 0) return error.InvalidBundle;
        const header = rows.items[0];
        const name_col = indexOf(header, "Eval Task") orelse return error.InvalidBundle;
        const base_col = indexOf(header, "Random baseline") orelse return error.InvalidBundle;
        for (tasks) |*t| {
            for (rows.items[1..]) |row| {
                if (row.len <= @max(name_col, base_col) or !std.mem.eql(u8, row[name_col], t.label)) continue;
                t.random_baseline = std.fmt.parseFloat(f64, std.mem.trim(u8, row[base_col], " ")) catch return error.InvalidBundle;
                break;
            } else {
                log.warn("no random baseline for CORE task {s}", .{t.label});
                return error.InvalidBundle;
            }
        }
    }

    fn indexOf(row: []const []const u8, name: []const u8) ?usize {
        for (row, 0..) |f, i| {
            if (std.mem.eql(u8, f, name)) return i;
        }
        return null;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "eval bundle extracts a deflated zip and reads core.yaml and the baselines" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    try EvalBundle.extract(allocator, std.testing.io, mod.build_options.source_root ++ "/testdata/eval_bundle.zip", root);
    const config = mod.Config{ .allocator = allocator, .base_dir = root, .nanochat_dir = "/nonexistent-nanochat", .data_url = "http://unused" };
    var bundle = try EvalBundle.open(allocator, std.testing.io, &config, null);
    defer bundle.deinit();
    try std.testing.expectEqual(@as(usize, 6), bundle.tasks.len);
    const arc = bundle.tasks[1];
    try std.testing.expectEqualStrings("arc_easy", arc.label);
    try std.testing.expectEqual(CoreTaskType.multiple_choice, arc.task_type);
    try std.testing.expectEqual(@as(usize, 2), arc.num_fewshot);
    try std.testing.expectEqualStrings("\nAnswer: ", arc.continuation_delimiter);
    try std.testing.expectEqual(@as(f64, 25), arc.random_baseline);
    try std.testing.expectEqualStrings(" ", bundle.tasks[0].continuation_delimiter);
    try std.testing.expectEqual(CoreTaskType.schema, bundle.tasks[3].task_type);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const examples = try bundle.readExamples(arena.allocator(), mod.Storage.init(allocator, std.testing.io), arc);
    try std.testing.expectEqual(@as(usize, 24), examples.len);
}
