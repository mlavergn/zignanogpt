const std = @import("std");
const log = std.log.scoped(.zignanogpt_overview);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// One labelled line of the overview.
pub const OverviewLine = struct {
    label: []const u8,
    value: []const u8,
    /// Whether the value reports something missing or wrong.
    problem: bool = false,
};

/// The console's first page: where things live and what is ready
/// (tokenizer, shards, checkpoints), read from disk on demand.
pub const Overview = struct {
    const Self = @This();

    arena: std.heap.ArenaAllocator,
    lines: []const OverviewLine = &.{},

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{ .arena = std.heap.ArenaAllocator.init(allocator) };
    }

    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    /// Re-reads everything.
    ///
    /// Parameters:
    /// - `self`: the overview.
    /// - `process`: process state (config, storage).
    ///
    /// Return: nothing; problems become lines rather than errors.
    pub fn refresh(self: *Self, process: std.process.Init) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        _ = self.arena.reset(.retain_capacity);
        self.lines = self.collect(process) catch |err| blk: {
            const a = self.arena.allocator();
            const lines = a.alloc(OverviewLine, 1) catch break :blk &.{};
            lines[0] = .{ .label = "error", .value = @errorName(err), .problem = true };
            break :blk lines;
        };
    }

    fn collect(self: *Self, process: std.process.Init) ![]const OverviewLine {
        const a = self.arena.allocator();
        const storage = mod.Storage.init(a, process.io);
        var config = try mod.Config.load(a, process.environ_map, storage);
        var lines: std.ArrayList(OverviewLine) = .empty;
        try lines.append(a, .{ .label = "base dir", .value = config.base_dir });
        try lines.append(a, .{ .label = "nanochat dir", .value = config.nanochat_dir });
        try lines.append(a, .{ .label = "data url", .value = config.data_url });
        try lines.append(a, .{ .label = "backend", .value = try std.fmt.allocPrint(a, "{s}, {d} threads", .{ mod.Backend.name, std.Thread.getCpuCount() catch 1 }) });

        const tok_dir = try std.fs.path.join(a, &.{ config.base_dir, "tokenizer" });
        if (mod.Tokenizer.load(a, storage, tok_dir)) |tok| {
            try lines.append(a, .{ .label = "tokenizer", .value = try std.fmt.allocPrint(a, "{d} tokens", .{tok.vocabSize()}) });
        } else |_| {
            try lines.append(a, .{ .label = "tokenizer", .value = "missing: train one, or import --tokenizer-only", .problem = true });
        }

        var dataset = try mod.Dataset.init(a, process.io, &config, mod.Dataset.default_name);
        const shards = try dataset.list(a);
        const has_val = shards.len > 0 and std.mem.endsWith(u8, shards[shards.len - 1], "06542.parquet");
        try lines.append(a, .{
            .label = "data shards",
            .value = if (shards.len == 0) "none: run Download data" else try std.fmt.allocPrint(a, "{d} train + {s} val", .{ shards.len - @intFromBool(has_val), if (has_val) "1" else "no" }),
            .problem = shards.len < 2 or !has_val,
        });
        // Datasets made by `repackage`: the last shard is always validation.
        for (try mod.Dataset.names(a, storage, config.base_dir)) |name| {
            if (std.mem.eql(u8, name, mod.Dataset.default_name)) continue;
            var other = try mod.Dataset.init(a, process.io, &config, name);
            const n = (try other.list(a)).len;
            try lines.append(a, .{
                .label = try std.fmt.allocPrint(a, "data {s}", .{name}),
                .value = if (n < 2) "fewer than 2 shards: run Prepare data again" else try std.fmt.allocPrint(a, "{d} train + 1 val", .{n - 1}),
                .problem = n < 2,
            });
        }

        const before_checkpoints = lines.items.len;
        for ([_]mod.CheckpointKind{ .base, .sft, .rl }) |kind| {
            const tags = try mod.Checkpoint.listTags(a, storage, config.base_dir, kind);
            if (tags.len == 0) continue;
            for (tags) |tag| {
                var ckpt = try mod.Checkpoint.init(a, storage, config.base_dir, kind, tag);
                const step = ckpt.lastStep("safetensors") catch {
                    try lines.append(a, .{ .label = @tagName(kind), .value = try std.fmt.allocPrint(a, "{s}: no model saved", .{tag}), .problem = true });
                    continue;
                };
                try lines.append(a, .{ .label = @tagName(kind), .value = try std.fmt.allocPrint(a, "{s} at step {d}", .{ tag, step }) });
            }
        }
        if (lines.items.len == before_checkpoints) try lines.append(a, .{ .label = "checkpoints", .value = "none yet" });
        return lines.items;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "overview reports a fresh base dir as empty" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    var env = std.process.Environ.Map.init(std.testing.allocator);
    defer env.deinit();
    try env.put(mod.Config.base_dir_env, root);
    try env.put(mod.Config.nanochat_dir_env, root);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const process: std.process.Init = .{ .minimal = .{ .environ = .empty, .args = undefined }, .arena = &arena, .gpa = std.testing.allocator, .io = std.testing.io, .environ_map = &env, .preopens = undefined };
    var overview = Overview.init(std.testing.allocator);
    defer overview.deinit();
    overview.refresh(process);
    var problems: usize = 0;
    for (overview.lines) |l| problems += @intFromBool(l.problem);
    try std.testing.expectEqual(@as(usize, 2), problems); // tokenizer and shards
    try std.testing.expectEqualStrings("none yet", overview.lines[overview.lines.len - 1].value);
}
