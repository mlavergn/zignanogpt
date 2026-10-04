const std = @import("std");
const log = std.log.scoped(.zignanogpt_tok_eval);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// `zignanogpt tok-eval`: compression (bytes per token) of the trained
/// tokenizer on a text file, optionally against another `.tiktoken` vocabulary
/// (e.g. GPT-4's `cl100k_base.tiktoken`, which splits digits in runs of 3).
pub const TokEval = struct {
    pub const usage =
        \\usage: zignanogpt tok-eval [--data <file>] [--compare <file.tiktoken>]
        \\  --data     text file; default: the first row group of the first train and the val shard
        \\
    ;

    /// Runs the command.
    ///
    /// Parameters:
    /// - `init`: process state.
    /// - `args`: the command's options.
    /// - `out`: the report.
    ///
    /// Return: the exit code; storage, tokenizer and argument errors.
    pub fn run(init: std.process.Init, args: *cli.Args, out: *std.Io.Writer) !u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const allocator = init.gpa;
        const data = try args.string("data");
        const compare = try args.string("compare");
        try args.finish();

        var config = try mod.Config.load(allocator, init.environ_map, mod.Storage.init(allocator, init.io));
        defer config.deinit();
        const storage = mod.Storage.init(allocator, init.io);
        const dir = try std.fs.path.join(allocator, &.{ config.base_dir, "tokenizer" });
        defer allocator.free(dir);
        var ours = try mod.Tokenizer.load(allocator, storage, dir);
        defer ours.deinit();
        var other: ?mod.Tokenizer = null;
        defer if (other) |*o| o.deinit();
        if (compare) |path| {
            const text = try storage.read(path);
            defer allocator.free(text);
            other = try mod.Tokenizer.parseTiktoken(allocator, text, &.{}, 3);
        }
        const other_name = if (compare) |path| std.fs.path.stem(path) else "";

        try out.print("{s:<16} {s:<12} {s:>12} {s:>12} {s:>10}\n", .{ "data", "tokenizer", "bytes", "tokens", "bytes/tok" });
        if (data) |path| {
            var dataset = try mod.TextDataset.load(allocator, storage, path);
            defer dataset.deinit();
            try report(allocator, out, std.fs.path.basename(path), "ours", &ours, dataset.docs);
            if (other) |*o| try report(allocator, out, std.fs.path.basename(path), other_name, o, dataset.docs);
            return 0;
        }
        var dataset = try mod.Dataset.init(allocator, init.io, &config);
        defer dataset.deinit();
        const paths = try dataset.list(allocator);
        defer {
            for (paths) |p| allocator.free(p);
            allocator.free(paths);
        }
        if (paths.len < 2) {
            try out.print("no shards; pass --data or run `zignanogpt download -n 1`\n", .{});
            return 1;
        }
        for ([_][]const u8{ paths[0], paths[paths.len - 1] }, [_][]const u8{ "climbmix-train", "climbmix-val" }) |path, label| {
            var file = try mod.ParquetFile.open(allocator, init.io, path);
            defer file.deinit();
            var strings: mod.ParquetStrings = .{};
            defer strings.deinit(allocator);
            try file.readStrings(0, try file.column("text"), &strings);
            const docs = try allocator.alloc([]const u8, strings.len());
            defer allocator.free(docs);
            for (docs, 0..) |*d, i| d.* = strings.get(i);
            try report(allocator, out, label, "ours", &ours, docs);
            if (other) |*o| try report(allocator, out, label, other_name, o, docs);
        }
        return 0;
    }

    fn report(allocator: std.mem.Allocator, out: *std.Io.Writer, data: []const u8, name: []const u8, tokenizer: *const mod.Tokenizer, docs: []const []const u8) !void {
        var bytes: usize = 0;
        var tokens: usize = 0;
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(allocator);
        for (docs) |doc| {
            ids.clearRetainingCapacity();
            try tokenizer.encodeAppend(allocator, &ids, doc);
            bytes += doc.len;
            tokens += ids.items.len;
        }
        const ratio = @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(@max(tokens, 1)));
        try out.print("{s:<16} {s:<12} {d:>12} {d:>12} {d:>10.2}\n", .{ data, name, bytes, tokens, ratio });
    }
};
