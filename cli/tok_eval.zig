const std = @import("std");
const log = std.log.scoped(.zignanogpt_tok_eval);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// `zignanogpt tok-eval`: compression (bytes per token) of the trained
/// tokenizer on a text file, optionally against another `.tiktoken` vocabulary
/// (e.g. GPT-4's `cl100k_base.tiktoken`, which splits digits in runs of 3).
pub const TokEval = struct {
    pub const usage =
        \\usage: zignanogpt tok-eval --data <file> [--compare <file.tiktoken>]
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
        const data = try args.string("data") orelse {
            try out.writeAll(usage);
            return 2;
        };
        const compare = try args.string("compare");
        try args.finish();

        var config = try mod.Config.init(allocator, init.environ_map);
        defer config.deinit();
        const storage = mod.Storage.init(allocator, init.io);
        const dir = try std.fs.path.join(allocator, &.{ config.base_dir, "tokenizer" });
        defer allocator.free(dir);
        var ours = try mod.Tokenizer.load(allocator, storage, dir);
        defer ours.deinit();
        var dataset = try mod.TextDataset.load(allocator, storage, data);
        defer dataset.deinit();

        try out.print("{s:<12} {s:>12} {s:>12} {s:>10}\n", .{ "tokenizer", "bytes", "tokens", "bytes/tok" });
        try report(allocator, out, "ours", &ours, dataset.docs);
        if (compare) |path| {
            const text = try storage.read(path);
            defer allocator.free(text);
            var other = try mod.Tokenizer.parseTiktoken(allocator, text, &.{}, 3);
            defer other.deinit();
            try report(allocator, out, std.fs.path.stem(path), &other, dataset.docs);
        }
        return 0;
    }

    fn report(allocator: std.mem.Allocator, out: *std.Io.Writer, name: []const u8, tokenizer: *const mod.Tokenizer, docs: []const []const u8) !void {
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
        try out.print("{s:<12} {d:>12} {d:>12} {d:>10.2}\n", .{ name, bytes, tokens, ratio });
    }
};
