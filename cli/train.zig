const std = @import("std");
const log = std.log.scoped(.zignanogpt_train);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// `zignanogpt train`: base pretraining (nanochat's `base_train.py`).
pub const Train = struct {
    pub const usage =
        \\usage: zignanogpt train [--preset cpu] [--dataset <name> | --data <file>] [--model-tag <tag>] [--resume-from-step <n>]
        \\                        [--threads <n>] [--no-tui] [base_train.py options...]
        \\  options (defaults as base_train.py; -1 disables like Python):
        \\    --depth --aspect-ratio --head-dim --max-seq-len --window-pattern
        \\    --num-iterations --target-flops --target-param-data-ratio
        \\    --device-batch-size --total-batch-size
        \\    --embedding-lr --unembedding-lr --weight-decay --matrix-lr --scalar-lr
        \\    --warmup-steps --warmdown-ratio --final-lr-frac
        \\    --eval-every --eval-tokens --sample-every --save-every
        \\  --preset cpu  runs/runcpu.sh's settings (d6, seq 512, window L, 5000 steps); flags override it
        \\  --dataset     the shards in <base>/base_data_<name> (default climbmix; others come from `repackage`)
        \\  --data        a text file (blank-line-separated documents; last 10% held out) instead of the shards
        \\  --no-tui      plain log lines even on a terminal (otherwise it runs in the console)
        \\
    ;

    /// Runs the command. On a terminal without an observer it opens the
    /// console on the training page and starts there.
    ///
    /// Parameters:
    /// - `init`: process state.
    /// - `args`: the command's options.
    /// - `out`: progress output.
    /// - `observer`: training hooks (the console's job), or null.
    ///
    /// Return: the exit code; setup and training errors.
    pub fn run(init: std.process.Init, args: *cli.Args, out: *std.Io.Writer, observer: ?mod.TrainObserver) !u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const allocator = init.gpa;
        if (args.flag("help")) {
            try out.writeAll(usage);
            return 0;
        }
        const no_tui = args.flag("no-tui");
        if (observer == null and !no_tui and try std.Io.File.stdout().isTty(init.io)) {
            try cli.Console.run(init, .{ .command = .train, .args = args.items });
            return 0;
        }
        var options: mod.TrainOptions = .{};
        if (try args.string("preset")) |preset| {
            if (!std.mem.eql(u8, preset, "cpu")) {
                try out.print("unknown preset: {s}\n", .{preset});
                return 2;
            }
            options = mod.TrainOptions.cpu;
        }
        try args.fill(mod.TrainOptions, &options);
        const data_path = try args.string("data");
        const dataset_name = try args.string("dataset") orelse mod.Dataset.default_name;
        const tag_arg = try args.string("model-tag");
        const resume_from = try args.int(i64, "resume-from-step", -1);
        const threads = try args.int(usize, "threads", 0);
        try args.finish();

        const storage = mod.Storage.init(allocator, init.io);
        var config = try mod.Config.load(allocator, init.environ_map, storage);
        defer config.deinit();
        const tok_dir = try std.fs.path.join(allocator, &.{ config.base_dir, "tokenizer" });
        defer allocator.free(tok_dir);
        var tokenizer = try mod.Tokenizer.load(allocator, storage, tok_dir);
        defer tokenizer.deinit();

        // Documents: a text file split 90/10, or the shards.
        var text: ?mod.TextDataset = null;
        defer if (text) |*t| t.deinit();
        var dataset = try mod.Dataset.init(allocator, init.io, &config, dataset_name);
        defer dataset.deinit();
        const shards = try dataset.list(allocator);
        defer {
            for (shards) |p| allocator.free(p);
            allocator.free(shards);
        }
        const data: mod.TrainData = if (data_path) |path| blk: {
            text = try mod.TextDataset.load(allocator, storage, path);
            const docs = text.?.docs;
            if (docs.len < 2) return error.EmptyDataset;
            const val = @max(docs.len / 10, 1);
            break :blk .{ .text = .{ .train = docs[0 .. docs.len - val], .val = docs[docs.len - val ..] } };
        } else blk: {
            if (shards.len < 2) {
                try out.print("{s}: need at least one train shard and the val shard; {s} (or pass --data)\n", .{ dataset.dir, cli.Download.hint(dataset_name) });
                return 1;
            }
            break :blk .{ .shards = shards };
        };

        var tag_buf: [32]u8 = undefined;
        const tag = tag_arg orelse try std.fmt.bufPrint(&tag_buf, "d{d}", .{options.depth});
        var backend = try mod.Backend.init(allocator, init.io, .{ .threads = threads });
        defer backend.deinit();
        const trainer = try allocator.create(mod.Trainer);
        defer allocator.destroy(trainer);
        try trainer.init(allocator, init.io, &backend, &tokenizer, options, data, config.base_dir, tag, if (resume_from >= 0) @intCast(resume_from) else null, out);
        defer trainer.deinit();
        trainer.observer = observer;
        try trainer.describe();
        try trainer.run();
        return 0;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "train maps base_train.py flags onto the options" {
    var args = try cli.Args.init(std.testing.allocator, &.{ "--depth", "8", "--window-pattern=SL", "--num-iterations", "-1", "--matrix-lr=0.05" });
    defer args.deinit();
    var options = mod.TrainOptions.cpu;
    try args.fill(mod.TrainOptions, &options);
    try args.finish();
    try std.testing.expectEqual(@as(usize, 8), options.depth);
    try std.testing.expectEqualStrings("SL", options.window_pattern);
    try std.testing.expectEqual(@as(usize, 0), options.num_iterations);
    try std.testing.expectEqual(@as(f64, 0.05), options.matrix_lr);
    try std.testing.expectEqual(@as(usize, 512), options.max_seq_len); // preset kept
}
