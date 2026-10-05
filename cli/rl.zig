const std = @import("std");
const log = std.log.scoped(.zignanogpt_rl);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// `zignanogpt rl`: reinforcement learning on GSM8K (nanochat's `chat_rl.py`).
pub const Rl = struct {
    pub const usage =
        \\usage: zignanogpt rl [--model-tag <tag>] [--model-step <n>] [--threads <n>] [--no-tui] [chat_rl.py options...]
        \\  options (defaults as chat_rl.py):
        \\    --num-epochs --device-batch-size --examples-per-step --num-samples --max-new-tokens
        \\    --temperature --top-k --embedding-lr --unembedding-lr --matrix-lr --weight-decay
        \\    --init-lr-frac --eval-every --eval-examples --save-every
        \\  Starts from the sft checkpoint; writes chatrl_checkpoints/<tag>.
        \\
    ;

    /// Runs the command. On a terminal without an observer it opens the console.
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
            try cli.Console.run(init, .{ .command = .rl, .args = args.items });
            return 0;
        }
        var options: mod.RlOptions = .{};
        args.fill(mod.RlOptions, &options) catch |err| {
            log.debug("rl usage error [{t}]", .{err});
            try out.writeAll(usage);
            return 2;
        };
        const threads = try args.int(usize, "threads", 0);
        try args.finish();

        const storage = mod.Storage.init(allocator, init.io);
        var config = try mod.Config.load(allocator, init.environ_map, storage);
        defer config.deinit();
        var backend = try mod.Backend.init(allocator, init.io, .{ .threads = threads });
        defer backend.deinit();
        const trainer = try allocator.create(mod.RlTrainer);
        defer allocator.destroy(trainer);
        try trainer.init(allocator, init.io, &backend, &config, options, out);
        defer trainer.deinit();
        trainer.observer = observer;
        try trainer.run();
        return 0;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "rl maps chat_rl.py flags onto the options" {
    var args = try cli.Args.init(std.testing.allocator, &.{ "--num-samples", "8", "--device-batch-size=4", "--temperature", "0.7", "--top-k", "0" });
    defer args.deinit();
    var options: mod.RlOptions = .{};
    try args.fill(mod.RlOptions, &options);
    try args.finish();
    try std.testing.expectEqual(@as(usize, 8), options.num_samples);
    try std.testing.expectEqual(@as(usize, 4), options.device_batch_size);
    try std.testing.expectEqual(@as(f64, 0.7), options.temperature);
    try std.testing.expectEqual(@as(usize, 0), options.top_k);
}
