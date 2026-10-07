const std = @import("std");
const log = std.log.scoped(.zignanogpt_sft);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// `zignanogpt sft`: supervised fine-tuning (nanochat's `chat_sft.py`).
pub const Sft = struct {
    pub const usage =
        \\usage: zignanogpt sft [--model-tag <tag>] [--model-step <n>] [--threads <n>] [--no-tui] [chat_sft.py options...]
        \\  options (defaults as chat_sft.py; empty batch/LR options inherit the base checkpoint's; -1 disables):
        \\    --load-optimizer --num-iterations --max-seq-len --device-batch-size --total-batch-size
        \\    --embedding-lr --unembedding-lr --matrix-lr --init-lr-frac --warmup-ratio --warmdown-ratio
        \\    --final-lr-frac --eval-every --eval-tokens --chatcore-every --chatcore-max-cat
        \\    --chatcore-max-sample --mmlu-epochs --gsm8k-epochs
        \\  --conversations <file.jsonl>  your own conversations in the training mixture, one per line:
        \\                  [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]
        \\                  (an optional system message first; or {"messages": [...]})
        \\  --conversations-epochs <n>    their copies in the mixture (default 1)
        \\  runs/runcpu.sh uses: --num-iterations 1500 --eval-every 200 --eval-tokens 524288
        \\  The task data (SmolTalk, MMLU, GSM8K; ~1 GB) downloads on first use.
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
            try cli.Console.run(init, .{ .command = .sft, .args = args.items });
            return 0;
        }
        var options: mod.SftOptions = .{};
        args.fill(mod.SftOptions, &options) catch |err| {
            log.debug("sft usage error [{t}]", .{err});
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
        try out.writeAll("Loading the task data...\n");
        try out.flush();
        const extra: ?mod.SftData.Extra = if (options.conversations) |path| .{ .path = path, .epochs = options.conversations_epochs } else null;
        const data = mod.SftData.openStandard(allocator, init.io, &config, options.mmlu_epochs, options.gsm8k_epochs, extra, out) catch |err| switch (err) {
            // Explained by a warning (file and line).
            error.InvalidConversation, error.EmptyConversations => return 1,
            else => return err,
        };
        defer data.destroy();
        const trainer = try allocator.create(mod.SftTrainer);
        defer allocator.destroy(trainer);
        try trainer.init(allocator, init.io, &backend, config.base_dir, options, data, out);
        defer trainer.deinit();
        trainer.observer = observer;
        try trainer.describe();
        try trainer.run();
        return 0;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "sft maps chat_sft.py flags onto the options" {
    var args = try cli.Args.init(std.testing.allocator, &.{ "--num-iterations", "1500", "--eval-tokens=524288", "--chatcore-every", "-1", "--matrix-lr", "0.01", "--model-tag", "d6" });
    defer args.deinit();
    var options: mod.SftOptions = .{};
    try args.fill(mod.SftOptions, &options);
    try args.finish();
    try std.testing.expectEqual(@as(usize, 1500), options.num_iterations);
    try std.testing.expectEqual(@as(usize, 524288), options.eval_tokens);
    try std.testing.expectEqual(@as(usize, 0), options.chatcore_every);
    try std.testing.expectEqual(@as(?f64, 0.01), options.matrix_lr);
    try std.testing.expectEqualStrings("d6", options.model_tag.?);
    try std.testing.expect(options.max_seq_len == null); // inherited
}
