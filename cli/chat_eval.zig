const std = @import("std");
const log = std.log.scoped(.zignanogpt_cli_chat_eval);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// `zignanogpt chat-eval`: nanochat's `chat_eval.py` (ARC-Easy, ARC-Challenge,
/// MMLU, GSM8K; HumanEval is out of scope). Task data downloads on first use.
pub const ChatEval = struct {
    pub const usage =
        \\usage: zignanogpt chat-eval [-i <source>] [-a <tasks>] [-x <max-problems>] [-g <tag>] [-s <step>]
        \\                            [-t <temperature>] [-k <top-k>] [-n <samples>] [-m <max-new-tokens>] [--threads <n>]
        \\  -i, --source          base, sft or rl (default: sft, or base while there is none)
        \\  -a, --task-name       ARC-Easy|ARC-Challenge|MMLU|GSM8K, '|'-separated (default: all)
        \\  -x, --max-problems    per task (default: all)
        \\  -t, --temperature     generative tasks (default 0)
        \\  -k, --top-k           (default 50)
        \\  -n, --num-samples     per generative problem; any correct sample passes (default 1)
        \\  -m, --max-new-tokens  (default 512)
        \\  -b, --batch-size      accepted for compatibility; problems run one at a time
        \\
    ;

    /// Runs the command.
    ///
    /// Parameters:
    /// - `init`: process state.
    /// - `args`: the command's options.
    /// - `out`: progress and results.
    ///
    /// Return: the exit code; loading and evaluation errors.
    pub fn run(init: std.process.Init, args: *cli.Args, out: *std.Io.Writer, observer: ?mod.TrainObserver) !u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (args.flag("help")) {
            try out.writeAll(usage);
            return 0;
        }
        const parsed = parse(args) catch |err| {
            log.debug("chat-eval usage error [{t}]", .{err});
            try out.writeAll(usage);
            return 2;
        };
        const allocator = init.gpa;
        var kinds: std.ArrayList(mod.TaskKind) = .empty;
        defer kinds.deinit(allocator);
        if (parsed.tasks) |names| {
            var it = std.mem.splitScalar(u8, names, '|');
            while (it.next()) |name| {
                const kind = mod.TaskKind.parse(name) orelse {
                    try out.print("unknown task: {s} (ARC-Easy, ARC-Challenge, MMLU, GSM8K)\n", .{name});
                    return 2;
                };
                if (kind == .smoltalk or kind == .custom) {
                    try out.print("{s} has no evaluation\n", .{kind.name()});
                    return 2;
                }
                try kinds.append(allocator, kind);
            }
        } else try kinds.appendSlice(allocator, &mod.chat_eval_tasks);

        var backend = try mod.Backend.init(allocator, init.io, .{ .threads = parsed.chat.threads });
        defer backend.deinit();
        const loaded = try allocator.create(mod.LoadedModel);
        defer allocator.destroy(loaded);
        try cli.Chat.open(loaded, init, &backend, parsed.chat, out);
        defer loaded.deinit();
        var config = try mod.Config.load(allocator, init.environ_map, mod.Storage.init(allocator, init.io));
        defer config.deinit();

        var eval = mod.ChatEval.init(allocator, &loaded.model, &loaded.tokenizer);
        eval.progress = out;
        eval.observer = observer;
        var results: std.ArrayList(mod.EvalResult) = .empty;
        defer results.deinit(allocator);
        for (kinds.items) |kind| {
            var task = try mod.Task.open(allocator, init.io, &config, kind, "test", out);
            defer task.deinit();
            const r = try eval.run(&task, parsed.generative, parsed.max_problems);
            try out.print("\nFinal: {d}/{d} ({d:.2}%)\n{s} accuracy: {d:.2}%\n", .{ r.passed, r.total, 100 * r.accuracy(), kind.name(), 100 * r.accuracy() });
            try out.flush();
            try results.append(allocator, r);
        }
        if (results.items.len == mod.chat_eval_tasks.len) {
            try out.print("ChatCORE metric (without HumanEval): {d:.4}\n", .{mod.ChatEval.chatCore(results.items)});
        }
        return 0;
    }

    const Parsed = struct {
        chat: cli.ChatSettings,
        tasks: ?[]const u8,
        max_problems: ?usize,
        generative: mod.GenerativeOptions,
    };

    fn parse(args: *cli.Args) !Parsed {
        var p = Parsed{ .chat = .{}, .tasks = try either(args, "task-name", "a"), .max_problems = null, .generative = .{} };
        if (try either(args, "max-problems", "x")) |t| p.max_problems = try number(t);
        if (try either(args, "temperature", "t")) |t| p.generative.temperature = std.fmt.parseFloat(f32, t) catch return error.InvalidValue;
        if (try either(args, "top-k", "k")) |t| {
            const k = try number(t);
            p.generative.top_k = if (k == 0) null else k;
        }
        if (try either(args, "num-samples", "n")) |t| p.generative.num_samples = @max(try number(t), 1);
        if (try either(args, "max-new-tokens", "m")) |t| p.generative.max_new_tokens = try number(t);
        _ = try either(args, "batch-size", "b");
        if (try either(args, "source", "i")) |t| p.chat.source = std.meta.stringToEnum(mod.CheckpointKind, t) orelse return error.InvalidValue;
        p.chat.tag = try either(args, "model-tag", "g");
        if (try either(args, "step", "s")) |t| p.chat.step = try number(t);
        p.chat.threads = try args.int(usize, "threads", 0);
        try args.finish();
        return p;
    }

    fn either(args: *cli.Args, long: []const u8, short: []const u8) !?[]const u8 {
        return try args.string(long) orelse try args.string(short);
    }

    fn number(text: []const u8) !usize {
        return std.fmt.parseInt(usize, text, 10) catch {
            log.warn("'{s}' is not a count", .{text});
            return error.InvalidValue;
        };
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "chat-eval reads chat_eval.py's flags" {
    var args = try cli.Args.init(std.testing.allocator, &.{ "-i", "base", "-a", "MMLU|GSM8K", "-x", "5", "-n", "2", "-m", "64", "-b", "8", "-k", "0" });
    defer args.deinit();
    const p = try ChatEval.parse(&args);
    try std.testing.expectEqual(mod.CheckpointKind.base, p.chat.source.?);
    try std.testing.expectEqualStrings("MMLU|GSM8K", p.tasks.?);
    try std.testing.expectEqual(@as(?usize, 5), p.max_problems);
    try std.testing.expectEqual(@as(usize, 2), p.generative.num_samples);
    try std.testing.expectEqual(@as(usize, 64), p.generative.max_new_tokens);
    try std.testing.expect(p.generative.top_k == null);
}
