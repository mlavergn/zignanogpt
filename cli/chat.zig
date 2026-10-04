const std = @import("std");
const log = std.log.scoped(.zignanogpt_chat);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// `chat`'s options (nanochat's `chat_cli.py` flags).
pub const ChatSettings = struct {
    /// Null picks sft, falling back to base while there is no sft model.
    source: ?mod.CheckpointKind = null,
    tag: ?[]const u8 = null,
    step: ?usize = null,
    prompt: ?[]const u8 = null,
    options: mod.ChatOptions = .{},
    threads: usize = 0,
    no_tui: bool = false,
};

/// `zignanogpt chat`: talk to a model (nanochat's `chat_cli.py`). On a
/// terminal it opens the console's chat page; `-p` answers one prompt.
pub const Chat = struct {
    pub const usage =
        \\usage: zignanogpt chat [-i <source>] [-g <tag>] [-s <step>] [-p <prompt>] [-t <temperature>] [-k <top-k>]
        \\                       [--max-tokens <n>] [--threads <n>] [--no-tui]
        \\  -i, --source       base, sft or rl (default: sft, or base while there is no sft model)
        \\  -g, --model-tag    e.g. d12 (default: the largest d<N>)
        \\  -s, --step         (default: the last)
        \\  -p, --prompt       answer this one prompt and exit
        \\  -t, --temperature  (default 0.6; 0 is greedy)
        \\  -k, --top-k        (default 50; 0 samples from every token)
        \\  --max-tokens       per reply (default 256)
        \\  --no-tui           the line-based chat even on a terminal
        \\
    ;

    /// Runs the command.
    ///
    /// Parameters:
    /// - `init`: process state.
    /// - `args`: the command's options.
    /// - `out`: the conversation.
    /// - `observer`: set when the console runs the command (never for chat).
    ///
    /// Return: the exit code; loading and generation errors.
    pub fn run(init: std.process.Init, args: *cli.Args, out: *std.Io.Writer, observer: ?mod.TrainObserver) !u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (args.flag("help")) {
            try out.writeAll(usage);
            return 0;
        }
        const settings = parse(args) catch |err| {
            log.debug("chat usage error [{t}]", .{err});
            try out.writeAll(usage);
            return 2;
        };
        if (observer == null and settings.prompt == null and !settings.no_tui and try std.Io.File.stdout().isTty(init.io)) {
            // The console's form knows the long names only.
            const long = try longFlags(init.arena.allocator(), args.items);
            try cli.Console.run(init, .{ .command = .chat, .args = long });
            return 0;
        }

        const allocator = init.gpa;
        var backend = try mod.Backend.init(allocator, init.io, .{ .threads = settings.threads });
        defer backend.deinit();
        const loaded = try allocator.create(mod.LoadedModel);
        defer allocator.destroy(loaded);
        try open(loaded, init, &backend, settings, out);
        defer loaded.deinit();
        var session = try mod.ChatSession.init(allocator, &loaded.model, &loaded.tokenizer, settings.options);
        defer session.deinit();

        if (settings.prompt) |prompt| {
            try session.reply(prompt, out, null);
            try out.writeAll("\n");
            return 0;
        }
        try out.writeAll("\nNanoChat Interactive Mode\n" ++ "-" ** 50 ++ "\nType 'quit' or 'exit' to end the conversation\nType 'clear' to start a new conversation\n" ++ "-" ** 50 ++ "\n");
        var in_buffer: [64 * 1024]u8 = undefined;
        var stdin = std.Io.File.stdin().reader(init.io, &in_buffer);
        while (true) {
            try out.writeAll("\nUser: ");
            try out.flush();
            const line = try stdin.interface.takeDelimiter('\n') orelse {
                try out.writeAll("\nGoodbye!\n");
                break;
            };
            const input = std.mem.trim(u8, line, " \t\r");
            if (std.ascii.eqlIgnoreCase(input, "quit") or std.ascii.eqlIgnoreCase(input, "exit")) {
                try out.writeAll("Goodbye!\n");
                break;
            }
            if (std.ascii.eqlIgnoreCase(input, "clear")) {
                session.clear();
                try out.writeAll("Conversation cleared.\n");
                continue;
            }
            if (input.len == 0) continue;
            try out.writeAll("\nAssistant: ");
            session.reply(input, out, null) catch |err| switch (err) {
                error.SequenceTooLong => {
                    try out.writeAll("(the conversation is too long for the model; type 'clear')\n");
                    continue;
                },
                else => return err,
            };
            try out.writeAll("\n");
        }
        return 0;
    }

    /// Reads `chat`'s options.
    ///
    /// Parameters:
    /// - `args`: the command's options; all must be consumed.
    ///
    /// Return: the settings; `error.InvalidValue`, `error.MissingValue`, `error.UnknownArgument`.
    pub fn parse(args: *cli.Args) !ChatSettings {
        var s: ChatSettings = .{};
        if (try either(args, "source", "i")) |text| {
            s.source = std.meta.stringToEnum(mod.CheckpointKind, text) orelse {
                log.warn("--source: '{s}' is not base, sft or rl", .{text});
                return error.InvalidValue;
            };
        }
        s.tag = try either(args, "model-tag", "g");
        if (try either(args, "step", "s")) |text| s.step = try number(usize, "step", text);
        s.prompt = try either(args, "prompt", "p");
        if (try either(args, "temperature", "t")) |text| {
            const t = std.fmt.parseFloat(f32, text) catch return invalid("temperature", text);
            if (!(t >= 0)) return invalid("temperature", text);
            s.options.temperature = t;
        }
        if (try either(args, "top-k", "k")) |text| {
            const k = try number(usize, "top-k", text);
            s.options.top_k = if (k == 0) null else k;
        }
        s.options.max_tokens = try args.int(usize, "max-tokens", s.options.max_tokens);
        s.threads = try args.int(usize, "threads", 0);
        s.no_tui = args.flag("no-tui");
        try args.finish();
        return s;
    }

    /// Loads the model the settings name. Without `--source` it takes the sft
    /// model, or the base model (with a note on `out`) while there is none.
    ///
    /// Parameters:
    /// - `loaded`: receives the model (stable address).
    /// - `init`: process state.
    /// - `backend`: holds the weights.
    /// - `settings`: source, tag and step.
    /// - `out`: notes.
    ///
    /// Return: nothing; `error.NoCheckpoint` and loading errors.
    pub fn open(loaded: *mod.LoadedModel, init: std.process.Init, backend: *mod.Backend, settings: ChatSettings, out: *std.Io.Writer) !void {
        const allocator = init.gpa;
        const storage = mod.Storage.init(allocator, init.io);
        var config = try mod.Config.load(allocator, init.environ_map, storage);
        defer config.deinit();
        const kind = settings.source orelse blk: {
            const tags = try mod.Checkpoint.listTags(allocator, storage, config.base_dir, .sft);
            defer {
                for (tags) |t| allocator.free(t);
                allocator.free(tags);
            }
            if (tags.len > 0) break :blk mod.CheckpointKind.sft;
            try out.writeAll("No sft model yet: chatting with the base model (it continues text rather than answering).\n");
            break :blk mod.CheckpointKind.base;
        };
        try loaded.init(allocator, backend, storage, config.base_dir, .{ .kind = kind, .tag = settings.tag, .step = settings.step });
        try out.print("Loaded {s} model {s} step {d}\n", .{ @tagName(loaded.kind), loaded.tag, loaded.step });
        try out.flush();
    }

    /// `chat_cli.py`'s one-letter flags spelled out (`-t` is `--temperature`).
    ///
    /// Parameters:
    /// - `allocator`: owns the result (an arena: unchanged items are borrowed).
    /// - `items`: the arguments.
    ///
    /// Return: the arguments with long flags.
    pub fn longFlags(allocator: std.mem.Allocator, items: []const []const u8) ![]const []const u8 {
        const names = [_][2][]const u8{ .{ "i", "source" }, .{ "g", "model-tag" }, .{ "s", "step" }, .{ "p", "prompt" }, .{ "t", "temperature" }, .{ "k", "top-k" } };
        const out = try allocator.alloc([]const u8, items.len);
        for (items, out) |item, *o| {
            o.* = item;
            if (item.len < 2 or item[0] != '-' or item[1] == '-') continue;
            for (names) |n| {
                if (item.len == 2 and item[1] == n[0][0]) {
                    o.* = try std.mem.concat(allocator, u8, &.{ "--", n[1] });
                } else if (item.len > 2 and item[1] == n[0][0] and item[2] == '=') {
                    o.* = try std.mem.concat(allocator, u8, &.{ "--", n[1], item[2..] });
                }
            }
        }
        return out;
    }

    fn either(args: *cli.Args, long: []const u8, short: []const u8) !?[]const u8 {
        return try args.string(long) orelse try args.string(short);
    }

    fn number(comptime T: type, name: []const u8, text: []const u8) !T {
        return std.fmt.parseInt(T, text, 10) catch invalid(name, text);
    }

    fn invalid(name: []const u8, text: []const u8) error{InvalidValue} {
        log.warn("--{s}: '{s}' is not valid", .{ name, text });
        return error.InvalidValue;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "chat reads chat_cli.py's short and long flags" {
    var args = try cli.Args.init(std.testing.allocator, &.{ "-i", "base", "--model-tag=d6", "-s", "150", "-p", "Why is the sky blue?", "-t", "0", "--top-k", "0", "--max-tokens", "32" });
    defer args.deinit();
    const s = try cli.Chat.parse(&args);
    try std.testing.expectEqual(mod.CheckpointKind.base, s.source.?);
    try std.testing.expectEqualStrings("d6", s.tag.?);
    try std.testing.expectEqual(@as(usize, 150), s.step.?);
    try std.testing.expectEqualStrings("Why is the sky blue?", s.prompt.?);
    try std.testing.expectEqual(@as(f32, 0), s.options.temperature);
    try std.testing.expect(s.options.top_k == null);
    try std.testing.expectEqual(@as(usize, 32), s.options.max_tokens);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const long = try cli.Chat.longFlags(arena.allocator(), &.{ "-i", "base", "-t=0", "--top-k", "5", "-x" });
    const want = [_][]const u8{ "--source", "base", "--temperature=0", "--top-k", "5", "-x" };
    for (want, long) |w, l| try std.testing.expectEqualStrings(w, l);

    var bad = try cli.Args.init(std.testing.allocator, &.{ "-i", "dpo" });
    defer bad.deinit();
    try std.testing.expectError(error.InvalidValue, cli.Chat.parse(&bad));
}
