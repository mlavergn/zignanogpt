const std = @import("std");
const log = std.log.scoped(.zignanogpt_command);
const cli = @import("module.zig");

/// A `zignanogpt` subcommand. The pipeline stages mirror nanochat's scripts.
pub const Command = enum {
    const Self = @This();

    help,
    tui,
    version,
    config,
    download,
    @"tok-train",
    @"tok-eval",
    train,
    eval,
    chat,
    import,
    sft,
    @"chat-eval",
    rl,

    /// Looks up a subcommand by its command-line name.
    ///
    /// Parameters:
    /// - `name`: the first argument, e.g. `tok-train`.
    ///
    /// Return: the command, or null when `name` is not one.
    pub fn parse(name: []const u8) ?Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (std.mem.eql(u8, name, "-h") or std.mem.eql(u8, name, "--help")) return .help;
        if (std.mem.eql(u8, name, "--version")) return .version;
        return std.meta.stringToEnum(Self, name);
    }

    /// The PLAN.md phase that implements this command.
    ///
    /// Parameters:
    /// - `self`: the command.
    ///
    /// Return: the phase number, or null when the command already works.
    pub fn phase(self: Self) ?u8 {
        return switch (self) {
            .help, .tui, .version, .config, .@"tok-train", .@"tok-eval", .download, .import, .train => null,
            .chat => 9,
            .eval, .sft, .@"chat-eval", .rl => 10,
        };
    }

    /// One line describing the command, for `help`.
    ///
    /// Parameters:
    /// - `self`: the command.
    ///
    /// Return: the description.
    pub fn summary(self: Self) []const u8 {
        return switch (self) {
            .help => "Show this help",
            .tui => "Open the console (the default on a terminal)",
            .version => "Print the version",
            .config => "Print the resolved base directories",
            .download => "Download ClimbMix pretraining shards",
            .@"tok-train" => "Train the BPE tokenizer",
            .@"tok-eval" => "Evaluate the tokenizer's compression",
            .train => "Pretrain the base model",
            .eval => "Evaluate the base model (bpb, CORE)",
            .chat => "Chat with a model",
            .import => "Import a Python nanochat checkpoint and tokenizer",
            .sft => "Supervised fine-tuning",
            .@"chat-eval" => "Evaluate the chat model (ARC, MMLU, GSM8K)",
            .rl => "Reinforcement learning on GSM8K",
        };
    }

    /// Writes the usage text listing every command.
    ///
    /// Parameters:
    /// - `out`: the writer.
    ///
    /// Return: nothing; propagates write errors.
    pub fn writeUsage(out: *std.Io.Writer) std.Io.Writer.Error!void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        try out.writeAll("usage: zignanogpt <command> [options]\n\ncommands:\n");
        for (std.enums.values(Self)) |command| {
            try out.print("  {s:<10} {s}", .{ @tagName(command), command.summary() });
            if (command.phase()) |number| try out.print(" (not yet: phase {d})", .{number});
            try out.writeByte('\n');
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "command parses names and help flags" {
    try std.testing.expectEqual(cli.Command.@"tok-train", cli.Command.parse("tok-train").?);
    try std.testing.expectEqual(cli.Command.help, cli.Command.parse("--help").?);
    try std.testing.expectEqual(cli.Command.version, cli.Command.parse("--version").?);
    try std.testing.expect(cli.Command.parse("bogus") == null);
}

test "command usage lists every command" {
    var buffer: [2048]u8 = undefined;
    var writer: std.Io.Writer = .fixed(&buffer);
    try cli.Command.writeUsage(&writer);
    const text = writer.buffered();
    for (std.enums.values(cli.Command)) |command| {
        try std.testing.expect(std.mem.indexOf(u8, text, @tagName(command)) != null);
    }
}
