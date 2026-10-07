const std = @import("std");
const log = std.log.scoped(.zignanogpt_runner);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// Runs a subcommand with its arguments: the one dispatch the command line and
/// the console's jobs share.
pub const Runner = struct {
    /// Runs `command`.
    ///
    /// Parameters:
    /// - `init`: process state.
    /// - `command`: an implemented pipeline command.
    /// - `items`: its arguments (borrowed).
    /// - `out`: its output.
    /// - `observer`: training and progress hooks (the console's job), or null on the command line.
    ///
    /// Return: the exit code; the command's errors, `error.NotImplemented`.
    pub fn execute(init: std.process.Init, command: cli.Command, items: []const []const u8, out: *std.Io.Writer, observer: ?mod.TrainObserver) !u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var args = try cli.Args.init(init.gpa, items);
        defer args.deinit();
        return switch (command) {
            .@"tok-train" => cli.TokTrain.run(init, &args, out, observer),
            .@"tok-eval" => cli.TokEval.run(init, &args, out),
            .download => cli.Download.run(init, &args, out, observer),
            .repackage => cli.Repackage.run(init, &args, out, observer),
            .import => cli.Import.run(init, &args, out),
            .train => cli.Train.run(init, &args, out, observer),
            .chat => cli.Chat.run(init, &args, out, observer),
            .@"chat-eval" => cli.ChatEval.run(init, &args, out, observer),
            .sft => cli.Sft.run(init, &args, out, observer),
            .eval => cli.BaseEval.run(init, &args, out, observer),
            .rl => cli.Rl.run(init, &args, out, observer),
            .@"infer-bench" => cli.InferBench.run(init, &args, out),
            else => {
                try out.print("{s}: not implemented yet (PLAN.md phase {d})\n", .{ @tagName(command), command.phase() orelse 0 });
                return error.NotImplemented;
            },
        };
    }
};
