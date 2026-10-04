const std = @import("std");
const log = std.log.scoped(.zignanogpt_console);
const cli = @import("module.zig");
const vaxis = cli.vaxis;

/// The full-screen console: `zignanogpt` with no arguments on a terminal, or
/// `zignanogpt tui`. While it holds the terminal, `std.log` output goes to the
/// running job's log instead of the screen.
pub const Console = struct {
    var owns_terminal: bool = false;
    var capture: ?*cli.Job = null;

    /// Whether the console holds the terminal (logging must not print).
    pub fn quiet() bool {
        return owns_terminal;
    }

    /// Appends a log line to the running job, if any (called from `std_options.logFn`).
    ///
    /// Parameters:
    /// - `text`: the formatted line, newline included.
    ///
    /// Return: nothing.
    pub fn captureLog(text: []const u8) void {
        if (capture) |job| job.append(text);
    }

    /// Runs the console until it is quit.
    ///
    /// Parameters:
    /// - `init`: process state.
    /// - `start`: a job to start at once (e.g. from `zignanogpt train`), or null.
    ///
    /// Return: nothing; terminal and allocation errors.
    pub fn run(init: std.process.Init, start: ?cli.ConsoleStart) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const job = try init.gpa.create(cli.Job);
        job.init(init.gpa, init.io);
        var abandoned = false;
        defer if (!abandoned) {
            job.deinit();
            init.gpa.destroy(job);
        };
        const chat = try init.gpa.create(cli.ChatJob);
        chat.init(init.gpa, init.io);
        // Stops a reply; waits for a load to finish.
        defer {
            chat.deinit();
            init.gpa.destroy(chat);
        }

        var stopping_train = false;
        {
            // Raised before the app exists: creating it puts the terminal in
            // raw mode, and a log line after that would tear the screen.
            owns_terminal = true;
            capture = job;
            defer {
                capture = null;
                owns_terminal = false;
            }
            var buffer: [4096]u8 = undefined;
            var app: vaxis.vxfw.App = try .init(init.io, init.gpa, init.environ_map, &buffer);
            defer app.deinit();
            // ESC[38;5;Nm rather than vaxis' colon form, which Terminal.app ignores.
            app.vx.sgr = .legacy;

            const console = try init.gpa.create(cli.ConsoleApp);
            defer init.gpa.destroy(console);
            console.* = try cli.ConsoleApp.create(init.gpa, init, job, chat);
            defer console.deinit();
            console.pending_start = start;
            try app.run(console.widget(), .{});

            if (job.running()) {
                const op = cli.Operation.all[job.operation orelse 0];
                if (op.command == .train or op.command == .sft) {
                    job.requestStop();
                    stopping_train = true;
                } else {
                    // A download or tokenizer run cannot be interrupted; the
                    // process exit ends it. Its memory stays with the thread.
                    abandoned = true;
                }
            }
        }
        var out_buffer: [256]u8 = undefined;
        var stdout = std.Io.File.stdout().writer(init.io, &out_buffer);
        if (stopping_train) {
            try stdout.interface.writeAll("Stopping training: saving a checkpoint after the current step...\n");
            try stdout.interface.flush();
        } else if (abandoned) {
            try stdout.interface.writeAll("The running job was abandoned.\n");
            try stdout.interface.flush();
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "console captures log lines only while it holds the terminal" {
    try std.testing.expect(!Console.quiet());
    Console.captureLog("dropped: no job\n");
}
