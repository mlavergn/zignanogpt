const std = @import("std");
const log = std.log.scoped(.zignanogpt_job);
const cli = @import("module.zig");
const mod = cli.nanogpt;

pub const JobState = enum { idle, running, succeeded, failed };

/// A validation result, for the training view.
pub const EvalPoint = struct { step: usize, bpb: f64 };

/// What the console draws for a training job, copied out under the lock.
pub const TrainSnapshot = struct {
    report: ?mod.StepReport = null,
    losses: []const f32 = &.{},
    evals: []const EvalPoint = &.{},
    samples: []const []const u8 = &.{},
    sample_step: usize = 0,
};

/// Log bytes kept per job; older output is dropped from the front.
const log_capacity = 256 * 1024;

/// One command running on a worker thread for the console: its output (a
/// `std.Io.Writer` the command writes to unchanged), its result, and for
/// training the step reports, evals and samples the observer delivers.
/// Everything the worker touches is behind `mutex`; the console reads copies.
pub const Job = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    state: JobState = .idle,
    /// The operation (index into `Operation.all`) this job ran.
    operation: ?usize = null,
    exit_code: u8 = 0,
    failure: ?[]const u8 = null,
    text: std.ArrayList(u8) = .empty,
    writer: std.Io.Writer,
    write_buffer: [512]u8 = undefined,
    stop_requested: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// The arguments, owned until the next start.
    arena: std.heap.ArenaAllocator,

    losses: std.ArrayList(f32) = .empty,
    evals: std.ArrayList(EvalPoint) = .empty,
    samples: std.ArrayList([]u8) = .empty,
    sample_step: usize = 0,
    report: ?mod.StepReport = null,

    /// Creates an idle job; it must stay at this address (the writer points into it).
    ///
    /// Parameters:
    /// - `self`: the storage.
    /// - `allocator`: owns the log and history.
    /// - `io`: locks.
    ///
    /// Return: nothing.
    pub fn init(self: *Self, allocator: std.mem.Allocator, io: std.Io) void {
        self.* = .{
            .allocator = allocator,
            .io = io,
            .writer = undefined,
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
        self.writer = .{ .vtable = &.{ .drain = drain }, .buffer = &self.write_buffer };
    }

    /// Waits for a running job, then frees everything.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (self.thread) |t| t.join();
        self.clearHistory();
        self.samples.deinit(self.allocator);
        self.evals.deinit(self.allocator);
        self.losses.deinit(self.allocator);
        self.text.deinit(self.allocator);
        self.arena.deinit();
    }

    /// Starts a command on a worker thread.
    ///
    /// Parameters:
    /// - `self`: an idle (or finished) job.
    /// - `process`: process state for the command.
    /// - `operation`: the console entry being run.
    /// - `command`: what to run.
    /// - `args`: its arguments (copied).
    ///
    /// Return: nothing; `error.JobRunning`, thread errors.
    pub fn start(self: *Self, process: std.process.Init, operation: usize, command: cli.Command, args: []const []const u8) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (self.running()) return error.JobRunning;
        if (self.thread) |t| t.join();
        self.thread = null;
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const owned = try a.alloc([]const u8, args.len);
        for (args, owned) |src, *dst| dst.* = try a.dupe(u8, src);

        self.lock();
        self.clearHistory();
        self.text.clearRetainingCapacity();
        self.state = .running;
        self.operation = operation;
        self.failure = null;
        self.exit_code = 0;
        self.stop_requested.store(false, .release);
        self.unlock();
        self.thread = try std.Thread.spawn(.{}, worker, .{ self, process, command, owned });
    }

    pub fn running(self: *Self) bool {
        self.lock();
        defer self.unlock();
        return self.state == .running;
    }

    pub fn currentState(self: *Self) JobState {
        self.lock();
        defer self.unlock();
        return self.state;
    }

    /// Asks a training job to save a checkpoint and stop after its current step.
    pub fn requestStop(self: *Self) void {
        self.stop_requested.store(true, .release);
    }

    /// Appends text to the log (also used for captured `std.log` lines).
    ///
    /// Parameters:
    /// - `self`: the job.
    /// - `bytes`: the text.
    ///
    /// Return: nothing.
    pub fn append(self: *Self, bytes: []const u8) void {
        self.lock();
        defer self.unlock();
        self.appendLocked(bytes);
    }

    /// The last `max_lines` lines of the log, copied into `allocator`.
    ///
    /// Parameters:
    /// - `self`: the job.
    /// - `allocator`: owns the result (a draw arena).
    /// - `max_lines`: at most this many.
    ///
    /// Return: the lines, oldest first.
    pub fn tail(self: *Self, allocator: std.mem.Allocator, max_lines: usize) ![]const []const u8 {
        self.lock();
        defer self.unlock();
        var lines: std.ArrayList([]const u8) = .empty;
        var end = self.text.items.len;
        while (end > 0 and self.text.items[end - 1] == '\n') end -= 1;
        while (lines.items.len < max_lines and end > 0) {
            const from = if (std.mem.lastIndexOfScalar(u8, self.text.items[0..end], '\n')) |i| i + 1 else 0;
            try lines.append(allocator, try allocator.dupe(u8, self.text.items[from..end]));
            end = if (from > 0) from - 1 else 0;
        }
        std.mem.reverse([]const u8, lines.items);
        return lines.items;
    }

    /// A copy of the training history, for drawing.
    pub fn snapshot(self: *Self, allocator: std.mem.Allocator) !TrainSnapshot {
        self.lock();
        defer self.unlock();
        const samples = try allocator.alloc([]const u8, self.samples.items.len);
        for (self.samples.items, samples) |s, *d| d.* = try allocator.dupe(u8, s);
        return .{
            .report = self.report,
            .losses = try allocator.dupe(f32, self.losses.items),
            .evals = try allocator.dupe(EvalPoint, self.evals.items),
            .samples = samples,
            .sample_step = self.sample_step,
        };
    }

    /// The trainer hooks feeding this job.
    pub fn observer(self: *Self) mod.TrainObserver {
        return .{ .context = self, .onStep = onStep, .onEval = onEval, .onSample = onSample, .shouldStop = shouldStop };
    }

    fn worker(self: *Self, process: std.process.Init, command: cli.Command, args: []const []const u8) void {
        const code = cli.Runner.execute(process, command, args, &self.writer, self.observer()) catch |err| {
            self.writer.flush() catch |flush_err| log.debug("job log flush [{t}]", .{flush_err});
            self.lock();
            defer self.unlock();
            self.state = .failed;
            self.failure = @errorName(err);
            return;
        };
        self.writer.flush() catch |err| log.debug("job log flush [{t}]", .{err});
        self.lock();
        defer self.unlock();
        self.exit_code = code;
        self.state = if (code == 0) .succeeded else .failed;
    }

    fn onStep(context: *anyopaque, report: mod.StepReport) void {
        const self: *Self = @ptrCast(@alignCast(context));
        self.lock();
        defer self.unlock();
        self.losses.append(self.allocator, @floatCast(report.loss)) catch |err| log.debug("loss history [{t}]", .{err});
        self.report = report;
    }

    fn onEval(context: *anyopaque, step: usize, bpb: f64) void {
        const self: *Self = @ptrCast(@alignCast(context));
        self.lock();
        defer self.unlock();
        self.evals.append(self.allocator, .{ .step = step, .bpb = bpb }) catch |err| log.debug("eval history [{t}]", .{err});
    }

    fn onSample(context: *anyopaque, step: usize, text: []const u8) void {
        const self: *Self = @ptrCast(@alignCast(context));
        self.lock();
        defer self.unlock();
        if (step != self.sample_step) {
            for (self.samples.items) |s| self.allocator.free(s);
            self.samples.clearRetainingCapacity();
            self.sample_step = step;
        }
        const copy = self.allocator.dupe(u8, text) catch return;
        self.samples.append(self.allocator, copy) catch self.allocator.free(copy);
    }

    fn shouldStop(context: *anyopaque) bool {
        const self: *Self = @ptrCast(@alignCast(context));
        return self.stop_requested.load(.acquire);
    }

    /// `std.Io.Writer` sink: everything written lands in the log.
    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *Self = @alignCast(@fieldParentPtr("writer", w));
        self.lock();
        defer self.unlock();
        self.appendLocked(w.buffered());
        w.end = 0;
        var n: usize = 0;
        for (data[0 .. data.len - 1]) |d| {
            self.appendLocked(d);
            n += d.len;
        }
        const last = data[data.len - 1];
        for (0..splat) |_| self.appendLocked(last);
        return n + last.len * splat;
    }

    fn appendLocked(self: *Self, bytes: []const u8) void {
        if (self.text.items.len + bytes.len > log_capacity) {
            // Keep the newer half (from a line start).
            const keep_from = self.text.items.len / 2;
            const cut = if (std.mem.indexOfScalarPos(u8, self.text.items, keep_from, '\n')) |i| i + 1 else keep_from;
            std.mem.copyForwards(u8, self.text.items[0 .. self.text.items.len - cut], self.text.items[cut..]);
            self.text.shrinkRetainingCapacity(self.text.items.len - cut);
        }
        self.text.appendSlice(self.allocator, bytes) catch |err| log.debug("job log [{t}]", .{err});
    }

    fn clearHistory(self: *Self) void {
        for (self.samples.items) |s| self.allocator.free(s);
        self.samples.clearRetainingCapacity();
        self.evals.clearRetainingCapacity();
        self.losses.clearRetainingCapacity();
        self.report = null;
        self.sample_step = 0;
    }

    fn lock(self: *Self) void {
        self.mutex.lockUncancelable(self.io);
    }

    fn unlock(self: *Self) void {
        self.mutex.unlock(self.io);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "job writer collects output and the tail returns the last lines" {
    var job: Job = undefined;
    job.init(std.testing.allocator, std.testing.io);
    defer job.deinit();
    try job.writer.print("one\ntwo\n", .{});
    try job.writer.writeAll("three\n");
    try job.writer.flush();
    job.append("four");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const lines = try job.tail(arena.allocator(), 2);
    try std.testing.expectEqual(@as(usize, 2), lines.len);
    try std.testing.expectEqualStrings("three", lines[0]);
    try std.testing.expectEqualStrings("four", lines[1]);
}

test "job runs a command on its worker and records the result" {
    var job: Job = undefined;
    job.init(std.testing.allocator, std.testing.io);
    defer job.deinit();
    var env = std.process.Environ.Map.init(std.testing.allocator);
    defer env.deinit();
    try env.put("HOME", "/nonexistent-home");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const process: std.process.Init = .{
        .minimal = .{ .environ = .empty, .args = undefined },
        .arena = &arena,
        .gpa = std.testing.allocator,
        .io = std.testing.io,
        .environ_map = &env,
        .preopens = undefined,
    };
    // Importing from a directory without checkpoints fails cleanly.
    try job.start(process, 4, .import, &.{ "--from", "/nonexistent-nanochat" });
    job.thread.?.join();
    job.thread = null;
    try std.testing.expectEqual(JobState.failed, job.currentState());
    try std.testing.expectEqualStrings("NoCheckpoint", job.failure.?);
    const lines = try job.tail(arena.allocator(), 10);
    try std.testing.expect(lines.len > 0);
}
