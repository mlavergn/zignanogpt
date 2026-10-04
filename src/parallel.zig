const std = @import("std");
const log = std.log.scoped(.zignanogpt_parallel);
const mod = @import("module.zig");

/// Spreads independent work items over the `std.Io` thread pool.
///
/// Workers pull items from a shared atomic counter, so uneven items (and uneven
/// cores, e.g. performance vs efficiency) balance themselves. The caller's
/// thread is worker 0. Each concurrently running worker has a distinct index
/// below `threads`, so per-worker scratch indexed by it needs no locking.
pub const Parallel = struct {
    const Self = @This();

    io: std.Io,
    /// Upper bound on concurrent workers, the caller included.
    threads: usize,

    /// Creates the dispatcher.
    ///
    /// Parameters:
    /// - `io`: the Io whose pool runs the workers.
    /// - `threads`: worker count, the caller included; 0 means one per CPU.
    ///
    /// Return: the dispatcher.
    pub fn init(io: std.Io, threads: usize) Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const count = if (threads > 0) threads else std.Thread.getCpuCount() catch 1;
        return Self{ .io = io, .threads = @max(count, 1) };
    }

    /// Calls `work(context, item, worker)` once for every item in `0..count`.
    ///
    /// Returns after every item has run. With one thread or one item it runs
    /// inline, with no dispatch at all.
    ///
    /// Parameters:
    /// - `self`: the dispatcher.
    /// - `count`: the number of items.
    /// - `context`: passed through to `work`.
    /// - `work`: the item function; `worker` is below `self.threads`.
    ///
    /// Return: nothing; `error.Canceled` when the Io cancels the wait.
    pub fn run(self: *const Self, count: usize, context: anytype, comptime work: fn (@TypeOf(context), usize, usize) void) std.Io.Cancelable!void {
        const workers = @min(self.threads, count);
        if (workers <= 1) {
            for (0..count) |item| work(context, item, 0);
            return;
        }
        const Worker = struct {
            fn loop(ctx: @TypeOf(context), next: *std.atomic.Value(usize), total: usize, worker: usize) void {
                while (true) {
                    const item = next.fetchAdd(1, .monotonic);
                    if (item >= total) return;
                    work(ctx, item, worker);
                }
            }
        };
        var next = std.atomic.Value(usize).init(0);
        var group: std.Io.Group = .init;
        for (1..workers) |worker| group.async(self.io, Worker.loop, .{ context, &next, count, worker });
        Worker.loop(context, &next, count, 0);
        try group.await(self.io);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

const TestContext = struct {
    hits: []std.atomic.Value(u32),
    max_worker: *std.atomic.Value(usize),
};

fn testWork(ctx: TestContext, item: usize, worker: usize) void {
    _ = ctx.hits[item].fetchAdd(1, .monotonic);
    _ = ctx.max_worker.fetchMax(worker, .monotonic);
}

test "parallel runs every item exactly once" {
    var hits: [1000]std.atomic.Value(u32) = undefined;
    for (&hits) |*h| h.* = .init(0);
    var max_worker = std.atomic.Value(usize).init(0);

    const parallel = mod.Parallel.init(std.testing.io, 4);
    try parallel.run(hits.len, TestContext{ .hits = &hits, .max_worker = &max_worker }, testWork);
    for (hits) |h| try std.testing.expectEqual(@as(u32, 1), h.load(.monotonic));
    try std.testing.expect(max_worker.load(.monotonic) < 4);
}

test "parallel with one thread runs inline" {
    var hits: [10]std.atomic.Value(u32) = undefined;
    for (&hits) |*h| h.* = .init(0);
    var max_worker = std.atomic.Value(usize).init(0);

    const parallel = mod.Parallel.init(std.testing.io, 1);
    try parallel.run(hits.len, TestContext{ .hits = &hits, .max_worker = &max_worker }, testWork);
    for (hits) |h| try std.testing.expectEqual(@as(u32, 1), h.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 0), max_worker.load(.monotonic));
}
