// ---------------------------------------------------------------------------
// zig build bench: backend matmul throughput (GFLOP/s) on the shapes the model
// runs, single-threaded and on every core. Always built ReleaseFast.
// Shapes follow the CPU preset (depth 6, n_embd 384, batch 32 x seq 512),
// plus single-token decoding at d20.
// ---------------------------------------------------------------------------

const std = @import("std");
const log = std.log.scoped(.zignanogpt_bench);
const mod = @import("zignanogpt");

pub const std_options: std.Options = .{ .log_level = .info };

/// One timed product.
const Case = struct {
    name: []const u8,
    m: usize,
    n: usize,
    k: usize,
    options: mod.MatmulOptions = .{},
};

const cases = [_]Case{
    .{ .name = "attn proj  x@W^T", .m = 16384, .n = 384, .k = 384, .options = .{ .transpose_b = true } },
    .{ .name = "mlp fc     x@W^T", .m = 16384, .n = 1536, .k = 384, .options = .{ .transpose_b = true } },
    .{ .name = "mlp proj   x@W^T", .m = 16384, .n = 384, .k = 1536, .options = .{ .transpose_b = true } },
    .{ .name = "lm_head    x@W^T", .m = 1024, .n = 32768, .k = 384, .options = .{ .transpose_b = true } },
    .{ .name = "input grad dy@W ", .m = 16384, .n = 384, .k = 1536 },
    .{ .name = "wgrad   dy^T@x +=", .m = 1536, .n = 384, .k = 16384, .options = .{ .transpose_a = true, .accumulate = true } },
    .{ .name = "square 1024", .m = 1024, .n = 1024, .k = 1024 },
    // Chat decoding: one token through a d20 model (n_embd 1280), memory-bound.
    .{ .name = "decode fc  x@W^T", .m = 1, .n = 5120, .k = 1280, .options = .{ .transpose_b = true } },
    .{ .name = "decode head x@W^T", .m = 1, .n = 32768, .k = 1280, .options = .{ .transpose_b = true } },
};

const runs = 5;

/// Times every case at one thread and at all threads.
///
/// Parameters:
/// - `init`: runtime-supplied process state.
///
/// Return: nothing; allocation and backend errors.
pub fn main(init: std.process.Init) !void {
    log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer: std.Io.File.Writer = .init(.stdout(), init.io, &stdout_buffer);
    const out = &stdout_writer.interface;

    const cpus = std.Thread.getCpuCount() catch 1;
    try out.print("matmul GFLOP/s, {s} backend, best / median of {d} runs\n\n", .{ mod.Backend.name, runs });
    try out.print("{s:<20} {s:>16} {s:>18}\n", .{ "case", "1 thread", "all threads" });
    try out.flush();

    var single = try mod.Backend.init(init.gpa, init.io, .{ .threads = 1 });
    defer single.deinit();
    var multi = try mod.Backend.init(init.gpa, init.io, .{ .threads = cpus });
    defer multi.deinit();

    for (cases) |case| {
        const one = try timeCase(init.io, &single, case);
        const all = try timeCase(init.io, &multi, case);
        try out.print("{s:<20} {d:>7.1} / {d:>6.1} {d:>8.1} / {d:>6.1}  (x{d:.1}, {d} threads)\n", .{
            case.name, one.best, one.median, all.best, all.median, all.median / one.median, cpus,
        });
        try out.flush();
    }
}

const Result = struct { best: f64, median: f64 };

/// Runs one case `runs` times after a warm-up and reports GFLOP/s.
fn timeCase(io: std.Io, backend: *mod.Backend, case: Case) !Result {
    const allocator = backend.allocator;
    const a = try backend.alloc(.f32, if (case.options.transpose_a) &.{ case.k, case.m } else &.{ case.m, case.k });
    defer backend.free(a);
    const b = try backend.alloc(.f32, if (case.options.transpose_b) &.{ case.n, case.k } else &.{ case.k, case.n });
    defer backend.free(b);
    const c = try backend.alloc(.f32, &.{ case.m, case.n });
    defer backend.free(c);

    var rng = mod.Random.init(1);
    const host = try allocator.alloc(f32, @max(case.m * case.k, case.k * case.n));
    defer allocator.free(host);
    rng.fillUniform(host, -1, 1);
    try backend.upload(a, f32, host[0 .. case.m * case.k]);
    try backend.upload(b, f32, host[0 .. case.k * case.n]);

    try backend.matmul(c, a, b, case.options); // warm-up
    var gflops: [runs]f64 = undefined;
    const flops = 2.0 * @as(f64, @floatFromInt(case.m * case.n * case.k));
    for (&gflops) |*g| {
        const start = std.Io.Clock.awake.now(io);
        try backend.matmul(c, a, b, case.options);
        try backend.sync();
        const ns = start.durationTo(std.Io.Clock.awake.now(io)).nanoseconds;
        g.* = flops / @as(f64, @floatFromInt(@max(ns, 1)));
    }
    std.mem.sort(f64, &gflops, {}, std.sort.desc(f64));
    return Result{ .best = gflops[0], .median = gflops[runs / 2] };
}
