const std = @import("std");
const log = std.log.scoped(.zignanogpt_cuda_backend);
const mod = @import("../module.zig");

/// A CUDA tensor's storage: managed memory (one address for host and
/// device; unified on the DGX Spark's GB10) and its device pointer.
pub const CudaBuffer = struct {
    bytes: []align(64) u8,
    /// 0 for host-only tensors (`CpuBackend.alloc`).
    device_ptr: u64 = 0,
};

/// The kernels' PTX, generated at build time from `kernels.zig`.
const ptx = @embedFile("cuda_kernels.ptx");

const Result = c_int;
const Device = c_int;
const Context = ?*anyopaque;
const Stream = ?*anyopaque;
const Module = ?*anyopaque;
const Function = ?*anyopaque;

/// The driver API entry points, resolved from `libcuda.so.1` at run time (no
/// link-time dependency: the backend cross-compiles from any host).
const Driver = struct {
    cuInit: *const fn (c_uint) callconv(.c) Result,
    cuDeviceGet: *const fn (*Device, c_int) callconv(.c) Result,
    cuDevicePrimaryCtxRetain: *const fn (*Context, Device) callconv(.c) Result,
    cuDevicePrimaryCtxRelease_v2: *const fn (Device) callconv(.c) Result,
    cuCtxSetCurrent: *const fn (Context) callconv(.c) Result,
    cuStreamCreate: *const fn (*Stream, c_uint) callconv(.c) Result,
    cuStreamDestroy_v2: *const fn (Stream) callconv(.c) Result,
    cuStreamSynchronize: *const fn (Stream) callconv(.c) Result,
    cuModuleLoadData: *const fn (*Module, *const anyopaque) callconv(.c) Result,
    cuModuleUnload: *const fn (Module) callconv(.c) Result,
    cuModuleGetFunction: *const fn (*Function, Module, [*:0]const u8) callconv(.c) Result,
    cuMemAllocManaged: *const fn (*u64, usize, c_uint) callconv(.c) Result,
    cuMemFree_v2: *const fn (u64) callconv(.c) Result,
    cuLaunchKernel: *const fn (Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Stream, [*]?*anyopaque, ?[*]?*anyopaque) callconv(.c) Result,
    cuGetErrorString: *const fn (Result, *?[*:0]const u8) callconv(.c) Result,

    fn load(lib: *std.DynLib) !Driver {
        var d: Driver = undefined;
        inline for (@typeInfo(Driver).@"struct".fields) |f| {
            @field(d, f.name) = lib.lookup(f.type, f.name) orelse {
                log.warn("libcuda lacks {s}", .{f.name});
                return error.CudaUnavailable;
            };
        }
        return d;
    }
};

/// The kernels, by PTX entry name.
const Kernel = enum {
    fill_f32,
    copy_u32,
    add_f32,
    mul_f32,
    scale_f32,
    combine_f32,
    relu_square_f32,
    relu_square_backward_f32,
    embedding_f32,
    rmsnorm_f32,
    rmsnorm_backward_f32,
    matmul_f32,
};

/// The CUDA backend (driver API; target: the DGX Spark, GB10, aarch64 Linux).
/// Ops launch on one stream in issue order; `sync` waits for it. Kernels are
/// written in Zig (`src/cuda/kernels.zig`) and loaded as PTX, so neither nvcc nor
/// cuBLAS is needed: elementwise ops, embedding, RMS norm and a plain matmul
/// run on the GPU; the rest run on the CPU kernels over managed memory after
/// a sync. Built and checked for type errors on macOS, not yet run on a GPU.
pub const CudaBackend = struct {
    const Self = @This();

    pub const Buffer = CudaBuffer;
    pub const name = "cuda";

    pub const Options = struct {
        /// CPU worker threads for the ops that run on the CPU; 0 means one per CPU.
        threads: usize = 0,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: mod.CpuBackend,
    lib: std.DynLib,
    driver: Driver,
    device: Device,
    context: Context,
    stream: Stream,
    module: Module,
    functions: [@typeInfo(Kernel).@"enum".fields.len]Function,

    /// Loads the driver, opens device 0 and loads the kernels.
    ///
    /// Parameters:
    /// - `allocator`: host-side bookkeeping.
    /// - `io`: the CPU fallback's thread pool.
    /// - `options`: CPU threads.
    ///
    /// Return: the backend; `error.CudaUnavailable`, `error.CudaFailed`.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, options: Options) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var lib = std.DynLib.open("libcuda.so.1") catch {
            log.warn("cannot load libcuda.so.1 (no NVIDIA driver)", .{});
            return error.CudaUnavailable;
        };
        errdefer lib.close();
        const driver = try Driver.load(&lib);
        var self = Self{
            .allocator = allocator,
            .io = io,
            .cpu = undefined,
            .lib = lib,
            .driver = driver,
            .device = 0,
            .context = null,
            .stream = null,
            .module = null,
            .functions = undefined,
        };
        try self.check(driver.cuInit(0), "cuInit");
        try self.check(driver.cuDeviceGet(&self.device, 0), "cuDeviceGet");
        try self.check(driver.cuDevicePrimaryCtxRetain(&self.context, self.device), "cuDevicePrimaryCtxRetain");
        errdefer _ = driver.cuDevicePrimaryCtxRelease_v2(self.device);
        try self.check(driver.cuCtxSetCurrent(self.context), "cuCtxSetCurrent");
        try self.check(driver.cuStreamCreate(&self.stream, 0), "cuStreamCreate");
        errdefer _ = driver.cuStreamDestroy_v2(self.stream);
        try self.check(driver.cuModuleLoadData(&self.module, ptx), "cuModuleLoadData");
        errdefer _ = driver.cuModuleUnload(self.module);
        inline for (@typeInfo(Kernel).@"enum".fields, 0..) |f, i| {
            try self.check(driver.cuModuleGetFunction(&self.functions[i], self.module, f.name), "cuModuleGetFunction " ++ f.name);
        }
        self.cpu = try mod.CpuBackend.init(allocator, io, .{ .threads = options.threads });
        return self;
    }

    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.sync() catch |err| log.warn("CUDA work failed at shutdown [{t}]", .{err});
        self.cpu.deinit();
        _ = self.driver.cuModuleUnload(self.module);
        _ = self.driver.cuStreamDestroy_v2(self.stream);
        _ = self.driver.cuDevicePrimaryCtxRelease_v2(self.device);
        self.lib.close();
    }

    /// Allocates a zeroed tensor in managed memory.
    pub fn alloc(self: *Self, dtype: mod.Dtype, dims: []const usize) !mod.Tensor {
        const shape = try mod.Shape.init(dims);
        const len = shape.numel() * dtype.size();
        try self.current();
        var ptr: u64 = 0;
        // CU_MEM_ATTACH_GLOBAL = 1; managed allocations are at least 256-byte aligned.
        try self.check(self.driver.cuMemAllocManaged(&ptr, @max(len, 16), 1), "cuMemAllocManaged");
        const base: [*]align(64) u8 = @ptrFromInt(ptr);
        const bytes = base[0..len];
        @memset(bytes, 0);
        return mod.Tensor{ .buffer = .{ .bytes = bytes, .device_ptr = ptr }, .dtype = dtype, .shape = shape };
    }

    /// Frees a tensor after the work issued so far completes.
    pub fn free(self: *Self, tensor: mod.Tensor) void {
        if (tensor.buffer.device_ptr == 0) return self.cpu.free(tensor);
        self.sync() catch |err| log.warn("CUDA work failed before a free [{t}]", .{err});
        self.check(self.driver.cuMemFree_v2(tensor.buffer.device_ptr), "cuMemFree") catch |err| log.warn("cuMemFree failed [{t}]", .{err});
    }

    pub fn sync(self: *Self) !void {
        try self.current();
        try self.check(self.driver.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
    }

    pub fn upload(self: *Self, tensor: mod.Tensor, comptime T: type, data: []const T) !void {
        try self.sync();
        return self.cpu.upload(tensor, T, data);
    }

    pub fn download(self: *Self, tensor: mod.Tensor, comptime T: type, out: []T) !void {
        try self.sync();
        return self.cpu.download(tensor, T, out);
    }

    // -------------------------------------------------------------------------
    // Ops with CUDA kernels

    pub fn fill(self: *Self, tensor: mod.Tensor, value: f32) !void {
        if (tensor.dtype != .f32) return error.DtypeMismatch;
        const n = try count(tensor);
        try self.launch(.fill_f32, n, .{ ptrOf(tensor), value, n });
    }

    pub fn copy(self: *Self, dst: mod.Tensor, src: mod.Tensor) !void {
        if (dst.dtype != src.dtype) return error.DtypeMismatch;
        if (dst.numel() != src.numel()) return error.ShapeMismatch;
        if (overlaps(dst, src)) return self.onCpu("copy", .{ dst, src });
        const n = try count(dst);
        try self.launch(.copy_u32, n, .{ ptrOf(dst), ptrOf(src), n });
    }

    pub fn add(self: *Self, out: mod.Tensor, a: mod.Tensor, b: mod.Tensor) !void {
        try checkSame(out, a);
        try checkSame(out, b);
        const n = try count(out);
        try self.launch(.add_f32, n, .{ ptrOf(out), ptrOf(a), ptrOf(b), n });
    }

    pub fn mul(self: *Self, out: mod.Tensor, a: mod.Tensor, b: mod.Tensor) !void {
        try checkSame(out, a);
        try checkSame(out, b);
        const n = try count(out);
        try self.launch(.mul_f32, n, .{ ptrOf(out), ptrOf(a), ptrOf(b), n });
    }

    pub fn scale(self: *Self, out: mod.Tensor, a: mod.Tensor, s: f32) !void {
        try checkSame(out, a);
        const n = try count(out);
        try self.launch(.scale_f32, n, .{ ptrOf(out), ptrOf(a), s, n });
    }

    pub fn combine(self: *Self, out: mod.Tensor, x: mod.Tensor, xs: mod.Scalar, y: mod.Tensor, ys: mod.Scalar) !void {
        try checkSame(out, x);
        try checkSame(out, y);
        try xs.validate();
        try ys.validate();
        const n = try count(out);
        try self.launch(.combine_f32, n, .{
            ptrOf(out),                ptrOf(x),                                  ptrOf(y),
            ptrOf(xs.tensor orelse x), ptrOf(ys.tensor orelse x),                 xs.factor,
            ys.factor,                 @as(u32, @intFromBool(xs.tensor != null)), @as(u32, @intFromBool(ys.tensor != null)),
            n,
        });
    }

    pub fn reluSquare(self: *Self, out: mod.Tensor, x: mod.Tensor) !void {
        try checkSame(out, x);
        const n = try count(out);
        try self.launch(.relu_square_f32, n, .{ ptrOf(out), ptrOf(x), n });
    }

    pub fn reluSquareBackward(self: *Self, dx: mod.Tensor, dy: mod.Tensor, x: mod.Tensor) !void {
        try checkSame(dx, x);
        try checkSame(dy, x);
        const n = try count(dx);
        try self.launch(.relu_square_backward_f32, n, .{ ptrOf(dx), ptrOf(dy), ptrOf(x), n });
    }

    pub fn embedding(self: *Self, out: mod.Tensor, table: mod.Tensor, ids: mod.Tensor) !void {
        if (out.dtype != .f32 or table.dtype != .f32 or ids.dtype != .i32) return error.DtypeMismatch;
        const cols = table.shape.cols();
        if (out.numel() != ids.numel() * cols) return error.ShapeMismatch;
        try self.sync(); // the ids are checked on the host
        const vocab = table.shape.rows();
        for (std.mem.bytesAsSlice(i32, ids.buffer.bytes[ids.offset * 4 ..][0 .. ids.numel() * 4])) |id| {
            if (id < 0 or id >= vocab) return error.OutOfBounds;
        }
        const n = try count(out);
        try self.launch(.embedding_f32, n, .{ ptrOf(out), ptrOf(table), ptrOf(ids), @as(u32, @intCast(cols)), n });
    }

    pub fn rmsnorm(self: *Self, out: mod.Tensor, x: mod.Tensor, eps: f32) !void {
        try checkSame(out, x);
        const rows: u32 = @intCast(x.shape.rows());
        try self.launch(.rmsnorm_f32, rows, .{ ptrOf(out), ptrOf(x), rows, @as(u32, @intCast(x.shape.cols())), eps });
    }

    pub fn rmsnormBackward(self: *Self, dx: mod.Tensor, dy: mod.Tensor, x: mod.Tensor, eps: f32, accumulate: bool) !void {
        try checkSame(dx, x);
        try checkSame(dy, x);
        const rows: u32 = @intCast(x.shape.rows());
        try self.launch(.rmsnorm_backward_f32, rows, .{ ptrOf(dx), ptrOf(dy), ptrOf(x), rows, @as(u32, @intCast(x.shape.cols())), eps, @as(u32, @intFromBool(accumulate)) });
    }

    pub fn matmul(self: *Self, c: mod.Tensor, a: mod.Tensor, b: mod.Tensor, options: mod.MatmulOptions) !void {
        if (c.dtype != .f32 or a.dtype != .f32 or b.dtype != .f32) return error.DtypeMismatch;
        const m, const n, const k = try options.dims(c.shape, a.shape, b.shape);
        if (overlaps(c, a) or overlaps(c, b)) return error.Aliasing;
        if (options.batch > 1) {
            var single = options;
            single.batch = 1;
            for (0..options.batch) |i| try self.matmul(try options.matrixOf(c, i), try options.matrixOf(a, i), try options.matrixOf(b, i), single);
            return;
        }
        const total = std.math.cast(u32, m * n) orelse return error.TensorTooLarge;
        try self.launch(.matmul_f32, total, .{
            ptrOf(c),                                    ptrOf(a),                                    ptrOf(b),
            @as(u32, @intCast(m)),                       @as(u32, @intCast(n)),                       @as(u32, @intCast(k)),
            @as(u32, @intFromBool(options.transpose_a)), @as(u32, @intFromBool(options.transpose_b)), @as(u32, @intFromBool(options.accumulate)),
            options.alpha,
        });
    }

    // -------------------------------------------------------------------------
    // Ops on the CPU kernels (managed memory, after a sync)

    pub fn rope(self: *Self, out: mod.Tensor, x: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize) !void {
        return self.onCpu("rope", .{ out, x, cos, sin, pos0 });
    }
    pub fn ropeBackward(self: *Self, dx: mod.Tensor, dy: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize) !void {
        return self.onCpu("ropeBackward", .{ dx, dy, cos, sin, pos0 });
    }
    pub fn attention(self: *Self, out: mod.Tensor, q: mod.Tensor, k: mod.Tensor, v: mod.Tensor, lse: ?mod.Tensor, options: mod.AttentionOptions) !void {
        return self.onCpu("attention", .{ out, q, k, v, lse, options });
    }
    pub fn attentionBackward(self: *Self, dq: mod.Tensor, dk: mod.Tensor, dv: mod.Tensor, dout: mod.Tensor, q: mod.Tensor, k: mod.Tensor, v: mod.Tensor, out: mod.Tensor, lse: mod.Tensor, options: mod.AttentionOptions) !void {
        return self.onCpu("attentionBackward", .{ dq, dk, dv, dout, q, k, v, out, lse, options });
    }
    pub fn gateLinear(self: *Self, out: mod.Tensor, x: mod.Tensor, w: mod.Tensor) !void {
        return self.onCpu("gateLinear", .{ out, x, w });
    }
    pub fn gateLinearBackward(self: *Self, dx: mod.Tensor, dw: mod.Tensor, dout: mod.Tensor, x: mod.Tensor, w: mod.Tensor) !void {
        return self.onCpu("gateLinearBackward", .{ dx, dw, dout, x, w });
    }
    pub fn smear(self: *Self, out: mod.Tensor, x: mod.Tensor, gate: mod.Tensor, lambda: mod.Scalar) !void {
        return self.onCpu("smear", .{ out, x, gate, lambda });
    }
    pub fn smearBackward(self: *Self, dx: mod.Tensor, dgate: mod.Tensor, dlambda: mod.Tensor, dout: mod.Tensor, x: mod.Tensor, gate: mod.Tensor, lambda: mod.Scalar) !void {
        return self.onCpu("smearBackward", .{ dx, dgate, dlambda, dout, x, gate, lambda });
    }
    pub fn gatedAdd(self: *Self, out: mod.Tensor, x: mod.Tensor, y: mod.Tensor, gate: mod.Tensor, s: mod.Scalar) !void {
        return self.onCpu("gatedAdd", .{ out, x, y, gate, s });
    }
    pub fn valueMix(self: *Self, v: mod.Tensor, ve: mod.Tensor, gate: mod.Tensor) !void {
        return self.onCpu("valueMix", .{ v, ve, gate });
    }
    pub fn valueMixBackward(self: *Self, dve: mod.Tensor, dgate: mod.Tensor, dv: mod.Tensor, ve: mod.Tensor, gate: mod.Tensor) !void {
        return self.onCpu("valueMixBackward", .{ dve, dgate, dv, ve, gate });
    }
    pub fn softcap(self: *Self, out: mod.Tensor, logits: mod.Tensor, cap: f32, lse: ?mod.Tensor) !void {
        return self.onCpu("softcap", .{ out, logits, cap, lse });
    }
    pub fn crossEntropy(self: *Self, loss: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor, lse: ?mod.Tensor) !void {
        return self.onCpu("crossEntropy", .{ loss, logits, targets, lse });
    }
    pub fn crossEntropyRows(self: *Self, losses: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor) !void {
        return self.onCpu("crossEntropyRows", .{ losses, logits, targets });
    }
    pub fn crossEntropyBackward(self: *Self, dpad: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor, cap: f32, scale_by: f32, lse: ?mod.Tensor) !void {
        return self.onCpu("crossEntropyBackward", .{ dpad, logits, targets, cap, scale_by, lse });
    }
    pub fn crossEntropyWeightedBackward(self: *Self, dpad: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor, weights: mod.Tensor, cap: f32, lse: ?mod.Tensor) !void {
        return self.onCpu("crossEntropyWeightedBackward", .{ dpad, logits, targets, weights, cap, lse });
    }
    pub fn dot(self: *Self, out: mod.Tensor, a: mod.Tensor, b: mod.Tensor, factor: f32, accumulate: bool) !void {
        return self.onCpu("dot", .{ out, a, b, factor, accumulate });
    }
    pub fn embeddingBackward(self: *Self, dtable: mod.Tensor, dout: mod.Tensor, ids: mod.Tensor) !void {
        return self.onCpu("embeddingBackward", .{ dtable, dout, ids });
    }
    pub fn adamwStep(self: *Self, p: mod.Tensor, g: mod.Tensor, m: mod.Tensor, v: mod.Tensor, params: mod.AdamWParams) !void {
        return self.onCpu("adamwStep", .{ p, g, m, v, params });
    }
    pub fn muonMomentum(self: *Self, g: mod.Tensor, buf: mod.Tensor, momentum: f32) !void {
        return self.onCpu("muonMomentum", .{ g, buf, momentum });
    }
    pub fn muonPrepare(self: *Self, x: mod.Tensor) !void {
        return self.onCpu("muonPrepare", .{x});
    }
    pub fn muonFinish(self: *Self, p: mod.Tensor, g: mod.Tensor, second: mod.Tensor, params: mod.MuonParams) !void {
        return self.onCpu("muonFinish", .{ p, g, second, params });
    }

    // -------------------------------------------------------------------------
    // Launching

    fn onCpu(self: *Self, comptime op: []const u8, args: anytype) !void {
        try self.sync();
        return @call(.auto, @field(mod.CpuBackend, op), .{&self.cpu} ++ args);
    }

    /// Launches `kernel` over `threads` threads (blocks of 256) with `args`
    /// passed by address, as `cuLaunchKernel` wants.
    fn launch(self: *Self, kernel: Kernel, threads: u32, args: anytype) !void {
        if (threads == 0) return;
        try self.current();
        var values = args;
        var params: [values.len]?*anyopaque = undefined;
        inline for (0..values.len) |i| params[i] = @ptrCast(&values[i]);
        const block: u32 = 256;
        self.check(self.driver.cuLaunchKernel(self.functions[@intFromEnum(kernel)], (threads + block - 1) / block, 1, 1, block, 1, 1, 0, self.stream, &params, null), "cuLaunchKernel") catch |err| {
            log.warn("launching {t} failed", .{kernel});
            return err;
        };
    }

    /// Makes the context current on the calling thread (jobs move between threads).
    fn current(self: *Self) !void {
        try self.check(self.driver.cuCtxSetCurrent(self.context), "cuCtxSetCurrent");
    }

    fn check(self: *Self, result: Result, what: []const u8) !void {
        if (result == 0) return;
        var text: ?[*:0]const u8 = null;
        _ = self.driver.cuGetErrorString(result, &text);
        log.warn("{s} failed: {s} ({d})", .{ what, if (text) |t| std.mem.span(t) else "unknown", result });
        return error.CudaFailed;
    }

    fn ptrOf(t: mod.Tensor) u64 {
        return t.buffer.device_ptr + t.offset * t.dtype.size();
    }

    fn count(t: mod.Tensor) !u32 {
        return std.math.cast(u32, t.numel()) orelse error.TensorTooLarge;
    }

    fn checkSame(x: mod.Tensor, y: mod.Tensor) !void {
        if (x.dtype != .f32 or y.dtype != .f32) return error.DtypeMismatch;
        if (!x.shape.eql(y.shape)) return error.ShapeMismatch;
    }

    fn overlaps(a: mod.Tensor, b: mod.Tensor) bool {
        const a_lo = @intFromPtr(a.buffer.bytes.ptr) + a.offset * a.dtype.size();
        const b_lo = @intFromPtr(b.buffer.bytes.ptr) + b.offset * b.dtype.size();
        return a_lo < b_lo + b.numel() * b.dtype.size() and b_lo < a_lo + a.numel() * a.dtype.size();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "cuda backend passes the conformance suite" {
    try mod.Conformance.runAll(CudaBackend, std.testing.allocator, std.testing.io);
}
