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
        inline for (@typeInfo(Driver).@"struct".field_names, @typeInfo(Driver).@"struct".field_types) |name, F| {
            @field(d, name) = lib.lookup(F, name) orelse {
                log.warn("libcuda lacks {s}", .{name});
                return error.CudaUnavailable;
            };
        }
        return d;
    }
};

/// The kernels, by PTX entry name (`kernels.zig` exports the same names).
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
    embedding_backward_f32,
    rmsnorm_f32,
    rmsnorm_backward_f32,
    matmul_f32,
    rope_f32,
    gate_linear_f32,
    gate_linear_backward_dx_f32,
    gate_linear_backward_dw_f32,
    smear_f32,
    smear_backward_f32,
    gated_add_f32,
    value_mix_f32,
    value_mix_backward_f32,
    softcap_f32,
    softcap_lse_f32,
    xent_rows_f32,
    xent_rows_lse_f32,
    xent_backward_f32,
    dot_partial_f32,
    reduce_sum_f64,
    reduce_sum_f32,
    attention_f32,
    attention_delta_f32,
    attention_dq_f32,
    attention_dkv_f32,
    adamw_step_f32,
    muon_momentum_f32,
    row_squares_f64,
    col_squares_f64,
    total_f64,
    muon_scale_rows_f32,
    muon_scale_all_f32,
    muon_renorm_f32,
    normuon_stats_f32,
    muon_update_f32,
};

/// The attention problem as the kernels take it (`AttnDims` in kernels.zig).
const AttnDims = extern struct { b: u32, tq: u32, tk: u32, keys: u32, h: u32, hkv: u32, d: u32, window: u32 };

/// Threads per block for elementwise and row kernels (a multiple of 32: the
/// block reductions sum per warp).
const block_threads: u32 = 256;

/// The CUDA backend (driver API; target: DGX OS on the DGX Spark, GB10,
/// aarch64 Linux). Ops launch on one stream in issue order; `sync` waits for
/// it. Kernels are Zig (`src/cuda/kernels.zig`) loaded as PTX, so neither nvcc
/// nor cuBLAS is needed. Every op the model trains with has a kernel (tiled
/// matmul, warp-per-row attention, row reductions, optimizer); index tensors
/// are still checked on the host after a sync, and head dims past 256 fall
/// back to the CPU attention.
///
/// Written blind: it compiles (and its tests compile) for aarch64-linux, but
/// has not yet run on a GPU. Memory is managed (`cuMemAllocManaged`), which the
/// GB10's coherent unified memory lets host and device touch freely; on a GPU
/// without concurrent managed access the host writes in `alloc` would need a
/// sync first.
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
    functions: [@typeInfo(Kernel).@"enum".field_names.len]Function,
    /// Device allocations to free once the work issued so far completes
    /// (freed tensors and per-op scratch).
    deferred: std.ArrayList(u64) = .empty,

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
        inline for (@typeInfo(Kernel).@"enum".field_names, 0..) |tag, i| {
            try self.check(driver.cuModuleGetFunction(&self.functions[i], self.module, tag), "cuModuleGetFunction " ++ tag);
        }
        self.cpu = try mod.CpuBackend.init(allocator, io, .{ .threads = options.threads });
        return self;
    }

    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.sync() catch |err| log.warn("CUDA work failed at shutdown [{t}]", .{err});
        self.deferred.deinit(self.allocator);
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

    /// Frees a tensor once the work issued so far completes.
    pub fn free(self: *Self, tensor: mod.Tensor) void {
        if (tensor.buffer.device_ptr == 0) return self.cpu.free(tensor);
        self.deferred.append(self.allocator, tensor.buffer.device_ptr) catch {
            self.sync() catch |err| log.warn("CUDA work failed before a free [{t}]", .{err});
            self.check(self.driver.cuMemFree_v2(tensor.buffer.device_ptr), "cuMemFree") catch |err| log.warn("cuMemFree failed [{t}]", .{err});
        };
    }

    /// Waits for the stream, then frees the deferred allocations.
    pub fn sync(self: *Self) !void {
        try self.current();
        try self.check(self.driver.cuStreamSynchronize(self.stream), "cuStreamSynchronize");
        for (self.deferred.items) |ptr| {
            self.check(self.driver.cuMemFree_v2(ptr), "cuMemFree") catch |err| log.warn("cuMemFree failed [{t}]", .{err});
        }
        self.deferred.clearRetainingCapacity();
    }

    pub fn upload(self: *Self, tensor: mod.Tensor, comptime T: type, data: []const T) !void {
        try self.sync();
        return self.cpu.upload(tensor, T, data);
    }

    pub fn download(self: *Self, tensor: mod.Tensor, comptime T: type, out: []T) !void {
        try self.sync();
        return self.cpu.download(tensor, T, out);
    }

    /// A zeroed scratch tensor of `len` f32 slots (2 per f64), freed at the next sync.
    fn scratch(self: *Self, len: usize) !mod.Tensor {
        const t = try self.alloc(.f32, &.{@max(len, 1)});
        try self.deferred.append(self.allocator, t.buffer.device_ptr);
        return t;
    }

    // -------------------------------------------------------------------------
    // Elementwise

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

    pub fn rope(self: *Self, out: mod.Tensor, x: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize) !void {
        try self.rotate(out, x, cos, sin, pos0, 1);
    }

    pub fn ropeBackward(self: *Self, dx: mod.Tensor, dy: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize) !void {
        try self.rotate(dx, dy, cos, sin, pos0, -1);
    }

    /// `CpuBackend.ropeNorm`, as its three kernels (a fused kernel can come
    /// once this runs on the Spark).
    pub fn ropeNorm(self: *Self, out: mod.Tensor, x: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize, eps: f32, gain: f32) !void {
        try self.rope(x, x, cos, sin, pos0);
        try self.rmsnorm(out, x, eps);
        try self.scale(out, out, gain);
    }

    pub fn ropeNormBackward(self: *Self, dx: mod.Tensor, dy: mod.Tensor, x: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize, eps: f32, gain: f32) !void {
        try self.scale(dx, dy, gain);
        try self.rmsnormBackward(dx, dx, x, eps, false);
        try self.ropeBackward(dx, dx, cos, sin, pos0);
    }

    fn rotate(self: *Self, out: mod.Tensor, x: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize, sign: f32) !void {
        try checkSame(out, x);
        try checkSame(cos, sin);
        if (x.shape.rank != 4 or x.shape.dims[3] % 2 != 0 or cos.shape.cols() * 2 != x.shape.dims[3]) return error.ShapeMismatch;
        if (pos0 + x.shape.dims[1] > cos.shape.rows()) return error.OutOfBounds;
        const n = try count(x) / 2;
        try self.launch(.rope_f32, n, .{
            ptrOf(out),                          ptrOf(x),                            ptrOf(cos),                              ptrOf(sin),
            @as(u32, @intCast(x.shape.dims[1])), @as(u32, @intCast(x.shape.dims[2])), @as(u32, @intCast(x.shape.dims[3] / 2)), @as(u32, @intCast(pos0)),
            sign,                                n,
        });
    }

    pub fn gateLinear(self: *Self, out: mod.Tensor, x: mod.Tensor, w: mod.Tensor) !void {
        if (out.dtype != .f32 or x.dtype != .f32 or w.dtype != .f32) return error.DtypeMismatch;
        const heads = w.shape.rows();
        const cin = w.shape.cols();
        if (cin > x.shape.cols() or out.shape.rows() != x.shape.rows() or out.shape.cols() != heads) return error.ShapeMismatch;
        const n = try count(out);
        try self.launch(.gate_linear_f32, n, .{ ptrOf(out), ptrOf(x), ptrOf(w), @as(u32, @intCast(x.shape.cols())), @as(u32, @intCast(cin)), @as(u32, @intCast(heads)), @as(u32, @intCast(x.shape.rows())) });
    }

    pub fn gateLinearBackward(self: *Self, dx: mod.Tensor, dw: mod.Tensor, dout: mod.Tensor, x: mod.Tensor, w: mod.Tensor) !void {
        try checkSame(dx, x);
        try checkSame(dw, w);
        if (dout.dtype != .f32) return error.DtypeMismatch;
        const rows = x.shape.rows();
        const heads = w.shape.rows();
        const cin = w.shape.cols();
        if (cin > x.shape.cols() or dout.shape.rows() != rows or dout.shape.cols() != heads) return error.ShapeMismatch;
        const shape = .{ @as(u32, @intCast(x.shape.cols())), @as(u32, @intCast(cin)), @as(u32, @intCast(heads)), @as(u32, @intCast(rows)) };
        try self.launch(.gate_linear_backward_dx_f32, @intCast(rows * cin), .{ ptrOf(dx), ptrOf(dout), ptrOf(w) } ++ shape);
        try self.launchBlocks(.gate_linear_backward_dw_f32, .{ @intCast(heads * cin), 1, 1 }, .{ ptrOf(dw), ptrOf(dout), ptrOf(x) } ++ shape);
    }

    pub fn smear(self: *Self, out: mod.Tensor, x: mod.Tensor, gate: mod.Tensor, lambda: mod.Scalar) !void {
        try checkSame(out, x);
        try lambda.validate();
        if (gate.dtype != .f32) return error.DtypeMismatch;
        if (x.shape.rank != 3 or gate.numel() != x.shape.dims[0] * x.shape.dims[1]) return error.ShapeMismatch;
        if (overlaps(out, x)) return error.Aliasing;
        const n = try count(out);
        try self.launch(.smear_f32, n, .{
            ptrOf(out),                          ptrOf(x),                            ptrOf(gate),   ptrOf(lambda.tensor orelse gate),
            @as(u32, @intCast(x.shape.dims[1])), @as(u32, @intCast(x.shape.dims[2])), lambda.factor, @as(u32, @intFromBool(lambda.tensor != null)),
            n,
        });
    }

    pub fn smearBackward(self: *Self, dx: mod.Tensor, dgate: mod.Tensor, dlambda: mod.Tensor, dout: mod.Tensor, x: mod.Tensor, gate: mod.Tensor, lambda: mod.Scalar) !void {
        try checkSame(dx, x);
        try checkSame(dout, x);
        try checkSame(dgate, gate);
        try lambda.validate();
        if (dlambda.dtype != .f32 or dlambda.numel() == 0) return error.DtypeMismatch;
        if (x.shape.rank != 3 or gate.numel() != x.shape.dims[0] * x.shape.dims[1]) return error.ShapeMismatch;
        if (overlaps(dx, dout) or overlaps(dx, x)) return error.Aliasing;
        const rows: u32 = @intCast(x.shape.rows());
        const partial = try self.scratch(rows);
        try self.launchBlocks(.smear_backward_f32, .{ rows, 1, 1 }, .{
            ptrOf(dx),                           ptrOf(dgate),  ptrOf(partial),                                ptrOf(dout),
            ptrOf(x),                            ptrOf(gate),   ptrOf(lambda.tensor orelse gate),              @as(u32, @intCast(x.shape.dims[1])),
            @as(u32, @intCast(x.shape.dims[2])), lambda.factor, @as(u32, @intFromBool(lambda.tensor != null)),
        });
        // dlambda += factor * sum(partial), rows in order.
        try self.launchBlocks(.reduce_sum_f32, .{ 1, 1, 1 }, .{ ptrOf(partial), ptrOf(dlambda), rows, lambda.factor, @as(u32, 1) });
    }

    pub fn gatedAdd(self: *Self, out: mod.Tensor, x: mod.Tensor, y: mod.Tensor, gate: mod.Tensor, s: mod.Scalar) !void {
        try checkSame(out, x);
        try checkSame(out, y);
        try s.validate();
        if (gate.dtype != .f32) return error.DtypeMismatch;
        if (gate.numel() != x.shape.rows()) return error.ShapeMismatch;
        const n = try count(out);
        try self.launch(.gated_add_f32, n, .{
            ptrOf(out),                         ptrOf(x), ptrOf(y),                                 ptrOf(gate), ptrOf(s.tensor orelse gate),
            @as(u32, @intCast(x.shape.cols())), s.factor, @as(u32, @intFromBool(s.tensor != null)), n,
        });
    }

    pub fn valueMix(self: *Self, v: mod.Tensor, ve: mod.Tensor, gate: mod.Tensor) !void {
        if (v.dtype != .f32 or ve.dtype != .f32 or gate.dtype != .f32) return error.DtypeMismatch;
        const pairs = gate.numel();
        if (v.numel() != ve.numel() or pairs == 0 or v.numel() % pairs != 0) return error.ShapeMismatch;
        const n = try count(v);
        try self.launch(.value_mix_f32, n, .{ ptrOf(v), ptrOf(ve), ptrOf(gate), @as(u32, @intCast(v.numel() / pairs)), n });
    }

    pub fn valueMixBackward(self: *Self, dve: mod.Tensor, dgate: mod.Tensor, dv: mod.Tensor, ve: mod.Tensor, gate: mod.Tensor) !void {
        try checkSame(dve, ve);
        try checkSame(dgate, gate);
        if (dv.dtype != .f32) return error.DtypeMismatch;
        const pairs = gate.numel();
        if (dv.numel() != ve.numel() or pairs == 0 or ve.numel() % pairs != 0) return error.ShapeMismatch;
        try self.launch(.value_mix_backward_f32, @intCast(pairs), .{ ptrOf(dve), ptrOf(dgate), ptrOf(dv), ptrOf(ve), ptrOf(gate), @as(u32, @intCast(ve.numel() / pairs)), @as(u32, @intCast(pairs)) });
    }

    // -------------------------------------------------------------------------
    // Embedding (ids are checked on the host, after a sync)

    pub fn embedding(self: *Self, out: mod.Tensor, table: mod.Tensor, ids: mod.Tensor) !void {
        if (out.dtype != .f32 or table.dtype != .f32 or ids.dtype != .i32) return error.DtypeMismatch;
        const cols = table.shape.cols();
        if (out.numel() != ids.numel() * cols) return error.ShapeMismatch;
        _ = try self.hostIds(ids, table.shape.rows());
        const n = try count(out);
        try self.launch(.embedding_f32, n, .{ ptrOf(out), ptrOf(table), ptrOf(ids), @as(u32, @intCast(cols)), n });
    }

    /// `dtable[ids[r]] += dout[r]`: the host groups the rows by id (stable);
    /// one thread per (id, column) adds that id's rows in order (the CPU's order).
    pub fn embeddingBackward(self: *Self, dtable: mod.Tensor, dout: mod.Tensor, ids: mod.Tensor) !void {
        if (dtable.dtype != .f32 or dout.dtype != .f32 or ids.dtype != .i32) return error.DtypeMismatch;
        const cols = dtable.shape.cols();
        const rows = ids.numel();
        if (dout.numel() != rows * cols) return error.ShapeMismatch;
        if (rows == 0 or cols == 0) return;
        _ = try count(dtable);
        const id_values = try self.hostIds(ids, dtable.shape.rows());
        const order = try self.allocator.alloc(u32, rows);
        defer self.allocator.free(order);
        for (order, 0..) |*o, r| o.* = @intCast(r);
        const ByRow = struct {
            ids: []align(1) const i32,
            fn less(ctx: @This(), x: u32, y: u32) bool {
                return ctx.ids[x] < ctx.ids[y] or (ctx.ids[x] == ctx.ids[y] and x < y);
            }
        };
        std.sort.pdq(u32, order, ByRow{ .ids = id_values }, ByRow.less);
        var unique: usize = 0;
        for (order, 0..) |r, k| {
            if (k == 0 or id_values[r] != id_values[order[k - 1]]) unique += 1;
        }
        // Layout: rows by id, each id's start (and the end), then the ids.
        const groups = try self.scratch(rows + 2 * unique + 1);
        const words = std.mem.bytesAsSlice(u32, groups.buffer.bytes);
        @memcpy(words[0..rows], order);
        const starts = words[rows..][0 .. unique + 1];
        const unique_ids = words[rows + unique + 1 ..][0..unique];
        var u: usize = 0;
        for (order, 0..) |r, k| {
            if (k == 0 or id_values[r] != id_values[order[k - 1]]) {
                starts[u] = @intCast(k);
                unique_ids[u] = @intCast(id_values[r]);
                u += 1;
            }
        }
        starts[unique] = @intCast(rows);
        try self.launch(.embedding_backward_f32, @intCast(unique * cols), .{ ptrOf(dtable), ptrOf(dout), ptrOf(groups), @as(u32, @intCast(cols)), @as(u32, @intCast(unique)), @as(u32, @intCast(rows)) });
    }

    /// The ids of `ids` on the host after a sync, each checked below `vocab`.
    fn hostIds(self: *Self, ids: mod.Tensor, vocab: usize) ![]align(1) const i32 {
        try self.sync();
        const values = std.mem.bytesAsSlice(i32, ids.buffer.bytes[ids.offset * 4 ..][0 .. ids.numel() * 4]);
        for (values) |id| {
            if (id < 0 or id >= vocab) {
                log.debug("id {d} outside vocab {d}", .{ id, vocab });
                return error.OutOfBounds;
            }
        }
        return values;
    }

    // -------------------------------------------------------------------------
    // Row reductions (one block per row)

    pub fn rmsnorm(self: *Self, out: mod.Tensor, x: mod.Tensor, eps: f32) !void {
        try checkSame(out, x);
        try self.launchBlocks(.rmsnorm_f32, .{ try rowCount(x), 1, 1 }, .{ ptrOf(out), ptrOf(x), @as(u32, @intCast(x.shape.cols())), eps });
    }

    pub fn rmsnormBackward(self: *Self, dx: mod.Tensor, dy: mod.Tensor, x: mod.Tensor, eps: f32, accumulate: bool) !void {
        try checkSame(dx, x);
        try checkSame(dy, x);
        try self.launchBlocks(.rmsnorm_backward_f32, .{ try rowCount(x), 1, 1 }, .{ ptrOf(dx), ptrOf(dy), ptrOf(x), @as(u32, @intCast(x.shape.cols())), eps, @as(u32, @intFromBool(accumulate)) });
    }

    pub fn softcap(self: *Self, out: mod.Tensor, logits: mod.Tensor, cap: f32, lse: ?mod.Tensor) !void {
        if (out.dtype != .f32 or logits.dtype != .f32) return error.DtypeMismatch;
        if (out.shape.rows() != logits.shape.rows() or out.shape.cols() > logits.shape.cols()) return error.ShapeMismatch;
        if (overlaps(out, logits)) return error.Aliasing;
        const n = try count(out);
        const cols: u32 = @intCast(out.shape.cols());
        const padded: u32 = @intCast(logits.shape.cols());
        if (lse) |t| {
            try checkRowValues(t, out);
            return self.launchBlocks(.softcap_lse_f32, .{ try rowCount(out), 1, 1 }, .{ ptrOf(out), ptrOf(logits), ptrOf(t), cols, padded, cap });
        }
        try self.launch(.softcap_f32, n, .{ ptrOf(out), ptrOf(logits), cols, padded, cap, n });
    }

    /// Checks targets on the host (after a sync) and counts the valid ones.
    fn checkTargets(self: *Self, logits: mod.Tensor, targets: mod.Tensor) !usize {
        if (logits.dtype != .f32 or targets.dtype != .i32) return error.DtypeMismatch;
        if (targets.numel() != logits.shape.rows()) return error.ShapeMismatch;
        try self.sync();
        const vocab = logits.shape.cols();
        var valid: usize = 0;
        for (std.mem.bytesAsSlice(i32, targets.buffer.bytes[targets.offset * 4 ..][0 .. targets.numel() * 4])) |t| {
            if (t < -1 or t >= vocab) return error.OutOfBounds;
            valid += @intFromBool(t >= 0);
        }
        return valid;
    }

    pub fn crossEntropy(self: *Self, loss: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor, lse: ?mod.Tensor) !void {
        const valid = try self.checkTargets(logits, targets);
        if (loss.dtype != .f32 or loss.numel() == 0) return error.DtypeMismatch;
        const rows = try rowCount(logits);
        const losses = try self.scratch(rows);
        try self.xentRows(losses, logits, targets, lse);
        // The mean over valid rows (NaN when there are none, as PyTorch).
        const factor = if (valid == 0) std.math.nan(f32) else 1 / @as(f32, @floatFromInt(valid));
        try self.launchBlocks(.reduce_sum_f32, .{ 1, 1, 1 }, .{ ptrOf(losses), ptrOf(loss), rows, factor, @as(u32, 0) });
    }

    pub fn crossEntropyRows(self: *Self, losses: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor) !void {
        _ = try self.checkTargets(logits, targets);
        if (losses.dtype != .f32 or losses.numel() != logits.shape.rows()) return error.ShapeMismatch;
        try self.xentRows(losses, logits, targets, null);
    }

    fn xentRows(self: *Self, losses: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor, lse: ?mod.Tensor) !void {
        const rows = try rowCount(logits);
        const vocab: u32 = @intCast(logits.shape.cols());
        if (lse) |t| {
            try checkRowValues(t, logits);
            return self.launch(.xent_rows_lse_f32, rows, .{ ptrOf(losses), ptrOf(logits), ptrOf(targets), ptrOf(t), vocab, rows });
        }
        try self.launchBlocks(.xent_rows_f32, .{ rows, 1, 1 }, .{ ptrOf(losses), ptrOf(logits), ptrOf(targets), vocab });
    }

    pub fn crossEntropyBackward(self: *Self, dpad: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor, cap: f32, scale_by: f32, lse: ?mod.Tensor) !void {
        const valid = try self.checkTargets(logits, targets);
        try self.xentBackward(dpad, logits, targets, null, cap, scale_by / @as(f32, @floatFromInt(valid)), lse);
    }

    pub fn crossEntropyWeightedBackward(self: *Self, dpad: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor, weights: mod.Tensor, cap: f32, lse: ?mod.Tensor) !void {
        _ = try self.checkTargets(logits, targets);
        if (weights.dtype != .f32) return error.DtypeMismatch;
        if (weights.numel() != logits.shape.rows()) return error.ShapeMismatch;
        try self.xentBackward(dpad, logits, targets, weights, cap, 1, lse);
    }

    fn xentBackward(self: *Self, dpad: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor, weights: ?mod.Tensor, cap: f32, factor: f32, lse: ?mod.Tensor) !void {
        if (dpad.dtype != .f32) return error.DtypeMismatch;
        if (lse) |t| try checkRowValues(t, logits);
        if (dpad.shape.rows() != logits.shape.rows() or dpad.shape.cols() < logits.shape.cols()) return error.ShapeMismatch;
        if (overlaps(dpad, logits)) return error.Aliasing;
        try self.launchBlocks(.xent_backward_f32, .{ try rowCount(logits), 1, 1 }, .{
            ptrOf(dpad),              ptrOf(logits),                           ptrOf(targets),                        ptrOf(weights orelse logits),
            ptrOf(lse orelse logits), @as(u32, @intCast(logits.shape.cols())), @as(u32, @intCast(dpad.shape.cols())), cap,
            factor,                   @as(u32, @intFromBool(weights != null)), @as(u32, @intFromBool(lse != null)),
        });
    }

    pub fn dot(self: *Self, out: mod.Tensor, a: mod.Tensor, b: mod.Tensor, factor: f32, accumulate: bool) !void {
        try checkSame(a, b);
        if (out.dtype != .f32 or out.numel() == 0) return error.DtypeMismatch;
        const n = try count(a);
        const groups: u32 = @intCast(@max(1, @min(256, (n + 4095) / 4096)));
        const partial = try self.scratch(2 * groups); // one f64 per block
        try self.launchBlocks(.dot_partial_f32, .{ groups, 1, 1 }, .{ ptrOf(a), ptrOf(b), ptrOf(partial), n });
        try self.launchBlocks(.reduce_sum_f64, .{ 1, 1, 1 }, .{ ptrOf(partial), ptrOf(out), groups, factor, @as(u32, @intFromBool(accumulate)) });
    }

    // -------------------------------------------------------------------------
    // Matmul: 64x64 tiles per block of 256 threads (`matmul_f32` in kernels.zig)

    pub fn matmul(self: *Self, c: mod.Tensor, a: mod.Tensor, b: mod.Tensor, options: mod.MatmulOptions) !void {
        if (c.dtype != .f32 or a.dtype != .f32 or b.dtype != .f32) return error.DtypeMismatch;
        const m, const n, const k = try options.dims(c.shape, a.shape, b.shape);
        if (overlaps(c, a) or overlaps(c, b)) return error.Aliasing;
        if (m == 0 or n == 0) return;
        _ = try count(c);
        _ = try count(a);
        _ = try count(b);
        const grid = [3]u32{ @intCast((n + 63) / 64), @intCast((m + 63) / 64), @intCast(options.batch) };
        try self.launchBlocks(.matmul_f32, grid, .{
            ptrOf(c),                                    ptrOf(a),                                    ptrOf(b),
            @as(u32, @intCast(m)),                       @as(u32, @intCast(n)),                       @as(u32, @intCast(k)),
            @as(u32, @intFromBool(options.transpose_a)), @as(u32, @intFromBool(options.transpose_b)), @as(u32, @intFromBool(options.accumulate)),
            options.alpha,
        });
    }

    // -------------------------------------------------------------------------
    // Attention: one warp per query row (forward, dq) or key row (dk/dv)

    /// Head dims the warp-per-row kernels hold (8 per lane).
    const max_head_dim = 256;

    pub fn attention(self: *Self, out: mod.Tensor, q: mod.Tensor, k: mod.Tensor, v: mod.Tensor, lse: ?mod.Tensor, options: mod.AttentionOptions) !void {
        try checkSame(out, q);
        try checkSame(k, v);
        const d = try options.dims(out.shape, q.shape, k.shape, v.shape, if (lse) |t| t.shape else null);
        if (lse) |t| if (t.dtype != .f32) return error.DtypeMismatch;
        if (d.d > max_head_dim) return self.onCpu("attention", .{ out, q, k, v, lse, options });
        for ([_]mod.Tensor{ q, k, v }) |t| if (overlaps(out, t)) return error.Aliasing;
        _ = try count(q);
        _ = try count(k);
        const warps = d.b * d.tq * d.h;
        if (warps == 0 or d.d == 0) return;
        try self.launch(.attention_f32, try warpThreads(warps), .{ ptrOf(out), ptrOf(q), ptrOf(k), ptrOf(v), ptrOf(lse orelse out), attnDims(d, options), attnScale(d), @as(u32, @intFromBool(lse != null)) });
    }

    pub fn attentionBackward(self: *Self, dq: mod.Tensor, dk: mod.Tensor, dv: mod.Tensor, dout: mod.Tensor, q: mod.Tensor, k: mod.Tensor, v: mod.Tensor, out: mod.Tensor, lse: mod.Tensor, options: mod.AttentionOptions) !void {
        try checkSame(dq, q);
        try checkSame(dout, q);
        try checkSame(out, q);
        try checkSame(dk, k);
        try checkSame(dv, k);
        try checkSame(k, v);
        if (lse.dtype != .f32) return error.DtypeMismatch;
        const d = try options.dims(out.shape, q.shape, k.shape, v.shape, lse.shape);
        if (d.d > max_head_dim) return self.onCpu("attentionBackward", .{ dq, dk, dv, dout, q, k, v, out, lse, options });
        const inputs = [_]mod.Tensor{ dout, q, k, v, out, lse };
        for ([_]mod.Tensor{ dq, dk, dv }) |o| for (inputs) |t| if (overlaps(o, t)) return error.Aliasing;
        _ = try count(q);
        _ = try count(k);
        if (d.b == 0 or d.tq == 0 or d.h == 0 or d.d == 0) return;
        const dims = attnDims(d, options);
        const rows: u32 = @intCast(d.b * d.h * d.tq);
        const delta = try self.scratch(rows);
        try self.launch(.attention_delta_f32, rows, .{ ptrOf(delta), ptrOf(dout), ptrOf(out), dims });
        try self.launch(.attention_dq_f32, try warpThreads(d.b * d.tq * d.h), .{ ptrOf(dq), ptrOf(dout), ptrOf(q), ptrOf(k), ptrOf(v), ptrOf(lse), ptrOf(delta), dims, attnScale(d) });
        try self.launch(.attention_dkv_f32, try warpThreads(d.b * d.tk * d.hkv), .{ ptrOf(dk), ptrOf(dv), ptrOf(dout), ptrOf(q), ptrOf(k), ptrOf(v), ptrOf(lse), ptrOf(delta), dims, attnScale(d) });
    }

    fn attnDims(d: mod.AttentionOptions.Dims, options: mod.AttentionOptions) AttnDims {
        return .{
            .b = @intCast(d.b),
            .tq = @intCast(d.tq),
            .tk = @intCast(d.tk),
            .keys = @intCast(d.keys),
            .h = @intCast(d.h),
            .hkv = @intCast(d.hkv),
            .d = @intCast(d.d),
            .window = @intCast(@min(options.window, 1 << 30)),
        };
    }

    fn attnScale(d: mod.AttentionOptions.Dims) f32 {
        return 1 / @sqrt(@as(f32, @floatFromInt(d.d)));
    }

    /// Threads for one warp per row.
    fn warpThreads(rows: usize) !u32 {
        return std.math.cast(u32, rows * 32) orelse error.TensorTooLarge;
    }

    // -------------------------------------------------------------------------
    // Optimizer (norms and sums in f64, as the CPU's)

    pub fn adamwStep(self: *Self, p: mod.Tensor, g: mod.Tensor, m: mod.Tensor, v: mod.Tensor, params: mod.AdamWParams) !void {
        try checkSame(p, g);
        try checkSame(p, m);
        try checkSame(p, v);
        try params.validate();
        const n = try count(p);
        const step: f32 = @floatFromInt(params.step);
        try self.launch(.adamw_step_f32, n, .{
            ptrOf(p),                                                ptrOf(g),         ptrOf(m),         ptrOf(v),
            1 - params.lr * params.weight_decay,                     1 - params.beta1, 1 - params.beta2, 1 - std.math.pow(f32, params.beta2, step),
            params.lr / (1 - std.math.pow(f32, params.beta1, step)), params.eps,       n,
        });
    }

    pub fn muonMomentum(self: *Self, g: mod.Tensor, buf: mod.Tensor, momentum: f32) !void {
        try checkSame(g, buf);
        const n = try count(g);
        try self.launch(.muon_momentum_f32, n, .{ ptrOf(g), ptrOf(buf), momentum, n });
    }

    /// MuonEq row equilibration, then `x /= ||x||_F * 1.01 + 1e-6`.
    pub fn muonPrepare(self: *Self, x: mod.Tensor) !void {
        if (x.dtype != .f32) return error.DtypeMismatch;
        const rows = try rowCount(x);
        const cols: u32 = @intCast(x.shape.cols());
        const n = try count(x);
        if (n == 0) return;
        const row_sq = try self.scratch(2 * @as(usize, rows));
        const total = try self.scratch(2);
        try self.launchBlocks(.row_squares_f64, .{ rows, 1, 1 }, .{ ptrOf(x), ptrOf(row_sq), cols });
        try self.launchBlocks(.total_f64, .{ 1, 1, 1 }, .{ ptrOf(row_sq), ptrOf(total), rows });
        try self.launch(.muon_scale_rows_f32, n, .{ ptrOf(x), ptrOf(row_sq), ptrOf(total), rows, cols, n });
        try self.launchBlocks(.row_squares_f64, .{ rows, 1, 1 }, .{ ptrOf(x), ptrOf(row_sq), cols });
        try self.launchBlocks(.total_f64, .{ 1, 1, 1 }, .{ ptrOf(row_sq), ptrOf(total), rows });
        try self.launch(.muon_scale_all_f32, n, .{ ptrOf(x), ptrOf(total), n });
    }

    /// Muon+ renormalization, NorMuon variance reduction and the cautious update.
    pub fn muonFinish(self: *Self, p: mod.Tensor, g: mod.Tensor, second: mod.Tensor, params: mod.MuonParams) !void {
        try checkSame(p, g);
        if (second.dtype != .f32) return error.DtypeMismatch;
        const rows = try rowCount(p);
        const cols: u32 = @intCast(p.shape.cols());
        const by_row = rows >= cols;
        const shape = mod.MuonParams.secondShape(rows, cols);
        if (second.shape.rows() != shape[0] or second.shape.cols() != shape[1]) return error.ShapeMismatch;
        const n = try count(p);
        if (n == 0) return;

        const row_sq = try self.scratch(2 * @as(usize, rows));
        const total = try self.scratch(2);
        try self.launchBlocks(.row_squares_f64, .{ rows, 1, 1 }, .{ ptrOf(g), ptrOf(row_sq), cols });
        try self.launchBlocks(.total_f64, .{ 1, 1, 1 }, .{ ptrOf(row_sq), ptrOf(total), rows });
        const target: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(@min(rows, cols)))));
        try self.launch(.muon_renorm_f32, n, .{ ptrOf(g), ptrOf(total), target, n });

        // Squares over the reduced dimension: per row when tall, per column when wide.
        const reduced: u32 = @intCast(second.numel());
        const sums = try self.scratch(2 * @as(usize, reduced));
        if (by_row) {
            try self.launchBlocks(.row_squares_f64, .{ rows, 1, 1 }, .{ ptrOf(g), ptrOf(sums), cols });
        } else {
            try self.launch(.col_squares_f64, cols, .{ ptrOf(g), ptrOf(sums), rows, cols });
        }
        const ratio = try self.scratch(1);
        const red: f32 = @floatFromInt(if (by_row) cols else rows);
        try self.launchBlocks(.normuon_stats_f32, .{ 1, 1, 1 }, .{ ptrOf(sums), ptrOf(second), ptrOf(ratio), reduced, red, params.beta2 });
        try self.launch(.muon_update_f32, n, .{ ptrOf(p), ptrOf(g), ptrOf(second), ptrOf(ratio), params.lr, params.lr * params.weight_decay, cols, @as(u32, @intFromBool(by_row)), n });
    }

    // -------------------------------------------------------------------------
    // Launching

    fn onCpu(self: *Self, comptime op: []const u8, args: anytype) !void {
        try self.sync();
        return @call(.auto, @field(mod.CpuBackend, op), .{&self.cpu} ++ args);
    }

    /// Launches `kernel` over `threads` threads in blocks of `block_threads`.
    fn launch(self: *Self, kernel: Kernel, threads: u32, args: anytype) !void {
        if (threads == 0) return;
        try self.launchBlocks(kernel, .{ (threads + block_threads - 1) / block_threads, 1, 1 }, args);
    }

    /// Launches `kernel` over a `grid` of `block_threads`-thread blocks, with
    /// `args` passed by address as `cuLaunchKernel` wants.
    fn launchBlocks(self: *Self, kernel: Kernel, grid: [3]u32, args: anytype) !void {
        if (grid[0] == 0 or grid[1] == 0 or grid[2] == 0) return;
        try self.current();
        // A runtime copy of the arguments: constants in `args` are comptime
        // tuple fields, which have no address to hand the driver.
        const types = @typeInfo(@TypeOf(args)).@"struct".field_types;
        var values: @Tuple(types) = undefined;
        inline for (0..types.len) |i| values[i] = args[i];
        var params: [types.len]?*anyopaque = undefined;
        inline for (0..types.len) |i| params[i] = @ptrCast(&values[i]);
        self.check(self.driver.cuLaunchKernel(self.functions[@backingInt(kernel)], grid[0], grid[1], grid[2], block_threads, 1, 1, 0, self.stream, &params, null), "cuLaunchKernel") catch |err| {
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

    fn rowCount(t: mod.Tensor) !u32 {
        return std.math.cast(u32, t.shape.rows()) orelse error.TensorTooLarge;
    }

    /// Fails unless `values` holds one f32 per row of `rows_of`.
    fn checkRowValues(values: mod.Tensor, rows_of: mod.Tensor) !void {
        if (values.dtype != .f32) return error.DtypeMismatch;
        if (values.numel() != rows_of.shape.rows()) return error.ShapeMismatch;
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
