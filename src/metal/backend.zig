const std = @import("std");
const log = std.log.scoped(.zignanogpt_metal_backend);
const mod = @import("../module.zig");
const Objc = mod.Objc;
const Id = mod.ObjcId;

/// A Metal tensor's storage: a shared-storage `MTLBuffer` (Apple silicon's
/// unified memory) and its contents as host bytes.
pub const MetalBuffer = struct {
    bytes: []align(64) u8,
    /// Null for host-only tensors (`CpuBackend.alloc`).
    mtl: ?Id = null,
};

extern fn MTLCreateSystemDefaultDevice() ?Id;

/// The MSL sources, compiled when the backend starts: the kernels with fast
/// math, the reductions without (their compensated sums need IEEE rounding).
const kernel_source = @embedFile("kernels.metal");
const reduce_source = @embedFile("reduce.metal");

/// `MTLSize`.
const Size = extern struct { width: u64, height: u64 = 1, depth: u64 = 1 };

/// The compute kernels, by MSL function name.
const Kernel = enum {
    fill_f32,
    copy_u32,
    add_f32,
    mul_f32,
    scale_f32,
    combine_f32,
    relu_square_f32,
    relu_square_backward_f32,
    softcap_f32,
    embedding_f32,
    rmsnorm_f32,
    rmsnorm_backward_f32,
    matmul_f32,
    dot_partial_f32,
    reduce_sum_f32,
    rope_f32,
    gate_linear_f32,
    gate_linear_backward_dx_f32,
    gate_linear_backward_dw_f32,
    smear_f32,
    gated_add_f32,
    smear_backward_f32,
    value_mix_f32,
    value_mix_backward_f32,
    xent_rows_f32,
    xent_backward_f32,
    attention_delta_f32,
    attention_f32_d8,
    attention_f32_d16,
    attention_f32_d32,
    attention_f32_d64,
    attention_f32_d128,
    attention_dq_f32_d8,
    attention_dq_f32_d16,
    attention_dq_f32_d32,
    attention_dq_f32_d64,
    attention_dq_f32_d128,
    attention_dkv_f32_d8,
    attention_dkv_f32_d16,
    attention_dkv_f32_d32,
    attention_dkv_f32_d64,
    attention_dkv_f32_d128,
};

/// The Metal backend (macOS, Apple silicon): ops encoded into one compute
/// command buffer in issue order, run when `sync` (or a download) commits it.
///
/// Matmuls go to Metal Performance Shaders; the model's other per-layer ops
/// (attention, norms, rotary, gates, smear, value mix, dot products, cross
/// entropy) are MSL kernels. Tensors live in shared-storage buffers, so the
/// remaining ops (embedding backward, optimizer steps) run on the CPU kernels
/// over the same memory after a `sync`.
pub const MetalBackend = struct {
    const Self = @This();

    pub const Buffer = MetalBuffer;
    pub const name = "metal";

    pub const Options = struct {
        /// CPU worker threads for the ops that run on the CPU; 0 means one per CPU.
        threads: usize = 0,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    /// Runs the ops without a Metal kernel, on the same (shared) memory.
    cpu: mod.CpuBackend,
    device: Id,
    queue: Id,
    pipelines: [@typeInfo(Kernel).@"enum".fields.len]Id,
    /// `MPSMatrixMultiplication` kernels by shape, transposes, alpha, beta.
    gemms: std.AutoHashMapUnmanaged(GemmKey, Id) = .empty,
    /// The open command buffer and encoder, with their autorelease pool.
    command: ?Id = null,
    encoder: ?Id = null,
    pool: ?*anyopaque = null,
    /// Buffers to release once the open command buffer completes (freed
    /// tensors and per-op scratch).
    deferred: std.ArrayList(Id) = .empty,

    /// Opens the default Metal device and compiles the kernels.
    ///
    /// Parameters:
    /// - `allocator`: host-side bookkeeping.
    /// - `io`: the CPU fallback's thread pool.
    /// - `options`: CPU threads.
    ///
    /// Return: the backend; `error.NoMetalDevice`, `error.MetalCompileFailed`.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, options: Options) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const device = MTLCreateSystemDefaultDevice() orelse {
            log.warn("no Metal device", .{});
            return error.NoMetalDevice;
        };
        errdefer Objc.release(device);
        var cpu = try mod.CpuBackend.init(allocator, io, .{ .threads = options.threads });
        errdefer cpu.deinit();
        const queue = Objc.call0(?Id, device, "newCommandQueue") orelse return error.MetalInitFailed;
        errdefer Objc.release(queue);

        const pool = Objc.poolPush();
        defer Objc.poolPop(pool);
        var err: ?Id = null;
        const library = Objc.call3(?Id, Id, ?Id, *?Id, device, "newLibraryWithSource:options:error:", try Objc.string(kernel_source), null, &err) orelse {
            log.warn("Metal kernels failed to compile: {s}", .{Objc.errorText(err)});
            return error.MetalCompileFailed;
        };
        defer Objc.release(library);
        const strict = Objc.call0(?Id, Objc.call0(Id, try Objc.class("MTLCompileOptions"), "alloc"), "init") orelse return error.MetalInitFailed;
        defer Objc.release(strict);
        Objc.call1(void, bool, strict, "setFastMathEnabled:", false);
        const reduce_library = Objc.call3(?Id, Id, ?Id, *?Id, device, "newLibraryWithSource:options:error:", try Objc.string(reduce_source), strict, &err) orelse {
            log.warn("Metal reductions failed to compile: {s}", .{Objc.errorText(err)});
            return error.MetalCompileFailed;
        };
        defer Objc.release(reduce_library);
        var pipelines: [@typeInfo(Kernel).@"enum".fields.len]Id = undefined;
        var made: usize = 0;
        errdefer for (pipelines[0..made]) |p| Objc.release(p);
        inline for (@typeInfo(Kernel).@"enum".fields, 0..) |f, i| {
            const source = if (std.mem.eql(u8, f.name, "dot_partial_f32") or std.mem.eql(u8, f.name, "reduce_sum_f32")) reduce_library else library;
            const function = Objc.call1(?Id, Id, source, "newFunctionWithName:", try Objc.string(f.name)) orelse {
                log.warn("Metal kernel {s} missing", .{f.name});
                return error.MetalCompileFailed;
            };
            defer Objc.release(function);
            pipelines[i] = Objc.call2(?Id, Id, *?Id, device, "newComputePipelineStateWithFunction:error:", function, &err) orelse {
                log.warn("Metal pipeline {s} failed: {s}", .{ f.name, Objc.errorText(err) });
                return error.MetalCompileFailed;
            };
            made += 1;
        }
        return Self{ .allocator = allocator, .io = io, .cpu = cpu, .device = device, .queue = queue, .pipelines = pipelines };
    }

    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.sync() catch |err| log.warn("Metal work failed at shutdown [{t}]", .{err});
        self.deferred.deinit(self.allocator);
        var it = self.gemms.valueIterator();
        while (it.next()) |g| Objc.release(g.*);
        self.gemms.deinit(self.allocator);
        for (self.pipelines) |p| Objc.release(p);
        Objc.release(self.queue);
        Objc.release(self.device);
        self.cpu.deinit();
    }

    /// Allocates a zeroed tensor in a shared-storage buffer.
    pub fn alloc(self: *Self, dtype: mod.Dtype, dims: []const usize) !mod.Tensor {
        const shape = try mod.Shape.init(dims);
        const len = shape.numel() * dtype.size();
        // MTLResourceStorageModeShared = 0; Metal rejects empty buffers.
        const buffer = Objc.call2(?Id, u64, u64, self.device, "newBufferWithLength:options:", @max(len, 16), 0) orelse return error.OutOfMemory;
        const contents = Objc.call0(?[*]align(64) u8, buffer, "contents") orelse return error.OutOfMemory;
        const bytes = contents[0..len];
        @memset(bytes, 0);
        return mod.Tensor{ .buffer = .{ .bytes = bytes, .mtl = buffer }, .dtype = dtype, .shape = shape };
    }

    /// Frees a tensor once the work issued so far completes.
    pub fn free(self: *Self, tensor: mod.Tensor) void {
        const b = tensor.buffer.mtl orelse return self.cpu.free(tensor);
        if (self.command == null) return Objc.release(b);
        self.deferred.append(self.allocator, b) catch {
            self.sync() catch |err| log.warn("Metal work failed before a free [{t}]", .{err});
            Objc.release(b);
        };
    }

    /// A zeroed f32 scratch tensor, released after the open command buffer completes.
    fn scratch(self: *Self, len: usize) !mod.Tensor {
        const t = try self.alloc(.f32, &.{@max(len, 1)});
        _ = try self.commandBuffer();
        try self.deferred.append(self.allocator, t.buffer.mtl.?);
        return t;
    }

    /// Commits the open command buffer and waits for it.
    ///
    /// Parameters:
    /// - `self`: the backend.
    ///
    /// Return: nothing; `error.MetalCommandFailed` when the GPU work failed.
    pub fn sync(self: *Self) !void {
        const command = self.command orelse return;
        if (self.encoder) |e| Objc.call0(void, e, "endEncoding");
        Objc.call0(void, command, "commit");
        Objc.call0(void, command, "waitUntilCompleted");
        // MTLCommandBufferStatusError = 5
        const status = Objc.call0(u64, command, "status");
        const err = Objc.call0(?Id, command, "error");
        const failed = status == 5;
        if (failed) log.warn("Metal command buffer failed: {s}", .{Objc.errorText(err)});
        self.command = null;
        self.encoder = null;
        Objc.poolPop(self.pool);
        self.pool = null;
        for (self.deferred.items) |b| Objc.release(b);
        self.deferred.clearRetainingCapacity();
        if (failed) return error.MetalCommandFailed;
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
    // Ops with Metal kernels

    pub fn fill(self: *Self, tensor: mod.Tensor, value: f32) !void {
        if (tensor.dtype != .f32) return error.DtypeMismatch;
        const n = try count(tensor);
        const e = try self.begin(.fill_f32);
        setTensor(e, tensor, 0);
        setValue(e, f32, value, 1);
        setValue(e, u32, n, 2);
        dispatch(e, n);
    }

    pub fn copy(self: *Self, dst: mod.Tensor, src: mod.Tensor) !void {
        if (dst.dtype != src.dtype) return error.DtypeMismatch;
        if (dst.numel() != src.numel()) return error.ShapeMismatch;
        // Overlapping copies need memmove order: leave them to the CPU.
        if (overlaps(dst, src)) return self.onCpu("copy", .{ dst, src });
        const n = try count(dst);
        const e = try self.begin(.copy_u32);
        setTensor(e, dst, 0);
        setTensor(e, src, 1);
        setValue(e, u32, n, 2);
        dispatch(e, n);
    }

    pub fn add(self: *Self, out: mod.Tensor, a: mod.Tensor, b: mod.Tensor) !void {
        try self.binary(.add_f32, out, a, b);
    }

    pub fn mul(self: *Self, out: mod.Tensor, a: mod.Tensor, b: mod.Tensor) !void {
        try self.binary(.mul_f32, out, a, b);
    }

    pub fn scale(self: *Self, out: mod.Tensor, a: mod.Tensor, s: f32) !void {
        try checkSame(out, a);
        const n = try count(out);
        const e = try self.begin(.scale_f32);
        setTensor(e, out, 0);
        setTensor(e, a, 1);
        setValue(e, f32, s, 2);
        setValue(e, u32, n, 3);
        dispatch(e, n);
    }

    pub fn combine(self: *Self, out: mod.Tensor, x: mod.Tensor, xs: mod.Scalar, y: mod.Tensor, ys: mod.Scalar) !void {
        try checkSame(out, x);
        try checkSame(out, y);
        try xs.validate();
        try ys.validate();
        const n = try count(out);
        const e = try self.begin(.combine_f32);
        setTensor(e, out, 0);
        setTensor(e, x, 1);
        setTensor(e, y, 2);
        // An absent scalar tensor still needs a bound buffer: x stands in.
        setTensor(e, xs.tensor orelse x, 3);
        setTensor(e, ys.tensor orelse x, 4);
        const Args = extern struct { x_factor: f32, y_factor: f32, x_has: u32, y_has: u32, n: u32 };
        setValue(e, Args, .{ .x_factor = xs.factor, .y_factor = ys.factor, .x_has = @intFromBool(xs.tensor != null), .y_has = @intFromBool(ys.tensor != null), .n = n }, 5);
        dispatch(e, n);
    }

    pub fn reluSquare(self: *Self, out: mod.Tensor, x: mod.Tensor) !void {
        try checkSame(out, x);
        const n = try count(out);
        const e = try self.begin(.relu_square_f32);
        setTensor(e, out, 0);
        setTensor(e, x, 1);
        setValue(e, u32, n, 2);
        dispatch(e, n);
    }

    pub fn reluSquareBackward(self: *Self, dx: mod.Tensor, dy: mod.Tensor, x: mod.Tensor) !void {
        try checkSame(dx, x);
        try checkSame(dy, x);
        const n = try count(dx);
        const e = try self.begin(.relu_square_backward_f32);
        setTensor(e, dx, 0);
        setTensor(e, dy, 1);
        setTensor(e, x, 2);
        setValue(e, u32, n, 3);
        dispatch(e, n);
    }

    pub fn softcap(self: *Self, out: mod.Tensor, logits: mod.Tensor, cap: f32) !void {
        if (out.dtype != .f32 or logits.dtype != .f32) return error.DtypeMismatch;
        if (out.shape.rows() != logits.shape.rows() or out.shape.cols() > logits.shape.cols()) return error.ShapeMismatch;
        if (overlaps(out, logits)) return error.Aliasing;
        const n = try count(out);
        const e = try self.begin(.softcap_f32);
        setTensor(e, out, 0);
        setTensor(e, logits, 1);
        const Args = extern struct { cols: u32, padded: u32, cap: f32, n: u32 };
        setValue(e, Args, .{ .cols = @intCast(out.shape.cols()), .padded = @intCast(logits.shape.cols()), .cap = cap, .n = n }, 2);
        dispatch(e, n);
    }

    pub fn embedding(self: *Self, out: mod.Tensor, table: mod.Tensor, ids: mod.Tensor) !void {
        if (out.dtype != .f32 or table.dtype != .f32 or ids.dtype != .i32) return error.DtypeMismatch;
        const cols = table.shape.cols();
        if (out.numel() != ids.numel() * cols) return error.ShapeMismatch;
        // The ids are checked on the host (the contract forbids reading out of bounds).
        try self.sync();
        const vocab = table.shape.rows();
        const id_bytes = ids.buffer.bytes[ids.offset * 4 ..][0 .. ids.numel() * 4];
        for (std.mem.bytesAsSlice(i32, id_bytes)) |id| {
            if (id < 0 or id >= vocab) {
                log.debug("embedding id {d} outside vocab {d}", .{ id, vocab });
                return error.OutOfBounds;
            }
        }
        const n = try count(out);
        const e = try self.begin(.embedding_f32);
        setTensor(e, out, 0);
        setTensor(e, table, 1);
        setTensor(e, ids, 2);
        const Args = extern struct { cols: u32, n: u32 };
        setValue(e, Args, .{ .cols = @intCast(cols), .n = n }, 3);
        dispatch(e, n);
    }

    pub fn rmsnorm(self: *Self, out: mod.Tensor, x: mod.Tensor, eps: f32) !void {
        try checkSame(out, x);
        const e = try self.begin(.rmsnorm_f32);
        setTensor(e, out, 0);
        setTensor(e, x, 1);
        const Args = extern struct { cols: u32, eps: f32, accumulate: u32 };
        setValue(e, Args, .{ .cols = @intCast(x.shape.cols()), .eps = eps, .accumulate = 0 }, 2);
        dispatchRows(e, x.shape.rows(), x.shape.cols());
    }

    pub fn rmsnormBackward(self: *Self, dx: mod.Tensor, dy: mod.Tensor, x: mod.Tensor, eps: f32, accumulate: bool) !void {
        try checkSame(dx, x);
        try checkSame(dy, x);
        const e = try self.begin(.rmsnorm_backward_f32);
        setTensor(e, dx, 0);
        setTensor(e, dy, 1);
        setTensor(e, x, 2);
        const Args = extern struct { cols: u32, eps: f32, accumulate: u32 };
        setValue(e, Args, .{ .cols = @intCast(x.shape.cols()), .eps = eps, .accumulate = @intFromBool(accumulate) }, 3);
        dispatchRows(e, x.shape.rows(), x.shape.cols());
    }

    /// `c (+)= alpha * op(a) @ op(b)` with Metal Performance Shaders (its
    /// GEMM is tuned per GPU); the MSL kernel covers what MPS cannot take.
    pub fn matmul(self: *Self, c: mod.Tensor, a: mod.Tensor, b: mod.Tensor, options: mod.MatmulOptions) !void {
        if (c.dtype != .f32 or a.dtype != .f32 or b.dtype != .f32) return error.DtypeMismatch;
        const m, const n, const k = try options.dims(c.shape, a.shape, b.shape);
        if (overlaps(c, a) or overlaps(c, b)) return error.Aliasing;
        if (m == 0 or n == 0) return;
        if (k > 0) return self.gemm(c, a, b, options, m, n, k);
        const e = try self.begin(.matmul_f32);
        setTensor(e, c, 0);
        setTensor(e, a, 1);
        setTensor(e, b, 2);
        const Args = extern struct { m: u32, n: u32, k: u32, transpose_a: u32, transpose_b: u32, accumulate: u32, alpha: f32 };
        setValue(e, Args, .{
            .m = @intCast(m),
            .n = @intCast(n),
            .k = @intCast(k),
            .transpose_a = @intFromBool(options.transpose_a),
            .transpose_b = @intFromBool(options.transpose_b),
            .accumulate = @intFromBool(options.accumulate),
            .alpha = options.alpha,
        }, 3);
        Objc.call2(void, Size, Size, e, "dispatchThreadgroups:threadsPerThreadgroup:", .{ .width = (n + 63) / 64, .height = (m + 63) / 64 }, .{ .width = 128 });
    }

    const GemmKey = struct { m: usize, n: usize, k: usize, ta: bool, tb: bool, alpha: u32, beta: bool };

    /// Encodes an `MPSMatrixMultiplication` (between compute encoders).
    fn gemm(self: *Self, c: mod.Tensor, a: mod.Tensor, b: mod.Tensor, options: mod.MatmulOptions, m: usize, n: usize, k: usize) !void {
        const command = try self.commandBuffer();
        const key = GemmKey{ .m = m, .n = n, .k = k, .ta = options.transpose_a, .tb = options.transpose_b, .alpha = @bitCast(options.alpha), .beta = options.accumulate };
        const kernel = self.gemms.get(key) orelse blk: {
            const raw = Objc.call0(?Id, try Objc.class("MPSMatrixMultiplication"), "alloc") orelse return error.MetalInitFailed;
            const F = *const fn (Id, mod.ObjcSel, Id, bool, bool, u64, u64, u64, f64, f64) callconv(.c) ?Id;
            const made = Objc.function(F)(raw, Objc.sel("initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:"), self.device, options.transpose_a, options.transpose_b, m, n, k, options.alpha, if (options.accumulate) 1 else 0) orelse return error.MetalInitFailed;
            try self.gemms.put(self.allocator, key, made);
            break :blk made;
        };
        // Stored layouts: A is [M, K] or [K, M], B is [K, N] or [N, K], C is [M, N].
        const left = try matrix(a, if (options.transpose_a) k else m, if (options.transpose_a) m else k);
        const right = try matrix(b, if (options.transpose_b) n else k, if (options.transpose_b) k else n);
        const result = try matrix(c, m, n);
        const Encode = *const fn (Id, mod.ObjcSel, Id, Id, Id, Id) callconv(.c) void;
        Objc.function(Encode)(kernel, Objc.sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"), command, left, right, result);
    }

    /// An autoreleased `MPSMatrix` over a tensor's elements, `rows x cols` row-major.
    fn matrix(t: mod.Tensor, rows: usize, cols: usize) !Id {
        // MPSDataTypeFloat32 = 0x10000000 | 32
        const F = *const fn (Id, mod.ObjcSel, u64, u64, u64, u32) callconv(.c) ?Id;
        const desc = Objc.function(F)(try Objc.class("MPSMatrixDescriptor"), Objc.sel("matrixDescriptorWithRows:columns:rowBytes:dataType:"), rows, cols, cols * 4, 0x10000020) orelse return error.MetalInitFailed;
        const raw = Objc.call0(?Id, try Objc.class("MPSMatrix"), "alloc") orelse return error.MetalInitFailed;
        const made = Objc.call3(?Id, Id, u64, Id, raw, "initWithBuffer:offset:descriptor:", t.buffer.mtl.?, t.offset * 4, desc) orelse return error.MetalInitFailed;
        return Objc.call0(Id, made, "autorelease");
    }

    // -------------------------------------------------------------------------
    // Ops on the CPU kernels (shared memory, after a sync)

    pub fn rope(self: *Self, out: mod.Tensor, x: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize) !void {
        try self.rotate(out, x, cos, sin, pos0, 1);
    }
    pub fn ropeBackward(self: *Self, dx: mod.Tensor, dy: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize) !void {
        try self.rotate(dx, dy, cos, sin, pos0, -1);
    }

    fn rotate(self: *Self, out: mod.Tensor, x: mod.Tensor, cos: mod.Tensor, sin: mod.Tensor, pos0: usize, sign: f32) !void {
        try checkSame(out, x);
        try checkSame(cos, sin);
        if (x.shape.rank != 4 or x.shape.dims[3] % 2 != 0 or cos.shape.cols() * 2 != x.shape.dims[3]) return error.ShapeMismatch;
        if (pos0 + x.shape.dims[1] > cos.shape.rows()) return error.OutOfBounds;
        const half = x.shape.dims[3] / 2;
        const n = try count(x) / 2;
        const e = try self.begin(.rope_f32);
        setTensor(e, out, 0);
        setTensor(e, x, 1);
        setTensor(e, cos, 2);
        setTensor(e, sin, 3);
        const Args = extern struct { t: u32, h: u32, half_d: u32, pos0: u32, sign: f32, n: u32 };
        setValue(e, Args, .{ .t = @intCast(x.shape.dims[1]), .h = @intCast(x.shape.dims[2]), .half_d = @intCast(half), .pos0 = @intCast(pos0), .sign = sign, .n = n }, 4);
        dispatch(e, n);
    }

    const AttnArgs = extern struct { b: u32, tq: u32, tk: u32, keys: u32, h: u32, hkv: u32, d: u32, window: u32, scale: f32, has_lse: u32 };

    /// Queries (forward, dq) or keys (dk/dv) per attention threadgroup.
    const attn_block = 32;

    /// Flash attention on simdgroup matrices (`kernels.metal`); head
    /// dimensions past 128 run on the CPU kernels.
    pub fn attention(self: *Self, out: mod.Tensor, q: mod.Tensor, k: mod.Tensor, v: mod.Tensor, lse: ?mod.Tensor, options: mod.AttentionOptions) !void {
        try checkSame(out, q);
        try checkSame(k, v);
        const d = try options.dims(out.shape, q.shape, k.shape, v.shape, if (lse) |t| t.shape else null);
        if (lse) |t| if (t.dtype != .f32) return error.DtypeMismatch;
        const padded = paddedHeadDim(d.d) orelse return self.onCpu("attention", .{ out, q, k, v, lse, options });
        for ([_]mod.Tensor{ q, k, v }) |t| if (overlaps(out, t)) return error.Aliasing;
        _ = try count(q);
        _ = try count(k);
        if (d.b == 0 or d.tq == 0 or d.h == 0 or d.d == 0) return;
        const e = try self.begin(attnKernel("attention_f32", padded));
        for ([_]mod.Tensor{ out, q, k, v, lse orelse out }, 0..) |t, i| setTensor(e, t, i);
        setValue(e, AttnArgs, attnArgs(d, options, lse != null), 5);
        dispatchGroups(e, .{ .width = (d.tq + attn_block - 1) / attn_block, .height = d.h, .depth = d.b });
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
        const padded = paddedHeadDim(d.d) orelse return self.onCpu("attentionBackward", .{ dq, dk, dv, dout, q, k, v, out, lse, options });
        const inputs = [_]mod.Tensor{ dout, q, k, v, out, lse };
        for ([_]mod.Tensor{ dq, dk, dv }) |o| for (inputs) |t| if (overlaps(o, t)) return error.Aliasing;
        _ = try count(q);
        _ = try count(k);
        if (d.b == 0 or d.tq == 0 or d.h == 0 or d.d == 0) return;
        const args = attnArgs(d, options, true);
        const rows = d.b * d.h * d.tq;
        const delta = try self.scratch(rows);
        var e = try self.begin(.attention_delta_f32);
        for ([_]mod.Tensor{ delta, dout, out }, 0..) |t, i| setTensor(e, t, i);
        setValue(e, AttnArgs, args, 3);
        dispatch(e, @intCast(rows));
        e = try self.begin(attnKernel("attention_dq_f32", padded));
        for ([_]mod.Tensor{ dq, dout, q, k, v, lse, delta }, 0..) |t, i| setTensor(e, t, i);
        setValue(e, AttnArgs, args, 7);
        dispatchGroups(e, .{ .width = (d.tq + attn_block - 1) / attn_block, .height = d.h, .depth = d.b });
        e = try self.begin(attnKernel("attention_dkv_f32", padded));
        for ([_]mod.Tensor{ dk, dv, dout, q, k, v, lse, delta }, 0..) |t, i| setTensor(e, t, i);
        setValue(e, AttnArgs, args, 8);
        dispatchGroups(e, .{ .width = (d.tk + attn_block - 1) / attn_block, .height = d.hkv, .depth = d.b });
    }

    fn attnArgs(d: mod.AttentionOptions.Dims, options: mod.AttentionOptions, has_lse: bool) AttnArgs {
        return .{
            .b = @intCast(d.b),
            .tq = @intCast(d.tq),
            .tk = @intCast(d.tk),
            .keys = @intCast(d.keys),
            .h = @intCast(d.h),
            .hkv = @intCast(d.hkv),
            .d = @intCast(d.d),
            .window = @intCast(@min(options.window, 1 << 30)),
            .scale = 1 / @sqrt(@as(f32, @floatFromInt(d.d))),
            .has_lse = @intFromBool(has_lse),
        };
    }

    /// The kernels' padded head dimension (a compiled instantiation), or null past 128.
    fn paddedHeadDim(d: usize) ?usize {
        inline for (.{ 8, 16, 32, 64, 128 }) |p| if (d <= p) return p;
        return null;
    }

    fn attnKernel(comptime prefix: []const u8, padded: usize) Kernel {
        return switch (padded) {
            8 => @field(Kernel, prefix ++ "_d8"),
            16 => @field(Kernel, prefix ++ "_d16"),
            32 => @field(Kernel, prefix ++ "_d32"),
            64 => @field(Kernel, prefix ++ "_d64"),
            else => @field(Kernel, prefix ++ "_d128"),
        };
    }

    const GateArgs = extern struct { cols: u32, cin: u32, heads: u32, rows: u32 };

    pub fn gateLinear(self: *Self, out: mod.Tensor, x: mod.Tensor, w: mod.Tensor) !void {
        if (out.dtype != .f32 or x.dtype != .f32 or w.dtype != .f32) return error.DtypeMismatch;
        const heads = w.shape.rows();
        const cin = w.shape.cols();
        if (cin > x.shape.cols() or out.shape.rows() != x.shape.rows() or out.shape.cols() != heads) return error.ShapeMismatch;
        const e = try self.begin(.gate_linear_f32);
        setTensor(e, out, 0);
        setTensor(e, x, 1);
        setTensor(e, w, 2);
        setValue(e, GateArgs, .{ .cols = @intCast(x.shape.cols()), .cin = @intCast(cin), .heads = @intCast(heads), .rows = @intCast(x.shape.rows()) }, 3);
        dispatch(e, try count(out));
    }

    pub fn gateLinearBackward(self: *Self, dx: mod.Tensor, dw: mod.Tensor, dout: mod.Tensor, x: mod.Tensor, w: mod.Tensor) !void {
        try checkSame(dx, x);
        try checkSame(dw, w);
        if (dout.dtype != .f32) return error.DtypeMismatch;
        const rows = x.shape.rows();
        const heads = w.shape.rows();
        const cin = w.shape.cols();
        if (cin > x.shape.cols() or dout.shape.rows() != rows or dout.shape.cols() != heads) return error.ShapeMismatch;
        const args = GateArgs{ .cols = @intCast(x.shape.cols()), .cin = @intCast(cin), .heads = @intCast(heads), .rows = @intCast(rows) };
        var e = try self.begin(.gate_linear_backward_dx_f32);
        setTensor(e, dx, 0);
        setTensor(e, dout, 1);
        setTensor(e, w, 2);
        setValue(e, GateArgs, args, 3);
        dispatch(e, @intCast(rows * cin));
        e = try self.begin(.gate_linear_backward_dw_f32);
        setTensor(e, dw, 0);
        setTensor(e, dout, 1);
        setTensor(e, x, 2);
        setValue(e, GateArgs, args, 3);
        dispatch(e, @intCast(heads * cin));
    }

    const SmearArgs = extern struct { t: u32, cols: u32, factor: f32, has_lambda: u32, n: u32 };

    pub fn smear(self: *Self, out: mod.Tensor, x: mod.Tensor, gate: mod.Tensor, lambda: mod.Scalar) !void {
        try checkSame(out, x);
        try lambda.validate();
        if (gate.dtype != .f32) return error.DtypeMismatch;
        if (x.shape.rank != 3 or gate.numel() != x.shape.dims[0] * x.shape.dims[1]) return error.ShapeMismatch;
        if (overlaps(out, x)) return error.Aliasing;
        const n = try count(out);
        const e = try self.begin(.smear_f32);
        setTensor(e, out, 0);
        setTensor(e, x, 1);
        setTensor(e, gate, 2);
        setTensor(e, lambda.tensor orelse gate, 3);
        setValue(e, SmearArgs, .{ .t = @intCast(x.shape.dims[1]), .cols = @intCast(x.shape.dims[2]), .factor = lambda.factor, .has_lambda = @intFromBool(lambda.tensor != null), .n = n }, 4);
        dispatch(e, n);
    }

    pub fn smearBackward(self: *Self, dx: mod.Tensor, dgate: mod.Tensor, dlambda: mod.Tensor, dout: mod.Tensor, x: mod.Tensor, gate: mod.Tensor, lambda: mod.Scalar) !void {
        try checkSame(dx, x);
        try checkSame(dout, x);
        try checkSame(dgate, gate);
        try lambda.validate();
        if (dlambda.dtype != .f32 or dlambda.numel() == 0) return error.DtypeMismatch;
        if (x.shape.rank != 3 or gate.numel() != x.shape.dims[0] * x.shape.dims[1]) return error.ShapeMismatch;
        if (overlaps(dx, dout) or overlaps(dx, x)) return error.Aliasing;
        const rows = x.shape.rows();
        const partial = try self.scratch(rows);
        const e = try self.begin(.smear_backward_f32);
        for ([_]mod.Tensor{ dx, dgate, partial, dout, x, gate, lambda.tensor orelse gate }, 0..) |t, i| setTensor(e, t, i);
        setValue(e, SmearArgs, .{ .t = @intCast(x.shape.dims[1]), .cols = @intCast(x.shape.dims[2]), .factor = lambda.factor, .has_lambda = @intFromBool(lambda.tensor != null), .n = 0 }, 7);
        dispatchRows(e, rows, x.shape.dims[2]);
        try self.reduce(partial, rows, dlambda, lambda.factor, true);
    }

    pub fn gatedAdd(self: *Self, out: mod.Tensor, x: mod.Tensor, y: mod.Tensor, gate: mod.Tensor, s: mod.Scalar) !void {
        try checkSame(out, x);
        try checkSame(out, y);
        try s.validate();
        if (gate.dtype != .f32) return error.DtypeMismatch;
        if (gate.numel() != x.shape.rows()) return error.ShapeMismatch;
        const n = try count(out);
        const e = try self.begin(.gated_add_f32);
        for ([_]mod.Tensor{ out, x, y, gate, s.tensor orelse gate }, 0..) |t, i| setTensor(e, t, i);
        setValue(e, SmearArgs, .{ .t = 1, .cols = @intCast(x.shape.cols()), .factor = s.factor, .has_lambda = @intFromBool(s.tensor != null), .n = n }, 5);
        dispatch(e, n);
    }

    const MixArgs = extern struct { d: u32, pairs: u32 };

    pub fn valueMix(self: *Self, v: mod.Tensor, ve: mod.Tensor, gate: mod.Tensor) !void {
        if (v.dtype != .f32 or ve.dtype != .f32 or gate.dtype != .f32) return error.DtypeMismatch;
        const pairs = gate.numel();
        if (v.numel() != ve.numel() or pairs == 0 or v.numel() % pairs != 0) return error.ShapeMismatch;
        const e = try self.begin(.value_mix_f32);
        setTensor(e, v, 0);
        setTensor(e, ve, 1);
        setTensor(e, gate, 2);
        setValue(e, MixArgs, .{ .d = @intCast(v.numel() / pairs), .pairs = @intCast(pairs) }, 3);
        dispatch(e, try count(v));
    }

    pub fn valueMixBackward(self: *Self, dve: mod.Tensor, dgate: mod.Tensor, dv: mod.Tensor, ve: mod.Tensor, gate: mod.Tensor) !void {
        try checkSame(dve, ve);
        try checkSame(dgate, gate);
        if (dv.dtype != .f32) return error.DtypeMismatch;
        const pairs = gate.numel();
        if (dv.numel() != ve.numel() or pairs == 0 or ve.numel() % pairs != 0) return error.ShapeMismatch;
        const e = try self.begin(.value_mix_backward_f32);
        for ([_]mod.Tensor{ dve, dgate, dv, ve, gate }, 0..) |t, i| setTensor(e, t, i);
        setValue(e, MixArgs, .{ .d = @intCast(ve.numel() / pairs), .pairs = @intCast(pairs) }, 5);
        dispatch(e, @intCast(pairs));
    }

    const XentArgs = extern struct { vocab: u32, padded: u32, cap: f32, factor: f32, has_weights: u32 };

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

    pub fn crossEntropyRows(self: *Self, losses: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor) !void {
        _ = try self.checkTargets(logits, targets);
        if (losses.dtype != .f32 or losses.numel() != logits.shape.rows()) return error.ShapeMismatch;
        try self.xentRows(losses, logits, targets);
    }

    fn xentRows(self: *Self, losses: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor) !void {
        const e = try self.begin(.xent_rows_f32);
        setTensor(e, losses, 0);
        setTensor(e, logits, 1);
        setTensor(e, targets, 2);
        setValue(e, XentArgs, .{ .vocab = @intCast(logits.shape.cols()), .padded = @intCast(logits.shape.cols()), .cap = 1, .factor = 0, .has_weights = 0 }, 3);
        dispatchRows(e, logits.shape.rows(), logits.shape.cols());
    }

    pub fn crossEntropy(self: *Self, loss: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor) !void {
        const valid = try self.checkTargets(logits, targets);
        if (loss.dtype != .f32 or loss.numel() == 0) return error.DtypeMismatch;
        const rows = logits.shape.rows();
        const losses = try self.scratch(rows);
        try self.xentRows(losses, logits, targets);
        // The mean over valid rows (NaN when there are none, as PyTorch).
        try self.reduce(losses, rows, loss, if (valid == 0) std.math.nan(f32) else 1 / @as(f32, @floatFromInt(valid)), false);
    }

    pub fn crossEntropyBackward(self: *Self, dpad: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor, cap: f32, scale_by: f32) !void {
        const valid = try self.checkTargets(logits, targets);
        try self.xentBackward(dpad, logits, targets, null, cap, scale_by / @as(f32, @floatFromInt(valid)));
    }

    pub fn crossEntropyWeightedBackward(self: *Self, dpad: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor, weights: mod.Tensor, cap: f32) !void {
        _ = try self.checkTargets(logits, targets);
        if (weights.dtype != .f32) return error.DtypeMismatch;
        if (weights.numel() != logits.shape.rows()) return error.ShapeMismatch;
        try self.xentBackward(dpad, logits, targets, weights, cap, 1);
    }

    fn xentBackward(self: *Self, dpad: mod.Tensor, logits: mod.Tensor, targets: mod.Tensor, weights: ?mod.Tensor, cap: f32, factor: f32) !void {
        if (dpad.dtype != .f32) return error.DtypeMismatch;
        if (dpad.shape.rows() != logits.shape.rows() or dpad.shape.cols() < logits.shape.cols()) return error.ShapeMismatch;
        if (overlaps(dpad, logits)) return error.Aliasing;
        const e = try self.begin(.xent_backward_f32);
        setTensor(e, dpad, 0);
        setTensor(e, logits, 1);
        setTensor(e, targets, 2);
        setTensor(e, weights orelse logits, 3);
        setValue(e, XentArgs, .{ .vocab = @intCast(logits.shape.cols()), .padded = @intCast(dpad.shape.cols()), .cap = cap, .factor = factor, .has_weights = @intFromBool(weights != null) }, 4);
        dispatchRows(e, logits.shape.rows(), dpad.shape.cols());
    }

    pub fn dot(self: *Self, out: mod.Tensor, a: mod.Tensor, b: mod.Tensor, factor: f32, accumulate: bool) !void {
        try checkSame(a, b);
        if (out.dtype != .f32 or out.numel() == 0) return error.DtypeMismatch;
        const n = try count(a);
        const groups: usize = @max(1, @min(256, (n + 4095) / 4096));
        const partial = try self.scratch(2 * groups); // (hi, lo) per group
        const e = try self.begin(.dot_partial_f32);
        setTensor(e, a, 0);
        setTensor(e, b, 1);
        setTensor(e, partial, 2);
        setValue(e, u32, n, 3);
        Objc.call2(void, Size, Size, e, "dispatchThreadgroups:threadsPerThreadgroup:", .{ .width = groups }, .{ .width = 256 });
        try self.reducePairs(partial, groups, out, factor, accumulate, true);
    }

    /// `out[0] (+)= factor * sum(values[0 .. n])` on the GPU, summed as double-floats.
    fn reduce(self: *Self, values: mod.Tensor, n: usize, out: mod.Tensor, factor: f32, accumulate: bool) !void {
        try self.reducePairs(values, n, out, factor, accumulate, false);
    }

    fn reducePairs(self: *Self, values: mod.Tensor, n: usize, out: mod.Tensor, factor: f32, accumulate: bool, pairs: bool) !void {
        const e = try self.begin(.reduce_sum_f32);
        setTensor(e, values, 0);
        setTensor(e, out, 1);
        const Args = extern struct { count: u32, factor: f32, accumulate: u32, pairs: u32 };
        setValue(e, Args, .{ .count = @intCast(n), .factor = factor, .accumulate = @intFromBool(accumulate), .pairs = @intFromBool(pairs) }, 2);
        Objc.call2(void, Size, Size, e, "dispatchThreadgroups:threadsPerThreadgroup:", .{ .width = 1 }, .{ .width = 256 });
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
    // Encoding

    /// Runs a `CpuBackend` op once the queued GPU work is done.
    fn onCpu(self: *Self, comptime op: []const u8, args: anytype) !void {
        try self.sync();
        return @call(.auto, @field(mod.CpuBackend, op), .{&self.cpu} ++ args);
    }

    /// The open command buffer with no encoder open (for MPS), creating it if needed.
    fn commandBuffer(self: *Self) !Id {
        if (self.encoder) |e| {
            Objc.call0(void, e, "endEncoding");
            self.encoder = null;
        }
        if (self.command) |c| return c;
        self.pool = Objc.poolPush();
        errdefer {
            Objc.poolPop(self.pool);
            self.pool = null;
        }
        const command = Objc.call0(?Id, self.queue, "commandBuffer") orelse return error.MetalCommandFailed;
        self.command = command;
        return command;
    }

    /// The compute encoder, with `kernel`'s pipeline set; opens one if needed.
    fn begin(self: *Self, kernel: Kernel) !Id {
        if (self.encoder == null) {
            const command = try self.commandBuffer();
            self.encoder = Objc.call0(?Id, command, "computeCommandEncoder") orelse return error.MetalCommandFailed;
        }
        const e = self.encoder.?;
        Objc.call1(void, Id, e, "setComputePipelineState:", self.pipelines[@intFromEnum(kernel)]);
        return e;
    }

    fn binary(self: *Self, kernel: Kernel, out: mod.Tensor, a: mod.Tensor, b: mod.Tensor) !void {
        try checkSame(out, a);
        try checkSame(out, b);
        const n = try count(out);
        const e = try self.begin(kernel);
        setTensor(e, out, 0);
        setTensor(e, a, 1);
        setTensor(e, b, 2);
        setValue(e, u32, n, 3);
        dispatch(e, n);
    }

    fn setTensor(e: Id, t: mod.Tensor, index: u64) void {
        Objc.call3(void, Id, u64, u64, e, "setBuffer:offset:atIndex:", t.buffer.mtl.?, t.offset * t.dtype.size(), index);
    }

    fn setValue(e: Id, comptime T: type, value: T, index: u64) void {
        Objc.call3(void, *const T, u64, u64, e, "setBytes:length:atIndex:", &value, @sizeOf(T), index);
    }

    /// One thread per element.
    fn dispatch(e: Id, n: u32) void {
        if (n == 0) return;
        Objc.call2(void, Size, Size, e, "dispatchThreads:threadsPerThreadgroup:", .{ .width = n }, .{ .width = @min(n, 256) });
    }

    /// `groups` threadgroups of 128 threads (4 simdgroups).
    fn dispatchGroups(e: Id, groups: Size) void {
        Objc.call2(void, Size, Size, e, "dispatchThreadgroups:threadsPerThreadgroup:", groups, .{ .width = 128 });
    }

    /// One threadgroup per row.
    fn dispatchRows(e: Id, rows: usize, cols: usize) void {
        if (rows == 0) return;
        const threads: u64 = @min(1024, @max(32, std.math.ceilPowerOfTwoAssert(usize, @max(cols, 1))));
        Objc.call2(void, Size, Size, e, "dispatchThreadgroups:threadsPerThreadgroup:", .{ .width = rows }, .{ .width = @min(threads, 256) });
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

test "metal backend passes the conformance suite" {
    try mod.Conformance.runAll(MetalBackend, std.testing.allocator, std.testing.io);
}

test "metal flash attention matches the cpu kernels at model sizes" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var gpu = try MetalBackend.init(allocator, io, .{});
    defer gpu.deinit();
    var cpu = try mod.CpuBackend.init(allocator, io, .{});
    defer cpu.deinit();
    const Case = struct { b: usize, tq: usize, tk: usize, keys: usize, h: usize, hkv: usize, d: usize, window: usize, backward: bool };
    const cases = [_]Case{
        .{ .b = 2, .tq = 100, .tk = 100, .keys = 100, .h = 4, .hkv = 2, .d = 64, .window = 1000, .backward = true },
        .{ .b = 1, .tq = 77, .tk = 77, .keys = 77, .h = 2, .hkv = 2, .d = 64, .window = 20, .backward = true },
        .{ .b = 1, .tq = 70, .tk = 70, .keys = 70, .h = 4, .hkv = 1, .d = 128, .window = 33, .backward = true },
        .{ .b = 1, .tq = 50, .tk = 50, .keys = 50, .h = 3, .hkv = 3, .d = 40, .window = 1000, .backward = true },
        .{ .b = 3, .tq = 1, .tk = 96, .keys = 61, .h = 4, .hkv = 2, .d = 64, .window = 1000, .backward = false },
        .{ .b = 1, .tq = 9, .tk = 64, .keys = 40, .h = 2, .hkv = 1, .d = 128, .window = 16, .backward = false },
    };
    var rng = mod.Random.init(5);
    for (cases) |c| {
        const qd = [_]usize{ c.b, c.tq, c.h, c.d };
        const kd = [_]usize{ c.b, c.tk, c.hkv, c.d };
        const ld = [_]usize{ c.b, c.h, c.tq };
        const options = mod.AttentionOptions{ .window = c.window, .keys = c.keys };
        const hq = try allocator.alloc(f32, c.b * c.tq * c.h * c.d);
        defer allocator.free(hq);
        const hk = try allocator.alloc(f32, c.b * c.tk * c.hkv * c.d);
        defer allocator.free(hk);
        const hv = try allocator.alloc(f32, hk.len);
        defer allocator.free(hv);
        const hg = try allocator.alloc(f32, hq.len);
        defer allocator.free(hg);
        rng.fillUniform(hq, -2, 2);
        rng.fillUniform(hk, -2, 2);
        rng.fillUniform(hv, -1, 1);
        rng.fillUniform(hg, -1, 1);

        const Run = struct {
            fn on(comptime B: type, be: *B, a: std.mem.Allocator, cc: Case, o: mod.AttentionOptions, dims: anytype, data: anytype) ![5][]f32 {
                const q = try be.alloc(.f32, &dims[0]);
                defer be.free(q);
                const k = try be.alloc(.f32, &dims[1]);
                defer be.free(k);
                const v = try be.alloc(.f32, &dims[1]);
                defer be.free(v);
                const g = try be.alloc(.f32, &dims[0]);
                defer be.free(g);
                const out = try be.alloc(.f32, &dims[0]);
                defer be.free(out);
                const lse = try be.alloc(.f32, &dims[2]);
                defer be.free(lse);
                const dq = try be.alloc(.f32, &dims[0]);
                defer be.free(dq);
                const dk = try be.alloc(.f32, &dims[1]);
                defer be.free(dk);
                const dv = try be.alloc(.f32, &dims[1]);
                defer be.free(dv);
                try be.upload(q, f32, data[0]);
                try be.upload(k, f32, data[1]);
                try be.upload(v, f32, data[2]);
                try be.upload(g, f32, data[3]);
                try be.attention(out, q, k, v, lse, o);
                if (cc.backward) try be.attentionBackward(dq, dk, dv, g, q, k, v, out, lse, o);
                var results: [5][]f32 = undefined;
                for ([_]mod.Tensor{ out, lse, dq, dk, dv }, 0..) |t, i| {
                    results[i] = try a.alloc(f32, t.numel());
                    try be.download(t, f32, results[i]);
                }
                return results;
            }
        };
        const dims = .{ qd, kd, ld };
        const data = .{ hq, hk, hv, hg };
        const want = try Run.on(mod.CpuBackend, &cpu, allocator, c, options, dims, data);
        defer for (want) |w| allocator.free(w);
        const got = try Run.on(MetalBackend, &gpu, allocator, c, options, dims, data);
        defer for (got) |g| allocator.free(g);
        const names = [_][]const u8{ "out", "lse", "dq", "dk", "dv" };
        const checked: usize = if (c.backward) 5 else 2;
        for (want[0..checked], got[0..checked], names[0..checked]) |w, g, name| {
            var worst: f32 = 0;
            for (w, g) |x, y| worst = @max(worst, @abs(x - y) / (1 + @abs(x)));
            if (worst > 2e-4) {
                std.debug.print("attention d={d} tq={d} {s}: max error {e}\n", .{ c.d, c.tq, name, worst });
                return error.TestUnexpectedResult;
            }
        }
    }
}
