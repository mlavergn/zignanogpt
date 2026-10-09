const std = @import("std");
const log = std.log.scoped(.zignanogpt_safe_tensors_writer);
const mod = @import("../module.zig");

/// A tensor queued for writing: backend memory, or host bytes.
const Pending = struct {
    name: []const u8,
    dtype: mod.Dtype,
    shape: mod.Shape,
    source: union(enum) { tensor: mod.Tensor, bytes: []const u8 },
};

/// Writes a safetensors file (the layout `SafeTensors` reads), streaming each
/// backend tensor through one reusable host buffer and committing atomically,
/// so a checkpoint never needs a second full copy in memory.
pub const SafeTensorsWriter = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    pending: std.ArrayList(Pending) = .empty,
    metadata: std.ArrayList([2][]const u8) = .empty,

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{ .allocator = allocator, .arena = std.heap.ArenaAllocator.init(allocator) };
    }

    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    /// Queues a backend tensor (f32 or i32), read at `write`.
    ///
    /// Parameters:
    /// - `self`: the writer.
    /// - `name`: the tensor's name (copied).
    /// - `tensor`: must stay alive until `write`.
    ///
    /// Return: nothing; allocation errors.
    pub fn addTensor(self: *Self, name: []const u8, tensor: mod.Tensor) !void {
        const a = self.arena.allocator();
        try self.pending.append(a, .{ .name = try a.dupe(u8, name), .dtype = tensor.dtype, .shape = tensor.shape, .source = .{ .tensor = tensor } });
    }

    /// Queues host data.
    ///
    /// Parameters:
    /// - `self`: the writer.
    /// - `name`: the tensor's name (copied).
    /// - `T`: `f32` or `i32`.
    /// - `dims`: the shape.
    /// - `data`: the elements (copied).
    ///
    /// Return: nothing; shape and allocation errors.
    pub fn addHost(self: *Self, name: []const u8, comptime T: type, dims: []const usize, data: []const T) !void {
        const shape = try mod.Shape.init(dims);
        if (shape.numel() != data.len) return error.ShapeMismatch;
        const a = self.arena.allocator();
        try self.pending.append(a, .{ .name = try a.dupe(u8, name), .dtype = mod.Dtype.of(T), .shape = shape, .source = .{ .bytes = try a.dupe(u8, std.mem.sliceAsBytes(data)) } });
    }

    /// Adds a `__metadata__` string entry.
    pub fn addMetadata(self: *Self, key: []const u8, value: []const u8) !void {
        const a = self.arena.allocator();
        try self.metadata.append(a, .{ try a.dupe(u8, key), try a.dupe(u8, value) });
    }

    /// Writes everything queued to `path`, atomically.
    ///
    /// Parameters:
    /// - `self`: the writer.
    /// - `backend`: downloads the queued tensors.
    /// - `storage`: the file helper.
    /// - `path`: the destination; its directory is created.
    ///
    /// Return: nothing; backend and storage errors.
    pub fn write(self: *Self, backend: *mod.Backend, storage: mod.Storage, path: []const u8) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var header: std.Io.Writer.Allocating = .init(self.allocator);
        defer header.deinit();
        const w = &header.writer;
        try w.writeByte('{');
        var offset: usize = 0;
        var largest: usize = 0;
        for (self.pending.items, 0..) |p, i| {
            const size = p.shape.numel() * p.dtype.size();
            if (i > 0) try w.writeByte(',');
            try std.json.Stringify.value(p.name, .{}, w);
            try w.print(":{{\"dtype\":\"{s}\",\"shape\":[", .{switch (p.dtype) {
                .f32 => "F32",
                .i32 => "I32",
            }});
            for (p.shape.slice(), 0..) |d, k| try w.print("{s}{d}", .{ if (k > 0) "," else "", d });
            try w.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + size });
            offset += size;
            largest = @max(largest, size);
        }
        if (self.metadata.items.len > 0) {
            try w.writeAll(",\"__metadata__\":{");
            for (self.metadata.items, 0..) |kv, i| {
                if (i > 0) try w.writeByte(',');
                try std.json.Stringify.value(kv[0], .{}, w);
                try w.writeByte(':');
                try std.json.Stringify.value(kv[1], .{}, w);
            }
            try w.writeByte('}');
        }
        try w.writeByte('}');
        while (header.written().len % 8 != 0) try w.writeByte(' ');

        if (std.Io.Dir.path.dirname(path)) |dir| try storage.makeDir(dir);
        var node = try mod.zigstorage.Node.init(self.allocator, storage.io, .empty, path);
        defer node.deinit();
        try node.open();
        var len_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_buf, header.written().len, .little);
        try node.append(&len_buf);
        try node.append(header.written());
        const buffer = try self.allocator.alignedAlloc(u8, .@"4", largest);
        defer self.allocator.free(buffer);
        for (self.pending.items) |p| switch (p.source) {
            .bytes => |b| try node.append(b),
            .tensor => |t| {
                const bytes = buffer[0 .. t.numel() * t.dtype.size()];
                switch (t.dtype) {
                    .f32 => try backend.download(t, f32, std.mem.bytesAsSlice(f32, bytes)),
                    .i32 => try backend.download(t, i32, std.mem.bytesAsSlice(i32, bytes)),
                }
                try node.append(bytes);
            },
        };
        try node.save();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "safetensors writer round-trips through the reader" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const path = try std.Io.Dir.path.join(allocator, &.{ root, "sub", "x.safetensors" });
    defer allocator.free(path);

    var backend = try mod.Backend.init(allocator, std.testing.io, .{ .threads = 1 });
    defer backend.deinit();
    const t = try backend.alloc(.f32, &.{ 2, 3 });
    defer backend.free(t);
    try backend.upload(t, f32, &.{ 1, 2, 3, 4, 5, 6 });

    var writer = mod.SafeTensorsWriter.init(allocator);
    defer writer.deinit();
    try writer.addTensor("w \"quoted\"", t);
    try writer.addHost("steps", i32, &.{2}, &.{ 7, -1 });
    try writer.addMetadata("step", "5");
    try writer.write(&backend, mod.Storage.init(allocator, std.testing.io), path);

    var file = try mod.SafeTensors.load(allocator, std.testing.io, path);
    defer file.deinit();
    var w: [6]f32 = undefined;
    try file.read("w \"quoted\"", f32, &w);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, &w);
    var s: [2]i32 = undefined;
    try file.read("steps", i32, &s);
    try std.testing.expectEqualSlices(i32, &.{ 7, -1 }, &s);
    try std.testing.expectEqual(@as(usize, 5), try file.metadataInt(usize, "step"));
}
