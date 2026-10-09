const std = @import("std");
const log = std.log.scoped(.zignanogpt_safe_tensors);
const mod = @import("../module.zig");

/// One named tensor in a safetensors file.
pub const SafeTensorEntry = struct {
    dtype: mod.Dtype,
    shape: mod.Shape,
    /// The raw little-endian elements, borrowed from the file bytes.
    bytes: []const u8,
};

/// A safetensors file held in memory: 8-byte little-endian header length, JSON
/// header (`name -> {dtype, shape, data_offsets}` plus optional
/// `__metadata__`), then the tensor data. The parity fixtures use it, and so
/// does this port's checkpoint format.
pub const SafeTensors = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    /// The whole file; entries borrow from it.
    data: []u8,
    header: std.json.Parsed(std.json.Value),
    entries: std.array_hash_map.String(SafeTensorEntry),

    /// Reads and parses a file through zigstorage.
    ///
    /// Parameters:
    /// - `allocator`: owns the bytes and index until `deinit`.
    /// - `io`: the Io the read runs on.
    /// - `location`: a path or `file://` URL.
    ///
    /// Return: the parsed file; storage or format errors.
    pub fn load(allocator: std.mem.Allocator, io: std.Io, location: []const u8) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var node = try mod.zigstorage.Node.init(allocator, io, .empty, location);
        defer node.deinit();
        const data = node.read(.all) catch |err| {
            log.err("cannot read {s} [{t}]", .{ location, err });
            return err;
        };
        return parse(allocator, data);
    }

    /// Parses file bytes, taking ownership of them.
    ///
    /// Parameters:
    /// - `allocator`: the allocator `data` came from; owns the index too.
    /// - `data`: the file bytes; freed by `deinit` (or here on error).
    ///
    /// Return: the parsed file; `error.InvalidSafeTensors` on a malformed file,
    /// `error.UnsupportedDtype` for an element type other than F32/I32.
    pub fn parse(allocator: std.mem.Allocator, data: []u8) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        errdefer allocator.free(data);
        if (data.len < 8) return error.InvalidSafeTensors;
        const header_len = std.mem.readInt(u64, data[0..8], .little);
        if (header_len > data.len - 8) return error.InvalidSafeTensors;
        const body = data[8 + header_len ..];

        var header = try std.json.parseFromSlice(std.json.Value, allocator, data[8..][0..header_len], .{});
        errdefer header.deinit();
        if (header.value != .object) return error.InvalidSafeTensors;

        var entries: std.array_hash_map.String(SafeTensorEntry) = .empty;
        errdefer entries.deinit(allocator);
        var it = header.value.object.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            try entries.put(allocator, kv.key_ptr.*, try parseEntry(kv.value_ptr.*, body));
        }
        return Self{ .allocator = allocator, .data = data, .header = header, .entries = entries };
    }

    /// Frees the index and the file bytes.
    ///
    /// Parameters:
    /// - `self`: the file.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.entries.deinit(self.allocator);
        self.header.deinit();
        self.allocator.free(self.data);
    }

    /// Looks up a tensor.
    ///
    /// Parameters:
    /// - `self`: the file.
    /// - `name`: the tensor name.
    ///
    /// Return: the entry, or `error.MissingTensor`.
    pub fn get(self: *const Self, name: []const u8) !SafeTensorEntry {
        return self.entries.get(name) orelse {
            log.debug("tensor '{s}' not found", .{name});
            return error.MissingTensor;
        };
    }

    /// Copies a tensor's elements into `out`.
    ///
    /// Parameters:
    /// - `self`: the file.
    /// - `name`: the tensor name.
    /// - `T`: `f32` or `i32`; must match the stored dtype.
    /// - `out`: exactly the tensor's element count.
    ///
    /// Return: nothing; missing, dtype or size errors.
    pub fn read(self: *const Self, name: []const u8, comptime T: type, out: []T) !void {
        const entry = try self.get(name);
        if (entry.dtype != mod.Dtype.of(T)) return error.DtypeMismatch;
        if (out.len != entry.shape.numel()) {
            log.debug("tensor '{s}' {f} does not fit {d} elements", .{ name, entry.shape, out.len });
            return error.ShapeMismatch;
        }
        @memcpy(std.mem.sliceAsBytes(out), entry.bytes);
    }

    /// Reads a tensor into newly allocated memory.
    ///
    /// Parameters:
    /// - `self`: the file.
    /// - `allocator`: allocates the result.
    /// - `name`: the tensor name.
    /// - `T`: `f32` or `i32`.
    ///
    /// Return: the elements, owned by the caller.
    pub fn readAlloc(self: *const Self, allocator: std.mem.Allocator, name: []const u8, comptime T: type) ![]T {
        const entry = try self.get(name);
        const out = try allocator.alloc(T, entry.shape.numel());
        errdefer allocator.free(out);
        try self.read(name, T, out);
        return out;
    }

    /// A `__metadata__` value.
    ///
    /// Parameters:
    /// - `self`: the file.
    /// - `key`: the metadata key.
    ///
    /// Return: the string, or null when absent.
    pub fn metadata(self: *const Self, key: []const u8) ?[]const u8 {
        const meta = self.header.value.object.get("__metadata__") orelse return null;
        if (meta != .object) return null;
        const value = meta.object.get(key) orelse return null;
        return if (value == .string) value.string else null;
    }

    /// A `__metadata__` value parsed as an integer.
    ///
    /// Parameters:
    /// - `self`: the file.
    /// - `key`: the metadata key.
    ///
    /// Return: the value; `error.MissingMetadata` or a parse error.
    pub fn metadataInt(self: *const Self, comptime T: type, key: []const u8) !T {
        const text = self.metadata(key) orelse return error.MissingMetadata;
        return std.fmt.parseInt(T, text, 10);
    }

    /// Decodes one header entry against the data section.
    fn parseEntry(value: std.json.Value, body: []const u8) !SafeTensorEntry {
        if (value != .object) return error.InvalidSafeTensors;
        const dtype_text = (value.object.get("dtype") orelse return error.InvalidSafeTensors);
        const shape_json = (value.object.get("shape") orelse return error.InvalidSafeTensors);
        const offsets = (value.object.get("data_offsets") orelse return error.InvalidSafeTensors);
        if (dtype_text != .string or shape_json != .array or offsets != .array or offsets.array.items.len != 2) {
            return error.InvalidSafeTensors;
        }
        const dtype: mod.Dtype = if (std.mem.eql(u8, dtype_text.string, "F32"))
            .f32
        else if (std.mem.eql(u8, dtype_text.string, "I32"))
            .i32
        else {
            log.err("unsupported safetensors dtype {s}", .{dtype_text.string});
            return error.UnsupportedDtype;
        };
        var dims: [mod.Shape.max_rank]usize = undefined;
        if (shape_json.array.items.len > dims.len) return error.InvalidSafeTensors;
        for (shape_json.array.items, 0..) |d, i| dims[i] = try jsonUsize(d);
        const shape = try mod.Shape.init(dims[0..shape_json.array.items.len]);
        const start = try jsonUsize(offsets.array.items[0]);
        const end = try jsonUsize(offsets.array.items[1]);
        if (start > end or end > body.len or end - start != shape.numel() * dtype.size()) return error.InvalidSafeTensors;
        return SafeTensorEntry{ .dtype = dtype, .shape = shape, .bytes = body[start..end] };
    }

    /// A non-negative JSON integer.
    fn jsonUsize(value: std.json.Value) !usize {
        if (value != .integer or value.integer < 0) return error.InvalidSafeTensors;
        return @intCast(value.integer);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "safetensors parses a hand-built file" {
    const allocator = std.testing.allocator;
    const header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"b\":{\"dtype\":\"I32\",\"shape\":[1,1],\"data_offsets\":[8,12]},\"__metadata__\":{\"n\":\"42\"}}";
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(allocator);
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try bytes.appendSlice(allocator, &len_buf);
    try bytes.appendSlice(allocator, header);
    try bytes.appendSlice(allocator, std.mem.sliceAsBytes(&[_]f32{ 1.5, -2 }));
    try bytes.appendSlice(allocator, std.mem.sliceAsBytes(&[_]i32{7}));

    var file = try mod.SafeTensors.parse(allocator, try bytes.toOwnedSlice(allocator));
    defer file.deinit();
    var a: [2]f32 = undefined;
    try file.read("a", f32, &a);
    try std.testing.expectEqualSlices(f32, &.{ 1.5, -2 }, &a);
    var b: [1]i32 = undefined;
    try file.read("b", i32, &b);
    try std.testing.expectEqual(@as(i32, 7), b[0]);
    try std.testing.expectEqual(@as(usize, 42), try file.metadataInt(usize, "n"));
    try std.testing.expectError(error.MissingTensor, file.get("c"));
    try std.testing.expectError(error.DtypeMismatch, file.read("a", i32, &b));
}

test "safetensors rejects a truncated file" {
    const allocator = std.testing.allocator;
    const data = try allocator.dupe(u8, &.{ 200, 0, 0, 0, 0, 0, 0, 0, '{', '}' });
    try std.testing.expectError(error.InvalidSafeTensors, mod.SafeTensors.parse(allocator, data));
}

test "safetensors loads the committed gpt fixture" {
    const path = mod.build_options.source_root ++ "/testdata/gpt.safetensors";
    var file = try mod.SafeTensors.load(std.testing.allocator, std.testing.io, path);
    defer file.deinit();
    const idx = try file.get("idx");
    try std.testing.expectEqual(mod.Dtype.i32, idx.dtype);
    try std.testing.expectEqual(@as(usize, 4), try file.metadataInt(usize, "n_layer"));
}
