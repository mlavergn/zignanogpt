const std = @import("std");
const log = std.log.scoped(.zignanogpt_zip_archive);
const mod = @import("module.zig");

/// One member of a zip archive.
pub const ZipEntry = struct {
    name: []const u8,
    method: u16,
    compressed_size: u64,
    size: u64,
    local_header_offset: u64,
};

/// A zip archive read through zigstorage ranged reads: the central directory
/// once, then each member on demand. Zip64 is supported (multi-GB PyTorch
/// checkpoints use it). Members are stored (what `torch.save` writes) or
/// deflated (e.g. nanochat's `eval_bundle.zip`).
pub const ZipArchive = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    node: mod.zigstorage.Node,
    /// The central directory; entry names borrow from it.
    directory: []u8,
    entries: []ZipEntry,

    /// Opens an archive and reads its central directory.
    ///
    /// Parameters:
    /// - `allocator`: owns the directory and the entry list.
    /// - `io`: the Io reads run on.
    /// - `location`: a path or URL.
    ///
    /// Return: the archive; storage errors, `error.InvalidZip`.
    pub fn open(allocator: std.mem.Allocator, io: std.Io, location: []const u8) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var node = try mod.zigstorage.Node.init(allocator, io, .empty, location);
        errdefer node.deinit();
        try node.fetchAttributes();
        const size = node.size orelse return error.InvalidZip;
        if (size < 22) return error.InvalidZip;

        // The end-of-central-directory record sits in the last 22 + 65535 bytes.
        const tail_len = @min(size, 22 + 0xffff + 20);
        const tail = try node.read(.{ .offset = size - tail_len, .length = tail_len });
        defer allocator.free(tail);
        const eocd = std.mem.lastIndexOf(u8, tail, "PK\x05\x06") orelse return error.InvalidZip;
        if (tail.len - eocd < 22) return error.InvalidZip;
        var count: u64 = std.mem.readInt(u16, tail[eocd + 10 ..][0..2], .little);
        var dir_size: u64 = std.mem.readInt(u32, tail[eocd + 12 ..][0..4], .little);
        var dir_offset: u64 = std.mem.readInt(u32, tail[eocd + 16 ..][0..4], .little);
        if (count == 0xffff or dir_size == 0xffffffff or dir_offset == 0xffffffff) {
            // Zip64: a locator right before the EOCD points at the zip64 EOCD record.
            if (eocd < 20 or !std.mem.eql(u8, tail[eocd - 20 ..][0..4], "PK\x06\x07")) return error.InvalidZip;
            const record_offset = std.mem.readInt(u64, tail[eocd - 20 + 8 ..][0..8], .little);
            const record = try node.read(.{ .offset = record_offset, .length = 56 });
            defer allocator.free(record);
            if (record.len < 56 or !std.mem.eql(u8, record[0..4], "PK\x06\x06")) return error.InvalidZip;
            count = std.mem.readInt(u64, record[32..40], .little);
            dir_size = std.mem.readInt(u64, record[40..48], .little);
            dir_offset = std.mem.readInt(u64, record[48..56], .little);
        }
        const directory = try node.read(.{ .offset = dir_offset, .length = dir_size });
        errdefer allocator.free(directory);
        if (directory.len != dir_size) return error.InvalidZip;
        const entries = try parseDirectory(allocator, directory, std.math.cast(usize, count) orelse return error.InvalidZip);
        return Self{ .allocator = allocator, .node = node, .directory = directory, .entries = entries };
    }

    /// Frees the directory.
    ///
    /// Parameters:
    /// - `self`: the archive.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.allocator.free(self.entries);
        self.allocator.free(self.directory);
        self.node.deinit();
    }

    /// The first member whose name ends with `suffix` (torch prefixes every
    /// member with the archive's folder name).
    ///
    /// Parameters:
    /// - `self`: the archive.
    /// - `suffix`: e.g. `/data.pkl`.
    ///
    /// Return: the entry, or `error.MissingZipEntry`.
    pub fn find(self: *const Self, suffix: []const u8) !ZipEntry {
        for (self.entries) |e| {
            if (std.mem.endsWith(u8, e.name, suffix)) return e;
        }
        log.debug("no zip member ending in {s}", .{suffix});
        return error.MissingZipEntry;
    }

    /// Reads a member, inflating it when deflated.
    ///
    /// Parameters:
    /// - `self`: the archive.
    /// - `entry`: from `entries` or `find`.
    ///
    /// Return: the bytes, owned by the caller; `error.UnsupportedZip` for other compression methods.
    pub fn read(self: *Self, entry: ZipEntry) ![]u8 {
        if (entry.method != 0 and entry.method != 8) {
            log.warn("zip member {s} uses compression method {d}; only stored and deflate are supported", .{ entry.name, entry.method });
            return error.UnsupportedZip;
        }
        const header = try self.node.read(.{ .offset = entry.local_header_offset, .length = 30 });
        defer self.allocator.free(header);
        if (header.len < 30 or !std.mem.eql(u8, header[0..4], "PK\x03\x04")) return error.InvalidZip;
        const name_len = std.mem.readInt(u16, header[26..28], .little);
        const extra_len = std.mem.readInt(u16, header[28..30], .little);
        const offset = entry.local_header_offset + 30 + name_len + extra_len;
        if (entry.method == 8) {
            const packed_bytes = try self.node.read(.{ .offset = offset, .length = entry.compressed_size });
            defer self.allocator.free(packed_bytes);
            if (packed_bytes.len != entry.compressed_size) return error.InvalidZip;
            var in: std.Io.Reader = .fixed(packed_bytes);
            var inflate: std.compress.flate.Decompress = .init(&in, .raw, &.{});
            var out: std.Io.Writer.Allocating = .init(self.allocator);
            errdefer out.deinit();
            _ = inflate.reader.streamRemaining(&out.writer) catch |err| {
                log.warn("zip member {s} failed to inflate [{t}]", .{ entry.name, inflate.err orelse err });
                return error.InvalidZip;
            };
            if (out.written().len != entry.size) return error.InvalidZip;
            return out.toOwnedSlice();
        }
        const data = try self.node.read(.{ .offset = offset, .length = entry.size });
        errdefer self.allocator.free(data);
        if (data.len != entry.size) return error.InvalidZip;
        return data;
    }

    /// Decodes `count` central-directory records.
    fn parseDirectory(allocator: std.mem.Allocator, dir: []const u8, count: usize) ![]ZipEntry {
        const entries = try allocator.alloc(ZipEntry, count);
        errdefer allocator.free(entries);
        var pos: usize = 0;
        for (entries) |*e| {
            if (dir.len - pos < 46 or !std.mem.eql(u8, dir[pos..][0..4], "PK\x01\x02")) return error.InvalidZip;
            const rec = dir[pos..];
            const name_len = std.mem.readInt(u16, rec[28..30], .little);
            const extra_len = std.mem.readInt(u16, rec[30..32], .little);
            const comment_len = std.mem.readInt(u16, rec[32..34], .little);
            if (rec.len < 46 + @as(usize, name_len) + extra_len + comment_len) return error.InvalidZip;
            e.* = .{
                .name = rec[46..][0..name_len],
                .method = std.mem.readInt(u16, rec[10..12], .little),
                .compressed_size = std.mem.readInt(u32, rec[20..24], .little),
                .size = std.mem.readInt(u32, rec[24..28], .little),
                .local_header_offset = std.mem.readInt(u32, rec[42..46], .little),
            };
            // Zip64 extended information replaces the saturated fields, in order.
            var extra = rec[46 + name_len ..][0..extra_len];
            while (extra.len >= 4) {
                const id = std.mem.readInt(u16, extra[0..2], .little);
                const len = std.mem.readInt(u16, extra[2..4], .little);
                if (extra.len < 4 + @as(usize, len)) return error.InvalidZip;
                if (id == 0x0001) {
                    var field = extra[4..][0..len];
                    inline for (.{ "size", "compressed_size", "local_header_offset" }) |name| {
                        if (@field(e, name) == 0xffffffff) {
                            if (field.len < 8) return error.InvalidZip;
                            @field(e, name) = std.mem.readInt(u64, field[0..8], .little);
                            field = field[8..];
                        }
                    }
                }
                extra = extra[4 + len ..];
            }
            pos += 46 + @as(usize, name_len) + extra_len + comment_len;
        }
        return entries;
    }
};
