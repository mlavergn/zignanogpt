const std = @import("std");
const log = std.log.scoped(.zignanogpt_snappy);
const mod = @import("../module.zig");

/// Raw (unframed) Snappy decompression, Parquet's `SNAPPY` codec: a varint
/// uncompressed length, then literals and back-references.
pub const Snappy = struct {
    /// Decompresses `src` into `dst`, which must be exactly the uncompressed size.
    ///
    /// Parameters:
    /// - `src`: the compressed block.
    /// - `dst`: receives the output; its length must match the encoded length.
    ///
    /// Return: nothing; `error.InvalidSnappy` on malformed or mismatched input.
    pub fn decompress(src: []const u8, dst: []u8) !void {
        var pos: usize = 0;
        var len: u64 = 0;
        var shift: u6 = 0;
        while (true) {
            if (pos >= src.len) return error.InvalidSnappy;
            const b = src[pos];
            pos += 1;
            len |= @as(u64, b & 0x7f) << shift;
            if (b & 0x80 == 0) break;
            if (shift >= 35) return error.InvalidSnappy;
            shift += 7;
        }
        if (len != dst.len) {
            log.debug("snappy length {d} != expected {d}", .{ len, dst.len });
            return error.InvalidSnappy;
        }
        var out: usize = 0;
        while (pos < src.len) {
            const tag = src[pos];
            pos += 1;
            switch (@as(u2, @truncate(tag))) {
                0 => {
                    var n: usize = (tag >> 2) + 1;
                    if (n > 60) {
                        const extra = n - 60;
                        if (pos + extra > src.len) return error.InvalidSnappy;
                        n = 0;
                        for (0..extra) |i| n |= @as(usize, src[pos + i]) << @intCast(8 * i);
                        n += 1;
                        pos += extra;
                    }
                    if (pos + n > src.len or out + n > dst.len) return error.InvalidSnappy;
                    @memcpy(dst[out..][0..n], src[pos..][0..n]);
                    pos += n;
                    out += n;
                },
                1 => {
                    if (pos >= src.len) return error.InvalidSnappy;
                    const n: usize = ((tag >> 2) & 7) + 4;
                    const offset: usize = (@as(usize, tag >> 5) << 8) | src[pos];
                    pos += 1;
                    try copyBack(dst, &out, offset, n);
                },
                2 => {
                    if (pos + 2 > src.len) return error.InvalidSnappy;
                    const offset: usize = std.mem.readInt(u16, src[pos..][0..2], .little);
                    pos += 2;
                    try copyBack(dst, &out, offset, (tag >> 2) + 1);
                },
                3 => {
                    if (pos + 4 > src.len) return error.InvalidSnappy;
                    const offset: usize = std.mem.readInt(u32, src[pos..][0..4], .little);
                    pos += 4;
                    try copyBack(dst, &out, offset, (tag >> 2) + 1);
                },
            }
        }
        if (out != dst.len) return error.InvalidSnappy;
    }

    /// Copies `n` bytes from `offset` back; the ranges may overlap (run-length style).
    fn copyBack(dst: []u8, out: *usize, offset: usize, n: usize) !void {
        if (offset == 0 or offset > out.* or out.* + n > dst.len) return error.InvalidSnappy;
        for (0..n) |i| dst[out.* + i] = dst[out.* - offset + i];
        out.* += n;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "snappy decodes literals and overlapping copies" {
    // "abcabcabcabc": literal "abc" then a 1-byte-offset copy of 9 bytes at offset 3.
    const block = [_]u8{ 12, (3 - 1) << 2, 'a', 'b', 'c', 0x01 | ((9 - 4) << 2), 3 };
    var out: [12]u8 = undefined;
    try mod.Snappy.decompress(&block, &out);
    try std.testing.expectEqualStrings("abcabcabcabc", &out);
    var short: [11]u8 = undefined;
    try std.testing.expectError(error.InvalidSnappy, mod.Snappy.decompress(&block, &short));
}
