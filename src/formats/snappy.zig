const std = @import("std");
const log = std.log.scoped(.zignanogpt_snappy);
const mod = @import("../module.zig");

/// Raw (unframed) Snappy, Parquet's `SNAPPY` codec: a varint uncompressed
/// length, then literals and back-references.
pub const Snappy = struct {
    /// Bytes compressed independently (back-references never cross one), as the
    /// reference implementation; offsets stay below 64 KiB.
    const block_size = 1 << 16;
    const hash_bits = 14;

    /// The largest output `compress` can produce for `n` input bytes.
    ///
    /// Parameters:
    /// - `n`: the input length.
    ///
    /// Return: the bound (the reference implementation's).
    pub fn maxCompressedLength(n: usize) usize {
        return 32 + n + n / 6;
    }

    /// Compresses `src` greedily (a 4-byte hash table per 64 KiB block, the
    /// reference implementation's scheme, without its unaligned fast paths).
    ///
    /// Parameters:
    /// - `src`: the input; at most 4 GiB.
    /// - `dst`: at least `maxCompressedLength(src.len)` bytes.
    ///
    /// Return: the compressed length written to `dst`.
    pub fn compress(src: []const u8, dst: []u8) usize {
        std.debug.assert(dst.len >= maxCompressedLength(src.len));
        var out: usize = 0;
        var len = src.len;
        while (true) {
            const b: u8 = @truncate(len);
            len >>= 7;
            dst[out] = if (len == 0) b & 0x7f else b | 0x80;
            out += 1;
            if (len == 0) break;
        }
        var table: [1 << hash_bits]u16 = undefined;
        var start: usize = 0;
        while (start < src.len) : (start += block_size) {
            out += compressBlock(src[start..@min(src.len, start + block_size)], dst[out..], &table);
        }
        return out;
    }
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

    /// One block: literals between matches of at least 4 bytes found by hash.
    fn compressBlock(block: []const u8, dst: []u8, table: *[1 << hash_bits]u16) usize {
        var out: usize = 0;
        if (block.len < 15) return emitLiteral(dst, block);
        @memset(table, 0);
        var literal: usize = 0;
        var ip: usize = 1;
        var skip: usize = 32;
        const last = block.len - 4;
        while (ip <= last) {
            const word = std.mem.readInt(u32, block[ip..][0..4], .little);
            const h = hash(word);
            const candidate: usize = table[h];
            table[h] = @intCast(ip);
            if (candidate >= ip or std.mem.readInt(u32, block[candidate..][0..4], .little) != word) {
                // Misses speed up over incompressible runs, as in the reference.
                ip += skip >> 5;
                skip += 1;
                continue;
            }
            skip = 32;
            out += emitLiteral(dst[out..], block[literal..ip]);
            var n: usize = 4;
            while (ip + n < block.len and block[candidate + n] == block[ip + n]) n += 1;
            out += emitCopy(dst[out..], ip - candidate, n);
            ip += n;
            literal = ip;
            if (ip - 1 <= last) table[hash(std.mem.readInt(u32, block[ip - 1 ..][0..4], .little))] = @intCast(ip - 1);
        }
        out += emitLiteral(dst[out..], block[literal..]);
        return out;
    }

    fn hash(word: u32) usize {
        return @as(u32, word *% 0x1e35a7bd) >> (32 - hash_bits);
    }

    /// A literal tag (length - 1 inline below 60, else in 1-4 trailing bytes), then the bytes.
    fn emitLiteral(dst: []u8, bytes: []const u8) usize {
        if (bytes.len == 0) return 0;
        const n = bytes.len - 1;
        var out: usize = 1;
        if (n < 60) {
            dst[0] = @intCast(n << 2);
        } else {
            var count: u8 = 0;
            var rest = n;
            while (rest > 0) : (rest >>= 8) {
                dst[out] = @truncate(rest);
                out += 1;
                count += 1;
            }
            dst[0] = (59 + count) << 2;
        }
        @memcpy(dst[out..][0..bytes.len], bytes);
        return out + bytes.len;
    }

    /// Copy tags for a match: 64-byte pieces, then a 1-byte-offset tag when it fits, else 2-byte.
    fn emitCopy(dst: []u8, offset: usize, length: usize) usize {
        var out: usize = 0;
        var n = length;
        while (n >= 68) : (n -= 64) out += copyTag(dst[out..], offset, 64);
        if (n > 64) {
            out += copyTag(dst[out..], offset, 60);
            n -= 60;
        }
        if (n < 12 and offset < 2048) {
            dst[out] = @intCast(1 | ((n - 4) << 2) | ((offset >> 8) << 5));
            dst[out + 1] = @truncate(offset);
            return out + 2;
        }
        return out + copyTag(dst[out..], offset, n);
    }

    fn copyTag(dst: []u8, offset: usize, n: usize) usize {
        dst[0] = @intCast(2 | ((n - 1) << 2));
        std.mem.writeInt(u16, dst[1..3], @intCast(offset), .little);
        return 3;
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

test "snappy compresses what it decompresses, text and random bytes, across blocks" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();
    const sentence = "The history of science is the study of the development of science. ";
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    while (text.items.len < 3 * (1 << 16) + 123) {
        try text.appendSlice(allocator, sentence);
        // A random word now and then keeps it from being one long repeat.
        for (0..random.intRangeAtMost(usize, 0, 9)) |_| try text.append(allocator, 'a' + random.intRangeLessThan(u8, 0, 26));
    }
    const noise = try allocator.alloc(u8, 70_000);
    defer allocator.free(noise);
    random.bytes(noise);
    for ([_][]const u8{ "", "short", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", text.items, noise }) |src| {
        const packed_bytes = try allocator.alloc(u8, mod.Snappy.maxCompressedLength(src.len));
        defer allocator.free(packed_bytes);
        const n = mod.Snappy.compress(src, packed_bytes);
        const back = try allocator.alloc(u8, src.len);
        defer allocator.free(back);
        try mod.Snappy.decompress(packed_bytes[0..n], back);
        try std.testing.expectEqualSlices(u8, src, back);
        if (src.len == text.items.len) try std.testing.expect(n * 3 < src.len);
    }
}
