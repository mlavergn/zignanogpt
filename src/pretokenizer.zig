const std = @import("std");
const log = std.log.scoped(.zignanogpt_pretokenizer);
const mod = @import("module.zig");

/// The code points of Unicode `White_Space` (`\s` in Rust's `regex`, which
/// tiktoken and rustbpe use). uucode ships no `PropList.txt`, so they are listed.
const white_space = [_]u21{
    0x09,   0x0A,   0x0B,   0x0C,   0x0D,   0x20,   0x85,   0xA0,   0x1680, 0x2000, 0x2001, 0x2002, 0x2003,
    0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
};

/// Character classes the split pattern distinguishes.
const Class = enum { letter, number, space, other };

/// Splits text into the pieces BPE runs on, exactly as the GPT-4 style split
/// pattern does under a backtracking regex engine (`fancy_regex`):
///
/// ```
/// '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,D}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
/// ```
///
/// with `D = max_digits` (nanochat 2, GPT-4's cl100k 3). Alternatives are
/// tried in order at each position and the first that matches wins; every code
/// point matches some alternative, so the pieces tile the text. Classes come
/// from uucode's general categories (`\p{L}` = L*, `\p{N}` = N*).
pub const Pretokenizer = struct {
    const Self = @This();

    text: []const u8,
    pos: usize = 0,
    max_digits: usize,

    /// Starts splitting `text`.
    ///
    /// Parameters:
    /// - `text`: valid UTF-8.
    /// - `max_digits`: the `\p{N}{1,D}` bound.
    ///
    /// Return: the iterator; `error.InvalidUtf8` for malformed text.
    pub fn init(text: []const u8, max_digits: usize) !Self {
        if (!std.unicode.utf8ValidateSlice(text)) {
            log.debug("text is not valid UTF-8", .{});
            return error.InvalidUtf8;
        }
        return Self{ .text = text, .max_digits = max_digits };
    }

    /// The next piece, or null at the end.
    ///
    /// Parameters:
    /// - `self`: the iterator.
    ///
    /// Return: a slice of the text.
    pub fn next(self: *Self) ?[]const u8 {
        if (self.pos >= self.text.len) return null;
        const start = self.pos;
        const end = self.contraction(start) orelse
            self.letters(start) orelse
            self.digits(start) orelse
            self.punctuation(start) orelse
            self.newlines(start) orelse
            self.trailingSpace(start) orelse
            self.spaces(start).?;
        self.pos = end;
        return self.text[start..end];
    }

    /// `'(?i:[sdmt]|ll|ve|re)`
    fn contraction(self: *const Self, at: usize) ?usize {
        const quote = self.decode(at) orelse return null;
        if (quote.cp != '\'') return null;
        const first = self.decode(quote.end) orelse return null;
        if (foldsTo(first.cp, 's') or foldsTo(first.cp, 'd') or foldsTo(first.cp, 'm') or foldsTo(first.cp, 't')) return first.end;
        const second = self.decode(first.end) orelse return null;
        const pairs = [_][2]u8{ .{ 'l', 'l' }, .{ 'v', 'e' }, .{ 'r', 'e' } };
        for (pairs) |pair| {
            if (foldsTo(first.cp, pair[0]) and foldsTo(second.cp, pair[1])) return second.end;
        }
        return null;
    }

    /// `[^\r\n\p{L}\p{N}]?+\p{L}+`
    fn letters(self: *const Self, at: usize) ?usize {
        var c = self.decode(at) orelse return null;
        var pos = at;
        if (c.cp != '\r' and c.cp != '\n' and classOf(c.cp) != .letter and classOf(c.cp) != .number) {
            pos = c.end; // possessive: never given back
            c = self.decode(pos) orelse return null;
        }
        if (classOf(c.cp) != .letter) return null;
        return self.run(pos, .letter, std.math.maxInt(usize));
    }

    /// `\p{N}{1,D}`
    fn digits(self: *const Self, at: usize) ?usize {
        const c = self.decode(at) orelse return null;
        if (classOf(c.cp) != .number) return null;
        return self.run(at, .number, self.max_digits);
    }

    /// ` ?[^\s\p{L}\p{N}]++[\r\n]*`
    fn punctuation(self: *const Self, at: usize) ?usize {
        var c = self.decode(at) orelse return null;
        var pos = at;
        if (c.cp == ' ') {
            const after = self.decode(c.end) orelse return null;
            if (classOf(after.cp) != .other) return null; // backing off the space cannot help: ' ' is \s
            pos = c.end;
            c = after;
        }
        if (classOf(c.cp) != .other) return null;
        pos = self.run(pos, .other, std.math.maxInt(usize));
        while (self.decode(pos)) |nl| {
            if (nl.cp != '\r' and nl.cp != '\n') break;
            pos = nl.end;
        }
        return pos;
    }

    /// `\s*[\r\n]`: the whitespace run, cut after its last `\r` or `\n`.
    fn newlines(self: *const Self, at: usize) ?usize {
        var pos = at;
        var last: ?usize = null;
        while (self.decode(pos)) |c| {
            if (classOf(c.cp) != .space) break;
            pos = c.end;
            if (c.cp == '\r' or c.cp == '\n') last = pos;
        }
        return last;
    }

    /// `\s+(?!\S)`: the whitespace run, minus its last code point unless it ends the text.
    fn trailingSpace(self: *const Self, at: usize) ?usize {
        var pos = at;
        var prev: ?usize = null;
        while (self.decode(pos)) |c| {
            if (classOf(c.cp) != .space) break;
            prev = pos;
            pos = c.end;
        }
        if (prev == null) return null; // no whitespace at all
        if (pos == self.text.len) return pos;
        // Backtrack one code point so the lookahead sees whitespace.
        return if (prev.? > at) prev.? else null;
    }

    /// `\s+`
    fn spaces(self: *const Self, at: usize) ?usize {
        const c = self.decode(at) orelse return null;
        if (classOf(c.cp) != .space) return null;
        return self.run(at, .space, std.math.maxInt(usize));
    }

    /// End of the run of `class` code points starting at `at`, at most `limit` long.
    fn run(self: *const Self, at: usize, class: Class, limit: usize) usize {
        var pos = at;
        var count: usize = 0;
        while (count < limit) : (count += 1) {
            const c = self.decode(pos) orelse break;
            if (classOf(c.cp) != class) break;
            pos = c.end;
        }
        return pos;
    }

    const Decoded = struct { cp: u21, end: usize };

    /// The code point at byte `at`, if any (the text is validated in `init`).
    fn decode(self: *const Self, at: usize) ?Decoded {
        if (at >= self.text.len) return null;
        const len = std.unicode.utf8ByteSequenceLength(self.text[at]) catch return null;
        const cp = std.unicode.utf8Decode(self.text[at..][0..len]) catch return null;
        return .{ .cp = cp, .end = at + len };
    }

    /// The split pattern's class of a code point.
    fn classOf(cp: u21) Class {
        if (cp < 0x80) {
            // ASCII fast path.
            if ((cp >= 'a' and cp <= 'z') or (cp >= 'A' and cp <= 'Z')) return .letter;
            if (cp >= '0' and cp <= '9') return .number;
            if ((cp >= 0x09 and cp <= 0x0D) or cp == 0x20) return .space;
            return .other;
        }
        if (std.mem.indexOfScalar(u21, &white_space, cp) != null) return .space;
        return switch (mod.uucode.get(.general_category, cp)) {
            .letter_uppercase, .letter_lowercase, .letter_titlecase, .letter_modifier, .letter_other => .letter,
            .number_decimal_digit, .number_letter, .number_other => .number,
            else => .other,
        };
    }

    /// Case-insensitive match against an ASCII letter, with the one non-ASCII
    /// simple case fold the contraction letters have (`ſ` U+017F folds to `s`).
    fn foldsTo(cp: u21, letter: u8) bool {
        if (cp == letter or cp == std.ascii.toUpper(letter)) return true;
        return letter == 's' and cp == 0x017F;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

fn expectPieces(text: []const u8, expected: []const []const u8) !void {
    var it = try mod.Pretokenizer.init(text, 2);
    for (expected) |want| try std.testing.expectEqualStrings(want, it.next() orelse return error.TooFewPieces);
    try std.testing.expect(it.next() == null);
}

test "pretokenizer splits like the gpt-4 style pattern" {
    try expectPieces("Hello world! It's 12345", &.{ "Hello", " world", "!", " It", "'s", " ", "12", "34", "5" });
    try expectPieces("a  b   \n\n c", &.{ "a", " ", " b", "   \n\n", " c" });
    try expectPieces("x   ", &.{ "x", "   " });
    try expectPieces(" 'mS", &.{ " '", "mS" });
}

test "pretokenizer matches python's regex on the fixture" {
    const allocator = std.testing.allocator;
    var node = try mod.zigstorage.Node.init(allocator, std.testing.io, .empty, mod.build_options.source_root ++ "/testdata/tokenizer.json");
    defer node.deinit();
    const bytes = try node.read(.all);
    defer allocator.free(bytes);
    const Case = struct { text: []const u8, pieces: []const []const u8 };
    const Fixture = struct { cases: []const Case };
    const parsed = try std.json.parseFromSlice(Fixture, allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    for (parsed.value.cases) |case| {
        var it = try mod.Pretokenizer.init(case.text, 2);
        for (case.pieces, 0..) |want, i| {
            const got = it.next() orelse {
                std.debug.print("ran out at piece {d} of {f}\n", .{ i, std.json.fmt(case.text, .{}) });
                return error.TooFewPieces;
            };
            if (!std.mem.eql(u8, want, got)) {
                std.debug.print("piece {d} of {f}: want {f}, got {f}\n", .{ i, std.json.fmt(case.text, .{}), std.json.fmt(want, .{}), std.json.fmt(got, .{}) });
                return error.PieceMismatch;
            }
        }
        try std.testing.expect(it.next() == null);
    }
}
