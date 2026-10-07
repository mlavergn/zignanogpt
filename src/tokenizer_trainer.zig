const std = @import("std");
const log = std.log.scoped(.zignanogpt_tokenizer_trainer);
const mod = @import("module.zig");

/// A pair of token ids packed as `left << 32 | right` (ordering = tuple ordering).
const Pair = u64;

fn pairOf(a: u32, b: u32) Pair {
    return (@as(u64, a) << 32) | b;
}

/// A queued merge candidate: a pair, its count when queued, and the words it
/// may occur in.
const Job = struct {
    pair: Pair,
    count: i64,
    words: std.ArrayList(u32),

    /// Most frequent first; on a tie, the smaller pair first (rustbpe's order).
    fn order(context: void, a: Job, b: Job) std.math.Order {
        _ = context;
        if (a.count != b.count) return if (a.count > b.count) .lt else .gt;
        return std.math.order(a.pair, b.pair);
    }
};

/// Trains a byte-level BPE vocabulary as rustbpe does (`train_from_iterator`):
/// texts are split with the pretokenizer, identical pieces are counted, and
/// merges are made one at a time on the most frequent adjacent pair (ties to
/// the smaller pair), leftmost non-overlapping within each piece. Counts are
/// updated incrementally from each merge's local deltas, with stale queue
/// entries refreshed lazily when popped.
pub const TokenizerTrainer = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    /// Owns the piece keys.
    arena: std.heap.ArenaAllocator,
    counts: std.StringHashMapUnmanaged(i64),
    max_digits: usize,
    /// Receives the merges' progress, or null.
    observer: ?mod.TrainObserver = null,

    /// Creates an empty trainer.
    ///
    /// Parameters:
    /// - `allocator`: backs the counts.
    /// - `max_digits`: the split pattern's digit-run bound.
    ///
    /// Return: the trainer.
    pub fn init(allocator: std.mem.Allocator, max_digits: usize) Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        return Self{ .allocator = allocator, .arena = std.heap.ArenaAllocator.init(allocator), .counts = .empty, .max_digits = max_digits };
    }

    /// Frees the counts.
    ///
    /// Parameters:
    /// - `self`: the trainer.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.counts.deinit(self.allocator);
        self.arena.deinit();
    }

    /// Splits one document and counts its pieces.
    ///
    /// Parameters:
    /// - `self`: the trainer.
    /// - `text`: valid UTF-8.
    ///
    /// Return: nothing; `error.InvalidUtf8`, allocation errors.
    pub fn addText(self: *Self, text: []const u8) !void {
        var pieces = try mod.Pretokenizer.init(text, self.max_digits);
        while (pieces.next()) |piece| {
            const entry = try self.counts.getOrPut(self.allocator, piece);
            if (!entry.found_existing) {
                entry.key_ptr.* = try self.arena.allocator().dupe(u8, piece);
                entry.value_ptr.* = 0;
            }
            entry.value_ptr.* += 1;
        }
    }

    /// Distinct pieces seen so far.
    pub fn uniquePieces(self: *const Self) usize {
        return self.counts.count();
    }

    /// Learns `vocab_size - 256` merges and returns the token bytes by rank
    /// (the 256 single bytes, then one entry per merge).
    ///
    /// Parameters:
    /// - `self`: the trainer.
    /// - `allocator`: owns the result (each entry and the outer slice).
    /// - `vocab_size`: mergeable vocabulary size, at least 256 (no specials).
    ///
    /// Return: the tokens; fewer merges if the pairs run out.
    pub fn train(self: *Self, allocator: std.mem.Allocator, vocab_size: usize) ![][]u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (vocab_size < 256) return error.VocabTooSmall;
        const gpa = self.allocator;

        // One word per distinct piece, as byte ids, with its count.
        const n_words = self.counts.count();
        const words = try gpa.alloc(std.ArrayList(u32), n_words);
        defer {
            for (words) |*w| w.deinit(gpa);
            gpa.free(words);
        }
        const word_counts = try gpa.alloc(i64, n_words);
        defer gpa.free(word_counts);
        var it = self.counts.iterator();
        var wi: usize = 0;
        while (it.next()) |kv| : (wi += 1) {
            words[wi] = .empty;
            try words[wi].ensureTotalCapacity(gpa, kv.key_ptr.len);
            for (kv.key_ptr.*) |b| words[wi].appendAssumeCapacity(b);
            word_counts[wi] = kv.value_ptr.*;
        }

        // Initial pair counts and the words each pair occurs in.
        var pair_counts: std.AutoHashMapUnmanaged(Pair, i64) = .empty;
        defer pair_counts.deinit(gpa);
        var where: std.AutoArrayHashMapUnmanaged(Pair, std.ArrayList(u32)) = .empty;
        defer where.deinit(gpa);
        for (words, word_counts, 0..) |w, count, i| {
            if (w.items.len < 2 or count == 0) continue;
            for (0..w.items.len - 1) |j| {
                const pair = pairOf(w.items[j], w.items[j + 1]);
                (try pair_counts.getOrPutValue(gpa, pair, 0)).value_ptr.* += count;
                const entry = try where.getOrPut(gpa, pair);
                if (!entry.found_existing) entry.value_ptr.* = .empty;
                try appendOnce(gpa, entry.value_ptr, @intCast(i));
            }
        }

        var heap = std.PriorityQueue(Job, void, Job.order).empty;
        defer {
            while (heap.pop()) |job| {
                var j = job;
                j.words.deinit(gpa);
            }
            heap.deinit(gpa);
        }
        var wit = where.iterator();
        while (wit.next()) |kv| {
            const count = pair_counts.get(kv.key_ptr.*) orelse 0;
            if (count > 0) {
                try heap.push(gpa, .{ .pair = kv.key_ptr.*, .count = count, .words = kv.value_ptr.* });
            } else {
                kv.value_ptr.deinit(gpa);
            }
        }

        // Token bytes by id.
        var tokens: std.ArrayList([]u8) = .empty;
        errdefer {
            for (tokens.items) |t| allocator.free(t);
            tokens.deinit(allocator);
        }
        for (0..256) |b| try tokens.append(allocator, try allocator.dupe(u8, &.{@intCast(b)}));

        var updates: std.AutoArrayHashMapUnmanaged(Pair, std.ArrayList(u32)) = .empty;
        defer updates.deinit(gpa);
        var deltas: std.ArrayList(Delta) = .empty;
        defer deltas.deinit(gpa);
        const merges = vocab_size - 256;
        while (tokens.items.len - 256 < merges) {
            const done = tokens.items.len - 256;
            if (done % 64 == 0) mod.TrainObserver.progress(self.observer, "merges", done, merges);
            var top = heap.pop() orelse break;
            const current = pair_counts.get(top.pair) orelse 0;
            if (current <= 0) {
                top.words.deinit(gpa);
                continue;
            }
            if (top.count != current) {
                top.count = current;
                try heap.push(gpa, top);
                continue;
            }
            defer top.words.deinit(gpa);

            const new_id: u32 = @intCast(tokens.items.len);
            const left: u32 = @intCast(top.pair >> 32);
            const right: u32 = @truncate(top.pair);
            try tokens.append(allocator, try std.mem.concat(allocator, u8, &.{ tokens.items[left], tokens.items[right] }));

            for (top.words.items) |w| {
                deltas.clearRetainingCapacity();
                try mergeWord(gpa, &words[w], left, right, new_id, &deltas);
                for (deltas.items) |d| {
                    const total = d.delta * word_counts[w];
                    if (total == 0) continue;
                    (try pair_counts.getOrPutValue(gpa, d.pair, 0)).value_ptr.* += total;
                    if (d.delta > 0) {
                        const entry = try updates.getOrPut(gpa, d.pair);
                        if (!entry.found_existing) entry.value_ptr.* = .empty;
                        try appendOnce(gpa, entry.value_ptr, w);
                    }
                }
            }
            var uit = updates.iterator();
            while (uit.next()) |kv| {
                const count = pair_counts.get(kv.key_ptr.*) orelse 0;
                if (count > 0) {
                    try heap.push(gpa, .{ .pair = kv.key_ptr.*, .count = count, .words = kv.value_ptr.* });
                } else {
                    kv.value_ptr.deinit(gpa);
                }
            }
            updates.clearRetainingCapacity();
            if (tokens.items.len % 1000 == 0) log.info("merges: {d}/{d}", .{ tokens.items.len - 256, merges });
        }
        return tokens.toOwnedSlice(allocator);
    }

    const Delta = struct { pair: Pair, delta: i64 };

    /// Merges every leftmost non-overlapping `(a, b)` in `word` into `new_id`,
    /// recording the pair-count changes (rustbpe's `Word::merge_pair`).
    fn mergeWord(gpa: std.mem.Allocator, word: *std.ArrayList(u32), a: u32, b: u32, new_id: u32, deltas: *std.ArrayList(Delta)) !void {
        const ids = word.items;
        if (ids.len < 2) return;
        var out: usize = 0;
        var i: usize = 0;
        while (i < ids.len) {
            if (i + 1 < ids.len and ids[i] == a and ids[i + 1] == b) {
                if (out > 0) {
                    const x = ids[out - 1];
                    try deltas.append(gpa, .{ .pair = pairOf(x, a), .delta = -1 });
                    try deltas.append(gpa, .{ .pair = pairOf(x, new_id), .delta = 1 });
                }
                try deltas.append(gpa, .{ .pair = pairOf(a, b), .delta = -1 });
                if (i + 2 < ids.len) {
                    const y = ids[i + 2];
                    try deltas.append(gpa, .{ .pair = pairOf(b, y), .delta = -1 });
                    try deltas.append(gpa, .{ .pair = pairOf(new_id, y), .delta = 1 });
                }
                ids[out] = new_id;
                i += 2;
            } else {
                ids[out] = ids[i];
                i += 1;
            }
            out += 1;
        }
        word.shrinkRetainingCapacity(out);
    }

    /// Appends `w` unless it is already the last entry (words are visited in order).
    fn appendOnce(gpa: std.mem.Allocator, list: *std.ArrayList(u32), w: u32) !void {
        if (list.items.len > 0 and list.items[list.items.len - 1] == w) return;
        try list.append(gpa, w);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "tokenizer trainer learns the same merges as rustbpe" {
    const allocator = std.testing.allocator;
    var node = try mod.zigstorage.Node.init(allocator, std.testing.io, .empty, mod.build_options.source_root ++ "/testdata/tokenizer.json");
    defer node.deinit();
    const bytes = try node.read(.all);
    defer allocator.free(bytes);
    // ranks are JSON arrays of byte values.
    const Raw = struct { docs: []const []const u8, ranks: []const []const u8, vocab_size: usize };
    const parsed = try std.json.parseFromSlice(Raw, allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    var trainer = mod.TokenizerTrainer.init(allocator, 2);
    defer trainer.deinit();
    for (parsed.value.docs) |doc| try trainer.addText(doc);
    const tokens = try trainer.train(allocator, parsed.value.vocab_size - 9);
    defer {
        for (tokens) |t| allocator.free(t);
        allocator.free(tokens);
    }
    try std.testing.expectEqual(parsed.value.ranks.len, tokens.len);
    for (parsed.value.ranks, tokens, 0..) |want, got, rank| {
        std.testing.expectEqualSlices(u8, want, got) catch |err| {
            std.debug.print("rank {d} differs\n", .{rank});
            return err;
        };
    }
}

test "tokenizer trainer merges the most frequent pair, ties to the smaller pair" {
    const allocator = std.testing.allocator;
    var trainer = mod.TokenizerTrainer.init(allocator, 2);
    defer trainer.deinit();
    // Pieces "ab", " ab" x2, " cd" x3: (a,b), ( ,c) and (c,d) all count 3, so the
    // smallest pair ( ,c) merges first; then (a,b) ties (" c",d) and is smaller.
    try trainer.addText("ab ab ab cd cd cd");
    const tokens = try trainer.train(allocator, 258);
    defer {
        for (tokens) |t| allocator.free(t);
        allocator.free(tokens);
    }
    try std.testing.expectEqualStrings(" c", tokens[256]);
    try std.testing.expectEqualStrings("ab", tokens[257]);
}
