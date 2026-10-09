const std = @import("std");
const log = std.log.scoped(.zignanogpt_data_loader);
const mod = @import("module.zig");

/// nanochat's BOS-aligned best-fit pretraining loader
/// (`tokenizing_distributed_data_loader_with_state_bos_bestfit`, one rank).
///
/// Each document becomes `[BOS] + tokens` in a buffer kept at `buffer_size`
/// documents. Each of the `B` rows (`T + 1` tokens) is filled by repeatedly
/// taking the largest buffered document that fits whole (first on a tie);
/// when none fits, the shortest is cropped to fill the row exactly and its
/// remainder dropped. Rows start with BOS and have no padding.
pub const DataLoader = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    tokenizer: *const mod.Tokenizer,
    stream: *mod.DocumentStream,
    batch: usize,
    seq: usize,
    buffer_size: usize,
    bos: u32,
    /// Tokenized documents, in arrival order.
    docs: std.ArrayList([]u32) = .empty,
    /// One row being filled: `seq + 1` tokens.
    row: []u32,
    /// The state of the last refill.
    state: mod.DataLoaderState = .{},

    /// Creates a loader over a document stream.
    ///
    /// Parameters:
    /// - `allocator`: owns the buffers.
    /// - `tokenizer`: encodes the documents; must outlive the loader.
    /// - `stream`: the documents; must outlive the loader.
    /// - `batch`: rows per batch (B).
    /// - `seq`: tokens per row (T).
    /// - `buffer_size`: documents kept for best fit (nanochat: 1000).
    ///
    /// Return: the loader; allocation errors.
    pub fn init(allocator: std.mem.Allocator, tokenizer: *const mod.Tokenizer, stream: *mod.DocumentStream, batch: usize, seq: usize, buffer_size: usize) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        return Self{
            .allocator = allocator,
            .tokenizer = tokenizer,
            .stream = stream,
            .batch = batch,
            .seq = seq,
            .buffer_size = @max(buffer_size, 1),
            .bos = try tokenizer.bos(),
            .row = try allocator.alloc(u32, seq + 1),
        };
    }

    /// Frees the buffers.
    ///
    /// Parameters:
    /// - `self`: the loader.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        for (self.docs.items) |d| self.allocator.free(d);
        self.docs.deinit(self.allocator);
        self.allocator.free(self.row);
    }

    /// Fills the next batch.
    ///
    /// Parameters:
    /// - `self`: the loader.
    /// - `inputs`: `B * T` token ids (`row[:-1]` per row).
    /// - `targets`: `B * T` next tokens (`row[1:]` per row).
    ///
    /// Return: the state to resume from; stream and tokenizer errors.
    pub fn next(self: *Self, inputs: []i32, targets: []i32) !mod.DataLoaderState {
        if (inputs.len != self.batch * self.seq or targets.len != inputs.len) return error.ShapeMismatch;
        const capacity = self.seq + 1;
        for (0..self.batch) |b| {
            var pos: usize = 0;
            while (pos < capacity) {
                while (self.docs.items.len < self.buffer_size) try self.refill();
                const remaining = capacity - pos;
                var best: ?usize = null;
                var best_len: usize = 0;
                for (self.docs.items, 0..) |doc, i| {
                    if (doc.len <= remaining and doc.len > best_len) {
                        best = i;
                        best_len = doc.len;
                    }
                }
                if (best) |i| {
                    const doc = self.docs.orderedRemove(i);
                    defer self.allocator.free(doc);
                    @memcpy(self.row[pos..][0..doc.len], doc);
                    pos += doc.len;
                } else {
                    var shortest: usize = 0;
                    for (self.docs.items, 0..) |doc, i| {
                        if (doc.len < self.docs.items[shortest].len) shortest = i;
                    }
                    const doc = self.docs.orderedRemove(shortest);
                    defer self.allocator.free(doc);
                    @memcpy(self.row[pos..capacity], doc[0..remaining]);
                    pos = capacity;
                }
            }
            for (inputs[b * self.seq ..][0..self.seq], targets[b * self.seq ..][0..self.seq], 0..) |*x, *y, t| {
                x.* = @intCast(self.row[t]);
                y.* = @intCast(self.row[t + 1]);
            }
        }
        return self.state;
    }

    /// Tokenizes the next batch of documents into the buffer.
    fn refill(self: *Self) !void {
        const batch = try self.stream.next();
        self.state = batch.state;
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(self.allocator);
        for (batch.docs) |text| {
            ids.clearRetainingCapacity();
            try ids.append(self.allocator, self.bos);
            try self.tokenizer.encodeAppend(self.allocator, &ids, text);
            const owned = try self.allocator.dupe(u32, ids.items);
            errdefer self.allocator.free(owned);
            try self.docs.append(self.allocator, owned);
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

const Fixture = struct {
    parsed: std.json.Parsed(std.json.Value),
    bytes: []u8,

    fn load(allocator: std.mem.Allocator, name: []const u8) !Fixture {
        const path = try std.Io.Dir.path.join(allocator, &.{ mod.build_options.source_root, "testdata", name });
        defer allocator.free(path);
        const bytes = try mod.Storage.init(allocator, std.testing.io).read(path);
        errdefer allocator.free(bytes);
        return .{ .parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}), .bytes = bytes };
    }

    fn deinit(self: *Fixture, allocator: std.mem.Allocator) void {
        self.parsed.deinit();
        allocator.free(self.bytes);
    }
};

fn fixtureTokenizer(allocator: std.mem.Allocator) !mod.Tokenizer {
    var fixture = try Fixture.load(allocator, "tokenizer.json");
    defer fixture.deinit(allocator);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ranks = fixture.parsed.value.object.get("ranks").?.array.items;
    const tokens = try arena.allocator().alloc([]const u8, ranks.len);
    for (ranks, tokens) |r, *t| {
        const bytes = try arena.allocator().alloc(u8, r.array.items.len);
        for (r.array.items, bytes) |b, *o| o.* = @intCast(b.integer);
        t.* = bytes;
    }
    return mod.Tokenizer.init(allocator, tokens, &mod.Tokenizer.special_tokens, mod.Tokenizer.nanochat_max_digits);
}

fn expectBatches(allocator: std.mem.Allocator, tok: *const mod.Tokenizer, paths: []const []const u8, want: []const std.json.Value, resume_from: ?mod.DataLoaderState, b: usize, t: usize, tok_batch: usize, buffer: usize) !void {
    var stream = try mod.DocumentStream.init(allocator, std.testing.io, .{ .parquet = paths }, tok_batch, resume_from);
    defer stream.deinit();
    var loader = try mod.DataLoader.init(allocator, tok, &stream, b, t, buffer);
    defer loader.deinit();
    const inputs = try allocator.alloc(i32, b * t);
    defer allocator.free(inputs);
    const targets = try allocator.alloc(i32, b * t);
    defer allocator.free(targets);
    for (want, 0..) |batch, n| {
        const state = try loader.next(inputs, targets);
        const want_state = batch.object.get("state").?.object;
        for (batch.object.get("inputs").?.array.items, inputs) |w, g| {
            if (w.integer != g) {
                std.debug.print("batch {d}: inputs differ\n", .{n});
                return error.ParityMismatch;
            }
        }
        for (batch.object.get("targets").?.array.items, targets) |w, g| try std.testing.expectEqual(w.integer, g);
        try std.testing.expectEqual(want_state.get("pq_idx").?.integer, @as(i64, @intCast(state.pq_idx)));
        try std.testing.expectEqual(want_state.get("rg_idx").?.integer, @as(i64, @intCast(state.rg_idx)));
        try std.testing.expectEqual(want_state.get("epoch").?.integer, @as(i64, @intCast(state.epoch)));
    }
}

test "data loader packs batches exactly like nanochat" {
    const allocator = std.testing.allocator;
    var tok = try fixtureTokenizer(allocator);
    defer tok.deinit();
    var fixture = try Fixture.load(allocator, "dataloader.json");
    defer fixture.deinit(allocator);
    const obj = fixture.parsed.value.object;
    const b: usize = @intCast(obj.get("B").?.integer);
    const t: usize = @intCast(obj.get("T").?.integer);
    const tok_batch: usize = @intCast(obj.get("tokenizer_batch_size").?.integer);
    const buffer: usize = @intCast(obj.get("buffer_size").?.integer);

    var paths: [3][]const u8 = undefined;
    var path_bufs: [3][256]u8 = undefined;
    for (&paths, &path_bufs, 0..) |*p, *buf, i| p.* = try std.mem.print(buf, "{s}/testdata/shards/shard_{d:0>5}.parquet", .{ mod.build_options.source_root, i });

    // Train: every shard but the last; val: the last.
    try expectBatches(allocator, &tok, paths[0..2], obj.get("train").?.array.items, null, b, t, tok_batch, buffer);
    const from = obj.get("train").?.array.items[@intCast(obj.get("resume_from").?.integer)].object.get("state").?.object;
    const resumed = mod.DataLoaderState{
        .pq_idx = @intCast(from.get("pq_idx").?.integer),
        .rg_idx = @intCast(from.get("rg_idx").?.integer),
        .epoch = @intCast(from.get("epoch").?.integer),
    };
    try expectBatches(allocator, &tok, paths[0..2], obj.get("resumed").?.array.items, resumed, b, t, tok_batch, buffer);
    try expectBatches(allocator, &tok, paths[2..3], obj.get("val").?.array.items, null, b, t, tok_batch, buffer);
}
