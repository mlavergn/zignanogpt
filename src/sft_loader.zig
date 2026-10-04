const std = @import("std");
const log = std.log.scoped(.zignanogpt_sft_loader);
const mod = @import("module.zig");

/// `chat_sft.py`'s `sft_data_generator_bos_bestfit` (one rank): conversations
/// rendered with their loss masks are packed whole into `seq + 1`-token rows,
/// largest that fits first from a 100-conversation buffer; when none fits the
/// row is padded with BOS (masked) instead of cropping anything. Targets are -1
/// wherever the mask is 0 and over the padding.
///
/// One deliberate change: conversations are rendered with at most
/// `min(2048, seq + 1)` tokens (Python always uses 2048). With a context below
/// 2048 Python's long conversations never fit, pile up in the buffer and leave
/// every later row padding; at 2048 the two agree.
pub const SftLoader = struct {
    const Self = @This();

    /// Conversations kept for best-fit selection.
    pub const buffer_size = 100;
    /// `render_conversation`'s default cap.
    pub const render_max = 2048;

    const Rendered = struct { ids: []u32, mask: []u8 };

    allocator: std.mem.Allocator,
    tokenizer: *const mod.Tokenizer,
    mixture: *const mod.TaskMixture,
    batch: usize,
    seq: usize,
    /// Stops after this many batches (0: one pass over the data); training only.
    num_iterations: usize,
    train: bool,
    buffer: std.ArrayList(Rendered) = .empty,
    arena: std.heap.ArenaAllocator,
    row: std.ArrayList(u32) = .empty,
    row_mask: std.ArrayList(u8) = .empty,

    cursor: usize = 0,
    consumed: usize = 0,
    epoch: usize = 1,
    it: usize = 0,
    /// Set once the data (or `num_iterations`) is used up.
    last_step: bool = false,
    /// 0 -> 1 over the run.
    progress: f64 = 0,
    current_epoch: usize = 1,

    /// Creates a loader at the start of the data.
    ///
    /// Parameters:
    /// - `allocator`: owns the buffer.
    /// - `tokenizer`: renders conversations (outlives the loader).
    /// - `mixture`: the conversations (outlives the loader).
    /// - `batch`: rows per batch.
    /// - `seq`: tokens per row.
    /// - `num_iterations`: batches before `last_step` (0: one pass).
    /// - `train`: false for validation (no progress or stopping).
    ///
    /// Return: the loader; `error.EmptyDataset`.
    pub fn init(allocator: std.mem.Allocator, tokenizer: *const mod.Tokenizer, mixture: *const mod.TaskMixture, batch: usize, seq: usize, num_iterations: usize, train: bool) !Self {
        if (mixture.len() == 0) return error.EmptyDataset;
        return .{ .allocator = allocator, .tokenizer = tokenizer, .mixture = mixture, .batch = batch, .seq = seq, .num_iterations = num_iterations, .train = train, .arena = std.heap.ArenaAllocator.init(allocator) };
    }

    pub fn deinit(self: *Self) void {
        for (self.buffer.items) |r| self.free(r);
        self.buffer.deinit(self.allocator);
        self.row.deinit(self.allocator);
        self.row_mask.deinit(self.allocator);
        self.arena.deinit();
    }

    /// Fills the next batch.
    ///
    /// Parameters:
    /// - `self`: the loader.
    /// - `inputs`: `[batch, seq]` token ids.
    /// - `targets`: `[batch, seq]` next tokens, -1 where not trained on.
    ///
    /// Return: nothing; rendering and allocation errors.
    pub fn next(self: *Self, inputs: []i32, targets: []i32) !void {
        const t = self.seq;
        const capacity = t + 1;
        const bos = try self.tokenizer.bos();
        for (0..self.batch) |b| {
            self.row.clearRetainingCapacity();
            self.row_mask.clearRetainingCapacity();
            var content: usize = capacity;
            while (self.row.items.len < capacity) {
                try self.refill();
                const remaining = capacity - self.row.items.len;
                var best: ?usize = null;
                var best_len: usize = 0;
                for (self.buffer.items, 0..) |r, i| {
                    if (r.ids.len <= remaining and r.ids.len > best_len) {
                        best = i;
                        best_len = r.ids.len;
                    }
                }
                if (best) |i| {
                    const r = self.buffer.orderedRemove(i);
                    defer self.free(r);
                    try self.row.appendSlice(self.allocator, r.ids);
                    try self.row_mask.appendSlice(self.allocator, r.mask);
                    self.consumed += 1;
                } else {
                    content = self.row.items.len;
                    try self.row.appendNTimes(self.allocator, bos, remaining);
                    try self.row_mask.appendNTimes(self.allocator, 0, remaining);
                    break;
                }
            }
            const ids = self.row.items;
            const mask = self.row_mask.items;
            const in = inputs[b * t ..][0..t];
            const out = targets[b * t ..][0..t];
            for (in, ids[0..t]) |*d, s| d.* = @intCast(s);
            for (out, ids[1..capacity], mask[1..capacity]) |*d, s, m| d.* = if (m == 0) -1 else @intCast(s);
            if (content < capacity) {
                // Python's targets[i, content_len - 1:] (index -1 when the row is all padding)
                const from = if (content > 0) content - 1 else t - 1;
                @memset(out[from..], -1);
            }
        }
        self.it += 1;
        if (!self.train) return;
        if (self.num_iterations > 0 and self.it >= self.num_iterations) self.last_step = true;
        self.current_epoch = self.epoch;
        const size: f64 = @floatFromInt(self.mixture.len());
        self.progress = if (self.num_iterations > 0)
            @as(f64, @floatFromInt(self.it)) / @as(f64, @floatFromInt(self.num_iterations))
        else
            @as(f64, @floatFromInt(self.consumed)) / size;
        if (self.consumed >= self.mixture.len()) self.last_step = true;
    }

    /// Tops the buffer up to `buffer_size` rendered conversations.
    fn refill(self: *Self) !void {
        const cap = @min(render_max, self.seq + 1);
        while (self.buffer.items.len < buffer_size) {
            _ = self.arena.reset(.retain_capacity);
            const conv = try self.mixture.conversation(self.arena.allocator(), self.cursor);
            var r = try self.tokenizer.renderConversation(self.allocator, conv, cap);
            errdefer r.deinit(self.allocator);
            const ids = try r.ids.toOwnedSlice(self.allocator);
            errdefer self.allocator.free(ids);
            const mask = try r.mask.toOwnedSlice(self.allocator);
            errdefer self.allocator.free(mask);
            try self.buffer.append(self.allocator, .{ .ids = ids, .mask = mask });
            self.cursor += 1;
            if (self.cursor >= self.mixture.len()) {
                self.cursor %= self.mixture.len();
                self.epoch += 1;
                log.debug("sft data epoch {d}", .{self.epoch});
            }
        }
    }

    fn free(self: *Self, r: Rendered) void {
        self.allocator.free(r.ids);
        self.allocator.free(r.mask);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "sft loader packs conversations exactly as chat_sft.py" {
    const allocator = std.testing.allocator;
    const root = mod.build_options.source_root ++ "/testdata";
    const storage = mod.Storage.init(allocator, std.testing.io);
    var tok = try mod.TorchImport.loadTokenizer(allocator, storage, root ++ "/nanochat_base/tokenizer/tokenizer.pkl");
    defer tok.deinit();
    const config = mod.Config{ .allocator = allocator, .base_dir = root ++ "/task_base", .nanochat_dir = "/nonexistent-nanochat", .data_url = "http://unused" };
    var smoltalk = try mod.Task.open(allocator, std.testing.io, &config, .smoltalk, "test", null);
    defer smoltalk.deinit();
    var mmlu = try mod.Task.open(allocator, std.testing.io, &config, .mmlu, "test", null);
    defer mmlu.deinit();
    var gsm8k = try mod.Task.open(allocator, std.testing.io, &config, .gsm8k, "test", null);
    defer gsm8k.deinit();
    var mixture = try mod.TaskMixture.init(allocator, &.{ &smoltalk, &mmlu, &gsm8k });
    defer mixture.deinit();

    const bytes = try storage.read(root ++ "/sft_loader.json");
    defer allocator.free(bytes);
    const Batch = struct { inputs: []const []const i32, targets: []const []const i32, last_step: bool, approx_progress: f64, current_epoch: usize };
    var parsed = try std.json.parseFromSlice(std.json.ArrayHashMap(struct { seq: usize, num_iterations: usize, batches: []const Batch }), allocator, bytes, .{});
    defer parsed.deinit();
    var it = parsed.value.map.iterator();
    while (it.next()) |entry| {
        const case = entry.value_ptr.*;
        var loader = try SftLoader.init(allocator, &tok, &mixture, 2, case.seq, case.num_iterations, true);
        defer loader.deinit();
        const inputs = try allocator.alloc(i32, 2 * case.seq);
        defer allocator.free(inputs);
        const targets = try allocator.alloc(i32, 2 * case.seq);
        defer allocator.free(targets);
        for (case.batches, 0..) |want, bi| {
            try loader.next(inputs, targets);
            for (0..2) |r| {
                std.testing.expectEqualSlices(i32, want.inputs[r], inputs[r * case.seq ..][0..case.seq]) catch |err| {
                    std.debug.print("{s} batch {d} row {d} inputs\n", .{ entry.key_ptr.*, bi, r });
                    return err;
                };
                try std.testing.expectEqualSlices(i32, want.targets[r], targets[r * case.seq ..][0..case.seq]);
            }
            try std.testing.expectEqual(want.last_step, loader.last_step);
            try std.testing.expectEqual(want.approx_progress, loader.progress);
            try std.testing.expectEqual(want.current_epoch, loader.current_epoch);
        }
    }
}
