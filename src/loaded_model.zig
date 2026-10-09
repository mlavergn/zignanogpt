const std = @import("std");
const log = std.log.scoped(.zignanogpt_loaded_model);
const mod = @import("module.zig");

/// Which checkpoint to load (nanochat's `load_model(source, model_tag, step)`).
pub const ModelSelection = struct {
    kind: mod.CheckpointKind = .base,
    /// Null for the largest `d<N>`.
    tag: ?[]const u8 = null,
    /// Null for the last.
    step: ?usize = null,
};

/// A model and tokenizer loaded from this port's base directory
/// (`<base>/<kind>_checkpoints/<tag>/model_<step>.safetensors`, `<base>/tokenizer`).
/// It must stay at its address: the model's config lives in its arena.
pub const LoadedModel = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    kind: mod.CheckpointKind,
    tag: []const u8,
    step: usize,
    model: mod.Gpt,
    tokenizer: mod.Tokenizer,

    /// Loads a checkpoint and the tokenizer.
    ///
    /// Parameters:
    /// - `self`: the storage.
    /// - `allocator`: owns everything.
    /// - `backend`: holds the weights.
    /// - `storage`: file access.
    /// - `base_dir`: this port's base directory.
    /// - `selection`: kind, tag and step.
    ///
    /// Return: nothing; `error.NoCheckpoint`, `error.TokenizerMismatch`, storage and format errors.
    pub fn init(self: *Self, allocator: std.mem.Allocator, backend: *mod.Backend, storage: mod.Storage, base_dir: []const u8, selection: ModelSelection) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.allocator = allocator;
        self.arena = std.heap.ArenaAllocator.init(allocator);
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        self.kind = selection.kind;
        self.tag = if (selection.tag) |t| try a.dupe(u8, t) else blk: {
            const t = try mod.Checkpoint.largestTag(allocator, storage, base_dir, selection.kind);
            defer allocator.free(t);
            break :blk try a.dupe(u8, t);
        };
        var ckpt = try mod.Checkpoint.init(allocator, storage, base_dir, selection.kind, self.tag);
        defer ckpt.deinit();
        self.step = selection.step orelse try ckpt.lastStep("safetensors");
        log.info("loading {s} {s} step {d}", .{ @tagName(selection.kind), self.tag, self.step });

        var meta = try ckpt.loadMeta(self.step);
        defer meta.deinit();
        const model_config = meta.value.object.get("model_config") orelse {
            log.warn("{s}: meta lacks model_config", .{ckpt.dir});
            return error.InvalidCheckpoint;
        };
        const config = try mod.GptConfig.fromJson(a, model_config);
        self.model = try mod.Gpt.init(allocator, backend, config);
        errdefer self.model.deinit();
        try ckpt.loadModel(self.step, &self.model.weights);

        const tok_dir = try std.Io.Dir.path.join(allocator, &.{ base_dir, "tokenizer" });
        defer allocator.free(tok_dir);
        self.tokenizer = try mod.Tokenizer.load(allocator, storage, tok_dir);
        errdefer self.tokenizer.deinit();
        if (self.tokenizer.vocabSize() != config.vocab_size) {
            log.warn("tokenizer vocab {d} does not match the model's {d}", .{ self.tokenizer.vocabSize(), config.vocab_size });
            return error.TokenizerMismatch;
        }
    }

    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.tokenizer.deinit();
        self.model.deinit();
        self.arena.deinit();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "loaded model picks the largest tag and last step of an imported checkpoint" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const base = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const storage = mod.Storage.init(allocator, std.testing.io);
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();
    const imported = try mod.TorchImport.importCheckpoint(allocator, &backend, storage, mod.build_options.source_root ++ "/testdata/nanochat_base", base, .base, null, null);
    defer allocator.free(imported.tag);

    var loaded: LoadedModel = undefined;
    try loaded.init(allocator, &backend, storage, base, .{});
    defer loaded.deinit();
    try std.testing.expectEqualStrings("d2", loaded.tag);
    try std.testing.expectEqual(@as(usize, 5), loaded.step);
    try std.testing.expectEqual(@as(usize, 2), loaded.model.config.n_layer);
    try std.testing.expectEqual(@as(usize, 1033), loaded.tokenizer.vocabSize());

    var missing: LoadedModel = undefined;
    try std.testing.expectError(error.NoCheckpoint, missing.init(allocator, &backend, storage, base, .{ .kind = .sft }));
}
