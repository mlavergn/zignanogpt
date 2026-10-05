const std = @import("std");
const log = std.log.scoped(.zignanogpt_torch_import);
const mod = @import("../module.zig");

/// One imported tensor, upcast to f32 and made contiguous.
pub const StateTensor = struct {
    name: []const u8,
    shape: mod.Shape,
    data: []f32,
};

/// A PyTorch `state_dict` read from a `torch.save` file.
pub const StateDict = struct {
    const Self = @This();

    arena: std.heap.ArenaAllocator,
    tensors: []StateTensor,

    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    /// Looks a tensor up by name.
    pub fn get(self: *const Self, name: []const u8) ?StateTensor {
        for (self.tensors) |t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }
};

/// Element types of PyTorch storages the importer converts to f32.
const StorageDtype = enum { f32, f64, f16, bf16 };

/// Imports Python nanochat artifacts: `torch.save`'d state dicts (a zip of a
/// pickle plus raw storages) and the pickled tiktoken `tokenizer.pkl`.
pub const TorchImport = struct {
    /// Reads a `model_*.pt` state dict into f32 tensors. Keys lose the
    /// `_orig_mod.` prefix `torch.compile` adds, as nanochat's `build_model` does.
    ///
    /// Parameters:
    /// - `allocator`: backs the result's arena.
    /// - `io`: the Io reads run on.
    /// - `path`: the `.pt` file.
    ///
    /// Return: the tensors; zip, pickle and format errors.
    pub fn loadStateDict(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !StateDict {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var zip = try mod.ZipArchive.open(allocator, io, path);
        defer zip.deinit();
        const pickle_entry = try zip.find("/data.pkl");
        const pickle_bytes = try zip.read(pickle_entry);
        defer allocator.free(pickle_bytes);
        var pickle = try mod.Pickle.parse(allocator, pickle_bytes);
        defer pickle.deinit();
        if (pickle.root != .dict) return invalid("state dict is not a dict");
        // Members are `<archive>/data/<key>`, with `<archive>` the pickle's folder.
        const archive = pickle_entry.name[0 .. pickle_entry.name.len - "data.pkl".len];

        var result = StateDict{ .arena = std.heap.ArenaAllocator.init(allocator), .tensors = &.{} };
        errdefer result.arena.deinit();
        const arena = result.arena.allocator();
        const tensors = try arena.alloc(StateTensor, pickle.root.dict.items.len);
        for (pickle.root.dict.items, tensors) |entry, *out| {
            if (entry.key != .string) return invalid("state dict key is not a string");
            const name = entry.key.string;
            const stripped = if (std.mem.startsWith(u8, name, "_orig_mod.")) name["_orig_mod.".len..] else name;
            out.* = try rebuildTensor(arena, &zip, archive, try arena.dupe(u8, stripped), entry.value);
        }
        result.tensors = tensors;
        return result;
    }

    /// Reads nanochat's `tokenizer.pkl` (a pickled `tiktoken.Encoding`).
    ///
    /// Parameters:
    /// - `allocator`: owns the tokenizer.
    /// - `storage`: reads the file.
    /// - `path`: the `.pkl` file.
    ///
    /// Return: the tokenizer; `error.UnsupportedPattern` for a split pattern this port does not implement.
    pub fn loadTokenizer(allocator: std.mem.Allocator, storage: mod.Storage, path: []const u8) !mod.Tokenizer {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const bytes = try storage.read(path);
        defer allocator.free(bytes);
        var pickle = try mod.Pickle.parse(allocator, bytes);
        defer pickle.deinit();
        const root = pickle.root;
        if (root != .call or !root.call.func.isGlobal("tiktoken.core", "Encoding")) return invalid("not a tiktoken Encoding");
        const state = root.call.state orelse return invalid("Encoding without state");
        const pattern = (state.get("pat_str") orelse return invalid("no pat_str")).string;
        const max_digits: usize = if (std.mem.indexOf(u8, pattern, "\\p{N}{1,2}") != null)
            2
        else if (std.mem.indexOf(u8, pattern, "\\p{N}{1,3}") != null)
            3
        else {
            log.warn("unsupported tokenizer split pattern: {s}", .{pattern});
            return error.UnsupportedPattern;
        };

        const ranks = state.get("mergeable_ranks") orelse return invalid("no mergeable_ranks");
        if (ranks != .dict) return invalid("mergeable_ranks is not a dict");
        const tokens = try allocator.alloc([]const u8, ranks.dict.items.len);
        defer allocator.free(tokens);
        @memset(tokens, &.{});
        for (ranks.dict.items) |entry| {
            if (entry.key != .bytes or entry.value != .int) return invalid("bad mergeable rank");
            const rank = std.math.cast(usize, entry.value.int) orelse return invalid("negative rank");
            if (rank >= tokens.len or tokens[rank].len != 0) return invalid("ranks are not 0..n-1");
            tokens[rank] = entry.key.bytes;
        }

        const specials_value = state.get("special_tokens") orelse return invalid("no special_tokens");
        if (specials_value != .dict) return invalid("special_tokens is not a dict");
        const specials = try allocator.alloc([]const u8, specials_value.dict.items.len);
        defer allocator.free(specials);
        @memset(specials, &.{});
        for (specials_value.dict.items) |entry| {
            if (entry.key != .string or entry.value != .int) return invalid("bad special token");
            const id = std.math.cast(usize, entry.value.int) orelse return invalid("negative special id");
            if (id < tokens.len or id - tokens.len >= specials.len) return invalid("special ids must follow the ranks");
            specials[id - tokens.len] = entry.key.string;
        }
        return mod.Tokenizer.init(allocator, tokens, specials, max_digits);
    }

    /// Uploads a state dict into model weights, patching what old checkpoints
    /// lack as `checkpoint_manager.py` does (`resid_lambdas` = 1, `x0_lambdas` = 0).
    ///
    /// Parameters:
    /// - `backend`: uploads the tensors.
    /// - `state`: the imported tensors.
    /// - `weights`: shaped for the checkpoint's config.
    ///
    /// Return: nothing; `error.MissingTensor`, `error.ShapeMismatch`.
    pub fn loadWeights(backend: *mod.Backend, state: *const StateDict, weights: *mod.GptWeights) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        for (weights.params) |p| {
            if (state.get(p.name)) |t| {
                if (!t.shape.eql(p.tensor.shape)) {
                    log.err("{s}: checkpoint has {f}, model expects {f}", .{ p.name, t.shape, p.tensor.shape });
                    return error.ShapeMismatch;
                }
                try backend.upload(p.tensor, f32, t.data);
                continue;
            }
            const fill: ?f32 = if (std.mem.eql(u8, p.name, "resid_lambdas")) 1 else if (std.mem.eql(u8, p.name, "x0_lambdas")) 0 else null;
            const value = fill orelse {
                log.err("checkpoint lacks {s}", .{p.name});
                return error.MissingTensor;
            };
            log.info("patching missing {s} to {d}", .{ p.name, value });
            try backend.fill(p.tensor, value);
        }
    }

    /// What `importCheckpoint` imported.
    pub const Imported = struct { tag: []u8, step: usize };

    /// Imports a Python nanochat checkpoint and its tokenizer into this port's
    /// base directory: `model_<step>.safetensors` (f32, patched), `meta_<step>.json`
    /// (with a complete `model_config`), `tokenizer/tokenizer.tiktoken`.
    ///
    /// Parameters:
    /// - `allocator`: scratch and the result's tag.
    /// - `backend`: stages the weights.
    /// - `storage`: file access.
    /// - `from`: the Python base directory (`$NANOCHAT_BASE_DIR`).
    /// - `to`: this port's base directory.
    /// - `kind`: base, sft or rl.
    /// - `tag`: the model tag, or null for the largest.
    /// - `step`: the step, or null for the last.
    ///
    /// Return: the tag and step imported (free `tag`); storage and format errors.
    pub fn importCheckpoint(allocator: std.mem.Allocator, backend: *mod.Backend, storage: mod.Storage, from: []const u8, to: []const u8, kind: mod.CheckpointKind, tag: ?[]const u8, step: ?usize) !Imported {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const model_tag = if (tag) |t| try allocator.dupe(u8, t) else try mod.Checkpoint.largestTag(allocator, storage, from, kind);
        errdefer allocator.free(model_tag);
        var source = try mod.Checkpoint.init(allocator, storage, from, kind, model_tag);
        defer source.deinit();
        const at = step orelse try source.lastStep("pt");

        var meta = try source.loadMeta(at);
        defer meta.deinit();
        const model_config = meta.value.object.get("model_config") orelse return invalid("meta lacks model_config");
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const config = try mod.GptConfig.fromJson(arena.allocator(), model_config);

        const model_path = try source.path("model", at, "pt");
        defer allocator.free(model_path);
        var state = try loadStateDict(allocator, storage.io, model_path);
        defer state.deinit();
        var weights = try mod.GptWeights.init(allocator, backend, config);
        defer weights.deinit();
        try loadWeights(backend, &state, &weights);

        var target = try mod.Checkpoint.init(allocator, storage, to, kind, model_tag);
        defer target.deinit();
        try target.saveModel(backend, at, &weights);
        // The meta keeps every field; model_config is rewritten complete (window_pattern patched).
        const config_json = try std.json.Stringify.valueAlloc(arena.allocator(), config, .{});
        const config_value = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{});
        meta.value.object.getPtr("model_config").?.* = config_value;
        const meta_json = try std.json.Stringify.valueAlloc(arena.allocator(), meta.value, .{ .whitespace = .indent_2 });
        try target.saveMeta(at, meta_json);

        const pkl = try std.fs.path.join(allocator, &.{ from, "tokenizer", "tokenizer.pkl" });
        defer allocator.free(pkl);
        var tokenizer = try loadTokenizer(allocator, storage, pkl);
        defer tokenizer.deinit();
        if (tokenizer.vocabSize() != config.vocab_size) {
            log.err("tokenizer vocab {d} does not match the model's {d}", .{ tokenizer.vocabSize(), config.vocab_size });
            return error.TokenizerMismatch;
        }
        const tok_dir = try std.fs.path.join(allocator, &.{ to, "tokenizer" });
        defer allocator.free(tok_dir);
        try tokenizer.save(storage, tok_dir);
        return .{ .tag = model_tag, .step = at };
    }

    /// Materializes `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...)`.
    fn rebuildTensor(arena: std.mem.Allocator, zip: *mod.ZipArchive, archive: []const u8, name: []const u8, value: mod.PickleValue) !StateTensor {
        if (value != .call or !value.call.func.isGlobal("torch._utils", "_rebuild_tensor_v2")) return invalid("value is not a tensor");
        const args = value.call.args.items() orelse return invalid("tensor args");
        if (args.len < 4 or args[0] != .persid) return invalid("tensor args");
        // persistent id: ('storage', <torch XStorage>, key, location, numel)
        const pid = args[0].persid.items() orelse return invalid("storage id");
        if (pid.len < 3 or pid[1] != .global or pid[2] != .string) return invalid("storage id");
        const dtype = std.meta.stringToEnum(enum { FloatStorage, DoubleStorage, HalfStorage, BFloat16Storage }, pid[1].global.name) orelse {
            log.warn("{s}: unsupported storage {s}", .{ name, pid[1].global.name });
            return error.UnsupportedDtype;
        };
        const storage_dtype: StorageDtype = switch (dtype) {
            .FloatStorage => .f32,
            .DoubleStorage => .f64,
            .HalfStorage => .f16,
            .BFloat16Storage => .bf16,
        };
        if (args[1] != .int) return invalid("storage offset");
        const offset = std.math.cast(usize, args[1].int) orelse return invalid("storage offset");
        const sizes = args[2].items() orelse return invalid("tensor size");
        const strides = args[3].items() orelse return invalid("tensor stride");
        if (sizes.len != strides.len or sizes.len > mod.Shape.max_rank) return invalid("tensor rank");
        var dims: [mod.Shape.max_rank]usize = undefined;
        var steps: [mod.Shape.max_rank]usize = undefined;
        for (sizes, strides, 0..) |s, st, i| {
            dims[i] = std.math.cast(usize, s.int) orelse return invalid("tensor size");
            steps[i] = std.math.cast(usize, st.int) orelse return invalid("tensor stride");
        }
        const shape = try mod.Shape.init(dims[0..sizes.len]);

        const member = try std.fmt.allocPrint(arena, "{s}data/{s}", .{ archive, pid[2].string });
        const raw = try zip.read(try zip.find(member));
        defer zip.allocator.free(raw);
        const elem: usize = switch (storage_dtype) {
            .f32 => 4,
            .f64 => 8,
            .f16, .bf16 => 2,
        };
        const data = try arena.alloc(f32, shape.numel());
        var index: [mod.Shape.max_rank]usize = @splat(0);
        for (data) |*out| {
            var at = offset;
            for (0..shape.rank) |k| at += index[k] * steps[k];
            if ((at + 1) * elem > raw.len) return invalid("tensor outside its storage");
            const bytes = raw[at * elem ..][0..elem];
            out.* = switch (storage_dtype) {
                .f32 => @bitCast(std.mem.readInt(u32, bytes[0..4], .little)),
                .f64 => @floatCast(@as(f64, @bitCast(std.mem.readInt(u64, bytes[0..8], .little)))),
                .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, bytes[0..2], .little)))),
                .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, bytes[0..2], .little)) << 16),
            };
            // Advance the multi-index, last dimension fastest.
            var k = shape.rank;
            while (k > 0) {
                k -= 1;
                index[k] += 1;
                if (index[k] < dims[k]) break;
                index[k] = 0;
            }
        }
        return StateTensor{ .name = name, .shape = shape, .data = data };
    }

    fn invalid(what: []const u8) error{InvalidCheckpoint} {
        log.warn("invalid PyTorch artifact: {s}", .{what});
        return error.InvalidCheckpoint;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "torch import reads a state dict with bf16 storages and legacy prefixes" {
    const allocator = std.testing.allocator;
    const base = mod.build_options.source_root ++ "/testdata/nanochat_base/base_checkpoints/";
    var modern = try mod.TorchImport.loadStateDict(allocator, std.testing.io, base ++ "d2/model_000005.pt");
    defer modern.deinit();
    const wte = modern.get("transformer.wte.weight").?; // stored as bf16
    try std.testing.expectEqual(@as(usize, 1088 * 64), wte.data.len);
    try std.testing.expect(modern.get("resid_lambdas") != null);

    var legacy = try mod.TorchImport.loadStateDict(allocator, std.testing.io, base ++ "d1/model_000005.pt");
    defer legacy.deinit();
    try std.testing.expect(legacy.get("lm_head.weight") != null); // "_orig_mod." stripped
    try std.testing.expect(legacy.get("resid_lambdas") == null);
}

test "torch import converts python checkpoints that reproduce pytorch's logits" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const to = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const storage = mod.Storage.init(allocator, std.testing.io);
    var backend = try mod.Backend.init(allocator, std.testing.io, .{});
    defer backend.deinit();
    var expected = try mod.SafeTensors.load(allocator, std.testing.io, mod.build_options.source_root ++ "/testdata/nanochat_import.safetensors");
    defer expected.deinit();
    const from = mod.build_options.source_root ++ "/testdata/nanochat_base";

    // The default picks the largest tag (d2) and its last step (5, not 3).
    const auto = try mod.TorchImport.importCheckpoint(allocator, &backend, storage, from, to, .base, null, null);
    defer allocator.free(auto.tag);
    try std.testing.expectEqualStrings("d2", auto.tag);
    try std.testing.expectEqual(@as(usize, 5), auto.step);
    const legacy = try mod.TorchImport.importCheckpoint(allocator, &backend, storage, from, to, .base, "d1", null);
    defer allocator.free(legacy.tag);

    for ([_][]const u8{ "d2", "d1" }) |tag| {
        var ckpt = try mod.Checkpoint.init(allocator, storage, to, .base, tag);
        defer ckpt.deinit();
        try std.testing.expectEqual(@as(usize, 5), try ckpt.lastStep("safetensors"));
        var meta = try ckpt.loadMeta(5);
        defer meta.deinit();
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const config = try mod.GptConfig.fromJson(arena.allocator(), meta.value.object.get("model_config").?);
        var model = try mod.Gpt.init(allocator, &backend, config);
        defer model.deinit();
        try ckpt.loadModel(5, &model.weights);

        var acts = try mod.GptActivations.init(allocator, &backend, config, 1, 20);
        defer acts.deinit();
        const ids = try expected.readAlloc(allocator, "idx", i32);
        defer allocator.free(ids);
        const idx = try backend.alloc(.i32, &.{ 1, 20 });
        defer backend.free(idx);
        try backend.upload(idx, i32, ids);
        try model.forward(&acts, idx);
        const got = try allocator.alloc(f32, acts.logits.numel());
        defer allocator.free(got);
        try backend.download(acts.logits, f32, got);
        var name_buf: [16]u8 = undefined;
        const want = try expected.readAlloc(allocator, try std.fmt.bufPrint(&name_buf, "logits.{s}", .{tag}), f32);
        defer allocator.free(want);
        for (want, got) |w, g| try std.testing.expectApproxEqAbs(w, g, 5e-5);
    }

    const tok_dir = try std.fs.path.join(allocator, &.{ to, "tokenizer" });
    defer allocator.free(tok_dir);
    var tok = try mod.Tokenizer.load(allocator, storage, tok_dir);
    defer tok.deinit();
    try std.testing.expectEqual(@as(usize, 1033), tok.vocabSize());
}

test "torch import reads tokenizer.pkl into the same tokenizer as the fixture" {
    const allocator = std.testing.allocator;
    const storage = mod.Storage.init(allocator, std.testing.io);
    var tok = try mod.TorchImport.loadTokenizer(allocator, storage, mod.build_options.source_root ++ "/testdata/nanochat_base/tokenizer/tokenizer.pkl");
    defer tok.deinit();
    try std.testing.expectEqual(@as(usize, 1033), tok.vocabSize());
    try std.testing.expectEqual(@as(u32, 1024), try tok.bos());
    const ids = try tok.encode(allocator, "Hello world");
    defer allocator.free(ids);
    const text = try tok.decode(allocator, ids);
    defer allocator.free(text);
    try std.testing.expectEqualStrings("Hello world", text);
}
