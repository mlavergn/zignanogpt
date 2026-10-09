const std = @import("std");
const log = std.log.scoped(.zignanogpt_checkpoint);
const mod = @import("../module.zig");

/// Which training stage a checkpoint belongs to (nanochat's `load_model` sources).
pub const CheckpointKind = enum {
    base,
    sft,
    rl,

    pub fn dirName(self: CheckpointKind) []const u8 {
        return switch (self) {
            .base => "base_checkpoints",
            .sft => "chatsft_checkpoints",
            .rl => "chatrl_checkpoints",
        };
    }
};

/// One model's checkpoint directory, nanochat's layout:
/// `<root>/<kind>_checkpoints/<tag>/{model,optim,meta}_<step:06>.<ext>`.
/// This port writes `model_*.safetensors` (f32, PyTorch names) and nanochat's
/// `meta_*.json` unchanged in shape; Python's are `model_*.pt`.
pub const Checkpoint = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    storage: mod.Storage,
    /// `<root>/<kind>_checkpoints/<tag>`.
    dir: []const u8,

    /// Addresses a checkpoint directory (nothing is read or created).
    ///
    /// Parameters:
    /// - `allocator`: owns the path.
    /// - `storage`: file access.
    /// - `root`: a base directory.
    /// - `kind`: base, sft or rl.
    /// - `tag`: the model tag, e.g. `d12`.
    ///
    /// Return: the checkpoint; allocation errors.
    pub fn init(allocator: std.mem.Allocator, storage: mod.Storage, root: []const u8, kind: CheckpointKind, tag: []const u8) !Self {
        return Self{ .allocator = allocator, .storage = storage, .dir = try std.Io.Dir.path.join(allocator, &.{ root, kind.dirName(), tag }) };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.dir);
    }

    /// `<dir>/<prefix>_<step:06>.<ext>`, owned by the caller.
    pub fn path(self: *const Self, prefix: []const u8, step: usize, ext: []const u8) ![]u8 {
        return self.allocator.print("{s}/{s}_{d:0>6}.{s}", .{ self.dir, prefix, step, ext });
    }

    /// Writes `model_<step>.safetensors`.
    ///
    /// Parameters:
    /// - `self`: the checkpoint.
    /// - `backend`: downloads the weights.
    /// - `step`: the step number.
    /// - `weights`: the parameters.
    ///
    /// Return: nothing; backend and storage errors.
    pub fn saveModel(self: *const Self, backend: *mod.Backend, step: usize, weights: *const mod.GptWeights) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var writer = mod.SafeTensorsWriter.init(self.allocator);
        defer writer.deinit();
        for (weights.params) |p| try writer.addTensor(p.name, p.tensor);
        const file = try self.path("model", step, "safetensors");
        defer self.allocator.free(file);
        try writer.write(backend, self.storage, file);
    }

    /// Loads `model_<step>.safetensors` into `weights`.
    ///
    /// Parameters:
    /// - `self`: the checkpoint.
    /// - `step`: the step number.
    /// - `weights`: shaped for the checkpoint's config.
    ///
    /// Return: nothing; storage and shape errors.
    pub fn loadModel(self: *const Self, step: usize, weights: *mod.GptWeights) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const file = try self.path("model", step, "safetensors");
        defer self.allocator.free(file);
        var tensors = try mod.SafeTensors.load(self.allocator, self.storage.io, file);
        defer tensors.deinit();
        try weights.load(self.allocator, &tensors, "");
    }

    /// Writes `optim_<step>.safetensors` (nanochat: `optim_<step>_rank0.pt`).
    ///
    /// Parameters:
    /// - `self`: the checkpoint.
    /// - `backend`: downloads the state.
    /// - `step`: the step number.
    /// - `optimizer`: the optimizer.
    /// - `weights`: names the parameters.
    ///
    /// Return: nothing; backend and storage errors.
    pub fn saveOptimizer(self: *const Self, backend: *mod.Backend, step: usize, optimizer: *const mod.MuonAdamW, weights: *const mod.GptWeights) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var writer = mod.SafeTensorsWriter.init(self.allocator);
        defer writer.deinit();
        try optimizer.addState(&writer, weights);
        const file = try self.path("optim", step, "safetensors");
        defer self.allocator.free(file);
        try writer.write(backend, self.storage, file);
    }

    /// Restores `optim_<step>.safetensors` into `optimizer`.
    pub fn loadOptimizer(self: *const Self, step: usize, optimizer: *mod.MuonAdamW, weights: *const mod.GptWeights) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const file = try self.path("optim", step, "safetensors");
        defer self.allocator.free(file);
        var tensors = try mod.SafeTensors.load(self.allocator, self.storage.io, file);
        defer tensors.deinit();
        try optimizer.loadState(&tensors, weights);
    }

    /// Writes `meta_<step>.json`.
    pub fn saveMeta(self: *const Self, step: usize, json: []const u8) !void {
        const file = try self.path("meta", step, "json");
        defer self.allocator.free(file);
        try self.storage.write(file, json);
    }

    /// Reads and parses `meta_<step>.json`.
    ///
    /// Parameters:
    /// - `self`: the checkpoint.
    /// - `step`: the step number.
    ///
    /// Return: the parsed JSON, owned by the caller.
    pub fn loadMeta(self: *const Self, step: usize) !std.json.Parsed(std.json.Value) {
        const file = try self.path("meta", step, "json");
        defer self.allocator.free(file);
        const bytes = try self.storage.read(file);
        defer self.allocator.free(bytes);
        return std.json.parseFromSlice(std.json.Value, self.allocator, bytes, .{ .allocate = .alloc_always });
    }

    /// The highest step with a `model_<step>.<ext>` (nanochat's `find_last_step`).
    ///
    /// Parameters:
    /// - `self`: the checkpoint.
    /// - `ext`: `safetensors` for this port, `pt` for Python's.
    ///
    /// Return: the step; `error.NoCheckpoint` when there is none.
    pub fn lastStep(self: *const Self, ext: []const u8) !usize {
        var best: ?usize = null;
        var it = try listDir(self.allocator, self.storage, self.dir);
        defer it.close();
        while (try it.next()) |url| {
            const name = basename(url);
            if (!std.mem.startsWith(u8, name, "model_") or !std.mem.endsWith(u8, name, ext) or name.len <= "model_".len + ext.len + 1) continue;
            const digits = name["model_".len .. name.len - ext.len - 1];
            const step = std.fmt.parseInt(usize, digits, 10) catch continue;
            best = @max(best orelse 0, step);
        }
        return best orelse {
            log.warn("no model_*.{s} in {s}", .{ ext, self.dir });
            return error.NoCheckpoint;
        };
    }

    /// The largest `d<N>` model tag under `<root>/<kind>_checkpoints` (nanochat's
    /// `find_largest_model`; without `d<N>` tags, the last name in order).
    ///
    /// Parameters:
    /// - `allocator`: owns the result.
    /// - `storage`: file access.
    /// - `root`: a base directory.
    /// - `kind`: base, sft or rl.
    ///
    /// Return: the tag; `error.NoCheckpoint`.
    pub fn largestTag(allocator: std.mem.Allocator, storage: mod.Storage, root: []const u8, kind: CheckpointKind) ![]u8 {
        const dir = try std.Io.Dir.path.join(allocator, &.{ root, kind.dirName() });
        defer allocator.free(dir);
        var it = listDir(allocator, storage, dir) catch |err| switch (err) {
            error.NotFound => {
                log.warn("no checkpoints in {s}", .{dir});
                return error.NoCheckpoint;
            },
            else => return err,
        };
        defer it.close();
        var best_depth: ?usize = null;
        var best: ?[]u8 = null;
        errdefer if (best) |b| allocator.free(b);
        var last_name: ?[]u8 = null;
        defer if (last_name) |n| allocator.free(n);
        while (try it.next()) |url| {
            if (!std.mem.endsWith(u8, url, "/")) continue; // directories only
            const name = basename(url[0 .. url.len - 1]);
            if (last_name == null or std.mem.lessThan(u8, last_name.?, name)) {
                if (last_name) |n| allocator.free(n);
                last_name = try allocator.dupe(u8, name);
            }
            if (name.len < 2 or name[0] != 'd') continue;
            var end: usize = 1;
            while (end < name.len and std.ascii.isDigit(name[end])) end += 1;
            if (end == 1) continue;
            const depth = std.fmt.parseInt(usize, name[1..end], 10) catch continue;
            if (best_depth == null or depth > best_depth.?) {
                if (best) |b| allocator.free(b);
                best = try allocator.dupe(u8, name);
                best_depth = depth;
            }
        }
        if (best) |b| return b;
        if (last_name) |n| {
            last_name = null;
            return n;
        }
        return error.NoCheckpoint;
    }

    /// Every model tag under `<root>/<kind>_checkpoints`, sorted.
    ///
    /// Parameters:
    /// - `allocator`: owns the result (each tag and the list).
    /// - `storage`: file access.
    /// - `root`: a base directory.
    /// - `kind`: base, sft or rl.
    ///
    /// Return: the tags (empty when there is no checkpoint directory).
    pub fn listTags(allocator: std.mem.Allocator, storage: mod.Storage, root: []const u8, kind: CheckpointKind) ![][]u8 {
        const dir = try std.Io.Dir.path.join(allocator, &.{ root, kind.dirName() });
        defer allocator.free(dir);
        var tags: std.ArrayList([]u8) = .empty;
        errdefer {
            for (tags.items) |t| allocator.free(t);
            tags.deinit(allocator);
        }
        var it = listDir(allocator, storage, dir) catch |err| switch (err) {
            error.NotFound => return tags.toOwnedSlice(allocator),
            else => return err,
        };
        defer it.close();
        while (try it.next()) |url| {
            if (!std.mem.endsWith(u8, url, "/")) continue;
            try tags.append(allocator, try allocator.dupe(u8, basename(url[0 .. url.len - 1])));
        }
        std.mem.sort([]u8, tags.items, {}, struct {
            fn lessThan(_: void, a: []u8, b: []u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lessThan);
        return tags.toOwnedSlice(allocator);
    }

    fn listDir(allocator: std.mem.Allocator, storage: mod.Storage, dir: []const u8) !mod.zigstorage.NodeIterator {
        var node = try mod.zigstorage.Node.init(allocator, storage.io, .empty, dir);
        defer node.deinit();
        return node.list();
    }

    fn basename(url: []const u8) []const u8 {
        return url[(std.mem.findScalarLast(u8, url, '/') orelse return url) + 1 ..];
    }
};
