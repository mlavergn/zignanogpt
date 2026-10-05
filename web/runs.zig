const std = @import("std");
const log = std.log.scoped(.zignanogpt_runs);
const web = @import("module.zig");
const mod = web.nanogpt;

/// One training run with metrics: `<kind>_checkpoints/<tag>`.
pub const Run = struct {
    name: []const u8,
    size: u64,
};

/// The new part of a run's `metrics.jsonl`.
pub const Tail = struct {
    /// Where the next read continues.
    offset: usize,
    /// Whole lines only.
    text: []const u8,
};

/// The training runs whose `metrics.jsonl` the dashboard tails (base, sft
/// and rl checkpoint directories under this port's base dir).
pub const Runs = struct {
    /// Every run with a `metrics.jsonl`, base first.
    ///
    /// Parameters:
    /// - `arena`: holds the result.
    /// - `storage`: file access.
    /// - `base_dir`: this port's base directory.
    ///
    /// Return: the runs; storage errors.
    pub fn list(arena: std.mem.Allocator, storage: mod.Storage, base_dir: []const u8) ![]Run {
        var runs: std.ArrayList(Run) = .empty;
        for ([_]mod.CheckpointKind{ .base, .sft, .rl }) |kind| {
            const tags = try mod.Checkpoint.listTags(arena, storage, base_dir, kind);
            for (tags) |tag| {
                const name = try std.fmt.allocPrint(arena, "{s}/{s}", .{ kind.dirName(), tag });
                const path = try std.fs.path.join(arena, &.{ base_dir, name, "metrics.jsonl" });
                if (!try storage.exists(path)) continue;
                const bytes = try storage.read(path);
                defer storage.allocator.free(bytes);
                try runs.append(arena, .{ .name = name, .size = bytes.len });
            }
        }
        return runs.items;
    }

    /// The complete lines of a run's metrics after `offset`.
    ///
    /// Parameters:
    /// - `arena`: holds the result.
    /// - `storage`: file access.
    /// - `base_dir`: this port's base directory.
    /// - `name`: a run from `list` (anything else is rejected).
    /// - `offset`: bytes already read.
    ///
    /// Return: the new lines and the next offset; `error.UnknownRun`, storage errors.
    pub fn tail(arena: std.mem.Allocator, storage: mod.Storage, base_dir: []const u8, name: []const u8, offset: usize) !Tail {
        const runs = try list(arena, storage, base_dir);
        for (runs) |r| {
            if (std.mem.eql(u8, r.name, name)) break;
        } else {
            log.debug("metrics requested for unknown run {s}", .{name});
            return error.UnknownRun;
        }
        const path = try std.fs.path.join(arena, &.{ base_dir, name, "metrics.jsonl" });
        const bytes = try storage.read(path);
        defer storage.allocator.free(bytes);
        // A file that shrank (a new run) starts over.
        const from = if (offset <= bytes.len) offset else 0;
        const end = if (std.mem.lastIndexOfScalar(u8, bytes[from..], '\n')) |i| from + i + 1 else from;
        return .{ .offset = end, .text = try arena.dupe(u8, bytes[from..end]) };
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "runs lists metrics files and tails whole lines" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const storage = mod.Storage.init(allocator, std.testing.io);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try storage.write(try std.fs.path.join(a, &.{ root, "base_checkpoints/d6/metrics.jsonl" }), "{\"step\":0}\n{\"step\":1}\n{\"st");
    try storage.write(try std.fs.path.join(a, &.{ root, "chatsft_checkpoints/d6/metrics.jsonl" }), "{\"step\":0}\n");
    try storage.write(try std.fs.path.join(a, &.{ root, "chatrl_checkpoints/d6/meta_000001.json" }), "{}");

    const runs = try Runs.list(a, storage, root);
    try std.testing.expectEqual(@as(usize, 2), runs.len);
    try std.testing.expectEqualStrings("base_checkpoints/d6", runs[0].name);
    try std.testing.expectEqualStrings("chatsft_checkpoints/d6", runs[1].name);
    const first = try Runs.tail(a, storage, root, "base_checkpoints/d6", 0);
    try std.testing.expectEqualStrings("{\"step\":0}\n{\"step\":1}\n", first.text);
    const next = try Runs.tail(a, storage, root, "base_checkpoints/d6", first.offset);
    try std.testing.expectEqualStrings("", next.text);
    try std.testing.expectError(error.UnknownRun, Runs.tail(a, storage, root, "../../etc", 0));
}
