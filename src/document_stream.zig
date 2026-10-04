const std = @import("std");
const log = std.log.scoped(.zignanogpt_document_stream);
const mod = @import("module.zig");

/// Where a document batch came from, for resuming (`dataloader_state_dict`).
pub const DataLoaderState = struct {
    pq_idx: usize = 0,
    rg_idx: usize = 0,
    epoch: usize = 1,
};

/// A batch of documents, valid until the next `DocumentStream.next`.
pub const DocumentBatch = struct {
    docs: []const []const u8,
    state: DataLoaderState,
};

/// The documents a stream reads: parquet shards, or an in-memory list
/// (treated as one file of `rows_per_group`-document row groups).
pub const DocumentSource = union(enum) {
    parquet: []const []const u8,
    text: []const []const u8,
};

/// Rows per row group when the source is an in-memory document list.
const text_rows_per_group = 1024;

/// An endless stream of document batches, as nanochat's `_document_batches`
/// on a single rank: files in order, row groups in order, each row group cut
/// into batches of `batch_size`, then the next epoch from the first file.
/// Resuming from a state skips to the row group after it.
pub const DocumentStream = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    io: std.Io,
    source: DocumentSource,
    batch_size: usize,

    pq_idx: usize,
    rg_idx: usize = 0,
    epoch: usize,
    first_pass: bool = true,
    resume_pq: usize,
    resume_rg: ?usize,

    file: ?mod.ParquetFile = null,
    column: usize = 0,
    num_row_groups: usize = 0,
    strings: mod.ParquetStrings = .{},
    /// The current row group's documents, and the next one to emit.
    rows: std.ArrayList([]const u8) = .empty,
    loaded: bool = false,
    offset: usize = 0,

    /// Starts a stream.
    ///
    /// Parameters:
    /// - `allocator`: owns the buffers.
    /// - `io`: reads parquet files.
    /// - `source`: shard paths (borrowed) or documents (borrowed).
    /// - `batch_size`: documents per batch (`tokenizer_batch_size`).
    /// - `resume_from`: a state returned by an earlier stream, or null.
    ///
    /// Return: the stream; `error.EmptyDataset` without documents.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, source: DocumentSource, batch_size: usize, resume_from: ?DataLoaderState) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const empty = switch (source) {
            inline else => |items| items.len == 0,
        };
        if (empty or batch_size == 0) return error.EmptyDataset;
        return Self{
            .allocator = allocator,
            .io = io,
            .source = source,
            .batch_size = batch_size,
            .pq_idx = if (resume_from) |r| r.pq_idx else 0,
            .epoch = if (resume_from) |r| r.epoch else 1,
            .resume_pq = if (resume_from) |r| r.pq_idx else 0,
            .resume_rg = if (resume_from) |r| r.rg_idx else null,
        };
    }

    /// Frees the buffers and closes the current file.
    ///
    /// Parameters:
    /// - `self`: the stream.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.closeFile();
        self.rows.deinit(self.allocator);
        self.strings.deinit(self.allocator);
    }

    /// The next batch (the stream never ends; it cycles epochs).
    ///
    /// Parameters:
    /// - `self`: the stream.
    ///
    /// Return: the batch; storage and parquet errors.
    pub fn next(self: *Self) !DocumentBatch {
        while (true) {
            if (self.loaded) {
                if (self.offset < self.rows.items.len) {
                    const end = @min(self.offset + self.batch_size, self.rows.items.len);
                    defer self.offset = end;
                    return .{
                        .docs = self.rows.items[self.offset..end],
                        .state = .{ .pq_idx = self.pq_idx, .rg_idx = self.rg_idx, .epoch = self.epoch },
                    };
                }
                self.loaded = false;
                self.rg_idx += 1;
            }
            if (!self.isOpen()) {
                if (self.pq_idx >= self.numFiles()) {
                    self.first_pass = false;
                    self.epoch += 1;
                    self.pq_idx = 0;
                    continue;
                }
                try self.openFile();
                self.rg_idx = 0;
                if (self.first_pass and self.resume_rg != null and self.pq_idx == self.resume_pq) {
                    self.rg_idx = self.resume_rg.? + 1; // don't repeat data after resuming
                    if (self.rg_idx >= self.num_row_groups) {
                        self.closeFile();
                        self.pq_idx += 1;
                        continue;
                    }
                    self.resume_rg = null;
                }
            }
            if (self.rg_idx >= self.num_row_groups) {
                self.closeFile();
                self.pq_idx += 1;
                continue;
            }
            try self.loadRowGroup();
        }
    }

    fn numFiles(self: *const Self) usize {
        return switch (self.source) {
            .parquet => |paths| paths.len,
            .text => 1,
        };
    }

    fn isOpen(self: *const Self) bool {
        return switch (self.source) {
            .parquet => self.file != null,
            .text => self.num_row_groups > 0,
        };
    }

    fn openFile(self: *Self) !void {
        switch (self.source) {
            .parquet => |paths| {
                var file = try mod.ParquetFile.open(self.allocator, self.io, paths[self.pq_idx]);
                errdefer file.deinit();
                self.column = try file.column("text");
                self.num_row_groups = file.row_groups.len;
                self.file = file;
            },
            .text => |docs| self.num_row_groups = (docs.len + text_rows_per_group - 1) / text_rows_per_group,
        }
    }

    fn closeFile(self: *Self) void {
        if (self.file) |*f| f.deinit();
        self.file = null;
        self.num_row_groups = 0;
    }

    fn loadRowGroup(self: *Self) !void {
        self.rows.clearRetainingCapacity();
        switch (self.source) {
            .parquet => {
                self.strings.clear();
                try self.file.?.readStrings(self.rg_idx, self.column, &self.strings);
                for (0..self.strings.len()) |i| try self.rows.append(self.allocator, self.strings.get(i));
            },
            .text => |docs| {
                const start = self.rg_idx * text_rows_per_group;
                try self.rows.appendSlice(self.allocator, docs[start..@min(start + text_rows_per_group, docs.len)]);
            },
        }
        self.loaded = true;
        self.offset = 0;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "document stream cycles epochs over an in-memory list" {
    const docs = [_][]const u8{ "a", "b", "c" };
    var stream = try mod.DocumentStream.init(std.testing.allocator, std.testing.io, .{ .text = &docs }, 2, null);
    defer stream.deinit();
    const first = try stream.next();
    try std.testing.expectEqual(@as(usize, 2), first.docs.len);
    const second = try stream.next();
    try std.testing.expectEqualStrings("c", second.docs[0]);
    const third = try stream.next();
    try std.testing.expectEqual(@as(usize, 2), third.state.epoch);
    try std.testing.expectEqualStrings("a", third.docs[0]);
}
