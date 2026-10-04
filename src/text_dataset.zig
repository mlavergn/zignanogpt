const std = @import("std");
const log = std.log.scoped(.zignanogpt_text_dataset);
const mod = @import("module.zig");

/// Documents from a local text file: paragraphs separated by blank lines
/// (`"\n\n"`), empty ones skipped. For tests, smoke runs and tokenizer training
/// before (or instead of) the ClimbMix shards.
pub const TextDataset = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    text: []u8,
    docs: [][]const u8,

    /// Reads and splits a file.
    ///
    /// Parameters:
    /// - `allocator`: owns the text and the document list.
    /// - `storage`: the file helper.
    /// - `path`: the text file (UTF-8).
    ///
    /// Return: the dataset; storage errors, `error.InvalidUtf8`, `error.EmptyDataset`.
    pub fn load(allocator: std.mem.Allocator, storage: mod.Storage, path: []const u8) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const text = try storage.read(path);
        errdefer allocator.free(text);
        return fromText(allocator, text);
    }

    /// Splits text the dataset now owns.
    ///
    /// Parameters:
    /// - `allocator`: the allocator `text` came from.
    /// - `text`: UTF-8; freed by `deinit` (or here on error).
    ///
    /// Return: the dataset; `error.InvalidUtf8`, `error.EmptyDataset`.
    pub fn fromText(allocator: std.mem.Allocator, text: []u8) !Self {
        errdefer allocator.free(text);
        if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidUtf8;
        var docs: std.ArrayList([]const u8) = .empty;
        errdefer docs.deinit(allocator);
        var it = std.mem.splitSequence(u8, text, "\n\n");
        while (it.next()) |doc| {
            if (doc.len > 0) try docs.append(allocator, doc);
        }
        if (docs.items.len == 0) {
            log.warn("no documents in the text", .{});
            return error.EmptyDataset;
        }
        return Self{ .allocator = allocator, .text = text, .docs = try docs.toOwnedSlice(allocator) };
    }

    /// Frees the text and the list.
    ///
    /// Parameters:
    /// - `self`: the dataset.
    ///
    /// Return: nothing.
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.docs);
        self.allocator.free(self.text);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "text dataset splits paragraphs and skips empty ones" {
    const allocator = std.testing.allocator;
    var ds = try mod.TextDataset.fromText(allocator, try allocator.dupe(u8, "one\ntwo\n\n\n\nthree\n\nfour"));
    defer ds.deinit();
    try std.testing.expectEqual(@as(usize, 3), ds.docs.len);
    try std.testing.expectEqualStrings("one\ntwo", ds.docs[0]);
    try std.testing.expectEqualStrings("four", ds.docs[2]);
    try std.testing.expectError(error.EmptyDataset, mod.TextDataset.fromText(allocator, try allocator.dupe(u8, "\n\n")));
}
