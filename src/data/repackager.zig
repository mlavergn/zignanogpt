const std = @import("std");
const log = std.log.scoped(.zignanogpt_repackager);
const mod = @import("../module.zig");

/// How `Repackager.run` shuffles and cuts a corpus.
pub const RepackageOptions = struct {
    /// The JSONL field or Parquet column holding each document's text.
    field: []const u8 = "text",
    /// Seeds the shuffle (bucket choice, then each bucket's permutation).
    seed: u64 = 42,
    /// A train shard closes at this many characters (code points, as Python's
    /// `len`) once its documents fill whole row groups.
    chars_per_shard: u64 = 250_000_000,
    /// Documents per Parquet row group.
    row_group: usize = 1024,
    /// Document bytes per shuffle bucket: a corpus estimated larger is scattered
    /// into bucket files on disk, one bucket in memory at a time.
    bucket_bytes: u64 = 512 << 20,
    /// Replace the shards already in the output directory.
    overwrite: bool = false,
    /// Progress (reading, then writing), or null.
    observer: ?mod.TrainObserver = null,
};

/// What a run read and wrote.
pub const RepackageSummary = struct {
    files: usize = 0,
    documents: usize = 0,
    characters: u64 = 0,
    /// Documents dropped: invalid UTF-8, or JSONL lines without a string field.
    skipped: usize = 0,
    shards: usize = 0,
    /// Bytes of shards written.
    bytes: u64 = 0,
};

const InputKind = enum { text, jsonl, parquet };

const Input = struct {
    path: []const u8,
    kind: InputKind,
    size: u64,
};

/// One on-disk shuffle bucket: documents appended through a buffer.
const Bucket = struct {
    path: []const u8,
    session: mod.StorageSession,
    buffer: std.ArrayList(u8) = .empty,
};

/// nanochat's `dev/repackage_data_reference.py` for any corpus: documents from
/// text files (separated by blank lines), JSONL and Parquet are shuffled and
/// written as `shard_NNNNN.parquet` files with one `text` column, the layout
/// `Dataset` reads; the last shard is the validation split.
///
/// The shuffle is a uniform permutation in two passes: reading scatters each
/// document into a random bucket (one in memory for a small corpus, files under
/// `<out>/.repackage` otherwise), then each bucket is shuffled in memory and
/// written in turn. The last min(one shard, 10% of the characters) go to the
/// validation shard, which always gets at least one document.
pub const Repackager = struct {
    const Self = @This();

    const read_chunk = 16 << 20;
    const bucket_flush = 1 << 20;
    const bucket_dir = ".repackage";
    /// Bytes before each record: document length, then its characters.
    const record_header = 8;
    /// Parquet text compresses about 3x: its share of the bucket estimate.
    const parquet_expansion = 3;

    allocator: std.mem.Allocator,
    io: std.Io,
    storage: mod.Storage,
    options: RepackageOptions,
    out_dir: []const u8,
    out: *std.Io.Writer,
    prng: std.Random.DefaultPrng,
    summary: RepackageSummary = .{},
    inputs: std.ArrayList(Input) = .empty,
    /// The single in-memory bucket (records), when there are no bucket files.
    memory: std.ArrayList(u8) = .empty,
    buckets: std.ArrayList(Bucket) = .empty,
    bytes_read: u64 = 0,
    bytes_total: u64 = 0,
    warned: usize = 0,

    /// Repackages `inputs` into shards in `out_dir`.
    ///
    /// Parameters:
    /// - `allocator`: buffers and the in-memory bucket.
    /// - `io`: storage.
    /// - `inputs`: files and directories (searched recursively for `.txt`,
    ///   `.md`, `.text`, `.jsonl`, `.ndjson` and `.parquet`; hidden entries skipped).
    ///   A file named directly is read as Parquet or JSONL by extension, else as text.
    /// - `out_dir`: the dataset directory (`<base>/base_data_<name>`).
    /// - `options`: shuffle, shard sizes, field.
    /// - `out`: a line per shard and the totals.
    ///
    /// Return: the totals; `error.DatasetExists` (shards present, no overwrite),
    /// `error.NoInputs`, `error.TooFewDocuments`, `error.MissingColumn`, storage errors.
    pub fn run(allocator: std.mem.Allocator, io: std.Io, inputs: []const []const u8, out_dir: []const u8, options: RepackageOptions, out: *std.Io.Writer) !RepackageSummary {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        std.debug.assert(options.row_group > 0 and options.chars_per_shard > 0 and options.bucket_bytes > 0);
        var self: Self = .{
            .allocator = allocator,
            .io = io,
            .storage = mod.Storage.init(allocator, io),
            .options = options,
            .out_dir = std.mem.trimEnd(u8, out_dir, "/"),
            .out = out,
            .prng = std.Random.DefaultPrng.init(options.seed),
        };
        defer self.deinit();
        for (inputs) |raw| {
            const path = try self.storage.absolute(raw);
            defer allocator.free(path);
            try self.collect(path, true);
        }
        if (self.inputs.items.len == 0) {
            log.warn("no input files (looked for .txt, .md, .text, .jsonl, .ndjson, .parquet)", .{});
            return error.NoInputs;
        }
        try self.prepareOutput();

        // Pass 1: read every document into a random bucket.
        var estimate: u64 = 0;
        for (self.inputs.items) |in| {
            self.bytes_total += in.size;
            estimate += if (in.kind == .parquet) in.size * parquet_expansion else in.size;
        }
        const bucket_count: usize = @intCast(@max(1, (estimate + options.bucket_bytes - 1) / options.bucket_bytes));
        if (bucket_count > 1) try self.openBuckets(bucket_count);
        try out.print("Reading {d} files ({d} MiB) into {d} shuffle bucket{s}\n", .{ self.inputs.items.len, self.bytes_total >> 20, bucket_count, if (bucket_count == 1) "" else "s" });
        try out.flush();
        for (self.inputs.items) |in| {
            switch (in.kind) {
                .text => try self.readDelimited(in, "\n\n", addText),
                .jsonl => try self.readDelimited(in, "\n", addJsonLine),
                .parquet => try self.readParquet(in),
            }
            self.summary.files += 1;
        }
        for (self.buckets.items) |*b| {
            try b.session.append(b.buffer.items);
            try b.session.save();
        }
        try out.print("{d} documents, {d} characters", .{ self.summary.documents, self.summary.characters });
        if (self.summary.skipped > 0) try out.print(" ({d} skipped)", .{self.summary.skipped});
        try out.writeByte('\n');
        try out.flush();
        if (self.summary.documents < 2) {
            log.warn("need at least 2 documents (one for training, one for validation); found {d}", .{self.summary.documents});
            return error.TooFewDocuments;
        }

        // Pass 2: shuffle each bucket and write the shards.
        const val_chars = @min(options.chars_per_shard, self.summary.characters / 10);
        var sink: ShardSink = .{ .owner = &self, .train_chars = self.summary.characters - val_chars, .remaining = self.summary.documents };
        defer sink.deinit();
        if (self.buckets.items.len == 0) {
            try self.emit(self.memory.items, &sink);
        } else {
            for (self.buckets.items) |b| {
                const bytes = try self.storage.read(b.path);
                defer allocator.free(bytes);
                try self.emit(bytes, &sink);
            }
        }
        try sink.close();
        mod.TrainObserver.progress(options.observer, "writing shards", self.summary.documents, self.summary.documents);
        if (self.buckets.items.len > 0) try self.removeBuckets();
        return self.summary;
    }

    fn deinit(self: *Self) void {
        for (self.buckets.items) |*b| {
            b.session.deinit();
            b.buffer.deinit(self.allocator);
            self.allocator.free(b.path);
        }
        self.buckets.deinit(self.allocator);
        self.memory.deinit(self.allocator);
        for (self.inputs.items) |in| self.allocator.free(in.path);
        self.inputs.deinit(self.allocator);
    }

    // -------------------------------------------------------------------------
    // Inputs

    /// Adds a file, or a directory's files recursively.
    fn collect(self: *Self, path: []const u8, named: bool) !void {
        const entries = self.storage.list(self.allocator, path) catch |err| switch (err) {
            error.NotADirectory => {
                const name = std.Io.Dir.path.basename(path);
                const kind = kindOf(name) orelse if (named) InputKind.text else return;
                if (std.mem.startsWith(u8, path, self.out_dir) and path.len > self.out_dir.len and path[self.out_dir.len] == '/') {
                    log.warn("{s} is inside the output directory", .{path});
                    return error.InputInOutput;
                }
                const size = try self.storage.size(path);
                const owned = try self.allocator.dupe(u8, path);
                errdefer self.allocator.free(owned);
                try self.inputs.append(self.allocator, .{ .path = owned, .kind = kind, .size = size });
                return;
            },
            error.NotFound => {
                log.warn("no such input: {s}", .{path});
                return err;
            },
            else => return err,
        };
        defer {
            for (entries) |e| self.allocator.free(e);
            self.allocator.free(entries);
        }
        if (std.mem.eql(u8, std.mem.trimEnd(u8, path, "/"), self.out_dir)) {
            log.warn("{s} is the output directory", .{path});
            return error.InputInOutput;
        }
        for (entries) |entry| {
            const name = std.Io.Dir.path.basename(std.mem.trimEnd(u8, entry, "/"));
            if (name.len == 0 or name[0] == '.') continue;
            try self.collect(std.mem.trimEnd(u8, entry, "/"), false);
        }
    }

    fn kindOf(name: []const u8) ?InputKind {
        const ext = std.Io.Dir.path.extension(name);
        if (std.ascii.eqlIgnoreCase(ext, ".parquet")) return .parquet;
        if (std.ascii.eqlIgnoreCase(ext, ".jsonl") or std.ascii.eqlIgnoreCase(ext, ".ndjson")) return .jsonl;
        for ([_][]const u8{ ".txt", ".md", ".text" }) |t| if (std.ascii.eqlIgnoreCase(ext, t)) return .text;
        return null;
    }

    /// Refuses (or, with overwrite, clears) existing shards; drops stale buckets.
    fn prepareOutput(self: *Self) !void {
        try self.storage.makeDir(self.out_dir);
        const entries = try self.storage.list(self.allocator, self.out_dir);
        defer {
            for (entries) |e| self.allocator.free(e);
            self.allocator.free(entries);
        }
        var shards: usize = 0;
        for (entries) |e| shards += @intFromBool(isShard(e));
        if (shards > 0 and !self.options.overwrite) {
            log.warn("{s} already has {d} shards; overwrite to replace them", .{ self.out_dir, shards });
            return error.DatasetExists;
        }
        for (entries) |e| {
            if (isShard(e)) try self.storage.remove(e);
            if (std.mem.endsWith(u8, e, "/" ++ bucket_dir ++ "/")) try self.storage.removeTree(e);
        }
    }

    fn isShard(path: []const u8) bool {
        const name = std.Io.Dir.path.basename(path);
        return std.mem.startsWith(u8, name, "shard_") and std.mem.endsWith(u8, name, ".parquet");
    }

    /// Reads a file in chunks, handing each delimited piece to `handle` (the
    /// tail after the last delimiter waits for the next chunk), as a whole-file
    /// `splitSequence` would cut it.
    fn readDelimited(self: *Self, in: Input, delimiter: []const u8, handle: *const fn (*Self, Input, []const u8, usize) anyerror!void) !void {
        var carry: std.ArrayList(u8) = .empty;
        defer carry.deinit(self.allocator);
        var offset: u64 = 0;
        var piece: usize = 0;
        while (true) {
            const chunk = try self.storage.readRange(in.path, offset, read_chunk);
            defer self.allocator.free(chunk);
            if (chunk.len == 0) break;
            offset += chunk.len;
            self.bytes_read += chunk.len;
            try carry.appendSlice(self.allocator, chunk);
            var it = std.mem.splitSequence(u8, carry.items, delimiter);
            var consumed: usize = 0;
            while (it.next()) |part| {
                const next = it.index orelse break;
                piece += 1;
                try handle(self, in, part, piece);
                consumed = next;
            }
            const rest = carry.items.len - consumed;
            std.mem.copyForwards(u8, carry.items[0..rest], carry.items[consumed..]);
            carry.shrinkRetainingCapacity(rest);
            self.reportRead();
        }
        if (carry.items.len > 0) try handle(self, in, carry.items, piece + 1);
    }

    fn addText(self: *Self, _: Input, doc: []const u8, _: usize) !void {
        try self.add(doc);
    }

    /// One JSONL line: its `field`, when that is a string.
    fn addJsonLine(self: *Self, in: Input, raw: []const u8, line: usize) !void {
        const text = std.mem.trim(u8, raw, " \t\r");
        if (text.len == 0) return;
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const value = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{}) catch |err| {
            return self.skip("{s}:{d}: not JSON [{t}]", .{ in.path, line, err });
        };
        const field = switch (value) {
            .object => |o| o.get(self.options.field),
            else => null,
        };
        if (field) |f| if (f == .string) return self.add(f.string);
        return self.skip("{s}:{d}: no string field '{s}'", .{ in.path, line, self.options.field });
    }

    fn readParquet(self: *Self, in: Input) !void {
        var file = try mod.ParquetFile.open(self.allocator, self.io, in.path);
        defer file.deinit();
        const col = file.column(self.options.field) catch |err| {
            log.warn("{s}: no column '{s}' [{t}]", .{ in.path, self.options.field, err });
            return error.MissingColumn;
        };
        var strings: mod.ParquetStrings = .{};
        defer strings.deinit(self.allocator);
        const groups = file.row_groups.len;
        const start = self.bytes_read;
        for (0..groups) |rg| {
            strings.clear();
            try file.readStrings(rg, col, &strings);
            for (0..strings.len()) |i| try self.add(strings.get(i));
            self.bytes_read = start + in.size * (rg + 1) / @max(groups, 1);
            self.reportRead();
        }
    }

    /// Counts a document and puts it in a random bucket (empty ones are dropped).
    fn add(self: *Self, doc: []const u8) !void {
        if (doc.len == 0) return;
        const chars = std.unicode.utf8CountCodepoints(doc) catch return self.skip("a document is not UTF-8", .{});
        if (doc.len > std.math.maxInt(u32)) return self.skip("a document is over 4 GiB", .{});
        self.summary.documents += 1;
        self.summary.characters += chars;
        const target = if (self.buckets.items.len == 0)
            &self.memory
        else
            &self.buckets.items[self.prng.random().uintLessThan(usize, self.buckets.items.len)].buffer;
        var header: [record_header]u8 = undefined;
        std.mem.writeInt(u32, header[0..4], @intCast(doc.len), .little);
        std.mem.writeInt(u32, header[4..8], @intCast(chars), .little);
        try target.appendSlice(self.allocator, &header);
        try target.appendSlice(self.allocator, doc);
        if (self.buckets.items.len > 0 and target.items.len >= bucket_flush) {
            const b: *Bucket = @alignCast(@fieldParentPtr("buffer", target));
            try b.session.append(target.items);
            target.clearRetainingCapacity();
        }
    }

    /// Drops a document, warning for the first few.
    fn skip(self: *Self, comptime fmt: []const u8, args: anytype) void {
        self.summary.skipped += 1;
        if (self.warned < 5) log.warn("skipped: " ++ fmt, args);
        self.warned += 1;
    }

    fn reportRead(self: *Self) void {
        mod.TrainObserver.progress(self.options.observer, "reading MiB", @intCast(self.bytes_read >> 20), @intCast(@max(self.bytes_total >> 20, 1)));
    }

    fn openBuckets(self: *Self, count: usize) !void {
        for (0..count) |i| {
            var name: [32]u8 = undefined;
            const path = try self.allocator.print("{s}/{s}/{s}", .{ self.out_dir, bucket_dir, try std.mem.print(&name, "bucket_{d:0>5}.bin", .{i}) });
            errdefer self.allocator.free(path);
            var session = try self.storage.create(path);
            errdefer session.deinit();
            try self.buckets.append(self.allocator, .{ .path = path, .session = session });
        }
    }

    fn removeBuckets(self: *Self) !void {
        const dir = try self.allocator.print("{s}/{s}", .{ self.out_dir, bucket_dir });
        defer self.allocator.free(dir);
        try self.storage.removeTree(dir);
    }

    // -------------------------------------------------------------------------
    // Output

    /// Shuffles one bucket's records and writes them in that order.
    fn emit(self: *Self, records: []const u8, sink: *ShardSink) !void {
        var starts: std.ArrayList(usize) = .empty;
        defer starts.deinit(self.allocator);
        var pos: usize = 0;
        while (pos < records.len) {
            if (pos + record_header > records.len) return error.CorruptBucket;
            try starts.append(self.allocator, pos);
            pos += record_header + std.mem.readInt(u32, records[pos..][0..4], .little);
        }
        if (pos != records.len) return error.CorruptBucket;
        self.prng.random().shuffle(usize, starts.items);
        for (starts.items) |s| {
            const len = std.mem.readInt(u32, records[s..][0..4], .little);
            const chars = std.mem.readInt(u32, records[s + 4 ..][0..4], .little);
            try sink.add(records[s + record_header ..][0..len], chars);
        }
    }

    /// Cuts the shuffled stream into train shards, then the validation shard.
    const ShardSink = struct {
        owner: *Self,
        train_chars: u64,
        /// Documents not yet written.
        remaining: usize,
        emitted: u64 = 0,
        validation: bool = false,
        index: usize = 0,
        writer: ?mod.ParquetWriter = null,
        shard_chars: u64 = 0,
        shard_docs: usize = 0,
        written: usize = 0,

        fn add(self: *ShardSink, doc: []const u8, chars: u64) !void {
            const o = self.owner;
            if (!self.validation and (self.emitted >= self.train_chars or self.remaining == 1)) {
                try self.close();
                self.validation = true;
            }
            if (self.writer == null) {
                var name: [32]u8 = undefined;
                const path = try o.allocator.print("{s}/{s}", .{ o.out_dir, try mod.Dataset.shardName(&name, self.index) });
                defer o.allocator.free(path);
                self.writer = try mod.ParquetWriter.create(o.allocator, o.storage, path, "text", o.options.row_group);
            }
            try self.writer.?.append(doc);
            self.shard_chars += chars;
            self.shard_docs += 1;
            self.emitted += chars;
            self.remaining -= 1;
            self.written += 1;
            if (self.written % 4096 == 0) mod.TrainObserver.progress(o.options.observer, "writing shards", self.written, o.summary.documents);
            if (!self.validation and self.shard_chars >= o.options.chars_per_shard and self.shard_docs % o.options.row_group == 0) try self.close();
        }

        /// Commits the open shard, if any.
        fn close(self: *ShardSink) !void {
            const o = self.owner;
            var writer = &(self.writer orelse return);
            const bytes = try writer.finish();
            writer.deinit();
            self.writer = null;
            var name: [32]u8 = undefined;
            try o.out.print("{s} {s}: {d} documents, {d} characters, {d} MiB\n", .{ try mod.Dataset.shardName(&name, self.index), if (self.validation) "(val)" else "", self.shard_docs, self.shard_chars, bytes >> 20 });
            try o.out.flush();
            o.summary.shards += 1;
            o.summary.bytes += bytes;
            self.index += 1;
            self.shard_chars = 0;
            self.shard_docs = 0;
        }

        fn deinit(self: *ShardSink) void {
            if (self.writer) |*w| w.deinit();
        }
    };
};

// -----------------------------------------------------------------------------
// Unit Tests

/// Every document in a dataset directory's shards, in order, and the shard count.
fn readShards(allocator: std.mem.Allocator, dir: []const u8, docs: *std.ArrayList([]u8)) !usize {
    const storage = mod.Storage.init(allocator, std.testing.io);
    const entries = try storage.list(allocator, dir);
    defer {
        for (entries) |e| allocator.free(e);
        allocator.free(entries);
    }
    var shards: usize = 0;
    var strings: mod.ParquetStrings = .{};
    defer strings.deinit(allocator);
    for (entries) |path| {
        if (!Repackager.isShard(path)) continue;
        shards += 1;
        var file = try mod.ParquetFile.open(allocator, std.testing.io, path);
        defer file.deinit();
        const col = try file.column("text");
        for (0..file.row_groups.len) |rg| {
            strings.clear();
            try file.readStrings(rg, col, &strings);
            for (0..strings.len()) |i| try docs.append(allocator, try allocator.dupe(u8, strings.get(i)));
        }
    }
    return shards;
}

fn freeDocs(allocator: std.mem.Allocator, docs: *std.ArrayList([]u8)) void {
    for (docs.items) |d| allocator.free(d);
    docs.deinit(allocator);
}

test "repackage shuffles text, JSONL and Parquet into shards with a validation split" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const storage = mod.Storage.init(allocator, std.testing.io);
    const in_dir = try std.Io.Dir.path.join(allocator, &.{ root, "corpus" });
    defer allocator.free(in_dir);

    // 60 text docs (one with a stray third newline), 40 JSONL rows (+2 bad lines), 50 Parquet rows.
    var expected: std.ArrayList([]u8) = .empty;
    defer freeDocs(allocator, &expected);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    for (0..60) |i| {
        const doc = try allocator.print("text document {d}\nsecond line, ünïcode", .{i});
        try expected.append(allocator, doc);
        try text.appendSlice(allocator, doc);
        try text.appendSlice(allocator, "\n\n");
    }
    const text_path = try std.Io.Dir.path.join(allocator, &.{ in_dir, "a.txt" });
    defer allocator.free(text_path);
    try storage.write(text_path, text.items);
    var jsonl: std.ArrayList(u8) = .empty;
    defer jsonl.deinit(allocator);
    for (0..40) |i| {
        const doc = try allocator.print("json \"document\" {d}", .{i});
        try expected.append(allocator, doc);
        try jsonl.print(allocator, "{{\"id\": {d}, \"text\": {f}}}\n", .{ i, std.json.fmt(doc, .{}) });
    }
    try jsonl.appendSlice(allocator, "not json\n{\"id\": 1}\n\n");
    const jsonl_path = try std.Io.Dir.path.join(allocator, &.{ in_dir, "sub", "b.jsonl" });
    defer allocator.free(jsonl_path);
    try storage.write(jsonl_path, jsonl.items);
    const parquet_path = try std.Io.Dir.path.join(allocator, &.{ in_dir, "c.parquet" });
    defer allocator.free(parquet_path);
    {
        var w = try mod.ParquetWriter.create(allocator, storage, parquet_path, "text", 16);
        defer w.deinit();
        for (0..50) |i| {
            const doc = try allocator.print("parquet document {d}", .{i});
            try expected.append(allocator, doc);
            try w.append(doc);
        }
        _ = try w.finish();
    }
    const ignored = try std.Io.Dir.path.join(allocator, &.{ in_dir, "notes.bin" });
    defer allocator.free(ignored);
    try storage.write(ignored, "not a corpus file");

    var discard: std.Io.Writer.Discarding = .init(&.{});
    const out_dir = try std.Io.Dir.path.join(allocator, &.{ root, "base_data_mine" });
    defer allocator.free(out_dir);
    // Tiny buckets (several on disk) and shards of ~600 characters in row groups of 4.
    const options: RepackageOptions = .{ .chars_per_shard = 600, .row_group = 4, .bucket_bytes = 1500 };
    const summary = try Repackager.run(allocator, std.testing.io, &.{in_dir}, out_dir, options, &discard.writer);
    try std.testing.expectEqual(@as(usize, 3), summary.files);
    try std.testing.expectEqual(@as(usize, 150), summary.documents);
    try std.testing.expectEqual(@as(usize, 2), summary.skipped);

    var got: std.ArrayList([]u8) = .empty;
    defer freeDocs(allocator, &got);
    const shards = try readShards(allocator, out_dir, &got);
    try std.testing.expectEqual(summary.shards, shards);
    try std.testing.expect(shards >= 3);
    // The same documents, in a different order.
    try std.testing.expectEqual(expected.items.len, got.items.len);
    var same_order = true;
    for (expected.items, got.items) |e, g| same_order = same_order and std.mem.eql(u8, e, g);
    try std.testing.expect(!same_order);
    const lessThan = struct {
        fn f(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.f;
    std.mem.sort([]u8, expected.items, {}, lessThan);
    const sorted = try allocator.dupe([]u8, got.items);
    defer allocator.free(sorted);
    std.mem.sort([]u8, sorted, {}, lessThan);
    for (expected.items, sorted) |e, g| try std.testing.expectEqualStrings(e, g);

    // The bucket files are gone; a second run refuses, then overwrites the same way (same seed).
    const buckets = try std.Io.Dir.path.join(allocator, &.{ out_dir, Repackager.bucket_dir });
    defer allocator.free(buckets);
    try std.testing.expectError(error.NotFound, storage.list(allocator, buckets));
    try std.testing.expectError(error.DatasetExists, Repackager.run(allocator, std.testing.io, &.{in_dir}, out_dir, options, &discard.writer));
    var overwrite = options;
    overwrite.overwrite = true;
    _ = try Repackager.run(allocator, std.testing.io, &.{in_dir}, out_dir, overwrite, &discard.writer);
    var again: std.ArrayList([]u8) = .empty;
    defer freeDocs(allocator, &again);
    _ = try readShards(allocator, out_dir, &again);
    for (got.items, again.items) |a, b| try std.testing.expectEqualStrings(a, b);

    // In memory (one bucket): the validation shard is the last ~10%, never empty.
    var memory = overwrite;
    memory.bucket_bytes = 1 << 30;
    memory.chars_per_shard = 1 << 40;
    const one = try Repackager.run(allocator, std.testing.io, &.{in_dir}, out_dir, memory, &discard.writer);
    try std.testing.expectEqual(@as(usize, 2), one.shards);
    var file = try mod.ParquetFile.open(allocator, std.testing.io, try std.mem.print(&root_buf, "{s}/shard_00001.parquet", .{out_dir}));
    defer file.deinit();
    var val_rows: usize = 0;
    for (file.row_groups) |rg| val_rows += @intCast(rg.num_rows);
    try std.testing.expect(val_rows >= 10 and val_rows <= 20);

    // The output directory cannot be an input; a lone document has no split.
    try std.testing.expectError(error.InputInOutput, Repackager.run(allocator, std.testing.io, &.{out_dir}, out_dir, overwrite, &discard.writer));
    const lone = try std.Io.Dir.path.join(allocator, &.{ root, "lone.txt" });
    defer allocator.free(lone);
    try storage.write(lone, "just one document");
    try std.testing.expectError(error.TooFewDocuments, Repackager.run(allocator, std.testing.io, &.{lone}, out_dir, overwrite, &discard.writer));
}
