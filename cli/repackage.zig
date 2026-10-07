const std = @import("std");
const log = std.log.scoped(.zignanogpt_repackage);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// `zignanogpt repackage`: nanochat's `dev/repackage_data_reference.py` for your
/// own corpus. Shuffles text, JSONL and Parquet documents into pretraining
/// shards at `<base>/base_data_<name>`, which `train --dataset <name>` reads.
pub const Repackage = struct {
    pub const usage =
        \\usage: zignanogpt repackage --name <name> [--input <path>]... [<path>...] [--field text] [--seed 42]
        \\                            [--chars-per-shard 250000000] [--row-group 1024] [--bucket-mib 512] [--overwrite]
        \\  --name             the dataset: shards go to <base>/base_data_<name> (letters, digits, _ and -)
        \\  --input, <path>    files or directories (searched for .txt .md .text .jsonl .ndjson .parquet);
        \\                     text files hold documents separated by blank lines
        \\  --field            the JSONL field / Parquet column with the text (default text)
        \\  --seed             shuffle seed (default 42)
        \\  --chars-per-shard  characters per train shard (default 250000000, nanochat's ~100 MB shards)
        \\  --row-group        documents per Parquet row group (default 1024)
        \\  --bucket-mib       memory for one shuffle bucket; larger corpora are bucketed on disk (default 512)
        \\  --overwrite        replace the dataset's existing shards
        \\  The last shard is validation: min(one shard, 10% of the characters).
        \\
    ;

    /// Runs the command.
    ///
    /// Parameters:
    /// - `init`: process state.
    /// - `args`: the command's options.
    /// - `out`: progress and the summary.
    /// - `observer`: progress hooks (the console's job), or null.
    ///
    /// Return: the exit code; storage errors.
    pub fn run(init: std.process.Init, args: *cli.Args, out: *std.Io.Writer, observer: ?mod.TrainObserver) !u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const allocator = init.gpa;
        if (args.flag("help")) {
            try out.writeAll(usage);
            return 0;
        }
        const name = try args.string("name") orelse {
            try out.writeAll("--name is required\n" ++ usage);
            return 2;
        };
        var options: mod.RepackageOptions = .{ .observer = observer };
        if (try args.string("field")) |field| options.field = field;
        options.seed = try args.int(u64, "seed", options.seed);
        options.chars_per_shard = try args.int(u64, "chars-per-shard", options.chars_per_shard);
        options.row_group = try args.int(usize, "row-group", options.row_group);
        const bucket_mib = try args.int(u64, "bucket-mib", options.bucket_bytes >> 20);
        options.overwrite = args.flag("overwrite");
        const named = try args.strings(allocator, "input");
        defer allocator.free(named);
        const rest = try args.positionals(allocator);
        defer allocator.free(rest);
        try args.finish();
        if (options.chars_per_shard == 0 or options.row_group == 0 or bucket_mib == 0) {
            try out.writeAll("--chars-per-shard, --row-group and --bucket-mib must be positive\n");
            return 2;
        }
        options.bucket_bytes = bucket_mib << 20;
        const inputs = try std.mem.concat(allocator, []const u8, &.{ named, rest });
        defer allocator.free(inputs);
        if (inputs.len == 0) {
            try out.writeAll("no inputs: pass files or directories\n" ++ usage);
            return 2;
        }

        const storage = mod.Storage.init(allocator, init.io);
        var config = try mod.Config.load(allocator, init.environ_map, storage);
        defer config.deinit();
        const dir = mod.Dataset.directory(allocator, config.base_dir, name) catch |err| switch (err) {
            error.InvalidDatasetName => return 2,
            else => return err,
        };
        defer allocator.free(dir);
        try out.print("Repackaging into {s}\n", .{dir});
        try out.flush();
        const start = std.Io.Clock.awake.now(init.io);
        const summary = mod.Repackager.run(allocator, init.io, inputs, dir, options, out) catch |err| switch (err) {
            // Explained by a warning already.
            error.DatasetExists, error.NoInputs, error.TooFewDocuments, error.InputInOutput, error.MissingColumn, error.NotFound => return 1,
            else => return err,
        };
        const seconds = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds)) / 1e9;
        try out.print("Done in {d:.1}s: {d} documents, {d} characters into {d} shards ({d} MiB); train with --dataset {s}\n", .{ seconds, summary.documents, summary.characters, summary.shards, summary.bytes >> 20, name });
        return 0;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "repackage writes a dataset that train's shard listing finds" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const storage = mod.Storage.init(allocator, std.testing.io);
    const corpus = try std.fs.path.join(allocator, &.{ root, "corpus.txt" });
    defer allocator.free(corpus);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(allocator);
    for (0..30) |i| try text.print(allocator, "document number {d}\n\n", .{i});
    try storage.write(corpus, text.items);

    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put(mod.Config.base_dir_env, root);
    try env.put(mod.Config.nanochat_dir_env, "/nonexistent-nanochat");
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const process: std.process.Init = .{ .minimal = .{ .environ = .empty, .args = undefined }, .arena = &arena, .gpa = allocator, .io = std.testing.io, .environ_map = &env, .preopens = undefined };
    var discard: std.Io.Writer.Discarding = .init(&.{});

    var args = try cli.Args.init(allocator, &.{ "--name", "mine", corpus, "--chars-per-shard", "100", "--row-group", "2" });
    defer args.deinit();
    try std.testing.expectEqual(@as(u8, 0), try Repackage.run(process, &args, &discard.writer, null));
    var config = try mod.Config.load(allocator, &env, storage);
    defer config.deinit();
    var dataset = try mod.Dataset.init(allocator, std.testing.io, &config, "mine");
    defer dataset.deinit();
    const shards = try dataset.list(allocator);
    defer {
        for (shards) |p| allocator.free(p);
        allocator.free(shards);
    }
    try std.testing.expect(shards.len >= 3);
    const names = try mod.Dataset.names(allocator, storage, root);
    defer {
        for (names) |n| allocator.free(n);
        allocator.free(names);
    }
    try std.testing.expectEqual(@as(usize, 1), names.len);
    try std.testing.expectEqualStrings("mine", names[0]);

    // Existing shards without --overwrite, a bad name, no name: not crashes, exit codes.
    var again = try cli.Args.init(allocator, &.{ "--name", "mine", corpus });
    defer again.deinit();
    try std.testing.expectEqual(@as(u8, 1), try Repackage.run(process, &again, &discard.writer, null));
    var bad = try cli.Args.init(allocator, &.{ "--name", "../x", corpus });
    defer bad.deinit();
    try std.testing.expectEqual(@as(u8, 2), try Repackage.run(process, &bad, &discard.writer, null));
    var unnamed = try cli.Args.init(allocator, &.{corpus});
    defer unnamed.deinit();
    try std.testing.expectEqual(@as(u8, 2), try Repackage.run(process, &unnamed, &discard.writer, null));
}
