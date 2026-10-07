const std = @import("std");
const log = std.log.scoped(.zignanogpt_download);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// `zignanogpt download`: fetches pretraining shards (nanochat's
/// `python -m nanochat.dataset -n N`): the first N train shards plus the
/// validation shard, skipping any already present.
pub const Download = struct {
    pub const usage =
        \\usage: zignanogpt download -n <shards>
        \\  -n, --num-files  train shards to fetch (the validation shard always comes too)
        \\
    ;

    /// Runs the command.
    ///
    /// Parameters:
    /// - `init`: process state.
    /// - `args`: the command's options.
    /// - `out`: progress output.
    ///
    /// Return: the exit code (1 if any shard failed); argument errors.
    pub fn run(init: std.process.Init, args: *cli.Args, out: *std.Io.Writer, observer: ?mod.TrainObserver) !u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const allocator = init.gpa;
        const short = try args.string("n");
        const count = if (short) |text|
            std.fmt.parseInt(usize, text, 10) catch return usageError(out)
        else
            try args.int(usize, "num-files", 0);
        try args.finish();
        if (count == 0) return usageError(out);

        var config = try mod.Config.load(allocator, init.environ_map, mod.Storage.init(allocator, init.io));
        defer config.deinit();
        var dataset = try mod.Dataset.init(allocator, init.io, &config, mod.Dataset.default_name);
        defer dataset.deinit();
        const train = @min(count, mod.Dataset.max_shard);
        try out.print("Downloading {d} shards to {s}\n", .{ train + 1, dataset.dir });
        try out.flush();
        var failed: usize = 0;
        for (0..train + 1) |i| {
            mod.TrainObserver.progress(observer, "shards", i, train + 1);
            const index = if (i == train) mod.Dataset.max_shard else i;
            _ = dataset.download(index, out) catch {
                failed += 1;
            };
            try out.flush();
        }
        mod.TrainObserver.progress(observer, "shards", train + 1, train + 1);
        try out.print("Done: {d}/{d} shards available\n", .{ train + 1 - failed, train + 1 });
        return if (failed > 0) 1 else 0;
    }

    /// How to get shards for a dataset, for "no shards" messages.
    ///
    /// Parameters:
    /// - `name`: the dataset.
    ///
    /// Return: the advice.
    pub fn hint(name: []const u8) []const u8 {
        return if (std.mem.eql(u8, name, mod.Dataset.default_name)) "run `zignanogpt download -n 8`" else "make them with `zignanogpt repackage`";
    }

    fn usageError(out: *std.Io.Writer) !u8 {
        try out.writeAll(usage);
        return 2;
    }
};
