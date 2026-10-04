const std = @import("std");
const log = std.log.scoped(.zignanogpt_import);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// `zignanogpt import`: converts a Python nanochat checkpoint (`model_*.pt`,
/// `meta_*.json`, `tokenizer/tokenizer.pkl`) into this port's base directory.
pub const Import = struct {
    pub const usage =
        \\usage: zignanogpt import [--from <dir>] [--source base|sft|rl] [--model-tag <tag>] [--step <n>] [--tokenizer-only]
        \\  --from       the Python nanochat base dir (default: $NANOCHAT_BASE_DIR or ~/.cache/nanochat)
        \\  --source     which checkpoints (default: base)
        \\  --model-tag  e.g. d12 (default: the largest d<N>)
        \\  --step       (default: the last)
        \\  --tokenizer-only  import <from>/tokenizer/tokenizer.pkl alone
        \\
    ;

    /// Runs the command.
    ///
    /// Parameters:
    /// - `init`: process state.
    /// - `args`: the command's options.
    /// - `out`: progress output.
    ///
    /// Return: the exit code; import errors.
    pub fn run(init: std.process.Init, args: *cli.Args, out: *std.Io.Writer) !u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const allocator = init.gpa;
        const from_arg = try args.string("from");
        const source_name = try args.string("source") orelse "base";
        const tag = try args.string("model-tag");
        const step_text = try args.string("step");
        const tokenizer_only = args.flag("tokenizer-only");
        try args.finish();
        const kind = std.meta.stringToEnum(mod.CheckpointKind, source_name) orelse {
            try out.writeAll(usage);
            return 2;
        };
        const step: ?usize = if (step_text) |t| std.fmt.parseInt(usize, t, 10) catch {
            try out.writeAll(usage);
            return 2;
        } else null;

        var config = try mod.Config.load(allocator, init.environ_map, mod.Storage.init(allocator, init.io));
        defer config.deinit();
        const storage = mod.Storage.init(allocator, init.io);
        const from = try storage.absolute(from_arg orelse config.nanochat_dir);
        defer allocator.free(from);
        if (tokenizer_only) {
            const pkl = try std.fs.path.join(allocator, &.{ from, "tokenizer", "tokenizer.pkl" });
            defer allocator.free(pkl);
            var tokenizer = try mod.TorchImport.loadTokenizer(allocator, storage, pkl);
            defer tokenizer.deinit();
            const dir = try std.fs.path.join(allocator, &.{ config.base_dir, "tokenizer" });
            defer allocator.free(dir);
            try tokenizer.save(storage, dir);
            try out.print("Imported tokenizer ({d} tokens) into {s}\n", .{ tokenizer.vocabSize(), dir });
            return 0;
        }
        var backend = try mod.Backend.init(allocator, init.io, .{});
        defer backend.deinit();
        try out.print("Importing {s} checkpoints from {s}\n", .{ source_name, from });
        try out.flush();
        const imported = try mod.TorchImport.importCheckpoint(allocator, &backend, storage, from, config.base_dir, kind, tag, step);
        defer allocator.free(imported.tag);
        try out.print("Imported {s} step {d} into {s}/{s}/{s}, tokenizer into {s}/tokenizer\n", .{ imported.tag, imported.step, config.base_dir, kind.dirName(), imported.tag, config.base_dir });
        return 0;
    }
};
