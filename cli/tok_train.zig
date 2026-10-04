const std = @import("std");
const log = std.log.scoped(.zignanogpt_tok_train);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// `zignanogpt tok-train`: trains the BPE tokenizer (nanochat's `tok_train.py`)
/// and writes it to `<base dir>/tokenizer/tokenizer.tiktoken`.
pub const TokTrain = struct {
    pub const usage =
        \\usage: zignanogpt tok-train --data <file> [--vocab-size 32768] [--max-chars 2000000000] [--doc-cap 10000]
        \\  --data        text file; documents are separated by blank lines
        \\  --vocab-size  total vocabulary including the 9 special tokens
        \\  --max-chars   stop after this many characters
        \\  --doc-cap     crop each document to this many characters
        \\
    ;

    /// Runs the command.
    ///
    /// Parameters:
    /// - `init`: process state.
    /// - `args`: the command's options.
    /// - `out`: progress output.
    ///
    /// Return: the exit code; storage, tokenizer and argument errors.
    pub fn run(init: std.process.Init, args: *cli.Args, out: *std.Io.Writer) !u8 {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const allocator = init.gpa;
        const data = try args.string("data") orelse {
            try out.writeAll(usage);
            return 2;
        };
        const vocab_size = try args.int(usize, "vocab-size", 32768);
        const max_chars = try args.int(usize, "max-chars", 2_000_000_000);
        const doc_cap = try args.int(usize, "doc-cap", 10_000);
        try args.finish();
        const specials = mod.Tokenizer.special_tokens.len;
        if (vocab_size < 256 + specials) {
            try out.print("--vocab-size must be at least {d}\n", .{256 + specials});
            return 2;
        }

        var config = try mod.Config.init(allocator, init.environ_map);
        defer config.deinit();
        const storage = mod.Storage.init(allocator, init.io);
        var dataset = try mod.TextDataset.load(allocator, storage, data);
        defer dataset.deinit();
        try out.print("max_chars: {d}\ndoc_cap: {d}\nvocab_size: {d}\n", .{ max_chars, doc_cap, vocab_size });
        try out.flush();

        const start = std.Io.Clock.awake.now(init.io);
        var trainer = mod.TokenizerTrainer.init(allocator, mod.Tokenizer.nanochat_max_digits);
        defer trainer.deinit();
        var chars: usize = 0;
        for (dataset.docs) |doc| {
            const capped = capChars(doc, doc_cap);
            try trainer.addText(capped.text);
            chars += capped.chars;
            if (chars > max_chars) break;
        }
        try out.print("{d} characters, {d} unique pieces\n", .{ chars, trainer.uniquePieces() });
        try out.flush();
        const tokens = try trainer.train(allocator, vocab_size - specials);
        defer {
            for (tokens) |t| allocator.free(t);
            allocator.free(tokens);
        }
        var tokenizer = try mod.Tokenizer.init(allocator, tokens, &mod.Tokenizer.special_tokens, mod.Tokenizer.nanochat_max_digits);
        defer tokenizer.deinit();
        const seconds = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(init.io)).nanoseconds)) / 1e9;
        try out.print("Training time: {d:.2}s\n", .{seconds});

        const dir = try std.fs.path.join(allocator, &.{ config.base_dir, "tokenizer" });
        defer allocator.free(dir);
        try tokenizer.save(storage, dir);
        try out.print("Saved tokenizer ({d} tokens) to {s}/{s}\n", .{ tokenizer.vocabSize(), dir, mod.Tokenizer.file_name });

        // Inline sanity check, as tok_train.py does.
        const sample = "Hello world! This is a test.\nNumbers: 123, 4567, 89\nContractions: I'm, you're, it's\nSpecial chars: @#$%^&*()\nUnicode: 你好世界 🌍";
        const ids = try tokenizer.encode(allocator, sample);
        defer allocator.free(ids);
        const decoded = try tokenizer.decode(allocator, ids);
        defer allocator.free(decoded);
        if (!std.mem.eql(u8, decoded, sample)) return error.RoundTripFailed;
        try out.print("Round trip ok: {d} bytes -> {d} tokens\n", .{ sample.len, ids.len });
        return 0;
    }

    const Capped = struct { text: []const u8, chars: usize };

    /// The first `cap` code points of `text` (Python's `doc[:cap]`) and their count.
    fn capChars(text: []const u8, cap: usize) Capped {
        var view = std.unicode.Utf8View.initUnchecked(text);
        var it = view.iterator();
        var count: usize = 0;
        while (count < cap) : (count += 1) {
            _ = it.nextCodepointSlice() orelse break;
        }
        return .{ .text = text[0..it.i], .chars = count };
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "tok-train caps documents by code points" {
    const capped = cli.TokTrain.capChars("héllo 世界", 7);
    try std.testing.expectEqualStrings("héllo 世", capped.text);
    try std.testing.expectEqual(@as(usize, 7), capped.chars);
    try std.testing.expectEqual(@as(usize, 2), cli.TokTrain.capChars("ab", 10).chars);
}
