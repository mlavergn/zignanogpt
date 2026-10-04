const std = @import("std");
const log = std.log.scoped(.zignanogpt_operation);
const cli = @import("module.zig");

/// One form field of an operation: a command-line flag and how it is edited.
pub const Field = struct {
    /// The flag without dashes; empty for the free-form "more options" field.
    flag: []const u8,
    label: []const u8,
    default: []const u8 = "",
    kind: Kind = .text,
    /// Shown while the field is selected.
    hint: []const u8 = "",

    pub const Kind = enum {
        /// A value; empty leaves the flag out.
        text,
        /// `yes` adds the bare flag, anything else leaves it out.
        toggle,
    };
};

/// One entry of the console's left pane: a command with a form, or the overview.
pub const Operation = struct {
    const Self = @This();

    title: []const u8,
    summary: []const u8,
    /// Null for the overview, which runs nothing.
    command: ?cli.Command,
    fields: []const Field = &.{},

    /// The most fields any operation has (the form's storage is sized by it).
    pub const max_fields = 16;

    /// The console's operations, in the order the left pane lists them.
    pub const all = [_]Operation{
        .{ .title = "Overview", .summary = "Directories, tokenizer, data shards and checkpoints.", .command = null },
        .{
            .title = "Download data",
            .summary = "Fetch ClimbMix pretraining shards (the validation shard always comes too).",
            .command = .download,
            .fields = &.{.{ .flag = "num-files", .label = "train shards", .default = "8", .hint = "runcpu.sh uses 8 (~830 MB)" }},
        },
        .{
            .title = "Train tokenizer",
            .summary = "Learn the BPE vocabulary from the train shards (or a text file).",
            .command = .@"tok-train",
            .fields = &.{
                .{ .flag = "vocab-size", .label = "vocab size", .default = "32768", .hint = "includes the 9 special tokens" },
                .{ .flag = "max-chars", .label = "max chars", .default = "2000000000" },
                .{ .flag = "doc-cap", .label = "doc cap", .default = "10000", .hint = "characters kept per document" },
                .{ .flag = "data", .label = "text file", .hint = "empty: the downloaded shards" },
            },
        },
        .{
            .title = "Evaluate tokenizer",
            .summary = "Bytes per token of the trained tokenizer, optionally against another .tiktoken.",
            .command = .@"tok-eval",
            .fields = &.{
                .{ .flag = "data", .label = "text file", .hint = "empty: the first train and val row groups" },
                .{ .flag = "compare", .label = "compare with", .hint = "a .tiktoken file, e.g. cl100k_base.tiktoken" },
            },
        },
        .{
            .title = "Import from nanochat",
            .summary = "Convert a Python nanochat checkpoint and tokenizer into this port's format.",
            .command = .import,
            .fields = &.{
                .{ .flag = "from", .label = "nanochat dir", .hint = "empty: $NANOCHAT_BASE_DIR or ~/.cache/nanochat" },
                .{ .flag = "source", .label = "source", .default = "base", .hint = "base, sft or rl" },
                .{ .flag = "model-tag", .label = "model tag", .hint = "empty: the largest d<N>" },
                .{ .flag = "step", .label = "step", .hint = "empty: the last" },
                .{ .flag = "tokenizer-only", .label = "tokenizer only", .default = "no", .kind = .toggle },
            },
        },
        .{
            .title = "Train model",
            .summary = "Base pretraining (base_train.py). `s` saves a checkpoint and stops.",
            .command = .train,
            .fields = &.{
                .{ .flag = "preset", .label = "preset", .default = "cpu", .hint = "cpu: runcpu.sh's settings; empty: base_train.py defaults" },
                .{ .flag = "depth", .label = "depth" },
                .{ .flag = "max-seq-len", .label = "max seq len" },
                .{ .flag = "num-iterations", .label = "iterations" },
                .{ .flag = "device-batch-size", .label = "device batch" },
                .{ .flag = "total-batch-size", .label = "total batch" },
                .{ .flag = "eval-every", .label = "eval every" },
                .{ .flag = "sample-every", .label = "sample every" },
                .{ .flag = "save-every", .label = "save every" },
                .{ .flag = "model-tag", .label = "model tag", .hint = "empty: d<depth>" },
                .{ .flag = "resume-from-step", .label = "resume from step" },
                .{ .flag = "data", .label = "text file", .hint = "empty: the downloaded shards" },
                .{ .flag = "threads", .label = "threads", .hint = "empty: one per CPU" },
                .{ .flag = "", .label = "more options", .hint = "any other base_train flags, e.g. --matrix-lr 0.03" },
            },
        },
        .{ .title = "Chat", .summary = "Talk to a trained model.", .command = .chat },
        .{ .title = "Evaluate model", .summary = "Base model bpb and CORE.", .command = .eval },
        .{ .title = "Fine-tune (SFT)", .summary = "Supervised fine-tuning on conversations.", .command = .sft },
        .{ .title = "Evaluate chat", .summary = "ARC, MMLU and GSM8K.", .command = .@"chat-eval" },
        .{ .title = "Reinforcement (RL)", .summary = "RL on GSM8K.", .command = .rl },
    };

    /// The PLAN.md phase that implements this operation, or null when it works.
    pub fn phase(self: Self) ?u8 {
        const command = self.command orelse return null;
        return command.phase();
    }

    /// Builds the command-line arguments a filled-in form stands for.
    ///
    /// Parameters:
    /// - `self`: the operation.
    /// - `allocator`: owns the result (an arena is simplest: strings are borrowed from `values`).
    /// - `values`: one per field, as typed.
    ///
    /// Return: the arguments; allocation errors.
    pub fn args(self: Self, allocator: std.mem.Allocator, values: []const []const u8) ![]const []const u8 {
        var out: std.ArrayList([]const u8) = .empty;
        for (self.fields, values) |field, raw| {
            const value = std.mem.trim(u8, raw, " \t");
            if (value.len == 0) continue;
            if (field.flag.len == 0) {
                var words = std.mem.tokenizeAny(u8, value, " \t");
                while (words.next()) |w| try out.append(allocator, w);
                continue;
            }
            const flag = try std.mem.concat(allocator, u8, &.{ "--", field.flag });
            switch (field.kind) {
                .text => {
                    try out.append(allocator, flag);
                    try out.append(allocator, value);
                },
                .toggle => if (std.ascii.eqlIgnoreCase(value, "yes")) try out.append(allocator, flag),
            }
        }
        log.debug("{s}: {d} arguments", .{ self.title, out.items.len });
        return out.toOwnedSlice(allocator);
    }

    /// The index of the operation running `command`.
    pub fn indexOf(command: cli.Command) ?usize {
        for (all, 0..) |op, i| {
            if (op.command == command) return i;
        }
        return null;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "operation forms become command-line arguments" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const train = Operation.all[Operation.indexOf(.train).?];
    var values: [Operation.max_fields][]const u8 = @splat("");
    values[0] = "cpu";
    values[3] = " 20 ";
    values[13] = "--matrix-lr 0.03";
    const args = try train.args(arena.allocator(), values[0..train.fields.len]);
    const want = [_][]const u8{ "--preset", "cpu", "--num-iterations", "20", "--matrix-lr", "0.03" };
    try std.testing.expectEqual(want.len, args.len);
    for (want, args) |w, a| try std.testing.expectEqualStrings(w, a);

    const import = Operation.all[Operation.indexOf(.import).?];
    const toggled = try import.args(arena.allocator(), &.{ "", "base", "", "", "yes" });
    try std.testing.expectEqualStrings("--tokenizer-only", toggled[2]);
    for (Operation.all) |op| try std.testing.expect(op.fields.len <= Operation.max_fields);
    try std.testing.expectEqual(@as(?u8, 9), Operation.all[Operation.indexOf(.chat).?].phase());
}
