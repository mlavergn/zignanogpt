const std = @import("std");
const log = std.log.scoped(.zignanogpt_operation);
const cli = @import("module.zig");
const mod = cli.nanogpt;

/// One form field of an operation: a command-line flag and how it is edited.
pub const Field = struct {
    /// The flag without dashes; empty for the free-form "more options" field.
    flag: []const u8,
    label: []const u8,
    default: []const u8 = "",
    kind: Kind = .text,
    /// Shown while the field is selected.
    hint: []const u8 = "",
    /// Shown dimmed while the field is empty: what the command uses then.
    placeholder: []const u8 = "",
    /// Passed only when the operation runs this command (see `Operation.alternate`).
    only: ?cli.Command = null,

    /// Whether the field is passed when `command` runs.
    pub fn appliesTo(self: Field, command: cli.Command) bool {
        return self.only == null or self.only.? == command;
    }

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
    /// A second command the operation runs when a field holds one of some values.
    alternate: ?Alternate = null,

    pub const Alternate = struct {
        command: cli.Command,
        /// The deciding field's index.
        field: usize,
        /// Values (case-insensitive) that select `command`.
        values: []const []const u8,
    };

    /// The most fields any operation has (the form's storage is sized by it).
    pub const max_fields = 20;

    /// The console's operations, in the order the left pane lists them.
    pub const all = [_]Operation{
        .{ .title = "Overview", .summary = "Directories, tokenizer, data shards and checkpoints.", .command = null },
        .{
            .title = "Prepare data",
            .summary = "Shuffle your own corpus (text, JSONL, Parquet) into pretraining shards; pick it with the dataset field.",
            .command = .repackage,
            .fields = &.{
                .{ .flag = "name", .label = "dataset name", .hint = "required; shards go to <base>/base_data_<name>" },
                .{ .flag = "input", .label = "input", .hint = "a file or directory (.txt .md .jsonl .ndjson .parquet); text: blank lines between documents" },
                .{ .flag = "field", .label = "text field", .placeholder = "text", .hint = "the JSONL field or Parquet column" },
                .{ .flag = "chars-per-shard", .label = "chars per shard", .placeholder = "250000000", .hint = "~100 MB shards, as nanochat" },
                .{ .flag = "row-group", .label = "row group", .placeholder = "1024", .hint = "documents per Parquet row group" },
                .{ .flag = "seed", .label = "shuffle seed", .placeholder = "42" },
                .{ .flag = "overwrite", .label = "overwrite", .kind = .toggle, .default = "no", .hint = "yes replaces the dataset's shards" },
                .{ .flag = "", .label = "more options", .placeholder = "none", .hint = "e.g. --input another/dir --bucket-mib 1024" },
            },
        },
        .{
            .title = "Download data",
            .summary = "Fetch ClimbMix pretraining shards (the validation shard always comes too).",
            .command = .download,
            .fields = &.{.{ .flag = "num-files", .label = "train shards", .default = "8", .hint = "runcpu.sh uses 8 (~830 MB)" }},
        },
        .{
            .title = "Train tokenizer",
            .summary = "Learn the BPE vocabulary from a dataset's train shards (or a text file).",
            .command = .@"tok-train",
            .fields = &.{
                .{ .flag = "vocab-size", .label = "vocab size", .default = "32768", .hint = "includes the 9 special tokens" },
                .{ .flag = "max-chars", .label = "max chars", .default = "2000000000" },
                .{ .flag = "doc-cap", .label = "doc cap", .default = "10000", .hint = "characters kept per document" },
                .{ .flag = "dataset", .label = "dataset", .placeholder = "climbmix", .hint = "shards from Download data or Prepare data" },
                .{ .flag = "data", .label = "text file", .placeholder = "the dataset's shards" },
            },
        },
        .{
            .title = "Train model",
            .summary = "Base pretraining (base_train.py). `s` saves and stops.",
            .command = .train,
            .fields = &.{
                .{ .flag = "depth", .label = "depth", .default = "6", .hint = "layers; width follows (64 x depth)" },
                .{ .flag = "head-dim", .label = "head dim", .default = "64" },
                .{ .flag = "max-seq-len", .label = "max seq len", .default = "512" },
                .{ .flag = "window-pattern", .label = "window pattern", .default = "L", .hint = "L full attention; S short window, repeated over layers" },
                .{ .flag = "num-iterations", .label = "iterations", .default = "5000", .hint = "empty: from the data:param ratio" },
                .{ .flag = "device-batch-size", .label = "device batch", .default = "32" },
                .{ .flag = "total-batch-size", .label = "total batch", .default = "16384", .hint = "tokens per step" },
                .{ .flag = "eval-every", .label = "eval every", .default = "100" },
                .{ .flag = "eval-tokens", .label = "eval tokens", .default = "524288" },
                .{ .flag = "sample-every", .label = "sample every", .default = "100" },
                .{ .flag = "save-every", .label = "save every" },
                .{ .flag = "model-tag", .label = "model tag" },
                .{ .flag = "resume-from-step", .label = "resume from step", .placeholder = "none (new run)" },
                .{ .flag = "dataset", .label = "dataset", .placeholder = "climbmix", .hint = "shards from Download data or Prepare data" },
                .{ .flag = "data", .label = "text file", .placeholder = "the dataset's shards" },
                .{ .flag = "threads", .label = "threads", .placeholder = "one per CPU" },
                .{ .flag = "", .label = "more options", .placeholder = "none", .hint = "any other base_train flags, e.g. --matrix-lr 0.03" },
            },
        },
        .{
            .title = "Fine-tune (SFT)",
            .summary = "chat_sft.py: SmolTalk + MMLU + GSM8K (+ your conversations) from a base checkpoint (data ~1 GB on first run). `s` saves and stops.",
            .command = .sft,
            .fields = &.{
                .{ .flag = "model-tag", .label = "base model tag", .placeholder = "largest d<N>" },
                .{ .flag = "model-step", .label = "base step", .placeholder = "last" },
                .{ .flag = "num-iterations", .label = "iterations", .default = "1500", .hint = "runcpu.sh: 1500; empty or -1: one epoch" },
                .{ .flag = "max-seq-len", .label = "max seq len", .placeholder = "base run's" },
                .{ .flag = "device-batch-size", .label = "device batch", .placeholder = "base run's" },
                .{ .flag = "total-batch-size", .label = "total batch", .placeholder = "base run's" },
                .{ .flag = "eval-every", .label = "eval every", .default = "200" },
                .{ .flag = "eval-tokens", .label = "eval tokens", .default = "524288", .hint = "runcpu.sh: 524288 (chat_sft.py: 20971520)" },
                .{ .flag = "chatcore-every", .label = "ChatCORE every", .default = "200", .hint = "-1 disables" },
                .{ .flag = "chatcore-max-cat", .label = "ChatCORE max cat", .placeholder = "all", .hint = "problems per multiple-choice task" },
                .{ .flag = "chatcore-max-sample", .label = "ChatCORE max gen", .default = "24", .hint = "GSM8K problems" },
                .{ .flag = "conversations", .label = "conversations", .placeholder = "none", .hint = "your JSONL file: one [{role, content}, ...] conversation per line" },
                .{ .flag = "conversations-epochs", .label = "conversation epochs", .placeholder = "1", .hint = "copies of your conversations in the mixture" },
                .{ .flag = "threads", .label = "threads", .placeholder = "one per CPU" },
                .{ .flag = "", .label = "more options", .placeholder = "none", .hint = "any other chat_sft flags, e.g. --mmlu-epochs 1" },
            },
        },
        .{
            .title = "Reinforcement (RL)",
            .summary = "chat_rl.py: on-policy REINFORCE on GSM8K from the sft checkpoint. `s` saves and stops.",
            .command = .rl,
            .fields = &.{
                .{ .flag = "model-tag", .label = "sft model tag", .placeholder = "largest d<N>" },
                .{ .flag = "model-step", .label = "sft step", .placeholder = "last" },
                .{ .flag = "examples-per-step", .label = "examples/step", .default = "16" },
                .{ .flag = "num-samples", .label = "samples", .default = "16", .hint = "per example" },
                .{ .flag = "device-batch-size", .label = "device batch", .default = "8", .hint = "divides samples" },
                .{ .flag = "max-new-tokens", .label = "max new tokens", .default = "256" },
                .{ .flag = "eval-every", .label = "eval every", .default = "60" },
                .{ .flag = "eval-examples", .label = "eval examples", .default = "400" },
                .{ .flag = "save-every", .label = "save every", .default = "60" },
                .{ .flag = "threads", .label = "threads", .placeholder = "one per CPU" },
                .{ .flag = "", .label = "more options", .placeholder = "none", .hint = "any other chat_rl flags, e.g. --temperature 0.8" },
            },
        },
        .{
            .title = "Evaluate",
            .summary = "base models: CORE, bpb and samples (base_eval.py); sft or rl: ARC, MMLU, GSM8K (chat_eval.py).",
            .command = .eval,
            .alternate = .{ .command = .@"chat-eval", .field = 0, .values = &.{ "sft", "rl" } },
            .fields = &.{
                .{ .flag = "source", .label = "model", .default = "base", .only = .@"chat-eval", .hint = "base, sft or rl; it picks the evaluation" },
                .{ .flag = "model-tag", .label = "model tag", .placeholder = "largest d<N>" },
                .{ .flag = "step", .label = "step", .placeholder = "last" },
                .{ .flag = "eval", .label = "modes", .default = "core,bpb,sample", .only = .eval, .hint = "base: any of core, bpb, sample" },
                .{ .flag = "max-per-task", .label = "max per task", .default = "16", .only = .eval, .placeholder = "all", .hint = "base: CORE examples per task (runcpu.sh: 16)" },
                .{ .flag = "device-batch-size", .label = "bpb batch", .default = "1", .only = .eval, .hint = "base: runcpu.sh uses 1" },
                .{ .flag = "split-tokens", .label = "bpb tokens", .default = "16384", .only = .eval, .hint = "base: per split (runcpu.sh: 16384)" },
                .{ .flag = "dataset", .label = "bpb dataset", .only = .eval, .placeholder = "climbmix", .hint = "base: the shards bpb reads" },
                .{ .flag = "task-name", .label = "tasks", .only = .@"chat-eval", .placeholder = "all four", .hint = "sft/rl: e.g. ARC-Easy|MMLU" },
                .{ .flag = "max-problems", .label = "max problems", .only = .@"chat-eval", .placeholder = "all", .hint = "sft/rl: per task" },
                .{ .flag = "max-new-tokens", .label = "max new tokens", .default = "512", .only = .@"chat-eval", .hint = "sft/rl: GSM8K" },
                .{ .flag = "threads", .label = "threads", .placeholder = "one per CPU" },
            },
        },
        .{
            .title = "Chat",
            .summary = "Talk to a trained model (chat_cli.py); runs beside other jobs.",
            .command = .chat,
            .fields = &.{
                .{ .flag = "source", .label = "source", .placeholder = "sft (base if none)", .hint = "base, sft or rl" },
                .{ .flag = "model-tag", .label = "model tag", .placeholder = "largest d<N>" },
                .{ .flag = "step", .label = "step", .placeholder = "last" },
                .{ .flag = "temperature", .label = "temperature", .default = "0.6", .hint = "0 is greedy" },
                .{ .flag = "top-k", .label = "top-k", .default = "50", .hint = "0 samples from every token" },
                .{ .flag = "max-tokens", .label = "max tokens", .default = "256", .hint = "per reply" },
                .{ .flag = "threads", .label = "threads", .placeholder = "one per CPU" },
            },
        },
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
        const command = self.commandFor(values);
        for (self.fields, values) |field, raw| {
            if (!field.appliesTo(command)) continue;
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

    /// What an empty field stands for: on the Train page base_train.py's
    /// default (an empty field passes no flag), elsewhere its placeholder.
    ///
    /// Parameters:
    /// - `self`: the operation.
    /// - `index`: the field.
    /// - `values`: the form, one value per field.
    /// - `buf`: formatting space.
    ///
    /// Return: the text (empty when there is nothing to show).
    pub fn placeholder(self: Self, index: usize, values: []const []const u8, buf: []u8) []const u8 {
        const field = self.fields[index];
        if (self.command == .train) {
            if (trainDefault(self, field.flag, values, buf)) |text| return text;
        }
        return field.placeholder;
    }

    /// The value `train` uses for a flag it is not given.
    fn trainDefault(self: Self, flag: []const u8, values: []const []const u8, buf: []u8) ?[]const u8 {
        const options = mod.TrainOptions{};
        if (std.mem.eql(u8, flag, "model-tag")) {
            // d<depth>, from the depth field when it is filled in.
            for (self.fields, values) |f, v| {
                const typed = std.mem.trim(u8, v, " \t");
                if (std.mem.eql(u8, f.flag, "depth") and typed.len > 0) return std.fmt.bufPrint(buf, "d{s}", .{typed}) catch null;
            }
            return std.fmt.bufPrint(buf, "d{d}", .{options.depth}) catch null;
        }
        inline for (@typeInfo(mod.TrainOptions).@"struct".fields) |f| {
            const name = comptime blk: {
                var kebab: [f.name.len]u8 = undefined;
                for (f.name, 0..) |c, i| kebab[i] = if (c == '_') '-' else c;
                const final = kebab;
                break :blk &final;
            };
            if (std.mem.eql(u8, flag, name)) {
                const value = @field(options, f.name);
                return switch (f.type) {
                    usize => if (value == 0)
                        (if (std.mem.eql(u8, f.name, "save_every")) "end only" else if (std.mem.eql(u8, f.name, "eval_every") or std.mem.eql(u8, f.name, "sample_every")) "off" else "auto")
                    else
                        std.fmt.bufPrint(buf, "{d}", .{value}) catch null,
                    f64 => std.fmt.bufPrint(buf, "{d}", .{value}) catch null,
                    []const u8 => value,
                    else => null,
                };
            }
        }
        return null;
    }

    /// The command a filled-in form runs: the alternate when its field selects it.
    ///
    /// Parameters:
    /// - `self`: the operation.
    /// - `values`: one per field, as typed.
    ///
    /// Return: the command (undefined for the overview: check `command` first).
    pub fn commandFor(self: Self, values: []const []const u8) cli.Command {
        if (self.alternate) |alt| {
            const value = std.mem.trim(u8, values[alt.field], " \t");
            for (alt.values) |v| {
                if (std.ascii.eqlIgnoreCase(value, v)) return alt.command;
            }
        }
        return self.command.?;
    }

    /// The index of the operation running `command` (directly or as its alternate).
    pub fn indexOf(command: cli.Command) ?usize {
        for (all, 0..) |op, i| {
            if (op.command == command) return i;
            if (op.alternate) |alt| if (alt.command == command) return i;
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
    values[0] = "8";
    values[4] = " 20 ";
    values[13] = "mine";
    values[16] = "--matrix-lr 0.03";
    const args = try train.args(arena.allocator(), values[0..train.fields.len]);
    const want = [_][]const u8{ "--depth", "8", "--num-iterations", "20", "--dataset", "mine", "--matrix-lr", "0.03" };
    try std.testing.expectEqual(want.len, args.len);
    for (want, args) |w, a| try std.testing.expectEqualStrings(w, a);

    // Prepare data: the overwrite toggle is a bare flag, only when "yes".
    const prepare = Operation.all[Operation.indexOf(.repackage).?];
    var form: [Operation.max_fields][]const u8 = @splat("");
    form[0] = "mine";
    form[1] = "corpus/";
    form[6] = "no";
    const quiet = try prepare.args(arena.allocator(), form[0..prepare.fields.len]);
    try std.testing.expectEqual(@as(usize, 4), quiet.len);
    form[6] = "Yes";
    const loud = try prepare.args(arena.allocator(), form[0..prepare.fields.len]);
    try std.testing.expectEqualStrings("--overwrite", loud[loud.len - 1]);
    for (Operation.all) |op| {
        try std.testing.expect(op.fields.len <= Operation.max_fields);
        try std.testing.expectEqual(@as(?u8, null), op.phase());
    }
}

test "empty fields show what the command will use" {
    const train = Operation.all[Operation.indexOf(.train).?];
    var values: [Operation.max_fields][]const u8 = @splat("");
    var buf: [32]u8 = undefined;
    const field = struct {
        fn of(op: Operation, flag: []const u8) usize {
            for (op.fields, 0..) |f, i| if (std.mem.eql(u8, f.flag, flag)) return i;
            unreachable;
        }
    };
    const v = values[0..train.fields.len];
    // Cleared fields fall back to base_train.py's defaults.
    try std.testing.expectEqualStrings("20", train.placeholder(field.of(train, "depth"), v, &buf));
    try std.testing.expectEqualStrings("auto", train.placeholder(field.of(train, "num-iterations"), v, &buf));
    try std.testing.expectEqualStrings("end only", train.placeholder(field.of(train, "save-every"), v, &buf));
    try std.testing.expectEqualStrings("d20", train.placeholder(field.of(train, "model-tag"), v, &buf));
    try std.testing.expectEqualStrings("one per CPU", train.placeholder(field.of(train, "threads"), v, &buf));
    values[field.of(train, "depth")] = "6";
    try std.testing.expectEqualStrings("d6", train.placeholder(field.of(train, "model-tag"), v, &buf));
    const chat = Operation.all[Operation.indexOf(.chat).?];
    try std.testing.expectEqualStrings("sft (base if none)", chat.placeholder(0, values[0..chat.fields.len], &buf));
}

test "evaluate runs base_eval or chat_eval with that command's flags only" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const index = Operation.indexOf(.eval).?;
    try std.testing.expectEqual(index, Operation.indexOf(.@"chat-eval").?);
    const op = Operation.all[index];
    var values: [Operation.max_fields][]const u8 = undefined;
    for (op.fields, 0..) |f, i| values[i] = f.default;
    values[1] = "d6";
    try std.testing.expectEqual(cli.Command.eval, op.commandFor(values[0..op.fields.len]));
    const base = try op.args(a, values[0..op.fields.len]);
    const want_base = [_][]const u8{ "--model-tag", "d6", "--eval", "core,bpb,sample", "--max-per-task", "16", "--device-batch-size", "1", "--split-tokens", "16384" };
    try std.testing.expectEqual(want_base.len, base.len);
    for (want_base, base) |w, g| try std.testing.expectEqualStrings(w, g);
    values[0] = "SFT";
    try std.testing.expectEqual(cli.Command.@"chat-eval", op.commandFor(values[0..op.fields.len]));
    const chat = try op.args(a, values[0..op.fields.len]);
    const want_chat = [_][]const u8{ "--source", "SFT", "--model-tag", "d6", "--max-new-tokens", "512" };
    try std.testing.expectEqual(want_chat.len, chat.len);
    for (want_chat, chat) |w, g| try std.testing.expectEqualStrings(w, g);
}
