const std = @import("std");
const log = std.log.scoped(.zignanogpt_task);
const mod = @import("module.zig");

/// nanochat's tasks (`tasks/*.py`).
pub const TaskKind = enum {
    smoltalk,
    mmlu,
    arc_easy,
    arc_challenge,
    gsm8k,

    /// The name chat_eval and chat_sft use, e.g. `ARC-Easy`.
    pub fn name(self: TaskKind) []const u8 {
        return switch (self) {
            .smoltalk => "SmolTalk",
            .mmlu => "MMLU",
            .arc_easy => "ARC-Easy",
            .arc_challenge => "ARC-Challenge",
            .gsm8k => "GSM8K",
        };
    }

    pub fn parse(text: []const u8) ?TaskKind {
        inline for (@typeInfo(TaskKind).@"enum".fields) |f| {
            const kind: TaskKind = @enumFromInt(f.value);
            if (std.mem.eql(u8, text, kind.name())) return kind;
        }
        return null;
    }
};

/// How a task is scored: compare the logits of the answer letters, or sample a completion.
pub const EvalType = enum { categorical, generative };

/// One task over a dataset split (nanochat's `Task`): `start/stop/step` slice
/// the shuffled rows, `conversation` builds the chat the model trains on or is
/// prompted with, and `evaluate` scores a response.
pub const Task = struct {
    const Self = @This();

    kind: TaskKind,
    data: mod.HubDataset,
    start: usize = 0,
    stop: ?usize = null,
    step: usize = 1,

    /// MMLU's answer letters.
    pub const mmlu_letters = [_][]const u8{ "A", "B", "C", "D" };

    /// Loads (downloading on first use) a task's split.
    ///
    /// Parameters:
    /// - `allocator`: owns the data.
    /// - `io`: file and network access.
    /// - `config`: the base directories.
    /// - `kind`: the task.
    /// - `split`: e.g. `train`, `test`, `auxiliary_train` (MMLU).
    /// - `out`: download progress, or null.
    ///
    /// Return: the task (rows shuffled with seed 42, as nanochat); loading errors.
    pub fn open(allocator: std.mem.Allocator, io: std.Io, config: *const mod.Config, kind: TaskKind, split: []const u8, out: ?*std.Io.Writer) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const source: struct { repo: []const u8, subset: []const u8, columns: []const []const u8 } = switch (kind) {
            .smoltalk => .{ .repo = "HuggingFaceTB/smol-smoltalk", .subset = "default", .columns = &.{ "messages.role", "messages.content" } },
            .mmlu => .{ .repo = "cais/mmlu", .subset = "all", .columns = &.{ "question", "choices", "answer" } },
            .arc_easy, .arc_challenge => .{ .repo = "allenai/ai2_arc", .subset = kind.name(), .columns = &.{ "question", "choices.text", "choices.label", "answerKey" } },
            .gsm8k => .{ .repo = "openai/gsm8k", .subset = "main", .columns = &.{ "question", "answer" } },
        };
        var data = try mod.HubDataset.open(allocator, io, config, source.repo, source.subset, split, source.columns, out);
        errdefer data.deinit();
        try data.shuffle(42);
        return Self{ .kind = kind, .data = data };
    }

    pub fn deinit(self: *Self) void {
        self.data.deinit();
    }

    pub fn evalType(self: *const Self) EvalType {
        return switch (self.kind) {
            .mmlu, .arc_easy, .arc_challenge => .categorical,
            .smoltalk, .gsm8k => .generative,
        };
    }

    /// The examples in the slice (`stop` is clamped to the data).
    pub fn len(self: *const Self) usize {
        const stop = @min(self.stop orelse self.data.len(), self.data.len());
        if (stop <= self.start) return 0;
        return (stop - self.start + self.step - 1) / self.step;
    }

    /// The `index`-th example as a conversation.
    ///
    /// Parameters:
    /// - `self`: the task.
    /// - `arena`: holds the messages (strings are borrowed from the dataset).
    /// - `index`: below `len`.
    ///
    /// Return: the conversation; `error.InvalidTaskRow` for a malformed row.
    pub fn conversation(self: *const Self, arena: std.mem.Allocator, index: usize) !mod.Conversation {
        const row = self.start + index * self.step;
        const d = &self.data;
        switch (self.kind) {
            .smoltalk => {
                const roles = try d.strings(arena, 0, row);
                const contents = try d.strings(arena, 1, row);
                if (roles.len != contents.len or roles.len < 2) return invalid("smoltalk messages");
                const messages = try arena.alloc(mod.Message, roles.len);
                for (messages, roles, contents) |*m, role, content| {
                    m.* = .{ .role = std.meta.stringToEnum(mod.Role, role) orelse return invalid("smoltalk role"), .content = .{ .text = content } };
                }
                return .{ .messages = messages };
            },
            .mmlu => {
                const choices = try d.strings(arena, 1, row);
                const answer = d.int(2, row) orelse return invalid("mmlu answer");
                if (choices.len != 4 or answer < 0 or answer >= 4) return invalid("mmlu choices");
                return multipleChoice(arena, d.string(0, row) orelse "", &mmlu_letters, choices, mmlu_letters[@intCast(answer)]);
            },
            .arc_easy, .arc_challenge => {
                const choices = try d.strings(arena, 1, row);
                const labels = try d.strings(arena, 2, row);
                const answer = d.string(3, row) orelse return invalid("arc answer");
                if (choices.len != labels.len) return invalid("arc choices");
                for (labels) |l| {
                    if (std.mem.eql(u8, l, answer)) break;
                } else return invalid("arc answer not among the labels");
                return multipleChoice(arena, d.string(0, row) orelse "", labels, choices, answer);
            },
            .gsm8k => {
                const question = d.string(0, row) orelse return invalid("gsm8k question");
                const answer = d.string(1, row) orelse return invalid("gsm8k answer");
                const messages = try arena.alloc(mod.Message, 2);
                messages[0] = .{ .role = .user, .content = .{ .text = question } };
                messages[1] = .{ .role = .assistant, .content = .{ .parts = try toolParts(arena, answer) } };
                return .{ .messages = messages };
            },
        }
    }

    /// A categorical example's answer letters (`conversation['letters']`).
    pub fn letters(self: *const Self, arena: std.mem.Allocator, index: usize) ![]const []const u8 {
        return switch (self.kind) {
            .mmlu => &mmlu_letters,
            .arc_easy, .arc_challenge => self.data.strings(arena, 2, self.start + index * self.step),
            .smoltalk, .gsm8k => error.NotCategorical,
        };
    }

    /// Scores a response (nanochat's `evaluate`): the letter for multiple
    /// choice, the `#### <number>` answer for GSM8K.
    ///
    /// Parameters:
    /// - `self`: the task.
    /// - `arena`: scratch.
    /// - `index`: the example.
    /// - `response`: the assistant's reply.
    ///
    /// Return: whether it is correct; `error.NotEvaluable` for SmolTalk.
    pub fn evaluate(self: *const Self, arena: std.mem.Allocator, index: usize, response: []const u8) !bool {
        const conv = try self.conversation(arena, index);
        const last = conv.messages[conv.messages.len - 1];
        switch (self.kind) {
            .mmlu, .arc_easy, .arc_challenge => return std.mem.eql(u8, response, last.content.text),
            .gsm8k => {
                const parts = last.content.parts;
                const reference = extractAnswer(arena, parts[parts.len - 1].text) catch null;
                const predicted = extractAnswer(arena, response) catch null;
                if (reference == null or predicted == null) return reference == null and predicted == null;
                return std.mem.eql(u8, reference.?, predicted.?);
            },
            .smoltalk => return error.NotEvaluable,
        }
    }

    /// GSM8K's `extract_answer`: the first `#### -?[0-9.,]+`, commas removed.
    ///
    /// Parameters:
    /// - `arena`: holds the result.
    /// - `text`: a completion.
    ///
    /// Return: the number's text, or null when there is none.
    pub fn extractAnswer(arena: std.mem.Allocator, text: []const u8) !?[]const u8 {
        var from: usize = 0;
        while (std.mem.indexOfPos(u8, text, from, "#### ")) |at| : (from = at + 1) {
            var i = at + 5;
            if (i < text.len and text[i] == '-') i += 1;
            const digits = i;
            while (i < text.len and (std.ascii.isDigit(text[i]) or text[i] == '.' or text[i] == ',')) i += 1;
            if (i == digits) continue;
            return try std.mem.replaceOwned(u8, arena, text[at + 5 .. i], ",", "");
        }
        return null;
    }

    /// `render_mc`: the question, `- choice=letter` lines, the instruction;
    /// the assistant answers with the letter.
    fn multipleChoice(arena: std.mem.Allocator, question: []const u8, choice_letters: []const []const u8, choices: []const []const u8, answer: []const u8) !mod.Conversation {
        var text: std.ArrayList(u8) = .empty;
        try text.print(arena, "Multiple Choice question: {s}\n", .{question});
        for (choice_letters, choices) |letter, choice| try text.print(arena, "- {s}={s}\n", .{ choice, letter });
        try text.appendSlice(arena, "\nRespond only with the letter of the correct answer.");
        const messages = try arena.alloc(mod.Message, 2);
        messages[0] = .{ .role = .user, .content = .{ .text = text.items } };
        messages[1] = .{ .role = .assistant, .content = .{ .text = answer } };
        return .{ .messages = messages };
    }

    /// GSM8K's answer split on `<<expr=result>>` calculator calls
    /// (`re.split(r'(<<[^>]+>>)')`, empty pieces kept).
    fn toolParts(arena: std.mem.Allocator, answer: []const u8) ![]const mod.MessagePart {
        var parts: std.ArrayList(mod.MessagePart) = .empty;
        var text_start: usize = 0;
        var i: usize = 0;
        while (i + 1 < answer.len) {
            if (!std.mem.startsWith(u8, answer[i..], "<<")) {
                i += 1;
                continue;
            }
            var j = i + 2;
            while (j < answer.len and answer[j] != '>') j += 1;
            if (j == i + 2 or !std.mem.startsWith(u8, answer[j..], ">>")) {
                i += 1;
                continue;
            }
            try parts.append(arena, .{ .kind = .text, .text = answer[text_start..i] });
            const inner = answer[i + 2 .. j];
            const eq = std.mem.lastIndexOfScalar(u8, inner, '=');
            try parts.append(arena, .{ .kind = .python, .text = if (eq) |e| inner[0..e] else inner });
            try parts.append(arena, .{ .kind = .python_output, .text = if (eq) |e| inner[e + 1 ..] else "" });
            i = j + 2;
            text_start = i;
        }
        try parts.append(arena, .{ .kind = .text, .text = answer[text_start..] });
        return parts.items;
    }

    fn invalid(what: []const u8) error{InvalidTaskRow} {
        log.warn("malformed task row: {s}", .{what});
        return error.InvalidTaskRow;
    }
};

/// nanochat's `TaskMixture`: every example of every task (a task listed twice
/// counts twice), in an order shuffled by Python's `random.Random(42)`.
pub const TaskMixture = struct {
    const Self = @This();

    pub const Entry = struct { task: u32, index: u32 };

    allocator: std.mem.Allocator,
    tasks: []const *const Task,
    entries: []Entry,

    /// Builds the shuffled index.
    ///
    /// Parameters:
    /// - `allocator`: owns the index.
    /// - `tasks`: the tasks (borrowed; may repeat).
    ///
    /// Return: the mixture; allocation errors.
    pub fn init(allocator: std.mem.Allocator, tasks: []const *const Task) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var entries: std.ArrayList(Entry) = .empty;
        errdefer entries.deinit(allocator);
        for (tasks, 0..) |t, ti| {
            for (0..t.len()) |i| try entries.append(allocator, .{ .task = @intCast(ti), .index = @intCast(i) });
        }
        var rng = mod.PythonRandom.init(42);
        rng.shuffle(Entry, entries.items);
        return Self{ .allocator = allocator, .tasks = tasks, .entries = try entries.toOwnedSlice(allocator) };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.entries);
    }

    pub fn len(self: *const Self) usize {
        return self.entries.len;
    }

    /// The `index`-th conversation of the mixture.
    pub fn conversation(self: *const Self, arena: std.mem.Allocator, index: usize) !mod.Conversation {
        const e = self.entries[index];
        return self.tasks[e.task].conversation(arena, e.index);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "tasks render the conversations, letters and scores of nanochat's tasks" {
    const allocator = std.testing.allocator;
    const root = mod.build_options.source_root ++ "/testdata";
    const storage = mod.Storage.init(allocator, std.testing.io);
    var tok = try mod.TorchImport.loadTokenizer(allocator, storage, root ++ "/nanochat_base/tokenizer/tokenizer.pkl");
    defer tok.deinit();
    const config = mod.Config{ .allocator = allocator, .base_dir = root ++ "/task_base", .nanochat_dir = "/nonexistent-nanochat", .data_url = "http://unused" };
    const bytes = try storage.read(root ++ "/tasks.json");
    defer allocator.free(bytes);
    const Row = struct {
        ids: []const u32,
        mask: []const u8,
        completion: []const u32,
        letters: ?[]const []const u8 = null,
        correct: ?[]const bool = null,
        parts: ?[]const [2][]const u8 = null,
        evaluate: ?[]const u8 = null,
    };
    const Expected = struct {
        tasks: std.json.ArrayHashMap(struct { len: usize, rows: []const Row }),
        extract: []const [2]?[]const u8,
        mixture: []const [2]u32,
    };
    var parsed = try std.json.parseFromSlice(Expected, allocator, bytes, .{});
    defer parsed.deinit();
    const want = parsed.value;

    var smoltalk = try Task.open(allocator, std.testing.io, &config, .smoltalk, "test", null);
    defer smoltalk.deinit();
    var mmlu = try Task.open(allocator, std.testing.io, &config, .mmlu, "test", null);
    defer mmlu.deinit();
    var arc = try Task.open(allocator, std.testing.io, &config, .arc_easy, "test", null);
    defer arc.deinit();
    var gsm8k = try Task.open(allocator, std.testing.io, &config, .gsm8k, "test", null);
    defer gsm8k.deinit();
    gsm8k.start = 2;
    gsm8k.stop = 25;
    gsm8k.step = 3;
    const answers = [_][]const u8{ "so 5 + 3 = 8\n#### 8", "#### -1,234.5", "no marker 42", "#### 7\n#### 9", "####8", "#### .5," };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]*const Task{ &smoltalk, &mmlu, &arc, &gsm8k }) |task| {
        const expected = want.tasks.map.get(task.kind.name()).?;
        try std.testing.expectEqual(expected.len, task.len());
        for (expected.rows, 0..) |row, i| {
            const conv = try task.conversation(a, i);
            var r = try tok.renderConversation(allocator, conv, 2048);
            defer r.deinit(allocator);
            std.testing.expectEqualSlices(u32, row.ids, r.ids.items) catch |err| {
                std.debug.print("{s} row {d}\n", .{ task.kind.name(), i });
                return err;
            };
            try std.testing.expectEqualSlices(u8, row.mask, r.mask.items);
            const completion = try tok.renderForCompletion(allocator, conv);
            defer allocator.free(completion);
            try std.testing.expectEqualSlices(u32, row.completion, completion);
            if (row.letters) |letters| {
                const got = try task.letters(a, i);
                try std.testing.expectEqual(letters.len, got.len);
                for (letters, got, row.correct.?) |l, g, correct| {
                    try std.testing.expectEqualStrings(l, g);
                    try std.testing.expectEqual(correct, try task.evaluate(a, i, l));
                }
            }
            if (row.evaluate) |scores| {
                for (answers, scores) |answer, score| try std.testing.expectEqual(score == 1, try task.evaluate(a, i, answer));
                try std.testing.expectEqual(row.parts.?.len, conv.messages[1].content.parts.len);
            }
        }
    }
    for (want.extract) |case| {
        const got = try Task.extractAnswer(a, case[0].?);
        if (case[1]) |w| try std.testing.expectEqualStrings(w, got.?) else try std.testing.expect(got == null);
    }
    var mixture = try TaskMixture.init(allocator, &.{ &smoltalk, &mmlu, &mmlu, &gsm8k });
    defer mixture.deinit();
    try std.testing.expectEqual(want.mixture.len, mixture.len());
    for (want.mixture, mixture.entries) |w, e| {
        try std.testing.expectEqual(w[0], e.task);
        try std.testing.expectEqual(w[1], e.index);
    }
}
