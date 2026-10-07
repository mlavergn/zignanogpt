const std = @import("std");
const log = std.log.scoped(.zignanogpt_sft_data);
const mod = @import("module.zig");

/// `chat_sft.py`'s data: the training mixture (SmolTalk, MMLU auxiliary_train
/// `mmlu_epochs` times, GSM8K train `gsm8k_epochs` times), the validation
/// mixture (SmolTalk test, the first 5,200 MMLU test rows, the first 420 GSM8K
/// test rows) and the ChatCORE tasks (ARC-Easy, ARC-Challenge, MMLU, GSM8K tests).
/// Your own conversations (a `ConversationFile`) can join the training mixture.
/// Heap-allocated: the mixtures point at its task lists.
pub const SftData = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    /// Tasks this data opened (and frees).
    owned: std.ArrayList(*mod.Task) = .empty,
    train_tasks: []const *const mod.Task = &.{},
    val_tasks: []const *const mod.Task = &.{},
    /// Evaluated for ChatCORE; opened on first use by `chatcoreTasks`.
    chatcore: []const *const mod.Task = &.{},
    chatcore_loaded: bool = false,
    /// Your conversations, when given (the custom task borrows them).
    conversations: ?*mod.ConversationFile = null,
    config: ?*const mod.Config = null,
    io: ?std.Io = null,
    train: mod.TaskMixture,
    val: mod.TaskMixture,

    /// Opens (downloading on first use) the standard mixtures.
    ///
    /// Parameters:
    /// - `allocator`: owns everything.
    /// - `io`: file and network access.
    /// - `config`: the base directories (outlives the data).
    /// - `mmlu_epochs`: copies of MMLU auxiliary_train in the training mixture.
    /// - `gsm8k_epochs`: copies of GSM8K train.
    /// - `extra`: your conversations (a JSONL path) and their copies in training, or null.
    /// - `out`: download progress, or null.
    ///
    /// Return: the data (`destroy` it); task and conversation-file errors.
    pub fn openStandard(allocator: std.mem.Allocator, io: std.Io, config: *const mod.Config, mmlu_epochs: usize, gsm8k_epochs: usize, extra: ?Extra, out: ?*std.Io.Writer) !*Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        const self = try allocator.create(Self);
        self.* = .{ .allocator = allocator, .train = undefined, .val = undefined, .config = config, .io = io };
        errdefer allocator.destroy(self);
        errdefer self.freeOwned();
        const smoltalk = try self.open(io, config, .smoltalk, "train", out);
        const mmlu = try self.open(io, config, .mmlu, "auxiliary_train", out);
        const gsm8k = try self.open(io, config, .gsm8k, "train", out);
        var train: std.ArrayList(*const mod.Task) = .empty;
        defer train.deinit(allocator);
        try train.append(allocator, smoltalk);
        try train.appendNTimes(allocator, mmlu, mmlu_epochs);
        try train.appendNTimes(allocator, gsm8k, gsm8k_epochs);
        if (extra) |e| try train.appendNTimes(allocator, try self.openConversations(io, e.path), e.epochs);

        const smoltalk_test = try self.open(io, config, .smoltalk, "test", out);
        const mmlu_test = try self.open(io, config, .mmlu, "test", out);
        mmlu_test.stop = 5200;
        const gsm8k_test = try self.open(io, config, .gsm8k, "test", out);
        gsm8k_test.stop = 420;
        try self.setMixtures(train.items, &.{ smoltalk_test, mmlu_test, gsm8k_test });
        return self;
    }

    /// Builds data from given tasks (borrowed), e.g. for tests.
    ///
    /// Parameters:
    /// - `allocator`: owns the mixtures.
    /// - `train`: the training tasks (repeat one to oversample it).
    /// - `val`: the validation tasks.
    /// - `chatcore`: the ChatCORE tasks.
    ///
    /// Return: the data (`destroy` it); allocation errors.
    pub fn fromTasks(allocator: std.mem.Allocator, train: []const *const mod.Task, val: []const *const mod.Task, chatcore: []const *const mod.Task) !*Self {
        const self = try allocator.create(Self);
        self.* = .{ .allocator = allocator, .train = undefined, .val = undefined };
        errdefer allocator.destroy(self);
        self.chatcore = try allocator.dupe(*const mod.Task, chatcore);
        self.chatcore_loaded = true;
        errdefer allocator.free(self.chatcore);
        try self.setMixtures(train, val);
        return self;
    }

    /// Conversations from your own file, mixed into training `epochs` times.
    pub const Extra = struct { path: []const u8, epochs: usize = 1 };

    pub fn destroy(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.val.deinit();
        self.train.deinit();
        self.allocator.free(self.val_tasks);
        self.allocator.free(self.train_tasks);
        self.allocator.free(self.chatcore);
        self.freeOwned();
        self.allocator.destroy(self);
    }

    /// The training rows that come from your conversations (all epochs).
    pub fn customRows(self: *const Self) usize {
        var n: usize = 0;
        for (self.train_tasks) |t| {
            if (t.kind == .custom) n += t.len();
        }
        return n;
    }

    /// The ChatCORE tasks, opened on first use.
    pub fn chatcoreTasks(self: *Self, out: ?*std.Io.Writer) ![]const *const mod.Task {
        if (!self.chatcore_loaded) {
            const config = self.config orelse return error.NoTaskConfig;
            var list: std.ArrayList(*const mod.Task) = .empty;
            defer list.deinit(self.allocator);
            for (mod.chat_eval_tasks) |kind| try list.append(self.allocator, try self.open(self.io.?, config, kind, "test", out));
            self.chatcore = try list.toOwnedSlice(self.allocator);
            self.chatcore_loaded = true;
        }
        return self.chatcore;
    }

    fn setMixtures(self: *Self, train: []const *const mod.Task, val: []const *const mod.Task) !void {
        self.train_tasks = try self.allocator.dupe(*const mod.Task, train);
        errdefer self.allocator.free(self.train_tasks);
        self.val_tasks = try self.allocator.dupe(*const mod.Task, val);
        errdefer self.allocator.free(self.val_tasks);
        self.train = try mod.TaskMixture.init(self.allocator, self.train_tasks);
        errdefer self.train.deinit();
        self.val = try mod.TaskMixture.init(self.allocator, self.val_tasks);
    }

    fn open(self: *Self, io: std.Io, config: *const mod.Config, kind: mod.TaskKind, split: []const u8, out: ?*std.Io.Writer) !*mod.Task {
        const task = try self.allocator.create(mod.Task);
        errdefer self.allocator.destroy(task);
        task.* = try mod.Task.open(self.allocator, io, config, kind, split, out);
        errdefer task.deinit();
        try self.owned.append(self.allocator, task);
        return task;
    }

    /// Loads the conversation file and a task over it (both owned).
    fn openConversations(self: *Self, io: std.Io, path: []const u8) !*mod.Task {
        const file = try self.allocator.create(mod.ConversationFile);
        errdefer self.allocator.destroy(file);
        file.* = try mod.ConversationFile.load(self.allocator, mod.Storage.init(self.allocator, io), path);
        self.conversations = file;
        const task = try self.allocator.create(mod.Task);
        errdefer self.allocator.destroy(task);
        task.* = mod.Task.fromConversations(self.allocator, file);
        try self.owned.append(self.allocator, task);
        return task;
    }

    fn freeOwned(self: *Self) void {
        for (self.owned.items) |t| {
            t.deinit();
            self.allocator.destroy(t);
        }
        self.owned.deinit(self.allocator);
        if (self.conversations) |file| {
            file.deinit();
            self.allocator.destroy(file);
            self.conversations = null;
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "your conversations join the training mixture and render for training" {
    const allocator = std.testing.allocator;
    const root = mod.build_options.source_root ++ "/testdata";
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    const storage = mod.Storage.init(allocator, std.testing.io);
    const path = try std.fs.path.join(allocator, &.{ base, "identity.jsonl" });
    defer allocator.free(path);
    try storage.write(path,
        \\[{"role": "user", "content": "What is your name?"}, {"role": "assistant", "content": "I am zignanogpt."}]
        \\[{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Who made you?"}, {"role": "assistant", "content": "Someone with a Mac."}, {"role": "user", "content": "Thanks"}, {"role": "assistant", "content": "Sure."}]
        \\
    );
    const config = mod.Config{ .allocator = allocator, .base_dir = base, .nanochat_dir = root ++ "/task_base", .data_url = "http://unused" };
    const plain = try SftData.openStandard(allocator, std.testing.io, &config, 1, 1, null, null);
    defer plain.destroy();
    const data = try SftData.openStandard(allocator, std.testing.io, &config, 1, 1, .{ .path = path, .epochs = 3 }, null);
    defer data.destroy();
    try std.testing.expectEqual(@as(usize, 6), data.customRows());
    try std.testing.expectEqual(plain.train.len() + 6, data.train.len());
    try std.testing.expectEqual(plain.val.len(), data.val.len());

    // Every custom row renders: the assistant's tokens are the ones trained on.
    var tok = try mod.TorchImport.loadTokenizer(allocator, storage, root ++ "/nanochat_base/tokenizer/tokenizer.pkl");
    defer tok.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var found: usize = 0;
    for (0..data.train.len()) |i| {
        const e = data.train.entries[i];
        if (data.train.tasks[e.task].kind != .custom) continue;
        found += 1;
        var rendered = try tok.renderConversation(allocator, try data.train.conversation(arena.allocator(), i), 2048);
        defer rendered.deinit(allocator);
        var trained: usize = 0;
        for (rendered.mask.items) |m| trained += m;
        try std.testing.expect(trained > 0 and trained < rendered.ids.items.len);
    }
    try std.testing.expectEqual(@as(usize, 6), found);

    // A bad line fails the load and frees what was opened.
    try storage.write(path, "[{\"role\": \"user\", \"content\": \"no reply\"}]\n");
    try std.testing.expectError(error.InvalidConversation, SftData.openStandard(allocator, std.testing.io, &config, 1, 1, .{ .path = path }, null));
}
