const std = @import("std");
const log = std.log.scoped(.zignanogpt_sft_data);
const mod = @import("module.zig");

/// `chat_sft.py`'s data: the training mixture (SmolTalk, MMLU auxiliary_train
/// `mmlu_epochs` times, GSM8K train `gsm8k_epochs` times), the validation
/// mixture (SmolTalk test, the first 5,200 MMLU test rows, the first 420 GSM8K
/// test rows) and the ChatCORE tasks (ARC-Easy, ARC-Challenge, MMLU, GSM8K tests).
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
    /// - `out`: download progress, or null.
    ///
    /// Return: the data (`destroy` it); task errors.
    pub fn openStandard(allocator: std.mem.Allocator, io: std.Io, config: *const mod.Config, mmlu_epochs: usize, gsm8k_epochs: usize, out: ?*std.Io.Writer) !*Self {
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

    fn freeOwned(self: *Self) void {
        for (self.owned.items) |t| {
            t.deinit();
            self.allocator.destroy(t);
        }
        self.owned.deinit(self.allocator);
    }
};
