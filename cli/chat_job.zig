const std = @import("std");
const log = std.log.scoped(.zignanogpt_chat_job);
const cli = @import("module.zig");
const mod = cli.nanogpt;

pub const ChatState = enum { idle, loading, ready, replying, failed };

pub const ChatRole = enum { user, assistant, note };

/// One transcript entry.
pub const ChatTurn = struct {
    role: ChatRole,
    text: std.ArrayList(u8) = .empty,
};

/// What the console draws, copied out under the lock.
pub const ChatSnapshot = struct {
    state: ChatState,
    failure: ?[]const u8,
    label: []const u8,
    turns: []const struct { role: ChatRole, text: []const u8 },
};

/// The console's chat: loads a model and generates replies on a worker
/// thread (one thread per load or reply), streaming the text into a transcript
/// the console draws. Everything the worker touches is behind `mutex`.
pub const ChatJob = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    state: ChatState = .idle,
    failure: ?[]const u8 = null,
    label: std.ArrayList(u8) = .empty,
    turns: std.ArrayList(ChatTurn) = .empty,
    thread: ?std.Thread = null,
    stop_requested: std.atomic.Value(bool) = .init(false),
    /// Streams the reply into the last turn.
    writer: std.Io.Writer,
    write_buffer: [256]u8 = undefined,

    backend: ?mod.Backend = null,
    loaded: ?*mod.LoadedModel = null,
    session: ?mod.ChatSession = null,
    /// The load's arguments, owned until the next load.
    arena: std.heap.ArenaAllocator,
    pending: std.ArrayList(u8) = .empty,

    /// Creates an idle chat; it must stay at this address (the writer points into it).
    ///
    /// Parameters:
    /// - `self`: the storage.
    /// - `allocator`: owns the transcript and the model.
    /// - `io`: locks.
    ///
    /// Return: nothing.
    pub fn init(self: *Self, allocator: std.mem.Allocator, io: std.Io) void {
        self.* = .{ .allocator = allocator, .io = io, .writer = undefined, .arena = std.heap.ArenaAllocator.init(allocator) };
        self.writer = .{ .vtable = &.{ .drain = drain }, .buffer = &self.write_buffer };
    }

    /// Stops a reply, waits for the worker, then frees everything.
    pub fn deinit(self: *Self) void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        self.requestStop();
        self.join();
        self.unload();
        for (self.turns.items) |*t| t.text.deinit(self.allocator);
        self.turns.deinit(self.allocator);
        self.label.deinit(self.allocator);
        self.pending.deinit(self.allocator);
        self.arena.deinit();
    }

    /// Loads a model on the worker (dropping any loaded one and the transcript).
    ///
    /// Parameters:
    /// - `self`: the chat, not busy.
    /// - `process`: process state.
    /// - `args`: `chat`'s options (copied).
    ///
    /// Return: nothing; `error.ChatBusy`, thread errors.
    pub fn load(self: *Self, process: std.process.Init, args: []const []const u8) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (self.busy()) return error.ChatBusy;
        self.join();
        self.unload();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        const owned = try a.alloc([]const u8, args.len);
        for (args, owned) |src, *dst| dst.* = try a.dupe(u8, src);
        self.lock();
        for (self.turns.items) |*t| t.text.deinit(self.allocator);
        self.turns.clearRetainingCapacity();
        self.label.clearRetainingCapacity();
        self.failure = null;
        self.state = .loading;
        self.unlock();
        self.thread = try std.Thread.spawn(.{}, loadWorker, .{ self, process, owned });
    }

    /// Sends a user message; the reply streams into the transcript.
    ///
    /// Parameters:
    /// - `self`: a ready chat.
    /// - `text`: the message (copied).
    ///
    /// Return: nothing; `error.ChatNotReady`, allocation and thread errors.
    pub fn send(self: *Self, text: []const u8) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        if (self.currentState() != .ready) return error.ChatNotReady;
        self.join();
        self.pending.clearRetainingCapacity();
        try self.pending.appendSlice(self.allocator, text);
        self.lock();
        defer self.unlock();
        try self.addTurnLocked(.user, text);
        try self.addTurnLocked(.assistant, "");
        self.stop_requested.store(false, .release);
        self.state = .replying;
        self.thread = std.Thread.spawn(.{}, replyWorker, .{self}) catch |err| {
            self.state = .ready;
            return err;
        };
    }

    /// Forgets the conversation (the model stays loaded).
    pub fn clear(self: *Self) !void {
        if (self.currentState() != .ready) return error.ChatNotReady;
        self.join();
        self.lock();
        defer self.unlock();
        if (self.session) |*s| s.clear();
        for (self.turns.items) |*t| t.text.deinit(self.allocator);
        self.turns.clearRetainingCapacity();
        try self.addTurnLocked(.note, "Conversation cleared.");
    }

    /// Ends the reply being generated after its current token.
    pub fn requestStop(self: *Self) void {
        self.stop_requested.store(true, .release);
    }

    pub fn currentState(self: *Self) ChatState {
        self.lock();
        defer self.unlock();
        return self.state;
    }

    /// Whether a load or a reply is running.
    pub fn busy(self: *Self) bool {
        const s = self.currentState();
        return s == .loading or s == .replying;
    }

    /// A copy of the transcript, for drawing.
    pub fn snapshot(self: *Self, allocator: std.mem.Allocator) !ChatSnapshot {
        self.lock();
        defer self.unlock();
        const Turn = @typeInfo(@FieldType(ChatSnapshot, "turns")).pointer.child;
        const turns = try allocator.alloc(Turn, self.turns.items.len);
        for (self.turns.items, turns) |t, *d| d.* = .{ .role = t.role, .text = try allocator.dupe(u8, t.text.items) };
        return .{ .state = self.state, .failure = self.failure, .label = try allocator.dupe(u8, self.label.items), .turns = turns };
    }

    fn loadWorker(self: *Self, process: std.process.Init, args: []const []const u8) void {
        self.loadModel(process, args) catch |err| {
            self.unload();
            self.lock();
            defer self.unlock();
            self.state = .failed;
            self.failure = @errorName(err);
            return;
        };
        self.lock();
        defer self.unlock();
        self.state = .ready;
    }

    fn loadModel(self: *Self, process: std.process.Init, items: []const []const u8) !void {
        var args = try cli.Args.init(self.allocator, items);
        defer args.deinit();
        const settings = try cli.Chat.parse(&args);
        self.backend = try mod.Backend.init(self.allocator, process.io, .{ .threads = settings.threads });
        const loaded = try self.allocator.create(mod.LoadedModel);
        errdefer self.allocator.destroy(loaded);
        // Notes ("no sft model yet", "Loaded ...") land in the transcript.
        self.lock();
        self.addTurnLocked(.note, "") catch |err| log.debug("chat note [{t}]", .{err});
        self.unlock();
        try cli.Chat.open(loaded, process, &self.backend.?, settings, &self.writer);
        self.loaded = loaded;
        self.session = try mod.ChatSession.init(self.allocator, &loaded.model, &loaded.tokenizer, settings.options);
        self.lock();
        defer self.unlock();
        self.label.print(self.allocator, "{s} {s} step {d}", .{ @tagName(loaded.kind), loaded.tag, loaded.step }) catch |err| log.debug("chat label [{t}]", .{err});
    }

    fn replyWorker(self: *Self) void {
        const result = self.session.?.reply(self.pending.items, &self.writer, &self.stop_requested);
        self.writer.flush() catch |err| log.debug("chat flush [{t}]", .{err});
        self.lock();
        defer self.unlock();
        if (result) |_| {} else |err| switch (err) {
            error.SequenceTooLong => self.addTurnLocked(.note, "The conversation is too long for the model; type 'clear'.") catch |e| log.debug("chat note [{t}]", .{e}),
            else => self.addTurnLocked(.note, @errorName(err)) catch |e| log.debug("chat note [{t}]", .{e}),
        }
        self.state = .ready;
    }

    fn unload(self: *Self) void {
        if (self.session) |*s| s.deinit();
        self.session = null;
        if (self.loaded) |l| {
            l.deinit();
            self.allocator.destroy(l);
        }
        self.loaded = null;
        if (self.backend) |*b| b.deinit();
        self.backend = null;
    }

    fn join(self: *Self) void {
        if (self.thread) |t| t.join();
        self.thread = null;
    }

    fn addTurnLocked(self: *Self, role: ChatRole, text: []const u8) !void {
        var turn = ChatTurn{ .role = role };
        try turn.text.appendSlice(self.allocator, text);
        errdefer turn.text.deinit(self.allocator);
        try self.turns.append(self.allocator, turn);
    }

    /// `std.Io.Writer` sink: appends to the last turn.
    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *Self = @alignCast(@fieldParentPtr("writer", w));
        self.lock();
        defer self.unlock();
        self.appendLocked(w.buffered());
        w.end = 0;
        var n: usize = 0;
        for (data[0 .. data.len - 1]) |d| {
            self.appendLocked(d);
            n += d.len;
        }
        const last = data[data.len - 1];
        for (0..splat) |_| self.appendLocked(last);
        return n + last.len * splat;
    }

    fn appendLocked(self: *Self, bytes: []const u8) void {
        if (bytes.len == 0 or self.turns.items.len == 0) return;
        const turn = &self.turns.items[self.turns.items.len - 1];
        turn.text.appendSlice(self.allocator, bytes) catch |err| log.debug("chat text [{t}]", .{err});
    }

    fn lock(self: *Self) void {
        self.mutex.lockUncancelable(self.io);
    }

    fn unlock(self: *Self) void {
        self.mutex.unlock(self.io);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "chat job loads an imported model, streams a reply and clears" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const base = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    {
        var backend = try mod.Backend.init(allocator, std.testing.io, .{});
        defer backend.deinit();
        const imported = try mod.TorchImport.importCheckpoint(allocator, &backend, mod.Storage.init(allocator, std.testing.io), mod.build_options.source_root ++ "/testdata/nanochat_base", base, .base, null, null);
        allocator.free(imported.tag);
    }
    var env = std.process.Environ.Map.init(allocator);
    defer env.deinit();
    try env.put(mod.Config.base_dir_env, base);
    try env.put(mod.Config.nanochat_dir_env, "/nonexistent-nanochat");
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const process: std.process.Init = .{ .minimal = .{ .environ = .empty, .args = undefined }, .arena = &arena, .gpa = allocator, .io = std.testing.io, .environ_map = &env, .preopens = undefined };

    var chat: ChatJob = undefined;
    chat.init(allocator, std.testing.io);
    defer chat.deinit();
    try std.testing.expectError(error.ChatNotReady, chat.send("hi"));
    try chat.load(process, &.{ "-t", "0", "--max-tokens", "4" });
    chat.join();
    try std.testing.expectEqual(ChatState.ready, chat.currentState());
    try chat.send("Hello");
    chat.join();
    try std.testing.expectEqual(ChatState.ready, chat.currentState());
    const snap = try chat.snapshot(arena.allocator());
    try std.testing.expectEqualStrings("base d2 step 5", snap.label);
    // note (fallback + loaded), user, assistant
    try std.testing.expectEqual(@as(usize, 3), snap.turns.len);
    try std.testing.expect(std.mem.find(u8, snap.turns[0].text, "No sft model") != null);
    try std.testing.expectEqualStrings("Hello", snap.turns[1].text);
    try std.testing.expect(snap.turns[2].text.len > 0);
    try chat.clear();
    try std.testing.expectEqual(@as(usize, 1), (try chat.snapshot(arena.allocator())).turns.len);

    // A missing checkpoint fails the load cleanly.
    try chat.load(process, &.{ "-i", "rl" });
    chat.join();
    try std.testing.expectEqual(ChatState.failed, chat.currentState());
    try std.testing.expectEqualStrings("NoCheckpoint", chat.failure.?);
}
