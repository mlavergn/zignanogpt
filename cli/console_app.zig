const std = @import("std");
const log = std.log.scoped(.zignanogpt_console_app);
const cli = @import("module.zig");
const mod = cli.nanogpt;
const vaxis = cli.vaxis;
const vxfw = vaxis.vxfw;
const Span = vxfw.RichText.TextSpan;
const Style = cli.TuiStyle;

/// Which pane the keys go to.
pub const Focus = enum { nav, form, chat };

/// A job to start as soon as the console is up (`zignanogpt train` on a terminal).
pub const ConsoleStart = struct {
    command: cli.Command,
    args: []const []const u8,
};

/// One key and what it does, for the status line.
const Binding = struct { key: []const u8, does: []const u8 };

/// Eighths of a cell, bottom-aligned, for the loss curve.
const blocks = [_][]const u8{ " ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█" };
const nav_width = 26;
const label_width = 18;
const operations = cli.Operation.all;

/// The console: operations on the left, the selected one's form, output and
/// (for training) live progress on the right, a status line of live keys
/// below. One job runs at a time on a worker thread; the screen redraws on a
/// timer while it does.
pub const ConsoleApp = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    init: std.process.Init,
    job: *cli.Job,
    chat: *cli.ChatJob,
    overview: cli.Overview,
    nav: usize = 0,
    focus: Focus = .nav,
    /// The selected form row; `fields.len` is the Run row.
    field: usize = 0,
    editing: bool = false,
    edit: std.ArrayList(u8) = .empty,
    values: [operations.len][cli.Operation.max_fields]std.ArrayList(u8),
    notice: ?[]const u8 = null,
    notice_problem: bool = false,
    quit_armed: bool = false,
    last_state: cli.JobState = .idle,
    pending_start: ?ConsoleStart = null,
    /// The chat page's message being typed.
    input: std.ArrayList(u8) = .empty,
    last_chat: cli.ChatState = .idle,

    /// Creates the console with every form at its defaults.
    ///
    /// Parameters:
    /// - `allocator`: owns the form values.
    /// - `init`: process state for jobs and the overview.
    /// - `job`: the job slot (stable address).
    /// - `chat`: the chat (stable address).
    ///
    /// Return: the app; allocation errors.
    pub fn create(allocator: std.mem.Allocator, init: std.process.Init, job: *cli.Job, chat: *cli.ChatJob) !Self {
        var self = Self{ .allocator = allocator, .init = init, .job = job, .chat = chat, .overview = cli.Overview.init(allocator), .values = undefined };
        for (&self.values, operations) |*row, op| {
            for (row, 0..) |*v, i| {
                v.* = .empty;
                if (i < op.fields.len) try v.appendSlice(allocator, op.fields[i].default);
            }
        }
        return self;
    }

    pub fn deinit(self: *Self) void {
        for (&self.values) |*row| for (row) |*v| v.deinit(self.allocator);
        self.edit.deinit(self.allocator);
        self.input.deinit(self.allocator);
        self.overview.deinit();
    }

    /// Selects `start`'s operation, fills its form from the arguments, and runs it.
    ///
    /// Parameters:
    /// - `self`: the app.
    /// - `start`: the command and its arguments.
    ///
    /// Return: nothing; allocation and job errors.
    pub fn startWith(self: *Self, start: ConsoleStart) !void {
        const index = cli.Operation.indexOf(start.command) orelse return;
        const op = operations[index];
        self.nav = index;
        self.focus = .form;
        self.field = op.fields.len;
        for (op.fields, 0..) |f, i| {
            if (f.kind == .toggle) self.values[index][i].clearRetainingCapacity();
        }
        // Known flags fill their fields; everything else goes to "more options".
        var more: std.ArrayList(u8) = .empty;
        defer more.deinit(self.allocator);
        var i: usize = 0;
        while (i < start.args.len) : (i += 1) {
            const arg = start.args[i];
            const name = if (std.mem.startsWith(u8, arg, "--")) arg[2..] else "";
            const eq = std.mem.indexOfScalar(u8, name, '=');
            const flag = if (eq) |e| name[0..e] else name;
            const slot = for (op.fields, 0..) |f, k| {
                if (f.flag.len > 0 and std.mem.eql(u8, f.flag, flag)) break k;
            } else null;
            if (slot) |k| {
                const v = &self.values[index][k];
                v.clearRetainingCapacity();
                if (op.fields[k].kind == .toggle) {
                    try v.appendSlice(self.allocator, "yes");
                } else if (eq) |e| {
                    try v.appendSlice(self.allocator, name[e + 1 ..]);
                } else if (i + 1 < start.args.len) {
                    i += 1;
                    try v.appendSlice(self.allocator, start.args[i]);
                }
                continue;
            }
            if (more.items.len > 0) try more.append(self.allocator, ' ');
            try more.appendSlice(self.allocator, arg);
        }
        for (op.fields, 0..) |f, k| {
            if (f.flag.len == 0) {
                self.values[index][k].clearRetainingCapacity();
                try self.values[index][k].appendSlice(self.allocator, more.items);
            }
        }
        try self.run(index);
    }

    pub fn widget(self: *Self) vxfw.Widget {
        return .{ .userdata = self, .eventHandler = typeErasedEventHandler, .drawFn = typeErasedDrawFn };
    }

    // -------------------------------------------------------------------------
    // Events

    pub fn handleEvent(self: *Self, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
        switch (event) {
            .init => {
                self.overview.refresh(self.init);
                if (self.pending_start) |start| {
                    self.pending_start = null;
                    self.startWith(start) catch |err| self.setNotice(true, @errorName(err));
                }
                try ctx.tick(250, self.widget());
                ctx.redraw = true;
            },
            .tick => {
                const state = self.job.currentState();
                if (state != self.last_state) {
                    if (state != .running) self.overview.refresh(self.init);
                    self.last_state = state;
                }
                const chat = self.chat.currentState();
                if (chat != self.last_chat) {
                    // A finished load opens the conversation.
                    if (self.last_chat == .loading and chat == .ready and operations[self.nav].command == .chat and self.focus != .nav) self.focus = .chat;
                    self.last_chat = chat;
                }
                ctx.redraw = true;
                const interval: u32 = if (chat == .replying) 100 else if (state == .running or chat == .loading) 250 else 1000;
                try ctx.tick(interval, self.widget());
            },
            .key_press => |key| {
                try self.handleKey(ctx, key);
                ctx.consumeAndRedraw();
            },
            else => {},
        }
    }

    fn handleKey(self: *Self, ctx: *vxfw.EventContext, key: vaxis.Key) !void {
        if (key.matches('c', .{ .ctrl = true })) {
            ctx.quit = true;
            return;
        }
        if (self.editing) return self.handleEditKey(key);
        if (self.focus == .chat) return self.handleChatKey(key);
        const armed = self.quit_armed;
        self.quit_armed = false;
        self.notice = null;
        const op = operations[self.nav];
        if (key.matches('s', .{})) return self.stopJob();
        switch (self.focus) {
            .nav => {
                if (key.matches(vaxis.Key.up, .{}) or key.matches('k', .{})) {
                    self.nav = if (self.nav == 0) operations.len - 1 else self.nav - 1;
                    self.field = 0;
                } else if (key.matches(vaxis.Key.down, .{}) or key.matches('j', .{})) {
                    self.nav = (self.nav + 1) % operations.len;
                    self.field = 0;
                } else if (key.matches(vaxis.Key.right, .{}) or key.matches(vaxis.Key.enter, .{}) or key.matches(vaxis.Key.tab, .{})) {
                    if (op.command == .chat and self.chatOpen()) {
                        self.focus = .chat;
                    } else if (op.command != null and op.phase() == null) self.focus = .form;
                } else if (key.matches('r', .{})) {
                    if (op.command == null) self.overview.refresh(self.init) else try self.run(self.nav);
                } else if (key.matches('q', .{}) or key.matches(vaxis.Key.escape, .{})) {
                    return self.quit(ctx, armed);
                }
            },
            .form => {
                const rows = op.fields.len + 1;
                if (key.matches(vaxis.Key.up, .{}) or key.matches('k', .{})) {
                    self.field = if (self.field == 0) rows - 1 else self.field - 1;
                } else if (key.matches(vaxis.Key.down, .{}) or key.matches('j', .{}) or key.matches(vaxis.Key.tab, .{})) {
                    self.field = (self.field + 1) % rows;
                } else if (key.matches(vaxis.Key.left, .{}) or key.matches(vaxis.Key.escape, .{})) {
                    self.focus = if (op.command == .chat and self.chatOpen()) .chat else .nav;
                } else if (key.matches('r', .{})) {
                    try self.run(self.nav);
                } else if (key.matches(vaxis.Key.enter, .{}) or key.matches(' ', .{})) {
                    if (self.field == op.fields.len) return self.run(self.nav);
                    const f = op.fields[self.field];
                    const value = &self.values[self.nav][self.field];
                    switch (f.kind) {
                        .toggle => {
                            const on = std.ascii.eqlIgnoreCase(value.items, "yes");
                            value.clearRetainingCapacity();
                            try value.appendSlice(self.allocator, if (on) "no" else "yes");
                        },
                        .text => {
                            self.edit.clearRetainingCapacity();
                            try self.edit.appendSlice(self.allocator, value.items);
                            self.editing = true;
                        },
                    }
                } else if (key.matches('q', .{})) {
                    return self.quit(ctx, armed);
                }
            },
            .chat => unreachable,
        }
    }

    /// The chat page's keys: typing, enter sends (`quit`/`exit` leave,
    /// `clear` forgets the conversation), esc stops a reply or leaves.
    fn handleChatKey(self: *Self, key: vaxis.Key) !void {
        self.notice = null;
        const state = self.chat.currentState();
        if (key.matches(vaxis.Key.escape, .{})) {
            if (state == .replying) {
                self.chat.requestStop();
            } else self.focus = .nav;
        } else if (key.matches(vaxis.Key.tab, .{})) {
            self.focus = .form;
        } else if (key.matches(vaxis.Key.enter, .{})) {
            const text = std.mem.trim(u8, self.input.items, " \t");
            if (std.ascii.eqlIgnoreCase(text, "quit") or std.ascii.eqlIgnoreCase(text, "exit")) {
                self.input.clearRetainingCapacity();
                self.focus = .nav;
                return;
            }
            if (state != .ready) {
                self.setNotice(true, if (state == .replying) "wait for the reply (esc stops it)" else "load a model first (tab: settings)");
                return;
            }
            if (std.ascii.eqlIgnoreCase(text, "clear")) {
                try self.chat.clear();
            } else if (text.len > 0) {
                try self.chat.send(text);
            }
            self.input.clearRetainingCapacity();
        } else if (key.matches(vaxis.Key.backspace, .{})) {
            var n = self.input.items.len;
            while (n > 0) {
                n -= 1;
                if (self.input.items[n] & 0xC0 != 0x80) break;
            }
            self.input.shrinkRetainingCapacity(n);
        } else if (key.matches('u', .{ .ctrl = true })) {
            self.input.clearRetainingCapacity();
        } else if (key.text) |text| {
            try self.input.appendSlice(self.allocator, text);
        }
    }

    /// Whether the chat has a model (loading, ready or replying).
    fn chatOpen(self: *Self) bool {
        return switch (self.chat.currentState()) {
            .loading, .ready, .replying => true,
            .idle, .failed => false,
        };
    }

    fn handleEditKey(self: *Self, key: vaxis.Key) !void {
        if (key.matches(vaxis.Key.enter, .{})) {
            const value = &self.values[self.nav][self.field];
            value.clearRetainingCapacity();
            try value.appendSlice(self.allocator, self.edit.items);
            self.editing = false;
        } else if (key.matches(vaxis.Key.escape, .{})) {
            self.editing = false;
        } else if (key.matches(vaxis.Key.backspace, .{})) {
            // Drop one UTF-8 code point.
            var n = self.edit.items.len;
            while (n > 0) {
                n -= 1;
                if (self.edit.items[n] & 0xC0 != 0x80) break;
            }
            self.edit.shrinkRetainingCapacity(n);
        } else if (key.matches('u', .{ .ctrl = true })) {
            self.edit.clearRetainingCapacity();
        } else if (key.text) |text| {
            try self.edit.appendSlice(self.allocator, text);
        }
    }

    /// Starts the operation's command with its form's arguments.
    fn run(self: *Self, index: usize) !void {
        const op = operations[index];
        const command = op.command orelse return;
        if (op.phase()) |p| {
            self.setNotice(true, try std.fmt.allocPrint(self.init.arena.allocator(), "{s} arrives in PLAN.md phase {d}", .{ op.title, p }));
            return;
        }
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        var values: [cli.Operation.max_fields][]const u8 = undefined;
        for (op.fields, 0..) |_, i| values[i] = self.values[index][i].items;
        const args = try op.args(arena.allocator(), values[0..op.fields.len]);
        if (command == .chat) {
            // The chat runs beside jobs, on its own worker.
            if (self.chat.busy()) {
                self.setNotice(true, "the chat is busy (esc stops a reply)");
                return;
            }
            try self.chat.load(self.init, args);
            self.last_chat = .loading;
            self.input.clearRetainingCapacity();
            self.setNotice(false, "loading the model");
            return;
        }
        if (self.job.running()) {
            self.setNotice(true, "a job is already running (s stops training)");
            return;
        }
        log.debug("starting {s} with {d} arguments", .{ op.title, args.len });
        try self.job.start(self.init, index, command, args);
        self.last_state = .running;
        self.setNotice(false, "started");
    }

    fn stopJob(self: *Self) void {
        if (!self.job.running()) return;
        const op = operations[self.job.operation orelse return];
        if (op.command != .train) {
            self.setNotice(true, "only training can be stopped; others run to the end");
            return;
        }
        self.job.requestStop();
        self.setNotice(false, "stopping after this step; saving a checkpoint");
    }

    /// Quits; while a job runs, only on the second `q` (`armed`).
    fn quit(self: *Self, ctx: *vxfw.EventContext, armed: bool) void {
        self.chat.requestStop();
        if (self.job.running() and !armed) {
            self.quit_armed = true;
            self.setNotice(true, "a job is running: q again quits (training saves a checkpoint first)");
            return;
        }
        ctx.quit = true;
    }

    fn setNotice(self: *Self, problem: bool, text: []const u8) void {
        self.notice = text;
        self.notice_problem = problem;
    }

    // -------------------------------------------------------------------------
    // Drawing

    pub fn draw(self: *Self, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
        const width = ctx.max.width orelse 0;
        const height = ctx.max.height orelse 0;
        const a = ctx.arena;
        var children: std.ArrayList(vxfw.SubSurface) = .empty;
        const split = @min(@as(u16, nav_width), width / 3);
        const body_top: u16 = 2;
        const body_height = height -| body_top -| 2;
        const op = operations[self.nav];

        try children.append(a, try self.line(ctx, 0, 0, split, &.{
            .{ .text = "  Operations", .style = .{ .fg = Style.primaryText.color(), .bold = true } },
        }));
        try children.append(a, try self.line(ctx, 0, split + 1, width -| split -| 1, &.{
            .{ .text = op.title, .style = .{ .fg = Style.primaryText.color(), .bold = true } },
        }));
        try children.append(a, try self.rule(ctx, 1, width, split, "┼"));

        var nav_lines: std.ArrayList([]const Span) = .empty;
        for (operations, 0..) |entry, i| try nav_lines.append(a, try self.navRow(a, entry, i, split));
        try children.append(a, try self.block(ctx, body_top, 0, split, body_height, nav_lines.items));
        for (0..body_height) |r| try children.append(a, try self.line(ctx, body_top + @as(u16, @intCast(r)), split, 1, &.{.{ .text = "│", .style = Style.divider.style() }}));
        try children.append(a, try self.pane(ctx, body_top, split + 2, width -| split -| 2, body_height));

        try children.append(a, try self.rule(ctx, body_top + body_height, width, split, "┴"));
        try children.append(a, try self.line(ctx, body_top + body_height + 1, 0, width, try self.statusSpans(a)));
        return .{
            .size = .{ .width = width, .height = height },
            .widget = self.widget(),
            .buffer = &.{},
            .children = children.items,
        };
    }

    fn navRow(self: *Self, a: std.mem.Allocator, entry: cli.Operation, index: usize, width: u16) ![]const Span {
        var spans: std.ArrayList(Span) = .empty;
        const on_cursor = index == self.nav;
        try spans.append(a, .{
            .text = if (on_cursor) "› " else "  ",
            .style = if (self.focus == .nav) .{ .fg = Style.cursor.color(), .bold = true } else Style.secondaryText.style(),
        });
        const dim = entry.phase() != null;
        try spans.append(a, .{ .text = entry.title, .style = .{ .fg = if (dim) Style.secondaryText.color() else Style.primaryText.color(), .bold = on_cursor } });
        const tag: Span = if (entry.command == .chat) switch (self.chat.currentState()) {
            .loading, .replying => .{ .text = "●", .style = Style.warningText.style() },
            .ready => .{ .text = "✔", .style = Style.successText.style() },
            .failed => .{ .text = "✗", .style = Style.errorText.style() },
            .idle => .{ .text = "" },
        } else if (self.job.operation == index) switch (self.job.currentState()) {
            .running => .{ .text = "●", .style = Style.warningText.style() },
            .succeeded => .{ .text = "✔", .style = Style.successText.style() },
            .failed => .{ .text = "✗", .style = Style.errorText.style() },
            .idle => .{ .text = "" },
        } else if (entry.phase()) |p| .{ .text = try std.fmt.allocPrint(a, "p{d}", .{p}), .style = Style.secondaryText.style() } else .{ .text = "" };
        const used = 2 + entry.title.len + displayWidth(tag.text);
        try spans.append(a, .{ .text = try spaces(a, if (width > used + 1) width - used - 1 else 1) });
        try spans.append(a, tag);
        return spans.items;
    }

    /// The right pane: summary, form, Run row, then output or training progress.
    fn pane(self: *Self, ctx: vxfw.DrawContext, row: u16, col: u16, width: u16, height: u16) !vxfw.SubSurface {
        const a = ctx.arena;
        const op = operations[self.nav];
        var lines: std.ArrayList([]const Span) = .empty;
        try lines.append(a, &.{.{ .text = op.summary, .style = Style.secondaryText.style() }});
        try lines.append(a, &.{});

        if (op.command == null) {
            for (self.overview.lines) |l| {
                try lines.append(a, try a.dupe(Span, &.{
                    .{ .text = try pad(a, l.label, label_width), .style = Style.secondaryText.style() },
                    .{ .text = l.value, .style = if (l.problem) Style.warningText.style() else Style.primaryText.style() },
                }));
            }
            try lines.append(a, &.{});
            try lines.append(a, try a.dupe(Span, &.{ .{ .text = "r", .style = Style.keyHint.style() }, .{ .text = " refreshes", .style = Style.secondaryText.style() } }));
            return self.block(ctx, row, col, width, height, lines.items);
        }
        if (op.phase()) |p| {
            try lines.append(a, &.{.{ .text = try std.fmt.allocPrint(a, "Arrives in PLAN.md phase {d}.", .{p}), .style = Style.warningText.style() }});
            return self.block(ctx, row, col, width, height, lines.items);
        }

        if (op.command == .chat) return self.chatPane(ctx, row, col, width, height, lines);
        return self.formPane(ctx, row, col, width, height, lines);
    }

    /// The form rows and the Run row, then this operation's output.
    fn formPane(self: *Self, ctx: vxfw.DrawContext, row: u16, col: u16, width: u16, height: u16, start: std.ArrayList([]const Span)) !vxfw.SubSurface {
        const a = ctx.arena;
        const op = operations[self.nav];
        var lines = start;
        try self.formLines(a, &lines);
        const running_here = self.job.operation == self.nav and self.job.currentState() == .running;
        const on_run = self.focus == .form and self.field == op.fields.len;
        try lines.append(a, try a.dupe(Span, &.{
            .{ .text = if (on_run) "› " else "  ", .style = .{ .fg = Style.cursor.color(), .bold = true } },
            if (running_here)
                .{ .text = if (op.command == .train) "■ running (s stops and saves)" else "■ running", .style = Style.warningText.style() }
            else
                .{ .text = "▶ Run", .style = .{ .fg = Style.keyHint.color(), .bold = on_run } },
        }));
        try lines.append(a, &.{});

        // Output of this operation's last job.
        if (self.job.operation == self.nav) {
            const state = self.job.currentState();
            try lines.append(a, &.{switch (state) {
                .running => .{ .text = "● running", .style = Style.warningText.style() },
                .succeeded => .{ .text = "✔ finished", .style = Style.successText.style() },
                .failed => .{ .text = try std.fmt.allocPrint(a, "✗ failed: {s}", .{self.job.failure orelse "exit code"}), .style = Style.errorText.style() },
                .idle => .{ .text = "" },
            }});
            const used: u16 = @intCast(@min(lines.items.len, height));
            const top = try self.block(ctx, row, col, width, used, lines.items);
            const rest = height -| used;
            const snap = try self.job.snapshot(a);
            const body = if (op.command == .train and snap.report != null)
                try self.trainView(ctx, snap, width, rest)
            else
                try self.logView(ctx, width, rest);
            return self.stack(ctx, row, col, width, height, top, body, used);
        }
        return self.block(ctx, row, col, width, height, lines.items);
    }

    /// The chat page: the settings form until a model is loaded (or while it
    /// is being edited), then the transcript and the message being typed.
    fn chatPane(self: *Self, ctx: vxfw.DrawContext, row: u16, col: u16, width: u16, height: u16, start: std.ArrayList([]const Span)) !vxfw.SubSurface {
        const a = ctx.arena;
        const op = operations[self.nav];
        var lines = start;
        const snap = try self.chat.snapshot(a);
        const open = snap.state == .loading or snap.state == .ready or snap.state == .replying;
        if (!open or self.focus == .form) {
            try self.formLines(a, &lines);
            const on_run = self.focus == .form and self.field == op.fields.len;
            try lines.append(a, try a.dupe(Span, &.{
                .{ .text = if (on_run) "› " else "  ", .style = .{ .fg = Style.cursor.color(), .bold = true } },
                .{ .text = if (open) "▶ Load another model" else "▶ Load model", .style = .{ .fg = Style.keyHint.color(), .bold = on_run } },
            }));
            try lines.append(a, &.{});
        }
        try lines.append(a, &.{switch (snap.state) {
            .idle => .{ .text = "" },
            .loading => .{ .text = "● loading the model", .style = Style.warningText.style() },
            .ready, .replying => .{ .text = try std.fmt.allocPrint(a, "✔ {s}", .{snap.label}), .style = Style.successText.style() },
            .failed => .{ .text = try std.fmt.allocPrint(a, "✗ failed: {s}", .{snap.failure orelse "error"}), .style = Style.errorText.style() },
        }});
        const used: u16 = @intCast(@min(lines.items.len, height));
        const top = try self.block(ctx, row, col, width, used, lines.items);
        const body = try self.chatView(ctx, snap, width, height -| used);
        return self.stack(ctx, row, col, width, height, top, body, used);
    }

    /// The transcript's last lines above the input line.
    fn chatView(self: *Self, ctx: vxfw.DrawContext, snap: cli.ChatSnapshot, width: u16, height: u16) !vxfw.Surface {
        const a = ctx.arena;
        var lines: std.ArrayList([]const Span) = .empty;
        const text_width: usize = @max(@as(usize, width) -| 2, 8);
        for (snap.turns, 0..) |turn, i| {
            const last = i + 1 == snap.turns.len;
            switch (turn.role) {
                .user => try lines.append(a, &.{.{ .text = "You", .style = .{ .fg = Style.keyHint.color(), .bold = true } }}),
                .assistant => try lines.append(a, &.{.{ .text = "Assistant", .style = .{ .fg = Style.successText.color(), .bold = true } }}),
                .note => {},
            }
            const style = if (turn.role == .note) Style.secondaryText.style() else Style.primaryText.style();
            var text = std.mem.trimEnd(u8, turn.text, "\n");
            if (turn.role == .assistant and last and snap.state == .replying) text = try std.fmt.allocPrint(a, "{s}▍", .{text});
            for (try wrap(a, text, text_width)) |l| try lines.append(a, try a.dupe(Span, &.{ .{ .text = "  " }, .{ .text = l, .style = style } }));
            try lines.append(a, &.{});
        }
        const chatting = self.focus == .chat and operations[self.nav].command == .chat;
        const input_line: []const Span = if (chatting)
            try a.dupe(Span, &.{
                .{ .text = "› ", .style = .{ .fg = Style.cursor.color(), .bold = true } },
                .{ .text = try a.dupe(u8, self.input.items), .style = .{ .fg = Style.primaryText.color(), .bold = true } },
                .{ .text = "▏", .style = Style.cursor.style() },
            })
        else if (snap.state == .ready or snap.state == .replying)
            &.{.{ .text = "  enter or → to type a message", .style = Style.secondaryText.style() }}
        else
            &.{};
        const room: usize = height -| 1;
        const shown = lines.items[lines.items.len -| room..];
        var all: std.ArrayList([]const Span) = .empty;
        try all.appendSlice(a, shown);
        for (shown.len..room) |_| try all.append(a, &.{});
        try all.append(a, input_line);
        return (try self.block(ctx, 0, 0, width, height, all.items)).surface;
    }

    /// Splits text into lines of at most `width` code points, at newlines and
    /// otherwise at the last space that fits.
    fn wrap(a: std.mem.Allocator, text: []const u8, width: usize) ![]const []const u8 {
        var out: std.ArrayList([]const u8) = .empty;
        var paragraphs = std.mem.splitScalar(u8, text, '\n');
        while (paragraphs.next()) |para| {
            var rest = para;
            while (true) {
                var count: usize = 0;
                var end: usize = 0;
                var space: ?usize = null;
                while (end < rest.len and count < width) {
                    if (rest[end] == ' ') space = end;
                    end += std.unicode.utf8ByteSequenceLength(rest[end]) catch 1;
                    count += 1;
                }
                end = @min(end, rest.len);
                if (end == rest.len) {
                    try out.append(a, rest);
                    break;
                }
                const cut = if (space) |sp| (if (sp > 0) sp else end) else end;
                try out.append(a, rest[0..cut]);
                rest = std.mem.trimStart(u8, rest[cut..], " ");
            }
        }
        return out.items;
    }

    /// The operation's form rows.
    fn formLines(self: *Self, a: std.mem.Allocator, lines: *std.ArrayList([]const Span)) !void {
        const op = operations[self.nav];
        const form_focus = self.focus == .form;
        for (op.fields, 0..) |f, i| {
            const selected = form_focus and self.field == i;
            var spans: std.ArrayList(Span) = .empty;
            try spans.append(a, .{ .text = if (selected) "› " else "  ", .style = .{ .fg = Style.cursor.color(), .bold = true } });
            try spans.append(a, .{ .text = try pad(a, f.label, label_width), .style = Style.secondaryText.style() });
            if (selected and self.editing) {
                try spans.append(a, .{ .text = try a.dupe(u8, self.edit.items), .style = .{ .fg = Style.primaryText.color(), .bold = true } });
                try spans.append(a, .{ .text = "▏", .style = Style.cursor.style() });
            } else {
                const value = self.values[self.nav][i].items;
                try spans.append(a, if (value.len == 0)
                    .{ .text = "—", .style = Style.secondaryText.style() }
                else
                    .{ .text = try a.dupe(u8, value), .style = .{ .fg = Style.primaryText.color(), .bold = selected } });
            }
            if (selected and f.hint.len > 0) try spans.append(a, .{ .text = try std.fmt.allocPrint(a, "   {s}", .{f.hint}), .style = Style.secondaryText.style() });
            try lines.append(a, spans.items);
        }
    }

    /// The last lines of the job's output.
    fn logView(self: *Self, ctx: vxfw.DrawContext, width: u16, height: u16) !vxfw.Surface {
        const a = ctx.arena;
        const tail = try self.job.tail(a, height);
        var lines: std.ArrayList([]const Span) = .empty;
        for (tail) |t| try lines.append(a, try a.dupe(Span, &.{.{ .text = t, .style = Style.secondaryText.style() }}));
        return (try self.block(ctx, 0, 0, width, height, lines.items)).surface;
    }

    /// Progress, live numbers, the loss curve, val bpb and samples.
    fn trainView(self: *Self, ctx: vxfw.DrawContext, snap: cli.TrainSnapshot, width: u16, height: u16) !vxfw.Surface {
        const a = ctx.arena;
        const r = snap.report.?;
        var lines: std.ArrayList([]const Span) = .empty;
        const done = @as(f64, @floatFromInt(r.step + 1)) / @as(f64, @floatFromInt(r.num_iterations));
        const bar_width: usize = @max(@as(usize, width) -| 28, 10);
        const filled: usize = @intFromFloat(@min(done, 1) * @as(f64, @floatFromInt(bar_width)));
        var bar: std.ArrayList(u8) = .empty;
        for (0..bar_width) |i| try bar.appendSlice(a, if (i < filled) "█" else "░");
        const eta = if (r.step > 10) r.total_time / @as(f64, @floatFromInt(r.step - 10)) * @as(f64, @floatFromInt(r.num_iterations - r.step)) / 60 else 0;
        try lines.append(a, try a.dupe(Span, &.{
            .{ .text = bar.items, .style = Style.chart.style() },
            .{ .text = try std.fmt.allocPrint(a, " {d}/{d}  eta {d:.1}m", .{ r.step + 1, r.num_iterations, eta }) },
        }));
        try lines.append(a, &.{.{ .text = try std.fmt.allocPrint(a, "loss {d:.4} · lrm {d:.2} · {d:.0} tok/s · {d:.3} TFLOP/s · {d:.0} ms/step · epoch {d}", .{ r.loss, r.lrm, r.tok_per_sec, r.tflops, r.dt * 1000, r.state.epoch }) }});
        var bpb: std.ArrayList(u8) = .empty;
        try bpb.appendSlice(a, "val bpb");
        const first = snap.evals.len -| 6;
        for (snap.evals[first..]) |e| try bpb.print(a, "  {d:.4}@{d}", .{ e.bpb, e.step });
        try lines.append(a, &.{.{ .text = bpb.items, .style = Style.secondaryText.style() }});
        const header: u16 = @intCast(lines.items.len);

        var tail: std.ArrayList([]const Span) = .empty;
        if (snap.samples.len > 0) {
            try tail.append(a, &.{.{ .text = try std.fmt.allocPrint(a, "samples at step {d}", .{snap.sample_step}), .style = .{ .bold = true } }});
            for (snap.samples) |s| try tail.append(a, &.{.{ .text = try std.mem.replaceOwned(u8, a, s, "\n", " ") }});
        }
        const tail_h: u16 = @intCast(@min(tail.items.len, height / 3));
        const curve_h = height -| header -| tail_h -| 1;

        var children: std.ArrayList(vxfw.SubSurface) = .empty;
        try children.append(a, try self.block(ctx, 0, 0, width, header, lines.items));
        if (curve_h >= 3) try children.append(a, .{ .origin = .{ .row = header, .col = 0 }, .surface = try self.curve(ctx, snap.losses, width, curve_h) });
        if (tail_h > 0) try children.append(a, try self.block(ctx, header + curve_h + 1, 0, width, tail_h, tail.items[0..tail_h]));
        return .{ .size = .{ .width = width, .height = height }, .widget = self.widget(), .buffer = &.{}, .children = children.items };
    }

    /// Block-character columns of the bucket-averaged loss history.
    fn curve(self: *Self, ctx: vxfw.DrawContext, losses: []const f32, width: u16, height: u16) !vxfw.Surface {
        const a = ctx.arena;
        var surface = try vxfw.Surface.init(a, self.widget(), .{ .width = width, .height = height });
        const label_w: u16 = 8;
        if (losses.len == 0 or width <= label_w + 2) return surface;
        const cols: usize = width - label_w;
        const values = try a.alloc(f32, @min(cols, losses.len));
        for (values, 0..) |*v, c| {
            const lo = c * losses.len / values.len;
            const hi = @max((c + 1) * losses.len / values.len, lo + 1);
            var sum: f32 = 0;
            for (losses[lo..hi]) |x| sum += x;
            v.* = sum / @as(f32, @floatFromInt(hi - lo));
        }
        var min = values[0];
        var max = values[0];
        for (values) |v| {
            min = @min(min, v);
            max = @max(max, v);
        }
        if (max - min < 1e-6) max = min + 1e-6;
        try writeText(&surface, a, 0, 0, try std.fmt.allocPrint(a, "{d:>7.3}", .{max}), Style.secondaryText.style());
        try writeText(&surface, a, height - 1, 0, try std.fmt.allocPrint(a, "{d:>7.3}", .{min}), Style.secondaryText.style());
        for (values, 0..) |v, c| {
            const eighths: usize = @intFromFloat(@round((v - min) / (max - min) * @as(f32, @floatFromInt(@as(usize, height) * 8 - 1))) + 1);
            for (0..height) |row| {
                const from_bottom = height - 1 - row;
                const level = @min(eighths -| from_bottom * 8, 8);
                if (level == 0) continue;
                surface.writeCell(@intCast(label_w + c), @intCast(row), .{ .char = .{ .grapheme = blocks[level], .width = 1 }, .style = Style.chart.style() });
            }
        }
        return surface;
    }

    fn statusSpans(self: *Self, a: std.mem.Allocator) ![]const Span {
        var spans: std.ArrayList(Span) = .empty;
        const badge = if (self.editing) "editing" else switch (self.focus) {
            .nav => "operations",
            .form => "form",
            .chat => "chat",
        };
        try spans.append(a, .{ .text = try std.fmt.allocPrint(a, "▸▸ {s}  ", .{badge}), .style = .{ .fg = Style.statusText.color(), .bold = true } });
        if (self.notice) |text| {
            try spans.append(a, .{ .text = text, .style = if (self.notice_problem) .{ .fg = Style.errorText.color(), .bold = true } else Style.successText.style() });
            return spans.items;
        }
        const keys: []const Binding = if (self.editing) &.{
            .{ .key = "enter", .does = "save" },
            .{ .key = "esc", .does = "cancel" },
            .{ .key = "ctrl-u", .does = "clear" },
        } else switch (self.focus) {
            .nav => &.{
                .{ .key = "↑↓", .does = "move" },
                .{ .key = "→ enter", .does = "open" },
                .{ .key = "r", .does = "run" },
                .{ .key = "s", .does = "stop training" },
                .{ .key = "q", .does = "quit" },
            },
            .chat => &.{
                .{ .key = "enter", .does = "send" },
                .{ .key = "esc", .does = "stop reply / leave" },
                .{ .key = "tab", .does = "settings" },
                .{ .key = "ctrl-u", .does = "clear line" },
                .{ .key = "clear", .does = "new conversation" },
            },
            .form => &.{
                .{ .key = "↑↓", .does = "field" },
                .{ .key = "enter", .does = "edit / run" },
                .{ .key = "r", .does = "run" },
                .{ .key = "s", .does = "stop training" },
                .{ .key = "← esc", .does = "operations" },
            },
        };
        for (keys, 0..) |k, i| {
            if (i > 0) try spans.append(a, .{ .text = " · ", .style = Style.secondaryText.style() });
            try spans.append(a, .{ .text = k.key, .style = Style.keyHint.style() });
            try spans.append(a, .{ .text = " " });
            try spans.append(a, .{ .text = k.does, .style = Style.secondaryText.style() });
        }
        return spans.items;
    }

    // -------------------------------------------------------------------------
    // Layout helpers

    /// One line of spans at `(row, col)`.
    fn line(self: *Self, ctx: vxfw.DrawContext, row: u16, col: u16, width: u16, spans: []const Span) !vxfw.SubSurface {
        _ = self;
        const text = try ctx.arena.create(vxfw.RichText);
        text.* = .{ .text = spans, .softwrap = false, .overflow = .ellipsis, .width_basis = .parent };
        return .{ .origin = .{ .row = row, .col = col }, .surface = try text.widget().draw(ctx.withConstraints(.{ .width = width, .height = 1 }, .{ .width = width, .height = 1 })) };
    }

    /// Lines stacked from `(row, col)`, clipped to `height`.
    fn block(self: *Self, ctx: vxfw.DrawContext, row: u16, col: u16, width: u16, height: u16, lines: []const []const Span) !vxfw.SubSurface {
        var children: std.ArrayList(vxfw.SubSurface) = .empty;
        for (lines[0..@min(lines.len, height)], 0..) |spans, r| try children.append(ctx.arena, try self.line(ctx, @intCast(r), 0, width, spans));
        return .{ .origin = .{ .row = row, .col = col }, .surface = .{ .size = .{ .width = width, .height = height }, .widget = self.widget(), .buffer = &.{}, .children = children.items } };
    }

    /// `top` above `body`, as one sub-surface.
    fn stack(self: *Self, ctx: vxfw.DrawContext, row: u16, col: u16, width: u16, height: u16, top: vxfw.SubSurface, body: vxfw.Surface, split: u16) !vxfw.SubSurface {
        const children = try ctx.arena.dupe(vxfw.SubSurface, &.{ .{ .origin = .{ .row = 0, .col = 0 }, .surface = top.surface }, .{ .origin = .{ .row = split, .col = 0 }, .surface = body } });
        return .{ .origin = .{ .row = row, .col = col }, .surface = .{ .size = .{ .width = width, .height = height }, .widget = self.widget(), .buffer = &.{}, .children = children } };
    }

    /// A divider rule across the screen with a junction at the pane split.
    fn rule(self: *Self, ctx: vxfw.DrawContext, row: u16, width: u16, joint: u16, glyph: []const u8) !vxfw.SubSurface {
        var text: std.ArrayList(u8) = .empty;
        for (0..width) |c| try text.appendSlice(ctx.arena, if (c == joint) glyph else "─");
        return self.line(ctx, row, 0, width, try ctx.arena.dupe(Span, &.{.{ .text = text.items, .style = Style.divider.style() }}));
    }

    fn writeText(surface: *vxfw.Surface, a: std.mem.Allocator, row: u16, col: u16, text: []const u8, style: vaxis.Style) !void {
        for (text, 0..) |_, i| {
            if (col + i >= surface.size.width) break;
            surface.writeCell(@intCast(col + i), row, .{ .char = .{ .grapheme = try a.dupe(u8, text[i .. i + 1]), .width = 1 }, .style = style });
        }
    }

    fn pad(a: std.mem.Allocator, text: []const u8, width: usize) ![]const u8 {
        const n = displayWidth(text);
        if (n >= width) return std.fmt.allocPrint(a, "{s} ", .{text});
        const out = try a.alloc(u8, text.len + width - n);
        @memcpy(out[0..text.len], text);
        @memset(out[text.len..], ' ');
        return out;
    }

    fn spaces(a: std.mem.Allocator, n: usize) ![]const u8 {
        const out = try a.alloc(u8, n);
        @memset(out, ' ');
        return out;
    }

    /// Code points, a good-enough width for labels and status marks.
    fn displayWidth(text: []const u8) usize {
        return std.unicode.utf8CountCodepoints(text) catch text.len;
    }

    fn typeErasedEventHandler(ptr: *anyopaque, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.handleEvent(ctx, event);
    }

    fn typeErasedDrawFn(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.draw(ctx);
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

const TestSupport = struct {
    env: std.process.Environ.Map,
    arena: std.heap.ArenaAllocator,
    job: cli.Job,
    chat: cli.ChatJob,

    fn create(allocator: std.mem.Allocator, self: *TestSupport) !void {
        self.env = std.process.Environ.Map.init(allocator);
        try self.env.put(mod.Config.base_dir_env, "/nonexistent-zignanogpt");
        try self.env.put(mod.Config.nanochat_dir_env, "/nonexistent-nanochat");
        self.arena = std.heap.ArenaAllocator.init(allocator);
        self.job.init(allocator, std.testing.io);
        self.chat.init(allocator, std.testing.io);
    }

    fn processInit(self: *TestSupport, allocator: std.mem.Allocator) std.process.Init {
        return .{ .minimal = .{ .environ = .empty, .args = undefined }, .arena = &self.arena, .gpa = allocator, .io = std.testing.io, .environ_map = &self.env, .preopens = undefined };
    }

    fn destroy(self: *TestSupport) void {
        self.chat.deinit();
        self.job.deinit();
        self.arena.deinit();
        self.env.deinit();
    }

    fn context(allocator: std.mem.Allocator) vxfw.EventContext {
        return .{ .io = std.testing.io, .alloc = allocator, .cmds = .empty, .phase = .at_target };
    }
};

fn press(app: *ConsoleApp, ctx: *vxfw.EventContext, key: vaxis.Key) !void {
    try app.handleEvent(ctx, .{ .key_press = key });
}

test "console navigates operations and edits a form field" {
    const allocator = std.testing.allocator;
    var support: TestSupport = undefined;
    try TestSupport.create(allocator, &support);
    defer support.destroy();
    var app = try ConsoleApp.create(allocator, support.processInit(allocator), &support.job, &support.chat);
    defer app.deinit();
    var ctx = TestSupport.context(allocator);
    defer ctx.cmds.deinit(allocator);

    try press(&app, &ctx, .{ .codepoint = vaxis.Key.down });
    try std.testing.expectEqualStrings("Download data", operations[app.nav].title);
    try press(&app, &ctx, .{ .codepoint = vaxis.Key.enter });
    try std.testing.expectEqual(Focus.form, app.focus);
    // Edit "train shards" from 8 to 2.
    try press(&app, &ctx, .{ .codepoint = vaxis.Key.enter });
    try std.testing.expect(app.editing);
    try press(&app, &ctx, .{ .codepoint = vaxis.Key.backspace });
    try press(&app, &ctx, .{ .codepoint = '2', .text = "2" });
    try press(&app, &ctx, .{ .codepoint = vaxis.Key.enter });
    try std.testing.expect(!app.editing);
    try std.testing.expectEqualStrings("2", app.values[1][0].items);
    try press(&app, &ctx, .{ .codepoint = vaxis.Key.escape });
    try std.testing.expectEqual(Focus.nav, app.focus);
    // Wrapping upward from the first entry lands on the last.
    try press(&app, &ctx, .{ .codepoint = vaxis.Key.up });
    try press(&app, &ctx, .{ .codepoint = vaxis.Key.up });
    try std.testing.expectEqualStrings("Reinforcement (RL)", operations[app.nav].title);
    try press(&app, &ctx, .{ .codepoint = 'q', .text = "q" });
    try std.testing.expect(ctx.quit);
}

test "console fills a form from command-line arguments" {
    const allocator = std.testing.allocator;
    var support: TestSupport = undefined;
    try TestSupport.create(allocator, &support);
    defer support.destroy();
    var app = try ConsoleApp.create(allocator, support.processInit(allocator), &support.job, &support.chat);
    defer app.deinit();
    // The job fails at once (no tokenizer here); what matters is the form.
    try app.startWith(.{ .command = .train, .args = &.{ "--preset", "cpu", "--num-iterations=20", "--matrix-lr", "0.03" } });
    support.job.thread.?.join();
    support.job.thread = null;
    const train = cli.Operation.indexOf(.train).?;
    try std.testing.expectEqual(train, app.nav);
    try std.testing.expectEqualStrings("20", app.values[train][3].items);
    try std.testing.expectEqualStrings("--matrix-lr 0.03", app.values[train][13].items);
    try std.testing.expectEqual(cli.JobState.failed, support.job.currentState());
}

test "console draws titles, panes, rules and the status line" {
    const allocator = std.testing.allocator;
    var support: TestSupport = undefined;
    try TestSupport.create(allocator, &support);
    defer support.destroy();
    var app = try ConsoleApp.create(allocator, support.processInit(allocator), &support.job, &support.chat);
    defer app.deinit();
    app.overview.refresh(app.init);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    vxfw.DrawContext.init(.unicode);
    const ctx: vxfw.DrawContext = .{ .arena = arena.allocator(), .min = .{ .width = 0, .height = 0 }, .max = .{ .width = 100, .height = 30 }, .cell_size = .{ .width = 10, .height = 20 } };
    for (0..operations.len) |i| {
        app.nav = i;
        app.focus = if (operations[i].command != null and operations[i].phase() == null) .form else .nav;
        const surface = try app.draw(ctx);
        try std.testing.expectEqual(@as(u16, 100), surface.size.width);
        // titles (2), top rule, nav, 26 divider cells, pane, bottom rule, status
        try std.testing.expectEqual(@as(usize, 2 + 1 + 1 + 26 + 1 + 1 + 1), surface.children.len);
        try std.testing.expectEqual(@as(u16, 29), surface.children[surface.children.len - 1].origin.row);
    }
}

test "console chat page loads a model, sends a message and draws the transcript" {
    const allocator = std.testing.allocator;
    var support: TestSupport = undefined;
    try TestSupport.create(allocator, &support);
    defer support.destroy();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = root_buf[0..try tmp.dir.realPath(std.testing.io, &root_buf)];
    {
        var backend = try mod.Backend.init(allocator, std.testing.io, .{});
        defer backend.deinit();
        const imported = try mod.TorchImport.importCheckpoint(allocator, &backend, mod.Storage.init(allocator, std.testing.io), mod.build_options.source_root ++ "/testdata/nanochat_base", base, .base, null, null);
        allocator.free(imported.tag);
    }
    try support.env.put(mod.Config.base_dir_env, base);
    var app = try ConsoleApp.create(allocator, support.processInit(allocator), &support.job, &support.chat);
    defer app.deinit();
    var ctx = TestSupport.context(allocator);
    defer ctx.cmds.deinit(allocator);

    try app.startWith(.{ .command = .chat, .args = &.{ "--temperature", "0", "--max-tokens", "3" } });
    support.chat.thread.?.join();
    support.chat.thread = null;
    try app.handleEvent(&ctx, .tick);
    try std.testing.expectEqual(Focus.chat, app.focus);
    try std.testing.expectEqualStrings("0", app.values[app.nav][3].items);
    for ("Hi") |c| try press(&app, &ctx, .{ .codepoint = c, .text = &.{c} });
    try press(&app, &ctx, .{ .codepoint = vaxis.Key.enter });
    try std.testing.expectEqual(@as(usize, 0), app.input.items.len);
    support.chat.thread.?.join();
    support.chat.thread = null;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const snap = try support.chat.snapshot(arena.allocator());
    try std.testing.expectEqual(@as(usize, 3), snap.turns.len);
    try std.testing.expectEqualStrings("Hi", snap.turns[1].text);
    vxfw.DrawContext.init(.unicode);
    const draw_ctx: vxfw.DrawContext = .{ .arena = arena.allocator(), .min = .{ .width = 0, .height = 0 }, .max = .{ .width = 100, .height = 30 }, .cell_size = .{ .width = 10, .height = 20 } };
    const surface = try app.draw(draw_ctx);
    try std.testing.expectEqual(@as(u16, 100), surface.size.width);
    // "quit" leaves the conversation; tab opens the settings.
    try press(&app, &ctx, .{ .codepoint = vaxis.Key.tab });
    try std.testing.expectEqual(Focus.form, app.focus);
    try press(&app, &ctx, .{ .codepoint = vaxis.Key.escape });
    try std.testing.expectEqual(Focus.chat, app.focus);
    for ("quit") |c| try press(&app, &ctx, .{ .codepoint = c, .text = &.{c} });
    try press(&app, &ctx, .{ .codepoint = vaxis.Key.enter });
    try std.testing.expectEqual(Focus.nav, app.focus);
}

test "console wraps chat text at spaces and newlines" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const lines = try ConsoleApp.wrap(arena.allocator(), "the quick brown fox\njumps", 10);
    const want = [_][]const u8{ "the quick", "brown fox", "jumps" };
    try std.testing.expectEqual(want.len, lines.len);
    for (want, lines) |w, l| try std.testing.expectEqualStrings(w, l);
}
