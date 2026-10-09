const std = @import("std");
const log = std.log.scoped(.zignanogpt_console_app);
const cli = @import("module.zig");
const mod = cli.nanogpt;
const tui = cli.tui;
const Span = tui.TextSpan;
const Theme = tui.Theme;

/// Which pane the keys go to.
pub const Focus = enum { nav, form, chat };

/// A job to start as soon as the console is up (`zignanogpt train` on a terminal).
pub const ConsoleStart = struct {
    command: cli.Command,
    args: []const []const u8,
};

/// One key and what it does, for the status line.
/// Eighths of a cell, bottom-aligned, for the loss curve.
const nav_width = 26;
const label_width = 18;
const operations = cli.Operation.all;

/// One of the console's panes as a draw-only widget, for `SplitPane` to place.
/// It has no event handler, so it lives in the frame arena.
const PaneView = struct {
    const Self = @This();

    app: *ConsoleApp,
    part: enum { split, nav, detail },

    fn widget(self: *const Self) tui.Widget {
        return .{ .userdata = @constCast(self), .drawFn = typeErasedDrawFn };
    }

    fn typeErasedDrawFn(ptr: *anyopaque, ctx: tui.DrawContext) std.mem.Allocator.Error!tui.Surface {
        const self: *const Self = @ptrCast(@alignCast(ptr));
        return switch (self.part) {
            .split => self.app.splitPane(ctx),
            .nav => self.app.navPane(ctx),
            .detail => self.app.detailPane(ctx),
        };
    }
};

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
    /// The selected operation's form; its row past the fields is the Run row.
    form: tui.Form,
    /// The two panes under their titles, with a draggable divider.
    split: tui.SplitPane = .{ .left_width = nav_width },
    values: [operations.len][cli.Operation.max_fields]std.ArrayList(u8),
    notice: ?[]const u8 = null,
    notice_problem: bool = false,
    quit_armed: bool = false,
    last_state: cli.JobState = .idle,
    pending_start: ?ConsoleStart = null,
    /// The chat page's message being typed.
    input: tui.LineInput,
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
        var self = Self{ .allocator = allocator, .init = init, .job = job, .chat = chat, .overview = cli.Overview.init(allocator), .form = .init(allocator), .input = .init(allocator), .values = undefined };
        self.form.label_width = label_width;
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
        self.form.deinit();
        self.input.deinit();
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
        self.form.reset();
        self.form.cursor = op.fields.len;
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
            const eq = std.mem.findScalar(u8, name, '=');
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

    pub fn widget(self: *Self) tui.Widget {
        return .{ .userdata = self, .eventHandler = typeErasedEventHandler, .drawFn = typeErasedDrawFn };
    }

    // -------------------------------------------------------------------------
    // Events

    pub fn handleEvent(self: *Self, ctx: *tui.EventContext, event: tui.vxfw.Event) anyerror!void {
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

    fn handleKey(self: *Self, ctx: *tui.EventContext, key: tui.Key) !void {
        if (key.matches('c', .{ .ctrl = true })) {
            ctx.quit = true;
            return;
        }
        if (self.form.editing) {
            _ = try self.formKey(key);
            return;
        }
        if (self.focus == .chat) return self.handleChatKey(key);
        const armed = self.quit_armed;
        self.quit_armed = false;
        self.notice = null;
        const op = operations[self.nav];
        if (key.matches('s', .{})) return self.stopJob();
        switch (self.focus) {
            .nav => {
                if (key.matches(tui.Key.up, .{}) or key.matches('k', .{})) {
                    self.nav = if (self.nav == 0) operations.len - 1 else self.nav - 1;
                    self.form.reset();
                } else if (key.matches(tui.Key.down, .{}) or key.matches('j', .{})) {
                    self.nav = (self.nav + 1) % operations.len;
                    self.form.reset();
                } else if (key.matches(tui.Key.right, .{}) or key.matches(tui.Key.enter, .{}) or key.matches(tui.Key.tab, .{})) {
                    if (op.command == .chat and self.chatOpen()) {
                        self.focus = .chat;
                    } else if (op.command != null and op.phase() == null) self.focus = .form;
                } else if (key.matches('r', .{})) {
                    if (op.command == null) self.overview.refresh(self.init) else try self.run(self.nav);
                } else if (key.matches('q', .{}) or key.matches(tui.Key.escape, .{})) {
                    return self.quit(ctx, armed);
                }
            },
            .form => {
                if (try self.formKey(key)) return;
                if (key.matches(tui.Key.left, .{}) or key.matches(tui.Key.escape, .{})) {
                    self.focus = if (op.command == .chat and self.chatOpen()) .chat else .nav;
                } else if (key.matches('r', .{})) {
                    try self.run(self.nav);
                } else if (key.matches('q', .{})) {
                    return self.quit(ctx, armed);
                }
            },
            .chat => unreachable,
        }
    }

    /// The chat page's keys: typing, enter sends (`quit`/`exit` leave,
    /// `clear` forgets the conversation), esc stops a reply or leaves.
    fn handleChatKey(self: *Self, key: tui.Key) !void {
        self.notice = null;
        const state = self.chat.currentState();
        if (key.matches(tui.Key.escape, .{})) {
            if (state == .replying) {
                self.chat.requestStop();
            } else self.focus = .nav;
        } else if (key.matches(tui.Key.tab, .{})) {
            self.focus = .form;
        } else if (key.matches(tui.Key.enter, .{})) {
            const text = std.mem.trim(u8, self.input.value(), " \t");
            if (std.ascii.eqlIgnoreCase(text, "quit") or std.ascii.eqlIgnoreCase(text, "exit")) {
                self.input.clear();
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
            self.input.clear();
        } else {
            _ = try self.input.handleKey(key);
        }
    }

    /// Whether the chat has a model (loading, ready or replying).
    fn chatOpen(self: *Self) bool {
        return switch (self.chat.currentState()) {
            .loading, .ready, .replying => true,
            .idle, .failed => false,
        };
    }

    /// Routes a key to the operation's form and acts on what it did: stores a
    /// saved value, flips a toggle, starts an edit, or runs from the Run row.
    ///
    /// Return: whether the key was the form's.
    fn formKey(self: *Self, key: tui.Key) !bool {
        const op = operations[self.nav];
        switch (try self.form.handleKey(key, op.fields.len + 1)) {
            .ignored => return false,
            .handled, .cancelled => {},
            .saved => |saved| {
                const value = &self.values[self.nav][saved.index];
                value.clearRetainingCapacity();
                try value.appendSlice(self.allocator, saved.value);
            },
            .chosen => |row| {
                if (row == op.fields.len) {
                    try self.run(self.nav);
                    return true;
                }
                const value = &self.values[self.nav][row];
                switch (op.fields[row].kind) {
                    .toggle => {
                        const on = std.ascii.eqlIgnoreCase(value.items, "yes");
                        value.clearRetainingCapacity();
                        try value.appendSlice(self.allocator, if (on) "no" else "yes");
                    },
                    .text => try self.form.edit(value.items),
                }
            },
        }
        return true;
    }

    /// Starts the operation's command with its form's arguments.
    fn run(self: *Self, index: usize) !void {
        const op = operations[index];
        if (op.command == null) return;
        if (op.phase()) |p| {
            self.setNotice(true, try self.init.arena.allocator().print("{s} arrives in PLAN.md phase {d}", .{ op.title, p }));
            return;
        }
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        var values: [cli.Operation.max_fields][]const u8 = undefined;
        for (op.fields, 0..) |_, i| values[i] = self.values[index][i].items;
        const args = try op.args(arena.allocator(), values[0..op.fields.len]);
        const command = op.commandFor(values[0..op.fields.len]);
        if (command == .chat) {
            // The chat runs beside jobs, on its own worker.
            if (self.chat.busy()) {
                self.setNotice(true, "the chat is busy (esc stops a reply)");
                return;
            }
            try self.chat.load(self.init, args);
            self.last_chat = .loading;
            self.input.clear();
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
        if (op.command != .train and op.command != .sft and op.command != .rl) {
            self.setNotice(true, "only training can be stopped; others run to the end");
            return;
        }
        self.job.requestStop();
        self.setNotice(false, "stopping; saving a checkpoint");
    }

    /// Quits; while a job runs, only on the second `q` (`armed`).
    fn quit(self: *Self, ctx: *tui.EventContext, armed: bool) void {
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

    pub fn draw(self: *Self, ctx: tui.DrawContext) std.mem.Allocator.Error!tui.Surface {
        const a = ctx.arena;
        const split = try a.create(PaneView);
        split.* = .{ .app = self, .part = .split };
        // The split takes every row but the last, which is the status line.
        const status = try textBlock(a, &.{try self.statusLine().spans(a)});
        const column = try a.create(tui.Column);
        column.* = .{ .items = try a.dupe(tui.ColumnItem, &.{ .{ .widget = split.widget() }, .{ .widget = status, .height = 1 } }) };
        var surface = try column.draw(ctx);
        surface.widget = self.widget();
        return surface;
    }

    /// The two panes under their titles.
    fn splitPane(self: *Self, ctx: tui.DrawContext) std.mem.Allocator.Error!tui.Surface {
        const a = ctx.arena;
        const bold: tui.Style = .{ .fg = Theme.primaryText.color(), .bold = true };
        const views = try a.alloc(PaneView, 2);
        views[0] = .{ .app = self, .part = .nav };
        views[1] = .{ .app = self, .part = .detail };
        return self.split.draw(ctx, .{
            .left_title = try textBlock(a, &.{try a.dupe(Span, &.{.{ .text = "  Operations", .style = bold }})}),
            .right_title = try textBlock(a, &.{try a.dupe(Span, &.{.{ .text = operations[self.nav].title, .style = bold }})}),
            .left = views[0].widget(),
            .right = views[1].widget(),
        });
    }

    /// The left pane: one row per operation.
    fn navPane(self: *Self, ctx: tui.DrawContext) std.mem.Allocator.Error!tui.Surface {
        const width = ctx.max.width orelse 0;
        var rows: std.ArrayList([]const Span) = .empty;
        for (operations, 0..) |entry, i| try rows.append(ctx.arena, try self.navRow(ctx.arena, entry, i, width));
        const block: tui.TextBlock = .{ .lines = rows.items };
        return block.draw(ctx);
    }

    /// The right pane, one column in from the divider.
    fn detailPane(self: *Self, ctx: tui.DrawContext) std.mem.Allocator.Error!tui.Surface {
        const content = try self.pane(ctx.arena, (ctx.max.width orelse 0) -| 1, ctx.max.height orelse 0);
        const column: tui.Column = .{ .items = &.{.{ .widget = content }}, .indent = 1 };
        return column.draw(ctx);
    }

    fn navRow(self: *Self, a: std.mem.Allocator, entry: cli.Operation, index: usize, width: u16) ![]const Span {
        var spans: std.ArrayList(Span) = .empty;
        const on_cursor = index == self.nav;
        try spans.append(a, .{
            .text = if (on_cursor) "› " else "  ",
            .style = if (self.focus == .nav) .{ .fg = Theme.cursor.color(), .bold = true } else Theme.secondaryText.style(),
        });
        const dim = entry.phase() != null;
        try spans.append(a, .{ .text = entry.title, .style = .{ .fg = if (dim) Theme.secondaryText.color() else Theme.primaryText.color(), .bold = on_cursor } });
        const mark: tui.StatusMark = if (entry.command == .chat) chatMark(self.chat.currentState()) else if (self.job.operation == index) jobMark(self.job.currentState()) else .idle;
        const tag: Span = if (mark != .idle) mark.span() else if (entry.phase()) |p| .{ .text = try a.print("p{d}", .{p}), .style = Theme.secondaryText.style() } else .{ .text = "" };
        const used = 2 + entry.title.len + displayWidth(tag.text);
        try spans.append(a, .{ .text = try spaces(a, if (width > used + 1) width - used - 1 else 1) });
        try spans.append(a, tag);
        return spans.items;
    }

    /// The right pane: summary, form, Run row, then output or training progress.
    fn pane(self: *Self, a: std.mem.Allocator, width: u16, height: u16) !tui.Widget {
        const op = operations[self.nav];
        var rows: std.ArrayList([]const Span) = .empty;
        for (try tui.TextWidth.wrap(a, op.summary, @max(width, 1))) |line| {
            try rows.append(a, try a.dupe(Span, &.{.{ .text = line, .style = Theme.secondaryText.style() }}));
        }
        try rows.append(a, &.{});

        if (op.command == null) {
            for (self.overview.lines) |l| {
                try rows.append(a, try a.dupe(Span, &.{
                    .{ .text = try pad(a, l.label, label_width), .style = Theme.secondaryText.style() },
                    .{ .text = l.value, .style = if (l.problem) Theme.warningText.style() else Theme.primaryText.style() },
                }));
            }
            try rows.append(a, &.{});
            try rows.append(a, try a.dupe(Span, &.{ .{ .text = "r", .style = Theme.keyHint.style() }, .{ .text = " refreshes", .style = Theme.secondaryText.style() } }));
            return textBlock(a, rows.items);
        }
        if (op.phase()) |p| {
            try rows.append(a, try a.dupe(Span, &.{.{ .text = try a.print("Arrives in PLAN.md phase {d}.", .{p}), .style = Theme.warningText.style() }}));
            return textBlock(a, rows.items);
        }

        if (op.command == .chat) return self.chatPane(a, height, rows);
        return self.formPane(a, width, height, rows);
    }

    /// The form rows and the Run row, then this operation's output.
    fn formPane(self: *Self, a: std.mem.Allocator, width: u16, height: u16, start: std.ArrayList([]const Span)) !tui.Widget {
        const op = operations[self.nav];
        var rows = start;
        try self.formLines(a, &rows);
        const running_here = self.job.operation == self.nav and self.job.currentState() == .running;
        const on_run = self.focus == .form and self.form.cursor == op.fields.len;
        try rows.append(a, try a.dupe(Span, &.{
            self.form.marker(op.fields.len, self.focus == .form),
            if (running_here)
                .{ .text = if (op.command == .train or op.command == .sft or op.command == .rl) "■ running (s stops and saves)" else "■ running", .style = Theme.warningText.style() }
            else
                .{ .text = "▶ Run", .style = .{ .fg = Theme.keyHint.color(), .bold = on_run } },
        }));
        try rows.append(a, &.{});
        if (self.job.operation != self.nav) return textBlock(a, rows.items);

        // Output of this operation's last job, below the form.
        const state = self.job.currentState();
        try rows.append(a, try markLine(a, jobMark(state), switch (state) {
            .running => " running",
            .succeeded => " finished",
            .failed => try a.print(" failed: {s}", .{self.job.failure orelse "exit code"}),
            .idle => "",
        }));
        const snap = try self.job.snapshot(a);
        if (snap.progress) |p| try rows.append(a, try progressLine(a, p, width));
        const used: u16 = @intCast(@min(rows.items.len, height));
        const rest = height -| used;
        const body = if ((op.command == .train or op.command == .sft) and snap.report != null)
            try trainView(a, snap, width, rest)
        else if (op.command == .rl and snap.rl_report != null)
            try rlView(a, snap, width, rest)
        else
            try self.logView(a, rest);
        return stacked(a, try textBlock(a, rows.items), used, body);
    }

    /// The chat page: the settings form until a model is loaded (or while it
    /// is being edited), then the transcript and the message being typed.
    fn chatPane(self: *Self, a: std.mem.Allocator, height: u16, start: std.ArrayList([]const Span)) !tui.Widget {
        const op = operations[self.nav];
        var rows = start;
        const snap = try self.chat.snapshot(a);
        const open = snap.state == .loading or snap.state == .ready or snap.state == .replying;
        if (!open or self.focus == .form) {
            try self.formLines(a, &rows);
            const on_run = self.focus == .form and self.form.cursor == op.fields.len;
            try rows.append(a, try a.dupe(Span, &.{
                self.form.marker(op.fields.len, self.focus == .form),
                .{ .text = if (open) "▶ Load another model" else "▶ Load model", .style = .{ .fg = Theme.keyHint.color(), .bold = on_run } },
            }));
            try rows.append(a, &.{});
        }
        try rows.append(a, try markLine(a, chatMark(snap.state), switch (snap.state) {
            .idle => "",
            .loading => " loading the model",
            .ready, .replying => try a.print(" {s}", .{snap.label}),
            .failed => try a.print(" failed: {s}", .{snap.failure orelse "error"}),
        }));
        const used: u16 = @intCast(@min(rows.items.len, height));
        return stacked(a, try textBlock(a, rows.items), used, try self.chatView(a, snap));
    }

    /// The transcript's last lines above the input line.
    fn chatView(self: *Self, a: std.mem.Allocator, snap: cli.ChatSnapshot) !tui.Widget {
        const entries = try a.alloc(tui.TranscriptEntry, snap.turns.len);
        for (snap.turns, entries, 0..) |turn, *entry, i| {
            var text = std.mem.trimEnd(u8, turn.text, "\n");
            if (turn.role == .assistant and i + 1 == snap.turns.len and snap.state == .replying) text = try a.print("{s}▍", .{text});
            entry.* = switch (turn.role) {
                .user => .{ .heading = "You", .heading_role = .keyHint, .text = text },
                .assistant => .{ .heading = "Assistant", .heading_role = .successText, .text = text },
                .note => .{ .text = text, .role = .secondaryText },
            };
        }
        const record = try a.create(tui.TranscriptView);
        record.* = .{ .entries = entries };

        const chatting = self.focus == .chat and operations[self.nav].command == .chat;
        const input_line: []const Span = if (chatting) blk: {
            const typed = try self.input.spans(a, .{ .fg = Theme.primaryText.color(), .bold = true });
            break :blk try a.dupe(Span, &.{ .{ .text = "› ", .style = .{ .fg = Theme.cursor.color(), .bold = true } }, typed[0], typed[1] });
        } else if (snap.state == .ready or snap.state == .replying)
            try a.dupe(Span, &.{.{ .text = "  enter or → to type a message", .style = Theme.secondaryText.style() }})
        else
            &.{};
        const column = try a.create(tui.Column);
        column.* = .{ .items = try a.dupe(tui.ColumnItem, &.{ .{ .widget = record.widget() }, .{ .widget = try textBlock(a, &.{input_line}), .height = 1 } }) };
        return column.widget();
    }

    /// The operation's form rows.
    fn formLines(self: *Self, a: std.mem.Allocator, lines: *std.ArrayList([]const Span)) !void {
        const op = operations[self.nav];
        var current: [cli.Operation.max_fields][]const u8 = undefined;
        for (op.fields, 0..) |_, i| current[i] = self.values[self.nav][i].items;
        // Fields for the command the form would not run (Evaluate: base vs sft/rl) are dimmed.
        const command = if (op.command != null) op.commandFor(current[0..op.fields.len]) else null;
        var fields: [cli.Operation.max_fields]tui.FormField = undefined;
        for (op.fields, 0..) |f, i| {
            var buf: [64]u8 = undefined;
            fields[i] = .{
                .label = f.label,
                .value = current[i],
                .placeholder = if (current[i].len == 0) try a.dupe(u8, op.placeholder(i, current[0..op.fields.len], &buf)) else "",
                .hint = f.hint,
                .active = command == null or f.appliesTo(command.?),
            };
        }
        try lines.appendSlice(a, try self.form.lines(a, fields[0..op.fields.len], self.focus == .form));
    }

    /// The last lines of the job's output.
    fn logView(self: *Self, a: std.mem.Allocator, height: u16) !tui.Widget {
        const tail = try self.job.tail(a, height);
        const entries = try a.alloc(tui.TranscriptEntry, tail.len);
        for (tail, entries) |t, *entry| entry.* = .{ .text = t, .role = .secondaryText };
        const record = try a.create(tui.TranscriptView);
        record.* = .{ .entries = entries, .indent = 0, .spaced = false, .wrap = false };
        return record.widget();
    }

    /// Progress, live numbers, the loss curve, val bpb and samples.
    fn trainView(a: std.mem.Allocator, snap: cli.TrainSnapshot, width: u16, height: u16) !tui.Widget {
        const r = snap.report.?;
        var rows: std.ArrayList([]const Span) = .empty;
        const done = @as(f64, @floatFromInt(r.step + 1)) / @as(f64, @floatFromInt(r.num_iterations));
        const bar: tui.ProgressBar = .{ .done = done, .width = @max(@as(usize, width) -| 28, 10) };
        const eta = if (r.step > 10) r.total_time / @as(f64, @floatFromInt(r.step - 10)) * @as(f64, @floatFromInt(r.num_iterations - r.step)) / 60 else 0;
        try rows.append(a, try a.dupe(Span, &.{
            try bar.span(a),
            .{ .text = try a.print(" {d}/{d}  eta {d:.1}m", .{ r.step + 1, r.num_iterations, eta }) },
        }));
        try rows.append(a, try a.dupe(Span, &.{.{ .text = try a.print("loss {d:.4} · lrm {d:.2} · {d:.0} tok/s · {d:.3} TFLOP/s · {d:.0} ms/step · epoch {d}", .{ r.loss, r.lrm, r.tok_per_sec, r.tflops, r.dt * 1000, r.state.epoch }) }}));
        var bpb: std.ArrayList(u8) = .empty;
        try bpb.appendSlice(a, "val bpb");
        const first = snap.evals.len -| 6;
        for (snap.evals[first..]) |e| try bpb.print(a, "  {d:.4}@{d}", .{ e.value, e.step });
        try rows.append(a, try a.dupe(Span, &.{.{ .text = bpb.items, .style = Theme.secondaryText.style() }}));
        const header: u16 = @intCast(rows.items.len);

        var tail: std.ArrayList([]const Span) = .empty;
        if (snap.samples.len > 0) {
            try tail.append(a, try a.dupe(Span, &.{.{ .text = try a.print("samples at step {d}", .{snap.sample_step}), .style = .{ .bold = true } }}));
            for (snap.samples) |s| try tail.append(a, try a.dupe(Span, &.{.{ .text = try std.mem.replaceOwned(u8, a, s, "\n", " ") }}));
        }
        const tail_h: u16 = @intCast(@min(tail.items.len, height / 3));
        const curve_h = height -| header -| tail_h -| 1;

        // Header, loss curve (when it has room), a blank row, samples.
        const chart = try a.create(tui.Sparkline);
        chart.* = .{ .values = snap.losses };
        const blank = try textBlock(a, &.{});
        const column = try a.create(tui.Column);
        column.* = .{ .items = try a.dupe(tui.ColumnItem, &.{
            .{ .widget = try textBlock(a, rows.items), .height = header },
            .{ .widget = if (curve_h >= 3) chart.widget() else blank, .height = curve_h },
            .{ .widget = blank, .height = 1 },
            .{ .widget = try textBlock(a, tail.items[0..tail_h]), .height = tail_h },
        }) };
        return column.widget();
    }

    /// RL progress: steps, reward and sequence length, pass@1, the reward curve.
    fn rlView(a: std.mem.Allocator, snap: cli.TrainSnapshot, width: u16, height: u16) !tui.Widget {
        const r = snap.rl_report.?;
        var rows: std.ArrayList([]const Span) = .empty;
        const finished = r.step + 1;
        const done = @as(f64, @floatFromInt(finished)) / @as(f64, @floatFromInt(@max(r.num_steps, 1)));
        const bar: tui.ProgressBar = .{ .done = done, .width = @max(@as(usize, width) -| 28, 10) };
        const eta = r.total_time / @as(f64, @floatFromInt(finished)) * @as(f64, @floatFromInt(r.num_steps -| finished)) / 60;
        try rows.append(a, try a.dupe(Span, &.{
            try bar.span(a),
            .{ .text = try a.print(" {d}/{d}  eta {d:.1}m", .{ finished, r.num_steps, eta }) },
        }));
        try rows.append(a, try a.dupe(Span, &.{.{ .text = try a.print("reward {d:.4} · sequence length {d:.1} · lrm {d:.2} · {d:.1} s/step", .{ r.reward, r.sequence_length, r.lrm, r.dt }) }}));
        var passk: std.ArrayList(u8) = .empty;
        try passk.appendSlice(a, "pass@1");
        const first = snap.evals.len -| 6;
        for (snap.evals[first..]) |e| try passk.print(a, "  {d:.4}@{d}", .{ e.value, e.step });
        try rows.append(a, try a.dupe(Span, &.{.{ .text = passk.items, .style = Theme.secondaryText.style() }}));
        try rows.append(a, try a.dupe(Span, &.{.{ .text = "mean reward per step", .style = .{ .bold = true } }}));
        const header: u16 = @intCast(rows.items.len);

        const chart = try a.create(tui.Sparkline);
        chart.* = .{ .values = snap.rewards };
        const curve_h = height -| header;
        const column = try a.create(tui.Column);
        column.* = .{ .items = try a.dupe(tui.ColumnItem, &.{
            .{ .widget = try textBlock(a, rows.items), .height = header },
            .{ .widget = if (curve_h >= 3) chart.widget() else try textBlock(a, &.{}), .height = curve_h },
        }) };
        return column.widget();
    }

    /// The job's current work as a bar: `label  ████░░░░ done/total`.
    fn progressLine(a: std.mem.Allocator, p: cli.JobProgress, width: u16) ![]const Span {
        const done = if (p.total == 0) 0 else @as(f64, @floatFromInt(p.done)) / @as(f64, @floatFromInt(p.total));
        const count = try a.print(" {d}/{d}", .{ p.done, p.total });
        const label = try a.print("{s}  ", .{p.label});
        const bar: tui.ProgressBar = .{ .done = done, .width = @max(@as(usize, width) -| (displayWidth(label) + count.len + 2), 10) };
        return a.dupe(Span, &.{
            .{ .text = label, .style = Theme.secondaryText.style() },
            try bar.span(a),
            .{ .text = count, .style = Theme.secondaryText.style() },
        });
    }

    /// The bottom line: where the keys go, then the keys that work there, or the notice.
    fn statusLine(self: *const Self) tui.StatusLine {
        const mode = if (self.form.editing) "editing" else switch (self.focus) {
            .nav => "operations",
            .form => "form",
            .chat => "chat",
        };
        const hints: []const tui.KeyHint = if (self.form.editing) &.{
            .{ .key = "enter", .action = "save" },
            .{ .key = "esc", .action = "cancel" },
            .{ .key = "ctrl-u", .action = "clear" },
        } else switch (self.focus) {
            .nav => &.{
                .{ .key = "↑↓", .action = "move" },
                .{ .key = "→ enter", .action = "open" },
                .{ .key = "r", .action = "run" },
                .{ .key = "s", .action = "stop training" },
                .{ .key = "q", .action = "quit" },
            },
            .chat => &.{
                .{ .key = "enter", .action = "send" },
                .{ .key = "esc", .action = "stop reply / leave" },
                .{ .key = "tab", .action = "settings" },
                .{ .key = "ctrl-u", .action = "clear line" },
                .{ .key = "clear", .action = "new conversation" },
            },
            .form => &.{
                .{ .key = "↑↓", .action = "field" },
                .{ .key = "enter", .action = "edit / run" },
                .{ .key = "r", .action = "run" },
                .{ .key = "s", .action = "stop training" },
                .{ .key = "← esc", .action = "operations" },
            },
        };
        return .{ .mode = mode, .hints = hints, .notice = if (self.notice) |text| .{ .text = text, .problem = self.notice_problem } else null };
    }

    /// A job's state as a mark.
    fn jobMark(state: cli.JobState) tui.StatusMark {
        return switch (state) {
            .idle => .idle,
            .running => .busy,
            .succeeded => .done,
            .failed => .failed,
        };
    }

    /// The chat's state as a mark (a loaded model is "done").
    fn chatMark(state: cli.ChatState) tui.StatusMark {
        return switch (state) {
            .idle => .idle,
            .loading, .replying => .busy,
            .ready => .done,
            .failed => .failed,
        };
    }

    /// A mark followed by text in the mark's role; empty for `idle`.
    fn markLine(a: std.mem.Allocator, mark: tui.StatusMark, text: []const u8) ![]const Span {
        if (mark == .idle) return &.{};
        return a.dupe(Span, &.{ mark.span(), .{ .text = text, .style = mark.role().style() } });
    }

    // -------------------------------------------------------------------------
    // Layout helpers

    /// Lines of spans as a frame-arena widget. The widget draws after its
    /// builder returns, so the list is copied into the arena; each row must
    /// already live there (`a.dupe` a literal holding runtime values).
    fn textBlock(a: std.mem.Allocator, rows: []const []const Span) !tui.Widget {
        const block = try a.create(tui.TextBlock);
        block.* = .{ .lines = try a.dupe([]const Span, rows) };
        return block.widget();
    }

    /// `top` (taking `top_rows`) over `body` (the rest), as a frame-arena widget.
    fn stacked(a: std.mem.Allocator, top: tui.Widget, top_rows: u16, body: tui.Widget) !tui.Widget {
        const column = try a.create(tui.Column);
        column.* = .{ .items = try a.dupe(tui.ColumnItem, &.{ .{ .widget = top, .height = top_rows }, .{ .widget = body } }) };
        return column.widget();
    }

    fn pad(a: std.mem.Allocator, text: []const u8, width: usize) ![]const u8 {
        const n = displayWidth(text);
        if (n >= width) return a.print("{s} ", .{text});
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

    fn typeErasedEventHandler(ptr: *anyopaque, ctx: *tui.EventContext, event: tui.vxfw.Event) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ptr));
        return self.handleEvent(ctx, event);
    }

    fn typeErasedDrawFn(ptr: *anyopaque, ctx: tui.DrawContext) std.mem.Allocator.Error!tui.Surface {
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

    fn context(allocator: std.mem.Allocator) tui.EventContext {
        return .{ .io = std.testing.io, .alloc = allocator, .cmds = .empty, .phase = .at_target };
    }
};

fn press(app: *ConsoleApp, ctx: *tui.EventContext, key: tui.Key) !void {
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

    try press(&app, &ctx, .{ .codepoint = tui.Key.down });
    try std.testing.expectEqualStrings("Prepare data", operations[app.nav].title);
    try press(&app, &ctx, .{ .codepoint = tui.Key.down });
    try std.testing.expectEqualStrings("Download data", operations[app.nav].title);
    try press(&app, &ctx, .{ .codepoint = tui.Key.enter });
    try std.testing.expectEqual(Focus.form, app.focus);
    // Edit "train shards" from 8 to 2.
    try press(&app, &ctx, .{ .codepoint = tui.Key.enter });
    try std.testing.expect(app.form.editing);
    try press(&app, &ctx, .{ .codepoint = tui.Key.backspace });
    try press(&app, &ctx, .{ .codepoint = '2', .text = "2" });
    try press(&app, &ctx, .{ .codepoint = tui.Key.enter });
    try std.testing.expect(!app.form.editing);
    try std.testing.expectEqualStrings("2", app.values[2][0].items);
    try press(&app, &ctx, .{ .codepoint = tui.Key.escape });
    try std.testing.expectEqual(Focus.nav, app.focus);
    // Wrapping upward from the first entry lands on the last.
    try press(&app, &ctx, .{ .codepoint = tui.Key.up });
    try press(&app, &ctx, .{ .codepoint = tui.Key.up });
    try std.testing.expectEqualStrings("Overview", operations[app.nav].title);
    try press(&app, &ctx, .{ .codepoint = tui.Key.up });
    try std.testing.expectEqualStrings("Chat", operations[app.nav].title);
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
    try app.startWith(.{ .command = .train, .args = &.{ "--depth", "4", "--num-iterations=20", "--matrix-lr", "0.03" } });
    support.job.thread.?.join();
    support.job.thread = null;
    const train = cli.Operation.indexOf(.train).?;
    try std.testing.expectEqual(train, app.nav);
    try std.testing.expectEqualStrings("4", app.values[train][0].items);
    try std.testing.expectEqualStrings("20", app.values[train][4].items);
    try std.testing.expectEqualStrings("--matrix-lr 0.03", app.values[train][16].items);
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
    tui.DrawContext.init(.unicode);
    const ctx: tui.DrawContext = .{ .arena = arena.allocator(), .min = .{ .width = 0, .height = 0 }, .max = .{ .width = 100, .height = 30 }, .cell_size = .{ .width = 10, .height = 20 } };
    for (0..operations.len) |i| {
        app.nav = i;
        app.focus = if (operations[i].command != null and operations[i].phase() == null) .form else .nav;
        const surface = try app.draw(ctx);
        try std.testing.expectEqual(@as(u16, 100), surface.size.width);
        // The split (titles, rules, panes) over the status line.
        try std.testing.expectEqual(@as(usize, 2), surface.children.len);
        try std.testing.expectEqual(@as(u16, 29), surface.children[0].surface.size.height);
        try std.testing.expectEqual(@as(u16, 29), surface.children[1].origin.row);
        // Titles, top rule, panes, bottom rule.
        try std.testing.expectEqual(@as(usize, 4), surface.children[0].surface.children.len);
    }
}

test "console chat page loads a model, sends a message and draws the transcript" {
    const allocator = std.testing.allocator;
    var support: TestSupport = undefined;
    try TestSupport.create(allocator, &support);
    defer support.destroy();
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
    try press(&app, &ctx, .{ .codepoint = tui.Key.enter });
    try std.testing.expectEqual(@as(usize, 0), app.input.value().len);
    support.chat.thread.?.join();
    support.chat.thread = null;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const snap = try support.chat.snapshot(arena.allocator());
    try std.testing.expectEqual(@as(usize, 3), snap.turns.len);
    try std.testing.expectEqualStrings("Hi", snap.turns[1].text);
    tui.DrawContext.init(.unicode);
    const draw_ctx: tui.DrawContext = .{ .arena = arena.allocator(), .min = .{ .width = 0, .height = 0 }, .max = .{ .width = 100, .height = 30 }, .cell_size = .{ .width = 10, .height = 20 } };
    const surface = try app.draw(draw_ctx);
    try std.testing.expectEqual(@as(u16, 100), surface.size.width);
    // "quit" leaves the conversation; tab opens the settings.
    try press(&app, &ctx, .{ .codepoint = tui.Key.tab });
    try std.testing.expectEqual(Focus.form, app.focus);
    try press(&app, &ctx, .{ .codepoint = tui.Key.escape });
    try std.testing.expectEqual(Focus.chat, app.focus);
    for ("quit") |c| try press(&app, &ctx, .{ .codepoint = c, .text = &.{c} });
    try press(&app, &ctx, .{ .codepoint = tui.Key.enter });
    try std.testing.expectEqual(Focus.nav, app.focus);
}

/// A drawn surface as text, one line per row: every cell of the surface and
/// its children at their origins (blank cells as spaces).
fn screenText(a: std.mem.Allocator, surface: tui.Surface) ![]const u8 {
    const w = surface.size.width;
    const grid = try a.alloc([]const u8, @as(usize, w) * surface.size.height);
    @memset(grid, " ");
    paint(grid, w, surface.size.height, surface, 0, 0);
    var out: std.ArrayList(u8) = .empty;
    for (0..surface.size.height) |r| {
        for (grid[r * w ..][0..w]) |g| try out.appendSlice(a, g);
        try out.append(a, '\n');
    }
    return out.items;
}

fn paint(grid: [][]const u8, w: u16, h: u16, surface: tui.Surface, row: i32, col: i32) void {
    for (surface.buffer, 0..) |cell, i| {
        const r = row + @as(i32, @intCast(i / surface.size.width));
        const c = col + @as(i32, @intCast(i % surface.size.width));
        if (r < 0 or c < 0 or r >= h or c >= w or cell.char.grapheme.len == 0) continue;
        grid[@as(usize, @intCast(r)) * w + @as(usize, @intCast(c))] = cell.char.grapheme;
    }
    for (surface.children) |child| paint(grid, w, h, child.surface, row + child.origin.row, col + child.origin.col);
}

test "console wraps an operation's summary to the pane instead of cutting it" {
    const allocator = std.testing.allocator;
    var support: TestSupport = undefined;
    try TestSupport.create(allocator, &support);
    defer support.destroy();
    var app = try ConsoleApp.create(allocator, support.processInit(allocator), &support.job, &support.chat);
    defer app.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    tui.DrawContext.init(.unicode);
    // The width of the screenshot's terminal.
    const ctx: tui.DrawContext = .{ .arena = a, .min = .{ .width = 0, .height = 0 }, .max = .{ .width = 80, .height = 24 }, .cell_size = .{ .width = 10, .height = 20 } };
    app.nav = cli.Operation.indexOf(.download).?;
    const text = try screenText(a, try app.draw(ctx));
    try std.testing.expect(std.mem.find(u8, text, "…") == null);
    try std.testing.expect(std.mem.find(u8, text, "always comes too).") != null);
}

test "console shows a running job's progress and the RL training view" {
    const allocator = std.testing.allocator;
    var support: TestSupport = undefined;
    try TestSupport.create(allocator, &support);
    defer support.destroy();
    var app = try ConsoleApp.create(allocator, support.processInit(allocator), &support.job, &support.chat);
    defer app.deinit();
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    tui.DrawContext.init(.unicode);
    const ctx: tui.DrawContext = .{ .arena = a, .min = .{ .width = 0, .height = 0 }, .max = .{ .width = 100, .height = 30 }, .cell_size = .{ .width = 10, .height = 20 } };

    // An RL job mid-step: its rollouts' progress over the log (no step yet).
    const rl = cli.Operation.indexOf(.rl).?;
    app.nav = rl;
    app.focus = .form;
    support.job.operation = rl;
    support.job.state = .running;
    const obs = support.job.observer();
    mod.TrainObserver.progress(obs, "rollouts", 3, 16);
    var text = try screenText(a, try app.draw(ctx));
    try std.testing.expect(std.mem.find(u8, text, "rollouts  ") != null);
    try std.testing.expect(std.mem.find(u8, text, " 3/16") != null);

    // After a step and an eval: the RL view, and the finished step cleared the progress.
    obs.onEval(obs.context, 0, 0.25);
    obs.onRlStep(obs.context, .{ .step = 0, .num_steps = 4, .reward = 0.5, .sequence_length = 120, .lrm = 1, .dt = 2, .total_time = 2 });
    text = try screenText(a, try app.draw(ctx));
    try std.testing.expect(std.mem.find(u8, text, "rollouts") == null);
    try std.testing.expect(std.mem.find(u8, text, "1/4  eta 0.1m") != null);
    try std.testing.expect(std.mem.find(u8, text, "reward 0.5000 · sequence length 120.0") != null);
    try std.testing.expect(std.mem.find(u8, text, "pass@1  0.2500@0") != null);

    // A finished job shows no progress.
    mod.TrainObserver.progress(obs, "rollouts", 5, 16);
    support.job.state = .succeeded;
    text = try screenText(a, try app.draw(ctx));
    try std.testing.expect(std.mem.find(u8, text, "5/16") == null);
}
