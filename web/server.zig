const std = @import("std");
const log = std.log.scoped(.zignanogpt_server);
const web = @import("module.zig");
const mod = web.nanogpt;

/// The single page (chat and training dashboard), embedded at build time.
const page = @embedFile("page.html");

/// The web console's HTTP server: the page, `POST /api/chat` (server-sent
/// events), `GET /api/info`, `GET /api/runs` and `GET /api/metrics?run=&offset=`.
/// Each connection runs as its own task; generation is one at a time.
pub const Server = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    io: std.Io,
    base_dir: []const u8,
    /// Null when there is no checkpoint yet (the dashboard still works).
    model: ?*mod.LoadedModel,
    model_mutex: std.Io.Mutex = .init,
    seed: std.atomic.Value(u64) = .init(42),

    pub fn init(allocator: std.mem.Allocator, io: std.Io, base_dir: []const u8, model: ?*mod.LoadedModel) Self {
        return .{ .allocator = allocator, .io = io, .base_dir = base_dir, .model = model };
    }

    /// Accepts connections forever.
    ///
    /// Parameters:
    /// - `self`: the server (stable address).
    /// - `listener`: a listening socket.
    ///
    /// Return: only on an accept error.
    pub fn serve(self: *Self, listener: *std.Io.net.Server) !void {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var group: std.Io.Group = .init;
        defer group.cancel(self.io);
        while (true) {
            const stream = try listener.accept(self.io);
            group.concurrent(self.io, connection, .{ self, stream }) catch |err| {
                log.debug("serving inline [{t}]", .{err});
                connection(self, stream);
            };
        }
    }

    /// Serves requests on one connection until it closes.
    fn connection(self: *Self, stream: std.Io.net.Stream) void {
        defer stream.close(self.io);
        var read_buffer: [16 * 1024]u8 = undefined;
        var write_buffer: [16 * 1024]u8 = undefined;
        var reader = stream.reader(self.io, &read_buffer);
        var writer = stream.writer(self.io, &write_buffer);
        var http = std.http.Server.init(&reader.interface, &writer.interface);
        while (true) {
            var request = http.receiveHead() catch |err| {
                if (err != error.HttpConnectionClosing) log.debug("receive [{t}]", .{err});
                return;
            };
            self.handle(&request) catch |err| {
                log.debug("request failed [{t}]", .{err});
                return;
            };
            if (!request.head.keep_alive) return;
        }
    }

    /// Routes one request.
    pub fn handle(self: *Self, request: *std.http.Server.Request) !void {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        // The head's strings die when the body is read: copy the target first.
        const target = try a.dupe(u8, request.head.target);
        const path = target[0 .. std.mem.indexOfScalar(u8, target, '?') orelse target.len];
        const query = if (std.mem.indexOfScalar(u8, target, '?')) |q| target[q + 1 ..] else "";
        const method = request.head.method;
        log.debug("{t} {s}", .{ method, target });

        if (method == .GET and (std.mem.eql(u8, path, "/") or std.mem.eql(u8, path, "/index.html"))) {
            return request.respond(page, .{ .extra_headers = &.{.{ .name = "content-type", .value = "text/html; charset=utf-8" }} });
        }
        if (method == .GET and std.mem.eql(u8, path, "/api/info")) {
            const label: ?[]const u8 = if (self.model) |m| try std.fmt.allocPrint(a, "{s} {s} step {d}", .{ @tagName(m.kind), m.tag, m.step }) else null;
            return json(request, a, .{ .model = label, .version = mod.build_options.version }, .ok);
        }
        if (method == .GET and std.mem.eql(u8, path, "/api/runs")) {
            return json(request, a, try web.Runs.list(a, mod.Storage.init(a, self.io), self.base_dir), .ok);
        }
        if (method == .GET and std.mem.eql(u8, path, "/api/metrics")) {
            const run = try param(a, query, "run") orelse return json(request, a, .{ .@"error" = "missing run" }, .bad_request);
            const offset_text = try param(a, query, "offset") orelse "0";
            const offset = std.fmt.parseInt(usize, offset_text, 10) catch 0;
            const tail = web.Runs.tail(a, mod.Storage.init(a, self.io), self.base_dir, run, offset) catch |err| switch (err) {
                error.UnknownRun => return json(request, a, .{ .@"error" = "unknown run" }, .not_found),
                else => return err,
            };
            return json(request, a, tail, .ok);
        }
        if (method == .POST and std.mem.eql(u8, path, "/api/chat")) return self.chat(request, a);
        return json(request, a, .{ .@"error" = "not found" }, .not_found);
    }

    /// Streams a reply as server-sent events.
    fn chat(self: *Self, request: *std.http.Server.Request, a: std.mem.Allocator) !void {
        var body_buffer: [4096]u8 = undefined;
        const reader = try request.readerExpectContinue(&body_buffer);
        const body = reader.allocRemaining(a, .limited(1 << 20)) catch return json(request, a, .{ .@"error" = "request too large" }, .bad_request);
        const loaded = self.model orelse return json(request, a, .{ .@"error" = "no model: train or import one first" }, .service_unavailable);
        var send_buffer: [4096]u8 = undefined;
        var response = try request.respondStreaming(&send_buffer, .{ .respond_options = .{
            .keep_alive = false,
            .extra_headers = &.{
                .{ .name = "content-type", .value = "text/event-stream" },
                .{ .name = "cache-control", .value = "no-cache" },
            },
        } });
        // One generation at a time: the model's scratch is shared.
        self.model_mutex.lockUncancelable(self.io);
        defer self.model_mutex.unlock(self.io);
        const seed = self.seed.fetchAdd(1, .monotonic);
        try web.ChatStream.run(a, &loaded.model, &loaded.tokenizer, body, seed, &response.writer);
        try response.end();
    }

    fn json(request: *std.http.Server.Request, a: std.mem.Allocator, value: anytype, status: std.http.Status) !void {
        const text = try std.json.Stringify.valueAlloc(a, value, .{});
        try request.respond(text, .{ .status = status, .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
    }

    /// A query parameter, percent-decoded.
    fn param(a: std.mem.Allocator, query: []const u8, name: []const u8) !?[]const u8 {
        var it = std.mem.splitScalar(u8, query, '&');
        while (it.next()) |pair| {
            const eq = std.mem.indexOfScalar(u8, pair, '=') orelse continue;
            if (!std.mem.eql(u8, pair[0..eq], name)) continue;
            const raw = try a.dupe(u8, pair[eq + 1 ..]);
            std.mem.replaceScalar(u8, raw, '+', ' ');
            return std.Uri.percentDecodeInPlace(raw);
        }
        return null;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "server query parameters decode" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqualStrings("base_checkpoints/d6", (try Server.param(arena.allocator(), "offset=3&run=base_checkpoints%2Fd6", "run")).?);
    try std.testing.expect(try Server.param(arena.allocator(), "offset=3", "run") == null);
}
