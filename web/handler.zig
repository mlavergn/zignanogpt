const std = @import("std");
const log = std.log.scoped(.zignanogpt_handler);
const zigwui = @import("zigwui");
const web = @import("module.zig");
const mod = web.nanogpt;

/// The web console's routes, for zigwui's `Server`: `POST /api/chat`
/// (server-sent events), `GET /api/info`, `GET /api/runs` and
/// `GET /api/metrics?run=&offset=`. Everything else falls through to zigwui
/// and the Svelte bundle. zigwui answers each connection on its own task;
/// generation is one at a time.
///
/// zigwui listens on loopback only. A request must also name the server in
/// `Host` (a loopback address or `localhost`, with its port), so a DNS-rebound
/// page cannot reach it, the bundle included, and a request carrying a
/// foreign `Origin` cannot POST.
pub const Handler = struct {
    const Self = @This();

    /// Largest chat request body.
    const max_chat_body = 1 << 20;

    allocator: std.mem.Allocator,
    io: std.Io,
    base_dir: []const u8,
    /// Null when there is no checkpoint yet (the dashboard still works).
    model: ?*mod.LoadedModel,
    /// The console's port, which `Host` must name.
    port: u16,
    model_mutex: std.Io.Mutex = .init,
    seed: std.atomic.Value(u64) = .init(42),

    /// Prepares the routes for a console on `port`.
    ///
    /// Parameters:
    /// - `allocator`: thread-safe; zigwui answers connections concurrently.
    /// - `io`: drives storage and the model lock.
    /// - `base_dir`: this port's base directory (training runs); borrowed.
    /// - `model`: the loaded model, or null; borrowed.
    /// - `port`: the console's port.
    ///
    /// Return: the routes.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, base_dir: []const u8, model: ?*mod.LoadedModel, port: u16) Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        return .{ .allocator = allocator, .io = io, .base_dir = base_dir, .model = model, .port = port };
    }

    /// Answers one request, or leaves it to zigwui's bundle.
    ///
    /// Parameters:
    /// - `ctx`: the request, its path and target already copied.
    ///
    /// Return: true once answered (refusals included); false for a GET of
    /// anything outside `/api/`, which zigwui serves from the bundle.
    pub fn route(self: *Self, ctx: *const zigwui.RequestContext) !bool {
        const request = ctx.request;
        const a = ctx.allocator;
        const path = ctx.path;
        log.debug("{t} {f}", .{ ctx.method, std.ascii.hexEscape(ctx.target, .lower) });
        var host: []const u8 = "";
        var origin: ?[]const u8 = null;
        var headers = request.iterateHeaders();
        while (headers.next()) |header| {
            if (std.ascii.eqlIgnoreCase(header.name, "host")) host = try a.dupe(u8, header.value);
            if (std.ascii.eqlIgnoreCase(header.name, "origin")) origin = try a.dupe(u8, header.value);
        }
        if (!self.allowedHost(host)) {
            try reject(request, "unexpected Host");
            return true;
        }
        // Another site's page may POST here (no preflight for text/plain):
        // only this page's own origin may run the model.
        if (ctx.method != .GET and origin != null and !sameOrigin(origin.?, host)) {
            try reject(request, "cross-origin request");
            return true;
        }

        if (ctx.method == .POST and std.mem.eql(u8, path, "/api/chat")) {
            try self.chat(request, a);
            return true;
        }
        if (!std.mem.startsWith(u8, path, "/api/")) return false;
        if (ctx.method != .GET) {
            try json(request, a, .{ .@"error" = "not found" }, .not_found);
            return true;
        }
        if (std.mem.eql(u8, path, "/api/info")) {
            const label: ?[]const u8 = if (self.model) |m| try a.print("{s} {s} step {d}", .{ @tagName(m.kind), m.tag, m.step }) else null;
            try json(request, a, .{ .model = label, .version = mod.build_options.version }, .ok);
        } else if (std.mem.eql(u8, path, "/api/runs")) {
            try json(request, a, try web.Runs.list(a, mod.Storage.init(a, self.io), self.base_dir), .ok);
        } else if (std.mem.eql(u8, path, "/api/metrics")) {
            try self.metrics(ctx);
        } else {
            try json(request, a, .{ .@"error" = "not found" }, .not_found);
        }
        return true;
    }

    /// The new part of a run's `metrics.jsonl`, from `offset`.
    fn metrics(self: *Self, ctx: *const zigwui.RequestContext) !void {
        const a = ctx.allocator;
        const run = try ctx.queryValue("run") orelse return json(ctx.request, a, .{ .@"error" = "missing run" }, .bad_request);
        const offset = std.fmt.parseInt(usize, try ctx.queryValue("offset") orelse "0", 10) catch 0;
        const tail = web.Runs.tail(a, mod.Storage.init(a, self.io), self.base_dir, run, offset) catch |err| switch (err) {
            error.UnknownRun => return json(ctx.request, a, .{ .@"error" = "unknown run" }, .not_found),
            else => return err,
        };
        try json(ctx.request, a, tail, .ok);
    }

    /// Streams a reply as server-sent events.
    fn chat(self: *Self, request: *std.http.Server.Request, a: std.mem.Allocator) !void {
        var body_buffer: [4096]u8 = undefined;
        const reader = try request.readerExpectContinue(&body_buffer);
        const body = reader.allocRemaining(a, .limited(max_chat_body)) catch return json(request, a, .{ .@"error" = "request too large" }, .bad_request);
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
        var event_buffer: [1024]u8 = undefined;
        var events = EventWriter.init(&response, &event_buffer);
        try web.ChatStream.run(a, &loaded.model, &loaded.tokenizer, body, seed, &events.writer);
        try response.end();
    }

    fn json(request: *std.http.Server.Request, a: std.mem.Allocator, value: anytype, status: std.http.Status) !void {
        const text = try std.json.Stringify.valueAlloc(a, value, .{});
        try request.respond(text, .{ .status = status, .keep_alive = false, .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
    }

    /// Refuses a request (403) without reading its body.
    fn reject(request: *std.http.Server.Request, reason: []const u8) !void {
        log.warn("Refused {f} [{s}]", .{ std.ascii.hexEscape(request.head.target, .lower), reason });
        try request.respond("{\"error\":\"forbidden\"}", .{ .status = .forbidden, .keep_alive = false, .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }} });
    }

    /// Whether `host` (a `Host` header) names this server: its port, and
    /// `localhost` or a loopback IP literal.
    fn allowedHost(self: *const Self, host: []const u8) bool {
        const name, const port_text = splitHost(host) orelse return false;
        if (port_text) |text| {
            const port = std.fmt.parseInt(u16, text, 10) catch return false;
            if (port != self.port) return false;
        } else if (self.port != 80) return false;
        if (std.ascii.eqlIgnoreCase(name, "localhost")) return true;
        const address = std.Io.net.IpAddress.parse(name, 0) catch return false;
        return switch (address) {
            .ip4 => |ip| ip.bytes[0] == 127,
            .ip6 => |ip| std.mem.eql(u8, &ip.bytes, &(@as([15]u8, @splat(0)) ++ [_]u8{1})),
        };
    }

    /// A `Host` header's name (IPv6 without brackets) and port text, if any.
    fn splitHost(host: []const u8) ?struct { []const u8, ?[]const u8 } {
        if (host.len == 0) return null;
        if (host[0] == '[') {
            const close = std.mem.findScalar(u8, host, ']') orelse return null;
            const rest = host[close + 1 ..];
            if (rest.len == 0) return .{ host[1..close], null };
            if (rest[0] != ':') return null;
            return .{ host[1..close], rest[1..] };
        }
        const colon = std.mem.findScalar(u8, host, ':') orelse return .{ host, null };
        return .{ host[0..colon], host[colon + 1 ..] };
    }

    /// Whether `origin` is this server as the page names it.
    fn sameOrigin(origin: []const u8, host: []const u8) bool {
        const scheme = "http://";
        return std.mem.startsWith(u8, origin, scheme) and std.mem.eql(u8, origin[scheme.len..], host);
    }
};

/// The writer `ChatStream` sends events through, whose `flush` reaches the
/// socket. The body writer's own `flush` stops at the connection's buffer
/// (only `BodyWriter.flush` sends it on), so a reply would arrive in one piece
/// when it ended.
const EventWriter = struct {
    const Self = @This();

    body: *std.http.BodyWriter,
    writer: std.Io.Writer,

    fn init(body: *std.http.BodyWriter, buffer: []u8) Self {
        return .{ .body = body, .writer = .{ .buffer = buffer, .vtable = &.{ .drain = drain, .flush = flush } } };
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *Self = @alignCast(@fieldParentPtr("writer", w));
        try self.body.writer.writeAll(w.buffered());
        _ = w.consumeAll();
        return self.body.writer.writeSplat(data, splat);
    }

    fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *Self = @alignCast(@fieldParentPtr("writer", w));
        try w.defaultFlush();
        try self.body.writer.flush();
        try self.body.flush();
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "handler accepts only this console's own Host" {
    var handler = Handler.init(std.testing.allocator, std.testing.io, "", null, 8000);
    try std.testing.expect(handler.allowedHost("127.0.0.1:8000"));
    try std.testing.expect(handler.allowedHost("localhost:8000"));
    try std.testing.expect(handler.allowedHost("[::1]:8000"));
    try std.testing.expect(!handler.allowedHost("127.0.0.1:8001"));
    try std.testing.expect(!handler.allowedHost("attacker.example:8000"));
    try std.testing.expect(!handler.allowedHost("10.0.0.2:8000"));
    try std.testing.expect(!handler.allowedHost(""));
    try std.testing.expect(Handler.sameOrigin("http://localhost:8000", "localhost:8000"));
    try std.testing.expect(!Handler.sameOrigin("https://attacker.example", "localhost:8000"));
}

/// A console with no model and an empty bundle on a random loopback port.
const TestServer = struct {
    const Self = @This();

    const no_files = [_]zigwui.AssetFile{};

    assets: zigwui.AssetStore,
    handler: Handler,
    server: zigwui.Server(Handler),
    task: std.Io.Future(@typeInfo(@TypeOf(zigwui.Server(Handler).serve)).@"fn".return_type.?),

    fn init(self: *Self, allocator: std.mem.Allocator) !void {
        const io = std.testing.io;
        self.assets = try zigwui.AssetStore.init(allocator, io, null, &no_files);
        const port = zigwui.Server(Handler).randomPort(io);
        self.handler = Handler.init(allocator, io, "/nonexistent-zignanogpt-base", null, port);
        self.server = try zigwui.Server(Handler).init(allocator, io, &self.handler, &self.assets, .{ .port = port, .open_browser = false });
        try self.server.bind();
        self.task = try io.concurrent(zigwui.Server(Handler).serve, .{&self.server});
    }

    fn deinit(self: *Self) void {
        const io = std.testing.io;
        self.task.cancel(io) catch |err| std.debug.assert(err == error.Canceled);
        self.server.deinit();
        self.assets.deinit();
    }

    /// Sends `head`, a format string given the port as `port` (so literal
    /// braces double), and reads the whole reply.
    fn exchange(self: *Self, allocator: std.mem.Allocator, comptime head: []const u8) ![]u8 {
        const io = std.testing.io;
        var stream = try self.server.address.connect(io, .{ .mode = .stream });
        defer stream.close(io);
        var write_buffer: [1024]u8 = undefined;
        var writer = stream.writer(io, &write_buffer);
        try writer.interface.print(head, .{ .port = self.handler.port });
        try writer.interface.flush();
        var read_buffer: [4096]u8 = undefined;
        var reader = stream.reader(io, &read_buffer);
        return reader.interface.allocRemaining(allocator, .limited(1 << 20));
    }
};

test "handler serves the API over loopback, leaves the page to the bundle, and refuses foreign requests" {
    std.testing.log_level = .err;
    const allocator = std.testing.allocator;
    var harness: TestServer = undefined;
    try harness.init(allocator);
    defer harness.deinit();

    // No bundle in this test: zigwui says so, rather than the handler answering.
    const page = try harness.exchange(allocator, "GET / HTTP/1.1\r\nHost: 127.0.0.1:{[port]d}\r\nConnection: close\r\n\r\n");
    defer allocator.free(page);
    try std.testing.expect(std.mem.indexOf(u8, page, "The web UI has not been built.") != null);

    const info = try harness.exchange(allocator, "GET /api/info HTTP/1.1\r\nHost: localhost:{[port]d}\r\nConnection: close\r\n\r\n");
    defer allocator.free(info);
    try std.testing.expect(std.mem.startsWith(u8, info, "HTTP/1.1 200 OK"));
    try std.testing.expect(std.mem.indexOf(u8, info, "{\"model\":null,\"version\":\"") != null);

    const missing = try harness.exchange(allocator, "GET /api/metrics HTTP/1.1\r\nHost: 127.0.0.1:{[port]d}\r\nConnection: close\r\n\r\n");
    defer allocator.free(missing);
    try std.testing.expect(std.mem.startsWith(u8, missing, "HTTP/1.1 400 Bad Request"));

    const no_model = try harness.exchange(allocator, "POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1:{[port]d}\r\nOrigin: http://127.0.0.1:{[port]d}\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}");
    defer allocator.free(no_model);
    try std.testing.expect(std.mem.startsWith(u8, no_model, "HTTP/1.1 503 Service Unavailable"));

    const rebound = try harness.exchange(allocator, "GET / HTTP/1.1\r\nHost: attacker.example:{[port]d}\r\nConnection: close\r\n\r\n");
    defer allocator.free(rebound);
    try std.testing.expect(std.mem.startsWith(u8, rebound, "HTTP/1.1 403 Forbidden"));

    const cross = try harness.exchange(allocator, "POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1:{[port]d}\r\nOrigin: https://attacker.example\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\n{{}}");
    defer allocator.free(cross);
    try std.testing.expect(std.mem.startsWith(u8, cross, "HTTP/1.1 403 Forbidden"));

    const unknown = try harness.exchange(allocator, "GET /api/nothing HTTP/1.1\r\nHost: 127.0.0.1:{[port]d}\r\nConnection: close\r\n\r\n");
    defer allocator.free(unknown);
    try std.testing.expect(std.mem.startsWith(u8, unknown, "HTTP/1.1 404 Not Found"));
}
