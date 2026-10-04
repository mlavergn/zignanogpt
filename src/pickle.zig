const std = @import("std");
const log = std.log.scoped(.zignanogpt_pickle);
const mod = @import("module.zig");

/// A dictionary entry, in insertion order.
pub const PickleEntry = struct { key: PickleValue, value: PickleValue };

/// A pickled value. Calls are not executed: `REDUCE` and `NEWOBJ` record the
/// callable and its arguments, and the caller interprets the few it knows
/// (`torch._utils._rebuild_tensor_v2`, `tiktoken.core.Encoding`, ...).
pub const PickleValue = union(enum) {
    none,
    boolean: bool,
    int: i64,
    float: f64,
    string: []const u8,
    bytes: []const u8,
    tuple: []const PickleValue,
    list: *std.ArrayList(PickleValue),
    dict: *std.ArrayList(PickleEntry),
    global: struct { module: []const u8, name: []const u8 },
    /// `BINPERSID`: an id the unpickler resolves (torch: storage references).
    persid: *const PickleValue,
    /// `REDUCE` (`func(*args)`) or `NEWOBJ` (`cls.__new__(cls, *args)`), plus `BUILD`'s state.
    call: *Call,

    pub const Call = struct {
        func: PickleValue,
        args: PickleValue,
        state: ?PickleValue = null,
    };

    /// Looks up a string key in a dict value.
    ///
    /// Parameters:
    /// - `self`: a dict.
    /// - `key`: the key.
    ///
    /// Return: the value, or null when absent or `self` is not a dict.
    pub fn get(self: PickleValue, key: []const u8) ?PickleValue {
        if (self != .dict) return null;
        for (self.dict.items) |entry| {
            if (entry.key == .string and std.mem.eql(u8, entry.key.string, key)) return entry.value;
        }
        return null;
    }

    /// Whether this is a global (or a call of one) named `module.name`.
    pub fn isGlobal(self: PickleValue, module: []const u8, name: []const u8) bool {
        return switch (self) {
            .global => |g| std.mem.eql(u8, g.module, module) and std.mem.eql(u8, g.name, name),
            else => false,
        };
    }

    /// A tuple's items (also accepts lists).
    pub fn items(self: PickleValue) ?[]const PickleValue {
        return switch (self) {
            .tuple => |t| t,
            .list => |l| l.items,
            else => null,
        };
    }
};

/// A pickle (protocols 2-4) decoded into `PickleValue`s, without executing
/// anything. Supports the opcodes `torch.save` and `pickle.dump` of a tiktoken
/// encoding produce. Everything is allocated in one arena, freed by `deinit`.
pub const Pickle = struct {
    const Self = @This();

    arena: std.heap.ArenaAllocator,
    root: PickleValue,

    /// Decodes a pickle.
    ///
    /// Parameters:
    /// - `allocator`: backs the arena.
    /// - `data`: the pickle bytes; strings borrow from them, so keep them alive.
    ///
    /// Return: the decoded tree; `error.InvalidPickle`, `error.UnsupportedPickle`.
    pub fn parse(allocator: std.mem.Allocator, data: []const u8) !Self {
        log.debug("{s}:{d} :: {s}", .{ @src().file, @src().line, @src().fn_name });
        var self = Self{ .arena = std.heap.ArenaAllocator.init(allocator), .root = .none };
        errdefer self.arena.deinit();
        self.root = try run(self.arena.allocator(), data);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }

    const Mark = struct { at: usize };

    fn run(a: std.mem.Allocator, data: []const u8) !PickleValue {
        var stack: std.ArrayList(PickleValue) = .empty;
        var marks: std.ArrayList(usize) = .empty;
        var memo: std.AutoHashMapUnmanaged(u32, PickleValue) = .empty;
        var memo_next: u32 = 0;
        var r = Reader{ .data = data };
        while (true) {
            const op = try r.byte();
            switch (op) {
                0x80 => _ = try r.byte(), // PROTO
                0x95 => _ = try r.take(8), // FRAME
                '.' => return stack.pop() orelse error.InvalidPickle, // STOP
                '(' => try marks.append(a, stack.items.len), // MARK
                '0' => _ = stack.pop() orelse return error.InvalidPickle, // POP
                '}' => {
                    const d = try a.create(std.ArrayList(PickleEntry));
                    d.* = .empty;
                    try stack.append(a, .{ .dict = d });
                },
                ']' => {
                    const l = try a.create(std.ArrayList(PickleValue));
                    l.* = .empty;
                    try stack.append(a, .{ .list = l });
                },
                ')' => try stack.append(a, .{ .tuple = &.{} }),
                't' => {
                    const start = marks.pop() orelse return error.InvalidPickle;
                    const items = try a.dupe(PickleValue, stack.items[start..]);
                    stack.shrinkRetainingCapacity(start);
                    try stack.append(a, .{ .tuple = items });
                },
                0x85, 0x86, 0x87 => { // TUPLE1..3
                    const n: usize = op - 0x84;
                    if (stack.items.len < n) return error.InvalidPickle;
                    const items = try a.dupe(PickleValue, stack.items[stack.items.len - n ..]);
                    stack.shrinkRetainingCapacity(stack.items.len - n);
                    try stack.append(a, .{ .tuple = items });
                },
                'q' => try memo.put(a, try r.byte(), try top(&stack)), // BINPUT
                'r' => try memo.put(a, try r.int(u32), try top(&stack)), // LONG_BINPUT
                0x94 => { // MEMOIZE
                    try memo.put(a, memo_next, try top(&stack));
                    memo_next += 1;
                },
                'h' => try stack.append(a, memo.get(try r.byte()) orelse return error.InvalidPickle), // BINGET
                'j' => try stack.append(a, memo.get(try r.int(u32)) orelse return error.InvalidPickle), // LONG_BINGET
                'J' => try stack.append(a, .{ .int = try r.int(i32) }), // BININT
                'K' => try stack.append(a, .{ .int = try r.byte() }), // BININT1
                'M' => try stack.append(a, .{ .int = try r.int(u16) }), // BININT2
                0x8a => { // LONG1: little-endian two's complement, up to 8 bytes
                    const n = try r.byte();
                    if (n > 8) return unsupported("integer wider than 64 bits");
                    const bytes = try r.take(n);
                    var v: i64 = 0;
                    for (bytes, 0..) |b, i| v |= @as(i64, b) << @intCast(8 * i);
                    if (n > 0 and n < 8 and bytes[n - 1] & 0x80 != 0) v -= @as(i64, 1) << @intCast(8 * n);
                    try stack.append(a, .{ .int = v });
                },
                'G' => try stack.append(a, .{ .float = @bitCast(std.mem.readInt(u64, (try r.take(8))[0..8], .big)) }), // BINFLOAT
                'N' => try stack.append(a, .none),
                0x88 => try stack.append(a, .{ .boolean = true }),
                0x89 => try stack.append(a, .{ .boolean = false }),
                'X' => try stack.append(a, .{ .string = try r.take(try r.int(u32)) }), // BINUNICODE
                0x8c => try stack.append(a, .{ .string = try r.take(try r.byte()) }), // SHORT_BINUNICODE
                0x8d => try stack.append(a, .{ .string = try r.take(try r.len64()) }), // BINUNICODE8
                'B' => try stack.append(a, .{ .bytes = try r.take(try r.int(u32)) }), // BINBYTES
                'C' => try stack.append(a, .{ .bytes = try r.take(try r.byte()) }), // SHORT_BINBYTES
                0x8e => try stack.append(a, .{ .bytes = try r.take(try r.len64()) }), // BINBYTES8
                'c' => { // GLOBAL "module\nname\n"
                    const module = try r.line();
                    const name = try r.line();
                    try stack.append(a, .{ .global = .{ .module = module, .name = name } });
                },
                0x93 => { // STACK_GLOBAL
                    const name = stack.pop() orelse return error.InvalidPickle;
                    const module = stack.pop() orelse return error.InvalidPickle;
                    if (name != .string or module != .string) return error.InvalidPickle;
                    try stack.append(a, .{ .global = .{ .module = module.string, .name = name.string } });
                },
                'Q' => { // BINPERSID
                    const id = try a.create(PickleValue);
                    id.* = stack.pop() orelse return error.InvalidPickle;
                    try stack.append(a, .{ .persid = id });
                },
                'R', 0x81 => { // REDUCE, NEWOBJ
                    const args = stack.pop() orelse return error.InvalidPickle;
                    const func = stack.pop() orelse return error.InvalidPickle;
                    if (func.isGlobal("collections", "OrderedDict")) {
                        const d = try a.create(std.ArrayList(PickleEntry));
                        d.* = .empty;
                        try stack.append(a, .{ .dict = d });
                    } else {
                        const call = try a.create(PickleValue.Call);
                        call.* = .{ .func = func, .args = args };
                        try stack.append(a, .{ .call = call });
                    }
                },
                'b' => { // BUILD
                    const state = stack.pop() orelse return error.InvalidPickle;
                    const target = try top(&stack);
                    switch (target) {
                        .call => |call| call.state = state,
                        .dict => {}, // OrderedDict state: nothing we need
                        else => return unsupported("BUILD on a non-object"),
                    }
                },
                's' => { // SETITEM
                    const value = stack.pop() orelse return error.InvalidPickle;
                    const key = stack.pop() orelse return error.InvalidPickle;
                    const target = try top(&stack);
                    if (target != .dict) return error.InvalidPickle;
                    try target.dict.append(a, .{ .key = key, .value = value });
                },
                'u' => { // SETITEMS
                    const start = marks.pop() orelse return error.InvalidPickle;
                    const pairs = stack.items[start..];
                    if (pairs.len % 2 != 0 or start == 0) return error.InvalidPickle;
                    const target = stack.items[start - 1];
                    if (target != .dict) return error.InvalidPickle;
                    var i: usize = 0;
                    while (i < pairs.len) : (i += 2) try target.dict.append(a, .{ .key = pairs[i], .value = pairs[i + 1] });
                    stack.shrinkRetainingCapacity(start);
                },
                'a' => { // APPEND
                    const value = stack.pop() orelse return error.InvalidPickle;
                    const target = try top(&stack);
                    if (target != .list) return error.InvalidPickle;
                    try target.list.append(a, value);
                },
                'e' => { // APPENDS
                    const start = marks.pop() orelse return error.InvalidPickle;
                    if (start == 0 or stack.items[start - 1] != .list) return error.InvalidPickle;
                    try stack.items[start - 1].list.appendSlice(a, stack.items[start..]);
                    stack.shrinkRetainingCapacity(start);
                },
                else => {
                    log.warn("unsupported pickle opcode 0x{x} at byte {d}", .{ op, r.pos - 1 });
                    return error.UnsupportedPickle;
                },
            }
        }
    }

    fn top(stack: *std.ArrayList(PickleValue)) !PickleValue {
        if (stack.items.len == 0) return error.InvalidPickle;
        return stack.items[stack.items.len - 1];
    }

    fn unsupported(what: []const u8) error{UnsupportedPickle} {
        log.warn("unsupported pickle content: {s}", .{what});
        return error.UnsupportedPickle;
    }

    const Reader = struct {
        data: []const u8,
        pos: usize = 0,

        fn byte(r: *Reader) !u8 {
            if (r.pos >= r.data.len) return error.InvalidPickle;
            defer r.pos += 1;
            return r.data[r.pos];
        }

        fn take(r: *Reader, n: usize) ![]const u8 {
            if (n > r.data.len - r.pos) return error.InvalidPickle;
            defer r.pos += n;
            return r.data[r.pos..][0..n];
        }

        fn int(r: *Reader, comptime T: type) !T {
            return std.mem.readInt(T, (try r.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
        }

        fn len64(r: *Reader) !usize {
            return std.math.cast(usize, try r.int(u64)) orelse error.InvalidPickle;
        }

        fn line(r: *Reader) ![]const u8 {
            const rest = r.data[r.pos..];
            const end = std.mem.indexOfScalar(u8, rest, '\n') orelse return error.InvalidPickle;
            r.pos += end + 1;
            return rest[0..end];
        }
    };
};

// -----------------------------------------------------------------------------
// Unit Tests

test "pickle decodes a protocol-4 tiktoken encoding" {
    // pickle.dumps(tiktoken.Encoding("x", pat_str="a", mergeable_ranks={b"a": 0, b"b": 1, b"ab": 2}, special_tokens={"<|bos|>": 3}))
    const data = "\x80\x04\x95\x88\x00\x00\x00\x00\x00\x00\x00\x8c\x0dtiktoken.core\x94\x8c\x08Encoding\x94\x93\x94)\x81\x94}\x94(\x8c\x04name\x94\x8c\x01x\x94\x8c\x07pat_str\x94\x8c\x01a\x94\x8c\x0fmergeable_ranks\x94}\x94(C\x01a\x94K\x00C\x01b\x94K\x01C\x02ab\x94K\x02u\x8c\x0especial_tokens\x94}\x94\x8c\x07<|bos|>\x94K\x03su";
    var p = try mod.Pickle.parse(std.testing.allocator, data ++ "b.");
    defer p.deinit();
    try std.testing.expect(p.root == .call);
    try std.testing.expect(p.root.call.func.isGlobal("tiktoken.core", "Encoding"));
    const state = p.root.call.state.?;
    try std.testing.expectEqualStrings("a", state.get("pat_str").?.string);
    const ranks = state.get("mergeable_ranks").?.dict.items;
    try std.testing.expectEqual(@as(usize, 3), ranks.len);
    try std.testing.expectEqualStrings("ab", ranks[2].key.bytes);
    try std.testing.expectEqual(@as(i64, 3), state.get("special_tokens").?.get("<|bos|>").?.int);
}
