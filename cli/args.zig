const std = @import("std");
const log = std.log.scoped(.zignanogpt_args);
const cli = @import("module.zig");

/// A subcommand's options: `--name=value`, `--name value`, or a bare `--flag`.
/// Every option must be consumed; `finish` rejects the rest, so a typo fails
/// loudly instead of silently using a default.
pub const Args = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    items: []const []const u8,
    used: []bool,

    /// Wraps the arguments after the subcommand name.
    ///
    /// Parameters:
    /// - `allocator`: owns the bookkeeping.
    /// - `items`: the raw arguments (borrowed).
    ///
    /// Return: the parser; allocation errors.
    pub fn init(allocator: std.mem.Allocator, items: []const []const u8) !Self {
        const used = try allocator.alloc(bool, items.len);
        @memset(used, false);
        return Self{ .allocator = allocator, .items = items, .used = used };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.used);
    }

    /// The value of `--name` (or `-n` for a one-letter name), if given.
    ///
    /// Parameters:
    /// - `self`: the parser.
    /// - `name`: without the dashes.
    ///
    /// Return: the value; `error.MissingValue` for a trailing `--name`.
    pub fn string(self: *Self, name: []const u8) !?[]const u8 {
        return self.take(name, false);
    }

    /// The first `--name` value, optionally skipping ones already consumed.
    fn take(self: *Self, name: []const u8, unused_only: bool) !?[]const u8 {
        for (self.items, 0..) |item, i| {
            if (unused_only and self.used[i]) continue;
            // `--name`, or `-n` for a one-letter name.
            const dashes: usize = if (std.mem.startsWith(u8, item, "--")) 2 else if (name.len == 1 and std.mem.startsWith(u8, item, "-")) 1 else continue;
            if (!std.mem.startsWith(u8, item[dashes..], name)) continue;
            const rest = item[dashes + name.len ..];
            if (rest.len > 0 and rest[0] == '=') {
                self.used[i] = true;
                return rest[1..];
            }
            if (rest.len == 0) {
                if (i + 1 >= self.items.len) {
                    log.warn("--{s} needs a value", .{name});
                    return error.MissingValue;
                }
                self.used[i] = true;
                self.used[i + 1] = true;
                return self.items[i + 1];
            }
        }
        return null;
    }

    /// Every value of a repeatable `--name` (or `-n`), in order.
    ///
    /// Parameters:
    /// - `self`: the parser.
    /// - `allocator`: owns the list (the values are borrowed from the arguments).
    /// - `name`: without the dashes.
    ///
    /// Return: the values (empty when absent); `error.MissingValue`.
    pub fn strings(self: *Self, allocator: std.mem.Allocator, name: []const u8) ![]const []const u8 {
        var values: std.ArrayList([]const u8) = .empty;
        errdefer values.deinit(allocator);
        while (try self.take(name, true)) |v| try values.append(allocator, v);
        return values.toOwnedSlice(allocator);
    }

    /// The arguments no option consumed that do not start with `-` (paths and
    /// the like), marked used. Call it after reading every option, so their
    /// values are already taken.
    ///
    /// Parameters:
    /// - `self`: the parser.
    /// - `allocator`: owns the list (the values are borrowed from the arguments).
    ///
    /// Return: the arguments, in order.
    pub fn positionals(self: *Self, allocator: std.mem.Allocator) ![]const []const u8 {
        var values: std.ArrayList([]const u8) = .empty;
        errdefer values.deinit(allocator);
        for (self.items, self.used) |item, *used| {
            if (used.* or std.mem.startsWith(u8, item, "-")) continue;
            used.* = true;
            try values.append(allocator, item);
        }
        return values.toOwnedSlice(allocator);
    }

    /// An integer option.
    ///
    /// Parameters:
    /// - `self`: the parser.
    /// - `T`: the integer type.
    /// - `name`: without the dashes.
    /// - `default`: when absent.
    ///
    /// Return: the value; `error.InvalidValue` when unparsable.
    pub fn int(self: *Self, comptime T: type, name: []const u8, default: T) !T {
        const text = try self.string(name) orelse return default;
        return std.fmt.parseInt(T, text, 10) catch {
            log.warn("--{s}: '{s}' is not an integer", .{ name, text });
            return error.InvalidValue;
        };
    }

    /// A float option.
    ///
    /// Parameters:
    /// - `self`: the parser.
    /// - `name`: without the dashes.
    /// - `default`: when absent.
    ///
    /// Return: the value; `error.InvalidValue` when unparsable.
    pub fn float(self: *Self, name: []const u8, default: f64) !f64 {
        const text = try self.string(name) orelse return default;
        return std.fmt.parseFloat(f64, text) catch {
            log.warn("--{s}: '{s}' is not a number", .{ name, text });
            return error.InvalidValue;
        };
    }

    /// Whether a bare `--name` flag is present.
    ///
    /// Parameters:
    /// - `self`: the parser.
    /// - `name`: without the dashes.
    ///
    /// Return: true when given.
    pub fn flag(self: *Self, name: []const u8) bool {
        for (self.items, 0..) |item, i| {
            if (std.mem.startsWith(u8, item, "--") and std.mem.eql(u8, item[2..], name)) {
                self.used[i] = true;
                return true;
            }
        }
        return false;
    }

    /// Fills a struct from `--kebab-case` flags named after its fields
    /// (`max_seq_len` <- `--max-seq-len`). `usize` and `f64` fields map Python's
    /// negative "disabled" sentinel to 0; optional fields stay null when absent.
    ///
    /// Parameters:
    /// - `self`: the parser.
    /// - `T`: a struct of `usize`, `i64`, `f64`, `[]const u8` and optional fields.
    /// - `options`: updated in place.
    ///
    /// Return: nothing; `error.InvalidValue`, `error.MissingValue`.
    pub fn fill(self: *Self, comptime T: type, options: *T) !void {
        inline for (@typeInfo(T).@"struct".fields) |field| {
            const name_flag = comptime blk: {
                var name: [field.name.len]u8 = undefined;
                for (field.name, 0..) |c, i| name[i] = if (c == '_') '-' else c;
                const final = name;
                break :blk &final;
            };
            const target = &@field(options, field.name);
            switch (field.type) {
                usize => {
                    const v = try self.int(i64, name_flag, @intCast(target.*));
                    target.* = if (v < 0) 0 else @intCast(v);
                },
                i64 => target.* = try self.int(i64, name_flag, target.*),
                f64 => {
                    const v = try self.float(name_flag, target.*);
                    target.* = if (v < 0) 0 else v;
                },
                []const u8 => if (try self.string(name_flag)) |v| {
                    target.* = v;
                },
                ?usize => if (try self.string(name_flag)) |_| {
                    const v = try self.int(i64, name_flag, 0);
                    target.* = if (v < 0) null else @intCast(v);
                },
                ?f64 => if (try self.string(name_flag)) |_| {
                    const v = try self.float(name_flag, 0);
                    target.* = if (v < 0) null else v;
                },
                ?[]const u8 => target.* = try self.string(name_flag),
                else => @compileError("unhandled option type for " ++ field.name),
            }
        }
    }

    /// Fails on any argument no option consumed.
    ///
    /// Parameters:
    /// - `self`: the parser.
    ///
    /// Return: nothing; `error.UnknownArgument`.
    pub fn finish(self: *const Self) !void {
        for (self.items, self.used) |item, used| {
            if (!used) {
                log.warn("unknown argument: {s}", .{item});
                return error.UnknownArgument;
            }
        }
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

test "args parse both spellings, flags, and reject leftovers" {
    var args = try cli.Args.init(std.testing.allocator, &.{ "--vocab-size=512", "--data", "x.txt", "--verbose", "--max-chars", "10", "-n", "3" });
    defer args.deinit();
    try std.testing.expectEqual(@as(usize, 512), try args.int(usize, "vocab-size", 0));
    try std.testing.expectEqualStrings("x.txt", (try args.string("data")).?);
    try std.testing.expect(args.flag("verbose"));
    try std.testing.expectEqual(@as(usize, 7), try args.int(usize, "doc-cap", 7));
    try std.testing.expectError(error.UnknownArgument, args.finish());
    _ = try args.int(usize, "max-chars", 0);
    try std.testing.expectEqual(@as(usize, 3), try args.int(usize, "n", 0));
    try args.finish();
}

test "args collect repeated options and positionals" {
    var args = try Args.init(std.testing.allocator, &.{ "a.txt", "--input", "b.jsonl", "--name", "mine", "--input=c/", "d.parquet", "--overwrite" });
    defer args.deinit();
    try std.testing.expectEqualStrings("mine", (try args.string("name")).?);
    const inputs = try args.strings(std.testing.allocator, "input");
    defer std.testing.allocator.free(inputs);
    try std.testing.expectEqual(@as(usize, 2), inputs.len);
    try std.testing.expectEqualStrings("b.jsonl", inputs[0]);
    try std.testing.expectEqualStrings("c/", inputs[1]);
    try std.testing.expect(args.flag("overwrite"));
    const rest = try args.positionals(std.testing.allocator);
    defer std.testing.allocator.free(rest);
    try std.testing.expectEqual(@as(usize, 2), rest.len);
    try std.testing.expectEqualStrings("a.txt", rest[0]);
    try std.testing.expectEqualStrings("d.parquet", rest[1]);
    try args.finish();
}
