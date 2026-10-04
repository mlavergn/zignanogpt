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

    /// The value of `--name`, if given.
    ///
    /// Parameters:
    /// - `self`: the parser.
    /// - `name`: without the dashes.
    ///
    /// Return: the value; `error.MissingValue` for a trailing `--name`.
    pub fn string(self: *Self, name: []const u8) !?[]const u8 {
        for (self.items, 0..) |item, i| {
            if (!std.mem.startsWith(u8, item, "--") or !std.mem.startsWith(u8, item[2..], name)) continue;
            const rest = item[2 + name.len ..];
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
    var args = try cli.Args.init(std.testing.allocator, &.{ "--vocab-size=512", "--data", "x.txt", "--verbose", "--max-chars", "10" });
    defer args.deinit();
    try std.testing.expectEqual(@as(usize, 512), try args.int(usize, "vocab-size", 0));
    try std.testing.expectEqualStrings("x.txt", (try args.string("data")).?);
    try std.testing.expect(args.flag("verbose"));
    try std.testing.expectEqual(@as(usize, 7), try args.int(usize, "doc-cap", 7));
    try std.testing.expectError(error.UnknownArgument, args.finish());
    _ = try args.int(usize, "max-chars", 0);
    try args.finish();
}
