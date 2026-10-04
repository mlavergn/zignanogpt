const std = @import("std");
const log = std.log.scoped(.zignanogpt_calculator);

/// nanochat's calculator tool (`use_calculator` in `engine.py`): the model
/// writes an expression between `<|python_start|>` and `<|python_end|>`, and
/// the engine forces `str(eval(expr))` back between the output tokens.
///
/// Python's `eval` is replaced by an evaluator for exactly what the filter
/// lets through: arithmetic on int and float literals (`+ - * / //`, unary
/// signs, parentheses; no `**`) and `'string'.count('sub')`. Results print
/// as Python's `str()` does. Anything Python would reject yields null, as do
/// integers beyond 128 bits (Python's are unbounded).
pub const Calculator = struct {
    const Self = @This();

    /// The characters of a "pure math" expression.
    const math_chars = "0123456789*+-/.() ";
    /// The characters of a string expression.
    const string_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ ";
    const dangerous = [_][]const u8{
        "__",      "import", "exec", "eval", "compile", "open",    "file",    "input",   "raw_input",
        "globals", "locals", "vars", "dir",  "getattr", "setattr", "delattr", "hasattr",
    };
    /// Python's parser nesting limit.
    const max_depth = 200;

    /// A Python number.
    const Value = union(enum) {
        int: i128,
        float: f64,

        fn toFloat(v: Value) f64 {
            return switch (v) {
                .int => |i| @floatFromInt(i),
                .float => |f| f,
            };
        }
    };

    /// Evaluates a tool expression.
    ///
    /// Parameters:
    /// - `allocator`: allocates the result.
    /// - `expr`: the decoded expression.
    ///
    /// Return: `str(result)` owned by the caller, or null when nanochat's tool
    /// returns None; allocation errors.
    pub fn evaluate(allocator: std.mem.Allocator, expr_in: []const u8) !?[]u8 {
        // Commas are dropped ("1,234" is a number).
        const expr = try std.mem.replaceOwned(u8, allocator, expr_in, ",", "");
        defer allocator.free(expr);
        if (onlyChars(expr, math_chars)) {
            if (std.mem.indexOf(u8, expr, "**") != null) return null;
            var parser = Arithmetic{ .text = expr };
            const value = parser.parse() catch |err| {
                log.debug("calculator rejected {s} [{t}]", .{ expr, err });
                return null;
            };
            return try format(allocator, value);
        }
        if (!onlyChars(expr, string_chars)) return null;
        const lower = try std.ascii.allocLowerString(allocator, expr);
        defer allocator.free(lower);
        for (dangerous) |pattern| {
            if (std.mem.indexOf(u8, lower, pattern) != null) return null;
        }
        if (std.mem.indexOf(u8, expr, ".count(") == null) return null;
        const count = countCall(allocator, expr) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => {
                log.debug("calculator rejected {s} [{t}]", .{ expr, err });
                return null;
            },
        };
        return try std.fmt.allocPrint(allocator, "{d}", .{count});
    }

    fn onlyChars(text: []const u8, allowed: []const u8) bool {
        for (text) |ch| {
            if (std.mem.indexOfScalar(u8, allowed, ch) == null) return false;
        }
        return true;
    }

    /// Python's `str()` of a number.
    fn format(allocator: std.mem.Allocator, value: Value) ![]u8 {
        return switch (value) {
            .int => |i| std.fmt.allocPrint(allocator, "{d}", .{i}),
            .float => |f| formatFloat(allocator, f),
        };
    }

    /// Python's float repr: the shortest round-trip digits, positional for
    /// decimal exponents in [-4, 16), otherwise `d.ddde+XX`.
    fn formatFloat(allocator: std.mem.Allocator, f: f64) ![]u8 {
        if (std.math.isNan(f)) return allocator.dupe(u8, "nan");
        if (std.math.isInf(f)) return allocator.dupe(u8, if (f < 0) "-inf" else "inf");
        // Zig's `{e}` gives the same shortest digits: "[-]d[.ddd]e[-]x".
        var buf: [64]u8 = undefined;
        const sci = try std.fmt.bufPrint(&buf, "{e}", .{f});
        const negative = sci[0] == '-';
        const body = if (negative) sci[1..] else sci;
        const e_at = std.mem.indexOfScalar(u8, body, 'e') orelse return error.InvalidFormat;
        const exponent = try std.fmt.parseInt(i32, body[e_at + 1 ..], 10);
        var digits_buf: [32]u8 = undefined;
        var n: usize = 0;
        for (body[0..e_at]) |ch| {
            if (ch == '.') continue;
            digits_buf[n] = ch;
            n += 1;
        }
        const digits = digits_buf[0..n];
        const sign: []const u8 = if (negative) "-" else "";

        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        try out.appendSlice(allocator, sign);
        if (exponent < -4 or exponent >= 16) {
            try out.append(allocator, digits[0]);
            if (digits.len > 1) {
                try out.append(allocator, '.');
                try out.appendSlice(allocator, digits[1..]);
            }
            try out.print(allocator, "e{c}{d:0>2}", .{ @as(u8, if (exponent < 0) '-' else '+'), @abs(exponent) });
        } else if (exponent < 0) {
            try out.appendSlice(allocator, "0.");
            try out.appendNTimes(allocator, '0', @intCast(-exponent - 1));
            try out.appendSlice(allocator, digits);
        } else {
            const whole: usize = @intCast(exponent + 1);
            if (digits.len <= whole) {
                try out.appendSlice(allocator, digits);
                try out.appendNTimes(allocator, '0', whole - digits.len);
                try out.appendSlice(allocator, ".0");
            } else {
                try out.appendSlice(allocator, digits[0..whole]);
                try out.append(allocator, '.');
                try out.appendSlice(allocator, digits[whole..]);
            }
        }
        return out.toOwnedSlice(allocator);
    }

    /// Recursive descent over Python's arithmetic grammar:
    /// `sum: term (('+'|'-') term)*`, `term: factor (('*'|'/'|'//') factor)*`,
    /// `factor: ('+'|'-') factor | atom`, `atom: NUMBER | '(' sum ')'`.
    const Arithmetic = struct {
        text: []const u8,
        at: usize = 0,
        depth: usize = 0,

        fn parse(p: *Arithmetic) !Value {
            const value = try p.sum();
            p.skipSpaces();
            if (p.at != p.text.len) return error.Syntax;
            return value;
        }

        fn sum(p: *Arithmetic) anyerror!Value {
            var left = try p.term();
            while (true) {
                p.skipSpaces();
                if (p.eat("+")) {
                    left = try add(left, try p.term(), .add);
                } else if (p.eat("-")) {
                    left = try add(left, try p.term(), .sub);
                } else return left;
            }
        }

        fn term(p: *Arithmetic) !Value {
            var left = try p.factor();
            while (true) {
                p.skipSpaces();
                if (p.eat("*")) {
                    left = try multiply(left, try p.factor());
                } else if (p.eat("//")) {
                    left = try floorDivide(left, try p.factor());
                } else if (p.eat("/")) {
                    left = try divide(left, try p.factor());
                } else return left;
            }
        }

        fn factor(p: *Arithmetic) anyerror!Value {
            p.skipSpaces();
            if (p.eat("+")) return p.nested(.plus);
            if (p.eat("-")) return p.nested(.minus);
            return p.atom();
        }

        fn nested(p: *Arithmetic, op: enum { plus, minus }) !Value {
            p.depth += 1;
            defer p.depth -= 1;
            if (p.depth > max_depth) return error.TooDeep;
            const v = try p.factor();
            if (op == .plus) return v;
            return switch (v) {
                .int => |i| .{ .int = std.math.negate(i) catch return error.Overflow },
                .float => |f| .{ .float = -f },
            };
        }

        fn atom(p: *Arithmetic) !Value {
            p.skipSpaces();
            if (p.eat("(")) {
                p.depth += 1;
                defer p.depth -= 1;
                if (p.depth > max_depth) return error.TooDeep;
                const v = try p.sum();
                p.skipSpaces();
                if (!p.eat(")")) return error.Syntax;
                return p.noCall(v);
            }
            return p.noCall(try p.number());
        }

        /// `2(3)` and `(2)(3)` are calls in Python: a TypeError.
        fn noCall(p: *Arithmetic, v: Value) !Value {
            p.skipSpaces();
            if (p.at < p.text.len and (p.text[p.at] == '(' or std.ascii.isDigit(p.text[p.at]) or p.text[p.at] == '.')) return error.Syntax;
            return v;
        }

        fn number(p: *Arithmetic) !Value {
            const start = p.at;
            while (p.at < p.text.len and std.ascii.isDigit(p.text[p.at])) p.at += 1;
            const int_end = p.at;
            var is_float = false;
            if (p.at < p.text.len and p.text[p.at] == '.') {
                is_float = true;
                p.at += 1;
                while (p.at < p.text.len and std.ascii.isDigit(p.text[p.at])) p.at += 1;
            }
            const literal = p.text[start..p.at];
            if (literal.len == 0 or std.mem.eql(u8, literal, ".")) return error.Syntax;
            if (is_float) return .{ .float = try std.fmt.parseFloat(f64, literal) };
            // "007" is a syntax error; "000" is zero.
            if (int_end - start > 1 and literal[0] == '0' and std.mem.indexOfNone(u8, literal, "0") != null) return error.Syntax;
            return .{ .int = try std.fmt.parseInt(i128, literal, 10) };
        }

        fn eat(p: *Arithmetic, token: []const u8) bool {
            if (!std.mem.startsWith(u8, p.text[p.at..], token)) return false;
            p.at += token.len;
            return true;
        }

        fn skipSpaces(p: *Arithmetic) void {
            while (p.at < p.text.len and p.text[p.at] == ' ') p.at += 1;
        }
    };

    fn add(a: Value, b: Value, op: enum { add, sub }) !Value {
        if (a == .int and b == .int) {
            return .{ .int = switch (op) {
                .add => std.math.add(i128, a.int, b.int) catch return error.Overflow,
                .sub => std.math.sub(i128, a.int, b.int) catch return error.Overflow,
            } };
        }
        return .{ .float = switch (op) {
            .add => a.toFloat() + b.toFloat(),
            .sub => a.toFloat() - b.toFloat(),
        } };
    }

    fn multiply(a: Value, b: Value) !Value {
        if (a == .int and b == .int) return .{ .int = std.math.mul(i128, a.int, b.int) catch return error.Overflow };
        return .{ .float = a.toFloat() * b.toFloat() };
    }

    fn divide(a: Value, b: Value) !Value {
        const d = b.toFloat();
        if (d == 0) return error.ZeroDivision;
        return .{ .float = a.toFloat() / d };
    }

    /// Python's `//`: floor division for ints; for floats CPython's
    /// `float_floor_div` (via fmod, so `7.5 // 2 == 3.0`).
    fn floorDivide(a: Value, b: Value) !Value {
        if (a == .int and b == .int) {
            if (b.int == 0) return error.ZeroDivision;
            return .{ .int = std.math.divFloor(i128, a.int, b.int) catch return error.Overflow };
        }
        const vx = a.toFloat();
        const wx = b.toFloat();
        if (wx == 0) return error.ZeroDivision;
        const m = @rem(vx, wx); // C fmod: the dividend's sign
        var div = (vx - m) / wx;
        if (m != 0 and ((wx < 0) != (m < 0))) div -= 1.0;
        if (div == 0) return .{ .float = std.math.copysign(@as(f64, 0), vx / wx) };
        var floored = @floor(div);
        if (div - floored > 0.5) floored += 1.0;
        return .{ .float = floored };
    }

    /// `'haystack'.count('needle')`, each side one or more adjacent string
    /// literals (Python concatenates them), with optional spaces.
    fn countCall(allocator: std.mem.Allocator, expr: []const u8) !usize {
        var at: usize = 0;
        skip(expr, &at);
        const haystack = try strings(allocator, expr, &at);
        defer allocator.free(haystack);
        skip(expr, &at);
        if (!std.mem.startsWith(u8, expr[at..], ".")) return error.Syntax;
        at += 1;
        skip(expr, &at);
        if (!std.mem.startsWith(u8, expr[at..], "count")) return error.Syntax;
        at += "count".len;
        skip(expr, &at);
        if (!std.mem.startsWith(u8, expr[at..], "(")) return error.Syntax;
        at += 1;
        skip(expr, &at);
        const needle = try strings(allocator, expr, &at);
        defer allocator.free(needle);
        skip(expr, &at);
        if (!std.mem.startsWith(u8, expr[at..], ")")) return error.Syntax;
        at += 1;
        skip(expr, &at);
        if (at != expr.len) return error.Syntax;
        // Python: an empty needle matches between every character.
        if (needle.len == 0) return haystack.len + 1;
        return std.mem.count(u8, haystack, needle);
    }

    fn strings(allocator: std.mem.Allocator, expr: []const u8, at: *usize) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        var any = false;
        while (true) {
            skip(expr, at);
            if (at.* >= expr.len or (expr[at.*] != '\'' and expr[at.*] != '"')) break;
            const quote = expr[at.*];
            const end = std.mem.indexOfScalarPos(u8, expr, at.* + 1, quote) orelse return error.Syntax;
            try out.appendSlice(allocator, expr[at.* + 1 .. end]);
            at.* = end + 1;
            any = true;
        }
        if (!any) return error.Syntax;
        return out.toOwnedSlice(allocator);
    }

    fn skip(expr: []const u8, at: *usize) void {
        while (at.* < expr.len and expr[at.*] == ' ') at.* += 1;
    }
};

// -----------------------------------------------------------------------------
// Unit Tests

fn expectCalc(expected: ?[]const u8, expr: []const u8) !void {
    const got = try Calculator.evaluate(std.testing.allocator, expr);
    defer if (got) |g| std.testing.allocator.free(g);
    if (expected) |e| {
        if (got == null) {
            std.debug.print("calculator {s}: expected {s}, got None\n", .{ expr, e });
            return error.TestUnexpectedResult;
        }
        try std.testing.expectEqualStrings(e, got.?);
    } else if (got) |g| {
        std.debug.print("calculator {s}: expected None, got {s}\n", .{ expr, g });
        return error.TestUnexpectedResult;
    }
}

test "calculator arithmetic and python number formatting" {
    try expectCalc("6", "2*3");
    try expectCalc("5.0", "10/2");
    try expectCalc("0.6666666666666666", "2/3");
    try expectCalc("1234567", "1,234,567");
    try expectCalc("-6", "-2*3");
    try expectCalc("8", "5 - - 3");
    try expectCalc("3", "7 // 2");
    try expectCalc("-4", "-7 // 2");
    try expectCalc("3.0", "7.5 // 2");
    try expectCalc("1e+16", "10000000000000000.0");
    try expectCalc("1.5e-05", "0.000015");
    try expectCalc("0.0001", ".0001");
    try expectCalc("-0.0", "-0.0");
    try expectCalc("0.30000000000000004", "0.1 + 0.2");
    try expectCalc("20", "(2 + 3) * 4");
    try expectCalc(null, "2 ** 3");
    try expectCalc(null, "1/0");
    try expectCalc(null, "007");
    try expectCalc(null, "2(3)");
    try expectCalc(null, "");
    try expectCalc(null, "1 2");
}

test "calculator string counts and rejected expressions" {
    try expectCalc("3", "'strawberry'.count('r')");
    try expectCalc("2", "\"banana\" .count( 'an' )");
    try expectCalc("4", "'abc'.count('')");
    try expectCalc(null, "'abc'.upper()");
    try expectCalc(null, "__import__('os')");
    try expectCalc(null, "x.count('a')");
    try expectCalc(null, "'a' + 'b'");
}
