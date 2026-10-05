// ---------------------------------------------------------------------------
// nvptx_fixup: a build-time tool for the CUDA backend. Zig 0.16 emits each
// exported kernel as a private function plus a public alias, which LLVM's
// NVPTX backend rejects ("aliasee must be a non-kernel function"). This
// rewrites the LLVM IR so each kernel is defined under its exported name;
// `zig cc` then lowers the IR to PTX.
//
//   nvptx_fixup <in.ll> <out.ll>
// ---------------------------------------------------------------------------

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = init.minimal.args.iterate();
    _ = args.skip();
    const in_path = args.next() orelse return error.MissingArgument;
    const out_path = args.next() orelse return error.MissingArgument;
    const cwd = std.Io.Dir.cwd();
    const text = try cwd.readFileAlloc(init.io, in_path, allocator, .limited(256 << 20));
    defer allocator.free(text);
    const fixed = try rewrite(allocator, text);
    defer allocator.free(fixed);
    try cwd.writeFile(init.io, .{ .sub_path = out_path, .data = fixed });
}

/// Drops `@pub = alias ..., ptr @priv` lines and renames each `@priv`
/// definition and use to `@pub`, removing its `private` linkage.
fn rewrite(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const Alias = struct { public: []const u8, private: []const u8 };
    var aliases: std.ArrayList(Alias) = .empty;
    defer aliases.deinit(allocator);
    var kept: std.ArrayList(u8) = .empty;
    defer kept.deinit(allocator);
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "@") and std.mem.indexOf(u8, line, " = alias ") != null) {
            const name_end = std.mem.indexOf(u8, line, " = alias ").?;
            const target = std.mem.lastIndexOf(u8, line, "ptr @") orelse return error.UnexpectedAlias;
            try aliases.append(allocator, .{ .public = line[1..name_end], .private = std.mem.trim(u8, line[target + 5 ..], " \r") });
            continue;
        }
        try kept.appendSlice(allocator, line);
        try kept.append(allocator, '\n');
    }
    var out = try kept.toOwnedSlice(allocator);
    for (aliases.items) |a| {
        const from = try std.mem.concat(allocator, u8, &.{ "@", a.private, "(" });
        defer allocator.free(from);
        const to = try std.mem.concat(allocator, u8, &.{ "@", a.public, "(" });
        defer allocator.free(to);
        const renamed = try std.mem.replaceOwned(u8, allocator, out, from, to);
        allocator.free(out);
        out = renamed;
    }
    // Kernels become externally visible definitions.
    const public = try std.mem.replaceOwned(u8, allocator, out, "define private ptx_kernel", "define ptx_kernel");
    allocator.free(out);
    return public;
}

test "nvptx fixup turns aliased kernels into named definitions" {
    const allocator = std.testing.allocator;
    const ir =
        \\@fill_f32 = alias void (ptr, float, i32), ptr @k.fill_f32
        \\define private ptx_kernel void @k.fill_f32(ptr %0, float %1, i32 %2) {
        \\  ret void
        \\}
        \\
    ;
    const fixed = try rewrite(allocator, ir);
    defer allocator.free(fixed);
    try std.testing.expect(std.mem.indexOf(u8, fixed, "alias") == null);
    try std.testing.expect(std.mem.indexOf(u8, fixed, "define ptx_kernel void @fill_f32(") != null);
}
