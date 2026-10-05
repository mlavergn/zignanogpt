const std = @import("std");
const log = std.log.scoped(.zignanogpt_objc);

/// An Objective-C object pointer.
pub const Id = *anyopaque;
/// A selector.
pub const Sel = *anyopaque;

extern "objc" fn objc_getClass(name: [*:0]const u8) ?Id;
extern "objc" fn sel_registerName(name: [*:0]const u8) Sel;
extern "objc" fn objc_msgSend() void;
extern "objc" fn objc_autoreleasePoolPush() ?*anyopaque;
extern "objc" fn objc_autoreleasePoolPop(pool: ?*anyopaque) void;

/// The Objective-C runtime through its C entry points: `objc_msgSend` cast to
/// each call's exact signature (as the arm64 ABI requires), selectors by
/// name, and autorelease pools. Only what the Metal backend needs.
pub const Objc = struct {
    /// A class object by name.
    pub fn class(name: [*:0]const u8) !Id {
        return objc_getClass(name) orelse {
            log.warn("Objective-C class {s} not found", .{name});
            return error.ObjcClassNotFound;
        };
    }

    pub fn sel(name: [*:0]const u8) Sel {
        return sel_registerName(name);
    }

    /// `[obj sel]`.
    pub fn call0(comptime R: type, obj: Id, name: [*:0]const u8) R {
        const F = *const fn (Id, Sel) callconv(.c) R;
        return @as(F, @ptrCast(&objc_msgSend))(obj, sel(name));
    }

    /// `[obj sel:a]`.
    pub fn call1(comptime R: type, comptime A: type, obj: Id, name: [*:0]const u8, a: A) R {
        const F = *const fn (Id, Sel, A) callconv(.c) R;
        return @as(F, @ptrCast(&objc_msgSend))(obj, sel(name), a);
    }

    /// `[obj sel:a b:b]`.
    pub fn call2(comptime R: type, comptime A: type, comptime B: type, obj: Id, name: [*:0]const u8, a: A, b: B) R {
        const F = *const fn (Id, Sel, A, B) callconv(.c) R;
        return @as(F, @ptrCast(&objc_msgSend))(obj, sel(name), a, b);
    }

    /// `[obj sel:a b:b c:c]`.
    pub fn call3(comptime R: type, comptime A: type, comptime B: type, comptime C: type, obj: Id, name: [*:0]const u8, a: A, b: B, c: C) R {
        const F = *const fn (Id, Sel, A, B, C) callconv(.c) R;
        return @as(F, @ptrCast(&objc_msgSend))(obj, sel(name), a, b, c);
    }

    /// `objc_msgSend` as a function of type `F` (`fn (Id, Sel, ...) callconv(.c) R`),
    /// for signatures the `call*` helpers do not cover.
    pub fn function(comptime F: type) F {
        return @ptrCast(&objc_msgSend);
    }

    pub fn release(obj: Id) void {
        call0(void, obj, "release");
    }

    /// An autoreleased `NSString` from UTF-8.
    pub fn string(text: [*:0]const u8) !Id {
        return call1(?Id, [*:0]const u8, try class("NSString"), "stringWithUTF8String:", text) orelse error.ObjcStringFailed;
    }

    /// An `NSError`'s description, for logs.
    pub fn errorText(err: ?Id) []const u8 {
        const e = err orelse return "(no error object)";
        const description = call0(?Id, e, "localizedDescription") orelse return "(no description)";
        const text = call0(?[*:0]const u8, description, "UTF8String") orelse return "(no text)";
        return std.mem.span(text);
    }

    pub fn poolPush() ?*anyopaque {
        return objc_autoreleasePoolPush();
    }

    pub fn poolPop(pool: ?*anyopaque) void {
        objc_autoreleasePoolPop(pool);
    }
};
