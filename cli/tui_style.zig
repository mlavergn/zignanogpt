const std = @import("std");
const log = std.log.scoped(.zignanogpt_tui_style);
const cli = @import("module.zig");
const vaxis = cli.vaxis;

/// The console's theme: every color it draws, named by what it is for
/// (after zigprompt's `TuiStyle`). Colors are the terminal's ANSI indices so
/// they follow its scheme, except safety orange (index 202), which reads as
/// orange everywhere. `NO_COLOR` drops them all; vaxis applies it.
pub const TuiStyle = enum {
    const Self = @This();

    /// Names: operations, field labels, values. Bright white.
    primaryText,
    /// Supporting text: summaries, hints, logs. Gray.
    secondaryText,
    /// The mode badge leading the status line. Yellow.
    statusText,
    /// Where the keys are: the cursor marker, the focused field. Green.
    cursor,
    /// A key's name in a hint. Safety orange.
    keyHint,
    /// Something that worked. Green.
    successText,
    /// Something to notice that is not a failure. Yellow.
    warningText,
    /// A failure. Red.
    errorText,
    /// Rules and the line between the panes. Gray.
    divider,
    /// Charts and progress bars. Cyan.
    chart,

    pub fn color(self: Self) vaxis.Color {
        return switch (self) {
            .primaryText => bright_white,
            .secondaryText, .divider => gray,
            .statusText, .warningText => yellow,
            .keyHint => safety_orange,
            .cursor, .successText => green,
            .errorText => red,
            .chart => cyan,
        };
    }

    pub fn style(self: Self) vaxis.Style {
        return .{ .fg = self.color() };
    }

    const safety_orange: vaxis.Color = .{ .index = 202 };
    const bright_white: vaxis.Color = .{ .index = 15 };
    const gray: vaxis.Color = .{ .index = 8 };
    const yellow: vaxis.Color = .{ .index = 3 };
    const green: vaxis.Color = .{ .index = 2 };
    const red: vaxis.Color = .{ .index = 1 };
    const cyan: vaxis.Color = .{ .index = 6 };
};

// -----------------------------------------------------------------------------
// Unit Tests

test "tui style gives each role its color" {
    try std.testing.expectEqual(vaxis.Color{ .index = 202 }, TuiStyle.keyHint.color());
    try std.testing.expectEqual(TuiStyle.cursor.color(), TuiStyle.successText.color());
    try std.testing.expectEqual(TuiStyle.divider.color(), TuiStyle.style(.secondaryText).fg);
    // The logger ZIGSTYLE asks every file for; a color lookup has nothing to log.
    _ = log;
}
