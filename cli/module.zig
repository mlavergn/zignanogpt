// ---------------------------------------------------------------------------
// The CLI barrel: every public type in cli/ plus the libraries it uses.
// ---------------------------------------------------------------------------

const std = @import("std");

pub const nanogpt = @import("zignanogpt");
pub const vaxis = @import("zigvaxis");

pub const Args = @import("args.zig").Args;
pub const Command = @import("command.zig").Command;
pub const Console = @import("console.zig").Console;
pub const ConsoleApp = @import("console_app.zig").ConsoleApp;
pub const ConsoleStart = @import("console_app.zig").ConsoleStart;
pub const Download = @import("download.zig").Download;
pub const EvalPoint = @import("job.zig").EvalPoint;
pub const Import = @import("import.zig").Import;
pub const Job = @import("job.zig").Job;
pub const JobState = @import("job.zig").JobState;
pub const Operation = @import("operation.zig").Operation;
pub const OperationField = @import("operation.zig").Field;
pub const Overview = @import("overview.zig").Overview;
pub const OverviewLine = @import("overview.zig").OverviewLine;
pub const Runner = @import("runner.zig").Runner;
pub const TokEval = @import("tok_eval.zig").TokEval;
pub const TokTrain = @import("tok_train.zig").TokTrain;
pub const Train = @import("train.zig").Train;
pub const TrainSnapshot = @import("job.zig").TrainSnapshot;
pub const TuiStyle = @import("tui_style.zig").TuiStyle;

test {
    std.testing.refAllDecls(@This());
}
