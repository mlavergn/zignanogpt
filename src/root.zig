//! zignanogpt — nanochat (tokenizer, pretraining, fine-tuning, inference) in Zig.
//!
//! This file exists so the generated documentation can find a root: Zig's autodoc
//! roots a module at `root.zig`, or failing that at the file whose basename
//! matches the artifact name — and ZIGSTYLE requires the barrel itself to be
//! `module.zig`, which can never match. Nothing imports this file; it re-exports
//! the barrel rather than duplicating it.

/// The library barrel: every public type and compile-time constant.
pub const nanogpt = @import("module.zig");
