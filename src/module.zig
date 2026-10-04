// ---------------------------------------------------------------------------
// The library barrel: every public type in src/ plus the external deps, so
// files import one module rather than their siblings.
// ---------------------------------------------------------------------------

const std = @import("std");
const builtin = @import("builtin");

// -----------------------------------------------------------------------------
// Compile-time invariants

/// Whether this is a debug build.
pub const is_debug: bool = builtin.mode == .Debug;

/// Constants supplied by `build.zig`: `backend`, `test_filter`, `version`, `source_root`.
pub const build_options = @import("config");

comptime {
    if (build_options.backend != .cpu) {
        @compileError("backend '" ++ @tagName(build_options.backend) ++ "' is not implemented yet (PLAN.md phase 13)");
    }
}

// -----------------------------------------------------------------------------
// External dependencies

pub const zigstorage = @import("zigstorage");
pub const uucode = @import("uucode");

// -----------------------------------------------------------------------------
// Library types

pub const AdamWParams = @import("adamw_params.zig").AdamWParams;
pub const AttentionOptions = @import("attention_options.zig").AttentionOptions;
pub const Config = @import("config.zig").Config;
pub const Conformance = @import("conformance.zig").Conformance;
pub const Conversation = @import("conversation.zig").Conversation;
pub const CpuAttention = @import("cpu_attention.zig").CpuAttention;
pub const CpuAttentionBackward = @import("cpu_attention_backward.zig").CpuAttentionBackward;
pub const CpuBackend = @import("cpu_backend.zig").CpuBackend;
pub const CpuMatmul = @import("cpu_matmul.zig").CpuMatmul;
pub const Dtype = @import("tensor.zig").Dtype;
pub const Gpt = @import("gpt.zig").Gpt;
pub const GptActivations = @import("gpt_activations.zig").GptActivations;
pub const GptConfig = @import("gpt_config.zig").GptConfig;
pub const GptGradBuffers = @import("gpt_grad_buffers.zig").GptGradBuffers;
pub const GptLayer = @import("gpt_weights.zig").GptLayer;
pub const GptLayerActivations = @import("gpt_activations.zig").GptLayerActivations;
pub const GptParam = @import("gpt_weights.zig").GptParam;
pub const GptWeights = @import("gpt_weights.zig").GptWeights;
pub const Message = @import("conversation.zig").Message;
pub const MessagePart = @import("conversation.zig").MessagePart;
pub const MatmulOptions = @import("matmul_options.zig").MatmulOptions;
pub const MuonAdamW = @import("muon_adam_w.zig").MuonAdamW;
pub const MuonParams = @import("muon_params.zig").MuonParams;
pub const OptimizerConfig = @import("muon_adam_w.zig").OptimizerConfig;
pub const polar_express_coeffs = @import("muon_params.zig").polar_express_coeffs;
pub const Parallel = @import("parallel.zig").Parallel;
pub const Pretokenizer = @import("pretokenizer.zig").Pretokenizer;
pub const Rendered = @import("tokenizer.zig").Rendered;
pub const Role = @import("conversation.zig").Role;
pub const Random = @import("random.zig").Random;
pub const SafeTensorEntry = @import("safe_tensors.zig").SafeTensorEntry;
pub const SafeTensors = @import("safe_tensors.zig").SafeTensors;
pub const Scalar = @import("scalar.zig").Scalar;
pub const Shape = @import("shape.zig").Shape;
pub const Storage = @import("storage.zig").Storage;
pub const Tensor = @import("tensor.zig").Tensor;
pub const TextDataset = @import("text_dataset.zig").TextDataset;
pub const Tokenizer = @import("tokenizer.zig").Tokenizer;
pub const TokenizerTrainer = @import("tokenizer_trainer.zig").TokenizerTrainer;
pub const TrainSchedule = @import("train_schedule.zig").TrainSchedule;

/// The compute backend this build targets (`-Dbackend=`). Model, optimizer and
/// loss code use only this type's ops; see `Conformance` for the contract.
pub const Backend = switch (build_options.backend) {
    .cpu => CpuBackend,
    .metal, .cuda => unreachable, // rejected by the comptime block above
};

test {
    std.testing.refAllDecls(@This());
}
