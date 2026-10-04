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
pub const Checkpoint = @import("checkpoint.zig").Checkpoint;
pub const CheckpointKind = @import("checkpoint.zig").CheckpointKind;
pub const Config = @import("config.zig").Config;
pub const Conformance = @import("conformance.zig").Conformance;
pub const Conversation = @import("conversation.zig").Conversation;
pub const CpuAttention = @import("cpu_attention.zig").CpuAttention;
pub const CpuAttentionBackward = @import("cpu_attention_backward.zig").CpuAttentionBackward;
pub const CpuBackend = @import("cpu_backend.zig").CpuBackend;
pub const CpuMatmul = @import("cpu_matmul.zig").CpuMatmul;
pub const DataLoader = @import("data_loader.zig").DataLoader;
pub const Dataset = @import("dataset.zig").Dataset;
pub const DataLoaderState = @import("document_stream.zig").DataLoaderState;
pub const DocumentBatch = @import("document_stream.zig").DocumentBatch;
pub const DocumentSource = @import("document_stream.zig").DocumentSource;
pub const DocumentStream = @import("document_stream.zig").DocumentStream;
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
pub const ParquetChunk = @import("parquet_file.zig").ParquetChunk;
pub const ParquetColumn = @import("parquet_file.zig").ParquetColumn;
pub const ParquetFile = @import("parquet_file.zig").ParquetFile;
pub const ParquetRowGroup = @import("parquet_file.zig").ParquetRowGroup;
pub const ParquetStrings = @import("parquet_file.zig").ParquetStrings;
pub const Pickle = @import("pickle.zig").Pickle;
pub const PickleEntry = @import("pickle.zig").PickleEntry;
pub const PickleValue = @import("pickle.zig").PickleValue;
pub const Pretokenizer = @import("pretokenizer.zig").Pretokenizer;
pub const Rendered = @import("tokenizer.zig").Rendered;
pub const Role = @import("conversation.zig").Role;
pub const Random = @import("random.zig").Random;
pub const SafeTensorEntry = @import("safe_tensors.zig").SafeTensorEntry;
pub const SafeTensors = @import("safe_tensors.zig").SafeTensors;
pub const SafeTensorsWriter = @import("safe_tensors_writer.zig").SafeTensorsWriter;
pub const Scalar = @import("scalar.zig").Scalar;
pub const Shape = @import("shape.zig").Shape;
pub const Snappy = @import("snappy.zig").Snappy;
pub const StateDict = @import("torch_import.zig").StateDict;
pub const StateTensor = @import("torch_import.zig").StateTensor;
pub const Storage = @import("storage.zig").Storage;
pub const Tensor = @import("tensor.zig").Tensor;
pub const TextDataset = @import("text_dataset.zig").TextDataset;
pub const ThriftField = @import("thrift_reader.zig").ThriftField;
pub const ThriftList = @import("thrift_reader.zig").ThriftList;
pub const ThriftReader = @import("thrift_reader.zig").ThriftReader;
pub const ThriftType = @import("thrift_reader.zig").ThriftType;
pub const Tokenizer = @import("tokenizer.zig").Tokenizer;
pub const TokenizerTrainer = @import("tokenizer_trainer.zig").TokenizerTrainer;
pub const TorchImport = @import("torch_import.zig").TorchImport;
pub const StepReport = @import("trainer.zig").StepReport;
pub const TrainData = @import("trainer.zig").TrainData;
pub const TrainObserver = @import("trainer.zig").TrainObserver;
pub const Trainer = @import("trainer.zig").Trainer;
pub const TrainOptions = @import("train_plan.zig").TrainOptions;
pub const TrainPlan = @import("train_plan.zig").TrainPlan;
pub const TrainSchedule = @import("train_schedule.zig").TrainSchedule;
pub const ZipArchive = @import("zip_archive.zig").ZipArchive;
pub const ZipEntry = @import("zip_archive.zig").ZipEntry;

/// The compute backend this build targets (`-Dbackend=`). Model, optimizer and
/// loss code use only this type's ops; see `Conformance` for the contract.
pub const Backend = switch (build_options.backend) {
    .cpu => CpuBackend,
    .metal, .cuda => unreachable, // rejected by the comptime block above
};

test {
    std.testing.refAllDecls(@This());
}
