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
    if (build_options.backend == .metal and builtin.os.tag != .macos) {
        @compileError("the metal backend needs macOS (Apple silicon)");
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
pub const Calculator = @import("calculator.zig").Calculator;
pub const ChatEval = @import("chat_eval.zig").ChatEval;
pub const chat_eval_tasks = @import("chat_eval.zig").chat_eval_tasks;
pub const ChatOptions = @import("chat_session.zig").ChatOptions;
pub const ChatSession = @import("chat_session.zig").ChatSession;
pub const Checkpoint = @import("data/checkpoint.zig").Checkpoint;
pub const CheckpointKind = @import("data/checkpoint.zig").CheckpointKind;
pub const Config = @import("config.zig").Config;
pub const Conformance = @import("conformance.zig").Conformance;
pub const Conversation = @import("conversation.zig").Conversation;
pub const CpuAttention = @import("cpu/attention.zig").CpuAttention;
pub const CpuAttentionBackward = @import("cpu/attention_backward.zig").CpuAttentionBackward;
pub const CoreEval = @import("core_eval.zig").CoreEval;
pub const CorePrepared = @import("core_eval.zig").CorePrepared;
pub const CoreTask = @import("data/eval_bundle.zig").CoreTask;
pub const CoreTaskType = @import("data/eval_bundle.zig").CoreTaskType;
pub const CpuBackend = @import("cpu/backend.zig").CpuBackend;
pub const CpuMath = @import("cpu/math.zig").CpuMath;
pub const CpuMatmul = @import("cpu/matmul.zig").CpuMatmul;
pub const DataLoader = @import("data_loader.zig").DataLoader;
pub const Dataset = @import("data/dataset.zig").Dataset;
pub const DataLoaderState = @import("data/document_stream.zig").DataLoaderState;
pub const DocumentBatch = @import("data/document_stream.zig").DocumentBatch;
pub const DocumentSource = @import("data/document_stream.zig").DocumentSource;
pub const DocumentStream = @import("data/document_stream.zig").DocumentStream;
pub const Dtype = @import("tensor.zig").Dtype;
pub const Gpt = @import("gpt.zig").Gpt;
pub const Column = @import("generation.zig").Column;
pub const Engine = @import("engine.zig").Engine;
pub const EvalBundle = @import("data/eval_bundle.zig").EvalBundle;
pub const EvalResult = @import("chat_eval.zig").EvalResult;
pub const EvalType = @import("task.zig").EvalType;
pub const GenerateOptions = @import("engine.zig").GenerateOptions;
pub const GenerativeOptions = @import("chat_eval.zig").GenerativeOptions;
pub const GeneratedBatch = @import("engine.zig").GeneratedBatch;
pub const Generation = @import("generation.zig").Generation;
pub const RlOptions = @import("rl_trainer.zig").RlOptions;
pub const RlTrainer = @import("rl_trainer.zig").RlTrainer;
pub const Rollout = @import("rl_trainer.zig").Rollout;
pub const RowState = @import("generation.zig").RowState;
pub const ToolTokens = @import("generation.zig").ToolTokens;
pub const GptActivations = @import("gpt_activations.zig").GptActivations;
pub const GptConfig = @import("gpt_config.zig").GptConfig;
pub const GptGradBuffers = @import("gpt_grad_buffers.zig").GptGradBuffers;
pub const GptLayer = @import("gpt_weights.zig").GptLayer;
pub const GptLayerActivations = @import("gpt_activations.zig").GptLayerActivations;
pub const GptParam = @import("gpt_weights.zig").GptParam;
pub const GptWeights = @import("gpt_weights.zig").GptWeights;
pub const Message = @import("conversation.zig").Message;
pub const MessagePart = @import("conversation.zig").MessagePart;
pub const HubColumn = @import("data/hub_dataset.zig").HubColumn;
pub const HubDataset = @import("data/hub_dataset.zig").HubDataset;
pub const InferenceBuffers = @import("inference_buffers.zig").InferenceBuffers;
pub const KvCache = @import("kv_cache.zig").KvCache;
pub const KvLayer = @import("kv_cache.zig").KvLayer;
pub const LoadedModel = @import("loaded_model.zig").LoadedModel;
pub const MatmulOptions = @import("matmul_options.zig").MatmulOptions;
pub const MuonAdamW = @import("muon_adam_w.zig").MuonAdamW;
pub const MuonParams = @import("muon_params.zig").MuonParams;
pub const OptimizerConfig = @import("muon_adam_w.zig").OptimizerConfig;
pub const polar_express_coeffs = @import("muon_params.zig").polar_express_coeffs;
pub const ModelSelection = @import("loaded_model.zig").ModelSelection;
pub const NumpyRandom = @import("numpy_random.zig").NumpyRandom;
pub const Parallel = @import("cpu/parallel.zig").Parallel;
pub const ParquetChunk = @import("formats/parquet_file.zig").ParquetChunk;
pub const ParquetColumn = @import("formats/parquet_file.zig").ParquetColumn;
pub const ParquetFile = @import("formats/parquet_file.zig").ParquetFile;
pub const ParquetRowGroup = @import("formats/parquet_file.zig").ParquetRowGroup;
pub const ParquetValues = @import("formats/parquet_file.zig").ParquetValues;
pub const ParquetStrings = @import("formats/parquet_file.zig").ParquetStrings;
pub const Pickle = @import("formats/pickle.zig").Pickle;
pub const PickleEntry = @import("formats/pickle.zig").PickleEntry;
pub const PickleValue = @import("formats/pickle.zig").PickleValue;
pub const Pretokenizer = @import("pretokenizer.zig").Pretokenizer;
pub const Rendered = @import("tokenizer.zig").Rendered;
pub const Role = @import("conversation.zig").Role;
pub const PythonRandom = @import("python_random.zig").PythonRandom;
pub const Random = @import("random.zig").Random;
pub const SafeTensorEntry = @import("formats/safe_tensors.zig").SafeTensorEntry;
pub const SafeTensors = @import("formats/safe_tensors.zig").SafeTensors;
pub const SafeTensorsWriter = @import("formats/safe_tensors_writer.zig").SafeTensorsWriter;
pub const Scalar = @import("scalar.zig").Scalar;
pub const SftData = @import("sft_data.zig").SftData;
pub const SftLoader = @import("sft_loader.zig").SftLoader;
pub const SftOptions = @import("sft_trainer.zig").SftOptions;
pub const SftTrainer = @import("sft_trainer.zig").SftTrainer;
pub const Shape = @import("shape.zig").Shape;
pub const Snappy = @import("formats/snappy.zig").Snappy;
pub const StateDict = @import("data/torch_import.zig").StateDict;
pub const StateTensor = @import("data/torch_import.zig").StateTensor;
pub const Storage = @import("data/storage.zig").Storage;
pub const Tensor = @import("tensor.zig").Tensor;
pub const Task = @import("task.zig").Task;
pub const TaskKind = @import("task.zig").TaskKind;
pub const TaskMixture = @import("task.zig").TaskMixture;
pub const TextDataset = @import("data/text_dataset.zig").TextDataset;
pub const ThriftField = @import("formats/thrift_reader.zig").ThriftField;
pub const ThriftList = @import("formats/thrift_reader.zig").ThriftList;
pub const ThriftReader = @import("formats/thrift_reader.zig").ThriftReader;
pub const ThriftType = @import("formats/thrift_reader.zig").ThriftType;
pub const Tokenizer = @import("tokenizer.zig").Tokenizer;
pub const TokenizerTrainer = @import("tokenizer_trainer.zig").TokenizerTrainer;
pub const TorchImport = @import("data/torch_import.zig").TorchImport;
pub const StepReport = @import("trainer.zig").StepReport;
pub const TrainData = @import("trainer.zig").TrainData;
pub const TrainObserver = @import("trainer.zig").TrainObserver;
pub const Trainer = @import("trainer.zig").Trainer;
pub const TrainOptions = @import("train_plan.zig").TrainOptions;
pub const TrainPlan = @import("train_plan.zig").TrainPlan;
pub const TrainSchedule = @import("train_schedule.zig").TrainSchedule;
pub const ZipArchive = @import("formats/zip_archive.zig").ZipArchive;
pub const ZipEntry = @import("formats/zip_archive.zig").ZipEntry;

/// The compute backend this build targets (`-Dbackend=`). Model, optimizer and
/// loss code use only this type's ops; see `Conformance` for the contract.
pub const Backend = switch (build_options.backend) {
    .cpu => CpuBackend,
    .metal => MetalBackend,
    .cuda => CudaBackend,
};

const metal = build_options.backend == .metal;
/// The Metal backend and its Objective-C bindings exist only in `-Dbackend=metal`
/// builds (they link Metal.framework); elsewhere they are `void`.
pub const MetalBackend = if (metal) @import("metal/backend.zig").MetalBackend else void;
pub const MetalBuffer = if (metal) @import("metal/backend.zig").MetalBuffer else void;
pub const Objc = if (metal) @import("metal/objc.zig").Objc else void;
pub const ObjcId = if (metal) @import("metal/objc.zig").Id else void;
pub const ObjcSel = if (metal) @import("metal/objc.zig").Sel else void;
/// The CUDA backend exists only in `-Dbackend=cuda` builds.
pub const CudaBackend = if (build_options.backend == .cuda) @import("cuda/backend.zig").CudaBackend else void;
pub const CudaBuffer = if (build_options.backend == .cuda) @import("cuda/backend.zig").CudaBuffer else void;

test {
    std.testing.refAllDecls(@This());
}
