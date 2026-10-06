# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A port of Karpathy's **nanochat** (tokenizer, pretraining, SFT/RL, eval, inference for a
GPT-2-class chat model) to Zig 0.16. **`PLAN.md` is the source of truth** for scope, phases,
and every design decision (resolved Q1–Q25); read it before starting a phase. Each phase
records its status there; the CLI's pipeline commands print which phase implements them.

It follows the sibling `../zigmicrogpt` (a Zig port of `microgpt`); read that repo's
`CLAUDE.md` for the inherited conventions.

## Platforms

Two supported targets, and both must always compile: **macOS on Apple silicon with Metal**
and **DGX OS (Linux ARM64, DGX Spark GB10) with CUDA**. A native `zig build` picks the
backend on each (`Backend.detect`). From macOS, `make cuda` cross-compiles the CUDA build and
compiles (does not run) every test binary for `aarch64-linux-gnu`; `make validate` includes it
there. Nothing on the build or test path may assume macOS (frameworks, `xcrun`, `open` stay
behind `builtin.os.tag == .macos` or in convenience targets). CUDA has not run on hardware yet.

## Git

**Never create branches, commits, tags, or pushes.** The user makes or explicitly requests
every git operation. Finish a phase at a green `make validate` and report.

## Layout

| Path | What it is |
| :--- | :--- |
| `nanochat/` | Git submodule (`karpathy/nanochat`): the Python/PyTorch reference. **Read-only upstream; never edit it.** |
| `src/` | The library: the core nanochat port (model, optimizer, tokenizer, data, training, inference, evals). `module.zig` is the barrel and the `zig build test` root; `root.zig` exists only for `zig build docs`. Supporting plug-ins live in subdirectories |
| `src/cpu/` | The CPU backend: `backend.zig`, `matmul.zig`, `attention.zig`, `attention_backward.zig`, `math.zig`, `parallel.zig` |
| `src/data/` | Data I/O: `storage.zig` (file read/write), pretraining shards (`dataset`, `document_stream`, `text_dataset`), Hugging Face downloads (`hub_dataset`), checkpoints and `.pt` import, the CORE eval bundle |
| `src/formats/` | File-format readers and writers: Parquet (+ Thrift, Snappy), safetensors, zip, pickle |
| `src/metal/` | The Metal backend: `backend.zig`, `objc.zig` (Objective-C runtime), MSL `kernels.metal` and `reduce.metal` |
| `src/cuda/` | The CUDA backend: `backend.zig` and `kernels.zig` (Zig compiled to PTX by `build.zig`) |
| `cli/` | The `zignanogpt` executable: `main.zig` (entry), `module.zig` (barrel + test root), `command.zig` (subcommands) |
| `web/` | `zignanogpt-web`: one embedded page (chat over SSE, training dashboard), `make web` |
| `bench/main.zig` | `zig build bench`: matmul GFLOP/s at 1 thread and all threads, always ReleaseFast |
| `tools/` | Build-time host tools (`nvptx_fixup.zig`: makes Zig's nvptx IR loadable as PTX) |
| `dev/fixtures.py` | Dev-only parity fixtures from the Python reference → `testdata/` (safetensors layout) |
| `testdata/` | Committed fixtures (created as phases add fixture groups) |

## Dependencies

All declared in `build.zig.zon`; `zigstorage` and `zigtui` are local checkouts at
`../../inferise/` (the build's `Git.cloneDeps` clones them if missing, then asks for a re-run).

- `zigstorage` → library only. All disk and HTTP I/O goes through its `Node` (`file://`, `https://`).
- `zigtui` → `cli/` only (`cli.tui`): the shared Inferise TUI components (`Theme`, `ProgressBar`, ...)
  and the whole vaxis API re-exported (`tui.vxfw`, `tui.Key`). Never import or declare zigvaxis.
  zigtui is shared across projects: add a component or role there only when it is generic.
- `uucode` → library (tokenizer `general_category`), taken from zigtui (`tui_dep.module("uucode")`).
  Zig allows a package's files in one module only, so the tokenizer must share zigtui's
  (zigvaxis's tables); a field zigvaxis does not build needs adding in zigtui's `uucode_fields`.

Changing the package `.name` invalidates `.fingerprint`; `zig build` prints the replacement value.

## Build graph

- Options: `-Dbackend=cpu|metal|cuda`, `-Dtest-filter=<substring>` (applied to every test binary).
  Without `-Dbackend`, `Backend.detect` in build.zig picks Metal on Apple-silicon macOS, CUDA on
  Linux with the NVIDIA driver (`libcuda.so.1`), else CPU; cross builds get CPU. Everything
  backend-specific in build.zig (choice, detection, linking, the CUDA PTX) lives in `Backend`.
  `metal` links Metal, MetalPerformanceShaders, Foundation and libobjc (macOS only); `cuda`
  builds the kernels' PTX (`cudaKernels` in build.zig) and embeds it. `module.zig` exports the
  Metal/CUDA types only in their own builds (`void` elsewhere), so CPU builds never reference them.
- The `config` options module (imported as `mod.build_options`) carries `backend`, `test_filter`,
  `version`, and `source_root` (absolute repo root, for tests reading `testdata/`).
- `zig build test` runs four test binaries: `src/module.zig`, `cli/module.zig`, `web/module.zig`, `tools/nvptx_fixup.zig`.
  A public type not re-exported from its directory's `module.zig` has its tests silently skipped.

## Backend layer

- `mod.Backend` is the build's backend type (`CpuBackend`, `MetalBackend` or `CudaBackend`), chosen at comptime in
  `src/module.zig`. Model/optimizer/loss code calls only its ops on `Tensor`s and never
  touches element memory; host data moves via `upload`/`download`.
- The contract is written down in `src/conformance.zig` (doc comment + `required_decls`):
  issue-order semantics, `sync`, no host-returning ops, contiguous tensors with an element
  offset (views via `Tensor.rows`/`reshape`), transposes as op flags.
- **Adding an op**: add it to `CpuBackend`, list it in `Conformance.required_decls`, and add
  a conformance case checked against a host (f64) reference. Shape rules shared across
  backends live in a shared type (e.g. `MatmulOptions.dims`), not per backend.
- CPU internals: `CpuMatmul` (packed GotoBLAS-style tiles, 6x16 FMA kernel) and
  `Parallel` (work items on the `std.Io` thread pool via `Io.Group.async`; worker index
  is unique among concurrent workers, so per-worker scratch needs no lock). Elementwise
  ops run in 64K-element chunks. No `std.Thread.Pool` in 0.16.
- GPU backends keep the CPU kernels as their fallback: `MetalBuffer`/`CudaBuffer` carry a
  `bytes` slice over unified memory, so `CpuBackend` ops run on GPU tensors after a `sync`
  (`onCpu`). A new op can start there and move to a kernel later; each `onCpu` stalls the GPU
  queue, so on hot paths it costs far more than the op. Metal attention is flash attention on
  simdgroup matrices (head dim padded to 8..128, compared against the CPU kernels in
  `src/metal/backend.zig`'s tests). Only `copy` writes index tensors on the GPU; the backend
  tracks those writes (`readyForHost`) so id/target checks read the host bytes without a sync.
  Optimizer and embedding-backward kernels live in the strict library (`reduce.metal`).
  Profile GPU time per kernel by timing command buffers with a sync per op (temporary patch;
  never commit it). Metal kernels are MSL in
  `src/metal/kernels.metal` (fast math) and `src/metal/reduce.metal` (strict, double-float sums);
  CUDA kernels are Zig in `src/cuda/kernels.zig` (camelCase functions `@export`ed under the
  snake_case PTX names the backend looks up; no libm on the device). Shared memory is a
  container-level `addrspace(.shared)` array; barriers, warp shuffles, `ex2`/`lg2` and
  `%nctaid` are inline PTX (checked only by the driver's JIT on the Spark). Every launch is
  256-thread blocks (the block reductions assume a multiple of 32); `launchBlocks` copies the
  arguments into a runtime tuple (constants are comptime fields with no address). A new CUDA
  kernel needs its name in both `Kernel` (backend) and the `@export` list. The CUDA backend
  has never run: see PLAN.md's DGX bring-up checklist before trusting it. `make test` runs the suite
  on the detected backend and `make test-cpu` on the CPU (`make validate` runs both); `make cuda`
  cross-compiles the CUDA build (no GPU to run it on yet).
- Run `make bench` after touching `CpuMatmul`; phase 1 baseline on an M5 Max is in PLAN.md.
  A with at most 4 rows times a transposed B (decoding) takes a separate matrix-vector path
  (column blocks, contiguous SIMD dots); the bench's `decode` cases cover it.

## Model code

- `GptWeights` is one model-shaped set of tensors (PyTorch names and registration order);
  `Gpt.weights` holds the parameters and a second `GptWeights` holds the gradients.
- `GptActivations` keeps every forward intermediate per layer (backward reads them);
  `GptGradBuffers` is one reusable set of activation gradients (backward walks layers in reverse).
- `Gpt.backward` adds into the grads (zero them per optimizer step) and takes `scale`
  (`1 / grad_accum_steps`). Backward ops that feed a residual or a shared input take an
  `accumulate` flag or add (`+=`) by contract; read the op's doc before reusing a buffer.
- `MuonAdamW` groups parameters by name exactly as `setup_optimizer`; hyperparameters stay
  f64 and are rounded to f32 per step (PyTorch's 0-D tensors). Elementwise optimizer math
  uses `lerp` with torch's two-sided formula. `step` uses the Muon grads as scratch.
  `TrainSchedule.apply(&optimizer, step)` before each `optimizer.step`.
- The optimizer fixture runs with `TORCH_COMPILE_DISABLE=1` (set in `dev/fixtures.py`).
- A new backward op needs a finite-difference case in `Conformance` (`gradCheck` against the
  matching forward op). Keep reductions deterministic: per-chunk partials summed in order,
  never per-worker accumulators.

## Tokenizer

- `Pretokenizer` hand-implements the split pattern's alternatives in order (it is not a
  regex engine); classes come from uucode categories plus the hardcoded `White_Space` list.
  Any change must keep `testdata/tokenizer.json` passing (pieces, ids, merges).
- `Tokenizer` = tiktoken semantics (rank by merged bytes, leftmost lowest rank);
  `TokenizerTrainer` = rustbpe (count desc, then smaller pair). Specials follow the ranks.
- CLI options go through `cli/args.zig` (`Args`): every flag must be consumed, `finish`
  rejects typos. User-facing problems log at `warn` (tests fail on `log.err`).
- `Storage` (src/data/storage.zig) is the one place files are read/written (zigstorage, atomic).

## Data

- `Dataset` lists/downloads ClimbMix shards (`shard_NNNNN.parquet`, last = val);
  `DocumentStream` is nanochat's `_document_batches` as a state machine (files -> row groups ->
  batches, epochs, resume skips to the row group after the saved one); `DataLoader` is the
  BOS-aligned best-fit packer. Its doc buffer must stay ordered (`orderedRemove`): selection
  ties go to the first match, exactly as Python's list scan.
- `ParquetFile` decodes byte-array, int32 and int64 columns, nested ones included (repetition and
  definition levels; `ParquetValues.rowStarts` splits list columns into rows). `zig` keywords
  bite: `resume` is reserved, `i0`/`u8`-style names are types.

## Checkpoints

- Layout follows nanochat: `<base>/{base,chatsft,chatrl}_checkpoints/<tag>/{model,optim,meta}_<step:06>`;
  this port writes `.safetensors` (f32, PyTorch names), Python's are `.pt`. `Checkpoint` finds
  the largest `d<N>` tag and last step. `zignanogpt import` converts Python checkpoints
  (`TorchImport`); it overwrites `<base>/tokenizer` with the checkpoint's tokenizer.
- zigstorage treats a scheme-less relative path as a URL host: build paths from `Config.load`
  (absolute) and pass user paths through `Storage.absolute`. `Storage` methods do this already;
  code that opens `zigstorage.Node` directly (Parquet, zip, safetensors, listings) does not.

## Training

- `TrainPlan` reproduces `base_train.py`'s derivations (model shape from depth, d12-relative
  scaling laws, batch-size LR and weight-decay scaling, iterations, grad accumulation); its
  tests pin Python's numbers for the CPU preset and default d20. `TrainOptions` fields map 1:1
  to `--kebab-case` flags (`cli/train.zig` reflects over them); 0 means Python's -1.
- `Trainer.run` mirrors base_train's loop (eval/sample/save at the top, `num_iterations + 1`
  passes). Resume is approximate by design (data restarts at the next row group). Checkpoints
  and `metrics.jsonl` live in `<base>/base_checkpoints/<tag>/`.

## Inference

- `Gpt.forwardStep(cache, bufs, idx)` is the cached forward: prefill `[1, T]` or decode
  `[B, 1]`; `KvCache` holds `[B, Tmax, Hkv, D]` per layer, the shared position and the last
  token's pre-smear embedding (`prev`); `InferenceBuffers` is per-step scratch (one set reused
  by every layer). Logits come out for each row's last position only.
- `Engine.generate` returns a `Generation` iterator (`next` -> token column + masks): one batch-1
  prefill copied into `num_samples` rows (`KvCache.copyFrom`), `RowState` per row (forced
  tokens, calculator block). The forward for a column runs lazily at the next `next` call, so
  nothing is computed after the last token. Only greedy is comparable with Python (different
  RNG); `testdata/engine.json` (fixture group `engine`) pins greedy tokens and calculator results.
- `Calculator` replaces Python `eval` for exactly what `use_calculator`'s filter admits; keep
  its output byte-identical to Python's `str()` (the fixture lists edge cases).
- `LoadedModel` loads `<base>/<kind>_checkpoints/<tag>/model_<step>.safetensors` + the
  tokenizer; it must not move after `init` (config strings live in its arena). `ChatSession`
  is `chat_cli.py`'s conversation; `cli/chat.zig` (CLI) and `cli/chat_job.zig` (console) both
  drive it.

## Tasks, SFT and evals

- `HubDataset` loads Hugging Face parquet exports by logical column path (`messages.role`,
  `choices.text`): writers name list elements `item` or `element`. Rows are ordered by
  `NumpyRandom` (numpy's PCG64 `permutation`); mixtures and few-shot picks by `PythonRandom`
  (CPython's MT19937). Changes must keep `testdata/tasks.json` and `core.json` passing.
- Fixtures under `testdata/task_base/task_data` are real slices in `load_hub_dataset`'s
  layout; tests point `Config.base_dir` (or `nanochat_dir`, the read-only fallback) at them.
- `SftLoader` caps renders at `min(2048, seq + 1)` (PLAN.md phase 10): the `sft` fixture runs
  nanochat's chat_sft.py through `runpy` with that one patch and `Task.__len__` clamped.
- `CoreEval` renders nanochat's three Jinja templates by hand (few-shot blocks joined by
  a blank line, `trim` = Python `str.strip`); `../zigjinja` exists if general templates are needed.

## Console (TUI)

- `zignanogpt` with no arguments on a terminal (or `zignanogpt tui`) opens the console;
  piped, it prints usage. `zignanogpt train` on a terminal opens the console on the Train
  page and starts the run there (`--no-tui` for plain logs). Layout after `../../inferise/zigprompt/cli`:
  operations left, the selected one's form + output (or live training view) right, a status
  line of live keys. Built on zigtui components (`cli.tui`): `SplitPane` (titles, rules, the
  draggable divider), `StatusLine`, `StatusMark` (job/chat state), `Form` + `LineInput` (the
  operation forms, the chat input), `ProgressBar`, `Sparkline` (loss curve) and `TranscriptView` (chat,
  job log); colors by `tui.Theme` role. `ConsoleApp` is the root widget; its panes are draw-only
  `PaneView`s. Generic UI belongs in zigtui (shared), not in `cli/`.
- `Operation.all` is the left pane, in pipeline order: each entry is a `Command` plus form
  `Field`s that map to its flags (`Operation.args`). An entry can run a second command
  (`alternate`, chosen by a field's value) with per-command fields (`Field.only`): Evaluate runs
  `eval` for base models and `chat-eval` for sft/rl. A new pipeline command gets an entry there
  and a case in `Runner`; diagnostics (`tok-eval`, `import`) stay CLI-only.
- Chat is the exception: the console's Chat page uses `ChatJob` (its own worker, one thread per
  load or reply, transcript behind a mutex) instead of `Job`, so it can run beside training.
  `zignanogpt chat` on a terminal opens that page (short flags are spelled out first, since
  the form only knows long names); `-p` or `--no-tui` stay line-based.
- `Job` runs one command on a worker thread through `Runner.execute` (the same dispatch the
  CLI uses), writing to a mutex-guarded log via a `std.Io.Writer`; training feeds it through
  `TrainObserver`. While the console holds the terminal, `std.log` goes into the job's log
  (`Console.captureLog`, wired in `main.zig`'s `logFn`). Quitting during training stops and
  saves; other running jobs are abandoned to the process exit (never freed under the thread).
- Tests drive `ConsoleApp` headlessly (`handleEvent` with key presses, `draw` into a vxfw
  `DrawContext`); flattening a drawn surface's cell buffers to text is a quick layout check.
  Live terminal behavior (raw mode, resize, colors) still needs a real terminal.

## Commands

- `make validate`: clean, format, lint, build, test (the gate). `make test`: tests only.
- Single test: `zig build test -Dtest-filter="<test name substring>"`.
- `make lint` reads `styleguide/zlint.json` (the `styleguide` submodule; `git submodule update
  --init` after a fresh clone). zlint warnings are errors, and it
  rejects `catch {}` and `catch unreachable` (flush explicitly; write `(a + b - 1) / b`, not
  `divCeil(...) catch unreachable`). It also fails on an unused `log`: a file with only hot
  functions logs on a cold path (an error branch) instead.
- `make format` / `make lint` cover `ZIG_SOURCES` (`build.zig`, `src/`, `cli/`, `web/`, `bench/`); add new
  source directories there, never `.` (that sweeps in `nanochat/`).
- `make linux`: cross-compiles x86_64 and aarch64 (CPU backend). `make cuda`: the DGX OS CUDA
  build plus compiled tests. `make docs`: autodoc on :8080.
- `make cli`: ReleaseFast build, then the console (TUI). `make run ARGS="config"` or
  `./zig-out/bin/zignanogpt <command>` for a single command.
- `make fixtures` (or `make fixtures ONLY=gpt`): runs `dev/fixtures.py` with
  `/usr/local/inferise/uv/bin/uv` (`UV=` to override), `--frozen`, venv at `./.venv` (kept out
  of the submodule). Regenerate only when a fixture group changes; the files are committed.

## Parity testing

- Each phase adds a `@fixture` group to `dev/fixtures.py` that dumps PyTorch inputs, weights
  and outputs (plus `debug.*` intermediates) to `testdata/<group>.safetensors`; Zig tests load
  it with `SafeTensors.load(.., mod.build_options.source_root ++ "/testdata/...")`.
- Fixture weights are perturbed after `init_weights` (which zeroes projections) so every path
  contributes. Python facts already pinned: CPU computes in f32; `rms_norm` eps = f32 machine
  eps; attention sees `0 <= i - j <= window`, scale `1/sqrt(D)`, kv head `h / (H / Hkv)`; the
  short window for a 2048 context is 512 (the code comment saying 768 is stale).
- `Gpt` parameters carry their PyTorch `state_dict` names (`Gpt.params`), so fixtures and
  Python checkpoints load by name. Tests fail on any `log.err`, so expected-error paths log at
  debug/warn.

## Runtime directories

`Config` (`src/config.zig`): Zig writes only to `$ZIGNANOGPT_BASE_DIR` (default
`~/.cache/zignanogpt`); `$NANOCHAT_BASE_DIR` (default `~/.cache/nanochat`) is read-only
(Python's shards, checkpoints, tokenizer). Never write into the nanochat dir.

## Style

`styleguide/ZIGSTYLE.md` (see `../zigmicrogpt/styleguide/` until the submodule exists):
one primary type per file named for its role (no prefix), import siblings only through
`module.zig`, scoped logger `.zignanogpt_<file>`, `log.debug` entry trace on non-hot
functions, doc comments with Parameters/Return, tests at the bottom under `// Unit Tests`.
Executables set `std_options.log_level = .info`, which compiles the traces out.
