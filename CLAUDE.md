# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A port of Karpathy's **nanochat** (tokenizer, pretraining, SFT/RL, eval, inference for a
GPT-2-class chat model) to Zig 0.16. **`PLAN.md` is the source of truth** for scope, phases,
and every design decision (resolved Q1–Q25); read it before starting a phase. Each phase
records its status there; the CLI's pipeline commands print which phase implements them.

It follows the sibling `../zigmicrogpt` (a Zig port of `microgpt`); read that repo's
`CLAUDE.md` for the inherited conventions.

## Git

**Never create branches, commits, tags, or pushes.** The user makes or explicitly requests
every git operation. Finish a phase at a green `make validate` and report.

## Layout

| Path | What it is |
| :--- | :--- |
| `nanochat/` | Git submodule (`karpathy/nanochat`): the Python/PyTorch reference. **Read-only upstream; never edit it.** |
| `src/` | The library. `module.zig` is the barrel and the `zig build test` root; `root.zig` exists only for `zig build docs` |
| `cli/` | The `zignanogpt` executable: `main.zig` (entry), `module.zig` (barrel + test root), `command.zig` (subcommands) |
| `web/` | The `zignanogpt-web` console (stub until phase 11) |
| `bench/main.zig` | `zig build bench`: matmul GFLOP/s at 1 thread and all threads, always ReleaseFast |
| `dev/fixtures.py` | Dev-only parity fixtures from the Python reference → `testdata/` (safetensors layout) |
| `testdata/` | Committed fixtures (created as phases add fixture groups) |

## Dependencies

All declared in `build.zig.zon`; `zigstorage` and `zigvaxis` are local checkouts at
`../../inferise/` (the build's `Git.cloneDeps` clones them if missing, then asks for a re-run).

- `zigstorage` → library only. All disk and HTTP I/O goes through its `Node` (`file://`, `https://`).
- `uucode` → library (tokenizer Unicode categories). One module is shared with zigvaxis
  (`external_uucode = true`); `uucode_fields` in `build.zig` must stay the union of what both need.
- `zigvaxis` → `cli/` only (TUI).

Changing the package `.name` invalidates `.fingerprint`; `zig build` prints the replacement value.

## Build graph

- Options: `-Dbackend=cpu|metal|cuda` (only `cpu` compiles; `src/module.zig` rejects the others),
  `-Dtest-filter=<substring>` (applied to all three test binaries).
- The `config` options module (imported as `mod.build_options`) carries `backend`, `test_filter`,
  `version`, and `source_root` (absolute repo root, for tests reading `testdata/`).
- `zig build test` runs three test binaries: `src/module.zig`, `cli/module.zig`, `web/module.zig`.
  A public type not re-exported from its directory's `module.zig` has its tests silently skipped.

## Backend layer

- `mod.Backend` is the build's backend type (`CpuBackend` today), chosen at comptime in
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
- Run `make bench` after touching `CpuMatmul`; phase 1 baseline on an M5 Max is in PLAN.md.

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
- `Storage` (src/storage.zig) is the one place files are read/written (zigstorage, atomic).

## Data

- `Dataset` lists/downloads ClimbMix shards (`shard_NNNNN.parquet`, last = val);
  `DocumentStream` is nanochat's `_document_batches` as a state machine (files -> row groups ->
  batches, epochs, resume skips to the row group after the saved one); `DataLoader` is the
  BOS-aligned best-fit packer. Its doc buffer must stay ordered (`orderedRemove`): selection
  ties go to the first match, exactly as Python's list scan.
- `ParquetFile` decodes flat byte-array columns only; nested columns (HF chat datasets) are
  phase 10. `zig` keywords bite: `resume` is reserved, `i0`/`u8`-style names are types.

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

## Console (TUI)

- `zignanogpt` with no arguments on a terminal (or `zignanogpt tui`) opens the console;
  piped, it prints usage. `zignanogpt train` on a terminal opens the console on the Train
  page and starts the run there (`--no-tui` for plain logs). Layout after `../../inferise/zigprompt/cli`:
  operations left, the selected one's form + output (or live training view) right, a status
  line of live keys. Built on zigvaxis `vxfw` (`ConsoleApp` is the root widget).
- `Operation.all` is the left pane: each entry is a `Command` plus form `Field`s that map to
  its flags (`Operation.args`). A new command gets an entry there and a case in `Runner`.
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
- `make linux`: cross-compiles x86_64 and aarch64 (DGX Spark). `make docs`: autodoc on :8080.
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
