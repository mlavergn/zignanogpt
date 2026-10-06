# PLAN.md — nanochat → pure Zig

Port `nanochat/` (Python/PyTorch, read-only submodule) to Zig 0.16.
Conventions: `../zigmicrogpt` + `styleguide/ZIGSTYLE.md` (one struct per file, `module.zig`
barrels, role-named files with no prefix, scoped logs, tests at file bottom).

Dependencies (all Zig):

| Dep | Used by | For |
| :--- | :--- | :--- |
| `zigstorage` | library `src/` | Disk + HTTP I/O |
| `uucode` | library `src/`, the module zigtui builds for zigvaxis (one per process) | Tokenizer `\p{L}`/`\p{N}` (`general_category`) |
| `zigtui` | `cli/` only | Shared TUI components over zigvaxis (re-exports its API; pulls in `zigimg`) |

No C/Python at runtime; Python only in dev-only fixture generation.

Supported platforms (both must compile at every commit; `make validate` on macOS cross-compiles
the second, tests included):

| Platform | Backend | Build |
| :--- | :--- | :--- |
| macOS, Apple silicon (ARM64) | Metal | native `zig build` (detected) |
| DGX OS (Ubuntu-based Linux, ARM64), DGX Spark GB10 (sm_121) | CUDA | native `zig build` on the Spark (detected via `libcuda.so.1`); `make cuda` cross-compiles from macOS |

CPU works on both (and elsewhere) with `-Dbackend=cpu`.

## Key decisions

| Area | Python | Zig |
| :--- | :--- | :--- |
| Compute | PyTorch, CUDA/MPS, bf16, FA3 | `Backend` chosen at build time (`-Dbackend=cpu\|metal\|cuda`); CPU first |
| Precision | bf16 compute, fp32 master | f32 everywhere; bf16 checkpoint tensors upcast on import |
| Autograd | torch autograd | Hand-written backward per layer (llm.c style), no generic tape |
| Attention | FA3 / SDPA | Causal + sliding-window attention, GQA, as backend ops |
| Tokenizer | rustbpe + tiktoken (Rust `fancy-regex`) | Hand-written pre-tokenizer + `uucode` categories + BPE |
| Data | pyarrow parquet | Minimal Parquet reader (Thrift compact; zstd via `std.compress.zstd`; snappy hand-rolled) |
| Disk I/O | `open`, `torch.save` | zigstorage `file://` (atomic sessions, ranged reads for Parquet footers) |
| Download | requests / urllib | zigstorage `https://` streamed into a `file://` session |
| Checkpoints | `torch.save` `.pt` + JSON | Own format (JSON header + raw f32) + meta JSON; Zig importer for Python `.pt` |
| CLI | one script per stage | One `zignanogpt` executable with subcommands; zigvaxis TUI for `chat` + `train` monitor |
| Monitoring | wandb | Trainer appends `metrics.jsonl` + samples to the run dir; TUI and web tail it |
| Distributed | DDP / torchrun | Out of scope (one process, many threads) |
| Parity | — | Tolerance tests against fixtures from a dev-only Python script |

## Backend design

- Selected at comptime: `-Dbackend=` picks one backend type; model code is generic over it.
- Shaped for Metal (unified memory, MSL kernels, command buffers) and CUDA (discrete memory,
  driver API, streams) from day one:
  - `Tensor` = shape + opaque device storage handle. No slice access outside a backend.
  - Explicit `toHost`/`fromHost` for I/O, checkpoints, sampling, tests.
  - Ops are enqueued; explicit `sync` points (loss readout, sampling, checkpoint, eval).
  - Device allocator per backend; activations/grad buffers preallocated per config.
- Op set (each with backward where needed): matmul, embedding gather/scatter-add, rmsnorm,
  rope, attention (causal, window, GQA, KV cache), relu², sigmoid/tanh/softcap, cross-entropy,
  elementwise/reduce, AdamW step, Muon pieces (Polar Express, norms, variance reduction).
- One conformance test suite runs every op against every backend (CPU is the reference).

## CLI

`zignanogpt <cmd>`: `download`, `tok-train`, `tok-eval`, `train`, `eval`, `chat`, `import`
(+ M2: `sft`, `chat-eval`, `rl`). Flags mirror the Python scripts.

## Milestones

**M1 = phases 0–9** (pretrain + chat inference). **M2 = phases 10–12** (SFT/evals, web, CPU perf).
**M3 = phase 13** (Metal + CUDA backends, after M2). Each phase ends green on `make validate`.
Git: Claude creates no branches or commits; the user makes or requests every git operation.

0. **Scaffold** — done
   - `src/module.zig`, `src/root.zig`, `cli/{main,module}.zig`, `web/{main,module}.zig` stubs.
   - `build.zig.zon`: dep paths → `../../inferise/<dep>`; add `uucode` (same commit zigvaxis pins).
   - `build.zig`: `zigstorage` + `uucode` → library; `zigvaxis` (`external_uucode = true`, fed our
     `uucode` with the union of fields) → `cli/` only.
   - Add `styleguide` submodule; wire `-Dtest-filter` into the 3 `addTest` calls; `-Dbackend` option.
   - `Config`: base dir `~/.cache/zignanogpt` (`$ZIGNANOGPT_BASE_DIR`) for everything Zig writes;
     nanochat dir `~/.cache/nanochat` (`$NANOCHAT_BASE_DIR`) is read-only (shards, Python checkpoints).
   - `make linux` also cross-compiles `aarch64-linux` (DGX Spark host).
   - `dev/fixtures.py` (run in the nanochat venv) dumps tensors/tokens to committed `testdata/`.
   - Update `CLAUDE.md` (deps, layout).
1. **Tensor + backend core** — done
   - `Tensor`, `Backend` contract, `CpuBackend` (tiled SIMD matmul, thread pool), `Random`.
   - Op conformance tests; `zig build bench` (matmul GFLOP/s).
   - Exit gate: review the `Backend` contract against Metal and CUDA execution models.
   - Result: ops so far are memory, `fill`/`copy`/`add`/`mul`/`scale`, `matmul` (both
     transposes, alpha, accumulate); later phases add theirs with conformance cases.
     Baseline (M5 Max, 18 threads): matmul 80–86 GFLOP/s per core, 0.6–1.1 TFLOP/s total.
   - Contract review (rules now in `src/conformance.zig`): issue-order semantics, errors may
     surface at `sync`, no op returns host data, `free` safe after issue, element offsets
     map to `setBuffer:offset:` / `CUdeviceptr + bytes`. No changes needed for either GPU.
2. **Model forward** (`gpt.py`) — done
   - `GptConfig`, `Gpt`: padded vocab, RoPE (base 100k, −θ convention), QK-norm ×1.2, ReLU²,
     GQA, window pattern `SSSL`, value embeds + `ve_gate`, smear, backout, resid/x0 lambdas,
     logit softcap 15, `init_weights`, flops/param counts.
   - Parity: logits vs fixture for a tiny config (atol ~1e-4).
   - Result: `testdata/gpt.safetensors` (4 layers, GQA 4/2, windows 128/512, 2 value-embed
     layers, vocab 300→320, every weight perturbed). Max |diff| vs PyTorch: blocks ≤ 7.6e-6,
     logits 2.6e-6, rotary 1 ulp; tests gate at 5e-5. Param/FLOP counts match exactly.
   - Ops added (each with a conformance case): embedding, rmsnorm, rope, attention (window,
     GQA, KV-cache key count, lse out), reluSquare, combine, gateLinear, smear, valueMix, softcap.
3. **Backward + loss** — done
   - Per-layer backward, cross-entropy with `ignore_index=-1`, `mean`/`none` reduction.
   - Tests: finite-difference grad check; grads vs fixture.
   - Result: `testdata/gpt_grad.safetensors` (targets with ignored rows). Loss within 5e-7;
     all 35 parameter grads within 3.6e-6 of the tensor's peak (gate 2e-5). Second backward
     accumulates exactly. `none` reduction not needed by training; deferred to evals (phase 10).
   - Ops added, each finite-difference checked in `Conformance`: crossEntropy(+Backward, fused
     with softcap), rmsnorm/rope/attention/reluSquare/gateLinear/smear/valueMix/embedding
     backward, dot. Reductions sum fixed chunks in a fixed order: results are bit-reproducible.
4. **Optimizer** (`optim.py`) — done
   - AdamW (decoupled WD, bias correction); Muon (Nesterov, MuonEq, Polar Express ×5,
     Muon+ renorm, NorMuon variance reduction, cautious WD); param groups + LR scaling.
   - LR schedule: warmup / warmdown / final-lr-frac; Muon momentum schedule.
   - Parity: one step on fixture params/grads.
   - Result: `testdata/optim.safetensors` (3 steps, schedule values varied per step, wd > 0).
     After 3 steps every parameter is within 2.6e-7 of PyTorch (relative to its peak; gate
     1e-5). `TrainSchedule` matches `base_train.py`'s functions (incl. Python's round-half-even).
     A 2-layer model memorizes one random batch in 60 steps (loss 3.69 -> 0.037).
   - Ops added: adamwStep, muonMomentum, muonPrepare, muonFinish (Polar Express = matmul +
     combine). Muon runs per matrix (nanochat stacks same-shape matrices only for speed).
5. **Tokenizer** (`tokenizer.py`, rustbpe) — done
   - `\p{L}`/`\p{N}` from `uucode` `general_category`; `\s` = Unicode `White_Space` (25 code
     points, hardcoded: uucode ships no `PropList.txt`).
   - Pre-tokenizer matching `SPLIT_PATTERN` exactly (possessive quantifiers, `\s+(?!\S)`).
   - BPE train (rustbpe algorithm), encode (rank merges), decode, 9 special tokens,
     `render_conversation` / `render_for_completion`, `token_bytes`.
   - CLI: `tok-train`, `tok-eval`.
   - Parity: exact token ids vs tiktoken on a fixture corpus (incl. Unicode edge cases).
   - Result: `testdata/tokenizer.json`. Pieces match Python `regex` and ids match tiktoken
     exactly on 58 cases (CJK, Hangul, emoji ZWJ, combining marks, `²½Ⅻ٣`, NBSP/U+3000/U+2028,
     CRLF, `ſ` contractions); all 768 merges match rustbpe; renders + masks + completion match.
     At 4,096 vocab on all nanochat .md/.py (447 KB) all 4,087 ranks match rustbpe. Encoding
     is ~2x tiktoken single-threaded (4.5 MB: 0.13 s vs 0.24 s), same 1,230,870 tokens.
   - rustbpe analysis: its lazy heap is deterministic standard BPE (a pair is only ever created
     in one merge step, so one job covers all its occurrences); ties go to the smaller pair.
     Encoding follows tiktoken's rank-based `_byte_pair_merge`, not rustbpe's merge-id loop.
   - `tok-train` / `tok-eval` read `--data <file>` (`TextDataset`: blank-line-separated
     documents) until phase 6 adds ClimbMix; `tok-eval --compare <x.tiktoken>` for GPT-4.
     Output: `<base>/tokenizer/tokenizer.tiktoken` (tiktoken's format; specials implied).
     `token_bytes` is computed from the tokenizer, not saved.
6. **Data** — done
   - `Dataset`: ClimbMix shard list; use a shard already in the nanochat dir, else download into
     the Zig base dir via zigstorage (retry, atomic commit); last shard = val.
   - `TextDataset`: tiny local corpus (`testdata/corpus.txt`, or `--data <file>`) for tests and smoke runs.
   - `Parquet` reader over zigstorage ranged reads; `DataLoader` (BOS-aligned best-fit packing, resume state).
   - CLI: `download -n N`.
   - Parity: same `(inputs, targets)` rows as Python for a fixed shard + tokenizer.
   - Result: `ParquetFile` (Thrift compact footer, v1/v2 data pages, PLAIN + dictionary,
     def levels, UNCOMPRESSED/SNAPPY/ZSTD) matches pyarrow on 4 fixture variants. `DataLoader`
     reproduces nanochat's best-fit batches exactly: 300 train batches over 2 shards and 3
     epochs, a mid-epoch resume, and val. Real ClimbMix: `download -n 1` (HTTPS via redirect,
     PAR1-checked, atomic); on shard 0 (86,016 docs, 226M chars) `tok-train` at 4,096 vocab
     takes 4.0 s single-threaded (rustbpe 4.1 s multi-threaded) and all 4,087 ranks match;
     `tok-eval` bytes/tokens match tiktoken on the train and val shards.
   - `Config.data_url` (`$ZIGNANOGPT_DATA_URL`) replaces the hardcoded BASE_URL. Shards are
     listed from both base dirs (nanochat's read-only; this port's copy wins a clash).
   - Deferred: parallel tokenization in the loader and parallel downloads (phase 12).
7. **Checkpoints + Python import** — done
   - Own format via zigstorage sessions: model + optimizer + meta (`model_NNNNNN`,
     `meta_NNNNNN.json`), find last step / largest model.
   - `.pt` importer: `std.zip` + pickle subset (`_rebuild_tensor_v2`, `OrderedDict`,
     Float/BFloat16 storages); strip `_orig_mod.`; patch missing `window_pattern`,
     `resid_lambdas`, `x0_lambdas` as `checkpoint_manager.py` does.
   - `tokenizer.pkl` importer (tiktoken `Encoding` state) + `token_bytes.pt`.
   - CLI: `import` (reads the nanochat dir, writes the Zig base dir). Test: imported checkpoint
     logits match PyTorch's.
   - Result: `ZipArchive` (zip64, stored members over ranged reads), `Pickle` (protocols 2-4,
     calls recorded not executed), `TorchImport` (`_rebuild_tensor_v2` with strides/offsets;
     f32/f64/f16/bf16 -> f32; `_orig_mod.` stripped; missing lambdas patched; tokenizer.pkl ->
     `Tokenizer`, digit bound read from its pattern). Fixture `testdata/nanochat_base` (d2:
     bf16 embeddings; d1: legacy keys, no window_pattern): imported models reproduce
     `build_model`'s logits within 5e-5; default picks the largest tag and last step.
   - Own format: `model_*.safetensors`, `optim_*.safetensors` (adamw.{m,v,step}, muon.{momentum,
     second}), nanochat's `meta_*.json` (model_config rewritten complete). `SafeTensorsWriter`
     streams tensors one at a time. Save + reload of model and optimizer gives a bit-identical
     next step.
   - `Storage.absolute` / `Config.load`: zigstorage needs absolute paths; every user path and
     env override is resolved against the working directory. CLI errors print one line.
8. **Base training** (`base_train.py`) — done (TUI needs a check in a real terminal)
   - CLI `train`: flags as Python (`--depth`, `--aspect-ratio`, `--head-dim`, `--max-seq-len`,
     `--window-pattern`, batch sizes, LRs, iterations, eval/sample/save cadence, resume).
   - Grad accumulation, val bpb (`loss_eval.py`), periodic samples, MFU/tok-s.
   - `metrics.jsonl` (step, loss, bpb, lr, tok/s, MFU, samples) in the run dir.
   - zigvaxis monitor (loss curve, tok/s, samples); plain-text log when stdout is not a tty.
   - Defaults = Python's (`--depth=20`, seq 2048, `SSSL`); `--preset=cpu` applies `runcpu.sh`
     (d6, seq 512, `L`, device batch 32, total batch 16384).
   - Target: `runs/runcpu.sh` equivalent (d6, seq 512) trains and loss tracks Python ±5%.
   - Result: same tokenizer (Python's 32,768 vocab on shard 0, `import --tokenizer-only`) and
     data, 150 steps of `--preset cpu`, different random init:

     | | Python (MPS) | Zig (CPU) |
     | :--- | ---: | ---: |
     | val bpb @0 / 50 / 100 / 150 | 3.1958 / 2.0891 / 1.9474 / 1.9016 | 3.1958 / 2.0894 / 1.9472 / 1.9011 |
     | loss @49 / 99 / 149 | 7.1411 / 6.3960 / 6.2286 | 7.1510 / 6.3963 / 6.2272 |
     | time per step | 0.26 s | 4.4-5.8 s (0.4-0.58 TFLOP/s), 11 GB RSS |

     Within 0.15% everywhere (target was 5%). A full 5,000-step preset run would take ~6 h on CPU.
   - `TrainPlan` pins base_train's derived numbers for the CPU preset and d20 exactly.
     `Trainer`: grad accumulation, schedules, val bpb (`crossEntropyRows`), greedy samples
     (KV-cached since phase 9), checkpoints + resume (approximate data
     resume, as nanochat), `metrics.jsonl`, stop-and-save on request.
   - Console (zigvaxis vxfw, after zigprompt's CLI): `zignanogpt` with no arguments opens it;
     operations on the left (Overview, Download, Train/Evaluate tokenizer, Import, Train; later
     phases listed with their phase), form + output on the right, jobs on a worker thread,
     training shown live (progress/eta, stats, loss curve, val bpb, samples), `s` stops and
     saves. `train` on a terminal runs inside it. Layout and keys tested headlessly.
9. **Inference** — done (console chat page needs a check in a real terminal)
   - `KvCache`, `Engine` (prefill once, batched samples, temperature/top-k, smear state,
     calculator tool via a safe arithmetic evaluator — no Python `eval`).
   - CLI `chat`: zigvaxis TUI (interactive) + `-p` one-shot prompt.
   - Result: `Gpt.forwardStep` (prefill at B=1 or decode at T=1 against `KvCache`; smear from
     the cached previous embedding via the new `gatedAdd` op; logits for the last position
     only) reproduces the full forward pass within 5e-5 across the sliding window, and from a
     prefill copied into several rows. `Engine`/`Generation` = `engine.py` (`RowState`, forced
     tokens, stop on `<|assistant_end|>`/`<|bos|>`, `generate_batch`); greedy generations on
     the imported d2 model match Python's `Engine` token for token (`testdata/engine.json`).
     Sampling (temperature, top-k) uses this port's generator, so only greedy is comparable.
   - `Calculator` = `use_calculator`: Python arithmetic (`+ - * / //`, unary signs, int vs
     float, `str()` formatting incl. `1e+16`/`1.5e-05`/`-0.0`, CPython's float `//`) and
     `'s'.count('t')`; 53 edge cases match Python. Ints beyond 128 bits and other string
     methods return None (Python would evaluate them).
   - `LoadedModel` (= `load_model`: kind, largest tag, last step), `ChatSession` (=
     `chat_cli.py`'s token layout and loop; seed 42 per reply like Python). `chat` defaults to
     sft and falls back to base (with a note) until phase 10 produces sft checkpoints; `-p`
     one-shot, line-based chat when piped or `--no-tui`, the console's Chat page on a terminal
     (settings form, streamed transcript, input line; `clear`/`quit`, esc stops a reply; runs
     beside a training job). `Gpt.greedy` (training samples) now uses the cache too.
   - CPU decode: `CpuMatmul` gained a matrix-vector path (A <= 4 rows, `x @ W^T`): d20-sized
     decode matmuls went from ~28 to ~112-138 GFLOP/s (~220-280 GB/s, memory bound).
     A 150-step d6 base model loads and answers a 40-token prompt in 0.27 s.
10. **Fine-tuning + evals** — done (console pages need a check in a real terminal)
    - SFT (`chat_sft.py`): task mixture, masked loss. Tasks: SmolTalk, ARC, MMLU, GSM8K (HF parquet,
      nested `messages` columns, snappy).
    - Evals: base bpb, CORE (`eval_bundle.zip` via `std.zip`), chat_eval (ARC/MMLU/GSM8K).
    - RL (`chat_rl.py`) on GSM8K — last, optional.
    - Result: data. `ParquetFile` reads nested columns (repetition/definition levels; logical
      paths like `choices.text` whatever the writer calls list elements) and int32/int64.
      `HubDataset` = `load_hub_dataset` (hub API listing, shards + `manifest.json` under
      `<base>/task_data`, nanochat's copy reused read-only); `NumpyRandom` (PCG64 +
      SeedSequence) reproduces `default_rng(42).permutation`, `PythonRandom` (MT19937)
      `random.Random(42).shuffle` and `.sample`. `Task` (SmolTalk, MMLU, ARC-Easy/Challenge,
      GSM8K with calculator parts) and `TaskMixture` match Python on renders, masks, letters,
      evaluations and mixture order (`testdata/task_base`: real slices, v1 and v2 pages).
    - chat_eval: `ChatEval` (categorical: letter logits after a cached prefill; generative:
      `Engine` samples) matches `run_categorical_eval`/`run_generative_eval` exactly on the d2
      model (accuracies and greedy completions). CLI `chat-eval` (-i -a -x -t -k -n -m), console
      page. ChatCORE is computed over ARC-E, ARC-C, MMLU, GSM8K (HumanEval out of scope).
    - SFT: `SftLoader` = the best-fit-pad generator (batches, progress, epochs, stop identical);
      `SftTrainer` = chat_sft.py (inherited batch/LRs, init_lr_frac, optimizer warm start,
      progress-based LR, Muon momentum, val bpb, ChatCORE, `chatsft_checkpoints/<tag>`).
      nanochat's own chat_sft.py and ours on the same base and data: losses, val bpb and the
      fine-tuned model's logits agree within 2e-4. CLI `sft`, console page (live view, `s`).
      Real CPU-preset run (d6, seq 512, full 790K-conversation mixture): 12.5 GB RSS, ~5.5 s/step.
    - Deliberate deviation: SFT renders conversations with at most `min(2048, max_seq_len + 1)`
      tokens. Python always uses 2048; below that its loader stalls (at seq 512, 43% of the
      mixture never fits, the buffer fills with them after ~10 batches and 95-99% of every later
      batch is padding; this hits runs/runcpu.sh). Identical to Python at 2048.
      Upstream quirks kept: `num_iterations` counts micro-batches; Muon weight decay starts at 0
      instead of the base run's last (near-zero) value.
    - Base eval: `EvalBundle` (download, deflate unzip, core.yaml subset, baselines CSV),
      `CoreEval` (Jinja templates rendered by hand, Python `str.strip`, few-shot `sample`,
      mean-loss / argmax scoring) matches `evaluate_core` on prompts, tokens, spans, outcomes and
      the metric (mini bundle fixture). CLI `eval` (core, bpb, sample; CSV in `<base>/base_eval`).
      Real run with runcpu.sh settings on the d6 model: all 22 tasks, 74 s.
    - RL: `RlTrainer` = chat_rl.py (sft start, rollouts with forced tool tokens masked, rewards
      from GSM8K's checker, advantage = reward - mean, token-level policy gradient through the new
      `crossEntropyWeightedBackward` op and `Gpt.backwardWeighted`, LR ramp to zero, pass@k evals,
      `chatrl_checkpoints/<tag>` model-only saves). On fixed rollouts the per-pass losses match
      PyTorch within 1e-5 and every gradient within 2e-5 of its scale; sampling uses this port's
      RNG. CLI `rl`, console page. SFT and RL now write `metrics.jsonl` too.
11. **Web console** (`web/`, one embedded HTML page) — done (page not yet checked in a browser)
    - Chat UI with token streaming (SSE).
    - Training dashboard tailing `metrics.jsonl`: loss/bpb curves, LR, tok/s, MFU, samples.
    - Result: `zignanogpt-web [--host] [--port] [-i -g -s]` (`make web`) on `std.http.Server`,
      one task per connection. `GET /` the embedded page (no external assets, light/dark);
      `POST /api/chat` streams the reply as server-sent events (whole UTF-8 characters, one
      generation at a time, stateless: the page sends the history); `GET /api/runs`,
      `/api/metrics?run=&offset=` tail `<base>/{base,chatsft,chatrl}_checkpoints/<tag>/metrics.jsonl`
      (run names validated against the listing). Dashboard: stat tiles, canvas charts (loss,
      val bpb / ChatCORE / pass@1, reward, LR multiplier, tok/s), latest samples. Without a
      checkpoint the dashboard still runs. Checked with curl against real runs.
12. **CPU performance** — done
    - Profile matmul/attention, cache blocking, thread tuning; compare tok/s vs PyTorch CPU.
    - Result (macOS `sample` on the CPU preset, d6, 32 x 512): attention backward was a third of
      the step and scalar `exp` (f64 in the cross entropy) another chunk. Changes: `CpuMath`
      (vectorized Cephes `exp`, <= 2 ulp; f64-summed log-sum-exp), query-blocked SIMD attention
      forward (flash style, 16 queries per vector, online softmax per 16-key chunk) and backward
      (scores, `dout.v` and `dq` as vector FMAs over transposed blocks), `k`-by-`k` packing of
      transposed A (weight gradients 650 -> 880 GFLOP/s), adaptive tile size for few-tile
      products, and the decode matrix-vector path (phase 9).
    - Step 5.5 s -> 3.68 s (0.45 -> 0.68 TFLOP/s, 4,450 tok/s). PyTorch CPU (eager, same preset,
      same machine, 18 cores): 2.95 s, 5,570 tok/s; its matmuls run on Accelerate (AMX), which
      plain SIMD cannot match. Remaining profile: matmul ~55%, attention backward ~15%, waits
      between short parallel ops ~15%. All parity tests unchanged.
13. **GPU backends** (M3) — Metal done; CUDA builds (compile-only until the Spark)
    - `MetalBackend`: Objective-C runtime via Zig `extern`, MSL kernels as source strings.
    - `CudaBackend`: driver API via `extern`; kernels in Zig built for `nvptx64-cuda`, PTX
      embedded and loaded with `cuModuleLoadData`.
    - CUDA target: DGX Spark (GB10, aarch64 Linux, unified memory, sm_121). Compile-only on macOS
      until the Spark is available; then conformance + parity run there.
    - Both pass the op conformance suite; parity of a training step vs CPU.
    - Decide then: Metal matmul via MPS (system framework, called like CoreFoundation) or own
      MSL kernels; CUDA has no cuBLAS (a C library), so its GEMM is our own Zig kernel.
    - Result, Metal (`-Dbackend=metal`, the default on Apple silicon): `MetalBackend` over the Objective-C
      runtime (`Objc`: typed `objc_msgSend` casts, selectors, autorelease pools), one compute
      command buffer in issue order, shared-storage buffers. Decision: matmul via MPS
      (`MPSMatrixMultiplication`, cached per shape; 7-12 TFLOP/s against ~0.9 for a first MSL
      simdgroup kernel, kept for k = 0). MSL kernels (compiled from source at startup): the
      elementwise ops, embedding, soft cap, RMS norm and backward, rotary, gates, smear, value
      mix, cross entropy (rows, mean, backward, weighted backward), and `dot`/reductions in a
      second library built without fast math (double-float sums, so near-zero gradients match
      the CPU's f64 sums). Ops without a kernel (embedding backward, optimizer steps)
      run on the CPU kernels over the same unified memory after a sync; frees and scratch are
      released when the command buffer completes. The whole suite passes on Metal (conformance,
      PyTorch parity of forward/gradients/optimizer, KV cache, engine, SFT end to end, RL,
      CORE). CPU preset step: 0.94 s (CPU backend 3.68 s, PyTorch CPU 2.95 s, PyTorch MPS
      0.26 s, f32 like ours). Profile: attention on the CPU was 58% of that step, with the GPU
      idle around each call.
    - Flash attention on the GPU: forward, `dq` and `dk`/`dv` kernels on 8x8 simdgroup
      matrices (4 simdgroups x 8 rows per threadgroup, online softmax, head dim zero-padded to
      8/16/32/64/128; larger falls back to the CPU). Backward recomputes probabilities from
      the saved log-sum-exp and splits `dq` from `dk`/`dv`, so nothing is atomic and sums keep
      a fixed order. Matches the CPU kernels within 2e-4 at model sizes (GQA, windows, cache
      decoding). CPU preset step: 0.94 s -> 0.37 s (44,500 tok/s, 6.8 TFLOP/s).
    - Remaining profile (GPU time per step ~0.29 s, kernels timed one by one): matmuls 61%
      (forward/backward plus Muon's Newton-Schulz), `gate_linear_backward_dw` 8% (one thread
      per weight over every row), cross entropy and soft cap 10%, attention 12%. The rest of
      the step is CPU work between syncs: the optimizer, embedding backward and the target check.
    - Closing the gap (each step measured on the CPU preset, d6, 32 x 512):
      - Optimizer (AdamW, Muon momentum, MuonEq, NorMuon) as GPU kernels in the strict
        library (double-float norms); embedding backward on the GPU (rows grouped by id on
        the host, summed in the CPU's order); index tensors read on the host without a sync
        unless a queued copy writes them. 0.37 s -> 0.307 s.
      - `gate_linear_backward_dw` as one threadgroup per weight: -> 0.284 s.
      - Attention threadgroups of 8 simdgroups for head dim <= 64 (occupancy; 36 -> 24 ms of
        GPU time): -> 0.271 s. Register-resident softmax via `thread_elements()` was slower
        on this compiler (it types the fragment as 64 elements) and was dropped.
      - `softcap` also writes each row's log-sum-exp (one pass, online max); the loss is then
        a gather and the backward skips its lse pass (optional `lse` on the soft-cap and
        cross-entropy ops; the CPU keeps its f64 loss): -> 0.259 s.
      - Muon parameters stacked by shape; Polar Express as batched MPS matmuls
        (`MatmulOptions.batch`): -> 0.249 s, 65,900 tok/s, 10.1 TFLOP/s (PyTorch MPS 0.26 s).
      - Now: GPU time is ~95% of the step; model matmuls ~163 ms of it (~14 TFLOP/s in MPS).
    - Result, CUDA (`-Dbackend=cuda`, `make cuda` cross-compiles for aarch64 Linux):
      `CudaBackend` loads `libcuda.so.1` at run time (`std.DynLib`; no link-time CUDA), one
      stream, managed memory (unified on GB10) with the same CPU fallback. Kernels are Zig
      (`src/cuda/kernels.zig`): Zig 0.16 emits exported kernels as aliases, which LLVM's NVPTX
      backend rejects, so the build emits LLVM IR, `tools/nvptx_fixup.zig` rewrites the aliases,
      and `zig cc` lowers it to sm_80 PTX (JIT-compiled forward by the driver, sm_121 included),
      embedded and loaded with `cuModuleLoadData`. GPU kernels: elementwise ops, embedding, RMS
      norm and backward, a one-thread-per-output matmul. Built and type-checked only: the
      conformance suite and a training step still have to run on the Spark, and the matmul is
      the first thing to tile there.
    - CUDA rough-out (blind, for the Spark): every op the model trains with now has a kernel
      (44 PTX entries, matched against `Kernel` by the build): a 64x64-tiled matmul through
      shared memory (batched via grid z), block-per-row norms / soft cap (+lse) / cross entropy,
      warp-per-row attention forward, dq and dk/dv (deterministic, head dim <= 256), the
      optimizer with f64 norms and sums, host-grouped embedding backward, rotary, gates, smear,
      value mix, and f64 two-stage `dot`. Frees and scratch are released at the next sync.
      Inline PTX for barriers, shuffles, `ex2`/`lg2` (tanh from `ex2`; PTX's own is ~11 bits).
      `make cuda` compiles it and every test binary; nothing has run on a GPU yet.
    - DGX bring-up checklist (native on DGX OS):
      1. `zig build`, then `./zig-out/bin/zignanogpt version` says `cuda backend`.
      2. `make test`: the conformance suite and the PyTorch parity fixtures on CUDA. First
         suspects on failure: the inline PTX (`kernels.zig`, validated only by the driver's JIT),
         block-size assumptions (256 threads, multiple of 32), the `AttnDims` byval parameter.
      3. `make train-bench` against PyTorch CUDA on the same Spark (nanochat's `base_train.py`,
         same d6 settings), and the Mac's Metal number (0.249 s per step).
      4. Speed, measured there: TF32 tensor-core matmul (`mma.sync`), flash-style attention,
         host-readable ids without a sync (GB10 has concurrent managed access), stream-ordered
         scratch (`cuMemAllocAsync`) instead of managed allocations per op.

## Out of scope

FP8, FA3, DDP, wandb, HumanEval (needs Python sandbox), notebooks, bf16 compute.

## Risks

- Bit-exact parity with PyTorch is not achievable (op ordering, bf16); tests use tolerances.
- CPU training is slow: d6 run ≈ hours; GPT-2-grade (d24+) needs a GPU backend.
- Tokenizer regex semantics are the most likely source of silent mismatch → exhaustive fixtures.
- Unicode version skew: `uucode` is Unicode 17.0; tiktoken/rustbpe use Rust `regex` tables
  (older). Characters assigned in between classify differently → fixtures cover them; document.
- Python checkpoints are bf16 for `wte`/`value_embeds`; f32 import is exact but Python-side
  logits ran in bf16, so imported-model parity needs looser tolerances.
- Designing the backend contract before Metal/CUDA exist may still need revision in M3.
- CUDA cannot run on macOS; until the DGX Spark is available the CUDA backend is compile-only.
- Zig's LLVM may not know `sm_121`; fallback is PTX for an older `sm_*`, JIT-compiled by the driver.
- Zig's `nvptx64` backend is less mature than its CPU targets; a kernel it cannot compile may
  force a hand-written PTX fallback for that op.
- `uucode` and `zigimg` are fetched from GitHub at build time (network on first build).
- Checkpoints get large (d20 ≈ 2 GB f32 weights, ×3 with AdamW/Muon state); zigstorage
  `file://` sessions stage to a temp file, so they need 2× free disk while saving.

## Resolved

| Q | Answer |
| :--- | :--- |
| Q1 First milestone | Pretrain + chat inference (0–9), then SFT/evals |
| Q2 Compute | CPU + SIMD now; design for multiple backends |
| Q3 Python checkpoints | Own format + Zig `.pt` importer |
| Q4 Fixtures | Dev-only Python script, committed to `testdata/` |
| Q5 Precision | f32 everywhere |
| Q6 Naming | ZIGSTYLE role names, no prefix |
| Q7 Web console | Chat UI + training dashboard |
| Q8 Backend selection | Build time, `-Dbackend=`, comptime dispatch |
| Q9 Second backends | Metal and CUDA |
| Q10 Dashboard data | Trainer writes `metrics.jsonl`; consumers tail it |
| Q11 CLI | One executable with subcommands; zigvaxis TUI; zigstorage for disk I/O |
| Q12 Python import | Weights + meta + `tokenizer.pkl` + `token_bytes` |
| Q13 Dev data | ClimbMix + tiny local corpus |
| Q14 Web console timing | M2 |
| Q15 TUI scope | `chat` + live `train` monitor; other commands plain text |
| Q16 Layering | zigstorage in library; zigtui (was zigvaxis directly) in `cli/` only |
| Q17 Unicode tables | `uucode` in the library, shared with zigvaxis: now the module zigtui builds and exports |
| Q18 Downloads | zigstorage `https://` into a `file://` session |
| Q19 GPU timing | M3; backend contract reviewed against both at phase 1 exit |
| Q20 CUDA kernels | Zig → `nvptx64-cuda` PTX via driver API |
| Q21 Dep paths | `../../inferise/<dep>` |
| Q22 Base dir | `~/.cache/zignanogpt`; `~/.cache/nanochat` read-only |
| Q23 Train defaults | Python's; `--preset=cpu` for `runcpu.sh` values |
| Q24 Git | No branches or commits by Claude; user drives all git |
| Q25 CUDA hardware | None yet (compile-only); target DGX Spark |
| Q26 Platforms | macOS ARM64 (Metal) and DGX OS Linux ARM64 (CUDA), both required; CUDA cross-checked by `make validate` on macOS |

## Open questions

None. Four rounds done; ready for review.
