"""
Parity fixtures for the Zig port, generated from the Python reference in ../nanochat.

Dev only: the Zig build never runs this. `make fixtures` runs it inside the
nanochat uv environment and writes testdata/, which is committed.

File format: safetensors (8-byte little-endian header length, JSON header, raw
little-endian tensor bytes), written by hand so no extra package is needed.
The Zig checkpoint format is the same layout, so one reader serves both.

Each phase registers a group with @fixture; `--only NAME` regenerates one group.
"""

import argparse
import json
import os
import struct

# The optimizer kernels are @torch.compile'd; run them eagerly (same math, no C++ toolchain).
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

FIXTURES = {}


def fixture(fn):
    """Register a fixture group under the function's name."""
    FIXTURES[fn.__name__] = fn
    return fn


def write_safetensors(path, tensors, metadata=None):
    """Write {name: torch.Tensor} to `path` in safetensors layout."""
    import torch
    dtypes = {torch.float32: "F32", torch.int32: "I32", torch.int64: "I64", torch.uint8: "U8"}
    header, blobs, offset = {}, [], 0
    for name, t in sorted(tensors.items()):
        t = t.detach().contiguous().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()  # the Zig port is f32 everywhere
        data = t.numpy().tobytes()
        header[name] = {"dtype": dtypes[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)  # align the data section to 8 bytes
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for blob in blobs:
            f.write(blob)


# -----------------------------------------------------------------------------
# Phase 2: model forward

# Small, but every feature active: GQA (4 query / 2 kv heads), sliding windows
# (seq 512 -> short window 128 < T), value embeddings on layers 1 and 3, a vocab
# that needs padding (300 -> 320).
GPT_CONFIG = dict(sequence_len=512, vocab_size=300, n_layer=4, n_head=4, n_kv_head=2, n_embd=64, window_pattern="SSSL")
GPT_BATCH, GPT_SEQ = 2, 200


@fixture
def gpt(out):
    """Weights, token ids, rotary tables, per-block outputs and logits of one forward pass."""
    import torch
    from nanochat.gpt import GPT, GPTConfig
    torch.manual_seed(1234)
    config = GPTConfig(**GPT_CONFIG)
    model = GPT(config)
    model.init_weights()
    # init_weights zeroes the output projections and most scalars, which would
    # leave whole paths untested; perturb every parameter so each one matters.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    model.eval()

    idx = torch.randint(0, config.vocab_size, (GPT_BATCH, GPT_SEQ))
    captured = {}
    hooks = [block.register_forward_hook(lambda m, i, o, n=n: captured.__setitem__(f"debug.block.{n}", o.detach().clone()))
             for n, block in enumerate(model.transformer.h)]
    with torch.no_grad():
        logits = model(idx)
    for h in hooks:
        h.remove()

    tensors = {name: p for name, p in model.state_dict().items()}
    tensors.update(captured)
    tensors["idx"] = idx.to(torch.int32)
    tensors["logits"] = logits
    tensors["cos"] = model.cos[0, :, 0, :]
    tensors["sin"] = model.sin[0, :, 0, :]
    metadata = dict(GPT_CONFIG, batch=GPT_BATCH, seq=GPT_SEQ,
                    window_sizes=",".join(str(w) for w, _ in model.window_sizes),
                    flops_per_token=model.estimate_flops(),
                    num_params=sum(p.numel() for p in model.parameters()))
    write_safetensors(os.path.join(out, "gpt.safetensors"), tensors, metadata)


# -----------------------------------------------------------------------------
# Phase 3: loss and backward

GRAD_BATCH, GRAD_SEQ = 2, 160


@fixture
def gpt_grad(out):
    """Weights, ids, targets (some ignored), the mean loss, and every parameter gradient."""
    import torch
    from nanochat.gpt import GPT, GPTConfig
    torch.manual_seed(4321)
    config = GPTConfig(**GPT_CONFIG)
    model = GPT(config)
    model.init_weights()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    idx = torch.randint(0, config.vocab_size, (GRAD_BATCH, GRAD_SEQ))
    targets = torch.randint(0, config.vocab_size, (GRAD_BATCH, GRAD_SEQ))
    targets[0, :7] = -1  # ignore_index, as in SFT masking
    targets[1, -3:] = -1
    loss = model(idx, targets)
    loss.backward()

    tensors = {name: p.detach() for name, p in model.named_parameters()}
    tensors.update({f"grad.{name}": p.grad for name, p in model.named_parameters()})
    tensors["idx"] = idx.to(torch.int32)
    tensors["targets"] = targets.to(torch.int32)
    tensors["loss"] = loss.detach().reshape(1)
    metadata = dict(GPT_CONFIG, batch=GRAD_BATCH, seq=GRAD_SEQ)
    write_safetensors(os.path.join(out, "gpt_grad.safetensors"), tensors, metadata)


# -----------------------------------------------------------------------------
# Phase 4: optimizer

OPTIM_STEPS = 3
# setup_optimizer arguments; non-zero weight decay so the cautious mask matters.
OPTIM_HPARAMS = dict(unembedding_lr=0.008, embedding_lr=0.3, matrix_lr=0.02, weight_decay=0.2, scalar_lr=0.5)
# Per-step schedule values: (lr multiplier, muon momentum, muon weight decay).
OPTIM_SCHEDULE = [(1.0, 0.85, 0.2), (0.5, 0.9, 0.15), (0.25, 0.97, 0.1)]


@fixture
def optim(out):
    """Initial weights, a gradient per step, and the weights after each MuonAdamW step."""
    import torch
    from nanochat.gpt import GPT, GPTConfig
    torch.manual_seed(777)
    model = GPT(GPTConfig(**GPT_CONFIG))
    model.init_weights()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    optimizer = model.setup_optimizer(**OPTIM_HPARAMS)
    tensors = {f"init.{n}": p.detach().clone() for n, p in model.named_parameters()}
    for step, (lrm, momentum, wd) in enumerate(OPTIM_SCHEDULE):
        for n, p in model.named_parameters():
            p.grad = torch.randn_like(p) * 0.01
            tensors[f"grad{step}.{n}"] = p.grad.clone()
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = momentum
                group["weight_decay"] = wd
        optimizer.step()
        for n, p in model.named_parameters():
            tensors[f"step{step}.{n}"] = p.detach().clone()
    metadata = dict(GPT_CONFIG, steps=OPTIM_STEPS,
                    schedule=";".join(f"{a},{b},{c}" for a, b, c in OPTIM_SCHEDULE),
                    **{k: str(v) for k, v in OPTIM_HPARAMS.items()})
    write_safetensors(os.path.join(out, "optim.safetensors"), tensors, metadata)


# -----------------------------------------------------------------------------
# Phase 5: tokenizer

TOKENIZER_VOCAB = 1024 + 9  # 768 merges + 256 bytes + 9 special tokens
TOKENIZER_SOURCES = ["README.md", "dev/LOG.md", "dev/LEADERBOARD.md", "nanochat/gpt.py", "nanochat/engine.py"]
TOKENIZER_EDGE_CASES = [
    "Hello world! This is a test.\nNumbers: 123, 4567, 89\nContractions: I'm, you're, it's",
    "Special chars: @#$%^&*()\nUnicode: 你好世界 🌍",
    "I'LL SAY it's DON'T we've they're ſ 'Re 'mS",
    "   leading spaces, trailing spaces   ",
    "tabs\tand\t\tmore\ttabs\n\n\nthree newlines\r\nwindows\r\n\r\nend",
    "nbsp here, ideographic　space, line sep, para sep, thin space",
    "digits 0 12 345 6789 ²³ ½ Ⅻ ٣٤٥ ١٢",
    "한국어 텍스트입니다. 日本語のテキスト。 Ελληνικά κείμενα. Русский текст.",
    "combining: é ä ñ — dashes – and … ellipsis «quotes» “curly” ‘single’",
    "emoji 👨‍👩‍👧 family 🏳️‍🌈 flag 1️⃣ keycap",
    "code: def f(x): return x**2 + 1  # comment\n    if a!=b and c<=d: pass",
    "math: ∑_{i=1}^{n} x_i ≤ ∞, α+β=γ, ∀x∈ℝ",
    "\n",
    " ",
    "a",
    "",
    "    \n    \n",
    "punct!!!???...,,,;;; ((( ))) ''' \"\"\"",
]
TOKENIZER_CONVERSATIONS = [
    {"messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 is 4."},
    ]},
    {"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Compute 12*12."},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me compute. "},
            {"type": "python", "text": "12*12"},
            {"type": "python_output", "text": "144"},
            {"type": "text", "text": "It is 144."},
        ]},
        {"role": "user", "content": "Thanks!"},
        {"role": "assistant", "content": "You're welcome."},
    ]},
]


@fixture
def tokenizer(out):
    """Train with rustbpe on a small corpus; dump the merges, encodings, renders and token bytes."""
    import regex
    from nanochat.tokenizer import RustBPETokenizer, SPLIT_PATTERN, SPECIAL_TOKENS
    root = os.path.dirname(os.path.abspath(__import__("nanochat").__file__))
    root = os.path.dirname(root)
    docs = []
    for rel in TOKENIZER_SOURCES:
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            docs.extend(p for p in f.read().split("\n\n") if p)
    docs.extend(c for c in TOKENIZER_EDGE_CASES if c)
    tok = RustBPETokenizer.train_from_iterator(iter(docs), TOKENIZER_VOCAB)
    enc = tok.enc
    n_merge = len(enc._mergeable_ranks)
    ranks = [None] * n_merge
    for b, r in enc._mergeable_ranks.items():
        ranks[r] = list(b)
    cases = []
    for text in TOKENIZER_EDGE_CASES + docs[:40]:
        cases.append({"text": text, "ids": tok.encode(text), "pieces": regex.findall(SPLIT_PATTERN, text)})
    convs = []
    for conv in TOKENIZER_CONVERSATIONS:
        ids, mask = tok.render_conversation(conv)
        convs.append({"conversation": conv, "ids": ids, "mask": mask})
    completion = tok.render_for_completion(TOKENIZER_CONVERSATIONS[1])
    special_ids = {tok.encode_special(s) for s in SPECIAL_TOKENS}
    token_bytes = [0 if i in special_ids else len(tok.decode_single_token_bytes(i)) for i in range(tok.get_vocab_size())]
    data = dict(pattern=SPLIT_PATTERN, vocab_size=TOKENIZER_VOCAB, docs=docs, ranks=ranks,
                specials={s: tok.encode_special(s) for s in SPECIAL_TOKENS},
                cases=cases, conversations=convs, completion=completion, token_bytes=token_bytes)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Phase 6: parquet and the dataloader

def _tokenizer_from_fixture(out):
    """The phase-5 fixture tokenizer, rebuilt as a nanochat RustBPETokenizer."""
    import tiktoken
    from nanochat.tokenizer import RustBPETokenizer, SPLIT_PATTERN, SPECIAL_TOKENS
    with open(os.path.join(out, "tokenizer.json"), encoding="utf-8") as f:
        data = json.load(f)
    ranks = {bytes(b): i for i, b in enumerate(data["ranks"])}
    specials = {s: len(ranks) + i for i, s in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name="fixture", pat_str=SPLIT_PATTERN, mergeable_ranks=ranks, special_tokens=specials)
    return RustBPETokenizer(enc, "<|bos|>"), data["docs"]


@fixture
def parquet(out):
    """Small parquet files covering the encodings the reader supports, and their strings."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.join(out, "parquet"), exist_ok=True)
    rng_texts = [f"document {i}: " + "word " * (i % 37) + "é" * (i % 3) for i in range(100)]
    with_nulls = [None if i % 11 == 5 else t for i, t in enumerate(rng_texts)]
    repeated = [["alpha", "beta", "gamma", "δέλτα"][i % 4] for i in range(100)]
    variants = {
        "v1_zstd_nulls": (with_nulls, True, dict(compression="zstd", data_page_version="1.0", use_dictionary=False, data_page_size=512)),
        "v2_zstd": (rng_texts, True, dict(compression="zstd", data_page_version="2.0", use_dictionary=False, data_page_size=512)),
        "dict_snappy": (repeated, True, dict(compression="snappy", data_page_version="1.0", use_dictionary=True)),
        "plain_required": (rng_texts, False, dict(compression="none", use_dictionary=False)),
    }
    expected = {}
    for name, (texts, nullable, opts) in variants.items():
        schema = pa.schema([pa.field("text", pa.string(), nullable=nullable)])
        table = pa.table({"text": pa.array(texts, type=pa.string())}, schema=schema)
        pq.write_table(table, os.path.join(out, "parquet", f"{name}.parquet"), row_group_size=32, **opts)
        pf = pq.ParquetFile(os.path.join(out, "parquet", f"{name}.parquet"))
        expected[name] = [[t for t in pf.read_row_group(i).column("text").to_pylist() if t is not None]
                          for i in range(pf.num_row_groups)]
    with open(os.path.join(out, "parquet", "expected.json"), "w", encoding="utf-8") as f:
        json.dump(expected, f, ensure_ascii=False)


LOADER_B, LOADER_T, LOADER_TOK_BATCH, LOADER_BUFFER = 2, 48, 8, 24


@fixture
def dataloader(out):
    """A 3-shard dataset (2 train + val) and the bestfit loader's batches, resume included."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import nanochat.dataloader as dl
    tok, docs = _tokenizer_from_fixture(out)
    shard_dir = os.path.join(out, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    thirds = [docs[:250], docs[250:450], docs[450:]]
    paths = []
    for i, part in enumerate(thirds):
        path = os.path.join(shard_dir, f"shard_{i:05d}.parquet")
        pq.write_table(pa.table({"text": part}), path, row_group_size=16, compression="zstd")
        paths.append(path)
    dl.list_parquet_files = lambda *a, **k: list(paths)

    def run(split, n, resume=None):
        loader = dl.tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tok, LOADER_B, LOADER_T, split, tokenizer_batch_size=LOADER_TOK_BATCH, device="cpu",
            resume_state_dict=resume, buffer_size=LOADER_BUFFER)
        batches = []
        for _ in range(n):
            x, y, state = next(loader)
            batches.append({"inputs": x.flatten().tolist(), "targets": y.flatten().tolist(), "state": state})
        return batches

    train = run("train", 300)  # crosses both train files and into epoch 3
    resumed = run("train", 10, resume=train[130]["state"])
    val = run("val", 5)
    data = dict(B=LOADER_B, T=LOADER_T, tokenizer_batch_size=LOADER_TOK_BATCH, buffer_size=LOADER_BUFFER,
                train=train, resume_from=130, resumed=resumed, val=val)
    with open(os.path.join(out, "dataloader.json"), "w", encoding="utf-8") as f:
        json.dump(data, f)


# -----------------------------------------------------------------------------
# Phase 7: importing Python checkpoints

IMPORT_CONFIGS = {
    # d2: a current checkpoint, as base_train writes it on a GPU (bf16 embeddings).
    "d2": dict(sequence_len=64, vocab_size=1033, n_layer=2, n_head=4, n_kv_head=2, n_embd=64, window_pattern="SL"),
    # d1: a legacy checkpoint: torch.compile key prefix, no resid/x0 lambdas, no window_pattern.
    "d1": dict(sequence_len=64, vocab_size=1033, n_layer=1, n_head=4, n_kv_head=2, n_embd=64),
}


@fixture
def nanochat_import(out):
    """A fake nanochat base dir (checkpoints, meta, tokenizer.pkl) and build_model's logits."""
    import shutil
    import torch
    from nanochat.gpt import GPT, GPTConfig
    base = os.path.join(out, "nanochat_base")
    shutil.rmtree(base, ignore_errors=True)
    tok, _ = _tokenizer_from_fixture(out)
    tok.save(os.path.join(base, "tokenizer"))
    torch.manual_seed(99)
    idx = torch.randint(0, 1033, (1, 20))
    tensors = {"idx": idx.to(torch.int32)}
    for tag, cfg in IMPORT_CONFIGS.items():
        model = GPT(GPTConfig(**cfg))
        model.init_weights()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        sd = model.state_dict()
        if tag == "d1":
            sd = {f"_orig_mod.{k}": v for k, v in sd.items() if k not in ("resid_lambdas", "x0_lambdas")}
            meta_cfg = {k: v for k, v in cfg.items()}
        else:
            sd = {k: (v.to(torch.bfloat16) if k.startswith(("transformer.wte", "value_embeds")) else v) for k, v in sd.items()}
            meta_cfg = dict(cfg)
        ckpt = os.path.join(base, "base_checkpoints", tag)
        os.makedirs(ckpt, exist_ok=True)
        torch.save(sd, os.path.join(ckpt, "model_000005.pt"))
        meta = {"step": 5, "val_bpb": 1.25, "model_config": meta_cfg, "user_config": {"depth": cfg["n_layer"]},
                "device_batch_size": 4, "max_seq_len": cfg["sequence_len"], "total_batch_size": 256,
                "dataloader_state_dict": {"pq_idx": 0, "rg_idx": 3, "epoch": 1},
                "loop_state": {"min_val_bpb": 1.25, "smooth_train_loss": 3.5, "total_training_time": 12.0}}
        with open(os.path.join(ckpt, "meta_000005.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if tag == "d2":
            open(os.path.join(ckpt, "model_000003.pt"), "wb").close()  # older step; never loaded
    # Expected logits come from nanochat's own loader, which upcasts bf16 and patches legacy keys.
    os.environ["NANOCHAT_BASE_DIR"] = base
    from nanochat.checkpoint_manager import build_model
    for tag in IMPORT_CONFIGS:
        model, _, _ = build_model(os.path.join(base, "base_checkpoints", tag), 5, torch.device("cpu"), "eval")
        with torch.no_grad():
            tensors[f"logits.{tag}"] = model(idx)
    write_safetensors(os.path.join(out, "nanochat_import.safetensors"), tensors)


# -----------------------------------------------------------------------------
# Phase 9: chat inference

CALC_EXPRS = [
    "2*3", "10/2", "2/3", "1,234 * 5", "1,000/4", "-7 // 2", "7 // -2", "7.5 // 2", "-7.5 // 2", "7 // -2.0",
    "100000000000000000 * 10", "10000000000000000.0", "123456789.123456789", "0.000015", ".0001", "5.",
    "1/3*3", "0.1 * 3", "1/7", "2/3*100000000000000000", "-1/3000000", "3.14159 * 2", "(((1+2)))",
    "3 -- 2", "+-+5", " 12 + 3", "-0.0", "0 * -1.5", "00", "0.5", "2 ** 10", "1/0", "1//0", "1.0//0",
    "007", "2(3)", "(2)(3)", "", "   ", "1.5.5", "1 2", "1 / / 2", "1e5",
    "'strawberry'.count('r')", "\"banana\".count(\"an\")", "'abc'.count('')", "'a' 'b'.count('a')",
    "'Mississippi' . count ( 'ss' )", "'abc'.upper()", "x.count('a')", "len('abc')", "__import__('os')",
    "'open'.count('o')",
]
ENGINE_PROMPTS = ["The capital of France is", "Hello"]


@fixture
def engine(out):
    """use_calculator on edge-case expressions; Engine greedy generations on the d2 import model."""
    import torch
    os.environ["NANOCHAT_BASE_DIR"] = os.path.join(out, "nanochat_base")
    from nanochat.engine import Engine, use_calculator
    from nanochat.checkpoint_manager import build_model
    calculator = []
    for expr in CALC_EXPRS:
        result = use_calculator(expr)
        calculator.append([expr, None if result is None else str(result)])
    model, tokenizer, _ = build_model(os.path.join(out, "nanochat_base", "base_checkpoints", "d2"), 5, torch.device("cpu"), "eval")
    engine = Engine(model, tokenizer)
    generations = []
    for prompt in ENGINE_PROMPTS:
        tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
        results, masks = engine.generate_batch(tokens, num_samples=2, max_tokens=24, temperature=0.0)
        generations.append({"prompt": prompt, "tokens": tokens, "results": results, "masks": masks})
    with open(os.path.join(out, "engine.json"), "w") as f:
        json.dump({"calculator": calculator, "generations": generations}, f, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="testdata", help="output directory")
    parser.add_argument("--only", action="append", help="fixture group to generate (repeatable)")
    parser.add_argument("--list", action="store_true", help="list fixture groups and exit")
    args = parser.parse_args()
    if args.list:
        print("\n".join(sorted(FIXTURES)) or "(no fixture groups yet)")
        return
    names = args.only or sorted(FIXTURES)
    unknown = [n for n in names if n not in FIXTURES]
    if unknown:
        parser.error(f"unknown fixture group(s): {', '.join(unknown)}")
    for name in names:
        print(f"fixture: {name}")
        FIXTURES[name](args.out)
    if not names:
        print("(no fixture groups yet)")


if __name__ == "__main__":
    main()
