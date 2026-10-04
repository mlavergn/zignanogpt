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
