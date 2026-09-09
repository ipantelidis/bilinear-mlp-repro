"""
Appendix I — Bilinear Transformers: A Loss Comparison (Table 6)
Reproduces the Bilinear / ReGLU / SwiGLU comparison from Pearce et al. (2025).

Three 6-layer TinyStories transformers (d_model 512, d_hidden 2048, 8 heads,
ctx 256 — the tdooms/ts-medium architecture) are trained for the same number
of epochs, differing only in the MLP activation:

    bilinear : (W x) ⊙ (V x)            gate=None
    reglu    : ReLU(W x) ⊙ (V x)        gate="relu"
    swiglu   : SiLU(W x) ⊙ (V x)        gate="silu"

The paper's two comparisons come from one set of runs:
  * constant epochs — final validation loss after EPOCHS epochs;
  * constant time   — validation loss at the wall-clock time when the
                      fastest variant finished (validation is evaluated
                      periodically with timestamps, so this is read off
                      the recorded curve).

USAGE
  # All three variants in parallel (one GPU each):
  python appendix_i.py --mode all --devices 3,4,5

  # Individually:
  python appendix_i.py --mode bilinear --device cuda:0

  # Build Table 6 + figure once all three results exist:
  python appendix_i.py --mode table

  # Smoke test (2% of data, 1 epoch):
  SUBSET=0.02 EPOCHS=1 python appendix_i.py --mode bilinear --device cuda:0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

os.chdir(Path(__file__).resolve().parents[3])
HERE = Path(__file__).parent

from language.transformer import Attention, Transformer

if not hasattr(Transformer, "all_tied_weights_keys"):
    Transformer.all_tied_weights_keys = property(lambda self: {})


def set_train(model):
    """train() everywhere except the Attention modules.

    Attention.forward branches on self.training: the training branch calls
    scaled_dot_product_attention with BOTH an explicit attn_mask and
    is_causal=True, which requires an attention_mask tensor and is rejected
    by current torch. The eval branch is functionally identical (the model
    has no dropout) and handles causal masking internally, and with
    right-padding real tokens never attend to pads, so training through it
    is exact.
    """
    model.train()
    for m in model.modules():
        if isinstance(m, Attention):
            m.eval()

# ════════════════════════════════════════════════════════════════
# CONFIGURATION (paper Table 2: TinyStories training setup)
# ════════════════════════════════════════════════════════════════

EPOCHS      = int(os.environ.get("EPOCHS", 5))       # paper: 5
SUBSET      = float(os.environ.get("SUBSET", 1.0))   # fraction of train data
BATCH       = 512                                    # paper: 512 sequences
MICRO_BATCH = 64                                     # per forward pass
GRAD_ACCUM  = BATCH // MICRO_BATCH
N_CTX       = 256                                    # paper: 256
LR          = 1e-3                                   # paper: 1e-3
WD          = 0.1                                    # paper: 0.1
WARMUP      = 50                                     # matches Transformer.fit
EVAL_EVERY  = 250                                    # opt steps between evals
LOG_EVERY   = 20
SEED        = 42

GATES = {"bilinear": None, "reglu": "relu", "swiglu": "silu"}

TOK_TRAIN = HERE / "tokens_train.pt"
TOK_VAL   = HERE / "tokens_val.pt"


# ════════════════════════════════════════════════════════════════
# DATA — tokenised once, shared by all three variants
# ════════════════════════════════════════════════════════════════

def build_token_cache():
    """Tokenise TinyStories into fixed-length id tensors (int16, vocab 4096)."""
    from datasets import load_dataset

    tok = Transformer.get_tokenizer("ts-4096")

    def encode(split):
        raw = load_dataset("roneneldan/TinyStories", split=split)
        texts = ["[BOS] " + t.lower() for t in raw["text"]]
        ids = tok(texts, truncation=True, padding="max_length",
                  max_length=N_CTX, return_tensors="pt",
                  add_special_tokens=False)["input_ids"]
        return ids.to(torch.int16), tok.pad_token_id

    if not TOK_TRAIN.exists():
        print("Tokenising train split (one-off, cached) …")
        ids, pad = encode("train")
        torch.save({"ids": ids, "pad": pad}, TOK_TRAIN)
        print(f"  {tuple(ids.shape)} → {TOK_TRAIN.name}")
    if not TOK_VAL.exists():
        print("Tokenising validation split (one-off, cached) …")
        ids, pad = encode("validation")
        torch.save({"ids": ids, "pad": pad}, TOK_VAL)
        print(f"  {tuple(ids.shape)} → {TOK_VAL.name}")


def load_tokens(path):
    data = torch.load(path, weights_only=True)
    return data["ids"], data["pad"]


# ════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def val_loss(model, ids, pad, device, max_seqs=None):
    model.eval()
    ids = ids[:max_seqs] if max_seqs else ids
    losses = []
    for s in range(0, len(ids), MICRO_BATCH):
        batch = ids[s:s + MICRO_BATCH].long().to(device)
        labels = batch.clone()
        labels[batch == pad] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch, labels=labels)
        losses.append(float(out.loss))
    set_train(model)
    return sum(losses) / len(losses)


def run_training(variant, device):
    build_token_cache()
    train_ids, pad = load_tokens(TOK_TRAIN)
    val_ids, _ = load_tokens(TOK_VAL)

    if SUBSET < 1.0:
        train_ids = train_ids[: int(len(train_ids) * SUBSET)]

    steps_per_epoch = len(train_ids) // BATCH
    total_steps = steps_per_epoch * EPOCHS

    print(f"\n{'='*60}")
    print(f"  Variant : {variant} (gate={GATES[variant]})")
    print(f"  Device  : {device}")
    print(f"  Data    : {len(train_ids)} seqs × {N_CTX} ctx, {EPOCHS} epochs")
    print(f"  Steps   : {total_steps} ({steps_per_epoch}/epoch, batch {BATCH})")
    print(f"{'='*60}\n")

    torch.manual_seed(SEED)
    model = Transformer.from_config(
        n_layer=6, d_model=512, d_hidden=2048, n_head=8, n_ctx=N_CTX,
        bilinear=True, gate=GATES[variant],
        normalization=True, tokenizer="ts-4096",
    ).to(device)
    set_train(model)
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

    optim = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WD,
        betas=(0.9, 0.95), fused=True,
    )

    # Linear warmup, then linear decay to 0 (paper: linear decay schedule)
    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP
        return max(0.0, 1.0 - (step - WARMUP) / max(1, total_steps - WARMUP))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    train_log, eval_log = [], []
    running, count = 0.0, 0
    t0 = time.time()
    opt_step = 0

    pbar = tqdm(total=total_steps, desc=variant)
    for epoch in range(EPOCHS):
        gen = torch.Generator().manual_seed(SEED + epoch)
        perm = torch.randperm(len(train_ids), generator=gen)

        for b in range(steps_per_epoch):
            idx = perm[b * BATCH:(b + 1) * BATCH]
            for m in range(GRAD_ACCUM):
                micro = train_ids[idx[m * MICRO_BATCH:(m + 1) * MICRO_BATCH]]
                batch = micro.long().to(device)
                labels = batch.clone()
                labels[batch == pad] = -100
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(batch, labels=labels)
                (out.loss / GRAD_ACCUM).backward()
                running += float(out.loss.detach())
                count += 1

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            opt_step += 1
            pbar.update(1)

            if opt_step % LOG_EVERY == 0:
                avg = running / count
                train_log.append((opt_step, opt_step * BATCH * N_CTX,
                                  time.time() - t0, avg))
                running, count = 0.0, 0
                pbar.set_postfix(loss=f"{avg:.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.1e}")

            if opt_step % EVAL_EVERY == 0:
                vl = val_loss(model, val_ids, pad, device, max_seqs=1024)
                eval_log.append((opt_step, time.time() - t0, vl))
                pbar.set_postfix(loss=f"{train_log[-1][3]:.4f}", val=f"{vl:.4f}")

    pbar.close()

    total_time = time.time() - t0
    final_val = val_loss(model, val_ids, pad, device)
    eval_log.append((opt_step, total_time, final_val))
    print(f"\n  Final validation loss: {final_val:.4f}  ({total_time/3600:.2f} h)")

    json.dump(
        {
            "variant": variant, "gate": GATES[variant],
            "epochs": EPOCHS, "subset": SUBSET, "batch": BATCH,
            "lr": LR, "wd": WD, "seed": SEED,
            "total_steps": opt_step, "total_seconds": total_time,
            "final_val_loss": final_val,
            "train_log": train_log,   # (step, tokens, seconds, train_loss)
            "eval_log": eval_log,     # (step, seconds, val_loss)
        },
        open(HERE / f"results_{variant}.json", "w"),
    )
    print(f"  Saved results_{variant}.json")


# ════════════════════════════════════════════════════════════════
# TABLE 6 + FIGURE
# ════════════════════════════════════════════════════════════════

def build_table():
    results = {}
    for v in GATES:
        p = HERE / f"results_{v}.json"
        if not p.exists():
            print(f"  Missing {p.name} — run --mode {v} first")
            return
        results[v] = json.load(open(p))

    # Constant time: the earliest total wall-clock time across variants;
    # for each variant take the last recorded validation loss before it.
    # Only the periodic evaluations (fixed 1024-sequence subset, steps at
    # multiples of EVAL_EVERY) are comparable across variants — the appended
    # final entry uses the full validation set and would bias the comparison
    # for whichever variant defines the cutoff.
    t_cut = min(r["total_seconds"] for r in results.values())

    def periodic(r):
        return [(st, s, vl) for st, s, vl in r["eval_log"] if st % EVAL_EVERY == 0]

    def loss_at(r, t):
        prior = [vl for _, s, vl in periodic(r) if s <= t]
        return prior[-1] if prior else float("nan")

    table = {"constant_epochs": {}, "constant_time": {},
             "cutoff_seconds": t_cut,
             "paper": {"constant_epochs": {"bilinear": 1.337, "reglu": 1.332, "swiglu": 1.321},
                        "constant_time": {"bilinear": 1.337, "reglu": 1.337, "swiglu": 1.336}}}
    for v, r in results.items():
        table["constant_epochs"][v] = r["final_val_loss"]
        table["constant_time"][v] = loss_at(r, t_cut)

    json.dump(table, open(HERE / "appendix_i_results.json", "w"), indent=2)

    print(f"\n{'':<16}{'Bilinear':>10}{'ReGLU':>10}{'SwiGLU':>10}")
    for row in ("constant_epochs", "constant_time"):
        ours = table[row]
        print(f"{row:<16}" + "".join(f"{ours[v]:>10.3f}" for v in GATES))
        pap = table["paper"][row]
        print(f"{'  (paper)':<16}" + "".join(f"{pap[v]:>10.3f}" for v in GATES))
    print(f"\nSaved appendix_i_results.json (cutoff {t_cut/3600:.2f} h)")

    # Loss curves over wall-clock time, with a zoomed inset on the final
    # stretch — the between-variant gaps (~0.006-0.02 nats) are invisible
    # at full scale.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=150)
    colors = {"bilinear": "#1f77b4", "reglu": "#2ca02c", "swiglu": "#d62728"}
    for v, r in results.items():
        steps, _, secs, losses = zip(*r["train_log"])
        axes[0].plot(steps, losses, color=colors[v], label=v, lw=1)
        # periodic subset evals only — the final full-set eval uses a
        # different protocol and would show as a spurious jump
        e_steps, e_secs, e_vl = zip(*periodic(r))
        axes[1].plot([s / 3600 for s in e_secs], e_vl, color=colors[v], label=v, lw=1.2)
    axes[1].axvline(t_cut / 3600, color="gray", ls="--", alpha=0.6, label="constant-time cut")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Train loss")
    axes[1].set_xlabel("Wall-clock (h)"); axes[1].set_ylabel("Validation loss")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    zoom_from = t_cut * 0.75 / 3600
    inset = axes[1].inset_axes([0.42, 0.42, 0.55, 0.53])
    lo, hi = 10.0, 0.0
    for v, r in results.items():
        pts = [(s / 3600, vl) for _, s, vl in periodic(r) if s / 3600 >= zoom_from]
        xs, ys = zip(*pts)
        inset.plot(xs, ys, color=colors[v], lw=1.2)
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
    inset.axvline(t_cut / 3600, color="gray", ls="--", alpha=0.6)
    inset.set_ylim(lo - 0.003, hi + 0.003)
    inset.tick_params(labelsize=6)
    inset.grid(True, alpha=0.3)
    axes[1].indicate_inset_zoom(inset, edgecolor="gray", alpha=0.5)

    fig.suptitle("Bilinear vs ReGLU vs SwiGLU on TinyStories (Appendix I)", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "appendix_i.png", bbox_inches="tight")
    print("Saved appendix_i.png")


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Appendix I — activation-function loss comparison")
    parser.add_argument("--mode", choices=list(GATES) + ["all", "table"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", default="3,4,5",
                        help="GPU indices for --mode all (one per variant)")
    args = parser.parse_args()

    if args.mode == "table":
        build_table()
    elif args.mode == "all":
        build_token_cache()   # tokenize once before forking
        gpus = [g.strip() for g in args.devices.split(",")]
        assert len(gpus) >= 3, "Need 3 GPU indices for --mode all"
        script = str(Path(__file__).resolve())
        procs = []
        for v, g in zip(GATES, gpus):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": g}
            procs.append(subprocess.Popen(
                [sys.executable, script, "--mode", v, "--device", "cuda:0"], env=env))
        print(f"Launched {list(GATES)} on GPUs {gpus[:3]} — waiting …")
        codes = [p.wait() for p in procs]
        if any(codes):
            print(f"WARNING: exit codes {codes}")
        else:
            build_table()
    else:
        run_training(args.mode, args.device)
