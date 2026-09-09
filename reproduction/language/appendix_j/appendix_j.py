"""
Appendix J — Finetuning: Your Transformer is Secretly Bilinear
Reproduces Figure 25 from Pearce et al. (2025)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY IDEA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TinyLlama's MLP uses SwiGLU:

    out = down_proj( gate · sigmoid(β · gate)  ⊙  up )
                     ───────────────────────
                         Swish_β(gate)

  β = 1.0  →  SiLU  (standard, the pretrained behaviour)
  β = 0.0  →  gate · 0.5  (linear gate)

When β = 0, the MLP becomes:
    out = down_proj( 0.5 · gate ⊙ up )
        ∝ down_proj( gate_proj(x) ⊙ up_proj(x) )

which is exactly a bilinear MLP — the element-wise product of
two linear projections of the input, with no nonlinearity.

We convert the pretrained model by linearly annealing β: 1 → 0
over the first 30 % of training, then holding at β = 0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run both experiments in parallel on GPUs 6 and 7 (recommended):
python appendix_j.py --mode both --devices 6,7

# Or run individually:
python appendix_j.py --mode finetune --device cuda:6
python appendix_j.py --mode baseline --device cuda:7

# Plot once both results files exist:
python appendix_j.py --mode plot

# Quick test (500 steps):
TOTAL_STEPS=500 python appendix_j.py --mode both --devices 6,7
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# Paper: ~400M tokens total = 100K steps × 8 seqs × 512 tokens
# ════════════════════════════════════════════════════════════════

TOTAL_STEPS  = int(os.environ.get("TOTAL_STEPS", 100_000))

# β annealing: linearly go from 1.0 → 0.0 over this fraction of steps
INTERP_FRAC  = 0.30      # first 30 % → 120M tokens (paper)

# Batching: micro-batch × grad-accum = effective batch per step
MICRO_BATCH  = 4         # sequences per forward pass (fits 24 GB A5500)
GRAD_ACCUM   = 2         # accumulate before each optimiser step
# Effective: 4 × 2 × 512 = 4096 tokens/step → 409M tokens total ✓

MAX_LENGTH   = 512       # context length (tokens)
# NOTE: 6e-4 (the paper's Table-3 value) is the LR for pretraining their small
# models FROM SCRATCH with large batches. Applied to fine-tuning a pretrained
# 1.1B model with a 4K-token batch it diverges (loss 2.7 → 7+ right after
# warmup — observed in the first run). Use a fine-tuning-scale LR instead.
LR           = float(os.environ.get("LR", 3e-5))
WEIGHT_DECAY = 0.1      # from paper
WARMUP_FRAC  = 0.01     # linear warmup over first 1 % of steps
GRAD_CLIP    = 1.0      # max gradient norm

LOG_EVERY    = 50       # log loss every N optimiser steps
SAVE_EVERY   = int(os.environ.get("SAVE_EVERY", 0))  # checkpoint every N steps (0 = disable; each is ~2.2 GB)

MODEL_ID     = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

INTERP_STEPS = int(TOTAL_STEPS * INTERP_FRAC)


# ════════════════════════════════════════════════════════════════
# β-PARAMETERISED ACTIVATION
# ════════════════════════════════════════════════════════════════

class BetaSiLU(nn.Module):
    """
    Swish_β(x) = x · sigmoid(β · x)

    We store β as a plain Python float — NOT a nn.Parameter — so the
    optimiser never touches it. Only our annealing schedule changes it.

    All 22 MLP blocks share the same BetaSiLU instance (set once in
    patch_model), so a single assignment like `act.beta = 0.5` instantly
    updates every layer simultaneously.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta   # mutable Python float, not a tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # At β=0: sigmoid(0)=0.5  →  output = 0.5·x  (linear, no nonlinearity)
        # At β=1: sigmoid(x)      →  standard SiLU
        return x * torch.sigmoid(self.beta * x)


def patch_model(model: nn.Module) -> BetaSiLU:
    """
    Surgically replace the SiLU activation in every LlamaMLP with BetaSiLU.

    TinyLlama's MLP forward (simplified):
        gate = gate_proj(x)          # shape: (seq, d_ffn)
        up   = up_proj(x)            # shape: (seq, d_ffn)
        out  = down_proj(act_fn(gate) * up)

    We replace act_fn in-place. Because all layers share one BetaSiLU
    object, changing act.beta changes all layers at once.
    """
    act = BetaSiLU(beta=1.0)
    for layer in model.model.layers:
        layer.mlp.act_fn = act
    print(f"  Patched act_fn in {len(model.model.layers)} MLP layers with BetaSiLU")
    return act


# ════════════════════════════════════════════════════════════════
# STREAMING DATASET
# ════════════════════════════════════════════════════════════════

class FineWebStream(IterableDataset):
    """
    Streams FineWeb text, tokenises documents, and packs tokens into
    fixed-length windows of MAX_LENGTH (no padding wasted).

    IMPORTANT: HuggingFace streaming datasets don't support multi-worker
    DataLoader sharding (SkipExamplesIterable raises NotImplementedError).
    Always use num_workers=0 with this class.
    """

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __iter__(self):
        ds = load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT",
            split="train", streaming=True,
        )
        buffer = []
        for example in ds:
            ids = self.tok(example["text"], add_special_tokens=False)["input_ids"]
            buffer.extend(ids)
            # Emit complete windows; discard the tail (< MAX_LENGTH tokens)
            while len(buffer) >= MAX_LENGTH:
                yield torch.tensor(buffer[:MAX_LENGTH], dtype=torch.long)
                buffer = buffer[MAX_LENGTH:]


# ════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════════

def run_training(mode: str, device: str) -> list[tuple[int, float]]:
    """
    Run one experiment and return a list of (optimizer_step, loss) pairs.

    mode = 'finetune':  β anneals 1.0 → 0.0 over the first INTERP_STEPS,
                        then stays at 0.0 (fully bilinear) for the rest.
    mode = 'baseline':  β stays at 1.0 throughout (continued pre-training).
    """
    eff_batch_tokens = MICRO_BATCH * GRAD_ACCUM * MAX_LENGTH
    total_tokens_M   = TOTAL_STEPS * eff_batch_tokens / 1e6

    print(f"\n{'='*60}")
    print(f"  Mode   : {mode}")
    print(f"  Device : {device}")
    print(f"  Steps  : {TOTAL_STEPS}  (interp ends at step {INTERP_STEPS})")
    print(f"  Batch  : {MICRO_BATCH} seqs × {GRAD_ACCUM} accum = "
          f"{eff_batch_tokens} tokens/step")
    print(f"  Total  : {total_tokens_M:.0f}M tokens")
    print(f"{'='*60}\n")

    # ── Tokeniser ────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── Model ────────────────────────────────────────────────
    print("Loading TinyLlama-1.1B (bfloat16) …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,   # ~2.2 GB weights (vs 4.4 GB in fp32)
        device_map=device,
    )
    model.train()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  {n_params:.2f}B parameters")

    # ── Patch activation ─────────────────────────────────────
    # Even for the baseline we patch (β stays 1.0) so both runs are
    # identical except for the β schedule — a clean controlled comparison.
    act = patch_model(model)

    # ── Optimiser ────────────────────────────────────────────
    # AdamW with betas=(0.9, 0.95) is common for LLM continued training
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # Linear warmup then linear decay to 10 % of peak LR
    WARMUP = max(100, int(TOTAL_STEPS * WARMUP_FRAC))
    def lr_lambda(step):
        if step < WARMUP:
            return step / WARMUP                                    # warmup
        frac = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
        return max(0.1, 1.0 - 0.9 * frac)                         # linear decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ── Dataset ──────────────────────────────────────────────
    # Baseline sees a different slice of FineWeb so we're comparing
    # β-annealing against the best possible continued pre-training baseline.
    ds = FineWebStream(tok)
    # num_workers=0 required: HF streaming datasets don't support worker sharding
    loader = DataLoader(ds, batch_size=MICRO_BATCH, num_workers=0)

    # ── Training ─────────────────────────────────────────────
    losses         = []        # (optimizer_step, loss) pairs for plotting
    opt_step       = 0         # counts complete optimiser updates
    running_loss   = 0.0       # accumulates raw (unscaled) loss for logging
    running_count  = 0         # number of micro-batches accumulated since last log

    pbar = tqdm(loader, total=TOTAL_STEPS, desc=mode)

    for micro_step, batch in enumerate(pbar):
        if opt_step >= TOTAL_STEPS:
            break

        # ── Update β schedule (finetune only) ────────────────
        if mode == "finetune":
            if opt_step < INTERP_STEPS:
                # Linearly anneal: β = 1 at step 0, β = 0 at step INTERP_STEPS
                act.beta = 1.0 - opt_step / INTERP_STEPS
            else:
                act.beta = 0.0   # fully bilinear — no more nonlinearity

        # ── Forward pass ─────────────────────────────────────
        ids  = batch.to(device)
        # labels = ids means we compute next-token prediction loss (causal LM)
        out  = model(input_ids=ids, labels=ids)
        # Divide by GRAD_ACCUM so the accumulated gradient equals the
        # gradient of the mean loss over the full effective batch
        loss = out.loss / GRAD_ACCUM
        loss.backward()
        running_loss += float(out.loss.detach())   # log the unscaled loss
        running_count += 1

        # ── Optimiser step (every GRAD_ACCUM micro-batches) ──
        if (micro_step + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)   # set_to_none saves memory

            # Log — average over the micro-batches accumulated since the last
            # log (LOG_EVERY optimiser steps × GRAD_ACCUM micro-batches).
            # Dividing by LOG_EVERY alone would inflate the loss by GRAD_ACCUM×.
            if opt_step % LOG_EVERY == 0:
                avg = running_loss / running_count
                losses.append((opt_step, avg))
                running_loss = 0.0
                running_count = 0
                pbar.set_postfix(
                    loss=f"{avg:.4f}",
                    beta=f"{act.beta:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            # Checkpoint
            if SAVE_EVERY and opt_step > 0 and opt_step % SAVE_EVERY == 0:
                ckpt = HERE / f"ckpt_{mode}_step{opt_step:06d}.pt"
                torch.save({"step": opt_step, "losses": losses,
                            "model_state": model.state_dict()}, ckpt)
                print(f"\n  Saved checkpoint: {ckpt.name}")

            opt_step += 1

    # Save results for plotting
    out_file = HERE / f"results_{mode}.json"
    json.dump(losses, open(out_file, "w"))
    print(f"\n  Saved {out_file.name}  ({len(losses)} log points)")

    del model
    torch.cuda.empty_cache()
    return losses


# ════════════════════════════════════════════════════════════════
# PLOT — Figure 25
# ════════════════════════════════════════════════════════════════

def plot():
    """Load results_{finetune,baseline}.json and reproduce Figure 25."""
    results = {}
    for mode in ("finetune", "baseline"):
        p = HERE / f"results_{mode}.json"
        if not p.exists():
            print(f"  Missing {p.name} — run --mode {mode} first")
            return
        results[mode] = json.load(open(p))

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=150)
    colors  = {"finetune": "#1f77b4", "baseline": "#ff7f0e"}

    for mode, data in results.items():
        xs = [s for s, _ in data]
        ys = [l for _, l in data]
        # 5-point moving average to smooth step noise. mode="valid" avoids the
        # zero-padded endpoints of mode="same", which would fake dips at the
        # start and end of the curve.
        ys_smooth = np.convolve(ys, np.ones(5) / 5, mode="valid")
        xs_smooth = xs[2:len(xs) - 2]
        ax.plot(xs_smooth, ys_smooth, color=colors[mode], label=mode, linewidth=1.5)

    # Mark end of β-annealing phase
    max_step = max(max(s for s, _ in d) for d in results.values())
    interp   = int(max_step * INTERP_FRAC)
    ax.axvline(interp, color="gray", linestyle="--", alpha=0.6,
               label=f"β reaches 0 (step {interp:,})")

    ax.set_xlabel("Step",  fontsize=11)
    ax.set_ylabel("Loss",  fontsize=11)
    ax.set_title("TinyLlama-1.1B: SwiGLU → bilinear fine-tuning (Fig. 25)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "fig_25.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → fig_25.png")

    ft_final   = results["finetune"][-1][1]
    base_final = results["baseline"][-1][1]
    print(f"\nFinal loss — finetune: {ft_final:.3f}  |  baseline: {base_final:.3f}")
    print(f"Paper target          — finetune: 2.217  |  baseline: 2.168")
    print(f"Gap: {ft_final - base_final:+.3f}  (paper: +0.049)")


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Appendix J — bilinear fine-tuning of TinyLlama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run on GPUs 6 & 7 in parallel (recommended):
  python appendix_j.py --mode both --devices 6,7

  # Individual runs (e.g. to resume one experiment):
  python appendix_j.py --mode finetune --device cuda:6
  python appendix_j.py --mode baseline --device cuda:7

  # Plot after both runs finish:
  python appendix_j.py --mode plot

  # Quick sanity-check (500 steps, ~5 min):
  TOTAL_STEPS=500 python appendix_j.py --mode both --devices 6,7
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["finetune", "baseline", "plot", "both"],
        default="both",
        help="Which experiment to run (default: both in parallel)",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device for a single run, e.g. --device cuda:6",
    )
    parser.add_argument(
        "--devices", default="6,7",
        help="Comma-separated GPU indices for --mode both (default: 6,7)",
    )
    args = parser.parse_args()

    if args.mode == "plot":
        plot()

    elif args.mode == "both":
        # ── Launch finetune and baseline in parallel ──────────
        # Each subprocess gets CUDA_VISIBLE_DEVICES=<gpu>, so from its
        # perspective the single visible GPU is always "cuda:0".
        gpus = [g.strip() for g in args.devices.split(",")]
        assert len(gpus) >= 2, "Need at least 2 GPU indices for --mode both"
        gpu_ft, gpu_bl = gpus[0], gpus[1]

        script = str(Path(__file__).resolve())
        total  = os.environ.get("TOTAL_STEPS", str(TOTAL_STEPS))

        def launch(mode, gpu):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "TOTAL_STEPS": total}
            return subprocess.Popen(
                [sys.executable, script, "--mode", mode, "--device", "cuda:0"],
                env=env,
            )

        print(f"Launching finetune on GPU {gpu_ft} and baseline on GPU {gpu_bl} …")
        p_ft = launch("finetune", gpu_ft)
        p_bl = launch("baseline", gpu_bl)
        print(f"  PIDs: finetune={p_ft.pid}  baseline={p_bl.pid}")
        print(f"  Waiting (TOTAL_STEPS={total}) …\n")

        p_ft.wait()
        p_bl.wait()

        if p_ft.returncode != 0 or p_bl.returncode != 0:
            print(f"WARNING: one or both runs exited with errors "
                  f"(codes: {p_ft.returncode}, {p_bl.returncode})")
        else:
            print("\nBoth runs complete. Plotting …")
            plot()

    else:
        # Single run
        losses = run_training(args.mode, args.device)
        # Auto-plot if the partner result already exists
        partner = "baseline" if args.mode == "finetune" else "finetune"
        if (HERE / f"results_{partner}.json").exists():
            print("\nPartner results found — plotting …")
            plot()
        else:
            print(f"\nRun --mode {partner} to get the other curve, then --mode plot.")
