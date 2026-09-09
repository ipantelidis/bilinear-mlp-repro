# ============================================================
# Appendix H — Correlation: Analyzing the Impact of Training Time
# Reproduces Figure 24 from Pearce et al. (2025)
#
# Uses the 5 publicly available x16 SAEs trained for increasing
# durations on fw-medium layer 12:
#   tdooms/fw-medium-scope  →  12-mlp-out-x16-k30-v{0..4}
# v0 = shortest training; v4 = longest (each 2× the previous).
# ============================================================

import gc
import json
import os
from pathlib import Path
from transformers import AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from tqdm import tqdm

os.chdir(Path(__file__).resolve().parents[3])
HERE = Path(__file__).parent

from language.transformer import Transformer, Attention, Rotary
from sae.sae import SAE
from safetensors.torch import load_model

if not hasattr(Transformer, "all_tied_weights_keys"):
    Transformer.all_tied_weights_keys = property(lambda self: {})


def fix_buffers(model):
    for m in model.modules():
        if isinstance(m, Attention):
            dev = next(m.parameters()).device
            m.mask = torch.tril(
                torch.ones(m.config.n_ctx, m.config.n_ctx)
            )[None, None].to(dev)
        if isinstance(m, Rotary):
            m.seq_len_cached = None


def load_tagged_sae(repo_id, layer, name, expansion, k, tag, device="cuda"):
    """Load a tagged SAE (e.g. 12-mlp-out-x16-k30-v0) from HuggingFace."""
    sae_name = f"{layer}-{name}-x{expansion}-k{k}-{tag}"
    cfg_path   = hf_hub_download(repo_id=repo_id, filename=f"{sae_name}/config.json")
    mdl_path   = hf_hub_download(repo_id=repo_id, filename=f"{sae_name}/model.safetensors")
    sae = SAE.from_config(**json.load(open(cfg_path)))
    load_model(sae, mdl_path)
    return sae.to(device)


def get_resid_mid(model, input_ids, layer):
    cap = {}
    h = model.transformer.h[layer].n2.register_forward_pre_hook(
        lambda _, inp: cap.__setitem__("x", inp[0].detach())
    )
    with torch.no_grad():
        model(input_ids)
    h.remove()
    return cap["x"]


def truncated_eigh(q, k):
    vals, vecs = torch.linalg.eigh(q)
    idxs = vals.abs().topk(k).indices
    return vals[idxs], vecs[:, idxs]


torch.set_grad_enabled(False)

LAYER      = 12
SAE_REPO   = "tdooms/fw-medium-scope"
EXPANSION  = 16
K_SAE      = 30
K_EIGH     = 2       # top-2 eigenvectors (as in paper's Panel C comparison)
N_SEQ       = 1024   # sequences — gives ~100K active token-feature pairs
BATCH_SEQ   = 1
FEAT_SAMPLE = 2048   # sample 2048/16384 features (paper uses all, but this is ~12%)

VERSIONS   = ["v0", "v1", "v2", "v3", "v4"]   # short → long training

# ── Load model ───────────────────────────────────────────────
print("Loading fw-medium …")
model = Transformer.from_pretrained("tdooms/fw-medium", device="cuda")
fix_buffers(model)

# ── Tokenise FineWeb data ────────────────────────────────────
print(f"Tokenising {N_SEQ} sequences …")
tok = AutoTokenizer.from_pretrained(
    "mistral-community/Mixtral-8x22B-v0.1",
    pad_token="</s>", padding_side="right"
)
raw   = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                     split=f"train[:{N_SEQ}]")
ids   = tok([r["text"] for r in raw], truncation=True,
            padding="max_length", max_length=512, return_tensors="pt")["input_ids"]

# ── Per-version computation ───────────────────────────────────
all_corrs = {}

for ver in VERSIONS:
    print(f"\n{'='*50}\nVersion {ver} …")
    cache = HERE / f"cache_{ver}.safetensors"

    sae     = load_tagged_sae(SAE_REPO, LAYER, "mlp-out", EXPANSION, K_SAE, ver)
    d_feats = sae.d_features   # 16 × 1024 = 16384
    d_model = model.config.d_model
    out_enc = sae.w_enc.weight.cpu()   # (d_feats, d_model)

    # Sample features
    feat_ids = torch.randperm(d_feats)[:FEAT_SAMPLE].tolist()

    # ── Eigendecomposition for sampled features ───────────────
    if not cache.exists():
        print(f"  Computing eigendecomposition ({FEAT_SAMPLE} features) …")
        w_l = model.w_l[LAYER].cpu()
        w_r = model.w_r[LAYER].cpu()
        w_p = model.w_p[LAYER].cpu()
        sv  = torch.empty(FEAT_SAMPLE, K_EIGH)
        sv_vecs = torch.empty(FEAT_SAMPLE, d_model, K_EIGH)
        for fi, f in enumerate(tqdm(feat_ids, desc="  eigh")):
            u   = out_enc[f]
            v   = w_p.T @ u
            wlv = w_l * v[:, None]
            q   = wlv.T @ w_r
            q   = 0.5 * (q + q.T)
            ev, ec = truncated_eigh(q, K_EIGH)
            sv[fi]      = ev
            sv_vecs[fi] = ec
        save_file({"vals": sv, "vecs": sv_vecs,
                   "feat_ids": torch.tensor(feat_ids)}, cache)
    else:
        data = load_file(cache, device="cpu")
        sv, sv_vecs = data["vals"], data["vecs"]
        feat_ids = data["feat_ids"].tolist()

    # ── Inference ─────────────────────────────────────────────
    mlp = model.transformer.h[LAYER].mlp
    raw_data = {f: {"y": [], "acts": []} for f in feat_ids}

    for start in tqdm(range(0, len(ids), BATCH_SEQ), desc="  data"):
        batch = ids[start:start + BATCH_SEQ].cuda()
        acts  = get_resid_mid(model, batch, LAYER)[:, 1:]
        for i in range(len(acts)):
            x     = acts[i]
            y_all = sae.encode(mlp(x))
            for fi, f in enumerate(feat_ids):
                yf   = y_all[:, f].cpu()
                mask = yf > 0
                if mask.sum() < 2:
                    continue
                raw_data[f]["y"].append(yf[mask])
                raw_data[f]["acts"].append(x[mask].cpu())
        torch.cuda.empty_cache()

    # ── Compute per-feature correlation (top-2 eigenvecs) ─────
    corrs = []
    for fi, f in enumerate(feat_ids):
        if not raw_data[f]["y"]:
            continue
        ya   = torch.cat(raw_data[f]["y"])
        xact = torch.cat(raw_data[f]["acts"])
        if len(ya) < 10:
            continue
        vk = sv_vecs[fi]          # (d_model, K_EIGH)
        lk = sv[fi]               # (K_EIGH,)
        ap = (xact @ vk).pow(2) * lk
        ap = ap.sum(-1)
        c  = float(torch.corrcoef(torch.stack([ya, ap]))[0, 1])
        if not np.isnan(c):
            corrs.append(c)

    all_corrs[ver] = corrs
    mean_c = np.mean(corrs) if corrs else float("nan")
    print(f"  {ver}: n={len(corrs)}, mean corr={mean_c:.3f}")

    del sv, sv_vecs, raw_data, sae
    gc.collect(); torch.cuda.empty_cache()

# ── Plot Figure 24 ────────────────────────────────────────────
print("\nPlotting Figure 24 …")

# Paper: darker = shorter training, lighter/brighter = longer training
cmap   = plt.cm.Blues
colors = [cmap(0.9 - 0.15 * i) for i in range(5)]    # dark → light

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
bins    = np.linspace(-0.5, 1.0, 80)

for i, (ver, corrs) in enumerate(all_corrs.items()):
    if not corrs:
        continue
    ax.hist(corrs, bins=bins, alpha=0.7, color=colors[i],
            label=f"{ver} (×{2**i})", density=False)

ax.set_xlabel("Correlation (active only)", fontsize=11)
ax.set_ylabel("Count",                    fontsize=11)
ax.set_title("Feature approximation correlations\n"
             "(fw-medium layer 12, x16 SAEs, top-2 eigenvectors)", fontsize=10)
ax.legend(title="Training steps", fontsize=9, title_fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "fig_24.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → fig_24.png")

# Table 5 comparison — printed and serialized
print("\nTable 5 comparison:")
print(f"{'Version':<8} {'Mean corr':>10}  {'Paper mean':>10}")
paper = [0.17, 0.28, 0.42, 0.52, 0.59]
table5 = {"paper_mean": dict(zip(VERSIONS, paper))}
for i, (ver, corrs) in enumerate(all_corrs.items()):
    mc = float(np.mean(corrs)) if corrs else float("nan")
    table5.setdefault("ours_mean", {})[ver] = mc
    table5.setdefault("n_features", {})[ver] = len(corrs)
    print(f"{ver:<8} {mc:>10.3f}  {paper[i]:>10.2f}")

with open(HERE / "appendix_h_results.json", "w") as f:
    json.dump(table5, f, indent=2)
print(f"Saved {HERE / 'appendix_h_results.json'}")

print("Done.")
