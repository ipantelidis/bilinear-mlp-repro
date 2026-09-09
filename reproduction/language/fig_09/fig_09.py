# ============================================================
# Figure 9: Activation correlations with low-rank approximations
# Reproduces panels A, B, C from Pearce et al. (2025) Fig. 9
#
# Models and SAEs (all public HuggingFace). The paper's names map to
# HF repos as follows — note the paper's "ts-tiny" (its 6-layer
# TinyStories model) is published as tdooms/ts-medium:
#   ts-tiny  → tdooms/ts-medium       @ layer 4,  SAE: ts-medium-scope (x4)
#   fw-small → tdooms/fw-small        @ layer 8,  SAE: fw-small-scope  (x4)
#   fw-medium→ tdooms/fw-medium       @ layer 10, SAE: fw-medium-scope (x8)
# Layers are chosen at ≈2/3 depth, as in the paper.
#
# Caching strategy (allows all 3 panels in one run after first run):
#   cache_{name}.safetensors  — eigendecomposition (vals + vecs)
#   fwmedium_results.json     — corr_by_k, per_feat_corr, d_feats
#   scatter_fw-medium.pt      — per-feature (ya, yhat) for Panel C
#
# First run computes and saves all caches; subsequent runs load them.
# ============================================================

import gc
import json
import os
from pathlib import Path
from transformers import AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from datasets import load_dataset
from safetensors.torch import save_file, load_file
from tqdm import tqdm

os.chdir(Path(__file__).resolve().parents[3])
HERE = Path(__file__).parent

from language.transformer import Transformer, Attention, Rotary
from sae import SAE

if not hasattr(Transformer, "all_tied_weights_keys"):
    Transformer.all_tied_weights_keys = property(lambda self: {})


# ── helpers ──────────────────────────────────────────────────
def fix_buffers(model):
    for m in model.modules():
        if isinstance(m, Attention):
            dev = next(m.parameters()).device
            m.mask = torch.tril(
                torch.ones(m.config.n_ctx, m.config.n_ctx)
            )[None, None].to(dev)
        if isinstance(m, Rotary):
            m.seq_len_cached = None


def get_resid_mid(model, input_ids, layer):
    cap = {}
    h = model.transformer.h[layer].n2.register_forward_pre_hook(
        lambda _, inp: cap.__setitem__("x", inp[0].detach())
    )
    with torch.no_grad():
        model(input_ids)
    h.remove()
    return cap["x"]


def truncated_eigh(tensor, k):
    vals, vecs = torch.linalg.eigh(tensor)
    idxs = vals.abs().topk(k).indices
    return vals[idxs], vecs[:, idxs]


# ── model / SAE configs ──────────────────────────────────────
K_MAX     = 64
N_C_FEATS = 9
SAMPLE    = 512

MODELS = {
    "ts-tiny": dict(
        repo="tdooms/ts-medium", layer=4,
        sae_repo="tdooms/ts-medium-scope", sae_exp=4,
        n_ctx=256, n_seq=512, tok="ts", batch=8,
    ),
    "fw-small": dict(
        repo="tdooms/fw-small", layer=8,
        sae_repo="tdooms/fw-small-scope", sae_exp=4,
        n_ctx=512, n_seq=256, tok="mistral", batch=8,
    ),
    "fw-medium": dict(
        repo="tdooms/fw-medium", layer=10,
        sae_repo="tdooms/fw-medium-scope", sae_exp=8,
        n_ctx=512, n_seq=512, tok="mistral", batch=1,
    ),
}
# Match paper's colour convention: fw-medium=blue, fw-small=red, ts-tiny=green
COLORS = {"ts-tiny": "#2ca02c", "fw-small": "#d62728", "fw-medium": "#1f77b4"}

# ── tokeniser cache ──────────────────────────────────────────
_tokenisers = {}


def tokenise(cfg, n_seq):
    key = cfg["tok"]
    if key not in _tokenisers:
        if key == "ts":
            _tokenisers["ts"] = AutoTokenizer.from_pretrained(
                "tdooms/ts-tokenizer-4096",
                pad_token="[EOS]", padding_side="right"
            )
        else:
            _tokenisers["mistral"] = AutoTokenizer.from_pretrained(
                "mistral-community/Mixtral-8x22B-v0.1",
                pad_token="</s>", padding_side="right"
            )
    tok = _tokenisers[key]
    if key == "ts":
        raw   = load_dataset("roneneldan/TinyStories", split=f"train[:{n_seq}]")
        texts = ["[BOS] " + r["text"].lower() for r in raw]
        return tok(texts, truncation=True, padding="max_length",
                   max_length=cfg["n_ctx"], return_tensors="pt",
                   add_special_tokens=False)["input_ids"]
    else:
        raw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                           split=f"train[:{n_seq}]")
        return tok([r["text"] for r in raw], truncation=True,
                   padding="max_length", max_length=cfg["n_ctx"],
                   return_tensors="pt")["input_ids"]


# ── per-model computation ─────────────────────────────────────
torch.set_grad_enabled(False)
all_results = {}

FW_JSON    = HERE / "fwmedium_results.json"
FW_SCATTER = HERE / "scatter_fw-medium.pt"

for name, cfg in MODELS.items():
    if name == "fw-medium" and FW_JSON.exists():
        print(f"\n{'='*50}\nLoading {name} from cache …")
        fw      = json.load(open(FW_JSON))
        d_feats = fw["d_feats"]
        feat_y  = [[] for _ in range(d_feats)]
        feat_yh = [[] for _ in range(d_feats)]
        if FW_SCATTER.exists():
            print("  Loading scatter data for Panel C …")
            scatter = torch.load(FW_SCATTER, weights_only=True)
            for f_str, data in scatter.items():
                f = int(f_str)
                feat_y[f]  = [data["ya"]]
                feat_yh[f] = [data["yhat"]]
        else:
            print("  Warning: scatter_fw-medium.pt not found — Panel C will be skipped.")
        all_results[name] = {
            "corr_by_k":     fw["corr_by_k"],
            "per_feat_corr": fw["per_feat_corr"],
            "feat_y":        feat_y,
            "feat_y_hat":    feat_yh,
        }
        continue

    print(f"\n{'='*50}\nProcessing {name} …")
    model   = Transformer.from_pretrained(cfg["repo"], device="cuda")
    fix_buffers(model)
    out_sae = SAE.from_pretrained(
        cfg["sae_repo"], point=("mlp-out", cfg["layer"]),
        expansion=cfg["sae_exp"], k=30
    ).cuda()
    d_feats = out_sae.d_features
    d_model = model.config.d_model
    out_enc = out_sae.w_enc.weight.cpu()

    # ── eigendecomposition (cached to disk) ──────────────────
    cache = HERE / f"cache_{name}.safetensors"
    if not cache.exists():
        print(f"  Computing eigendecomposition ({d_feats} features) …")
        w_l = model.w_l[cfg["layer"]].cpu()
        w_r = model.w_r[cfg["layer"]].cpu()
        w_p = model.w_p[cfg["layer"]].cpu()
        all_vals = torch.empty(d_feats, K_MAX)
        all_vecs = torch.empty(d_feats, d_model, K_MAX)
        for f in tqdm(range(d_feats), desc="  eigh"):
            u   = out_enc[f]
            v   = w_p.T @ u
            wlv = w_l * v[:, None]
            q   = wlv.T @ w_r
            q   = 0.5 * (q + q.T)
            ev, ec = truncated_eigh(q, K_MAX)
            all_vals[f] = ev
            all_vecs[f] = ec
        save_file({"vals": all_vals, "vecs": all_vecs}, cache)
        print(f"  Saved {cache.name}")
    else:
        print(f"  Loading cached eigendecomposition …")
        data     = load_file(cache, device="cpu")
        all_vals = data["vals"]
        all_vecs = data["vecs"]

    # ── dataset ───────────────────────────────────────────────
    print(f"  Tokenising {cfg['n_seq']} sequences …")
    input_ids = tokenise(cfg, cfg["n_seq"])

    # ── inference ─────────────────────────────────────────────
    mlp      = model.transformer.h[cfg["layer"]].mlp
    feat_ids = torch.randperm(d_feats)[:SAMPLE].tolist()
    raw_data = {f: {"y": [], "acts": []} for f in feat_ids}
    feat_y   = [[] for _ in range(d_feats)]
    feat_yh  = [[] for _ in range(d_feats)]

    for start in tqdm(range(0, len(input_ids), cfg["batch"]), desc="  data"):
        batch = input_ids[start:start + cfg["batch"]].cuda()
        acts  = get_resid_mid(model, batch, cfg["layer"])[:, 1:]
        for i in range(len(acts)):
            x     = acts[i]
            y_all = out_sae.encode(mlp(x))
            for f in range(d_feats):
                yf   = y_all[:, f].cpu()
                mask = yf > 0
                if mask.sum() < 2:
                    continue
                feat_y[f].append(yf[mask])
                v2 = all_vecs[f, :, :2]
                l2 = all_vals[f, :2]
                ap = ((x[mask].cpu() @ v2).pow(2) * l2).sum(-1).clamp(min=0)
                feat_yh[f].append(ap)
                if f in raw_data:
                    raw_data[f]["y"].append(yf[mask])
                    raw_data[f]["acts"].append(x[mask].cpu())
        torch.cuda.empty_cache()

    # ── per-feature correlation (top-2 eigenvecs) ────────────
    pfc = []
    for f in range(d_feats):
        if not feat_y[f]:
            pfc.append(float("nan"))
            continue
        ya = torch.cat(feat_y[f])
        yh = torch.cat(feat_yh[f])
        pfc.append(
            float(torch.corrcoef(torch.stack([ya, yh]))[0, 1])
            if len(ya) >= 10 else float("nan")
        )

    # ── correlation vs k ─────────────────────────────────────
    cbk = []
    for k in tqdm(range(1, K_MAX + 1), desc="  corr-vs-k"):
        cs = []
        for f in feat_ids:
            if not raw_data[f]["y"]:
                continue
            ya   = torch.cat(raw_data[f]["y"])
            xact = torch.cat(raw_data[f]["acts"])
            if len(ya) < 2:
                continue
            vk = all_vecs[f, :, :k]
            lk = all_vals[f, :k]
            ap = (xact @ vk).pow(2) * lk
            ap = ap.sum(-1)   # no clamp — matches paper's corrcoef computation
            c  = float(torch.corrcoef(torch.stack([ya, ap]))[0, 1])
            if not np.isnan(c):
                cs.append(c)
        cbk.append(np.mean(cs) if cs else np.nan)

    all_results[name] = {
        "corr_by_k":     cbk,
        "per_feat_corr": pfc,
        "feat_y":        feat_y,
        "feat_y_hat":    feat_yh,
    }

    # ── save fw-medium results for reuse (all 3 panels) ─────
    if name == "fw-medium":
        json.dump(
            {"corr_by_k": cbk, "per_feat_corr": pfc, "d_feats": d_feats},
            open(FW_JSON, "w"),
        )
        print(f"  Saved fwmedium_results.json")
        scatter = {
            str(f): {"ya": torch.cat(feat_y[f]), "yhat": torch.cat(feat_yh[f])}
            for f in feat_ids if feat_y[f]
        }
        torch.save(scatter, FW_SCATTER)
        print(f"  Saved scatter_fw-medium.pt ({len(scatter)} features)")

    del mlp, out_sae, model, all_vals, all_vecs, raw_data
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# Panel A — Average correlation vs number of eigenvectors (plotly style)
# ============================================================
print("\nPlotting Panel A …")
pio.templates.default = "plotly_white"

fig_a = go.Figure()
for n in MODELS:
    c  = np.array(all_results[n]["corr_by_k"])
    xs = np.arange(1, len(c) + 1)
    fig_a.add_trace(go.Scatter(
        x=xs, y=c, name=n,
        line=dict(color=COLORS[n], width=2),
        mode="lines",
    ))

fig_a.update_layout(
    title=dict(text="Feature activation approximation", x=0.5, font_size=14),
    xaxis=dict(title="Top eigenvectors", range=[0, K_MAX]),
    yaxis=dict(title="Correlation",      range=[0.5, 1.0]),
    legend=dict(x=0.02, y=0.98, font_size=11),
    width=600, height=450,
    margin=dict(l=60, r=20, t=50, b=60),
)
fig_a.write_image(HERE / "fig_09a.png", scale=2)
print("  → fig_09a.png")

# ============================================================
# Panel B — Histogram of per-feature correlations (top-2 eigvecs)
# ============================================================
print("Plotting Panel B …")
fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
bins = np.linspace(-0.25, 1.0, 60)
for n in MODELS:
    c = np.array(all_results[n]["per_feat_corr"])
    c = c[~np.isnan(c)]
    ax.hist(c, bins=bins, alpha=0.5, color=COLORS[n], label=n)
ax.set_xlabel("Correlation (active only)", fontsize=11)
ax.set_ylabel("Count",                    fontsize=11)
ax.set_title("Approx. by top 2 eigenvectors", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "fig_09b.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → fig_09b.png")

# ============================================================
# Panel C — 9 scatter plots for fw-medium
# ============================================================
print("Plotting Panel C …")
fw_feats = all_results["fw-medium"]["feat_y"]
fw_hats  = all_results["fw-medium"]["feat_y_hat"]

if any(fw_feats):   # available if not loaded from JSON
    # Paper protocol: "a random set of nine output features". Sample with a
    # fixed seed among features with enough active tokens and a
    # non-degenerate approximation (zero-variance yhat gives r = nan).
    valid = [
        f for f in range(len(fw_feats))
        if fw_feats[f]
        and sum(len(x) for x in fw_feats[f]) >= 200
        and torch.cat(fw_hats[f]).std() > 0
    ]
    gen = torch.Generator().manual_seed(0)
    chosen = [valid[i] for i in torch.randperm(len(valid), generator=gen)[:N_C_FEATS]]

    fig, axes = plt.subplots(3, 3, figsize=(7, 7), dpi=150)
    for idx, f in enumerate(chosen):
        ax   = axes[idx // 3][idx % 3]
        ya   = torch.cat(fw_feats[f]).numpy()
        yhat = torch.cat(fw_hats[f]).numpy()
        corr = float(torch.corrcoef(
            torch.stack([torch.tensor(ya), torch.tensor(yhat)])
        )[0, 1])
        ax.scatter(ya, yhat, s=3, alpha=0.3, color="#1f77b4")
        lim = max(ya.max(), yhat.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="orange", lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(f"r={corr:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("fw-medium (top-2 eigenvectors)", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "fig_09c.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → fig_09c.png")
else:
    print("  Panel C skipped (fw-medium loaded from JSON — rerun without JSON to generate)")

print("Done.")
