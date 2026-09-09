# ============================================================
# Figure 8: Sentiment Negation Circuit
# Reproduces panels A, B, C from Pearce et al. (2025) Fig. 8
#
# Requires (both public on HuggingFace):
#   tdooms/ts-medium       — 6-layer bilinear transformer
#   tdooms/ts-medium-scope — SAEs on resid-mid and mlp-out
# Panel C tokenizes roneneldan/TinyStories with the model's own
# tokenizer (the authors' tokenized dataset is not public).
# ============================================================

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_dataset
from torch.nn.functional import cosine_similarity as cos_sim
from tqdm import tqdm

os.chdir(Path(__file__).resolve().parents[3])
HERE = Path(__file__).parent

from language import Transformer
from language.transformer import Attention, Rotary
from sae.tracer import Tracer

# transformers 5.x compatibility: add missing all_tied_weights_keys shim.
if not hasattr(Transformer, "all_tied_weights_keys"):
    Transformer.all_tied_weights_keys = property(lambda self: {})


def fix_model_buffers(model):
    """Move all plain-tensor module attributes to the right device.
    transformers 5.x with device_map leaves non-registered tensors on meta
    (e.g. Attention.mask, Rotary caches). We recreate them on CUDA here.
    """
    for m in model.modules():
        if isinstance(m, Attention):
            m.mask = torch.tril(
                torch.ones(m.config.n_ctx, m.config.n_ctx)
            )[None, None].to(next(m.parameters()).device)
        if isinstance(m, Rotary):
            m.seq_len_cached = None   # force recompute on first forward


def get_resid_mid(model, input_ids, layer):
    """Capture residual stream at resid-mid (before MLP) via a forward hook."""
    captured = {}
    handle = model.transformer.h[layer].n2.register_forward_pre_hook(
        lambda _, inp: captured.__setitem__("x", inp[0].detach())
    )
    with torch.no_grad():
        model(input_ids)
    handle.remove()
    return captured["x"]


torch.set_grad_enabled(False)

# ============================================================
# Configuration
# ============================================================
LAYER    = 4
NOT_GOOD = 1882   # "not-good" output SAE feature (paper Section 5.1)

# The input features contributing to the circuit are listed with the
# paper's human labels in appendix_o/appendix_o.py (FEATURES dict);
# here only their ids, plot symbols and colours are needed
# (PAPER_ORDER / PAPER_SYMBOL / FID_MARKER below).

# ============================================================
# Load model and SAEs
# ============================================================
print("Loading model …")
model = Transformer.from_pretrained("tdooms/ts-medium", device="cuda")
fix_model_buffers(model)

print("Loading SAEs …")
tracer = Tracer(model, layer=LAYER, inp=dict(name="resid-mid"))

# ============================================================
# Panel A — Interaction submatrix heatmap
# ============================================================
print("Computing Panel A …")
# Q projected into SAE input latent space: (d_features_in, d_features_in)
q = tracer.q(NOT_GOOD, project=True)

# Top-15 cross-interactions (off-diagonal elements, ranked by |value|)
cross = torch.tril(2 * q, diagonal=-1)
_, flat_idxs = cross.abs().flatten().topk(15)
i1, i2 = torch.unravel_index(flat_idxs, q.shape)
feature_set = torch.cat([i1, i2]).unique().sort().values.tolist()

# Exact axis ordering and symbol assignment from the paper
PAPER_ORDER = [326, 1376, 1636, 123, 990, 1929, 461, 947, 882, 240, 766, 1395, 1604]
PAPER_SYMBOL = {
    326:  ("■", "#636EFA"), 1376: ("■", "#636EFA"), 1636: ("■", "#636EFA"),
    123:  ("■", "#636EFA"), 990:  ("■", "#636EFA"), 1929: ("■", "#636EFA"),
    461:  ("■", "#636EFA"), 947:  ("■", "#636EFA"),
    882:  ("▼", "#FFA15A"),
    240:  ("▲", "#00CC96"), 766:  ("▲", "#00CC96"),
    1395: ("▲", "#00CC96"), 1604: ("▲", "#00CC96"),
}

# Use paper order, keeping only features that appear in the computed set
feature_set = [f for f in PAPER_ORDER if f in set(feature_set)]
# Append any extra computed features not in PAPER_ORDER
extra = [f for f in torch.cat([i1, i2]).unique().sort().values.tolist()
         if f not in PAPER_ORDER]
feature_set += extra

def tick_label(fid):
    if fid in PAPER_SYMBOL:
        sym, hex_col = PAPER_SYMBOL[fid]
        return f"<span style='color:{hex_col}'>{sym}</span> {fid}"
    return str(fid)

tick_labels = [tick_label(f) for f in feature_set]
positions   = list(range(len(feature_set)))

submatrix = q[feature_set][:, feature_set].cpu().float()

fig = px.imshow(
    submatrix,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
)
fig.update_xaxes(
    side="top",
    tickmode="array",
    tickvals=positions,
    ticktext=tick_labels,
    tickfont_size=10,
)
fig.update_yaxes(
    tickmode="array",
    tickvals=positions,
    ticktext=tick_labels,
    tickfont_size=10,
)
fig.update_layout(
    width=450, height=450,
    margin=dict(l=0, r=60, b=0, t=50),
    template="plotly_white",
)
fig.write_image(HERE / "fig_08a.png", scale=2)
print("  → fig_08a.png")

# ============================================================
# Panel B — Feature projections onto top eigenvectors
# ============================================================
print("Computing Panel B …")

# ── Eigenvectors in MODEL HIDDEN SPACE (project=False) ──────
# This gives the ~0.6 range seen in the paper; SAE-latent eigenvectors
# give ~0.05 values because the 2048-dim space spreads the norm thin.
q_model = tracer.q(NOT_GOOD, project=False)
q_model = 0.5 * (q_model + q_model.T)
vals_m, vecs_m = torch.linalg.eigh(q_model.cpu())   # ascending
v_pos = vecs_m[:, -1]   # most positive eigenvector  (d_model,)
v_neg = vecs_m[:,  0]   # most negative eigenvector

# Canonicalize signs so the not-good output feature (1882) lands in
# the bottom-right quadrant (positive x, negative y), matching the paper.
out_enc = tracer.out.w_enc.weight.cpu()          # (d_features_out, d_model)
not_good_dir = out_enc[NOT_GOOD, :]
if cos_sim(not_good_dir.unsqueeze(0), v_pos.unsqueeze(0)) < 0:
    v_pos = -v_pos
if cos_sim(not_good_dir.unsqueeze(0), v_neg.unsqueeze(0)) > 0:
    v_neg = -v_neg

def cs(a, b):
    """Cosine similarity of two 1-D tensors."""
    return float(cos_sim(a.unsqueeze(0), b.unsqueeze(0)))

inp_dec  = tracer.inp.w_dec.weight.cpu()          # (d_model, d_features_in)

# ── Cluster marker mapping (same symbols as Panel A) ────────
FID_MARKER = {
    **{f: ("square",   "#636EFA", 12) for f in [326,1376,1636,123,990,1929,461,947]},
    882:  ("triangle-down", "#FFA15A", 12),
    **{f: ("triangle-up",  "#00CC96", 12) for f in [240,766,1395,1604]},
}

fig_b = go.Figure()

# Input SAE features: cos-sim of decoder direction with model-space eigenvectors
for fid, (sym, col, sz) in FID_MARKER.items():
    d = inp_dec[:, fid]
    fig_b.add_trace(go.Scatter(
        x=[cs(d, v_pos)], y=[cs(d, v_neg)],
        mode="markers",
        marker=dict(color=col, size=sz, symbol=sym),
        showlegend=False,
    ))

# Anchor: "bad"–"good" and "good"–"bad" token UNEMBEDDING differences
# (paper explicitly says "token unembeddings" = w_u rows, not embedding w_e)
bad_id  = model.tokenizer.encode(" bad",  add_special_tokens=False)[0]
good_id = model.tokenizer.encode(" good", add_special_tokens=False)[0]
d_bg = model.w_u[bad_id, :].cpu() - model.w_u[good_id, :].cpu()   # bad–good
d_gb = -d_bg                                                        # good–bad

for d, label, pos in [
    (d_bg, '"bad"–"good"<br>unembed',  "top left"),
    (d_gb, '"good"–"bad"<br>unembed', "bottom right"),
]:
    fig_b.add_trace(go.Scatter(
        x=[cs(d, v_pos)], y=[cs(d, v_neg)],
        mode="markers+text",
        marker=dict(color="black", size=10, symbol="circle"),
        text=[label], textposition=pos, textfont=dict(size=8),
        showlegend=False,
    ))

# Anchor: "[BOS] not" resid-mid activation at the "not" token
# "[BOS] not" tokenises as ['[BOS]', '[BOS]', 'not'] → "not" is at position 2
not_ids = model.tokenizer("[BOS] not", return_tensors="pt")["input_ids"].to("cuda")
act_not = get_resid_mid(model, not_ids, LAYER)[0, 2].cpu()
fig_b.add_trace(go.Scatter(
    x=[cs(act_not, v_pos)], y=[cs(act_not, v_neg)],
    mode="markers+text",
    marker=dict(color="black", size=10, symbol="circle"),
    text=['"not"<br>input'], textposition="top right", textfont=dict(size=8),
    showlegend=False,
))

# Output SAE features: not-good (1882) and not-bad (1179)
NOT_BAD = 1179
for fid, label, tpos in [
    (NOT_GOOD, "not-good<br>feature", "bottom right"),
    (NOT_BAD,  "not-bad<br>feature",  "top left"),
]:
    d = out_enc[fid, :]
    fig_b.add_trace(go.Scatter(
        x=[cs(d, v_pos)], y=[cs(d, v_neg)],
        mode="markers+text",
        marker=dict(color="black", size=10, symbol="circle"),
        text=[label], textposition=tpos, textfont=dict(size=8),
        showlegend=False,
    ))

AX = 0.7
fig_b.update_layout(
    xaxis=dict(
        title="Top positive eigenvector",
        range=[-AX, AX], zeroline=True, zerolinewidth=1, zerolinecolor="gray",
        tickvals=[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
    ),
    yaxis=dict(
        title="Top negative eigenvector",
        range=[-AX, AX], zeroline=True, zerolinewidth=1, zerolinecolor="gray",
        tickvals=[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
    ),
    width=520, height=520,
    template="plotly_white",
    margin=dict(l=60, r=20, b=60, t=20),
)
fig_b.write_image(HERE / "fig_08b.png", scale=2)
print("  → fig_08b.png")

# ============================================================
# Panel C — Activation vs approximation scatter (feature 1882)
# ============================================================
print("Computing Panel C …")

# q_model, vals_m, vecs_m already computed in Panel B above.
# Top-2 eigenvectors by absolute eigenvalue
top2_idx  = vals_m.abs().topk(2).indices
vals_top2 = vals_m[top2_idx]       # (2,)
vecs_top2 = vecs_m[:, top2_idx]    # (d_model, 2)

# Load and tokenise 256 stories from TinyStories (tdooms tokenised dataset
# is private, so we tokenise roneneldan/TinyStories with the model tokenizer).
dataset = load_dataset("roneneldan/TinyStories", split="train[:8192]")
n_ctx   = model.config.n_ctx  # 256

# Match the paper's training format: prepend "[BOS]", lowercase, truncate.
# The model was trained on TinyStories with a custom lowercase-only BPE
# tokenizer and "[BOS]" as the sentence-start token.
texts = ["[BOS] " + row["text"].lower() for row in dataset]
tokenized = model.tokenizer(
    texts,
    truncation=True,
    padding="max_length",
    max_length=n_ctx,
    return_tensors="pt",
    add_special_tokens=False,   # [BOS] already in the text
)
input_ids = tokenized["input_ids"]

# Process in batches of 32 to stay within GPU memory
BATCH = 32
mlp   = model.transformer.h[LAYER].mlp
ys, y_hats = [], []

for start in tqdm(range(0, len(input_ids), BATCH), desc="  Panel C"):
    batch    = input_ids[start:start + BATCH].to("cuda")
    acts     = get_resid_mid(model, batch, LAYER)[:, 1:]  # skip BOS

    for i in range(len(acts)):
        x      = acts[i]                            # (seq_len, d_model)

        # Actual SAE output activation for NOT_GOOD feature
        y_all  = tracer.out.encode(mlp(x))          # (seq_len, d_features_out)
        y_feat = y_all[:, NOT_GOOD].cpu()           # (seq_len,)

        # Quadratic approximation with top-2 eigenvectors
        proj   = x.cpu() @ vecs_top2               # (seq_len, 2)
        approx = (proj.pow(2) * vals_top2).sum(-1) # (seq_len,)

        # Clip approximation at 0 (matches paper: negative-eigenvalue term
        # occasionally dominates, but paper shows non-negative y-axis only)
        approx = approx.clamp(min=0)

        mask = y_feat > 0
        if mask.sum() > 0:
            ys.append(y_feat[mask])
            y_hats.append(approx[mask])

    torch.cuda.empty_cache()

if not ys:
    print("  WARNING: feature 1882 never activated — try more stories")
    raise RuntimeError("No active tokens found for feature 1882")

y_actual = torch.cat(ys).numpy()
y_approx = torch.cat(y_hats).numpy()

corr = float(torch.corrcoef(torch.stack([
    torch.tensor(y_actual), torch.tensor(y_approx)
]))[0, 1])
print(f"  Correlation (active only): {corr:.3f}  [paper reports ~0.66]")

import json
with open(HERE / "fig_08_results.json", "w") as f:
    json.dump({
        "panel_c_correlation_active_only": corr,
        "n_active_tokens": len(y_actual),
        "top_positive_eigenvalue": float(vals_m[-1]),
        "top_negative_eigenvalue": float(vals_m[0]),
        "paper": {"correlation": 0.66,
                  "top_positive_eigenvalue": 0.62,
                  "top_negative_eigenvalue": -0.66},
    }, f, indent=2)
print(f"  Saved {HERE / 'fig_08_results.json'}")

AX_MAX = 40  # match paper's axis range

fig_c, ax_c = plt.subplots(figsize=(5, 5), dpi=150)
ax_c.scatter(y_actual, y_approx, s=5, alpha=0.3, color="#1f77b4", linewidths=0)
ax_c.plot([0, AX_MAX], [0, AX_MAX], "--", color="darkorange", linewidth=1.5)
ax_c.set_xlim(0, AX_MAX)
ax_c.set_ylim(0, AX_MAX)
ax_c.set_xticks(range(0, AX_MAX + 1, 10))
ax_c.set_yticks(range(0, AX_MAX + 1, 5))
ax_c.set_xlabel("Output feature activation", fontsize=11)
ax_c.set_ylabel("Eigenvector-based activation", fontsize=11)
ax_c.grid(True, alpha=0.3)
fig_c.tight_layout()
fig_c.savefig(HERE / "fig_08c.png", dpi=150, bbox_inches="tight")
plt.close(fig_c)
print("  → fig_08c.png")
print("Done.")
