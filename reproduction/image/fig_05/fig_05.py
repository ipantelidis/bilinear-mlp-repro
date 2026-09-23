# ============================================================
# Figure 5 — Eigenvector consistency and truncation across sizes
# Reproduces Figure 5 from Pearce et al. (2025).
#
# Trains 5 seeds × 6 hidden sizes (30 … 1000) of the MNIST bilinear
# model, then plots (A) cross-seed cosine similarity of the top
# eigenvectors by rank and (B) classification error when the model
# is truncated to its top-k eigenvectors per digit.
# Produces fig_05a.png, fig_05b.png and fig_05_results.json;
# trained features are cached in features_sim.pt / results_sim.pt.
# ============================================================

import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import torch
from einops import *
from image import MNIST, Model
from kornia.augmentation import RandomGaussianNoise
from scipy import stats
from torch import nn
from torch.nn.functional import cosine_similarity

# Run from repo root so ./data always maps to <repo>/data
os.chdir(Path(__file__).resolve().parents[3])
HERE = Path(__file__).parent

pio.templates.default = "plotly_white"

sizes = [30, 50, 100, 300, 500, 1000]

# =====================================
# Statistical helper
# =====================================

def conf_interval(sims, conf=0.95):
    """Compute t-based confidence interval across runs."""
    mean = torch.mean(sims, dim=-2)
    sem = torch.std(sims, dim=-2) / torch.sqrt(torch.tensor(sims.shape[-2]))
    df = sims.shape[-2] - 1
    t_value = stats.t.ppf((1 + conf) / 2, df)
    return mean - t_value * sem, mean + t_value * sem


# =====================================
# Single training loop — shared across both figures
# =====================================

N_VECS = 15

features = torch.empty(6, 5, 10, N_VECS, 784)
results  = torch.empty(6, 5, 31)
ground   = torch.empty(6, 5)

def eval_truncated(data, vals, vecs, k):
    top_k_vals, top_k_indices = vals.abs().topk(k, dim=-1)
    top_k_vals = torch.gather(vals, -1, top_k_indices)
    expanded_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, vecs.size(-1))
    top_k_vecs = torch.gather(vecs, 1, expanded_indices)
    p = einsum(
        data.flatten(start_dim=1),
        top_k_vecs,
        "batch inp, out hid inp -> batch hid out",
    ).pow(2)
    return einsum(p, top_k_vals, "batch hid out, out hid -> batch out")

if (HERE / "features_sim.pt").exists() and (HERE / "results_sim.pt").exists():
    print("Loading cached features/results — delete the .pt files to retrain.")
    features = torch.load(HERE / "features_sim.pt")
    cached = torch.load(HERE / "results_sim.pt")
    results, ground = cached["results"], cached["ground"]
else:
    for d, i in product(range(6), range(5)):
        mnist = Model.from_config(
            epochs=100,
            wd=1.0,
            d_hidden=sizes[d],
            n_layer=1,
            residual=False,
            seed=i,
        ).cuda()

        transform = nn.Sequential(
            RandomGaussianNoise(mean=0, std=0.4, p=1),
        )

        torch.set_grad_enabled(True)
        train, test = MNIST(train=True), MNIST(train=False)
        mnist.fit(train, test, transform)
        torch.set_grad_enabled(False)

        vals, vecs = mnist.decompose()

        # Store top N_VECS eigenvectors for the similarity figure.
        # eigh returns ascending eigenvalues, so the most positive
        # eigenvectors are the LAST columns; flip so rank 0 = largest λ.
        features[d, i] = vecs[:, -N_VECS:, :].flip(1)

        # Compute truncation results for this model
        for k in range(31):
            logits = eval_truncated(test.x, vals, vecs, k)
            results[d, i, k] = (logits.argmax(dim=1) == test.y).float().mean().cpu()

        ground[d, i] = (mnist(test.x).argmax(dim=1) == test.y).float().mean().item()

    torch.save(features, HERE / "features_sim.pt")
    torch.save(dict(results=results, ground=ground), HERE / "results_sim.pt")

# =====================================
# FIGURE 1 — Similarity across eigenvectors
# =====================================

# features is already ordered rank 0 = top positive eigenvector
sims = cosine_similarity(
    features[..., None, :, :, :, :],
    features[..., :, None, :, :, :],
    dim=-1,
)

# offset=1 excludes the diagonal: comparing a run with itself always
# gives cosine 1.0 and would inflate the similarity curves.
idxs = torch.triu_indices(5, 5, offset=1)
sims = rearrange(
    sims[:, idxs[0], idxs[1]].abs(),
    "... batch cls comp -> ... (batch cls) comp",
)

fig = go.Figure()

viridis = plt.colormaps["viridis"]
colors = [
    f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    for r, g, b in [viridis(x)[:3] for x in [0., 0.25, 0.5, 0.75, 0.9, 1.]]
]

for i in range(6):
    mean = torch.mean(sims[i], dim=-2)
    low, up = conf_interval(sims[i], conf=0.9)
    x = torch.arange(len(mean))

    fig.add_trace(go.Scatter(
        x=x,
        y=mean,
        mode="lines",
        name=f"{sizes[i]}",
        line=dict(color=colors[i]),
    ))

    fig.add_trace(go.Scatter(
        x=torch.cat([x, x.flip(0)]),
        y=torch.cat([up, low.flip(0)]),
        fill="toself",
        fillcolor=colors[i].replace("rgb", "rgba").replace(")", ", 0.2)"),
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))

fig.update_layout(
    title="Similarity Across Eigenvectors",
    title_x=0.5,
    width=600,
    height=400,
    legend_title_text="Model Size",
)
# Presentation only: pad the x-range so curves/bands do not run into the
# plot border, and show the full y-range so no curve is clipped below 0.4
# (rank-15 similarity for size 30 is ~0.13). scale=4 for print resolution.
fig.update_xaxes(title="Eigenvector rank", range=[-0.3, 14.3])
fig.update_yaxes(title="Cosine similarity", range=[0, 1.01])

fig.write_image(HERE / "fig_05a.png", scale=4)

# =====================================
# FIGURE 2 — Truncation error across sizes
# =====================================

error = 1 - results

fig = go.Figure()

for i in range(6):
    mean = torch.mean(error[i], dim=0)
    low, up = conf_interval(error[i], conf=0.9)
    x = torch.arange(len(mean))

    fig.add_trace(go.Scatter(
        x=x,
        y=mean,
        mode="lines",
        name=f"{sizes[i]}",
        line=dict(color=colors[i]),
    ))

    fig.add_trace(go.Scatter(
        x=torch.cat([x, x.flip(0)]),
        y=torch.cat([up, low.flip(0)]),
        fill="toself",
        fillcolor=colors[i].replace("rgb", "rgba").replace(")", ", 0.2)"),
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))

small = lambda x: f"<span style='font-size: 9px;'>{x}</span>"

fig.update_layout(
    title="Truncation Across Sizes",
    title_x=0.5,
    width=600,
    height=400,
    legend_title_text="Model Size",
)
fig.update_xaxes(title="Eigenvector rank (per digit)")
fig.update_yaxes(
    title="Classification error",
    tickvals=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1],
    ticktext=["1%", small("2%"), small("5%"), "10%", small("20%"), small("50%"), "100%"],
    range=[-2.02, 0.02],
    type="log",
)

fig.write_image(HERE / "fig_05b.png")

# =====================================
# Serialize quantitative results
# =====================================

import json

summary = {
    "sizes": sizes,
    "similarity_mean_by_rank": {
        str(sizes[i]): torch.mean(sims[i], dim=-2).tolist() for i in range(6)
    },
    "truncation_error_mean_by_rank": {
        str(sizes[i]): torch.mean(error[i], dim=0).tolist() for i in range(6)
    },
    "full_model_accuracy_mean": {
        str(sizes[i]): ground[i].mean().item() for i in range(6)
    },
}

with open(HERE / "fig_05_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Saved {HERE / 'fig_05_results.json'}")
