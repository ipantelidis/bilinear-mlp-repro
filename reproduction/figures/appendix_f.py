# =====================================
# Imports and global setup
# =====================================

import os
from itertools import product

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from einops import *
from image import MNIST, Model
from kornia.augmentation import RandomGaussianNoise
from scipy import stats
from torch import einsum, nn
from torch.nn.functional import cosine_similarity

pio.templates.default = "plotly_white"

# Shared color settings
color = dict(
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
)

sizes = [30, 50, 100, 300, 500, 1000]

# =====================================
# Statistical helper
# =====================================

def conf_interval(sims, conf=0.95):
    mean = torch.mean(sims, dim=-2)
    sem = torch.std(sims, dim=-2) / torch.sqrt(torch.tensor(sims.shape[-2]))
    df = sims.shape[-2] - 1
    t_value = stats.t.ppf((1 + conf) / 2, df)
    return mean - t_value * sem, mean + t_value * sem

# =====================================
# Shared feature extraction + truncation evaluation
# =====================================

features = torch.empty(6, 5, 10, 20, 784)
results  = torch.empty(6, 5, 31)
ground   = torch.empty(6, 5)

for d, i in product(range(6), range(5)):
    model = Model.from_config(
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
    model.fit(train, test, transform)

    vals, vecs = model.decompose()
    features[d, i] = vecs[:, :20, :]

    def eval_truncated(data, vals, vecs, k):
        top_k_vals, top_k_idx = vals.abs().topk(k, dim=-1)
        top_k_vals = torch.gather(vals, -1, top_k_idx)

        expanded_idx = top_k_idx.unsqueeze(-1).expand(-1, -1, vecs.size(-1))
        top_k_vecs = torch.gather(vecs, 1, expanded_idx)

        p = einsum(
            data.flatten(start_dim=1),
            top_k_vecs,
            "batch inp, out hid inp -> batch hid out",
        ).pow(2)

        return einsum(p, top_k_vals, "batch hid out, out hid -> batch out")

    for k in range(31):
        logits = eval_truncated(test.x, vals, vecs, k)
        results[d, i, k] = (
            logits.argmax(dim=1) == test.y
        ).float().mean().cpu()

    ground[d, i] = (
        model(test.x).argmax(dim=1) == test.y
    ).float().mean().item()

torch.set_grad_enabled(False)

# Cache features
os.makedirs("..outputs/cache", exist_ok=True)
torch.save(features, "..outputs/cache/features_trunc.pt")
features = torch.load("..outputs/cache/features_trunc.pt")

# =====================================
# FIGURE 1 — Truncation across sizes (accuracy drop)
# =====================================

diff = ground[..., None] - results

fig = go.Figure()

viridis = plt.colormaps["viridis"]
colors = [
    f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    for r, g, b in [viridis(x)[:3] for x in [0., 0.25, 0.5, 0.75, 0.9, 1.]]
]

for i in range(6):
    mean = torch.mean(diff[i], dim=0)
    low, up = conf_interval(diff[i], conf=0.9)
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
    title="Truncation Across Sizes",
    title_x=0.5,
    width=600,
    height=400,
)
fig.update_xaxes(title="Eigenvector rank (per digit)")
fig.update_yaxes(
    title="Accuracy Drop",
    tickvals=[0.001, 0.01, 0.1, 1],
    ticktext=["0.1%", "1%", "10%", "100%"],
    range=[-3.02, 0.02],
    type="log",
)

fig.write_image(
    "..outputs/figures/appendix_f/acc_drop_trunc.png"
)

# =====================================
# FIGURE 2 — Similarity across eigenvectors
# =====================================

s = slice(-20, None)

sims = cosine_similarity(
    features[3, None, None, :, :, s, :],
    features[:, :, None, :, s, :],
    dim=-1,
)

idxs = torch.triu_indices(5, 5)
sims = rearrange(
    sims[:, idxs[0], idxs[1]].abs(),
    "... batch cls comp -> ... (batch cls) comp",
)

fig = go.Figure()

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
)
fig.update_xaxes(title="Eigenvector rank")
fig.update_yaxes(title="Cosine similarity", range=[0.0, 1.01])

fig.write_image(
    "..outputs/figures/appendix_f/sim_eigenvecs.png"
)

# =====================================
# FIGURE 3 — Heatmap
# =====================================

sims = cosine_similarity(
    features[:, None, None, :, :, s, :],
    features[:, :, None, :, s, :],
    dim=-1,
)

sims = rearrange(
    sims[:, :, idxs[0], idxs[1]].abs(),
    "... batch cls comp -> ... (batch cls) comp",
)

fig = px.imshow(
    sims[..., 0].mean(-1),
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.5,
    zmin=0,
    zmax=1,
)

fig.update_layout(width=463, height=400)
fig.update_xaxes(
    ticktext=[f"{sizes[i]}" for i in range(6)],
    tickvals=torch.arange(6),
)
fig.update_yaxes(
    ticktext=[f"{sizes[i]}" for i in range(6)],
    tickvals=torch.arange(6),
)

fig.write_image(
    "..outputs/figures/appendix_f/heatmap.png"
)
