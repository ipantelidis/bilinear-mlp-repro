# ============================================================
# Figure 2 — Top eigenvectors for MNIST and Fashion-MNIST
# Reproduces Figure 2B from Pearce et al. (2025).
#
# Trains one bilinear model per dataset (paper setup: d_hidden 512,
# wd 1.0, input noise std 0.5, 100 epochs), decomposes each class's
# interaction matrix, and plots the top positive eigenvector for
# classes 1–5. Produces fig_02.png.
# ============================================================

import os
from pathlib import Path

import plotly.express as px
import plotly.io as pio
import torch
from image import FMNIST, MNIST, Model
from kornia.augmentation import RandomGaussianNoise
from torch import nn

# Run from repo root so ./data always maps to <repo>/data
os.chdir(Path(__file__).resolve().parents[3])
HERE = Path(__file__).parent

pio.templates.default = "plotly_white"

# Shared color settings (paper style)
color = dict(
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
)

# =====================================
# Model initialization
# =====================================

mnist = Model.from_config(
    epochs=100,
    wd=1.0,
    d_hidden=512,
    n_layer=1,
    residual=False,
    seed=420,
).cuda()

fmnist = Model.from_config(
    epochs=100,
    wd=1.0,
    d_hidden=512,
    n_layer=1,
    residual=False,
    seed=420,
).cuda()

# =====================================
# Data augmentation (regularization)
# =====================================

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.5, p=1),
)

# =====================================
# Train MNIST model
# =====================================

torch.set_grad_enabled(True)

train, test = MNIST(train=True), MNIST(train=False)
mnist.fit(train, test, transform)

# =====================================
# Train Fashion-MNIST model
# =====================================

train, test = FMNIST(train=True), FMNIST(train=False)
fmnist.fit(train, test, transform)

torch.set_grad_enabled(False)

# =====================================
# Decompose models into eigenvectors
# =====================================

m_vals, m_vecs = mnist.decompose()
f_vals, f_vecs = fmnist.decompose()

# =====================================
# Select and normalize top eigenvectors
# =====================================

idxs = slice(1, 6)

vecs = torch.cat([
    m_vecs[idxs, -1],
    f_vecs[idxs, -1],
])

# Normalize each eigenvector independently
vecs /= vecs.abs().max(1, keepdim=True).values

# =====================================
# Plot eigenvectors (Figure 2)
# =====================================

fig = px.imshow(
    vecs.view(-1, 28, 28).cpu(),
    facet_col=0,
    facet_col_wrap=5,
    height=330,
    width=1000,
    facet_row_spacing=0.1,
    **color,
)

fig.update_layout(
    coloraxis_showscale=False,
    margin=dict(l=0, r=0, b=0, t=20),
)

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

# =====================================
# Add class labels
# =====================================

m_labels = [f"{i}" for i in range(10)]
f_labels = [
    "t-shirt/top", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot",
]

# Plotly numbers facet annotations bottom-row-first, so the label list
# must be Fashion-MNIST (bottom row) before MNIST (top row) even though
# the image tensor is concatenated MNIST-first.
labels = f_labels[idxs] + m_labels[idxs]

for i, annotation in enumerate(fig.layout.annotations):
    annotation.update(
        text=f"<b>{labels[i]}</b>",
        y=annotation["y"] + 0.005,
    )

# =====================================
# Save figure
# =====================================

fig.write_image(
    HERE / "fig_02.png",
    scale=4,
)

