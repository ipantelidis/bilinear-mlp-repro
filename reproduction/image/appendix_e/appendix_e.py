# ============================================================
# Appendix E — Sparsity: weight decay versus input noise
# Reproduces Figure 19 from Pearce et al. (2025).
#
# Trains a 6×6 grid of MNIST bilinear models over weight decay ×
# input-noise level and measures (L1/L2)² near-sparsity of the
# eigenvalues (effective rank) and of the top-5 eigenvectors
# (effective pixel count). Produces eigenval_sparsity.png and
# eigenvec_sparsity.png; the grid is cached in
# sparsity_grid.safetensors.
# ============================================================
import os
from itertools import product
from pathlib import Path

import plotly.express as px
import plotly.io as pio
import torch
from image import MNIST, Model
from kornia.augmentation import RandomGaussianNoise
from safetensors.torch import load_file, save_file
from torch import nn

# Run from repo root so ./data always maps to <repo>/data
os.chdir(Path(__file__).resolve().parents[3])
HERE = Path(__file__).parent

pio.templates.default = "plotly_white"

device = "cuda" if torch.cuda.is_available() else "cpu"

CACHE = HERE / "sparsity_grid.safetensors"

if CACHE.exists():
    print(f"Loading cached {CACHE.name} — delete it to retrain.")
    tensors = load_file(CACHE)
    avals, avecs = tensors["vals"], tensors["vecs"]
else:
    # =====================================
    # Preallocate storage for grid experiment
    # =====================================

    # Dimensions:
    #   wd_index × noise_index × digit × eigenvalue × pixel
    # Only the top-5 (most positive) eigenvectors are kept — the sparsity
    # analysis below uses nothing else, and the full grid would be ~6.6 GB.
    avecs = torch.empty(6, 6, 10, 5, 784)
    avals = torch.empty(6, 6, 10, 512)

    torch.set_grad_enabled(True)

    # =====================================
    # Grid search over weight decay and noise
    # =====================================

    for wd, i in product(range(6), range(6)):
        mnist = Model.from_config(
            epochs=50,
            wd=wd * 0.2,
            d_hidden=512,
            n_layer=1,
            residual=False,
            # Fixed seed: tying the seed to the noise index would confound
            # noise effects with initialization effects.
            seed=42,
        ).to(device)

        # Input noise increases with index i
        transform = nn.Sequential(
            RandomGaussianNoise(mean=0, std=0.2 * i, p=1),
        )

        train, test = MNIST(train=True), MNIST(train=False)
        mnist.fit(train, test, transform)

        # Decompose trained model
        vals, vecs = mnist.decompose()

        # Store eigenvalues and the top-5 eigenvectors.
        # eigh returns eigenvalues in ascending order, so the most positive
        # eigenvectors are the LAST five; flip to get rank 0 = largest.
        avals[wd, i] = vals
        avecs[wd, i] = vecs[:, -5:, :].flip(1)

    torch.set_grad_enabled(False)

    save_file(dict(vals=avals, vecs=avecs), CACHE)

# =====================================
# Eigenvalue sparsity analysis
# =====================================

l2 = avals.pow(2).sum(-1).sqrt()
l1 = avals.abs().sum(-1)

fig = px.imshow(
    (l1 / l2).pow(2).mean(-1).flip(0).cpu(),
    color_continuous_scale="Viridis",
    zmin=0,
)

fig.update_xaxes(
    tickvals=list(range(6)),
    ticktext=[f"{i * 0.2:.1f}" for i in range(6)],
    title="Input Noise",
)

fig.update_yaxes(
    tickvals=list(range(6)),
    ticktext=[f"{i * 0.2:.1f}" for i in reversed(range(6))],
    title="Weight Decay",
)

fig.update_layout(
    width=600,
    height=500,
    margin=dict(l=0, r=0, b=0, t=30),
    title="Eigenvalue Sparsity",
    title_x=0.45,
)

fig.write_image(HERE / "eigenval_sparsity.png", scale=4)

# =====================================
# Eigenvector sparsity analysis
# =====================================

# avecs already holds only the top-5 positive eigenvectors (see above)
l2 = avecs.pow(2).sum(-1).sqrt()
l1 = avecs.abs().sum(-1)

fig = px.imshow(
    (l1 / l2).pow(2).mean(-1).mean(-1).flip(0).cpu(),
    color_continuous_scale="Viridis",
)

fig.update_xaxes(
    tickvals=list(range(6)),
    ticktext=[f"{i * 0.2:.1f}" for i in range(6)],
    title="Input Noise",
)

fig.update_yaxes(
    tickvals=list(range(6)),
    ticktext=[f"{i * 0.2:.1f}" for i in reversed(range(6))],
    title="Weight Decay",
)

fig.update_layout(
    width=600,
    height=500,
    margin=dict(l=0, r=0, b=0, t=30),
    title="Eigenvector Sparsity",
    title_x=0.45,
)

fig.write_image(HERE / "eigenvec_sparsity.png", scale=4)
