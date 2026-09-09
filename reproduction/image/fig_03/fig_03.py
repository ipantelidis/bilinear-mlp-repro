# ============================================================
# Figure 3 — Eigenspectrum and top eigenvectors for digit 5
# Reproduces Figure 3 from Pearce et al. (2025).
#
# Trains one MNIST bilinear model (paper setup) and plots the top
# four positive and negative eigenvectors of digit 5's interaction
# matrix alongside its eigenvalue spectrum. Produces fig_03.png.
# ============================================================

import os
from pathlib import Path

import plotly.io as pio
import torch
from image import MNIST, Model
from image.plotting import plot_eigenspectrum
from kornia.augmentation import RandomGaussianNoise
from torch import nn

# Run from repo root so ./data always maps to <repo>/data
os.chdir(Path(__file__).resolve().parents[3])
HERE = Path(__file__).parent

pio.templates.default = "plotly_white"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# Model initialization
# =====================================

model = Model.from_config(
    epochs=100,
    wd=1.0,
    d_hidden=512,
    n_layer=1,
    residual=False,
    seed=420,
).to(device)

# =====================================
# Data augmentation
# =====================================

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.5, p=1),
)

# =====================================
# Train MNIST model
# =====================================

torch.set_grad_enabled(True)

train, test = MNIST(train=True), MNIST(train=False)
model.fit(train, test, transform)

torch.set_grad_enabled(False)

# =====================================
# Plot Figure 3: eigenspectrum for digit 5
# =====================================
fig = plot_eigenspectrum(
    model,
    digit=5,
    eigenvectors=4,
    eigenvalues=20,
)

fig.write_image(HERE / "fig_03.png", scale=4)


