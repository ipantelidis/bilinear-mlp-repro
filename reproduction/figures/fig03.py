# =====================================
# Imports and global setup
# =====================================

import os

import plotly.io as pio
import torch
from image import MNIST, Model
from image.plotting import plot_eigenspectrum
from kornia.augmentation import RandomAffine, RandomGaussianNoise
from torch import nn

pio.templates.default = "plotly_white"

# =====================================
# Paths and device
# =====================================

out_dir = "../outputs/figures/"
os.makedirs(out_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# Model initialization
# =====================================

model = Model.from_config(
    epochs=100,
    wd=1.0,
    n_layer=1,
    residual=False,
    seed=420,
).to(device)

# =====================================
# Data augmentation (same as Figure 2)
# =====================================

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.5, p=1),
    # RandomAffine(degrees=0, translate=(0.25, 0.25), p=1),
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

out_path = os.path.join(out_dir, "fig3.png")
fig.write_image(out_path, scale=4)


