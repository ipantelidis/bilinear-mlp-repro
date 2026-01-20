# =====================================
# Imports and global setup
# =====================================

import plotly.express as px
import plotly.io as pio
import torch
from einops import *
from image import FMNIST, MNIST, Model
from kornia.augmentation import RandomAffine, RandomGaussianNoise
from torch import nn

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
    n_layer=1,
    residual=False,
    seed=420,
).cuda()

fmnist = Model.from_config(
    epochs=100,
    wd=1.0,
    n_layer=1,
    residual=False,
    seed=420,
).cuda()

# =====================================
# Data augmentation (regularization)
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
    "../outputs/figures/fig2/eigenfeatures.png",
    scale=4,
)

