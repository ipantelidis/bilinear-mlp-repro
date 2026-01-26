# =====================================
# Imports and global setup
# =====================================
import os
from collections import namedtuple
from itertools import product

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from einops import *
from image import MNIST, Model
from kornia.augmentation import (RandomAffine, RandomGaussianBlur,
                                 RandomGaussianNoise, RandomSaltAndPepperNoise)
from plotly.subplots import make_subplots
from safetensors.torch import load_file, save_file
from torch import nn

pio.templates.default = "plotly_white"

# Shared color settings
color = dict(
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
    zmin=-1.15,
    zmax=1.15,
)

# =====================================
# Parameter grid definition
# =====================================

Params = namedtuple(
    'Params',
    ['rotation', 'translation', 'noise', 'blur', 'pepper', 'dropout'],
    defaults=(None,) * 6,
)

# Select experiment axis
params = Params(noise=3, blur=5)
# params = Params(noise=3, rotation=5)
# params = Params(noise=3, translation=5)

params = {
    k: range(v) if v is not None else [0]
    for k, v in params._asdict().items()
}

shape = [len(v) for v in params.values()]

# =====================================
# Preallocate storage
# =====================================

all_vecs = torch.empty(shape + [10, 10, 28 * 28])
all_vals = torch.empty(shape + [10, 512])
all_accs = torch.empty(shape)

train, test = MNIST(train=True), MNIST(train=False)

# =====================================
# Run experiments
# =====================================

torch.set_grad_enabled(True)

for run in [Params(*values) for values in product(*params.values())]:
    print(run)

    rotation, translation, noise, blur, pepper, dropout = run

    transform = nn.Sequential(
        RandomGaussianNoise(mean=0, std=noise * 0.2, p=1),
        # Comment if not using blur
        RandomGaussianBlur(
            kernel_size=5,
            sigma=(0.01 + 0.2 * blur, 0.01 + 0.2 * blur), 
            p=1,
        ),
        # Uncomment if using rotation/translation
        # RandomAffine(degrees=rotation * 8, translate=translation * 0.05, p=1.0),
    )

    model = Model.from_config(
        epochs=50,
        wd=1.0,
        n_layer=1,
        residual=False,
    ).cuda()

    metrics = model.fit(train, test, transform)
    vals, vecs = model.decompose()

    acc = metrics["val/acc"].iloc[-1]

    idx = tuple(getattr(run, k) for k in Params._fields)
    all_vecs[idx] = vecs[..., -10:, :]
    all_vals[idx] = vals
    all_accs[idx] = acc

torch.set_grad_enabled(False)

# =====================================
# Cache results
# =====================================

name = "blur"  # or "rotation" or "translation"

save_file(
    dict(vecs=all_vecs, vals=all_vals, accs=all_accs),
    f"{name}.safetensors",
)

tensors = load_file(f"{name}.safetensors")
all_vecs = tensors["vecs"]
all_vals = tensors["vals"]
all_accs = tensors["accs"]

# =====================================
# Select subset for visualization
# =====================================

dims_map = {
    "blur": [0, 0, slice(None), slice(None), 0, 0],
    "rotation": [slice(None), 0, slice(None), 0, 0, 0],
    "translation": [0, slice(None), slice(None), 0, 0, 0],
}

if name == "blur":
    subset = rearrange(
        all_vecs[tuple(dims_map[name] + [0, -1])],
        "... (w h) -> ... w h",
        w=28,
        h=28,
    )
elif name == "rotation":
    subset = rearrange(
        all_vecs[tuple(dims_map[name] + [5, -1])],
        "... (w h) -> ... w h",
        w=28,
        h=28,
    ).transpose(0, 1)
elif name == "translation":
    subset = rearrange(
        all_vecs[tuple(dims_map[name] + [0, -1])],
        "... (w h) -> ... w h",
        w=28,
        h=28,
    ).transpose(0, 1)

rows, cols = subset.size(0), subset.size(1)

# =====================================
# Titles 
# =====================================

if name=="blur":
    titles = [f"{all_accs[tuple(dims_map[name])].mT[c, r]:.1%}" for r, c in product(range(rows), range(cols))]
else:
    titles = [f"{all_accs[tuple(dims_map[name])][c, r]:.1%}" for r, c in product(range(rows), range(cols))]

fig = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=titles,
    horizontal_spacing=0.03,
    vertical_spacing=0.05,
)

# =====================================
# Manual sign alignment 
# =====================================

if name == "translation":
    idxs = [(0, 1), (1, 1), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2)]
elif name == "rotation":
    idxs = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3)]
elif name == "blur":
    idxs = [(1, 0), (1, 1), (2, 1)]

flips = torch.ones(subset.shape[:2])
[flips.__setitem__(idx, -1) for idx in idxs]
subset = einsum(subset, flips, "a b w h, a b -> a b w h").flip(0)

# =====================================
# Add heatmaps
# =====================================

for row, col in product(range(rows), range(cols)):
    fig.add_trace(
        go.Heatmap(
            z=subset[row, col].flip(0),
            showscale=False,
            colorscale="RdBu",
            zmid=0,
        ),
        row=row + 1,
        col=col + 1,
    )

# =====================================
# Layout & annotations
# =====================================

width, height = 800, 600

fig.update_layout(
    showlegend=False,
    paper_bgcolor="white",
    plot_bgcolor="white",
    height=height,
    width=width,
    margin=dict(l=50, r=0, b=50, t=20),
)

fig.update_annotations(font_size=13)
for a in fig.layout.annotations:
    a.update(y=a["y"] - 0.02)

for row, col in product(range(rows), range(cols)):
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        scaleanchor=f"y{row * cols + col + 1}",
        scaleratio=1,
        row=row + 1,
        col=col + 1,
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        constrain="domain",
        row=row + 1,
        col=col + 1,
    )

# =====================================
# Axis labels and arrows
# =====================================

if name == "rotation":
    label, start, end = "Rotation", "0 degrees", "40 degrees"
elif name == "translation":
    label, start, end = "Translation", "0 pixels", "7 pixels"
elif name == "blur":
    label, start, end = "Blur", "0 sigma", "1 sigma"

# =====================================
# Axis labels and arrows
# =====================================

# Horizontal axis (Rotation / Translation / Blur)
fig.add_annotation(
    x=0, y=-10 / height,
    xref="paper", yref="paper",
    showarrow=True,
    ax=320, ay=0,
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2
)

fig.add_annotation(
    x=1, y=-10 / height,
    xref="paper", yref="paper",
    showarrow=True,
    ax=-320, ay=0,
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2
)

fig.add_annotation(
    text=label,
    x=0.5, y=-20 / height,
    font=dict(size=16),
    xref="paper", yref="paper",
    showarrow=False
)

fig.add_annotation(
    text=start,
    x=0.02, y=-0.07,
    font=dict(size=14),
    xref="paper", yref="paper",
    showarrow=False
)

fig.add_annotation(
    text=end,
    x=0.98, y=-0.07,
    font=dict(size=14),
    xref="paper", yref="paper",
    showarrow=False
)

# Vertical axis (Noise)
fig.add_annotation(
    x=-22 / width, y=0.02,
    xref="paper", yref="paper",
    showarrow=True,
    ax=0, ay=-220,
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
)

fig.add_annotation(
    x=-22 / width, y=0.98,
    xref="paper", yref="paper",
    showarrow=True,
    ax=0, ay=220,
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
)

fig.add_annotation(
    text="Noise",
    x=-13 / width, y=0.5,
    font=dict(size=16),
    textangle=-90,
    xref="paper", yref="paper",
    showarrow=False
)

fig.add_annotation(
    text="0.4 norm",
    x=-0.06, y=0.95,
    font=dict(size=14),
    textangle=-90,
    xref="paper", yref="paper",
    showarrow=False
)

fig.add_annotation(
    text="0 norm",
    x=-0.06, y=0.05,
    font=dict(size=14),
    textangle=-90,
    xref="paper", yref="paper",
    showarrow=False
)

fig.write_image(f"{name}.png")
