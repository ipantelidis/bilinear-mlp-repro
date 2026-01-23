# =====================================
# Imports and global setup
# =====================================

import plotly.express as px
import plotly.io as pio
import torch
from einops import *
from image import MNIST, Model
from kornia.augmentation import RandomGaussianNoise, RandomSaltAndPepperNoise
from torch import nn

pio.templates.default = "plotly_white"

# Shared color settings
color = dict(
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
)

# =====================================
# Preallocate storage for experiments
# =====================================

# all_vecs: [noise_level, digit, top_eigenvectors, pixels]
all_vecs = torch.empty([5, 10, 10, 28 * 28])

# all_vals: [noise_level, digit, eigenvalues]
all_vals = torch.empty([5, 10, 512])

# all_accs: [noise_level]
all_accs = torch.empty([5])

# =====================================
# Train models with increasing input noise
# =====================================

for i in range(5):
    model = Model.from_config(
        epochs=50,
        wd=1.0,
        n_layer=1,
        residual=False,
    ).cuda()

    # Input noise increases with i
    transform = nn.Sequential(
        RandomGaussianNoise(mean=0, std=i * 0.2, p=1),
        # RandomSaltAndPepperNoise(amount=0.04 * i, salt_vs_pepper=0.5, p=1),
    )

    torch.set_grad_enabled(True)

    train, test = MNIST(train=True), MNIST(train=False)
    metrics = model.fit(train, test, transform)

    torch.set_grad_enabled(False)

    # Decompose trained model
    vals, vecs = model.decompose()

    # Store top 10 eigenvectors per digit
    all_vecs[i] = vecs[..., -10:, :]

    # Store all eigenvalues
    all_vals[i] = vals

    # Store final validation accuracy
    all_accs[i] = metrics["val/acc"].iloc[-1]

# Aliases for readability
vecs, vals, accs = all_vecs, all_vals, all_accs

# =====================================
# Select subset for visualization
# =====================================

# Top eigenvector of digit 0 across noise levels
subset = vecs[:, 0, -1]

# Normalize each eigenvector independently
subset /= subset.abs().max(1, keepdim=True).values

# =====================================
# Plot eigenvectors vs noise
# =====================================

fig = px.imshow(
    subset.view(-1, 28, 28).cpu(),
    facet_col=0,
    facet_col_wrap=5,
    height=250,
    width=1000,
    **color,
)

fig.update_layout(
    coloraxis_showscale=False,
    margin=dict(l=0, r=0, b=20, t=5),
)

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

# =====================================
# Annotate accuracy per noise level
# =====================================

for i, annotation in enumerate(fig.layout.annotations):
    annotation.update(
        text=f"{accs[i]:.1%}",
        y=annotation["y"] - 0.04,
    )

# =====================================
# Add noise direction annotations
# =====================================

fig.add_annotation(
    x=0, y=-0.0,
    xref="paper", yref="paper",
    showarrow=True,
    ax=450, ay=0,
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
)

fig.add_annotation(
    x=1, y=-0.0,
    xref="paper", yref="paper",
    showarrow=True,
    ax=-450, ay=0,
    axref="pixel", ayref="pixel",
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
)

fig.add_annotation(
    text="Noise",
    ax=0.5, y=-0.05,
    font=dict(size=16),
    xref="paper", yref="paper",
    axref="pixel", ayref="pixel",
    showarrow=False,
)

fig.add_annotation(
    text="norm=1",
    x=0.97, y=-0.1,
    font=dict(size=14),
    xref="paper", yref="paper",
    axref="pixel", ayref="pixel",
    showarrow=False,
)

fig.add_annotation(
    text="norm=0",
    x=0.02, y=-0.1,
    font=dict(size=14),
    xref="paper", yref="paper",
    axref="pixel", ayref="pixel",
    showarrow=False,
)

# =====================================
# Save figure
# =====================================

fig.write_image(
    "..outputs/figures/fig4.png",
    scale=4,
)

