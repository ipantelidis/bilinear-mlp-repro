# ============================================================
# Imports
# ============================================================
import plotly.express as px
import torch
from einops import einsum
from image import MNIST, Model
from image.plotting import plot_eigenspectrum
from kornia.augmentation import RandomGaussianNoise
from torch.nn.functional import cosine_similarity

# ============================================================
# Config
# ============================================================
device = "cuda"
torch.set_grad_enabled(True)

# Shared color settings
color = dict(
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0
)

DIGIT = 1

# ============================================================
# Label construction (similarity-based binary task)
# ============================================================
def make_label_one_similarity(dataset, target):
    dataset = dataset.view(dataset.size(0), -1)
    target = target.view(-1)

    pos_sims = cosine_similarity(dataset, target)
    neg_sims = cosine_similarity(dataset, 1 - target)

    return ((pos_sims > 0.4) | (neg_sims > 0.4)).long()

# ============================================================
# Train model (single-layer BiMLP)
# ============================================================
model = Model.from_config(
    epochs=100,
    wd=2.0,
    n_layer=1,
    d_output=2,
    bias=True,
    d_hidden=64
).to(device)

transform = torch.nn.Sequential(
    RandomGaussianNoise(mean=0.0, std=0.1, p=1.0),
)

train, test = MNIST(train=True), MNIST(train=False)

target = train.x[6].view(-1)
train.y = make_label_one_similarity(train.x, target)
test.y  = make_label_one_similarity(test.x, target)

model.fit(train, test, transform)

torch.set_grad_enabled(False)

# ============================================================
# Bilinear construction (needed for bias term)
# ============================================================
w_l = torch.block_diag(model.w_l[0], torch.eye(1, device=device))
w_l[:-1, -1] = model.blocks[0].bias.chunk(2)[0]

w_r = torch.block_diag(model.w_r[0], torch.eye(1, device=device))
w_r[:-1, -1] = model.blocks[0].bias.chunk(2)[1]

w_u = torch.cat(
    [model.w_u, torch.tensor([[1], [1]], device=device)],
    dim=1
)

w_e = torch.block_diag(model.w_e, torch.eye(1, device=device))

b = einsum(
    w_u[1],
    w_l,
    w_r,
    "out, out in1, out in2 -> in1 in2"
)
b = 0.5 * (b + b.mT)

# ============================================================
# FIGURE 1 — Eigen-spectrum (paper-style)
# ============================================================
fig = plot_eigenspectrum(
    model,
    digit=DIGIT,
    eigenvectors=2,
    eigenvalues=20
)
fig.update_coloraxes(showscale=False)
fig.write_image(
    f"eigenspectrum.png",
    scale=2
)

# ============================================================
# FIGURE 2 — Target image
# ============================================================
fig = px.imshow(
    target.view(28, 28).cpu(),
    **color
)
fig.update_coloraxes(showscale=False)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.write_image(
    f"target.png",
    scale=2
)

# ============================================================
# FIGURE 3 — Bias term
# ============================================================
bias_img = (b[:-1, -1] @ w_e[:-1, :-1]).cpu().view(28, 28)

fig = px.imshow(
    bias_img,
    **color
)
fig.update_coloraxes(showscale=False)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.write_image(
    "bias.png",
    scale=2
)
