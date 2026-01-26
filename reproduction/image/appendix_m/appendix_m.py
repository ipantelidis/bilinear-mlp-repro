# ============================================================
# Imports
# ============================================================
import os

import matplotlib.pyplot as plt
import torch
from einops import einsum, rearrange
from image.datasets import MNIST
from image.model import Model
from kornia.augmentation import RandomGaussianNoise
from mpl_toolkits.axes_grid1 import ImageGrid

# ============================================================
# Global config
# ============================================================
device = "cuda"
torch.set_grad_enabled(True)
os.makedirs("images", exist_ok=True)

VMIN_DEC, VMAX_DEC = -0.25, 0.25
VMIN_ENC, VMAX_ENC = -0.50, 0.50

DIGIT = 3
TOPK = 10
IDXs = torch.arange(-3, 0).flip(0)

# ============================================================
# Experiment variants
# ============================================================
EXPERIMENTS = {
    "A": dict(noise=None, pixel_mask=True, label="gaussian_0.00"),
    "B": dict(noise=0.15, pixel_mask=False, label="gaussian_0.15"),
    "C": dict(noise=0.30, pixel_mask=False, label="gaussian_0.30"),
}

# ============================================================
# Shared dataset
# ============================================================
train = MNIST(train=True, download=True)
test  = MNIST(train=False, download=True)

# ============================================================
# Run experiments
# ============================================================
for tag, cfg in EXPERIMENTS.items():

    print(f"\n=== Running variant {tag} ===")

    # ----------------------------
    # Model
    # ----------------------------
    model = Model.from_config(
        epochs=30,
        wd=0.0,
        n_layer=1,
        residual=True,
        seed=42
    ).to(device)

    transform = None
    if cfg["noise"] is not None:
        transform = torch.nn.Sequential(
            RandomGaussianNoise(mean=0.0, std=cfg["noise"], p=1.0)
        )

    model.fit(train, test, transform)

    torch.set_grad_enabled(False)
    model.to("cpu")

    torch.save(
        {"config": model.config, "model": model.state_dict()},
        f"adversarial_model_{cfg['label']}.pt"
    )

    # ----------------------------
    # Bilinear decomposition
    # ----------------------------
    l = model.w_l[0]
    r = model.w_r[0]

    B = einsum(
        model.w_u, l, r,
        "cls out, out in1, out in2 -> cls in1 in2"
    )
    B = 0.5 * (B + B.mT)

    _, eigvecs = torch.linalg.eigh(B)
    eigvecs = rearrange(eigvecs, "cls model eig -> cls eig model")

    # ----------------------------
    # Pseudoinverse encoders / decoders
    # ----------------------------
    decoders = eigvecs[:, -TOPK:]
    encoders = torch.linalg.pinv(
        rearrange(decoders, "cls eig model -> (cls eig) model")
    )
    encoders = rearrange(encoders, "model (cls eig) -> cls eig model", cls=10)

    decoders_px = einsum(decoders, model.w_e, "cls eig model, model pix -> cls eig pix")
    encoders_px = einsum(encoders, model.w_e, "cls eig model, model pix -> cls eig pix")

    # ----------------------------
    # Optional pixel masking (variant A)
    # ----------------------------
    if cfg["pixel_mask"]:
        threshold = 0.01
        pix_means = train.x.mean(dim=0).cpu()
        pix_mask = (pix_means < threshold).float().view(-1)
        encoders_px *= pix_mask[None, None, :]

    # ----------------------------
    # Fix sign ambiguity
    # ----------------------------
    data = torch.stack(
        [test.x[test.y == i][:500].view(-1, 28 * 28) for i in range(10)],
        dim=0
    ).to(decoders_px.device)  # [digit, samp, pix]

    overlaps = einsum(
        encoders_px, data,
        "d1 eig pix, d2 samp pix -> d1 eig samp d2"
    )

    mask = (1 - torch.eye(10))
    overlaps *= mask[:, None, None, :]

    signs = overlaps.sum(dim=-1).sum(dim=-1).sign()
    decoders_px *= signs[:, :, None]
    encoders_px *= signs[:, :, None]

    # ----------------------------
    # Plot
    # ----------------------------
    fig = plt.figure(figsize=(6, 4), dpi=150)

    # Decoders
    grid_dec = ImageGrid(
        fig, 211,
        nrows_ncols=(1, len(IDXs)),
        axes_pad=0.1,
        cbar_location='right',
        cbar_mode='single'
    )

    for i, k in enumerate(IDXs):
        im = grid_dec[i].imshow(
            decoders_px[DIGIT, k].view(28, 28),
            cmap="RdBu",
            vmin=VMIN_DEC,
            vmax=VMAX_DEC
        )
        grid_dec[i].axis("off")

    grid_dec.cbar_axes[0].colorbar(im)

    # Encoders
    grid_enc = ImageGrid(
        fig, 212,
        nrows_ncols=(1, len(IDXs)),
        axes_pad=0.1,
        cbar_location='right',
        cbar_mode='single'
    )

    for i, k in enumerate(IDXs):
        im = grid_enc[i].imshow(
            encoders_px[DIGIT, k].view(28, 28),
            cmap="RdBu",
            vmin=VMIN_ENC,
            vmax=VMAX_ENC
        )
        grid_enc[i].axis("off")

    grid_enc.cbar_axes[0].colorbar(im)

    plt.savefig(
        f"adversarial_masks_{tag}.png",
        bbox_inches="tight"
    )
    plt.close()

    torch.set_grad_enabled(True)
