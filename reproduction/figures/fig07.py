# ============================================================
# Imports 
# ============================================================
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
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
os.makedirs("adv", exist_ok=True)

VMIN_DEC, VMAX_DEC = -0.25, 0.25
VMIN_ENC, VMAX_ENC = -0.50, 0.50

DIGIT = 3
TOPK = 10
IDX = -2
IDX_EX = 0
STRENGTH = 0.15

IDXs = torch.arange(-3, 0).flip(0)

# ============================================================
# Experiment definitions
# ============================================================
EXPERIMENTS = {
    "1": dict(
        noise=0.15,
        pixel_mask=False,
        random_mode="permute"
    ),
    "2": dict(
        noise=None,
        pixel_mask=True,
        random_mode="gaussian"
    )
}

# ============================================================
# Shared dataset
# ============================================================
train = MNIST(train=True)
test  = MNIST(train=False)

# ============================================================
# Helper: evaluation
# ============================================================
def evaluate_mask(model, mask, digit, strength):
    inputs = train.x.cpu()

    if cfg["noise"] != None:
        mask_rand = mask.clone()
        perm = torch.randperm(len(mask))
        mask_rand = mask[perm]
    else:
        mask_rand = mask.clone()
        mask_rand[mask_rand > 0] = strength * torch.randn((mask_rand > 0).sum())

    inputs_vec = inputs.view(inputs.size(0), -1)

    adv_inputs  = (inputs_vec + mask.unsqueeze(0)).view(-1, 1, 28, 28)
    rand_inputs = (inputs_vec + mask_rand.unsqueeze(0)).view(-1, 1, 28, 28)

    logits = {
        "orig": model(inputs),
        "adv":  model(adv_inputs),
        "rand": model(rand_inputs),
    }

    metrics = {}
    for k, v in logits.items():
        preds = v.argmax(dim=-1)
        metrics[f"acc_{k}"] = (preds == train.y.cpu()).float().mean()
        metrics[f"mis_{k}"] = ((preds == digit) & (train.y.cpu() != digit)).float().mean()

    return metrics

# ============================================================
# Run experiments
# ============================================================
for tag, cfg in EXPERIMENTS.items():

    print(f"\n=== Running experiment {tag} ===")

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

    # ----------------------------
    # Bilinear decomposition
    # ----------------------------
    l = model.w_l[0]
    r = model.w_r[0]

    B = einsum(model.w_u, l, r, "cls o, o i1, o i2 -> cls i1 i2")
    B = 0.5 * (B + B.mT)

    _, eigvecs = torch.linalg.eigh(B)
    eigvecs = rearrange(eigvecs, "cls model eig -> cls eig model")

    # ----------------------------
    # Encoders / decoders
    # ----------------------------
    decoders = eigvecs[:, -TOPK:]
    encoders = torch.linalg.pinv(
        rearrange(decoders, "cls eig model -> (cls eig) model")
    )
    encoders = rearrange(encoders, "model (cls eig) -> cls eig model", cls=10)

    decoders_px = einsum(decoders, model.w_e, "cls eig model, model pix -> cls eig pix")
    encoders_px = einsum(encoders, model.w_e, "cls eig model, model pix -> cls eig pix")

    # ----------------------------
    # Optional pixel mask (exp 1 only)
    # ----------------------------
    if cfg["pixel_mask"]:
        pix_means = train.x.mean(dim=0).cpu()
        pix_mask  = (pix_means < 0.01).float().view(-1)
        encoders_px *= pix_mask[None, None, :]

    # ----------------------------
    # Sign alignment
    # ----------------------------
    data = torch.stack(
        [test.x[test.y == i][:500].view(-1, 28*28) for i in range(10)],
        dim=0
    ).to(decoders_px.device)

    overlaps = einsum(encoders_px, data, "d1 eig pix, d2 samp pix -> d1 eig samp d2")
    overlaps *= (1 - torch.eye(10))[:, None, None, :]
    signs = overlaps.sum(dim=-1).sum(dim=-1).sign()

    decoders_px *= signs[:, :, None]
    encoders_px *= signs[:, :, None]
    encoders_px_plt = encoders_px.clone()

    # ========================================================
    # FIGURE 1 — Four-panel visualization
    # ========================================================
    mask = encoders_px[DIGIT, IDX]
    mask /= mask[mask > 0].std()
    mask = STRENGTH * mask

    x = train.x[IDX_EX].cpu().view(-1)
    adv_img = (x + mask).view(28, 28)

    if cfg["random_mode"] == "gaussian":
        mask_rand = mask.clone()
        mask_rand[mask_rand > 0] = STRENGTH * torch.randn((mask_rand > 0).sum())
    else:
        perm = torch.randperm(len(mask))
        mask_rand = mask[perm]

    rand_img = (x + mask_rand).view(28, 28)

    fig = plt.figure(figsize=(8, 2.2), dpi=150)
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.25)

    images = [
        decoders_px[DIGIT, IDX].view(28, 28),
        encoders_px_plt[DIGIT, IDX].view(28, 28),
        adv_img,
        rand_img
    ]

    titles = [
        "Eigenvector",
        "Adversarial Mask",
        "Misclassified Example",
        "Random Mask"
    ]

    vmins = [VMIN_DEC, VMIN_ENC, -1, -1]
    vmaxs = [VMAX_DEC, VMAX_ENC,  1,  1]

    for ax, img, title, vmin, vmax in zip(grid, images, titles, vmins, vmaxs):
        ax.imshow(img, cmap="RdBu", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, -0.18, title, ha="center", va="top",
                transform=ax.transAxes, fontsize=9)

    plt.savefig(f"../outputs/figures/fig7a{tag}.png", bbox_inches="tight")
    plt.close()

    # ========================================================
    # FIGURE 2 — Evaluation curves
    # ========================================================
    strengths = np.arange(0.025, .301, 0.025)
    eig_idxs = [-1, -2, -3]

    def buf():
        return torch.zeros(10, len(eig_idxs), len(strengths))

    eval_metrics = defaultdict(buf)

    for d in range(10):
        for e, ei in enumerate(eig_idxs):
            for s, st in enumerate(strengths):
                m = encoders_px_plt[d, ei]
                m /= m.std()
                m = st * m
                metrics = evaluate_mask(model, m, d, st)
                print((d, ei, st, metrics['acc_adv'], metrics['mis_adv']))
                for k in metrics:
                    eval_metrics[k][d, e, s] = metrics[k]

    plt.figure(figsize=(6, 2.5), dpi=150)

    plt.subplot(1,2,1)
    for key, label in zip(['acc_orig','acc_adv','acc_rand'],
                          ['Original','Adversarial','Random']):
        vals = eval_metrics[key].view(-1, len(strengths))
        plt.errorbar(strengths, vals.mean(0),
                     yerr=1.96 * vals.std(0) / np.sqrt(vals.shape[0]),
                     fmt='o-', markersize=3, label=label)
    plt.xlabel("Mask Std Dev")
    plt.ylabel("Accuracy")
    plt.legend(prop={'size':9})

    plt.subplot(1,2,2)
    for key, label in zip(['mis_orig','mis_adv','mis_rand'],
                          ['Original','Adversarial','Random']):
        vals = eval_metrics[key].view(-1, len(strengths))
        plt.errorbar(strengths, vals.mean(0),
                     yerr=1.96 * vals.std(0) / np.sqrt(vals.shape[0]),
                     fmt='o-', markersize=3)
    plt.xlabel("Mask Std Dev")
    plt.ylabel("Misclassification")

    plt.tight_layout()
    plt.savefig(f"../outputs/figures/fig7b{tag}.png", bbox_inches="tight")
    plt.close()

    torch.set_grad_enabled(True)
