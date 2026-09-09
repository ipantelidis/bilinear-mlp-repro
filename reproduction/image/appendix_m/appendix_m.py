# ============================================================
# Appendix M — Adversarial masks under different regularization
# Reproduces Figure 26 from Pearce et al. (2025).
#
# Trains three MNIST bilinear models (no noise / Gaussian noise
# std 0.15 / std 0.30), visualizes the top-3 eigenvectors and their
# pseudoinverse adversarial masks for digit 3, and evaluates each
# mask set on the test set against a permuted-mask baseline.
# Produces adversarial_masks_{label}.png,
# adversarial_results_{label}.json and reusable checkpoints
# adversarial_model_{label}.pt.
# ============================================================
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops import einsum, rearrange
from image import MNIST, Model
from kornia.augmentation import RandomGaussianNoise
from mpl_toolkits.axes_grid1 import ImageGrid

# ============================================================
# Global config
# ============================================================
os.chdir(Path(__file__).resolve().parents[3])
HERE = Path(__file__).parent

device = "cuda"
torch.set_grad_enabled(True)

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
    # Model (loaded from checkpoint when available)
    # ----------------------------
    ckpt_path = HERE / f"adversarial_model_{cfg['label']}.pt"

    model = Model.from_config(
        epochs=30,
        wd=0.0,
        d_hidden=512,
        n_layer=1,
        residual=True,
        seed=42
    ).to(device)

    if ckpt_path.exists():
        print(f"Loading checkpoint {ckpt_path.name} — delete it to retrain.")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        torch.set_grad_enabled(False)
        model.to("cpu")
    else:
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
            ckpt_path,
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

    # File name follows the variant label so reruns overwrite the
    # committed artifacts instead of orphaning them.
    plt.savefig(HERE / f"adversarial_masks_{cfg['label']}.png", bbox_inches="tight")
    plt.close()

    # ----------------------------
    # Quantitative evaluation (test set, permuted-mask baseline)
    # ----------------------------
    strengths = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    eig_idxs = [-1, -2, -3]

    inputs_vec = test.x.cpu().view(test.x.size(0), -1)
    labels = test.y.cpu()

    acc = lambda logits: (logits.argmax(-1) == labels).float().mean().item()
    mis = lambda logits, d: (
        (logits.argmax(-1) == d) & (labels != d)
    ).float().mean().item()

    results = {
        "config": {k: v for k, v in cfg.items()},
        "strengths": strengths,
        "acc_orig": acc(model(test.x.cpu())),
        "acc_adv": [], "acc_rand": [],
        "mis_adv": [], "mis_rand": [],
    }

    for st in strengths:
        accs_a, accs_r, miss_a, miss_r = [], [], [], []
        for d in range(10):
            for ei in eig_idxs:
                m = encoders_px[d, ei].clone()
                m /= m.std()
                m = st * m

                perm = torch.randperm(len(m))
                m_rand = m[perm]

                adv = (inputs_vec + m).view(-1, 1, 28, 28)
                rnd = (inputs_vec + m_rand).view(-1, 1, 28, 28)

                logits_a, logits_r = model(adv), model(rnd)
                accs_a.append(acc(logits_a)); accs_r.append(acc(logits_r))
                miss_a.append(mis(logits_a, d)); miss_r.append(mis(logits_r, d))

        results["acc_adv"].append(sum(accs_a) / len(accs_a))
        results["acc_rand"].append(sum(accs_r) / len(accs_r))
        results["mis_adv"].append(sum(miss_a) / len(miss_a))
        results["mis_rand"].append(sum(miss_r) / len(miss_r))
        print(f"[{cfg['label']}] std={st:.2f} "
              f"acc adv/rand = {results['acc_adv'][-1]:.3f}/{results['acc_rand'][-1]:.3f} "
              f"mis adv/rand = {results['mis_adv'][-1]:.3f}/{results['mis_rand'][-1]:.3f}")

    out_json = HERE / f"adversarial_results_{cfg['label']}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_json}")

    torch.set_grad_enabled(True)
