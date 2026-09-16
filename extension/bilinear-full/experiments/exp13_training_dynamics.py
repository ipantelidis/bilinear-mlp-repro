"""
Exp 13 — Training Dynamics

Tracks how key spectral properties emerge during training, using per-epoch
checkpoints from train_epochs.py.

For each epoch checkpoint:
  1. Near-universal direction cos: mean pairwise cos of Q_dec top eigvecs across classes
  2. Encoder cross-class cos: mean pairwise cos of encoder Q_in top eigvecs
  3. Test ELBO (from checkpoint metadata)

Requires: checkpoints/mnist/epochs/epoch01.pt .. epoch20.pt
Run train_epochs.py first.

Figure saved:
    figures/mnist/exp13_training_dynamics.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE
from analysis import (get_decoder_interaction_matrix, get_encoder_interaction_matrix,
                      decompose)
from visualize import save_fig

EPOCH_DIR = "checkpoints/mnist/epochs"
DATA      = "/home/v25/ippa6201/bilinear-mlp-repro/data"


def _cos(a, b):
    return float(torch.dot(a.float(), b.float()) / (a.float().norm() * b.float().norm() + 1e-8))


def _near_universal_cos(model, mean_imgs, classes):
    """Mean pairwise cos of Q_dec top +eigvecs across classes."""
    eigvecs = {}
    for c in classes:
        Q = get_decoder_interaction_matrix(model, mean_imgs[c])
        vals, vecs = decompose(Q)
        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        eigvecs[c] = vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(model.d_latent)
    pairs = list(combinations(classes, 2))
    cos_vals = [abs(_cos(eigvecs[a], eigvecs[b])) for a, b in pairs]
    return float(np.mean(cos_vals))


def _enc_cross_class_cos(model, classes):
    """Mean pairwise cos of encoder Q_in top +eigvecs across classes."""
    eigvecs = {}
    for c in classes:
        d = torch.zeros(model.d_latent); d[c] = 1.0
        Q = get_encoder_interaction_matrix(model, d)
        vals, vecs = decompose(Q)
        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        eigvecs[c] = vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(model.d_input)
    pairs = list(combinations(classes, 2))
    cos_vals = [abs(_cos(eigvecs[a], eigvecs[b])) for a, b in pairs]
    return float(np.mean(cos_vals))


def main():
    epoch_dir = Path(EPOCH_DIR)
    ckpts = sorted(epoch_dir.glob("epoch*.pt"))
    if not ckpts:
        print(f"No epoch checkpoints found in {EPOCH_DIR}.")
        print("Please run: python train_epochs.py")
        return

    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    buckets = {}
    for x, y in loader:
        for i, lbl in enumerate(y.tolist()):
            buckets.setdefault(lbl, []).append(x[i])
    mean_imgs = {c: torch.stack(v).mean(0) for c, v in buckets.items()}
    classes   = sorted(mean_imgs.keys())

    epochs, test_elbo, dec_cos, enc_cos = [], [], [], []

    print(f"\n{'Ep':>4}  {'Test ELBO':>10}  {'Dec univ cos':>14}  {'Enc cross cos':>14}")
    print("-" * 50)

    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = FullBilinearVAE()
        model.load_state_dict(ckpt["model_state"]); model.eval()

        ep   = ckpt["epoch"]
        elbo = ckpt.get("test_total", float("nan"))
        dc   = _near_universal_cos(model, mean_imgs, classes)
        ec   = _enc_cross_class_cos(model, classes)

        epochs.append(ep); test_elbo.append(elbo)
        dec_cos.append(dc); enc_cos.append(ec)

        print(f"{ep:>4}  {elbo:>10.2f}  {dc:>14.4f}  {ec:>14.4f}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, test_elbo, "o-", color="steelblue")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Test ELBO (per sample)")
    axes[0].set_title("Test ELBO over training", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, dec_cos, "s-", color="seagreen", label="trained")
    axes[1].axhline(dec_cos[0] if dec_cos else 0, color="gray", linestyle="--",
                    linewidth=0.8, label="epoch 1 baseline")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Mean pairwise cos (Q_dec top eigvec)")
    axes[1].set_title("Near-universal direction\n(decoder cross-class cos)", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, enc_cos, "^-", color="darkorchid")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Mean pairwise cos (Q_in top eigvec)")
    axes[2].set_title("Encoder cross-class cos\n(lower = more discriminative)", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Exp 13 — Training dynamics (FullBilinearVAE on MNIST)", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp13_training_dynamics.png")


if __name__ == "__main__":
    main()
