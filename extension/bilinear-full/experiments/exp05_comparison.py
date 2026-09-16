"""
Exp 05 — Four-Way Model Comparison

Compares reconstruction quality and latent structure across all four model variants:
  (1) VanillaVAE           — standard MLP encoder + MLP decoder
  (2) BilinearVAE          — bilinear encoder + MLP decoder
  (3) DecBilinearVAE       — MLP encoder + bilinear decoder
  (4) FullBilinearVAE      — bilinear encoder + bilinear decoder  ← this work

Metrics:
  - Per-class reconstruction MSE
  - Test ELBO (from checkpoint history)
  - Latent effective rank

Figure saved:
    figures/mnist/exp05_comparison.png
"""

import os
import json
import importlib.util
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import FullBilinearVAE
from train  import load_checkpoint, elbo_loss
from visualize import save_fig

DATA     = "/home/v25/ippa6201/bilinear-mlp-repro/data"
REPO     = "/home/v25/ippa6201/bilinear-mlp-repro"
ENC_DIR  = os.path.join(REPO, "extension/bilinear-encoder")
DEC_DIR  = os.path.join(REPO, "extension/bilinear-decoder")
EXT_FULL = os.path.join(REPO, "extension_full")


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eff_rank(model, loader, device="cpu"):
    model.eval()
    all_mu = []
    with torch.no_grad():
        for x, _ in loader:
            mu, _ = model.encode(x.to(device))
            all_mu.append(mu.cpu())
    all_mu = torch.cat(all_mu)
    mu_c = all_mu - all_mu.mean(0)
    cov  = (mu_c.T @ mu_c) / (len(all_mu) - 1)
    eigs = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0)
    p    = eigs / (eigs.sum() + 1e-10)
    return float(torch.exp(-(p * torch.log(p + 1e-10)).sum()))


def _per_class_mse(model, loader, device="cpu"):
    model.eval()
    img_buckets, recon_buckets = {}, {}
    with torch.no_grad():
        for x, y in loader:
            recon, _, _ = model(x.to(device))
            for i, lbl in enumerate(y.tolist()):
                img_buckets.setdefault(lbl, []).append(x[i])
                recon_buckets.setdefault(lbl, []).append(recon[i].cpu())
    mses = {}
    for c in sorted(img_buckets):
        imgs   = torch.stack(img_buckets[c])
        recons = torch.stack(recon_buckets[c])
        mses[c] = ((imgs - recons) ** 2).mean().item()
    return mses


def main():
    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    # Load all four models
    models = {}

    # 1. VanillaVAE
    try:
        ext_models = _load_module("ext_models", os.path.join(EXT_FULL, "models.py"))
        ext_train  = _load_module("ext_train",  os.path.join(EXT_FULL, "train.py"))
        van = ext_models.VanillaVAE()
        ext_train.load_checkpoint(van, os.path.join(EXT_FULL, "checkpoints/mnist/vanilla_vae/model.pt"))
        models["VanillaVAE"] = van
    except Exception as e:
        print(f"  VanillaVAE not loaded: {e}")

    # 2. BilinearVAE (encoder only)
    try:
        enc_models = _load_module("enc_models", os.path.join(ENC_DIR, "models.py"))
        enc_train  = _load_module("enc_train",  os.path.join(ENC_DIR, "train.py"))
        enc = enc_models.BilinearVAE()
        enc_train.load_checkpoint(enc, os.path.join(ENC_DIR, "checkpoints/mnist/model.pt"))
        models["BilinearVAE\n(enc)"] = enc
    except Exception as e:
        print(f"  BilinearVAE not loaded: {e}")

    # 3. DecBilinearVAE (decoder only)
    try:
        dec_models = _load_module("dec_models", os.path.join(DEC_DIR, "models.py"))
        dec_train  = _load_module("dec_train",  os.path.join(DEC_DIR, "train.py"))
        dec = dec_models.DecBilinearVAE()
        dec_train.load_checkpoint(dec, os.path.join(DEC_DIR, "checkpoints/mnist/model.pt"))
        models["DecBilinearVAE\n(dec)"] = dec
    except Exception as e:
        print(f"  DecBilinearVAE not loaded: {e}")

    # 4. FullBilinearVAE (this)
    full = FullBilinearVAE(); load_checkpoint(full, "checkpoints/mnist/model.pt")
    models["FullBilinearVAE\n(enc+dec)"] = full

    # Compute metrics
    print(f"\n{'Model':<28} {'best_test':>10}  {'eff_rank':>10}  {'mean_mse':>10}")
    print("-" * 62)
    all_mses   = {}
    eff_ranks  = {}
    best_tests = {}

    for name, model in models.items():
        model.eval()
        mses = _per_class_mse(model, loader)
        er   = _eff_rank(model, loader)
        all_mses[name]  = mses
        eff_ranks[name] = er
        label = name.replace("\n", " ")
        print(f"  {label:<26} {'n/a':>10}  {er:>10.2f}  {np.mean(list(mses.values())):>10.4f}")

    with open("figures/mnist/exp05_results.json", "w") as f:
        json.dump({name.replace("\n", " "): {
                       "effective_rank": eff_ranks[name],
                       "mean_mse": float(np.mean(list(all_mses[name].values()))),
                       "per_class_mse": {str(c): v for c, v in all_mses[name].items()}}
                   for name in models}, f, indent=2)

    classes = sorted(all_mses[list(all_mses.keys())[0]].keys())
    model_names = list(models.keys())
    colors = ["steelblue", "indianred", "seagreen", "darkorchid"]

    # Figure: grouped bars (per-class MSE) + effective rank
    x     = np.arange(len(classes))
    w     = 0.2
    offsets = np.linspace(-(len(model_names)-1)/2, (len(model_names)-1)/2, len(model_names)) * w

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        mses = [all_mses[name][c] for c in classes]
        axes[0].bar(x + offsets[i], mses, w, label=name.replace("\n", " "),
                    color=color, alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"d{c}" for c in classes])
    axes[0].set_ylabel("Reconstruction MSE", fontsize=11)
    axes[0].set_title("Per-class reconstruction MSE", fontsize=11)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, axis="y")

    # Effective rank bar
    axes[1].bar(range(len(model_names)),
                [eff_ranks[n] for n in model_names],
                color=colors, alpha=0.85, edgecolor="white")
    for i, (n, er) in enumerate(zip(model_names, [eff_ranks[n] for n in model_names])):
        axes[1].text(i, er + 0.05, f"{er:.2f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels([n.replace("\n", "\n") for n in model_names], fontsize=8)
    axes[1].set_ylabel("Effective rank of latent space", fontsize=11)
    axes[1].set_title("Latent space effective rank", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exp 05 — Four-way model comparison: Vanilla / Enc-bilinear / Dec-bilinear / Full-bilinear",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp05_comparison.png")


if __name__ == "__main__":
    main()
