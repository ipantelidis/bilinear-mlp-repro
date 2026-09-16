"""
Exp 10 — Synthesis Quality & Mass Ratio Correlation

Two analyses:
  (a) Per-class MSE comparison: weight-based synthesis (decode top +eigvec)
      vs. actual VAE reconstruction (decode mean latent code).
      Measures how much quality is lost by using weights alone.

  (b) Scatter plot: positive eigenvalue mass ratio vs. per-class reconstruction
      MSE.  Tests whether classes with more generative eigenvalue mass are
      also easier to reconstruct.

Robustness of (b): the Pearson correlation (r ≈ −0.70, p ≈ 0.025) visually
hinges on outlier class d1, so we also report the Spearman rank correlation
and a leave-one-out Pearson sweep (drop each class in turn). If the
correlation collapses without d1, the association is fragile — reported
honestly in the JSON and on the figure.

Outputs:
    figures/mnist/exp10_synthesis_quality.png
    figures/mnist/exp10_mass_mse_scatter.png
    figures/exp10_results.json
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import DecBilinearVAE
from train    import load_checkpoint
from analysis import get_decoder_interaction_matrix, decompose, mean_lat_norm
from visualize import save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = "/home/v25/ippa6201/bilinear-mlp-repro/data"


def main():
    model = DecBilinearVAE(); load_checkpoint(model, CKPT); model.eval()

    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    img_buckets, lat_buckets = {}, {}
    with torch.no_grad():
        for x, y in loader:
            mu, _ = model.encode(x)
            for i, lbl in enumerate(y.tolist()):
                img_buckets.setdefault(lbl, []).append(x[i])
                lat_buckets.setdefault(lbl, []).append(mu[i])

    mean_imgs = {c: torch.stack(v).mean(0) for c, v in img_buckets.items()}
    mean_lats = {c: torch.stack(v).mean(0) for c, v in lat_buckets.items()}
    scale = mean_lat_norm(model, loader)

    synth_mses, recon_mses, mass_ratios = [], [], []

    with torch.no_grad():
        for c in range(10):
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]

            # Weight-based synthesis
            v      = vecs[pos_idx[0]] * scale if len(pos_idx) else torch.zeros(model.d_latent)
            synth  = model.decode(v.unsqueeze(0)).squeeze(0)
            # Actual VAE reconstruction
            recon  = model.decode(mean_lats[c].unsqueeze(0)).squeeze(0)

            synth_mses.append(((synth - mean_imgs[c]) ** 2).mean().item())
            recon_mses.append(((recon - mean_imgs[c]) ** 2).mean().item())

            # Mass ratio
            pos_mass = vals[pos_idx].abs().sum().item() if len(pos_idx) else 0.0
            neg_idx  = (vals < 0).nonzero(as_tuple=True)[0]
            neg_mass = vals[neg_idx].abs().sum().item() if len(neg_idx) else 1e-8
            mass_ratios.append(pos_mass / (neg_mass + 1e-8))

    ratios = [s / r for s, r in zip(synth_mses, recon_mses)]

    print(f"{'Class':<8} {'synth_mse':>12} {'recon_mse':>12} {'ratio':>8}")
    for c in range(10):
        print(f"  d{c}      {synth_mses[c]:>12.4f} {recon_mses[c]:>12.4f} {ratios[c]:>8.2f}x")
    print(f"\nMean quality ratio: {np.mean(ratios):.2f}x")

    # Figure 1: grouped bar chart
    x     = np.arange(10)
    width = 0.38
    fig1, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].bar(x - width/2, synth_mses, width, label="Weight-based synthesis",
                color="steelblue", alpha=0.85)
    axes[0].bar(x + width/2, recon_mses, width, label="Actual reconstruction",
                color="seagreen", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"d{c}" for c in range(10)])
    axes[0].set_ylabel("MSE to class mean image", fontsize=11)
    axes[0].set_title("Absolute MSE per class", fontsize=11)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis="y")

    cmap = plt.get_cmap("tab10")
    bars = axes[1].bar(range(10), ratios, color=[cmap(c) for c in range(10)], edgecolor="white")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1, label="equal quality")
    for bar, ratio in zip(bars, ratios):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{ratio:.1f}×", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(range(10)); axes[1].set_xticklabels([f"d{c}" for c in range(10)])
    axes[1].set_ylabel("Synthesis MSE / Reconstruction MSE", fontsize=11)
    axes[1].set_title(f"Quality ratio  (mean = {np.mean(ratios):.1f}×)", fontsize=11)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, axis="y")

    fig1.suptitle("Exp 10 — Synthesis quality: weight-based vs actual reconstruction",
                  fontsize=12, y=1.02)
    fig1.tight_layout()
    save_fig(fig1, "figures/mnist/exp10_synthesis_quality.png")

    # Figure 2: mass ratio vs reconstruction MSE scatter
    r_stat, p_val = stats.pearsonr(mass_ratios, recon_mses)
    rho, rho_p    = stats.spearmanr(mass_ratios, recon_mses)
    print(f"\nMass ratio vs recon MSE: Pearson r={r_stat:.3f}  p={p_val:.3f}")
    print(f"                         Spearman ρ={rho:.3f}  p={rho_p:.3f}")

    # Leave-one-out Pearson: drop each class in turn
    loo = {}
    for drop in range(10):
        mr = [mass_ratios[c] for c in range(10) if c != drop]
        rm = [recon_mses[c]  for c in range(10) if c != drop]
        r_i, p_i = stats.pearsonr(mr, rm)
        loo[drop] = {"r": float(r_i), "p": float(p_i)}
        print(f"  drop d{drop}: r={r_i:.3f}  p={p_i:.3f}")
    loo_rs   = [loo[c]["r"] for c in range(10)]
    loo_span = (min(loo_rs), max(loo_rs))
    print(f"  leave-one-out r range: [{loo_span[0]:.3f}, {loo_span[1]:.3f}]  "
          f"(without d1: r={loo[1]['r']:.3f}, p={loo[1]['p']:.3f})")

    with open("figures/exp10_results.json", "w") as f:
        json.dump({
            "synth_mse_per_class":  synth_mses,
            "recon_mse_per_class":  recon_mses,
            "quality_ratio_per_class": ratios,
            "quality_ratio_mean":   float(np.mean(ratios)),
            "mass_ratio_per_class": mass_ratios,
            "pearson":  {"r": float(r_stat), "p": float(p_val)},
            "spearman": {"rho": float(rho), "p": float(rho_p)},
            "leave_one_out_pearson": loo,
            "leave_one_out_r_range": list(loo_span),
        }, f, indent=2)
    print("Saved figures/exp10_results.json")

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for c in range(10):
        ax2.scatter(mass_ratios[c], recon_mses[c], color=cmap(c), s=120, zorder=3)
        ax2.annotate(f"d{c}", (mass_ratios[c], recon_mses[c]),
                     fontsize=9, ha="left", va="bottom", color=cmap(c))
    m, b = np.polyfit(mass_ratios, recon_mses, 1)
    xs = np.linspace(min(mass_ratios), max(mass_ratios), 50)
    ax2.plot(xs, m * xs + b, "k--", linewidth=1.2, alpha=0.6)
    ax2.set_xlabel("Positive eigenvalue mass ratio", fontsize=11)
    ax2.set_ylabel("Reconstruction MSE",             fontsize=11)
    ax2.set_title(f"Exp 10 — Mass ratio vs reconstruction MSE\n"
                  f"Pearson r={r_stat:.3f} (p={p_val:.3f}), "
                  f"Spearman ρ={rho:.3f} (p={rho_p:.3f})",
                  fontsize=11)
    ax2.text(0.02, 0.02,
             f"leave-one-out Pearson r ∈ [{loo_span[0]:.2f}, {loo_span[1]:.2f}]\n"
             f"without d1: r={loo[1]['r']:.2f} (p={loo[1]['p']:.2f})",
             transform=ax2.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8,
                       edgecolor="gray"))
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    save_fig(fig2, "figures/mnist/exp10_mass_mse_scatter.png")


if __name__ == "__main__":
    main()
