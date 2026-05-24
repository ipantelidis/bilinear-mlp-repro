"""
Exp 03 — Causal Generation Test

Synthesise one image per digit class from decoder weights (Exp 01 pipeline),
then encode it back through the encoder and check which class the resulting
latent code lands nearest to.  Closes the generate→encode loop without any
training data or optimisation.

Also runs on a randomly initialised model as a baseline.

Additionally produces a 2-D PCA visualisation explaining why the accuracy is
limited: synthesised codes cluster near one class, revealing that the decoder
uses a near-universal generative direction.

Figures saved:
    figures/mnist/exp03_causal_gen.png
    figures/mnist/exp03_latent_pca.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import DecBilinearVAE
from train    import load_checkpoint
from analysis import get_decoder_interaction_matrix, decompose, compute_class_means, mean_lat_norm
from visualize import save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = "/home/v25/ippa6201/bilinear-mlp-repro/data"


def _build_loader():
    return DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)


def run_causal_test(model, loader):
    buckets = {}
    for x, y in loader:
        for i, lbl in enumerate(y.tolist()):
            buckets.setdefault(lbl, []).append(x[i])
    mean_imgs = {c: torch.stack(v).mean(0) for c, v in buckets.items()}

    lat_means = compute_class_means(model, loader)
    scale     = mean_lat_norm(model, loader)

    results = []
    with torch.no_grad():
        for c in sorted(mean_imgs.keys()):
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            if not len(pos_idx):
                continue
            v_synth  = vecs[pos_idx[0]] * scale
            x_synth  = model.decode(v_synth.unsqueeze(0)).squeeze(0)
            mu_back, _ = model.encode(x_synth.unsqueeze(0))
            mu_back    = mu_back.squeeze(0)
            dists   = {lbl: (mu_back - m).norm().item() for lbl, m in lat_means.items()}
            nearest = min(dists, key=dists.get)
            cos     = float(torch.cosine_similarity(mu_back.unsqueeze(0),
                                                     lat_means[c].unsqueeze(0)))
            results.append({"class": c, "nearest": nearest, "correct": nearest == c,
                             "cos": cos, "synth_img": x_synth.detach(),
                             "synth_code": v_synth})
    return results


def main():
    loader = _build_loader()
    model  = DecBilinearVAE(); load_checkpoint(model, CKPT); model.eval()
    rand   = DecBilinearVAE(); rand.eval()

    trained_res = run_causal_test(model, loader)
    random_res  = run_causal_test(rand,  loader)

    n_tr = sum(r["correct"] for r in trained_res)
    n_rn = sum(r["correct"] for r in random_res)

    print(f"\nTrained: {n_tr}/10 correct")
    for r in trained_res:
        status = "✓" if r["correct"] else f"→{r['nearest']}"
        print(f"  digit {r['class']}: {status:<6}  cos={r['cos']:.3f}")
    print(f"\nRandom: {n_rn}/10 correct")

    # ── Figure 1: synthesised images with tick/cross ──────────────────────
    n   = len(trained_res)
    fig, axes = plt.subplots(4, n, figsize=(2.0 * n, 8),
                              gridspec_kw={"hspace": 0.08, "wspace": 0.05})

    def fill(results, row_img, row_lbl, label):
        for col, r in enumerate(results):
            axes[row_img, col].imshow(r["synth_img"].view(28, 28).numpy(),
                                      cmap="gray_r", vmin=0, vmax=1)
            axes[row_img, col].axis("off")
            if row_img == 0:
                axes[row_img, col].set_title(f"d{r['class']}", fontsize=9)
            tick  = "✓" if r["correct"] else f"→{r['nearest']}"
            color = "darkgreen" if r["correct"] else "firebrick"
            axes[row_lbl, col].text(0.5, 0.65, tick, ha="center", va="center",
                fontsize=13, color=color, transform=axes[row_lbl, col].transAxes)
            axes[row_lbl, col].text(0.5, 0.2, f"cos={r['cos']:.2f}", ha="center",
                va="center", fontsize=7, transform=axes[row_lbl, col].transAxes)
            axes[row_lbl, col].axis("off")
        axes[row_img, 0].set_ylabel(f"{label}\nsynth", fontsize=8, labelpad=4)
        axes[row_lbl, 0].set_ylabel(f"nearest\n({sum(r['correct'] for r in results)}/{n})",
                                     fontsize=8, labelpad=4)

    fill(trained_res, 0, 1, "trained")
    fill(random_res,  2, 3, "random")
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.505, 0.505],
                               transform=fig.transFigure, color="gray",
                               linestyle="--", linewidth=0.9))
    fig.suptitle(f"Exp 03 — Causal generation test\nTrained: {n_tr}/10  Random: {n_rn}/10",
                 fontsize=11, y=1.01)
    save_fig(fig, "figures/mnist/exp03_causal_gen.png")

    # ── Figure 2: 2-D PCA of class means vs synthesised codes ─────────────
    lat_means = compute_class_means(model, loader)
    classes   = sorted(lat_means.keys())

    synth_stack = torch.stack([r["synth_code"] for r in trained_res])
    centroid    = synth_stack.mean(0)

    all_pts  = torch.cat([torch.stack([lat_means[c] for c in classes]), synth_stack]).numpy()
    pca      = PCA(n_components=2)
    coords   = pca.fit_transform(all_pts)
    mean_c   = coords[:len(classes)]
    synth_c  = coords[len(classes):]
    cent_2d  = pca.transform(centroid.unsqueeze(0).numpy())[0]

    dists_to_means = {c: (centroid - lat_means[c]).norm().item() for c in classes}
    nearest_c = min(dists_to_means, key=dists_to_means.get)
    print(f"\nSynthesised code centroid nearest to: d{nearest_c}")

    fig2, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(classes):
        ax.scatter(*mean_c[i],  color=cmap(c), s=150, marker="o", zorder=4)
        ax.scatter(*synth_c[i], color=cmap(c), s=80,  marker="^", zorder=3, alpha=0.8)
        ax.plot([mean_c[i, 0], synth_c[i, 0]], [mean_c[i, 1], synth_c[i, 1]],
                color=cmap(c), linewidth=0.7, alpha=0.5)
        ax.annotate(f"d{c}", mean_c[i], fontsize=8, ha="left", va="bottom", color=cmap(c))
    ax.scatter(*cent_2d, color="black", s=200, marker="*", zorder=5, label="synth centroid")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title(f"Exp 03 — Latent PCA: class means (●) vs synthesised codes (▲)\n"
                 f"Centroid nearest to d{nearest_c}  — explains limited causal accuracy",
                 fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig2.tight_layout()
    save_fig(fig2, "figures/mnist/exp03_latent_pca.png")


if __name__ == "__main__":
    main()
