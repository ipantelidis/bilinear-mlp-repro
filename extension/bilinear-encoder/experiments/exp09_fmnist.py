"""
Experiment 09 — Fashion-MNIST Extension
=========================================
Runs three analyses on the Fashion-MNIST BilinearVAE to show the encoder
analysis generalises beyond digits:

  (a) Latent dictionary        — pixel patterns per latent dimension
  (b) Maximally activating     — causal test, trained vs. random
  (c) Cross-class similarity   — does visual taxonomy appear in weights?

Direct comparison with Pearce et al. Figure 2B (FMNIST classifier eigenvectors).

Outputs: figures/fashion_mnist/exp09_latent_dictionary.png
         figures/fashion_mnist/exp09_max_activating.png
         figures/fashion_mnist/exp09_cross_class.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
sys.path.insert(0, str(Path(__file__).parent.parent))

from models    import BilinearVAE
from train     import load_checkpoint
from analysis  import interaction_matrix, decompose, class_means
from visualize import plot_heatmap

# ── Constants ────────────────────────────────────────────────────────────────
CKPT   = Path("checkpoints/fashion_mnist/model.pt")
OUTDIR = Path("figures/fashion_mnist")
DEVICE = "cpu"
CLASS_NAMES = {
    0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress",  4: "Coat",
    5: "Sandal",  6: "Shirt",   7: "Sneaker",  8: "Bag",     9: "Boot",
}
# ─────────────────────────────────────────────────────────────────────────────


# ── (a) Latent dictionary ────────────────────────────────────────────────────

@torch.no_grad()
def plot_latent_dictionary(model) -> None:
    d = model.d_latent
    fig, axes = plt.subplots(2, d, figsize=(d * 1.4, 3.4),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.05})
    for k in range(d):
        e_k = torch.zeros(d); e_k[k] = 1.0
        Q = interaction_matrix(model, e_k)
        vals, vecs = decompose(Q)
        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        neg_idx = (vals < 0).nonzero(as_tuple=True)[0]

        for row, idx in enumerate([pos_idx, neg_idx]):
            img  = vecs[idx[0]].view(28, 28).numpy() if len(idx) else np.zeros((28, 28))
            lam  = float(vals[idx[0]]) if len(idx) else 0.0
            vmax = max(abs(img).max(), 1e-8)
            axes[row, k].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, k].set_title(f"z{k}\nλ={lam:.2f}", fontsize=7)
            axes[row, k].axis("off")

    axes[0, 0].set_ylabel("Activates z_k", fontsize=8, labelpad=4)
    axes[1, 0].set_ylabel("Suppresses z_k", fontsize=8, labelpad=4)
    fig.suptitle("Exp 09a — Fashion-MNIST latent dictionary  (μ*=e_k)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = OUTDIR / "exp09_latent_dictionary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── (b) Maximally activating ─────────────────────────────────────────────────

@torch.no_grad()
def run_max_activating(model, loader) -> tuple[list, float]:
    means = class_means(model, loader, DEVICE)
    norms = torch.cat([x.view(x.size(0), -1).norm(dim=1) for x, _ in loader])
    mean_norm = norms.mean().item()

    results = []
    for c in sorted(means.keys()):
        Q = interaction_matrix(model, means[c])
        vals, vecs = decompose(Q)
        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        if not len(pos_idx):
            continue
        v1 = vecs[pos_idx[0]]
        mu_synth, _ = model.encode((v1 * mean_norm).unsqueeze(0))
        mu_synth = mu_synth.squeeze(0)
        dists   = {lbl: (mu_synth - m).norm().item() for lbl, m in means.items()}
        nearest = min(dists, key=dists.get)
        results.append({
            "class": c, "nearest": nearest, "correct": nearest == c,
            "cos_to_true": float(torch.cosine_similarity(
                               mu_synth.unsqueeze(0), means[c].unsqueeze(0))),
            "eigenvector": v1,
        })
    return results, mean_norm


def plot_max_activating(trained, random, mean_norm) -> None:
    n = len(trained)
    fig, axes = plt.subplots(4, n, figsize=(2.0 * n, 8.0),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.05})

    def fill(results, row_img, row_lbl, label):
        for col, r in enumerate(results):
            img = (r["eigenvector"] * mean_norm).view(28, 28).numpy()
            vmax = max(abs(img).max(), 1e-8)
            axes[row_img, col].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row_img, col].axis("off")
            if row_img == 0:
                axes[row_img, col].set_title(CLASS_NAMES[r["class"]][:6], fontsize=7)
            tick  = "✓" if r["correct"] else f"→{CLASS_NAMES[r['nearest']][:4]}"
            color = "darkgreen" if r["correct"] else "firebrick"
            axes[row_lbl, col].text(0.5, 0.65, tick, ha="center", va="center",
                fontsize=11, color=color, transform=axes[row_lbl, col].transAxes)
            axes[row_lbl, col].text(0.5, 0.2, f"cos={r['cos_to_true']:.2f}",
                ha="center", va="center", fontsize=7,
                transform=axes[row_lbl, col].transAxes)
            axes[row_lbl, col].axis("off")
        n_ok = sum(r["correct"] for r in results)
        axes[row_img, 0].set_ylabel(f"{label}\n({n_ok}/{n})", fontsize=8, labelpad=4)

    fill(trained, 0, 1, "trained")
    fill(random,  2, 3, "random")
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.505, 0.505],
                               transform=fig.transFigure,
                               color="gray", linestyle="--", linewidth=0.9))
    n_tr = sum(r["correct"] for r in trained)
    n_rn = sum(r["correct"] for r in random)
    fig.suptitle(f"Exp 09b — Fashion-MNIST maximally activating test\n"
                 f"Trained: {n_tr}/{n}    Random: {n_rn}/{n}",
                 fontsize=11, y=1.01)
    out = OUTDIR / "exp09_max_activating.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── (c) Cross-class similarity ───────────────────────────────────────────────

@torch.no_grad()
def plot_cross_class(model, loader) -> None:
    means  = class_means(model, loader, DEVICE)
    labels = sorted(means.keys())
    top_vecs = {}
    for c, direction in means.items():
        Q = interaction_matrix(model, direction)
        vals, vecs = decompose(Q)
        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        top_vecs[c] = vecs[pos_idx[0]] if len(pos_idx) else torch.zeros(784)

    n   = len(labels)
    mat = np.zeros((n, n))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            cos = float(torch.dot(top_vecs[a], top_vecs[b]) /
                        (top_vecs[a].norm() * top_vecs[b].norm() + 1e-10))
            mat[i, j] = abs(cos)

    tick_labels = [CLASS_NAMES[l] for l in labels]
    plot_heatmap(
        matrix=mat, row_labels=tick_labels, col_labels=tick_labels,
        title="Exp 09c — Fashion-MNIST cross-class similarity  |cos(v1_A, v1_B)|",
        out_path=OUTDIR / "exp09_cross_class.png",
    )

    print("\n  Top-5 most similar pairs:")
    pairs = [(mat[i, j], labels[i], labels[j])
             for i in range(n) for j in range(i + 1, n)]
    for sim, a, b in sorted(pairs, reverse=True)[:5]:
        print(f"    ({CLASS_NAMES[a]}, {CLASS_NAMES[b]}): {sim:.3f}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    loader = DataLoader(
        datasets.FashionMNIST("/home/v25/ippa6201/bilinear-mlp-repro/data", train=False, download=True, transform=transform),
        batch_size=512, shuffle=False)

    trained = BilinearVAE()
    load_checkpoint(trained, str(CKPT), DEVICE)

    random_model = BilinearVAE(); random_model.eval()

    print("Running Exp 09: Fashion-MNIST Extension...")

    print("\n  (a) Latent dictionary...")
    plot_latent_dictionary(trained)

    print("\n  (b) Maximally activating test...")
    tr_results, mean_norm = run_max_activating(trained, loader)
    rn_results, _         = run_max_activating(random_model, loader)
    n_tr = sum(r["correct"] for r in tr_results)
    n_rn = sum(r["correct"] for r in rn_results)
    print(f"    Trained {n_tr}/{len(tr_results)} correct  |  Random {n_rn}/{len(rn_results)} correct")
    for r in tr_results:
        status = "✓" if r["correct"] else f"→ {CLASS_NAMES[r['nearest']]}"
        print(f"    {CLASS_NAMES[r['class']]:<12}: {status}")
    plot_max_activating(tr_results, rn_results, mean_norm)

    print("\n  (c) Cross-class similarity...")
    plot_cross_class(trained, loader)

    print("Done.")
