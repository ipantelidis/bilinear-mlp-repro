"""
Experiment 10 — Adversarial Mask
==================================
Direct analog of Pearce et al. Section 4.4, applied to the VAE encoder.

The pseudoinverse of the joint eigenvector matrix gives masks that activate
exactly one eigenvector without cross-activating others across different classes.
Adding a scaled mask to a test image from a different class should redirect the
encoder toward the target class.

    V  = top-K positive eigenvectors from all classes  (10K, 784)
    V+ = pseudoinverse(V)                              (784, 10K)
    mask_c = V+[:, c*K]   — activates class c's first eigenvector specifically

Measure: fraction of images from other classes that switch nearest-class to c.

Outputs: figures/mnist/exp10_mask_grid.png
         figures/mnist/exp10_steering_rate.png
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
sys.path.insert(0, str(Path(__file__).parent.parent))

from models   import BilinearVAE
from train    import load_checkpoint
from analysis import interaction_matrix, decompose, class_means

# ── Constants ────────────────────────────────────────────────────────────────
CKPT      = Path("checkpoints/mnist/model.pt")
OUT_GRID  = Path("figures/mnist/exp10_mask_grid.png")
OUT_STEER = Path("figures/mnist/exp10_steering_rate.png")
DEVICE    = "cpu"
K         = 3       # top-K eigenvectors per class used to build V
SCALES    = [0.5, 1.0, 2.0, 4.0]   # σ in units of mean image norm
N_TEST    = 500     # test images per source class
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def build_masks(model, means: dict, k: int = K) -> tuple[torch.Tensor, dict, dict]:
    """
    Stack top-k positive eigenvectors from all classes into V (10k, 784),
    compute the pseudoinverse, and return one mask per class.

    Returns:
        V      : (10k, 784) eigenvector matrix
        masks  : {class → (784,) adversarial mask}
        eigvecs: {class → (784,) top eigenvector, for comparison}
    """
    rows, class_idx, eigvecs = [], {}, {}
    for c in sorted(means.keys()):
        Q = interaction_matrix(model, means[c])
        vals, vecs = decompose(Q)
        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        start = len(rows)
        for i in range(k):
            rows.append(vecs[pos_idx[i]] if i < len(pos_idx) else torch.zeros(784))
        class_idx[c] = list(range(start, start + k))
        eigvecs[c]   = rows[start]   # top eigenvector for this class

    V      = torch.stack(rows)                 # (10k, 784)
    V_pinv = torch.linalg.pinv(V)             # (784, 10k)
    masks  = {c: V_pinv[:, class_idx[c][0]]
              for c in sorted(means.keys())}

    return V, masks, eigvecs


@torch.no_grad()
def steering_experiment(
    model, loader, masks, means, mean_norm
) -> tuple[dict, dict]:
    """
    For each target class c, add scaled mask to images from all other classes.
    Returns (results_adv, results_random) — both are {scale → {class → rate}}.
    """
    classes = sorted(means.keys())

    # Collect N_TEST images per class
    buckets: dict[int, list] = {}
    for x, y in loader:
        x = x.view(x.size(0), -1)
        for i, lbl in enumerate(y.tolist()):
            if len(buckets.get(lbl, [])) < N_TEST:
                buckets.setdefault(lbl, []).append(x[i])
        if all(len(buckets.get(c, [])) >= N_TEST for c in classes):
            break
    buckets = {c: torch.stack(imgs[:N_TEST]) for c, imgs in buckets.items()}

    adv_results  = {s: {} for s in SCALES}
    rand_results = {s: {} for s in SCALES}

    for target in classes:
        mask_unit = masks[target] / (masks[target].norm() + 1e-8)
        rand_unit = torch.randn_like(mask_unit)
        rand_unit = rand_unit / (rand_unit.norm() + 1e-8)

        # Images from all classes except target
        sources = torch.cat([imgs for c, imgs in buckets.items() if c != target])

        for sigma in SCALES:
            x_adv  = sources + sigma * mean_norm * mask_unit
            x_rand = sources + sigma * mean_norm * rand_unit

            def nearest_class(xs):
                results = []
                for i in range(0, len(xs), 512):
                    mu, _ = model.encode(xs[i:i+512])
                    for mu_i in mu:
                        d = {c: (mu_i - m).norm().item() for c, m in means.items()}
                        results.append(min(d, key=d.get))
                return results

            near_adv  = nearest_class(x_adv)
            near_rand = nearest_class(x_rand)

            adv_results[sigma][target]  = sum(n == target for n in near_adv)  / len(near_adv)
            rand_results[sigma][target] = sum(n == target for n in near_rand) / len(near_rand)

    return adv_results, rand_results


def plot_mask_grid(model, means, masks, eigvecs) -> None:
    classes = sorted(means.keys())
    n = len(classes)
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4.0),
                             gridspec_kw={"hspace": 0.06, "wspace": 0.04})
    for col, c in enumerate(classes):
        for row, (vec, label) in enumerate([(eigvecs[c], "Eigenvector"),
                                            (masks[c],   "Mask (V⁺)")]):
            img  = vec.view(28, 28).numpy()
            vmax = max(abs(img).max(), 1e-8)
            axes[row, col].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"d{c}", fontsize=9)
        axes[0, 0].set_ylabel("Eigenvector\n(v₁)", fontsize=8, labelpad=4)
        axes[1, 0].set_ylabel("Pseudo-\ninverse mask", fontsize=8, labelpad=4)
    fig.suptitle("Exp 10 — Top eigenvectors and their pseudoinverse masks",
                 fontsize=11, y=1.01)
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_GRID, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {OUT_GRID}")


def plot_steering(adv_results, rand_results) -> None:
    chance    = 1.0 / 9.0
    adv_means  = [np.mean(list(adv_results[s].values()))  for s in SCALES]
    rand_means = [np.mean(list(rand_results[s].values())) for s in SCALES]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(SCALES, adv_means,  "o-",  color="steelblue", linewidth=2,
            markersize=6, label="Adversarial mask (pseudoinverse)")
    ax.plot(SCALES, rand_means, "s--", color="indianred", linewidth=2,
            markersize=6, label="Random mask (control)")
    ax.axhline(chance, color="gray", linestyle=":", linewidth=1,
               label=f"Chance (1/9 ≈ {chance:.2f})")
    ax.set_xlabel("Mask scale σ (× mean image norm)", fontsize=11)
    ax.set_ylabel("Steering rate toward target class", fontsize=11)
    ax.set_title("Exp 10 — Adversarial steering rate vs. mask scale", fontsize=12)
    ax.set_ylim(0, 1.0); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    OUT_STEER.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEER, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {OUT_STEER}")


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    loader = DataLoader(
        datasets.MNIST("/home/v25/ippa6201/bilinear-mlp-repro/data", train=False, download=False, transform=transform),
        batch_size=512, shuffle=False)

    model = BilinearVAE()
    load_checkpoint(model, str(CKPT), DEVICE)
    print("Running Exp 10: Adversarial Mask...")

    means = class_means(model, loader, DEVICE)
    norms = torch.cat([x.view(x.size(0), -1).norm(dim=1) for x, _ in loader])
    mean_norm = norms.mean().item()

    V, masks, eigvecs = build_masks(model, means)
    print(f"\n  V shape: {tuple(V.shape)}")

    print("\n  cos(v1, mask) per class:")
    for c in sorted(means.keys()):
        cos = float(torch.dot(eigvecs[c], masks[c]) /
                    (eigvecs[c].norm() * masks[c].norm() + 1e-8))
        print(f"    digit {c}: {cos:.3f}")

    print(f"\n  Running steering experiment (scales={SCALES})...")
    adv, rand = steering_experiment(model, loader, masks, means, mean_norm)

    print(f"\n  {'Scale':<8} {'Adv':>8} {'Random':>8} {'Ratio':>8}")
    print("  " + "-" * 36)
    for s in SCALES:
        a = np.mean(list(adv[s].values()))
        r = np.mean(list(rand[s].values()))
        print(f"  σ={s:<5}  {a:>7.3f}  {r:>7.3f}  {a/r:>7.2f}×")

    plot_mask_grid(model, means, masks, eigvecs)
    plot_steering(adv, rand)
    print("Done.")
