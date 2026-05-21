"""
Experiment 05 — Maximally Activating Input Test
=================================================
The top positive eigenvector v1_c is by construction the unit-norm input that
most activates class c.  We test whether this is causally true: scale v1_c to
the mean image norm and check if encoding it lands nearest to class c.

Run on both the trained model and a random initialisation as a control.

Output: figures/mnist/exp05_max_activating.png
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
CKPT   = Path("checkpoints/mnist/model.pt")
OUT    = Path("figures/mnist/exp05_max_activating.png")
DEVICE = "cpu"
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_test(model, loader) -> tuple[list, float]:
    """Return (results list, mean image norm)."""
    means = class_means(model, loader, DEVICE)

    # Mean L2 norm of real images (sets the natural scale for the eigenvector)
    norms = torch.cat([x.view(x.size(0), -1).norm(dim=1)
                       for x, _ in loader])
    mean_norm = norms.mean().item()

    results = []
    for c in sorted(means.keys()):
        Q = interaction_matrix(model, means[c])
        vals, vecs = decompose(Q)

        pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
        if not len(pos_idx):
            continue

        v1 = vecs[pos_idx[0]]
        x_synth = (v1 * mean_norm).unsqueeze(0)          # (1, 784)

        mu_synth, _ = model.encode(x_synth)
        mu_synth = mu_synth.squeeze(0)

        dists   = {lbl: (mu_synth - m).norm().item() for lbl, m in means.items()}
        nearest = min(dists, key=dists.get)

        results.append({
            "class":       c,
            "nearest":     nearest,
            "correct":     nearest == c,
            "cos_to_true": float(torch.cosine_similarity(
                               mu_synth.unsqueeze(0), means[c].unsqueeze(0))),
            "eigenvector": v1,
        })
    return results, mean_norm


def plot(trained: list, random: list, mean_norm: float) -> None:
    n = len(trained)
    fig, axes = plt.subplots(4, n, figsize=(2.0 * n, 8.0),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.05})

    def fill(results, row_img, row_lbl, label):
        for col, r in enumerate(results):
            img  = (r["eigenvector"] * mean_norm).view(28, 28).numpy()
            vmax = max(abs(img).max(), 1e-8)
            axes[row_img, col].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[row_img, col].axis("off")
            if row_img == 0:
                axes[row_img, col].set_title(f"d{r['class']}", fontsize=9)

            tick  = "✓" if r["correct"] else f"→{r['nearest']}"
            color = "darkgreen" if r["correct"] else "firebrick"
            axes[row_lbl, col].text(0.5, 0.65, tick, ha="center", va="center",
                fontsize=13, color=color, transform=axes[row_lbl, col].transAxes)
            axes[row_lbl, col].text(0.5, 0.2, f"cos={r['cos_to_true']:.2f}",
                ha="center", va="center", fontsize=7,
                transform=axes[row_lbl, col].transAxes)
            axes[row_lbl, col].axis("off")

        n_ok = sum(r["correct"] for r in results)
        axes[row_img, 0].set_ylabel(f"{label}\neigenvector", fontsize=8, labelpad=4)
        axes[row_lbl, 0].set_ylabel(f"nearest class\n({n_ok}/{n})", fontsize=8, labelpad=4)

    fill(trained, 0, 1, "trained")
    fill(random,  2, 3, "random")

    fig.add_artist(plt.Line2D([0.02, 0.98], [0.505, 0.505],
                               transform=fig.transFigure,
                               color="gray", linestyle="--", linewidth=0.9))
    n_tr = sum(r["correct"] for r in trained)
    n_rn = sum(r["correct"] for r in random)
    fig.suptitle(f"Exp 05 — Maximally activating input test\n"
                 f"Trained: {n_tr}/{n} correct    Random: {n_rn}/{n} correct",
                 fontsize=11, y=1.01)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {OUT}")


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    loader = DataLoader(
        datasets.MNIST("/home/v25/ippa6201/bilinear-mlp-repro/data", train=False, download=False, transform=transform),
        batch_size=512, shuffle=False)

    trained_model = BilinearVAE()
    load_checkpoint(trained_model, str(CKPT), DEVICE)

    random_model = BilinearVAE()   # untrained control
    random_model.eval()

    print("Running Exp 05: Maximally Activating Input Test...")
    trained_results, mean_norm = run_test(trained_model, loader)
    random_results,  _         = run_test(random_model,  loader)

    n_tr = sum(r["correct"] for r in trained_results)
    n_rn = sum(r["correct"] for r in random_results)
    print(f"\n  Trained: {n_tr}/{len(trained_results)} correct")
    for r in trained_results:
        status = "✓" if r["correct"] else f"→ {r['nearest']}"
        print(f"    digit {r['class']}: {status}  cos={r['cos_to_true']:.3f}")
    print(f"\n  Random:  {n_rn}/{len(random_results)} correct")

    plot(trained_results, random_results, mean_norm)
    print("Done.")
