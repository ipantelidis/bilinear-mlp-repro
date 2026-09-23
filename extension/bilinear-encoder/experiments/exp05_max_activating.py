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
DATA   = str(Path(__file__).resolve().parents[3] / "data")
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


def _clean(ax):
    """Hide ticks and frame but keep axis labels (axis('off') erases labels)."""
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plot(trained: list, random: list, mean_norm: float) -> None:
    n = len(trained)
    fig = plt.figure(figsize=(6.2, 1.88))
    gs = fig.add_gridspec(4, n, height_ratios=[1, 0.36, 1, 0.36],
                          hspace=0.06, wspace=0.06,
                          left=0.075, right=0.995, top=0.92, bottom=0.005)
    axes = gs.subplots()

    def fill(results, row_img, row_lbl, label):
        for col, r in enumerate(results):
            img  = (r["eigenvector"] * mean_norm).view(28, 28).numpy()
            vmax = max(abs(img).max(), 1e-8)
            axes[row_img, col].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            _clean(axes[row_img, col])
            if row_img == 0:
                axes[row_img, col].set_title(f"{r['class']}", fontsize=7.5, pad=2)

            tick  = "✓" if r["correct"] else f"→{r['nearest']}"
            color = "darkgreen" if r["correct"] else "firebrick"
            axes[row_lbl, col].text(0.5, 0.66, tick, ha="center", va="center",
                fontsize=8.5, color=color, fontweight="bold",
                transform=axes[row_lbl, col].transAxes)
            axes[row_lbl, col].text(0.5, 0.08, f"cos {r['cos_to_true']:.2f}",
                ha="center", va="center", fontsize=5.6, color="0.35",
                transform=axes[row_lbl, col].transAxes)
            _clean(axes[row_lbl, col])

        n_ok = sum(r["correct"] for r in results)
        axes[row_img, 0].set_ylabel(label, fontsize=7.5, fontweight="bold",
                                    labelpad=3)
        axes[row_lbl, 0].set_ylabel(f"{n_ok}/{n}", fontsize=6.5, labelpad=3)

    fill(trained, 0, 1, "trained")
    fill(random,  2, 3, "untrained")

    # thin divider between the trained block and the untrained control block
    y_mid = 0.5 * (axes[1, 0].get_position().y0 +
                   axes[2, 0].get_position().y1)
    fig.add_artist(plt.Line2D([0.01, 0.995], [y_mid, y_mid],
                              transform=fig.transFigure,
                              color="0.6", linestyle="--", linewidth=0.7))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved → {OUT}")


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False, transform=transform),
        batch_size=512, shuffle=False)

    trained_model = BilinearVAE()
    load_checkpoint(trained_model, str(CKPT), DEVICE)

    # Untrained control: seed it so the figure is reproducible, and measure
    # the baseline over 10 seeded inits (a single init is high-variance).
    torch.manual_seed(0)
    random_model = BilinearVAE()
    random_model.eval()

    print("Running Exp 05: Maximally Activating Input Test...")
    trained_results, mean_norm = run_test(trained_model, loader)
    random_results,  _         = run_test(random_model,  loader)

    rn_counts = []
    for s in range(10):
        torch.manual_seed(s)
        rm = BilinearVAE(); rm.eval()
        rs, _ = run_test(rm, loader)
        rn_counts.append(sum(r["correct"] for r in rs))

    n_tr = sum(r["correct"] for r in trained_results)
    n_rn = sum(r["correct"] for r in random_results)
    import numpy as _np
    print(f"  Random baseline over 10 inits: "
          f"{_np.mean(rn_counts):.1f} ± {_np.std(rn_counts):.1f} /10")
    print(f"\n  Trained: {n_tr}/{len(trained_results)} correct")
    for r in trained_results:
        status = "✓" if r["correct"] else f"→ {r['nearest']}"
        print(f"    digit {r['class']}: {status}  cos={r['cos_to_true']:.3f}")
    print(f"\n  Random:  {n_rn}/{len(random_results)} correct")

    import json
    def _rows(rs):
        return [{"class": int(r["class"]), "correct": bool(r["correct"]),
                 "nearest": int(r["nearest"]),
                 "cos_to_true": float(r["cos_to_true"])} for r in rs]
    with open("figures/mnist/exp05_results.json", "w") as f:
        json.dump({"trained_correct": int(n_tr),
                   "random_correct_seed0": int(n_rn),
                   "random_correct_10init_mean": float(_np.mean(rn_counts)),
                   "random_correct_10init_std": float(_np.std(rn_counts)),
                   "random_correct_10init_counts": [int(c) for c in rn_counts],
                   "trained": _rows(trained_results),
                   "random": _rows(random_results)}, f, indent=2)
    print("  Saved → figures/mnist/exp05_results.json")

    plot(trained_results, random_results, mean_norm)
    print("Done.")
