"""
Experiment 12 — KMNIST Extension (dataset generality)
=====================================================
Runs the established encoder battery on a BilinearVAE trained on KMNIST
(10 Hiragana classes, much higher intra-class variability than digits):

  (a) Latent dictionary        — pixel patterns per latent dimension
  (b) Maximally activating     — causal test, trained vs. 10-seeded random baseline
  (c) Cross-class similarity   — |cos| of top eigenvectors of class-mean directions

Same protocol as exp05 (MNIST) / exp09 (FMNIST), including the 10-seeded-init
random baseline convention.

Outputs: figures/kmnist/exp12_latent_dictionary.png
         figures/kmnist/exp12_max_activating.png
         figures/kmnist/exp12_cross_class.png
         figures/kmnist/exp12_results.json
"""

import json
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
CKPT   = Path("checkpoints/kmnist/model.pt")
OUTDIR = Path("figures/kmnist")
DEVICE = "cpu"
# KMNIST class index → romanised Hiragana character (torchvision order)
CLASS_NAMES = {0: "o", 1: "ki", 2: "su", 3: "tsu", 4: "na",
               5: "ha", 6: "ma", 7: "ya", 8: "re", 9: "wo"}
# ─────────────────────────────────────────────────────────────────────────────


def _clean(ax):
    """Remove ticks and spines but keep the axis alive so labels render."""
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


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
            _clean(axes[row, k])

    axes[0, 0].set_ylabel("Activates z_k", fontsize=8, labelpad=4)
    axes[1, 0].set_ylabel("Suppresses z_k", fontsize=8, labelpad=4)
    fig.suptitle("Exp 12a — KMNIST latent dictionary  (μ*=e_k)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = OUTDIR / "exp12_latent_dictionary.png"
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
            _clean(axes[row_img, col])
            if row_img == 0:
                axes[row_img, col].set_title(CLASS_NAMES[r["class"]], fontsize=9)
            tick  = "✓" if r["correct"] else f"→{CLASS_NAMES[r['nearest']]}"
            color = "darkgreen" if r["correct"] else "firebrick"
            axes[row_lbl, col].text(0.5, 0.65, tick, ha="center", va="center",
                fontsize=11, color=color, transform=axes[row_lbl, col].transAxes)
            axes[row_lbl, col].text(0.5, 0.2, f"cos={r['cos_to_true']:.2f}",
                ha="center", va="center", fontsize=7,
                transform=axes[row_lbl, col].transAxes)
            _clean(axes[row_lbl, col])
        n_ok = sum(r["correct"] for r in results)
        axes[row_img, 0].set_ylabel(f"{label}\neigenvector", fontsize=8, labelpad=4)
        axes[row_lbl, 0].set_ylabel(f"nearest class\n({n_ok}/{n})", fontsize=8, labelpad=4)

    fill(trained, 0, 1, "trained")
    fill(random,  2, 3, "random (seed 0)")
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.505, 0.505],
                               transform=fig.transFigure,
                               color="gray", linestyle="--", linewidth=0.9))
    n_tr = sum(r["correct"] for r in trained)
    n_rn = sum(r["correct"] for r in random)
    fig.suptitle(f"Exp 12b — KMNIST maximally activating test\n"
                 f"Trained: {n_tr}/{n}    Random (seed 0): {n_rn}/{n}",
                 fontsize=11, y=1.01)
    out = OUTDIR / "exp12_max_activating.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ── (c) Cross-class similarity ───────────────────────────────────────────────

@torch.no_grad()
def cross_class(model, loader):
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
        title="Exp 12c — KMNIST cross-class similarity  |cos(v1_A, v1_B)|",
        out_path=OUTDIR / "exp12_cross_class.png",
    )

    pairs = sorted(([float(mat[i, j]), labels[i], labels[j]]
                    for i in range(n) for j in range(i + 1, n)), reverse=True)
    print("\n  Top-5 most similar pairs:")
    for sim, a, b in pairs[:5]:
        print(f"    ({CLASS_NAMES[a]}, {CLASS_NAMES[b]}): {sim:.3f}")
    return labels, mat, pairs


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    transform = transforms.Compose([transforms.ToTensor()])
    loader = DataLoader(
        datasets.KMNIST("/home/v25/ippa6201/bilinear-mlp-repro/data",
                        train=False, download=True, transform=transform),
        batch_size=512, shuffle=False)

    trained = BilinearVAE()
    load_checkpoint(trained, str(CKPT), DEVICE)

    torch.manual_seed(0)
    random_model = BilinearVAE(); random_model.eval()

    print("Running Exp 12: KMNIST Extension...")

    print("\n  (a) Latent dictionary...")
    plot_latent_dictionary(trained)

    print("\n  (b) Maximally activating test...")
    tr_results, mean_norm = run_max_activating(trained, loader)
    rn_results, _         = run_max_activating(random_model, loader)
    rn_counts = []
    for s in range(10):
        torch.manual_seed(s)
        rm = BilinearVAE(); rm.eval()
        rs, _ = run_max_activating(rm, loader)
        rn_counts.append(sum(r["correct"] for r in rs))
    print(f"    Random baseline over 10 inits: "
          f"{np.mean(rn_counts):.1f} ± {np.std(rn_counts):.1f} /10")
    n_tr = sum(r["correct"] for r in tr_results)
    n_rn = sum(r["correct"] for r in rn_results)
    print(f"    Trained {n_tr}/{len(tr_results)} correct  |  "
          f"Random (seed 0) {n_rn}/{len(rn_results)} correct")
    for r in tr_results:
        status = "✓" if r["correct"] else f"→ {CLASS_NAMES[r['nearest']]}"
        print(f"    {CLASS_NAMES[r['class']]:<4}: {status}  cos={r['cos_to_true']:.3f}")
    plot_max_activating(tr_results, rn_results, mean_norm)

    print("\n  (c) Cross-class similarity...")
    cc_labels, cc_mat, cc_pairs = cross_class(trained, loader)
    off = [float(cc_mat[i, j]) for i in range(len(cc_labels))
           for j in range(len(cc_labels)) if i != j]
    offdiag_mean = sum(off) / len(off)
    print(f"    Off-diagonal mean: {offdiag_mean:.3f}")

    def _rows(rs):
        return [{"class": int(r["class"]), "correct": bool(r["correct"]),
                 "nearest": int(r["nearest"]),
                 "cos_to_true": float(r["cos_to_true"])} for r in rs]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / "exp12_results.json", "w") as f:
        json.dump({"class_names": {int(k): v for k, v in CLASS_NAMES.items()},
                   "trained_correct": int(n_tr),
                   "random_correct_seed0": int(n_rn),
                   "random_correct_10init_mean": float(np.mean(rn_counts)),
                   "random_correct_10init_std": float(np.std(rn_counts)),
                   "random_correct_10init_counts": [int(c) for c in rn_counts],
                   "mean_image_norm": float(mean_norm),
                   "trained": _rows(tr_results),
                   "random": _rows(rn_results),
                   "crossclass_matrix": [[float(v) for v in row] for row in cc_mat],
                   "crossclass_offdiag_mean": float(offdiag_mean),
                   "crossclass_top5_pairs": [
                       {"pair": [CLASS_NAMES[a], CLASS_NAMES[b]],
                        "classes": [int(a), int(b)], "abs_cos": float(sim)}
                       for sim, a, b in cc_pairs[:5]]}, f, indent=2)
    print(f"  Saved → {OUTDIR / 'exp12_results.json'}")

    print("Done.")
