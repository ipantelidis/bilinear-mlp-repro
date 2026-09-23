"""
Exp 15 — GANSpace/PCA control: weight-based directions are not PCA directions

The working-directory prototype (ganspace_efficiency.py) suggested PCA over
latent codes "converges to" the analytical Q_dec eigenvectors at ~0.65-0.70
best-match |cos|. That experiment had two flaws: every repeat used the
identical first-n samples of an unshuffled loader (std was exactly 0), and
best-match |cos| against up to 10 PCA directions in a 10-d latent space has a
high chance floor by pure geometry (10 directions can form a complete
orthonormal basis).

This experiment redoes the comparison correctly: true random subsets per
repeat (5 repeats per sample count) and a random-orthonormal-basis baseline
(200 draws) as the chance floor of the metric.

Result: the top PCA directions (m=1, m=3) match the analytical
(centered-target) eigenvectors no better than a random orthonormal basis at
any sample count; only the complete 10-direction basis edges marginally above
its floor (0.70 vs 0.61 +/- 0.04). GANSpace-style PCA of the posterior and
weight-based eigendecomposition of Q_dec therefore recover essentially
unrelated directions. PCA captures data-variance
structure; Q_dec eigenvectors capture generative-output structure. The two
methods are complementary, not interchangeable, and no amount of sampling
makes PCA reproduce the weight-based directions.

Outputs:
    figures/mnist/exp15_ganspace_control.png
    figures/exp15_results.json
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as Fn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models    import DecBilinearVAE
from train     import load_checkpoint
from analysis  import get_decoder_interaction_matrix, decompose
from visualize import save_fig

DATA    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")
CKPT    = "checkpoints/mnist/model.pt"
OUT_FIG = "figures/mnist/exp15_ganspace_control.png"
OUT_JSON = "figures/exp15_results.json"

SAMPLE_COUNTS = [10, 25, 50, 100, 250, 1000, 2500, 5000]
BASIS_SIZES   = [1, 3, 10]
N_REPEATS     = 5
N_FLOOR_DRAWS = 200


def top_pos_eigvec(model, p):
    vals, vecs = decompose(get_decoder_interaction_matrix(model, p))
    pos = (vals > 0).nonzero(as_tuple=True)[0]
    return vecs[pos[0]] if len(pos) else vecs[0]


def best_match(an, basis, m):
    """Mean over analytical dirs of the best |cos| against the top-m basis rows."""
    A = Fn.normalize(an, dim=1)
    B = Fn.normalize(basis[:m], dim=1)
    return (A @ B.T).abs().max(dim=1).values.mean().item()


def pca_dirs(sub):
    c = sub - sub.mean(0)
    _, _, Vh = torch.linalg.svd(c, full_matrices=False)
    return Vh   # rows ordered by explained variance


@torch.no_grad()
def main():
    model = DecBilinearVAE(); load_checkpoint(model, CKPT); model.eval()
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    loader = DataLoader(datasets.MNIST(DATA, train=False, download=True, transform=tfm),
                        batch_size=512, shuffle=False)

    codes, buckets = [], {}
    for x, y in loader:
        mu, _ = model.encode(x)
        codes.append(mu)
        for i, l in enumerate(y.tolist()):
            buckets.setdefault(l, []).append(x[i])
    codes = torch.cat(codes)
    mean_imgs = {c: torch.stack(v).mean(0) for c, v in sorted(buckets.items())}
    gmean = torch.stack(list(mean_imgs.values())).mean(0)

    an_cen = torch.stack([top_pos_eigvec(model, mean_imgs[c] - gmean) for c in range(10)])
    an_raw = torch.stack([top_pos_eigvec(model, mean_imgs[c]) for c in range(10)])

    # chance floor: best-match against random orthonormal bases
    gen = torch.Generator().manual_seed(123)
    floor = {}
    for m in BASIS_SIZES:
        draws = []
        for _ in range(N_FLOOR_DRAWS):
            Q, _ = torch.linalg.qr(torch.randn(10, 10, generator=gen))
            draws.append(best_match(an_cen, Q.T, m))
        floor[m] = {"mean": float(np.mean(draws)), "std": float(np.std(draws))}
    print("random-basis floor:",
          {m: f"{v['mean']:.3f}±{v['std']:.3f}" for m, v in floor.items()})

    curves = {m: {"mean": [], "std": []} for m in BASIS_SIZES}
    raw10 = {"mean": [], "std": []}
    for n in SAMPLE_COUNTS:
        reps = {m: [] for m in BASIS_SIZES}
        reps_raw = []
        for r in range(N_REPEATS):
            g = torch.Generator().manual_seed(1000 * r + n)
            idx = torch.randperm(len(codes), generator=g)[:n]
            V = pca_dirs(codes[idx])
            for m in BASIS_SIZES:
                reps[m].append(best_match(an_cen, V, m))
            reps_raw.append(best_match(an_raw, V, 10))
        for m in BASIS_SIZES:
            curves[m]["mean"].append(float(np.mean(reps[m])))
            curves[m]["std"].append(float(np.std(reps[m])))
        raw10["mean"].append(float(np.mean(reps_raw)))
        raw10["std"].append(float(np.std(reps_raw)))
        print(f"n={n:>5}: " + "  ".join(
            f"m={m} {curves[m]['mean'][-1]:.3f}±{curves[m]['std'][-1]:.3f}"
            for m in BASIS_SIZES))

    with open(OUT_JSON, "w") as f:
        json.dump({"sample_counts": SAMPLE_COUNTS, "basis_sizes": BASIS_SIZES,
                   "n_repeats": N_REPEATS,
                   "floor_random_orthonormal_basis": floor,
                   "pca_vs_centered_eigvecs": curves,
                   "pca_vs_raw_eigvecs_m10": raw10}, f, indent=2)
    print(f"Saved {OUT_JSON}")

    # ── figure ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    colors = {1: "#1f77b4", 3: "#2ca02c", 10: "#d62728"}
    for m in BASIS_SIZES:
        mu = np.array(curves[m]["mean"]); sd = np.array(curves[m]["std"])
        ax.semilogx(SAMPLE_COUNTS, mu, "o-", color=colors[m], lw=1.8,
                    label=f"PCA top-{m} vs analytical (centered)")
        ax.fill_between(SAMPLE_COUNTS, mu - sd, mu + sd, color=colors[m], alpha=0.15)
        fm, fs = floor[m]["mean"], floor[m]["std"]
        ax.axhline(fm, color=colors[m], ls="--", lw=1.2, alpha=0.7)
        ax.fill_between([SAMPLE_COUNTS[0], SAMPLE_COUNTS[-1]],
                        fm - fs, fm + fs, color=colors[m], alpha=0.08)
        ax.text(SAMPLE_COUNTS[-1] * 1.05, fm, f"floor m={m}", fontsize=7,
                color=colors[m], va="center")
    ax.set_xlabel("Samples used for PCA")
    ax.set_ylabel("Mean best-match |cos| with analytical directions")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Exp 15 — PCA of latent codes does not recover the weight-based directions:\n"
                 "top PCA components match no better than a random basis (dashed floors);\n"
                 "only the complete 10-dir basis edges marginally above its floor", fontsize=10)
    save_fig(fig, OUT_FIG)


if __name__ == "__main__":
    main()
