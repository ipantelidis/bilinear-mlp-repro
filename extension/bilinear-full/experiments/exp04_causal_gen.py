"""
Exp 04 — Causal Generation Test

Same protocol as bilinear-decoder exp03: synthesise one image per class
import os
from decoder weights, encode back, check nearest class mean.

Key question: does having a bilinear encoder improve causal accuracy
beyond the 3/10 achieved by DecBilinearVAE alone?

Also runs on random model as baseline.

Figure saved:
    figures/mnist/exp04_causal_gen.png
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE
from train    import load_checkpoint
from analysis import (get_decoder_interaction_matrix, decompose,
                      compute_class_means, mean_lat_norm)
from visualize import save_fig

CKPT = "checkpoints/mnist/model.pt"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")


def _run_causal(model, loader):
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
            z = vecs[pos_idx[0]] * scale if len(pos_idx) else torch.zeros(model.d_latent)
            x_synth    = model.decode(z.unsqueeze(0)).squeeze(0)
            mu_back, _ = model.encode(x_synth.unsqueeze(0))
            mu_back    = mu_back.squeeze(0)
            dists   = {lbl: (mu_back - m).norm().item() for lbl, m in lat_means.items()}
            nearest = min(dists, key=dists.get)
            cos     = float(torch.cosine_similarity(mu_back.unsqueeze(0),
                                                     lat_means[c].unsqueeze(0)))
            results.append({"class": c, "nearest": nearest, "correct": nearest == c,
                             "cos": cos, "synth_img": x_synth.detach()})
    return results


def main():
    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    trained = FullBilinearVAE(); load_checkpoint(trained, CKPT); trained.eval()
    rand    = FullBilinearVAE(); rand.eval()

    tr_res = _run_causal(trained, loader)
    rn_res = _run_causal(rand, loader)

    n_tr = sum(r["correct"] for r in tr_res)
    n_rn = sum(r["correct"] for r in rn_res)

    print(f"\nFull bilinear trained: {n_tr}/10 correct  (DecBilinearVAE: 3/10)")
    for r in tr_res:
        status = "✓" if r["correct"] else f"→{r['nearest']}"
        print(f"  d{r['class']}: {status:<6}  cos={r['cos']:.3f}")
    print(f"\nRandom: {n_rn}/10 correct")

    with open("figures/mnist/exp04_results.json", "w") as f:
        json.dump({"causal_accuracy_trained": n_tr,
                   "causal_accuracy_random":  n_rn,
                   "trained_per_class": [{"class": r["class"], "nearest": r["nearest"],
                                          "correct": bool(r["correct"]), "cos": r["cos"]}
                                         for r in tr_res]}, f, indent=2)

    n = len(tr_res)
    fig, axes = plt.subplots(4, n, figsize=(2.0 * n, 8),
                              gridspec_kw={"hspace": 0.08, "wspace": 0.05})

    def fill(results, row_img, row_lbl, label):
        for col, r in enumerate(results):
            axes[row_img, col].imshow(r["synth_img"].view(28,28).numpy(), cmap="gray_r", vmin=0, vmax=1)
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

    fill(tr_res, 0, 1, "trained")
    fill(rn_res, 2, 3, "random")
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.505, 0.505],
                               transform=fig.transFigure, color="gray",
                               linestyle="--", linewidth=0.9))
    fig.suptitle(f"Exp 04 — Causal generation test (full bilinear)\n"
                 f"Trained: {n_tr}/10  Random: {n_rn}/10  (DecBilinearVAE baseline: 3/10)",
                 fontsize=11, y=1.01)
    save_fig(fig, "figures/mnist/exp04_causal_gen.png")


if __name__ == "__main__":
    main()
