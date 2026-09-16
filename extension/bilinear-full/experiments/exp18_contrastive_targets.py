"""
Exp 18 — Contrastive (centered) targets on the full bilinear VAE

Companion to bilinear-decoder exp13. Q_dec is linear in the target
direction p*, and raw class-mean images overlap heavily, so the
"near-universal generative direction" partly inherits that overlap.
Here the synthesis / cross-class / causal analyses are repeated with
centered targets p*_c − mean_c'(p*_c') on the FullBilinearVAE, across
the main checkpoint and all 5 seeds, giving seed-level uncertainty for
the causal-accuracy improvement.

Outputs:
    figures/mnist/exp18_synthesis_centered.png
    figures/mnist/exp18_seed_summary.png
    figures/mnist/exp18_results.json
"""

import json
import torch
import torch.nn.functional as Fn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models    import FullBilinearVAE
from train     import load_checkpoint
from analysis  import (get_decoder_interaction_matrix, decompose,
                       compute_class_means, mean_lat_norm)
from visualize import save_fig

DATA  = "/home/v25/ippa6201/bilinear-mlp-repro/data"
CKPTS = [("main", "checkpoints/mnist/model.pt")] + \
        [(f"seed{s}", f"checkpoints/mnist/seeds/seed{s}.pt") for s in range(5)]


def build_loader():
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    return DataLoader(datasets.MNIST(DATA, train=False, download=False, transform=tfm),
                      batch_size=512, shuffle=False)


def class_mean_images(loader):
    buckets = {}
    for x, y in loader:
        for i, l in enumerate(y.tolist()):
            buckets.setdefault(l, []).append(x[i])
    return {c: torch.stack(v).mean(0) for c, v in sorted(buckets.items())}


def top_pos_eigvec(model, p):
    vals, vecs = decompose(get_decoder_interaction_matrix(model, p))
    pos = (vals > 0).nonzero(as_tuple=True)[0]
    return vecs[pos[0]] if len(pos) else vecs[0]


def offdiag_mean(vecs):
    M = torch.stack(vecs)
    M = M / M.norm(dim=1, keepdim=True)
    S = (M @ M.T).abs()
    return S[~torch.eye(len(vecs), dtype=bool)].mean().item()


@torch.no_grad()
def evaluate(model, loader, mean_imgs, targets):
    """Cross-class eigvec similarity + causal accuracy for given targets."""
    lat_means = compute_class_means(model, loader)
    scale     = mean_lat_norm(model, loader)
    M = torch.stack([lat_means[c] for c in sorted(lat_means)])
    vecs, ok, landing, synth = [], 0, [], []
    for c in sorted(targets):
        v = top_pos_eigvec(model, targets[c])
        vecs.append(v)
        img = model.decode((v * scale).unsqueeze(0))
        synth.append(img.squeeze(0))
        mu, _ = model.encode(img)
        pred = Fn.cosine_similarity(mu, M).argmax().item()
        landing.append(pred); ok += int(pred == c)
    return offdiag_mean(vecs), ok, landing, synth


@torch.no_grad()
def main():
    loader    = build_loader()
    mean_imgs = class_mean_images(loader)
    gmean     = torch.stack(list(mean_imgs.values())).mean(0)
    raw       = {c: mean_imgs[c] for c in mean_imgs}
    centered  = {c: mean_imgs[c] - gmean for c in mean_imgs}

    results = {"per_ckpt": {}}
    for name, path in CKPTS:
        if not Path(path).exists():
            print(f"{name}: {path} missing — skipped"); continue
        model = FullBilinearVAE(); load_checkpoint(model, path); model.eval()
        sim_r, ok_r, land_r, synth_r = evaluate(model, loader, mean_imgs, raw)
        sim_c, ok_c, land_c, synth_c = evaluate(model, loader, mean_imgs, centered)
        results["per_ckpt"][name] = {
            "eigvec_similarity_raw": sim_r, "eigvec_similarity_centered": sim_c,
            "causal_raw": ok_r, "causal_centered": ok_c,
            "landing_centered": land_c,
        }
        print(f"{name}: sim {sim_r:.3f}->{sim_c:.3f} | causal {ok_r}/10 -> {ok_c}/10")
        if name == "main":
            main_synth_r, main_synth_c, main_land_c = synth_r, synth_c, land_c

    per = results["per_ckpt"]
    for key in ["eigvec_similarity_raw", "eigvec_similarity_centered",
                "causal_raw", "causal_centered"]:
        vals = [per[n][key] for n in per]
        results[f"{key}_mean"] = float(np.mean(vals))
        results[f"{key}_std"]  = float(np.std(vals))
    with open("figures/mnist/exp18_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\ncausal: raw {results['causal_raw_mean']:.1f}±{results['causal_raw_std']:.1f} "
          f"-> centered {results['causal_centered_mean']:.1f}±{results['causal_centered_std']:.1f}")

    # ── synthesis grid (main checkpoint) ─────────────────────
    def clean(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig, axes = plt.subplots(3, 10, figsize=(18, 6.2),
                             gridspec_kw={"hspace": 0.3, "wspace": 0.04})
    for c in range(10):
        axes[0, c].imshow(mean_imgs[c].view(28, 28), cmap="gray_r", vmin=0, vmax=1)
        axes[0, c].set_title(f"c{c}", fontsize=9); clean(axes[0, c])
        axes[1, c].imshow(main_synth_r[c].view(28, 28), cmap="gray_r", vmin=0, vmax=1)
        clean(axes[1, c])
        axes[2, c].imshow(main_synth_c[c].view(28, 28), cmap="gray_r", vmin=0, vmax=1)
        ok = main_land_c[c] == c
        axes[2, c].set_xlabel("✓" if ok else f"→{main_land_c[c]}", fontsize=9,
                              color="green" if ok else "red")
        clean(axes[2, c])
    axes[0, 0].set_ylabel("Mean image",      fontsize=9, labelpad=6)
    axes[1, 0].set_ylabel("Raw target",      fontsize=9, labelpad=6)
    axes[2, 0].set_ylabel("Centered target", fontsize=9, labelpad=6)
    fig.suptitle("Exp 18 — FullBilinearVAE synthesis, raw vs centered targets (main ckpt)",
                 fontsize=11, y=1.02)
    save_fig(fig, "figures/mnist/exp18_synthesis_centered.png")

    # ── seed summary ─────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    names = list(per)
    x = np.arange(len(names))
    axes2[0].bar(x - 0.2, [per[n]["causal_raw"] for n in names], 0.4, label="raw")
    axes2[0].bar(x + 0.2, [per[n]["causal_centered"] for n in names], 0.4, label="centered")
    axes2[0].set_xticks(x); axes2[0].set_xticklabels(names, rotation=45, fontsize=8)
    axes2[0].set_ylabel("Causal accuracy (/10)"); axes2[0].set_ylim(0, 10)
    axes2[0].legend(fontsize=8); axes2[0].grid(True, alpha=0.3, axis="y")
    axes2[1].bar(x - 0.2, [per[n]["eigvec_similarity_raw"] for n in names], 0.4, label="raw")
    axes2[1].bar(x + 0.2, [per[n]["eigvec_similarity_centered"] for n in names], 0.4, label="centered")
    axes2[1].set_xticks(x); axes2[1].set_xticklabels(names, rotation=45, fontsize=8)
    axes2[1].set_ylabel("Cross-class eigvec cos"); axes2[1].set_ylim(0, 1)
    axes2[1].legend(fontsize=8); axes2[1].grid(True, alpha=0.3, axis="y")
    fig2.suptitle("Exp 18 — Raw vs centered targets across seeds (FullBilinearVAE)",
                  fontsize=11)
    fig2.tight_layout()
    save_fig(fig2, "figures/mnist/exp18_seed_summary.png")


if __name__ == "__main__":
    main()
