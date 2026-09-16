"""
Exp 20 — Truncated Q_dec reconstruction with centered targets

Exp16 found no low-rank plateau when truncating Q_dec built from RAW
class-mean targets.  Exp18 showed that CENTERED targets (p*_c minus the
global mean image) make weight-based synthesis nearly perfect.  Does
centering also reveal low-rank structure that raw targets hid?

Definitions (careful):
  - centered target  t_c = mean_img_c − global_mean   (as in exp18)
  - Q = Q_dec(t_c), eigendecomposed via analysis.decompose (|λ| order)
  - two truncation rules:
      absk : exp16 protocol unchanged — z_k = Σ_{i<k} sign(λ_i)·(scale/√k)·v_i
             over the top-k |λ| eigvecs
      posk : top-k POSITIVE-λ eigvecs, λ-weighted, renormalised to `scale`
             (k=1 posk is exactly exp18's synthesis direction)
  - error metric: MSE(decode(z_k), mean_img_c).  MSE against the centered
    target itself is ill-defined (decode output lives in [0,1], the target
    is signed); comparing the centered versions (decode−g) vs (mean_c−g)
    is algebraically identical to this MSE, so only one number is needed.
  - causal check: encode(decode(z_k)) → cosine-nearest latent class mean
    (exp18's landing rule)

Outputs:
    figures/mnist/exp20_truncation_curves.png
    figures/mnist/exp20_truncation_images.png
    figures/mnist/exp20_results.json
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

DATA   = "/home/v25/ippa6201/bilinear-mlp-repro/data"
CKPTS  = [("main", "checkpoints/mnist/model.pt")] + \
         [(f"seed{s}", f"checkpoints/mnist/seeds/seed{s}.pt") for s in range(5)]
K_VALS = [1, 2, 3, 5, 10]
COMBOS = [("raw", "absk"), ("raw", "posk"), ("centered", "absk"), ("centered", "posk")]


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


def trunc_z(model, target, k, scale, rule):
    vals, vecs = decompose(get_decoder_interaction_matrix(model, target))
    z = torch.zeros(model.d_latent)
    if rule == "absk":                       # exp16 protocol, unchanged
        n = min(k, len(vals))
        for i in range(n):
            z = z + vecs[i] * (1.0 if vals[i] >= 0 else -1.0) * scale / (n ** 0.5)
    else:                                    # posk: top-k positive eigvecs
        pos = (vals > 0).nonzero(as_tuple=True)[0][:k]
        if len(pos):
            w = vals[pos] / vals[pos].sum()
            for i, wi in zip(pos.tolist(), w.tolist()):
                z = z + vecs[i] * wi
            z = z / z.norm() * scale
    return z, vals


@torch.no_grad()
def evaluate_ckpt(model, loader, mean_imgs, gmean, classes):
    """MSE / causal accuracy per (target kind, rule, k) + spectral mass."""
    scale = mean_lat_norm(model, loader)
    lat_means = compute_class_means(model, loader)
    M = torch.stack([lat_means[c] for c in classes])

    def landing(img):
        mu, _ = model.encode(img.unsqueeze(0))
        return Fn.cosine_similarity(mu, M).argmax().item()

    out = {"combo": {}, "mass": {}, "imgs": {}}
    for kind, rule in COMBOS:
        mse = {k: [] for k in K_VALS}
        causal = {k: 0 for k in K_VALS}
        mass = []
        for c in classes:
            t = mean_imgs[c] if kind == "raw" else mean_imgs[c] - gmean
            for k in K_VALS:
                z, vals = trunc_z(model, t, k, scale, rule)
                img = model.decode(z.unsqueeze(0)).squeeze(0)
                mse[k].append(((img - mean_imgs[c]) ** 2).mean().item())
                causal[k] += int(landing(img) == c)
                if rule == "posk":
                    out["imgs"][(kind, c, k)] = img
            a = vals.abs()
            mass.append((a.cumsum(0) / a.sum()).tolist())
        out["combo"][(kind, rule)] = {
            "mse_by_k": {k: float(np.mean(v)) for k, v in mse.items()},
            "causal_by_k": dict(causal)}
        out["mass"][kind] = np.array(mass).mean(0).tolist()

    full_mse = []
    for c in classes:
        r, _, _ = model(mean_imgs[c].unsqueeze(0))
        full_mse.append(((r.squeeze(0) - mean_imgs[c]) ** 2).mean().item())
    out["full_mse"] = float(np.mean(full_mse))
    return out


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


@torch.no_grad()
def main():
    loader    = build_loader()
    mean_imgs = class_mean_images(loader)
    gmean     = torch.stack(list(mean_imgs.values())).mean(0)
    classes   = sorted(mean_imgs)

    per_ckpt = {}
    main_out = None
    for name, path in CKPTS:
        if not Path(path).exists():
            print(f"{name}: {path} missing — skipped"); continue
        model = FullBilinearVAE(); load_checkpoint(model, path); model.eval()
        out = evaluate_ckpt(model, loader, mean_imgs, gmean, classes)
        per_ckpt[name] = out
        if name == "main":
            main_out = out
        line = "  ".join(f"{kind[:4]}/{rule} k=1: mse={out['combo'][(kind,rule)]['mse_by_k'][1]:.4f} "
                         f"causal={out['combo'][(kind,rule)]['causal_by_k'][1]}/10"
                         for kind, rule in COMBOS)
        print(f"{name}: {line} | full={out['full_mse']:.4f}")

    # Seed-level summary of the headline (k=1, posk) numbers
    summary = {}
    for kind in ["raw", "centered"]:
        for stat, fn in [("mse_k1", lambda o: o["combo"][(kind, "posk")]["mse_by_k"][1]),
                         ("causal_k1", lambda o: o["combo"][(kind, "posk")]["causal_by_k"][1])]:
            vals = [fn(per_ckpt[n]) for n in per_ckpt]
            summary[f"{kind}_posk_{stat}_mean"] = float(np.mean(vals))
            summary[f"{kind}_posk_{stat}_std"]  = float(np.std(vals))
    summary["full_mse_mean"] = float(np.mean([per_ckpt[n]["full_mse"] for n in per_ckpt]))
    print(f"\n{len(per_ckpt)}-ckpt summary (posk, k=1):")
    print(f"  MSE    raw {summary['raw_posk_mse_k1_mean']:.4f}±{summary['raw_posk_mse_k1_std']:.4f} "
          f"-> centered {summary['centered_posk_mse_k1_mean']:.4f}±{summary['centered_posk_mse_k1_std']:.4f} "
          f"(full decode {summary['full_mse_mean']:.4f})")
    print(f"  causal raw {summary['raw_posk_causal_k1_mean']:.1f}±{summary['raw_posk_causal_k1_std']:.1f} "
          f"-> centered {summary['centered_posk_causal_k1_mean']:.1f}±{summary['centered_posk_causal_k1_std']:.1f}")

    results = {
        "k_vals": K_VALS,
        "per_ckpt": {n: {
            "full_mse": per_ckpt[n]["full_mse"],
            "combos": {f"{kind}_{rule}": {
                "mse_by_k":    {str(k): v for k, v in per_ckpt[n]["combo"][(kind, rule)]["mse_by_k"].items()},
                "causal_by_k": {str(k): v for k, v in per_ckpt[n]["combo"][(kind, rule)]["causal_by_k"].items()}}
                for kind, rule in COMBOS},
            "mean_cum_spectral_mass": per_ckpt[n]["mass"]}
            for n in per_ckpt},
        "summary": summary,
    }
    with open("figures/mnist/exp20_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Figure 1: curves (main checkpoint) ───────────────────
    styles = {("raw", "absk"):      dict(color="gray",       ls="--", marker="s"),
              ("raw", "posk"):      dict(color="steelblue",  ls="-",  marker="o"),
              ("centered", "absk"): dict(color="darkkhaki",  ls="--", marker="s"),
              ("centered", "posk"): dict(color="darkorange", ls="-",  marker="o")}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for kind, rule in COMBOS:
        d = main_out["combo"][(kind, rule)]
        axes[0].plot(K_VALS, [d["mse_by_k"][k] for k in K_VALS],
                     label=f"{kind}/{rule}", **styles[(kind, rule)])
        axes[1].plot(K_VALS, [d["causal_by_k"][k] for k in K_VALS],
                     label=f"{kind}/{rule}", **styles[(kind, rule)])
    axes[0].axhline(main_out["full_mse"], color="seagreen", ls=":",
                    label=f"full decode ({main_out['full_mse']:.4f})")
    axes[0].set_xlabel("top-k eigenvectors"); axes[0].set_ylabel("MSE to class-mean image")
    axes[0].set_title("Truncated synthesis error vs k", fontsize=10)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("top-k eigenvectors"); axes[1].set_ylabel("Causal accuracy (/10)")
    axes[1].set_ylim(0, 10)
    axes[1].set_title("Causal landing vs k", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    ks = np.arange(1, len(main_out["mass"]["raw"]) + 1)
    axes[2].plot(ks, main_out["mass"]["raw"],      "s--", color="gray",       label="raw target")
    axes[2].plot(ks, main_out["mass"]["centered"], "o-",  color="darkorange", label="centered target")
    axes[2].set_xlabel("top-k eigenvalues"); axes[2].set_ylabel("cumulative |λ| mass")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("Q_dec spectral concentration", fontsize=10)
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
    fig.suptitle("Exp 20 — Truncated Q_dec reconstruction with centered targets (main ckpt)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp20_truncation_curves.png")

    # ── Figure 2: image grid (posk rule, main ckpt) ──────────
    rows = [("Mean image", None, None)] + \
           [(f"raw k={k}", "raw", k) for k in [1, 10]] + \
           [(f"centered k={k}", "centered", k) for k in [1, 2, 10]]
    fig2, axes2 = plt.subplots(len(rows), 10, figsize=(15, 1.55 * len(rows)),
                               gridspec_kw={"hspace": 0.1, "wspace": 0.04})
    for ci, c in enumerate(classes):
        for ri, (lbl, kind, k) in enumerate(rows):
            img = mean_imgs[c] if kind is None else main_out["imgs"][(kind, c, k)]
            axes2[ri, ci].imshow(img.view(28, 28), cmap="gray_r", vmin=0, vmax=1)
            _clean(axes2[ri, ci])
            if ri == 0:
                axes2[ri, ci].set_title(f"c{c}", fontsize=9)
            if ci == 0:
                axes2[ri, ci].set_ylabel(lbl, fontsize=8, labelpad=6)
    fig2.suptitle("Exp 20 — Truncated synthesis (positive-eigvec rule): k=1 already saturates",
                  fontsize=11, y=0.995)
    save_fig(fig2, "figures/mnist/exp20_truncation_images.png")


if __name__ == "__main__":
    main()
