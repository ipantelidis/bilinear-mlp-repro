"""
Exp 18 — KMNIST: raw vs centered targets (dataset generality)
=============================================================
Repeats the exp13 contrastive-target battery on a DecBilinearVAE trained on
KMNIST (10 Hiragana classes, high intra-class variability):

    - target overlap, raw vs centered (p*_c vs p*_c − mean_c' p*_c')
    - top-eigenvector cross-class |cos|, raw vs centered
      (+ random-init model control for both)
    - causal generation (synthesize → encode → nearest class-mean),
      raw vs centered (+ random-model control)
    - synthesis quality: MSE(decoded top eigvec, class mean image) vs the
      actual VAE reconstruction MSE (decode mean latent)

Same protocol as exp13 (MNIST / FMNIST).

Outputs:
    figures/kmnist/exp18_synthesis_centered.png
    figures/kmnist/exp18_crossclass_centered.png
    figures/exp18_results.json
"""

import json

import torch
import torch.nn.functional as Fn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models    import DecBilinearVAE
from train     import load_checkpoint
from analysis  import (get_decoder_interaction_matrix, decompose,
                       compute_class_means, mean_lat_norm)
from visualize import save_fig

DATA    = "/home/v25/ippa6201/bilinear-mlp-repro/data"
CKPT    = "checkpoints/kmnist/model.pt"
RESULTS = "figures/exp18_results.json"
# KMNIST class index → romanised Hiragana character (torchvision order)
KMNIST_NAMES = {0: "o", 1: "ki", 2: "su", 3: "tsu", 4: "na",
                5: "ha", 6: "ma", 7: "ya", 8: "re", 9: "wo"}


def build_loader():
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    return DataLoader(datasets.KMNIST(DATA, train=False, download=True,
                                      transform=tfm),
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
    return S, S[~torch.eye(len(vecs), dtype=bool)].mean().item()


@torch.no_grad()
def causal_accuracy(model, targets, scale, class_mu):
    """Synthesize from each target's top eigenvector, encode, classify."""
    M = torch.stack([class_mu[c] for c in sorted(class_mu)])
    ok, landing = 0, []
    for c in sorted(targets):
        v = top_pos_eigvec(model, targets[c]) * scale
        img = model.decode(v.unsqueeze(0))
        mu, _ = model.encode(img)
        pred = Fn.cosine_similarity(mu, M).argmax().item()
        landing.append(pred)
        ok += int(pred == c)
    return ok, landing


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    name = "kmnist"
    print(f"=== {name} ===")
    model = DecBilinearVAE(); load_checkpoint(model, CKPT); model.eval()
    loader = build_loader()

    mean_imgs   = class_mean_images(loader)
    global_mean = torch.stack(list(mean_imgs.values())).mean(0)
    raw      = {c: mean_imgs[c] for c in mean_imgs}
    centered = {c: mean_imgs[c] - global_mean for c in mean_imgs}

    scale    = mean_lat_norm(model, loader)
    class_mu = compute_class_means(model, loader)

    # target overlap and eigenvector overlap, raw vs centered
    _, t_raw = offdiag_mean([raw[c] for c in raw])
    _, t_cen = offdiag_mean([centered[c] for c in centered])
    S_raw, e_raw = offdiag_mean([top_pos_eigvec(model, raw[c]) for c in raw])
    S_cen, e_cen = offdiag_mean([top_pos_eigvec(model, centered[c]) for c in centered])

    # synthesis quality: MSE(decoded top eigvec, class mean image) for raw vs
    # centered targets, against actual VAE reconstruction (decode mean latent)
    mse_raw, mse_cen, mse_rec = [], [], []
    for c in sorted(mean_imgs):
        img_r = model.decode((top_pos_eigvec(model, raw[c]) * scale).unsqueeze(0)).squeeze(0)
        img_c = model.decode((top_pos_eigvec(model, centered[c]) * scale).unsqueeze(0)).squeeze(0)
        img_v = model.decode(class_mu[c].unsqueeze(0)).squeeze(0)
        mse_raw.append(((img_r - mean_imgs[c]) ** 2).mean().item())
        mse_cen.append(((img_c - mean_imgs[c]) ** 2).mean().item())
        mse_rec.append(((img_v - mean_imgs[c]) ** 2).mean().item())
    penalty_raw = float(torch.tensor([s / r for s, r in zip(mse_raw, mse_rec)]).mean())
    penalty_cen = float(torch.tensor([s / r for s, r in zip(mse_cen, mse_rec)]).mean())

    # causal generation, raw vs centered; random-model control
    ok_raw, land_raw = causal_accuracy(model, raw, scale, class_mu)
    ok_cen, land_cen = causal_accuracy(model, centered, scale, class_mu)
    torch.manual_seed(0)
    rand_model = DecBilinearVAE(); rand_model.eval()
    _, e_raw_rand = offdiag_mean(
        [top_pos_eigvec(rand_model, raw[c]) for c in raw])
    _, e_cen_rand = offdiag_mean(
        [top_pos_eigvec(rand_model, centered[c]) for c in centered])
    ok_rand, _ = causal_accuracy(rand_model, centered, scale, class_mu)

    results = {name: {
        "class_names": {int(k): v for k, v in KMNIST_NAMES.items()},
        "target_similarity_raw": t_raw,
        "target_similarity_centered": t_cen,
        "eigvec_similarity_raw": e_raw,
        "eigvec_similarity_centered": e_cen,
        "eigvec_similarity_raw_random_model": e_raw_rand,
        "eigvec_similarity_centered_random_model": e_cen_rand,
        "causal_raw": ok_raw, "causal_raw_landing": land_raw,
        "causal_centered": ok_cen, "causal_centered_landing": land_cen,
        "causal_centered_random_model": ok_rand,
        "synth_mse_raw_per_class":      mse_raw,
        "synth_mse_centered_per_class": mse_cen,
        "recon_mse_per_class":          mse_rec,
        "synth_mse_raw_mean":      float(torch.tensor(mse_raw).mean()),
        "synth_mse_centered_mean": float(torch.tensor(mse_cen).mean()),
        "recon_mse_mean":          float(torch.tensor(mse_rec).mean()),
        "mse_penalty_raw_vs_recon":      penalty_raw,
        "mse_penalty_centered_vs_recon": penalty_cen,
        "mean_latent_norm": float(scale),
    }}
    print(f"  targets: raw {t_raw:.3f} -> centered {t_cen:.3f}")
    print(f"  eigvecs: raw {e_raw:.3f} -> centered {e_cen:.3f} "
          f"(random model: raw {e_raw_rand:.3f}, centered {e_cen_rand:.3f})")
    print(f"  causal:  raw {ok_raw}/10 -> centered {ok_cen}/10 "
          f"(random model, centered: {ok_rand}/10)")
    print(f"  synth MSE: raw {torch.tensor(mse_raw).mean():.4f} -> "
          f"centered {torch.tensor(mse_cen).mean():.4f} "
          f"(VAE recon {torch.tensor(mse_rec).mean():.4f}; "
          f"penalty raw {penalty_raw:.1f}x -> centered {penalty_cen:.1f}x)")

    # ── figure 1: synthesis grid, mean / raw / centered ──────────────────────
    fig, axes = plt.subplots(3, 10, figsize=(18, 6.2),
                             gridspec_kw={"hspace": 0.3, "wspace": 0.04})
    for c in sorted(raw):
        axes[0, c].imshow(mean_imgs[c].view(28, 28), cmap="gray_r", vmin=0, vmax=1)
        axes[0, c].set_title(f"{KMNIST_NAMES[c]} (c{c})", fontsize=9)
        _clean(axes[0, c])
        img_r = model.decode((top_pos_eigvec(model, raw[c]) * scale).unsqueeze(0))
        axes[1, c].imshow(img_r.view(28, 28), cmap="gray_r", vmin=0, vmax=1)
        ok_r = land_raw[c] == c
        axes[1, c].set_xlabel("✓" if ok_r else f"→{KMNIST_NAMES[land_raw[c]]}",
                              fontsize=9, color="green" if ok_r else "red")
        _clean(axes[1, c])
        img_c = model.decode((top_pos_eigvec(model, centered[c]) * scale).unsqueeze(0))
        axes[2, c].imshow(img_c.view(28, 28), cmap="gray_r", vmin=0, vmax=1)
        ok_c = land_cen[c] == c
        axes[2, c].set_xlabel("✓" if ok_c else f"→{KMNIST_NAMES[land_cen[c]]}",
                              fontsize=9, color="green" if ok_c else "red")
        _clean(axes[2, c])
    axes[0, 0].set_ylabel("Mean image",      fontsize=9, labelpad=6)
    axes[1, 0].set_ylabel("Raw target",      fontsize=9, labelpad=6)
    axes[2, 0].set_ylabel("Centered target", fontsize=9, labelpad=6)
    fig.suptitle(f"Exp 18 — KMNIST synthesis with raw vs centered targets\n"
                 f"causal: raw {ok_raw}/10 → centered {ok_cen}/10   |   "
                 f"MSE to class mean: raw {torch.tensor(mse_raw).mean():.3f} → "
                 f"centered {torch.tensor(mse_cen).mean():.3f} "
                 f"(VAE recon {torch.tensor(mse_rec).mean():.3f})",
                 fontsize=11, y=1.02)
    save_fig(fig, f"figures/{name}/exp18_synthesis_centered.png")

    # ── figure 2: cross-class similarity heatmaps ────────────────────────────
    labels = [KMNIST_NAMES[c] for c in sorted(raw)]
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, S, ttl in [(axes2[0], S_raw, f"Raw targets  mean={e_raw:.3f}"),
                       (axes2[1], S_cen, f"Centered targets  mean={e_cen:.3f}")]:
        im = ax.imshow(S, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if S[i, j] > 0.6 else "black")
        fig2.colorbar(im, ax=ax, fraction=0.046)
    fig2.suptitle("Exp 18 — KMNIST top-eigenvector cross-class similarity:\n"
                  "raw vs centered targets", fontsize=11)
    save_fig(fig2, f"figures/{name}/exp18_crossclass_centered.png")

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()
