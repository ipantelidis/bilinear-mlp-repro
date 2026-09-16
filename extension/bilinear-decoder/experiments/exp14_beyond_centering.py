"""
Exp 14 — Beyond centering: alternative contrastive target constructions

Exp 13 showed that centering the target (p*_c − global mean) rescues
weight-based synthesis (causal 3/10 → 9/10 on MNIST). This experiment asks
whether anything smarter beats simple centering. Three alternatives:

  (a) pairwise contrast   p* = mean_c − mean_{c'}, c' = nearest-confusable
      class (highest pixel-space cosine to mean_c);
  (b) deflation           Q̄ = Q(global mean); project its top +eigenvector
      (the "universal direction") out of Q_c and take the leading remaining
      +eigenvector (depth sweep k = 1..6 in |λ| order also reported);
  (c) generalized direction  maximize zᵀQ_c z relative to zᵀQ̄₊z where
      Q̄₊ = |Q̄| (eigendecomposition with absolute eigenvalues) + ε·λ_max·I,
      via the generalized symmetric eigenproblem (ε sweep reported).

Result (MNIST): NOTHING beats centering (9/10 causal, cross-class 0.429,
synthesis MSE 0.040). Pairwise contrast is the runner-up (7/10, cross-class
0.365 — actually lower overlap, but worse causal accuracy and MSE). Deflation
and the generalized eigenproblem fail (≤ 4/10). The explanation is exact:
Q_dec is LINEAR in p*, so Q(p_c − g) = Q_c − Q̄ — centering already performs
full-matrix deflation of the shared component (verified to ~1e-4). Q̄ is far
from rank-1 (top-|λ| eigenvalue share 0.39), so projecting out one or a few
of its eigenvectors removes only a fraction of the shared structure while
distorting the eigenspace. Note also that the one-vs-rest contrast
mean_c − mean_{c'≠c} is exactly (10/9)·(mean_c − g): direction-wise it IS
centering, so pairwise-style contrasts cannot generalize past it.

Outputs:
    figures/mnist/exp14_method_grid.png     synthesis grid, one row per method
    figures/mnist/exp14_method_summary.png  causal / cross-class / MSE bars
    figures/exp14_results.json              all quantitative results
"""

import json
import numpy as np
import scipy.linalg
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
CKPT    = "checkpoints/mnist/model.pt"
RESULTS = "figures/exp14_results.json"
EXP13   = "figures/exp13_results.json"   # random-model reference, if present

EPS_REL       = [0.01, 0.1, 1.0]   # ε values for (c), relative to max |λ| of Q̄
EPS_MAIN      = 0.01               # ε shown in the figures
DEFLATE_DEPTH = 6                  # depth sweep for (b), |λ| order


def build_loader():
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    return DataLoader(datasets.MNIST(DATA, train=False, download=True, transform=tfm),
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


def top_pos_of(Q):
    vals, vecs = decompose(Q)
    pos = (vals > 0).nonzero(as_tuple=True)[0]
    return vecs[pos[0]] if len(pos) else vecs[0]


def offdiag_mean(vecs):
    M = torch.stack(vecs)
    M = M / M.norm(dim=1, keepdim=True)
    S = (M @ M.T).abs()
    return S[~torch.eye(len(vecs), dtype=bool)].mean().item()


@torch.no_grad()
def evaluate(model, dirs, scale, class_mu, mean_imgs):
    """dirs: {class → latent direction}. Synthesize, encode, classify, MSE."""
    M = torch.stack([class_mu[c] for c in sorted(class_mu)])
    ok, landing, mses, imgs = 0, [], [], {}
    for c in sorted(dirs):
        v = dirs[c] / dirs[c].norm() * scale
        img = model.decode(v.unsqueeze(0))
        mu, _ = model.encode(img)
        pred = Fn.cosine_similarity(mu, M).argmax().item()
        landing.append(pred)
        ok += int(pred == c)
        mses.append(((img.squeeze(0) - mean_imgs[c]) ** 2).mean().item())
        imgs[c] = img.squeeze(0)
    return {"causal": ok, "landing": landing,
            "crossclass": offdiag_mean([dirs[c] for c in sorted(dirs)]),
            "synth_mse_per_class": mses,
            "synth_mse_mean": float(np.mean(mses))}, imgs


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


@torch.no_grad()
def main():
    model = DecBilinearVAE(); load_checkpoint(model, CKPT); model.eval()
    loader = build_loader()
    mean_imgs   = class_mean_images(loader)
    global_mean = torch.stack(list(mean_imgs.values())).mean(0)
    scale       = mean_lat_norm(model, loader)
    class_mu    = compute_class_means(model, loader)
    classes     = sorted(mean_imgs)

    # nearest-confusable class per class (pixel-space cosine of mean images)
    T  = torch.stack([mean_imgs[c] for c in classes])
    Tn = T / T.norm(dim=1, keepdim=True)
    C  = Tn @ Tn.T
    nearest = {}
    for c in classes:
        row = C[c].clone(); row[c] = -2
        nearest[c] = int(row.argmax())

    # shared-component matrix Q̄ and its spectrum
    Qbar = get_decoder_interaction_matrix(model, global_mean)
    bvals, bvecs = decompose(Qbar)
    bpos = (bvals > 0).nonzero(as_tuple=True)[0]
    qbar_top_share = float(bvals.abs()[0] / bvals.abs().sum())

    # linearity check: Q(p−g) = Q(p) − Q(g)  ⇒ centering ≡ full-matrix deflation
    lin_dev = max(float((get_decoder_interaction_matrix(model, mean_imgs[c] - global_mean)
                         - (get_decoder_interaction_matrix(model, mean_imgs[c]) - Qbar))
                        .abs().max()) for c in classes)

    Qc = {c: get_decoder_interaction_matrix(model, mean_imgs[c]) for c in classes}

    # ── directions per method ────────────────────────────────────────────
    methods = {
        "raw":      {c: top_pos_eigvec(model, mean_imgs[c]) for c in classes},
        "centered": {c: top_pos_eigvec(model, mean_imgs[c] - global_mean)
                     for c in classes},
        "pairwise": {c: top_pos_eigvec(model, mean_imgs[c] - mean_imgs[nearest[c]])
                     for c in classes},
    }

    # (b) deflation of Q̄'s top +eigvec (the universal direction)
    u = bvecs[bpos[0]] if len(bpos) else bvecs[0]
    P = torch.eye(model.d_latent) - torch.outer(u, u)
    methods["deflate"] = {c: top_pos_of(P @ Qc[c] @ P) for c in classes}

    # (c) generalized eigenproblem for each ε
    Qbar_abs = (bvecs.T * bvals.abs()) @ bvecs        # V|Λ|Vᵀ (rows are eigvecs)
    lam_max  = float(bvals.abs().max())
    for eps in EPS_REL:
        Qplus = Qbar_abs + eps * lam_max * torch.eye(model.d_latent)
        d = {}
        for c in classes:
            _, V = scipy.linalg.eigh(Qc[c].numpy(), Qplus.numpy())
            v = torch.from_numpy(V[:, -1]).float()
            d[c] = v / v.norm()
        methods[f"generalized_eps{eps}"] = d

    # ── evaluate everything ──────────────────────────────────────────────
    results = {"nearest_confusable": nearest,
               "linearity_max_abs_dev": lin_dev,
               "qbar_eigenvalues": [float(v) for v in bvals],
               "qbar_top_abs_eigval_share": qbar_top_share,
               "methods": {}}
    imgs_by_method = {}
    print(f"\n{'method':<22} {'causal':>7} {'crossclass':>11} {'synthMSE':>9}")
    for name, dirs in methods.items():
        m, imgs = evaluate(model, dirs, scale, class_mu, mean_imgs)
        results["methods"][name] = m
        imgs_by_method[name] = imgs
        print(f"{name:<22} {m['causal']:>5}/10 {m['crossclass']:>11.3f} "
              f"{m['synth_mse_mean']:>9.4f}")

    # deflation depth sweep (|λ| order), numbers only
    sweep = {}
    for k in range(1, DEFLATE_DEPTH + 1):
        Pk = torch.eye(model.d_latent)
        for i in range(k):
            Pk = Pk - torch.outer(bvecs[i], bvecs[i])
        dirs = {c: top_pos_of(Pk @ Qc[c] @ Pk) for c in classes}
        m, _ = evaluate(model, dirs, scale, class_mu, mean_imgs)
        sweep[k] = {"causal": m["causal"], "crossclass": m["crossclass"]}
        print(f"deflate depth {k} (|λ| order): causal {m['causal']}/10  "
              f"crossclass {m['crossclass']:.3f}")
    results["deflation_depth_sweep"] = sweep

    # random-model reference from exp13, if available
    rand_ref = None
    try:
        with open(EXP13) as f:
            rand_ref = json.load(f)["mnist"]["eigvec_similarity_centered_random_model"]
    except (FileNotFoundError, KeyError):
        pass
    results["random_model_crossclass_ref_exp13"] = rand_ref

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {RESULTS}")

    # ── figure 1: synthesis grid, one row per method ─────────────────────
    grid_rows = [("Mean image", None),
                 ("Raw (exp09)", "raw"),
                 ("Centered (exp13)", "centered"),
                 ("Pairwise contrast", "pairwise"),
                 ("Deflation (top +eig)", "deflate"),
                 (f"Generalized (ε={EPS_MAIN})", f"generalized_eps{EPS_MAIN}")]
    fig, axes = plt.subplots(len(grid_rows), 10,
                             figsize=(18, 2.05 * len(grid_rows)),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.04})
    for r, (label, key) in enumerate(grid_rows):
        for c in classes:
            ax = axes[r, c]
            if key is None:
                ax.imshow(mean_imgs[c].view(28, 28), cmap="gray_r", vmin=0, vmax=1)
                ax.set_title(f"c{c}", fontsize=9)
            else:
                ax.imshow(imgs_by_method[key][c].view(28, 28),
                          cmap="gray_r", vmin=0, vmax=1)
                pred = results["methods"][key]["landing"][c]
                ok = pred == c
                ax.set_xlabel("✓" if ok else f"→{pred}", fontsize=9,
                              color="green" if ok else "red")
            _clean(ax)
        score = ("" if key is None
                 else f"\n{results['methods'][key]['causal']}/10")
        axes[r, 0].set_ylabel(label + score, fontsize=8, labelpad=6)
    fig.suptitle("Exp 14 — Alternative contrastive constructions vs simple centering "
                 "(MNIST)\nnothing beats centering; pairwise contrast is the runner-up",
                 fontsize=11, y=1.0)
    save_fig(fig, "figures/mnist/exp14_method_grid.png")

    # ── figure 2: summary bars ───────────────────────────────────────────
    names  = ["raw", "centered", "pairwise", "deflate", f"generalized_eps{EPS_MAIN}"]
    labels = ["raw", "centered", "pairwise", "deflate", f"gen. ε={EPS_MAIN}"]
    colors = ["#9e9e9e", "#2e7d32", "#1565c0", "#c62828", "#6a1b9a"]
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("causal", "Causal accuracy (/10)", "{:d}"),
              ("crossclass", "Cross-class |cos| of directions", "{:.3f}"),
              ("synth_mse_mean", "Synthesis MSE to class mean", "{:.3f}")]
    for ax, (metric, title, fmt) in zip(axes2, panels):
        vals = [results["methods"][n][metric] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt.format(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        if metric == "causal":
            ax.set_ylim(0, 10.8)
        if metric == "crossclass" and rand_ref is not None:
            ax.axhline(rand_ref, color="gray", linestyle="--", linewidth=1)
            ax.text(len(names) - 0.4, rand_ref + 0.01,
                    f"random model {rand_ref:.3f}", fontsize=7,
                    ha="right", color="gray")
    fig2.suptitle("Exp 14 — Method summary: centering is sufficient "
                  "(Q(p−g) = Q(p) − Q(g) exactly — centering already deflates "
                  "the full shared matrix)", fontsize=10, y=1.02)
    fig2.tight_layout()
    save_fig(fig2, "figures/mnist/exp14_method_summary.png")


if __name__ == "__main__":
    main()
