"""
Exp 16 — Latent-dimension sweep: do the rank-1 finding and the centering fix
survive when d_latent >> number of classes?

At d_latent=10 the number of latent dimensions coincides with the number of
MNIST classes, so both headline results — (i) centering the target rescues
weight-based synthesis (cross-class cos 0.84 -> 0.43, causal 3/10 -> 9/10)
and (ii) the centered interaction matrix is effectively rank-1 (top positive
eigenvector saturates the truncation curve) — could in principle be artifacts
of that coincidence.  This experiment repeats the full exp13-style battery at
d_latent in {10, 20, 32}, 3 seeds each, same training protocol as the
existing checkpoints (30 epochs, lr 1e-3, wd 0.01, beta 1, input noise 0.3,
BCE, batch 128, AdamW + cosine LR, best-test-loss checkpoint).

The d=10 group reuses checkpoints/mnist/seeds/seed{0,1,2}.pt (trained with
this exact protocol) copied to the latent_sweep naming; d=20 and d=32 seeds
are trained fresh if their checkpoints are missing.

Per (d_latent, seed):
  - raw / centered top-eigvec cross-class |cos| (off-diagonal mean)
  - raw / centered causal accuracy (decode top +eigvec scaled to the mean
    latent norm -> encode -> nearest class-mean by cosine)
  - random-init model control on centered targets (exp13 protocol)
  - truncation curve: z_k = lambda-weighted combination of the top-k POSITIVE
    centered eigenvectors, renormalized to the mean latent norm; MSE to the
    class mean image and causal accuracy at each k in {1,2,3,5,10,min(d,20)}
  - centered top-positive-eigenvalue share (rank-1 concentration) and number
    of positive eigenvalues
  - test reconstruction MSE (training sanity)

Outputs:
    figures/mnist/exp16_latent_sweep.png    4-panel sweep summary
    figures/mnist/exp16_synthesis_grid.png  centered synthesis per d (seed 0)
    figures/exp16_results.json              per-model + per-dim mean/std
"""

import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as Fn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models    import DecBilinearVAE
from train     import train, load_checkpoint
from analysis  import (get_decoder_interaction_matrix, decompose,
                       compute_class_means, mean_lat_norm)
from visualize import save_fig

DATA      = str(Path(__file__).resolve().parents[3] / "data")
CKPT_DIR  = Path("checkpoints/mnist/latent_sweep")
SEEDS_DIR = Path("checkpoints/mnist/seeds")
RESULTS   = "figures/exp16_results.json"
DIMS      = [10, 20, 32]
SEEDS     = [0, 1, 2]
K_BASE    = [1, 2, 3, 5, 10]
DIM_COLORS = {10: "tab:blue", 20: "tab:orange", 32: "tab:green"}


# ── data helpers ─────────────────────────────────────────────────────────────

def build_test_loader():
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    return DataLoader(datasets.MNIST(DATA, train=False, download=True, transform=tfm),
                      batch_size=512, shuffle=False)


def build_train_loaders(seed):
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    train_set = datasets.MNIST(DATA, train=True,  download=True, transform=tfm)
    test_set  = datasets.MNIST(DATA, train=False, download=True, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(train_set, batch_size=128, shuffle=True, generator=g,
                       num_workers=2, pin_memory=True),
            DataLoader(test_set, batch_size=512, shuffle=False,
                       num_workers=2, pin_memory=True))


def class_mean_images(loader):
    buckets = {}
    for x, y in loader:
        for i, l in enumerate(y.tolist()):
            buckets.setdefault(l, []).append(x[i])
    return {c: torch.stack(v).mean(0) for c, v in sorted(buckets.items())}


# ── training (only runs if a checkpoint is missing) ─────────────────────────

def ensure_checkpoints():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for d in DIMS:
        for s in SEEDS:
            out = CKPT_DIR / f"d{d}_seed{s}.pt"
            if out.exists():
                continue
            if d == 10 and (SEEDS_DIR / f"seed{s}.pt").exists():
                shutil.copy(SEEDS_DIR / f"seed{s}.pt", out)
                print(f"[exp16] reused existing d=10 seed{s} (same protocol) -> {out}")
                continue
            print(f"[exp16] training d_latent={d} seed={s} on {device}")
            torch.manual_seed(s)
            model = DecBilinearVAE(d_latent=d)
            tr_loader, te_loader = build_train_loaders(s)
            train(model, tr_loader, te_loader,
                  epochs=30, lr=1e-3, weight_decay=0.01, beta=1.0,
                  noise_std=0.3, device=device,
                  checkpoint_dir=str(CKPT_DIR), run_name=f"d{d}_seed{s}")


# ── battery primitives ───────────────────────────────────────────────────────

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
    """Decode each target's top +eigvec, re-encode, classify by nearest class mean."""
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


def trunc_z(model, target, k, scale):
    """Lambda-weighted combination of the top-k positive eigvecs, |z| = scale."""
    vals, vecs = decompose(get_decoder_interaction_matrix(model, target))
    pos = (vals > 0).nonzero(as_tuple=True)[0][:k]
    z = torch.zeros(model.d_latent)
    if len(pos):
        w = vals[pos] / vals[pos].sum()
        for i, wi in zip(pos.tolist(), w.tolist()):
            z = z + vecs[i] * wi
        z = z / z.norm() * scale
    return z


@torch.no_grad()
def test_recon_mse(model, loader):
    tot, n = 0.0, 0
    for x, _ in loader:
        recon, _, _ = model(x)
        tot += ((recon - x) ** 2).mean(dim=1).sum().item()
        n += x.size(0)
    return tot / n


# ── per-model battery ────────────────────────────────────────────────────────

@torch.no_grad()
def run_battery(d, seed, loader, mean_imgs, global_mean):
    model = DecBilinearVAE(d_latent=d)
    load_checkpoint(model, CKPT_DIR / f"d{d}_seed{seed}.pt")
    model.eval()

    classes  = sorted(mean_imgs)
    raw      = {c: mean_imgs[c] for c in classes}
    centered = {c: mean_imgs[c] - global_mean for c in classes}

    scale    = mean_lat_norm(model, loader)
    class_mu = compute_class_means(model, loader)
    M        = torch.stack([class_mu[c] for c in classes])

    _, e_raw = offdiag_mean([top_pos_eigvec(model, raw[c]) for c in classes])
    _, e_cen = offdiag_mean([top_pos_eigvec(model, centered[c]) for c in classes])
    ok_raw, land_raw = causal_accuracy(model, raw, scale, class_mu)
    ok_cen, land_cen = causal_accuracy(model, centered, scale, class_mu)

    # random-init control (exp13 protocol: trained model's scale + class means)
    torch.manual_seed(1000 + 10 * d + seed)
    rand_model = DecBilinearVAE(d_latent=d)
    rand_model.eval()
    _, e_rand = offdiag_mean([top_pos_eigvec(rand_model, centered[c]) for c in classes])
    ok_rand, _ = causal_accuracy(rand_model, centered, scale, class_mu)

    # truncation curve (centered targets)
    k_vals = sorted(set([k for k in K_BASE if k <= d] + [min(d, 20)]))
    trunc = {}
    for k in k_vals:
        mses, okk = [], 0
        for c in classes:
            z = trunc_z(model, centered[c], k, scale)
            img = model.decode(z.unsqueeze(0)).squeeze(0)
            mses.append(((img - mean_imgs[c]) ** 2).mean().item())
            mu, _ = model.encode(img.unsqueeze(0))
            okk += int(Fn.cosine_similarity(mu, M).argmax().item() == c)
        trunc[str(k)] = {"mse": float(torch.tensor(mses).mean()), "causal": okk}

    # rank-1 concentration of the centered spectrum
    shares, n_pos = [], []
    for c in classes:
        vals, _ = decompose(get_decoder_interaction_matrix(model, centered[c]))
        pos = vals[vals > 0]
        shares.append((pos.max() / pos.sum()).item() if len(pos) else 0.0)
        n_pos.append(int((vals > 0).sum()))

    # reference floor: VAE decode of the class-mean latent vs class mean image
    rec_cm = []
    for c in classes:
        img = model.decode(class_mu[c].unsqueeze(0)).squeeze(0)
        rec_cm.append(((img - mean_imgs[c]) ** 2).mean().item())

    return {
        "eigvec_cos_raw": e_raw,
        "eigvec_cos_centered": e_cen,
        "eigvec_cos_centered_random": e_rand,
        "causal_raw": ok_raw, "causal_raw_landing": land_raw,
        "causal_centered": ok_cen, "causal_centered_landing": land_cen,
        "causal_centered_random": ok_rand,
        "trunc": trunc, "k_vals": k_vals,
        "top_pos_eigval_share_mean": float(torch.tensor(shares).mean()),
        "n_pos_eig_mean": float(torch.tensor(n_pos, dtype=torch.float).mean()),
        "recon_mse_class_mean": float(torch.tensor(rec_cm).mean()),
        "test_recon_mse": test_recon_mse(model, loader),
        "mean_lat_norm": scale,
        "best_epoch": int(torch.load(CKPT_DIR / f"d{d}_seed{seed}.pt",
                                     map_location="cpu", weights_only=True)["epoch"]),
    }


# ── aggregation ──────────────────────────────────────────────────────────────

SCALARS = ["eigvec_cos_raw", "eigvec_cos_centered", "eigvec_cos_centered_random",
           "causal_raw", "causal_centered", "causal_centered_random",
           "top_pos_eigval_share_mean", "n_pos_eig_mean",
           "recon_mse_class_mean", "test_recon_mse", "mean_lat_norm"]


def mean_std(xs):
    t = torch.tensor(xs, dtype=torch.float)
    return float(t.mean()), float(t.std()) if len(xs) > 1 else 0.0


def aggregate(per_model, d):
    runs = [per_model[f"d{d}_seed{s}"] for s in SEEDS]
    agg = {}
    for key in SCALARS:
        m, sd = mean_std([r[key] for r in runs])
        agg[key] = {"mean": m, "std": sd}
    k_vals = runs[0]["k_vals"]
    agg["trunc"] = {}
    for k in k_vals:
        mm, ms = mean_std([r["trunc"][str(k)]["mse"] for r in runs])
        cm, cs = mean_std([r["trunc"][str(k)]["causal"] for r in runs])
        agg["trunc"][str(k)] = {"mse_mean": mm, "mse_std": ms,
                                "causal_mean": cm, "causal_std": cs}
    agg["k_vals"] = k_vals
    return agg


# ── figures ──────────────────────────────────────────────────────────────────

def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def sweep_figure(agg, t_raw, t_cen):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    dims = DIMS

    def series(key):
        m = [agg[d][key]["mean"] for d in dims]
        s = [agg[d][key]["std"] for d in dims]
        return m, s

    # (a) cross-class cosine vs d
    ax = axes[0, 0]
    for key, lbl, c in [("eigvec_cos_raw", "raw targets", "tab:red"),
                        ("eigvec_cos_centered", "centered targets", "tab:blue"),
                        ("eigvec_cos_centered_random", "random model (centered)", "tab:gray")]:
        m, s = series(key)
        ax.errorbar(dims, m, yerr=s, marker="o", capsize=3, label=lbl, color=c)
    ax.axhline(t_raw, ls="--", lw=1, color="tab:red", alpha=0.5)
    ax.axhline(t_cen, ls="--", lw=1, color="tab:blue", alpha=0.5)
    ax.text(dims[-1], t_raw + 0.012, f"target overlap raw {t_raw:.3f}",
            ha="right", fontsize=8, color="tab:red", alpha=0.8)
    ax.text(dims[-1], t_cen + 0.012, f"target overlap centered {t_cen:.3f}",
            ha="right", fontsize=8, color="tab:blue", alpha=0.8)
    ax.set_xlabel("d_latent"); ax.set_ylabel("mean off-diagonal |cos|")
    ax.set_title("(a) Top-eigvec cross-class similarity", fontsize=10)
    ax.set_xticks(dims); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (b) causal accuracy vs d
    ax = axes[0, 1]
    for key, lbl, c in [("causal_raw", "raw targets", "tab:red"),
                        ("causal_centered", "centered targets", "tab:blue"),
                        ("causal_centered_random", "random model (centered)", "tab:gray")]:
        m, s = series(key)
        ax.errorbar(dims, m, yerr=s, marker="o", capsize=3, label=lbl, color=c)
    ax.set_xlabel("d_latent"); ax.set_ylabel("causal accuracy (/10 classes)")
    ax.set_title("(b) Causal generation (top eigvec → encode → classify)", fontsize=10)
    ax.set_xticks(dims); ax.set_ylim(0, 10.5); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (c) truncation MSE vs k
    ax = axes[1, 0]
    for d in dims:
        ks = agg[d]["k_vals"]
        m = [agg[d]["trunc"][str(k)]["mse_mean"] for k in ks]
        s = [agg[d]["trunc"][str(k)]["mse_std"] for k in ks]
        ax.errorbar(ks, m, yerr=s, marker="o", capsize=3,
                    label=f"d={d}", color=DIM_COLORS[d])
        ax.axhline(agg[d]["recon_mse_class_mean"]["mean"], ls=":",
                   color=DIM_COLORS[d], alpha=0.6, lw=1)
    ax.set_xlabel("k (top-k positive centered eigvecs, λ-weighted)")
    ax.set_ylabel("MSE(decode($z_k$), class mean image)")
    ax.set_title("(c) Truncation: synthesis MSE vs k\n(dotted: decode(class-mean latent) floor)",
                 fontsize=10)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10, 20])
    ax.set_xticklabels([1, 2, 3, 5, 10, 20]); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # (d) truncation causal accuracy vs k
    ax = axes[1, 1]
    for d in dims:
        ks = agg[d]["k_vals"]
        m = [agg[d]["trunc"][str(k)]["causal_mean"] for k in ks]
        s = [agg[d]["trunc"][str(k)]["causal_std"] for k in ks]
        ax.errorbar(ks, m, yerr=s, marker="o", capsize=3,
                    label=f"d={d}", color=DIM_COLORS[d])
    ax.set_xlabel("k (top-k positive centered eigvecs, λ-weighted)")
    ax.set_ylabel("causal accuracy (/10 classes)")
    ax.set_title("(d) Truncation: causal accuracy vs k", fontsize=10)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10, 20])
    ax.set_xticklabels([1, 2, 3, 5, 10, 20]); ax.set_ylim(0, 10.5)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    cen = {d: agg[d]["eigvec_cos_centered"]["mean"] for d in dims}
    cau = {d: agg[d]["causal_centered"]["mean"] for d in dims}
    fig.suptitle("Exp 16 — Latent-dimension sweep (MNIST, 3 seeds per d): "
                 "centering and rank-1 truncation vs d_latent\n"
                 f"centered cross-class cos: " +
                 ", ".join(f"d={d}: {cen[d]:.3f}" for d in dims) +
                 "   |   centered causal: " +
                 ", ".join(f"d={d}: {cau[d]:.1f}/10" for d in dims),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "figures/mnist/exp16_latent_sweep.png")


@torch.no_grad()
def synthesis_grid(loader, mean_imgs, global_mean, per_model):
    """Centered top-eigvec synthesis at each d (seed 0), vs the class means."""
    classes = sorted(mean_imgs)
    centered = {c: mean_imgs[c] - global_mean for c in classes}
    fig, axes = plt.subplots(1 + len(DIMS), 10, figsize=(16, 2.1 * (1 + len(DIMS))),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    for c in classes:
        axes[0, c].imshow(mean_imgs[c].view(28, 28), cmap="gray_r", vmin=0, vmax=1)
        axes[0, c].set_title(f"c{c}", fontsize=9)
        _clean(axes[0, c])
    axes[0, 0].set_ylabel("Class mean", fontsize=9, labelpad=6)

    for row, d in enumerate(DIMS, start=1):
        model = DecBilinearVAE(d_latent=d)
        load_checkpoint(model, CKPT_DIR / f"d{d}_seed0.pt")
        model.eval()
        scale = per_model[f"d{d}_seed0"]["mean_lat_norm"]
        landing = per_model[f"d{d}_seed0"]["causal_centered_landing"]
        for c in classes:
            v = top_pos_eigvec(model, centered[c]) * scale
            img = model.decode(v.unsqueeze(0))
            axes[row, c].imshow(img.view(28, 28), cmap="gray_r", vmin=0, vmax=1)
            ok = landing[c] == c
            axes[row, c].set_xlabel("✓" if ok else f"→{landing[c]}",
                                    fontsize=9, color="green" if ok else "red")
            _clean(axes[row, c])
        axes[row, 0].set_ylabel(f"d={d}\n(seed 0)", fontsize=9, labelpad=6)
    fig.suptitle("Exp 16 — Centered top-eigvec synthesis at each latent dimension "
                 "(seed 0; ✓ = causal landing correct)", fontsize=11, y=0.99)
    save_fig(fig, "figures/mnist/exp16_synthesis_grid.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ensure_checkpoints()

    torch.manual_seed(0)
    loader = build_test_loader()
    mean_imgs = class_mean_images(loader)
    global_mean = torch.stack(list(mean_imgs.values())).mean(0)
    _, t_raw = offdiag_mean([mean_imgs[c] for c in sorted(mean_imgs)])
    _, t_cen = offdiag_mean([mean_imgs[c] - global_mean for c in sorted(mean_imgs)])
    print(f"target overlap (dataset property): raw {t_raw:.3f}, centered {t_cen:.3f}")

    per_model = {}
    for d in DIMS:
        for s in SEEDS:
            r = run_battery(d, s, loader, mean_imgs, global_mean)
            per_model[f"d{d}_seed{s}"] = r
            print(f"  d={d} seed={s}: raw cos {r['eigvec_cos_raw']:.3f} "
                  f"causal {r['causal_raw']}/10 | centered cos "
                  f"{r['eigvec_cos_centered']:.3f} causal {r['causal_centered']}/10 "
                  f"(random {r['eigvec_cos_centered_random']:.3f}, "
                  f"{r['causal_centered_random']}/10) | "
                  f"top+share {r['top_pos_eigval_share_mean']:.3f} | "
                  f"recon MSE {r['test_recon_mse']:.4f}")

    agg = {d: aggregate(per_model, d) for d in DIMS}

    print("\nPer-dimension summary (mean±std over 3 seeds):")
    for d in DIMS:
        a = agg[d]
        print(f"  d={d:>2}: raw cos {a['eigvec_cos_raw']['mean']:.3f}±{a['eigvec_cos_raw']['std']:.3f}  "
              f"cen cos {a['eigvec_cos_centered']['mean']:.3f}±{a['eigvec_cos_centered']['std']:.3f}  "
              f"rand {a['eigvec_cos_centered_random']['mean']:.3f}  "
              f"causal raw {a['causal_raw']['mean']:.1f} cen {a['causal_centered']['mean']:.1f} "
              f"rand {a['causal_centered_random']['mean']:.1f}  "
              f"top+share {a['top_pos_eigval_share_mean']['mean']:.3f}")
        ks = a["k_vals"]
        print("        trunc " + "  ".join(
            f"k={k}: {a['trunc'][str(k)]['mse_mean']:.4f}/{a['trunc'][str(k)]['causal_mean']:.1f}"
            for k in ks))

    results = {
        "config": {"dims": DIMS, "seeds": SEEDS, "epochs": 30, "lr": 1e-3,
                   "weight_decay": 0.01, "beta": 1.0, "noise_std": 0.3,
                   "batch_size": 128, "recon_loss": "bce",
                   "d10_source": "checkpoints/mnist/seeds/seed{0,1,2}.pt (same protocol, reused)",
                   "truncation_rule": "top-k positive centered eigvecs, lambda-weighted, |z|=mean latent norm",
                   "random_control_seed": "1000 + 10*d + seed"},
        "target_similarity_raw": t_raw,
        "target_similarity_centered": t_cen,
        "per_model": per_model,
        "per_dim": {str(d): agg[d] for d in DIMS},
    }
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS}")

    sweep_figure(agg, t_raw, t_cen)
    synthesis_grid(loader, mean_imgs, global_mean, per_model)


if __name__ == "__main__":
    main()
