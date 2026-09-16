"""
Exp 17 — Gradient-based (Jacobian) baseline: what does the bilinear
constraint buy over plain autograd on a normal VAE?

The weight-based protocol (exp13, centered targets) finds generative latent
directions from the decoder weights alone: eigendecompose Q_dec for
p*_c = mean_c − global_mean, decode the top positive eigenvector at the mean
latent norm, and check causal generation (decode → encode → nearest class
mean; 9/10 on MNIST).  A fair skeptic asks: can ordinary gradient machinery
on a standard MLP-decoder VAE do the same, without any bilinear structure?

Methods compared (all with the SAME centered targets and the same causal
test, every model using its own class-mean latents and mean latent norm):

    weight-eig  (DecBilinearVAE)   exp13 reference row, reproduced here.
    grad-ascent (VanillaVAE)       maximize <p*_c, decode(z)> over z by Adam,
                                   projecting z to the model's mean latent
                                   norm after every step (300 steps, lr 0.05,
                                   8 seeded restarts, best kept).
    jacobian    (VanillaVAE)       cheap one-step variant: z ∝ J^T p*
                                   evaluated at z=0, scaled to the norm budget.
    grad-ascent (DecBilinearVAE)   sanity/ablation: gradient search on the
                                   bilinear model itself (can exploit the same
                                   structure plus the output sigmoid).
    jacobian    (DecBilinearVAE)   structurally degenerate: the bias-free
                                   bilinear decoder is purely quadratic in z,
                                   so J^T p* ≡ 0 at z=0 (reported, not run).
    controls                       the same procedures on random-init models
                                   (all statistics from the random model).

Outputs:
    figures/mnist/exp17_jacobian_baseline.png   comparison grid + summary bars
    figures/exp17_results.json                  all numbers incl. the method
                                                table (causal accuracy, MSE,
                                                needs-data / needs-gradients /
                                                global-validity columns)
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as Fn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models    import DecBilinearVAE
from train     import load_checkpoint
from analysis  import (get_decoder_interaction_matrix, decompose,
                       compute_class_means, mean_lat_norm)
from visualize import save_fig

DATA         = "/home/v25/ippa6201/bilinear-mlp-repro/data"
CKPT_BILINEAR = "checkpoints/mnist/model.pt"
_VANILLA_DIR  = os.path.join(os.path.dirname(__file__),
                             "../../../extension_full/checkpoints/mnist/vanilla_vae")
CKPT_VANILLA  = os.path.join(_VANILLA_DIR, "model.pt")
CKPT_VANILLA_SEEDS = {s: os.path.join(_VANILLA_DIR, f"seeds/seed{s}.pt")
                      for s in (1, 2, 3)}
RESULTS = "figures/exp17_results.json"

# gradient-ascent hyperparameters (documented in the JSON)
N_RESTARTS = 8
N_STEPS    = 300
LR         = 0.05
SEED       = 0


# ── VanillaVAE matching the extension_full checkpoint key layout ───────────
# (same inline definition as exp12; encoder mirrors DecBilinearVAE's encoder,
#  decoder is a plain MLP: z(10) → Linear(256) → ReLU → Linear(784) → Sigmoid)
class VanillaVAE(nn.Module):
    def __init__(self, d_input=784, d_enc1=256, d_enc2=512, d_latent=10, d_dec=256):
        super().__init__()
        self.d_latent  = d_latent
        self.enc_fc1   = nn.Linear(d_input, d_enc1)
        self.enc_fc2   = nn.Linear(d_enc1,  d_enc2)
        self.fc_mu     = nn.Linear(d_enc2,  d_latent)
        self.fc_logvar = nn.Linear(d_enc2,  d_latent)

        class _Dec(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(d_latent, d_dec)
                self.fc2 = nn.Linear(d_dec, d_input)
            def forward(self, z):
                return torch.sigmoid(self.fc2(Fn.relu(self.fc1(z))))

        self.decoder = _Dec()

    def encode(self, x):
        h = Fn.relu(self.enc_fc1(x))
        h = Fn.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z)


def load_vanilla(path):
    model = VanillaVAE()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Shared protocol pieces ─────────────────────────────────────────────────

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


@torch.no_grad()
def causal_test(model, z_by_class, class_mu, mean_imgs):
    """Decode each class's latent, re-encode, classify by nearest class-mean
    latent (cosine).  Returns (n correct, landing list, per-class MSE to the
    class-mean image, decoded images)."""
    M = torch.stack([class_mu[c] for c in sorted(class_mu)])
    landing, mses, imgs = [], [], {}
    for c in sorted(z_by_class):
        img = model.decode(z_by_class[c].unsqueeze(0))
        mu, _ = model.encode(img)
        landing.append(Fn.cosine_similarity(mu, M).argmax().item())
        mses.append(((img.squeeze(0) - mean_imgs[c]) ** 2).mean().item())
        imgs[c] = img.squeeze(0)
    ok = sum(int(p == c) for c, p in enumerate(landing))
    return ok, landing, mses, imgs


@torch.no_grad()
def recon_mse(model, class_mu, mean_imgs):
    """Anchor: MSE of decoded class-mean latents to class-mean images."""
    return [((model.decode(class_mu[c].unsqueeze(0)).squeeze(0)
              - mean_imgs[c]) ** 2).mean().item() for c in sorted(mean_imgs)]


# ── Direction finders ──────────────────────────────────────────────────────

def grad_ascent_z(model, target, radius, generator,
                  n_restarts=N_RESTARTS, n_steps=N_STEPS, lr=LR):
    """
    Maximize f(z) = <target, decode(z)> subject to ||z|| = radius:
    Adam on z, projecting back to the sphere after every step, batched over
    n_restarts random inits on the sphere; the best final z is returned.
    """
    z0 = torch.randn(n_restarts, model.d_latent, generator=generator)
    z0 = z0 / z0.norm(dim=1, keepdim=True) * radius
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        obj = (model.decode(z) * target.unsqueeze(0)).sum(dim=1)
        (-obj.sum()).backward()
        opt.step()
        with torch.no_grad():
            z.data = z.data / z.data.norm(dim=1, keepdim=True) * radius
    with torch.no_grad():
        obj = (model.decode(z) * target.unsqueeze(0)).sum(dim=1)
        return z[obj.argmax()].detach().clone()


def jacobian_z(model, target, radius):
    """One-step variant: z ∝ J^T p* at z=0 (= ∇_z <p*, decode(z)>|_{z=0}),
    scaled to the norm budget.  Returns (z, grad_norm); z is None when the
    gradient at the origin vanishes (no linear term — the bilinear case)."""
    z = torch.zeros(1, model.d_latent, requires_grad=True)
    (model.decode(z) * target.unsqueeze(0)).sum().backward()
    g = z.grad.squeeze(0)
    gn = g.norm().item()
    if gn < 1e-8:
        return None, gn
    return (g / gn * radius).detach(), gn


# ── Per-model evaluation ───────────────────────────────────────────────────

def eval_model(model, loader, mean_imgs, centered, methods):
    """Run the requested methods on one model.  All statistics (class-mean
    latents, norm budget) come from this model itself."""
    class_mu = compute_class_means(model, loader)
    radius   = mean_lat_norm(model, loader)
    out = {"mean_lat_norm": radius,
           "recon_mse_per_class": recon_mse(model, class_mu, mean_imgs)}
    out["recon_mse_mean"] = float(torch.tensor(out["recon_mse_per_class"]).mean())

    for method in methods:
        if method == "weight_eig":
            z_by_class = {c: top_pos_eigvec(model, centered[c]) * radius
                          for c in sorted(centered)}
        elif method == "grad_ascent":
            gen = torch.Generator().manual_seed(SEED)
            z_by_class = {c: grad_ascent_z(model, centered[c], radius, gen)
                          for c in sorted(centered)}
        elif method == "jacobian":
            z_by_class, grad_norms = {}, []
            for c in sorted(centered):
                z, gn = jacobian_z(model, centered[c], radius)
                grad_norms.append(gn)
                if z is not None:
                    z_by_class[c] = z
            if len(z_by_class) < 10:
                out["jacobian"] = {"degenerate": True,
                                   "grad_norms_at_origin": grad_norms,
                                   "note": "decoder output has no linear term "
                                           "at z=0 (bias-free quadratic + "
                                           "sigmoid), so J^T p* ≡ 0 there"}
                continue
        ok, landing, mses, imgs = causal_test(model, z_by_class, class_mu, mean_imgs)
        out[method] = {"causal": ok, "landing": landing,
                       "synth_mse_per_class": mses,
                       "synth_mse_mean": float(torch.tensor(mses).mean()),
                       "images": imgs}
    return out


def strip_images(d):
    return {k: ({kk: vv for kk, vv in v.items() if kk != "images"}
                if isinstance(v, dict) else v) for k, v in d.items()}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    loader = build_loader()
    mean_imgs   = class_mean_images(loader)
    global_mean = torch.stack(list(mean_imgs.values())).mean(0)
    centered    = {c: mean_imgs[c] - global_mean for c in mean_imgs}

    # trained models
    bilinear = DecBilinearVAE(); load_checkpoint(bilinear, CKPT_BILINEAR)
    vanilla  = load_vanilla(CKPT_VANILLA)
    print(f"Loaded {CKPT_VANILLA}")

    print("\n[1/4] DecBilinearVAE: weight-eig (reference) + grad-ascent + jacobian")
    res_bil = eval_model(bilinear, loader, mean_imgs, centered,
                         ["weight_eig", "grad_ascent", "jacobian"])

    print("[2/4] VanillaVAE (main): grad-ascent + jacobian")
    res_van = eval_model(vanilla, loader, mean_imgs, centered,
                         ["grad_ascent", "jacobian"])

    print("[3/4] VanillaVAE seeds 1-3 (spread)")
    res_seeds = {}
    for s, path in CKPT_VANILLA_SEEDS.items():
        if not os.path.exists(path):
            print(f"  seed{s}: checkpoint missing, skipped"); continue
        res_seeds[s] = eval_model(load_vanilla(path), loader, mean_imgs,
                                  centered, ["grad_ascent", "jacobian"])
        print(f"  seed{s}: grad {res_seeds[s]['grad_ascent']['causal']}/10, "
              f"jac {res_seeds[s]['jacobian']['causal']}/10")

    print("[4/4] Random-init controls (all statistics from the random model)")
    torch.manual_seed(123)
    res_rand_van = eval_model(VanillaVAE().eval(), loader, mean_imgs, centered,
                              ["grad_ascent", "jacobian"])
    torch.manual_seed(123)
    res_rand_bil = eval_model(DecBilinearVAE().eval(), loader, mean_imgs,
                              centered, ["weight_eig", "grad_ascent"])

    # ── seed spread (main model + 3 seeds) ──────────────────
    spread = {}
    for method in ("grad_ascent", "jacobian"):
        accs = [res_van[method]["causal"]] + \
               [r[method]["causal"] for r in res_seeds.values()]
        mses = [res_van[method]["synth_mse_mean"]] + \
               [r[method]["synth_mse_mean"] for r in res_seeds.values()]
        spread[method] = {
            "causal_per_model": accs,
            "causal_mean": float(torch.tensor(accs, dtype=torch.float).mean()),
            "causal_min": min(accs), "causal_max": max(accs),
            "synth_mse_per_model": mses,
            "synth_mse_mean": float(torch.tensor(mses).mean()),
        }

    # ── the key table ───────────────────────────────────────
    needs_data_note = ("all methods use data for the targets p*_c (class-mean "
                       "images) and the norm budget (mean latent norm); "
                       "'needs data' below refers to the direction extraction "
                       "itself, given p*")
    table = [
        {"method": "weight-eig (DecBilinearVAE)",
         "causal_accuracy": res_bil["weight_eig"]["causal"],
         "synth_mse": res_bil["weight_eig"]["synth_mse_mean"],
         "needs_data": False,
         "needs_gradients": False,
         "global_validity": ("yes — Q_dec is the exact quadratic form of the "
                             "pre-sigmoid output over ALL of latent space; one "
                             "decomposition yields the full generative + "
                             "suppressor eigenbasis")},
        {"method": "grad-ascent (DecBilinearVAE)",
         "causal_accuracy": res_bil["grad_ascent"]["causal"],
         "synth_mse": res_bil["grad_ascent"]["synth_mse_mean"],
         "needs_data": False,
         "needs_gradients": True,
         "global_validity": ("no — one local optimum per run "
                             f"({N_RESTARTS}x{N_STEPS} decoder fwd+bwd passes "
                             "per class), depends on init/lr/restarts")},
        {"method": "grad-ascent (VanillaVAE)",
         "causal_accuracy": res_van["grad_ascent"]["causal"],
         "synth_mse": res_van["grad_ascent"]["synth_mse_mean"],
         "needs_data": False,
         "needs_gradients": True,
         "global_validity": ("no — one local optimum per run "
                             f"({N_RESTARTS}x{N_STEPS} decoder fwd+bwd passes "
                             "per class), depends on init/lr/restarts")},
        {"method": "jacobian 1-step (VanillaVAE)",
         "causal_accuracy": res_van["jacobian"]["causal"],
         "synth_mse": res_van["jacobian"]["synth_mse_mean"],
         "needs_data": False,
         "needs_gradients": True,
         "global_validity": ("no — first-order linearization, valid only "
                             "near z=0 (one bwd pass per class)")},
        {"method": "jacobian 1-step (DecBilinearVAE)",
         "causal_accuracy": None,
         "synth_mse": None,
         "needs_data": False,
         "needs_gradients": True,
         "global_validity": ("degenerate — bias-free quadratic decoder has "
                             "J^T p* = 0 at z=0")},
    ]

    results = {
        "config": {
            "targets": "centered class means, p*_c = mean_c - global_mean (MNIST test set)",
            "norm_budget": "per-model mean latent norm, z projected to it after every step",
            "grad_ascent": {"n_restarts": N_RESTARTS, "n_steps": N_STEPS,
                            "lr": LR, "optimizer": "Adam", "seed": SEED},
            "causal_test": "decode z -> encode -> argmax cosine to that model's class-mean latents",
            "needs_data_note": needs_data_note,
        },
        "dec_bilinear_vae":   strip_images(res_bil),
        "vanilla_vae_main":   strip_images(res_van),
        "vanilla_vae_seeds":  {f"seed{s}": strip_images(r)
                               for s, r in res_seeds.items()},
        "vanilla_seed_spread": spread,
        "controls_random_init": {
            "vanilla_vae":      strip_images(res_rand_van),
            "dec_bilinear_vae": strip_images(res_rand_bil),
            "note": "controls are fully self-consistent: class-mean latents "
                    "and norm budget come from the random model itself",
        },
        "table": table,
    }
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS}")

    # ── console summary ─────────────────────────────────────
    print("\nmethod                          causal   synth MSE")
    for row in table:
        acc = "n/a " if row["causal_accuracy"] is None else f"{row['causal_accuracy']}/10"
        mse = " n/a"  if row["synth_mse"] is None else f"{row['synth_mse']:.4f}"
        print(f"{row['method']:<32}{acc:>6}   {mse}")
    print(f"vanilla spread (main+3 seeds): grad "
          f"{spread['grad_ascent']['causal_per_model']}, "
          f"jac {spread['jacobian']['causal_per_model']}")
    print(f"controls (random init): vanilla grad "
          f"{res_rand_van['grad_ascent']['causal']}/10, "
          f"bilinear weight-eig {res_rand_bil['weight_eig']['causal']}/10, "
          f"bilinear grad {res_rand_bil['grad_ascent']['causal']}/10")

    # ── figure ──────────────────────────────────────────────
    def clean(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    rows = [
        ("Class mean\n(target basis)", None, None),
        ("DecBilinearVAE\nweight-eig",  res_bil["weight_eig"], "tab:blue"),
        ("DecBilinearVAE\ngrad-ascent", res_bil["grad_ascent"], "tab:cyan"),
        ("VanillaVAE\ngrad-ascent",     res_van["grad_ascent"], "tab:orange"),
        ("VanillaVAE\njacobian 1-step", res_van["jacobian"], "tab:red"),
    ]
    fig = plt.figure(figsize=(18, 13.5))
    gs_img = fig.add_gridspec(5, 10, top=0.90, bottom=0.33,
                              hspace=0.45, wspace=0.05)
    gs_bar = fig.add_gridspec(1, 2, top=0.25, bottom=0.05,
                              left=0.06, right=0.97, wspace=0.25)
    for r, (label, res, _) in enumerate(rows):
        for c in range(10):
            ax = fig.add_subplot(gs_img[r, c])
            img = mean_imgs[c] if res is None else res["images"][c]
            ax.imshow(img.view(28, 28), cmap="gray_r", vmin=0, vmax=1)
            clean(ax)
            if r == 0:
                ax.set_title(f"c{c}", fontsize=9)
            else:
                land = res["landing"][c]
                ok = land == c
                ax.set_xlabel("✓" if ok else f"→{land}", fontsize=9,
                              color="green" if ok else "red")
            if c == 0:
                ax.set_ylabel(label, fontsize=8.5, labelpad=6)

    # summary bars
    names  = ["weight-eig\n(bilinear)", "grad-ascent\n(bilinear)",
              "grad-ascent\n(vanilla)", "jacobian\n(vanilla)",
              "grad-ascent\n(random ctrl)"]
    colors = ["tab:blue", "tab:cyan", "tab:orange", "tab:red", "tab:gray"]
    accs   = [res_bil["weight_eig"]["causal"], res_bil["grad_ascent"]["causal"],
              spread["grad_ascent"]["causal_mean"],
              spread["jacobian"]["causal_mean"],
              res_rand_van["grad_ascent"]["causal"]]
    mses   = [res_bil["weight_eig"]["synth_mse_mean"],
              res_bil["grad_ascent"]["synth_mse_mean"],
              spread["grad_ascent"]["synth_mse_mean"],
              spread["jacobian"]["synth_mse_mean"],
              res_rand_van["grad_ascent"]["synth_mse_mean"]]
    acc_err = [[0, 0,
                accs[2] - spread["grad_ascent"]["causal_min"],
                accs[3] - spread["jacobian"]["causal_min"], 0],
               [0, 0,
                spread["grad_ascent"]["causal_max"] - accs[2],
                spread["jacobian"]["causal_max"] - accs[3], 0]]

    ax_acc = fig.add_subplot(gs_bar[0, 0])
    ax_acc.bar(names, accs, color=colors, yerr=acc_err, capsize=4)
    ax_acc.axhline(1, ls="--", c="gray", lw=1, label="chance (1/10)")
    ax_acc.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax_acc.set_ylim(0, 10.8)
    ax_acc.set_ylabel("causal accuracy (/10)", fontsize=10)
    ax_acc.set_title("Causal generation (vanilla bars: mean over main+3 seeds, "
                     "error bars min–max)", fontsize=10)
    ax_acc.tick_params(axis="x", labelsize=8.5)
    for i, a in enumerate(accs):
        ax_acc.text(i, a + 0.55, f"{a:.1f}" if i in (2, 3) else f"{int(a)}",
                    ha="center", fontsize=9)

    # MSE panel: random ctrl (0.179, flat gray output) omitted to keep the
    # scale readable — its value is reported in the title and JSON.
    ax_mse = fig.add_subplot(gs_bar[0, 1])
    ax_mse.bar(names[:4], mses[:4], color=colors[:4])
    ax_mse.axhline(res_bil["recon_mse_mean"], ls="--", c="tab:blue", lw=1,
                   label=f"bilinear VAE recon {res_bil['recon_mse_mean']:.4f}")
    ax_mse.axhline(res_van["recon_mse_mean"], ls="--", c="tab:orange", lw=1,
                   label=f"vanilla VAE recon {res_van['recon_mse_mean']:.4f}")
    ax_mse.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax_mse.set_ylim(0, 0.05)
    ax_mse.set_ylabel("synthesis MSE to class mean", fontsize=10)
    ax_mse.set_title("Synthesis quality (lower is better; dashed = VAE recon "
                     f"anchor; random ctrl {mses[4]:.3f} omitted)", fontsize=10)
    ax_mse.tick_params(axis="x", labelsize=8.5)
    for i, m in enumerate(mses[:4]):
        ax_mse.text(i, m + 0.001, f"{m:.3f}", ha="center", fontsize=9)

    fig.suptitle(
        "Exp 17 — Gradient (Jacobian) baseline vs weight-based synthesis "
        "(MNIST, centered targets)\n"
        f"weight-eig {res_bil['weight_eig']['causal']}/10  |  "
        f"bilinear grad-ascent {res_bil['grad_ascent']['causal']}/10  |  "
        f"vanilla grad-ascent {spread['grad_ascent']['causal_mean']:.1f}/10  |  "
        f"vanilla jacobian {spread['jacobian']['causal_mean']:.1f}/10  |  "
        f"random ctrl {res_rand_van['grad_ascent']['causal']}/10   —   "
        "gradient methods match weight-based capability; the bilinear win is "
        "closed-form global structure, not accuracy",
        fontsize=11.5, y=0.975)
    save_fig(fig, "figures/mnist/exp17_jacobian_baseline.png")


if __name__ == "__main__":
    main()
