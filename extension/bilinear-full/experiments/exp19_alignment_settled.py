"""
Exp 19 — The encoder↔decoder alignment question, settled

Consolidates the alignment story (exp03 → exp07 → exp15) into one experiment.
The corrected latent-space alignment measure (exp07):

    z_enc[c] = encode( Q_in top +eigvec for latent dim c, normalised to [0,1] )
    z_dec[c] = Q_dec(mean_img_c) top +eigvec · scale
    metric   = mean_c cos(z_enc[c], z_dec[c])

is computed for four checkpoint groups:

  v2               : main + 5 seeds (no alignment loss) — the exp07/exp15 numbers
  v3-shipped       : checkpoints/mnist/v3/shipped.pt, the prototype v3
                     checkpoint, nominally trained WITH the explicit
                     alignment loss
  v3-fixed λ=1     : 3 fresh seeds trained with a REPAIRED alignment loss at the
                     recommended strength (run.py --align_lambda help: "1.0")
  v3-fixed λ=1000  : 3 fresh seeds at 1000× strength, where the loss demonstrably
                     optimises its own objective

Discovery along the way: the alignment_loss of the shipped v3 prototype (its
loss is reproduced verbatim below) detaches BOTH Jacobians — the encoder one
via .detach(), the decoder one under torch.no_grad() — so the returned scalar
has grad_fn=None and align_lambda · L_align contributes ZERO gradient.  Any v3
checkpoint trained with that loss is therefore effectively a v2 run (this
experiment re-verifies the detachment programmatically and the shipped
checkpoint's Jacobian cosine confirms it: 0.375 vs v2's 0.374).  The
"v3-fixed" seeds repair the loss with create_graph=True autograd +
differentiable finite differences (see train_v3.py).

Manipulation check: the mean Jacobian cosine |cos(∂μ_k/∂x, ∂out/∂z_k)| — the
quantity the alignment loss actually optimises — is reported per checkpoint.
The empirical |random| baseline (500 unit-vector draws against the same
z_enc codes) carries its own spread.

Expected/honest outcome: the corrected metric stays inside the random band for
every group — i.e. even a working alignment loss that moves its own Jacobian
objective does not create weight-basis eigenvector alignment.  A null.

Outputs:
    figures/mnist/exp19_alignment_settled.png
    figures/mnist/exp19_results.json
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
from analysis  import (get_encoder_interaction_matrix, get_decoder_interaction_matrix,
                       decompose, mean_lat_norm)
from visualize import save_fig

DATA    = str(Path(__file__).resolve().parents[3] / "data")
V3DIR   = Path("checkpoints/mnist/v3")
N_RAND  = 500
N_JAC   = 256   # test images for the Jacobian manipulation check

CKPT_GROUPS = (
    [("v2", "main",  Path("checkpoints/mnist/model.pt"))] +
    [("v2", f"seed{s}", Path(f"checkpoints/mnist/seeds/seed{s}.pt")) for s in range(5)] +
    [("v3-shipped", "main", V3DIR / "shipped.pt")] +
    [("v3-fixed-lam1",    f"seed{s}", V3DIR / f"lam1_seed{s}.pt")    for s in range(3)] +
    [("v3-fixed-lam1000", f"seed{s}", V3DIR / f"lam1000_seed{s}.pt") for s in range(3)]
)
GROUP_COLORS = {"v2": "steelblue", "v3-shipped": "darkorchid",
                "v3-fixed-lam1": "seagreen", "v3-fixed-lam1000": "darkorange"}


def _cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


class _ShippedV3(FullBilinearVAE):
    """The shipped v3 prototype, reproduced in-project.

    The prototype's v3 architecture is identical to FullBilinearVAE here
    (bilinear encoder + linear skip, bilinear decoder + linear skip); the only
    addition is alignment_loss.  The method below is a verbatim copy of the
    shipped v3 alignment_loss, kept so the detachment check runs without the
    prototype tree.  (Only the functional-module alias differs: this file
    imports torch.nn.functional as Fn.)
    """

    def alignment_loss(self, x: torch.Tensor, z: torch.Tensor,
                       delta: float = 0.3) -> torch.Tensor:
        """
        Compute the alignment loss for one randomly sampled latent dimension.

        Args:
            x     : clean input batch, shape (B, 784)
            z     : reparameterised latent codes, shape (B, d_latent)
            delta : finite-difference step for the decoder Jacobian

        Returns:
            scalar loss (1 − mean |cosine similarity|)
        """
        B = x.size(0)
        k = torch.randint(0, self.d_latent, (1,)).item()

        # ── encoder Jacobian ∂μ_k/∂x via autograd ────────────────────
        x_ag = x.detach().requires_grad_(True)
        mu_ag, _ = self.encode(x_ag)
        mu_ag[:, k].sum().backward()
        enc_jac = x_ag.grad.detach()                        # (B, 784)

        # ── decoder Jacobian ∂output/∂z_k via finite differences ─────
        z_det = z.detach()
        e_k   = torch.zeros_like(z_det)
        e_k[:, k] = delta
        with torch.no_grad():
            dec_jac = (self.decoder(z_det + e_k)
                       - self.decoder(z_det - e_k)) / (2 * delta)   # (B, 784)

        # ── cosine similarity averaged over batch ─────────────────────
        cos = Fn.cosine_similarity(enc_jac.view(B, -1),
                                   dec_jac.view(B, -1), dim=1)        # (B,)
        return 1.0 - cos.abs().mean()


def verify_shipped_loss_detached():
    """Instantiate the shipped v3 loss (verbatim copy above) and check that it
    carries no gradient.  Returns True/False/None (None = could not check)."""
    try:
        torch.manual_seed(0)
        m = _ShippedV3()
        x = torch.rand(4, 784)
        mu, logvar = m.encode(x)
        z = m.reparametrize(mu.detach(), logvar.detach())
        return not m.alignment_loss(x, z).requires_grad
    except Exception as e:                                  # pragma: no cover
        print(f"  (could not verify shipped loss: {e})")
        return None


def corrected_alignment(model, mean_imgs, scale, classes):
    """exp07/exp15 measure; also returns the z_enc codes for the baseline."""
    per_class, z_encs = [], []
    with torch.no_grad():
        for c in classes:
            d = torch.zeros(model.d_latent); d[c] = 1.0
            vals_e, vecs_e = decompose(get_encoder_interaction_matrix(model, d))
            pos_e = (vals_e > 0).nonzero(as_tuple=True)[0]
            ev = vecs_e[pos_e[0]] if len(pos_e) else torch.zeros(model.d_input)
            v = ev - ev.min(); v = v / (v.max() + 1e-8)
            mu, _ = model.encode(v.unsqueeze(0))
            z_enc = mu.squeeze(0); z_encs.append(z_enc)
            vals_d, vecs_d = decompose(get_decoder_interaction_matrix(model, mean_imgs[c]))
            pos_d = (vals_d > 0).nonzero(as_tuple=True)[0]
            z_dec = vecs_d[pos_d[0]] * scale if len(pos_d) else torch.zeros(model.d_latent)
            per_class.append(_cos(z_enc, z_dec))
    return per_class, z_encs


def random_baseline(z_encs, d_latent, n=N_RAND, seed=42):
    """Distribution over draws of mean_c |cos(z_enc[c], random unit)|."""
    g = torch.Generator().manual_seed(seed)
    draws = []
    for _ in range(n):
        vals = []
        for z in z_encs:
            r = torch.randn(d_latent, generator=g); r = r / r.norm()
            vals.append(abs(_cos(z, r)))
        draws.append(float(np.mean(vals)))
    return float(np.mean(draws)), float(np.std(draws))


def jacobian_cos(model, x, delta=0.3):
    """Mean over k and batch of |cos(∂μ_k/∂x, ∂out/∂z_k)| — the alignment
    loss's own objective (its manipulation check)."""
    B = x.size(0)
    with torch.no_grad():
        mu, _ = model.encode(x)
    vals = []
    for k in range(model.d_latent):
        x_ag = x.detach().requires_grad_(True)
        mu_ag, _ = model.encode(x_ag)
        mu_ag[:, k].sum().backward()
        enc_jac = x_ag.grad.detach()
        e_k = torch.zeros_like(mu); e_k[:, k] = delta
        with torch.no_grad():
            dec_jac = (model.decode(mu + e_k) - model.decode(mu - e_k)) / (2 * delta)
        vals.append(Fn.cosine_similarity(enc_jac.view(B, -1),
                                         dec_jac.view(B, -1), dim=1).abs().mean().item())
    return float(np.mean(vals))


def main():
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    loader = DataLoader(datasets.MNIST(DATA, train=False, download=False, transform=tfm),
                        batch_size=512, shuffle=False)
    buckets = {}
    for x, y in loader:
        for i, l in enumerate(y.tolist()):
            buckets.setdefault(l, []).append(x[i])
    mean_imgs = {c: torch.stack(v).mean(0) for c, v in sorted(buckets.items())}
    classes = sorted(mean_imgs)
    x_jac, _ = next(iter(loader)); x_jac = x_jac[:N_JAC]

    detached = verify_shipped_loss_detached()
    print(f"Shipped v3 alignment_loss detached from graph (zero gradient): {detached}")

    rows = []
    print(f"\n{'group':<18}{'ckpt':<8}{'align':>8}{'|rand| baseline':>18}{'jac|cos|':>10}")
    print("-" * 62)
    for group, name, path in CKPT_GROUPS:
        if not path.exists():
            print(f"{group:<18}{name:<8}  missing ({path}) — skipped"); continue
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = FullBilinearVAE(); model.load_state_dict(ckpt["model_state"]); model.eval()
        scale = mean_lat_norm(model, loader)
        per_class, z_encs = corrected_alignment(model, mean_imgs, scale, classes)
        rb_m, rb_s = random_baseline(z_encs, model.d_latent)
        jc = jacobian_cos(model, x_jac)
        rows.append({"group": group, "ckpt": name,
                     "alignment": float(np.mean(per_class)),
                     "alignment_per_class": per_class,
                     "rand_baseline_mean": rb_m, "rand_baseline_std": rb_s,
                     "jacobian_cos": jc})
        print(f"{group:<18}{name:<8}{np.mean(per_class):>8.3f}"
              f"{rb_m:>12.3f}±{rb_s:.3f}{jc:>10.3f}")

    groups = list(dict.fromkeys(r["group"] for r in rows))
    group_stats = {}
    for g in groups:
        a = [r["alignment"] for r in rows if r["group"] == g]
        j = [r["jacobian_cos"] for r in rows if r["group"] == g]
        group_stats[g] = {"n": len(a),
                          "alignment_mean": float(np.mean(a)), "alignment_std": float(np.std(a)),
                          "jacobian_mean":  float(np.mean(j)), "jacobian_std":  float(np.std(j))}
    rb_pool_m = float(np.mean([r["rand_baseline_mean"] for r in rows]))
    rb_pool_s = float(np.mean([r["rand_baseline_std"]  for r in rows]))

    print(f"\n{'group':<18}{'n':>3}{'align mean±std':>18}{'jac|cos| mean±std':>20}")
    print("-" * 60)
    for g in groups:
        s = group_stats[g]
        print(f"{g:<18}{s['n']:>3}{s['alignment_mean']:>10.3f}±{s['alignment_std']:.3f}"
              f"{s['jacobian_mean']:>13.3f}±{s['jacobian_std']:.3f}")
    print(f"\n|random| baseline (pooled): {rb_pool_m:.3f} ± {rb_pool_s:.3f}")

    with open("figures/mnist/exp19_results.json", "w") as f:
        json.dump({"shipped_v3_loss_detached": detached,
                   "per_ckpt": rows,
                   "group_stats": group_stats,
                   "rand_baseline_pooled_mean": rb_pool_m,
                   "rand_baseline_pooled_std":  rb_pool_s,
                   "note": ("Corrected exp07 measure. v3-shipped was trained through "
                            "run.py whose alignment loss is detached (zero gradient); "
                            "v3-fixed seeds use a repaired differentiable loss "
                            "(train_v3.py).")},
                  f, indent=2)

    # ── Figure: one row of two panels ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    xpos, labels = [], []
    x = 0
    for g in groups:
        for r in [r for r in rows if r["group"] == g]:
            xpos.append(x); labels.append(f"{r['ckpt']}")
            x += 1
        x += 0.8  # gap between groups

    # Panel A: corrected alignment + random band
    axes[0].axhspan(rb_pool_m - rb_pool_s, rb_pool_m + rb_pool_s,
                    color="gray", alpha=0.25,
                    label=f"|random| baseline {rb_pool_m:.2f}±{rb_pool_s:.2f}")
    axes[0].axhline(rb_pool_m, color="gray", ls="--", lw=1)
    axes[0].axhline(-rb_pool_m, color="gray", ls="--", lw=1)
    axes[0].axhspan(-rb_pool_m - rb_pool_s, -rb_pool_m + rb_pool_s,
                    color="gray", alpha=0.25)
    axes[0].axhline(0, color="black", lw=0.7)
    i = 0
    for g in groups:
        rs = [r for r in rows if r["group"] == g]
        xs = xpos[i:i + len(rs)]
        axes[0].scatter(xs, [r["alignment"] for r in rs], s=70,
                        color=GROUP_COLORS[g], zorder=3, label=g)
        axes[1].scatter(xs, [r["jacobian_cos"] for r in rs], s=70,
                        color=GROUP_COLORS[g], zorder=3, label=g)
        i += len(rs)
    axes[0].set_xticks(xpos); axes[0].set_xticklabels(labels, rotation=45, fontsize=8)
    axes[0].set_ylabel("Corrected latent enc↔dec cos (exp07 measure)")
    axes[0].set_ylim(-0.6, 0.6)
    axes[0].set_title("Weight-basis alignment: inside the noise band\nfor every group (null)",
                      fontsize=10)
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Panel B: Jacobian objective (manipulation check)
    v2_jac = group_stats["v2"]["jacobian_mean"]
    axes[1].axhline(v2_jac, color="steelblue", ls="--", lw=1,
                    label=f"v2 level ({v2_jac:.3f})")
    axes[1].set_xticks(xpos); axes[1].set_xticklabels(labels, rotation=45, fontsize=8)
    axes[1].set_ylabel("Jacobian |cos(∂μ/∂x, ∂out/∂z)| (loss objective)")
    axes[1].set_title("Manipulation check: only the repaired loss at λ=1000\nmoves its own objective",
                      fontsize=10)
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Exp 19 — Encoder↔decoder alignment settled: explicit alignment loss "
                 "does not move the corrected metric", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp19_alignment_settled.png")


if __name__ == "__main__":
    main()
