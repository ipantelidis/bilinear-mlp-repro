"""
train_v3.py — Train the repaired-loss "v3" seeds for exp19.

The shipped v3 prototype augments the v2 architecture (== FullBilinearVAE in
this project) with a per-batch encoder↔decoder alignment loss

    L_align = 1 − |cos( ∂μ_k/∂x , ∂output/∂z_k )|      (one random k per batch)

but its implementation detaches BOTH Jacobians (encoder via .detach(),
decoder under torch.no_grad()), so the returned scalar has grad_fn=None and
align_lambda · L_align contributes ZERO gradient — every checkpoint trained
with it is effectively a v2 run.  Exp19 verifies this programmatically
against a verbatim copy of the shipped loss.

FixedV3 below is the REPAIRED, differentiable version: the encoder Jacobian
is taken with create_graph=True autograd and the decoder Jacobian with
differentiable finite differences, so gradient actually flows through the
alignment term into both codec sides.

Hyperparameters mirror the v2 main/seed checkpoints:
    EPOCHS=30  LR=1e-3  WD=0.01  BETA=1.0  KL-anneal=15 epochs  NOISE=0.3
    batch 128, BCE reconstruction, AdamW + cosine LR, best-test checkpoint.

Usage (from bilinear-full/):
    python train_v3.py probe                 # 6-epoch lambda probe (1, 100, 1000)
    python train_v3.py train LAM S0 [S1 ..]  # full 30-epoch seeds

    Optional overrides (mainly for smoke tests):
        --epochs N --ckpt-dir DIR --data-dir DIR --device DEV

Checkpoints (exp19's v3-fixed groups):
    checkpoints/mnist/v3/lam{L}_seed{S}.pt
    Format: {"epoch", "model_state", "history", "config"} — what exp19 loads.
    Existing checkpoint files are never overwritten (the run is skipped).
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import FullBilinearVAE

HERE     = Path(__file__).resolve().parent
DATA_DIR = str(HERE.parents[1] / "data")           # <repo>/data
CKPT_DIR = HERE / "checkpoints/mnist/v3"
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"

# Mirror the hyperparameters used for the clean-project v2 seeds
EPOCHS, LR, WD, BETA, ANNEAL, NOISE = 30, 1e-3, 0.01, 1.0, 15, 0.3
BATCH = 128


class FixedV3(FullBilinearVAE):
    """v3 with a differentiable alignment loss (gradient actually flows)."""

    def alignment_loss(self, x, z, delta=0.3):
        B = x.size(0)
        k = torch.randint(0, self.d_latent, (1,)).item()
        # encoder Jacobian dmu_k/dx, kept in the graph
        x_ag = x.detach().requires_grad_(True)
        mu_ag, _ = self.encode(x_ag)
        enc_jac = torch.autograd.grad(mu_ag[:, k].sum(), x_ag,
                                      create_graph=True)[0]         # (B, 784)
        # decoder Jacobian doutput/dz_k via differentiable finite differences
        z_det = z.detach()
        e_k = torch.zeros_like(z_det)
        e_k[:, k] = delta
        dec_jac = (self.decoder(z_det + e_k)
                   - self.decoder(z_det - e_k)) / (2 * delta)       # (B, 784)
        cos = F.cosine_similarity(enc_jac.view(B, -1), dec_jac.view(B, -1), dim=1)
        return 1.0 - cos.abs().mean()


# ── data ───────────────────────────────────────────────────────────────────

def get_loaders(data_dir, batch_size=BATCH):
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST(data_dir, train=True,  download=True, transform=tfm)
    te = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True),
            DataLoader(te, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True))


# ── ELBO + training loop (ported from the prototype's train.py) ────────────

def elbo_loss(recon, x, mu, logvar, beta=1.0):
    recon_l = F.binary_cross_entropy(recon, x, reduction="sum")
    kl_l    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_l + beta * kl_l, recon_l, kl_l


def run_epoch(model, loader, optimizer, device, beta=1.0, noise_std=0.0,
              training=True, align_lambda=0.0):
    model.train(training)
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0
    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(device)
            if training and noise_std > 0:
                x_in = (x + noise_std * torch.randn_like(x)).clamp(0, 1)
            else:
                x_in = x
            recon, mu, logvar = model(x_in)
            loss, recon_l, kl_l = elbo_loss(recon, x, mu, logvar, beta)
            if training and align_lambda > 0:
                z = model.reparametrize(mu.detach(), logvar.detach())
                loss = loss + align_lambda * model.alignment_loss(x, z)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            totals["total"] += loss.item()
            totals["recon"] += recon_l.item()
            totals["kl"]    += kl_l.item()
            n += x.size(0)
    return {k: v / n for k, v in totals.items()}


def train(model, train_loader, test_loader, *, epochs, lr=LR, weight_decay=WD,
          beta=BETA, kl_anneal_epochs=0, align_lambda=0.0, noise_std=NOISE,
          device=DEVICE, checkpoint_dir=str(CKPT_DIR), run_name="model"):
    """Train, saving the best-test-loss checkpoint as {run_name}.pt."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history, best_loss = [], float("inf")
    print(f"\nTraining  : {run_name}")
    print(f"β={beta}  kl_anneal={kl_anneal_epochs}  align_lambda={align_lambda}  "
          f"noise_std={noise_std}  epochs={epochs}  lr={lr}  wd={weight_decay}")
    print(f"{'Epoch':>5}  {'Train':>10}  {'Test':>10}  {'KL':>8}  {'β_eff':>6}")
    print("-" * 48)

    for epoch in range(1, epochs + 1):
        if kl_anneal_epochs > 0 and epoch <= kl_anneal_epochs:
            beta_eff = beta * epoch / kl_anneal_epochs
        else:
            beta_eff = beta
        tr = run_epoch(model, train_loader, optimizer, device, beta=beta_eff,
                       noise_std=noise_std, training=True,
                       align_lambda=align_lambda)
        te = run_epoch(model, test_loader, optimizer, device, beta=beta_eff,
                       noise_std=0.0, training=False, align_lambda=0.0)
        scheduler.step()
        history.append({"epoch": epoch, "beta_eff": beta_eff,
                        "train_total": tr["total"], "train_recon": tr["recon"],
                        "train_kl": tr["kl"], "test_total": te["total"],
                        "test_recon": te["recon"], "test_kl": te["kl"],
                        "lr": scheduler.get_last_lr()[0]})
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>5}  {tr['total']:>10.2f}  {te['total']:>10.2f}  "
                  f"{te['kl']:>8.2f}  {beta_eff:>6.3f}")
        if te["total"] < best_loss:
            best_loss = te["total"]
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "history": history,
                        "config": {"beta": beta, "noise_std": noise_std,
                                   "recon_loss": "bce", "lr": lr,
                                   "weight_decay": weight_decay,
                                   "align_lambda": align_lambda,
                                   "run_name": run_name}},
                       ckpt_dir / f"{run_name}.pt")

    with open(ckpt_dir / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest test loss : {best_loss:.2f}")
    print(f"Checkpoint saved → {ckpt_dir / run_name}.pt")
    return history


# ── evaluation: the loss's own Jacobian objective ──────────────────────────

def jacobian_cos(model, x, delta=0.3):
    """Mean |cos| over all k on one batch (evaluation, not training)."""
    B = x.size(0)
    with torch.no_grad():
        mu, _ = model.encode(x)
    vals = []
    for k in range(model.d_latent):
        x_ag = x.detach().requires_grad_(True)
        mu_ag, _ = model.encode(x_ag)
        mu_ag[:, k].sum().backward()
        enc_jac = x_ag.grad.detach()
        e_k = torch.zeros_like(mu)
        e_k[:, k] = delta
        with torch.no_grad():
            dec_jac = (model.decoder(mu + e_k) - model.decoder(mu - e_k)) / (2 * delta)
        vals.append(F.cosine_similarity(enc_jac.view(B, -1),
                                        dec_jac.view(B, -1), dim=1).abs().mean().item())
    return sum(vals) / len(vals)


# ── entry points ───────────────────────────────────────────────────────────

def train_one(lam, seed, epochs, run_name, *, ckpt_dir, data_dir, device):
    torch.manual_seed(seed)
    model = FixedV3()
    tr, te = get_loaders(data_dir)
    train(model, tr, te, epochs=epochs,
          kl_anneal_epochs=ANNEAL if epochs >= EPOCHS else epochs // 2,
          align_lambda=lam, device=device,
          checkpoint_dir=str(ckpt_dir), run_name=run_name)
    model.eval().cpu()
    x, _ = next(iter(te))
    jc = jacobian_cos(model, x[:256])
    print(f"[{run_name}] final-epoch jacobian |cos| = {jc:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("mode", choices=["probe", "train"],
                    help="probe: short lambda sweep; train: full seeds")
    ap.add_argument("args", nargs="*",
                    help="for 'train': LAM SEED [SEED ...] (default seeds 0 1 2)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epoch count (default: 30 for train, 6 for probe)")
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR),
                    help="checkpoint directory (default: checkpoints/mnist/v3)")
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="dataset directory (default: <repo>/data)")
    ap.add_argument("--device", default=DEVICE)
    a = ap.parse_args()
    ckpt_dir = Path(a.ckpt_dir)

    if a.mode == "probe":
        for lam in [1.0, 100.0, 1000.0]:
            epochs = a.epochs or 6
            print(f"\n===== probe lambda={lam} ({epochs} epochs, seed 0) =====")
            name = f"probe_lam{lam:g}"
            if (ckpt_dir / f"{name}.pt").exists():
                print(f"{name} exists — skipping (never overwritten)")
                continue
            train_one(lam, seed=0, epochs=epochs, run_name=name,
                      ckpt_dir=ckpt_dir, data_dir=a.data_dir, device=a.device)
    else:
        if not a.args:
            ap.error("train mode needs: LAM [SEED ...]")
        lam = float(a.args[0])
        seeds = [int(s) for s in a.args[1:]] or [0, 1, 2]
        for s in seeds:
            name = f"lam{lam:g}_seed{s}"
            if (ckpt_dir / f"{name}.pt").exists():
                print(f"{name} exists — skipping (never overwritten)")
                continue
            epochs = a.epochs or EPOCHS
            print(f"\n===== train lambda={lam} seed={s} ({epochs} epochs) =====")
            train_one(lam, seed=s, epochs=epochs, run_name=name,
                      ckpt_dir=ckpt_dir, data_dir=a.data_dir, device=a.device)


if __name__ == "__main__":
    main()
