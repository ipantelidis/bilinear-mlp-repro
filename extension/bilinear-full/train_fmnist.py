"""
train_fmnist.py — Train FullBilinearVAE on FashionMNIST.

Saves:
    checkpoints/fashion_mnist/model.pt         (best over 30 epochs)
    checkpoints/fashion_mnist/seeds/seed0..4.pt (5 independent seeds)

Run from bilinear-full/:
    python train_fmnist.py
"""

import json, sys, torch, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent))
from models import FullBilinearVAE

import os; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
DATA    = "/home/v25/ippa6201/bilinear-mlp-repro/data"
DEVICE  = "cuda:0" if torch.cuda.is_available() else "cpu"
EPOCHS  = 30
LR      = 1e-3
WD      = 0.01
NOISE   = 0.3
BETA    = 1.0
ANNEAL  = 15   # ramp beta 0→1 over first 15 epochs
BATCH   = 128
GRAD_CLIP = 5.0  # max grad norm — prevents the seed4-style divergence


def get_loaders():
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    tr = DataLoader(datasets.FashionMNIST(DATA, train=True,  download=False, transform=tf),
                    batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
    te = DataLoader(datasets.FashionMNIST(DATA, train=False, download=False, transform=tf),
                    batch_size=512,   shuffle=False, num_workers=2, pin_memory=True)
    return tr, te


def elbo_loss(recon, x, mu, logvar, beta=1.0):
    # Clamp guards the BCE against numerical blowup (seed4 once produced
    # out-of-[0,1] decoder outputs and hit the CUDA BCE assert).
    r = F.binary_cross_entropy(recon.clamp(1e-6, 1 - 1e-6), x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return r + beta * kl, r, kl


def run_epoch(model, loader, opt, beta, noise, training):
    model.train(training)
    tot = rec = klt = n = 0
    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(DEVICE)
            xi = (x + noise * torch.randn_like(x)).clamp(0, 1) if training and noise > 0 else x
            recon, mu, logvar = model(xi)
            loss, r, kl = elbo_loss(recon, x, mu, logvar, beta)
            if training:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            tot += loss.item(); rec += r.item(); klt += kl.item(); n += x.size(0)
    return tot / n, rec / n, klt / n


def train_one(ckpt_path, seed=None, label="model"):
    if Path(ckpt_path).exists():
        print(f"{label}: {ckpt_path} exists — skipping (delete to retrain)")
        return
    if seed is not None:
        torch.manual_seed(seed)
    model = FullBilinearVAE().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    tr_loader, te_loader = get_loaders()

    best = float("inf"); hist = []
    print(f"\n--- {label} ---")
    print(f"{'Ep':>4}  {'Train':>10}  {'Test':>10}  {'KL':>8}  {'β':>5}")
    print("-" * 45)

    for ep in range(1, EPOCHS + 1):
        beta = min(1.0, ep / ANNEAL) * BETA if ANNEAL > 0 else BETA
        tr_l, _, _ = run_epoch(model, tr_loader, opt, beta, NOISE, True)
        te_l, te_r, te_kl = run_epoch(model, te_loader, opt, beta, 0.0, False)
        sched.step()
        hist.append({"epoch": ep, "train": tr_l, "test": te_l, "kl": te_kl})
        if ep % 5 == 0 or ep == 1:
            print(f"{ep:>4}  {tr_l:>10.2f}  {te_l:>10.2f}  {te_kl:>8.2f}  {beta:>5.2f}")
        if te_l < best:
            best = te_l
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": ep, "best_test": best, "model_state": model.state_dict(),
                        "history": hist}, ckpt_path)
    print(f"  Best test: {best:.2f}  →  {ckpt_path}")
    return model


if __name__ == "__main__":
    # Main checkpoint
    train_one(Path("checkpoints/fashion_mnist/model.pt"), label="FashionMNIST main")

    # 5 seeds
    for s in range(5):
        train_one(Path(f"checkpoints/fashion_mnist/seeds/seed{s}.pt"),
                  seed=s, label=f"FashionMNIST seed{s}")

    print("\nAll done.")
