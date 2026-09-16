"""
train_epochs.py — Train FullBilinearVAE on MNIST, saving a checkpoint every epoch.

Used by exp13_training_dynamics.py to track how spectral structure emerges
during training.

Saves:
    checkpoints/mnist/epochs/epoch01.pt .. epoch20.pt

Run from bilinear-full/:
    python train_epochs.py
"""

import sys, torch, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).parent))
from models import FullBilinearVAE

DATA   = "/home/v25/ippa6201/bilinear-mlp-repro/data"
import os; os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
LR     = 1e-3
WD     = 0.01
NOISE  = 0.3
BETA   = 1.0
ANNEAL = 10
BATCH  = 128
SEED   = 42


def get_loaders():
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Lambda(lambda x: x.view(-1))])
    tr = DataLoader(datasets.MNIST(DATA, train=True,  download=False, transform=tf),
                    batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
    te = DataLoader(datasets.MNIST(DATA, train=False, download=False, transform=tf),
                    batch_size=512,   shuffle=False, num_workers=2, pin_memory=True)
    return tr, te


def elbo_loss(recon, x, mu, logvar, beta=1.0):
    r  = F.binary_cross_entropy(recon, x, reduction="sum")
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
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); rec += r.item(); klt += kl.item(); n += x.size(0)
    return tot / n, rec / n, klt / n


if __name__ == "__main__":
    torch.manual_seed(SEED)
    model = FullBilinearVAE().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    tr_loader, te_loader = get_loaders()

    out_dir = Path("checkpoints/mnist/epochs")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training for dynamics analysis ({EPOCHS} epochs, saving every epoch)")
    print(f"{'Ep':>4}  {'Train':>10}  {'Test':>10}  {'KL':>8}  {'β':>5}")
    print("-" * 45)

    for ep in range(1, EPOCHS + 1):
        beta = min(1.0, ep / ANNEAL) * BETA if ANNEAL > 0 else BETA
        tr_l, _, _ = run_epoch(model, tr_loader, opt, beta, NOISE, True)
        te_l, _, te_kl = run_epoch(model, te_loader, opt, beta, 0.0, False)
        sched.step()
        print(f"{ep:>4}  {tr_l:>10.2f}  {te_l:>10.2f}  {te_kl:>8.2f}  {beta:>5.2f}")

        torch.save({"epoch": ep, "test_total": te_l, "test_kl": te_kl,
                    "model_state": model.state_dict()},
                   out_dir / f"epoch{ep:02d}.pt")

    print(f"\nCheckpoints saved to {out_dir}/")
