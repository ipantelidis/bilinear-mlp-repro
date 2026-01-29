import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from encoder import VAE


# ---------- importance: rank latents ----------
@torch.no_grad()
def latent_importance(model, loader, device):
    model.eval()
    acc = None
    n = 0

    for x, _ in loader:
        x = x.view(x.size(0), -1).to(device)
        mu, _ = model.encoder(x)
        s = mu.pow(2).sum(0)
        acc = s if acc is None else acc + s
        n += mu.size(0)

    return acc / n


# ---------- eval with latent truncation ----------
@torch.no_grad()
def eval_truncated(model, loader, topk_idx, device):
    model.eval()
    total = 0.0

    for x, _ in loader:
        x = x.view(x.size(0), -1).to(device)

        logits, mu, logvar = model(x)
        z = model.reparameterize(mu, logvar)

        z_trunc = torch.zeros_like(z)
        z_trunc[:, topk_idx] = z[:, topk_idx]

        logits = model.decoder(z_trunc)

        recon = F.binary_cross_entropy_with_logits(
            logits, x, reduction="sum"
        ) / x.size(0)

        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ) / x.size(0)

        total += (recon + kl).item()

    return total / len(loader)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # data
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # load model
    model = VAE().to(device)
    model.load_state_dict(torch.load("vae.pt", map_location=device))
    model.eval()

    # compute importance
    imp = latent_importance(model, loader, device)
    all_idx = torch.arange(imp.numel(), device=device)

    # baseline
    baseline = eval_truncated(model, loader, all_idx, device)

    print(f"baseline ELBO = {baseline:.4f}")

    for k in [1, 2, 5, 10, 20]:
        topk = imp.topk(k).indices
        loss = eval_truncated(model, loader, topk, device)
        print(f"k={k:2d} | ELBO={loss:.4f} | Δ={loss - baseline:.4f}")
