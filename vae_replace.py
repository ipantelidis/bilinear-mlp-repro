# vae_replace.py

import torch
from encoder import VAE, elbo_loss
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def evaluate_latent(model, loader, latent_idx, k, device):
    model.eval()
    total_loss = 0.0
    total_count = 0

    # decompose ONCE
    vals, vecs = model.encoder.decompose()

    lam = vals[latent_idx]          # [d_hidden]
    V   = vecs[latent_idx]          # [d_hidden, d_input]

    idx = lam.abs().topk(k).indices
    lam = lam[idx]                  # [k]
    V   = V[idx]                    # [k, d_input]

    with torch.no_grad():
        for x, _ in loader:
            x = x.view(x.size(0), -1).to(device)

            # full encoder output
            mu, logvar = model.encoder(x)

            # replace ONLY one latent
            proj = x @ V.T                  # [batch, k]
            mu[:, latent_idx] = (proj ** 2) @ lam

            # deterministic evaluation
            z = mu
            logits = model.decoder(z)

            recon, kl = elbo_loss(x, logits, mu, logvar)
            total_loss += (recon + kl).item()
            total_count += x.size(0)

    # per-sample ELBO
    return total_loss / total_count


if __name__ == "__main__":
    device = torch.device("cpu")

    transform = transforms.ToTensor()
    test_set = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model = VAE().to(device)
    model.load_state_dict(torch.load("encoder_vae.pt", map_location=device))

    baseline = evaluate_latent(
        model, loader, latent_idx=0, k=400, device=device
    )

    for k in [1, 2, 5, 10, 20]:
        loss = evaluate_latent(
            model, loader, latent_idx=0, k=k, device=device
        )
        print(f"k={k:2d} | ELBO={loss:.4f} | Δ={loss - baseline:.4f}")
