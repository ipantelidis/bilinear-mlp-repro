import torch
from encoder import VAE
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



device = torch.device("cpu")
model = VAE().to(device)
model.load_state_dict(torch.load("encoder_vae.pt", map_location=device))
model.eval()


transform = transforms.ToTensor()
dataset = datasets.MNIST(root="./data", train=False, transform=transform)
x, _ = dataset[0]
x = x.view(-1)

with torch.no_grad():
    h = model.encoder.embed(x) 

# dcompose all latents once
vals, vecs = model.encoder.decompose()

with torch.no_grad():
    mu_full, _ = model.encoder(x.unsqueeze(0))
    z_full = mu_full.clone()

    # -------- NEW compute full quadratic + offset --------
    full_quad = torch.zeros_like(z_full)
    for ℓ in range(z_full.shape[1]):
        proj = vecs[ℓ] @ h
        full_quad[0, ℓ] = torch.sum(vals[ℓ] * proj**2)

    offset = z_full - full_quad
    # ------------------------------------------------------


def truncate_all_latents(h, vals, vecs, offset, k=None):
    z_trunc = torch.zeros_like(offset)

    for latent_idx in range(z_trunc.shape[1]):
        v = vals[latent_idx]
        V = vecs[latent_idx]

        if k is None:
            v_top = v
            V_top = V
        else:
            order = v.abs().argsort(descending=True)
            v_top = v[order][:k]
            V_top = V[order][:k]

        proj = V_top @ h
        z_trunc[0, latent_idx] = (
            torch.sum(v_top * proj**2) + offset[0, latent_idx]
        )

    return z_trunc


ks = [1, 5, 10, 20, None]

for k in ks:
    z_trunc = truncate_all_latents(h, vals, vecs, offset, k)
    recon = model.decoder(z_trunc)
    loss = F.mse_loss(recon, x.unsqueeze(0))

    label = "full" if k is None else k
    print(f"k = {label} --> reconstruction loss = {loss.item():.6f}")


    if k in [10, 20]:
        plt.imshow(recon.detach().view(28, 28), cmap="gray")
        plt.title(f"Reconstruction with k={k}")
        plt.axis("off")
        plt.show()