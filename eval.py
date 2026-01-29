import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from encoder import VAE  # your VAE code with Encoder, Decoder

os.makedirs("outputs", exist_ok=True)

DEVICE = torch.device("cpu")  # safer for eigen-decomposition
DIGIT = 3  # example digit to visualize
LATENT_IDX = 0  # which latent dimension to analyze

# -----------------------------
# Load trained model
# -----------------------------
model = VAE().to(DEVICE)
model.load_state_dict(torch.load("encoder_vae.pt", map_location=DEVICE))
model.eval()

# -----------------------------
# Data
# -----------------------------
transform = transforms.ToTensor()
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(train_set, batch_size=128, shuffle=False)

# -----------------------------
# Eigen-decomposition
# -----------------------------
with torch.no_grad():
    E = model.encoder.embed.weight
    Wl = model.encoder.block1.w_l
    Wr = model.encoder.block1.w_r
    Wu_mu = model.encoder.mu.weight
    Wu_logvar = model.encoder.logvar.weight

    # --- MU slice
    B_mu = torch.einsum("i,oi,oj->ij", Wu_mu[LATENT_IDX], Wl, Wr)
    B_mu = 0.5 * (B_mu + B_mu.T)
    B_mu_pixel = E.T @ B_mu @ E
    B_mu_pixel = 0.5 * (B_mu_pixel + B_mu_pixel.T)
    vals_mu, vecs_mu = torch.linalg.eigh(B_mu_pixel)
    top_vec_mu = vecs_mu[:, -1] / vecs_mu[:, -1].norm()

    # --- LOGVAR slice
    B_var = torch.einsum("i,oi,oj->ij", Wu_logvar[LATENT_IDX], Wl, Wr)
    B_var = 0.5 * (B_var + B_var.T)
    B_var_pixel = E.T @ B_var @ E
    B_var_pixel = 0.5 * (B_var_pixel + B_var_pixel.T)
    vals_var, vecs_var = torch.linalg.eigh(B_var_pixel)
    top_vec_var = vecs_var[:, -1] / vecs_var[:, -1].norm()

# -----------------------------
# Visualize top eigenvectors
# -----------------------------
for name, vec in [("mu", top_vec_mu), ("logvar", top_vec_var)]:
    plt.imshow(vec.detach().view(28,28).cpu(), cmap="RdBu")
    plt.colorbar()
    plt.axis("off")
    plt.title(f"Top eigenvector ({name})")
    plt.savefig(f"outputs/top_eigenvector_{name}.png", dpi=300)
    plt.close()

# -----------------------------
# Eigenvector-guided traversal
# -----------------------------
# pick an example digit
for x, y in loader:
    mask = (y == DIGIT)
    if mask.any():
        x0 = x[mask][0].view(1,-1).to(DEVICE)
        break

values = torch.linspace(-2, 2, 7).to(DEVICE)
imgs = []

with torch.no_grad():
    for v in values:
        x_pert = (x0 + v * top_vec_mu.unsqueeze(0)).clamp(0,1)
        mu_p, _ = model.encoder(x_pert)
        out = model.decoder(mu_p)
        imgs.append(out.view(1,1,28,28))

imgs = torch.cat(imgs)
grid = make_grid(imgs, nrow=7)

plt.figure(figsize=(10,2))
plt.imshow(grid.permute(1,2,0).cpu(), cmap="gray")
plt.axis("off")
plt.title("Eigenvector-guided traversal")
plt.savefig("outputs/eigen_guided_traversal.png", dpi=300)
plt.close()

# -----------------------------
# Comparison plot
# -----------------------------
fig, axs = plt.subplots(1,2,figsize=(6,3))
axs[0].imshow(top_vec_mu.detach().view(28,28).cpu(), cmap="RdBu")
axs[0].set_title("Eigenvector (mu)")
axs[0].axis("off")

axs[1].imshow(imgs[3,0].cpu(), cmap="gray")
axs[1].set_title("Generated image")
axs[1].axis("off")

plt.tight_layout()
plt.savefig("outputs/eigen_vs_generation.png", dpi=300)
plt.close()

print("✔ All outputs saved to ./outputs/")
