# ============================================================
# Interpretable BiMLP-VAE with Eigenvector-Guided Traversals
# ============================================================
# This version incorporates the following fixes:
# 1. KL weight increased to enforce latent geometry
# 2. Two-layer bilinear encoder (decompose first layer only)
# 3. Eigenvector-based *input* perturbation, not latent-axis traversal
# 4. Eigen-decomposition for both mu and logvar slices
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
os.makedirs("outputs", exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import einsum
from torchvision.utils import make_grid

from image.datasets import MNIST
from shared.components import Bilinear, Linear

# ============================================================
# Config
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 10
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
KL_WEIGHT = 1e-3
DIGIT = 3
LATENT_IDX = 3

torch.manual_seed(42)

# ============================================================
# BiMLP Encoder (2-layer, decompose first layer only)
# ============================================================
class BiEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Bilinear(784, 256)
        self.fc2 = Bilinear(256, 256)
        self.mu = Linear(256, LATENT_DIM)
        self.logvar = Linear(256, LATENT_DIM)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.mu(h), self.logvar(h)

# ============================================================
# Decoder
# ============================================================
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)

# ============================================================
# VAE
# ============================================================
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BiEncoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# ============================================================
# Loss
# ============================================================
def vae_loss(recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KL_WEIGHT * kl

# ============================================================
# Data
# ============================================================
train = MNIST(train=True)
loader = torch.utils.data.DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True
)

# ============================================================
# Train
# ============================================================
vae = VAE().to(DEVICE)
opt = torch.optim.Adam(vae.parameters(), lr=LR)

vae.train()
for epoch in range(EPOCHS):
    losses = []
    for x, _ in loader:
        x = x.view(x.size(0), -1).to(DEVICE)
        recon, mu, logvar = vae(x)
        loss = vae_loss(recon, x, mu, logvar)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"Epoch {epoch:03d} | Loss {sum(losses)/len(losses):.4f}")

vae.eval()
torch.set_grad_enabled(False)

# ============================================================
# Reconstruction sanity check
# ============================================================
x, _ = next(iter(loader))
x = x[:16].view(16, -1).to(DEVICE)

with torch.no_grad():
    recon, _, _ = vae(x)

grid = make_grid(
    torch.cat([x.view(-1,1,28,28), recon.view(-1,1,28,28)], 0),
    nrow=16
)

plt.figure(figsize=(12,2))
plt.imshow(grid.permute(1,2,0).cpu(), cmap="gray")
plt.axis("off")
plt.savefig("outputs/reconstructions.png", dpi=300)
plt.close()

# ============================================================
# Bilinear slice decomposition (first layer only)
# ============================================================
Wl, Wr = vae.encoder.fc1.w_l, vae.encoder.fc1.w_r
Wu_mu = vae.encoder.mu.weight
Wu_var = vae.encoder.logvar.weight

# --- MU slice
B_mu = einsum(
    Wu_mu[LATENT_IDX],
    Wl,
    Wr,
    "o, o i1, o i2 -> i1 i2"
)
B_mu = 0.5 * (B_mu + B_mu.T)

vals_mu, vecs_mu = torch.linalg.eigh(B_mu)
top_vec_mu = vecs_mu[:, -1]
top_vec_mu = top_vec_mu / top_vec_mu.norm()

# --- LOGVAR slice
B_var = einsum(
    Wu_var[LATENT_IDX],
    Wl,
    Wr,
    "o, o i1, o i2 -> i1 i2"
)
B_var = 0.5 * (B_var + B_var.T)

vals_var, vecs_var = torch.linalg.eigh(B_var)
top_vec_var = vecs_var[:, -1]
top_vec_var = top_vec_var / top_vec_var.norm()

# ============================================================
# Visualize eigenvectors
# ============================================================
for name, vec in [("mu", top_vec_mu), ("logvar", top_vec_var)]:
    img = vec.view(28, 28)
    plt.imshow(img.cpu(), cmap="RdBu")
    plt.colorbar()
    plt.axis("off")
    plt.savefig(f"outputs/top_eigenvector_{name}.png", dpi=300)
    plt.close()

# ============================================================
# Eigenvector-guided INPUT perturbation
# ============================================================
x0 = train.x[train.y == DIGIT][0].view(1,-1).to(DEVICE)

values = torch.linspace(-2, 2, 7).to(DEVICE)
imgs = []

for v in values:
    x_perturbed = x0 + v * top_vec_mu.unsqueeze(0)
    mu_p, _ = vae.encoder(x_perturbed)
    img = vae.decoder(mu_p).view(1,28,28)
    imgs.append(img)

imgs = torch.cat(imgs).unsqueeze(1)

plt.figure(figsize=(10,2))
plt.imshow(make_grid(imgs, nrow=7).permute(1,2,0).cpu(), cmap="gray")
plt.axis("off")
plt.savefig("outputs/eigen_guided_traversal.png", dpi=300)
plt.close()

# ============================================================
# Comparison plot
# ============================================================
fig, axs = plt.subplots(1,2,figsize=(6,3))

axs[0].imshow(top_vec_mu.view(28,28).cpu(), cmap="RdBu")
axs[0].set_title("Projected Eigenvector (μ)")
axs[0].axis("off")

axs[1].imshow(imgs[3,0].cpu(), cmap="gray")
axs[1].set_title("Generated Image")
axs[1].axis("off")

plt.tight_layout()
plt.savefig("outputs/eigen_vs_generation.png", dpi=300)
plt.close()

print("✔ All outputs saved to ./outputs/")