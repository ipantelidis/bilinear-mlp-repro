import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bvae import VAE

# -----------------------------
# Setup
# -----------------------------
device = torch.device("cpu")

batch_size = 256

transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: (x > 0.5).float()
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Load trained model
# -----------------------------
model = VAE().to(device)
model.load_state_dict(torch.load("bvae.pt", map_location=device))
model.eval()

# -----------------------------
# Encode dataset
# -----------------------------
Z = []
Y = []

with torch.no_grad():
    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        mu, _ = model.encoder(x)
        Z.append(mu.cpu())
        Y.append(y)

Z = torch.cat(Z)          # (N, d_latent)
Y = torch.cat(Y)          # (N,)

# -----------------------------
# PCA to 2D (pure PyTorch)
# -----------------------------
Z_centered = Z - Z.mean(dim=0)

U, S, V = torch.pca_lowrank(Z_centered, q=2)
Z2 = Z_centered @ V[:, :2]   # (N, 2)

Z2 = Z2.numpy()
Y = Y.numpy()

# -----------------------------
# Plot latent scatter
# -----------------------------
plt.figure(figsize=(6, 6))
plt.scatter(Z2[:, 0], Z2[:, 1], c=Y, s=3, cmap="tab10")
plt.colorbar()
plt.title("Latent space (mu, PCA)")
plt.tight_layout()
plt.show()

# -----------------------------
# 2D latent traversal (dims 0,1)
# -----------------------------
grid = 20
lims = (-3, 3)

z1 = np.linspace(*lims, grid)
z2 = np.linspace(*lims, grid)

images = []

with torch.no_grad():
    for y in z2:
        row = []
        for x in z1:
            z = torch.zeros(1, model.encoder.mu.out_features, device=device)
            z[0, 0] = x
            z[0, 1] = y
            img = model.decoder(z).view(28, 28)
            row.append(img.cpu())
        images.append(torch.cat(row, dim=1))

images = torch.cat(images, dim=0)

plt.figure(figsize=(6, 6))
plt.imshow(images, cmap="gray")
plt.axis("off")
plt.title("Latent traversal (z0, z1)")
plt.show()

"""# -----------------------------
# Per-dimension traversal
# -----------------------------
vals = [-3, 0, 3]
d_latent = model.encoder.mu.out_features

with torch.no_grad():
    for i in range(d_latent):
        imgs = []
        for v in vals:
            z = torch.zeros(1, d_latent, device=device)
            z[0, i] = v
            img = model.decoder(z).view(28, 28)
            imgs.append(img.cpu())

        imgs = torch.cat(imgs, dim=1)

        plt.figure(figsize=(3, 1))
        plt.imshow(imgs, cmap="gray")
        plt.axis("off")
        plt.title(f"latent dim {i}")
        plt.show()
"""


mus = []
labels = []

with torch.no_grad():
    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        mu, _ = model.encoder(x)
        mus.append(mu.cpu())
        labels.append(y)

mus = torch.cat(mus)      # [N, d_latent]
labels = torch.cat(labels)

# -----------------------------
# Pick top-variance dims
# -----------------------------
var = mus.var(dim=0)
i, j = torch.topk(var, 2).indices.tolist()

print(f"Plotting z[{i}] vs z[{j}]")

# -----------------------------
# Normalize (important)
# -----------------------------
mus = (mus - mus.mean(0)) / mus.std(0)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6, 6))
sc = plt.scatter(
    mus[:, i],
    mus[:, j],
    c=labels,
    s=4,
    cmap="tab10",
    alpha=0.6,
)

plt.xlabel(f"z[{i}]")
plt.ylabel(f"z[{j}]")
plt.colorbar(sc)
plt.title("β-VAE posterior means μ(x)")
plt.tight_layout()
plt.show()