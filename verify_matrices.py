from encoder import VAE

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

from mlp import Bilinear, Linear, RMSNorm


device = torch.device("cpu")

model = VAE().to(device)
model.load_state_dict(torch.load("encoder_vae.pt"))
model.eval()

vals, vecs = model.encoder.decompose()


print("Any NaNs:", torch.isnan(vals).any().item())
print("Any Infs:", torch.isinf(vals).any().item())
print("Max |eigenvalue|:", vals.abs().max().item())
print("Mean eigenvalue:", vals.mean().item())
print("Std eigenvalue:", vals.std().item())

vals, _ = model.encoder.decompose()

for k in range(vals.shape[0]):
    eigs = vals[k].abs().sort(descending=True).values
    print(f"latent {k}: top 10 eigenvalues:", eigs[:10].tolist())

