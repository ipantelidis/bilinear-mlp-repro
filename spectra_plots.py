import torch
import matplotlib.pyplot as plt
from encoder import VAE

device = torch.device("cpu")
model = VAE().to(device)

vals, _ = model.encoder.decompose()

for k in range(vals.shape[0]):
    eigs = vals[k].abs().sort(descending=True).values
    plt.plot(eigs.cpu(), alpha=0.5)

plt.yscale("log")
plt.xlabel("eigenvalue index")
plt.ylabel("|eigenvalue|")
plt.title("Eigenvalue decay for all latents")
plt.show()
