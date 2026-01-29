import torch
import torch.nn.functional as F
from conditional import VAE
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cpu")
model = VAE().to(device)
model.load_state_dict(torch.load("conditional_vae.pt", map_location=device))
model.eval()

# Load one sample (x, label y)
transform = transforms.ToTensor()
dataset = datasets.MNIST(root="./data", train=False, transform=transform)
x, y_label = dataset[0]
x = x.view(-1)
y_label = torch.tensor([y_label], device=device)  # keep as batch of size 1

# Number of top eigenpairs to keep per latent
k_values = [0, 1, 5, 10, 20]

# Step 1: compute full latent
with torch.no_grad():
    mu_full, logvar_full = model.encoder(x.unsqueeze(0), y_label)
    z_full = mu_full.clone()
    recon_full = model.decoder(z_full, y_label)
    loss_full = F.mse_loss(recon_full, x.unsqueeze(0))
    print(f"Full latent reconstruction loss : {loss_full.item():.6f}")

# Step 2: decompose all latents
vals, vecs = model.encoder.decompose()

# Step 3: loop over k values and truncate
for k in k_values:
    z_trunc = torch.zeros_like(z_full)

    for latent_idx in range(z_full.shape[1]):
        if k == 0:
            mu_trunc = torch.tensor(0.0)
        else:
            v = vals[latent_idx]
            V = vecs[latent_idx]
            order = v.abs().argsort(descending=True)
            v_top = v[order][:k]
            V_top = V[order][:k]
            proj = torch.matmul(V_top, x)
            mu_trunc = torch.sum(v_top * proj**2)

        z_trunc[0, latent_idx] = mu_trunc

    # Step 4: decode
    with torch.no_grad():
        recon_trunc = model.decoder(z_trunc, y_label)
        loss = F.mse_loss(recon_trunc, x.unsqueeze(0))
        print(f"k={k:2d}, truncated reconstruction loss: {loss.item():.6f}")

    # Optional: visualize
    plt.imshow(recon_trunc.view(28,28), cmap='gray')
    plt.title(f"Truncated reconstruction k={k}")
    plt.axis('off')
    plt.show()
