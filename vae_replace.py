"""
If replace part of the model with the new low-rank eigendecomposition does it work? YES!!!
"""


import torch
from encoder import VAE, elbo_loss   # your file
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Q matrix for one latent dim μi(x)
def compute_Q_for_latent(encoder, latent_idx):
    W_L = encoder.block2.w_l          # [m, d_hidden]
    W_R = encoder.block2.w_r          # [m, d_hidden]
    w   = encoder.mu.weight[latent_idx]  # [m]

    Q = torch.einsum("mi,mj,m->ij", W_L, W_R, w)
    return 0.5 * (Q + Q.T)


# Truncate Q (eigen decompose q - top k eigenvectors + rebuild low rank approx
def truncate_Q(Q, k):
    Q_cpu = Q.detach().cpu()                 # move ONLY Q to CPU
    vals, vecs = torch.linalg.eigh(Q_cpu)
    idx = vals.abs().topk(k).indices
    Qk = (vecs[:, idx] * vals[idx]) @ vecs[:, idx].T
    return Qk.to(Q.device)                   # move back 



# Recompute mu_i from Q 
def compute_mu_i_from_Q(h, Qk):
    return torch.einsum("bi,ij,bj->b", h, Qk, h)



def evaluate_latent(model, loader, latent_idx, k, device):
    model.eval()
    total_loss = 0.0

    # STEP 3–5 once
    Q = compute_Q_for_latent(model.encoder, latent_idx)
    Qk = truncate_Q(Q, k)

    with torch.no_grad():
        for x, _ in loader:
            x = x.view(x.size(0), -1).to(device)

            # run encoder manually to get h
            h = model.encoder.embed(x)
            h = model.encoder.block1(h)
            h = model.encoder.block2(h)

            mu, logvar = model.encoder(x)   # Modify computation

            # replace ONE latent (step 6)
            mu[:, latent_idx] = compute_mu_i_from_Q(h, Qk)

            z = model.reparameterize(mu, logvar)
            logits = model.decoder(z)

            loss, _, _ = elbo_loss(x, logits, mu, logvar)
            total_loss += loss.item()

    return total_loss / len(loader)



if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.ToTensor()
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
    loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model = VAE().to(device)
    model.load_state_dict(torch.load("vae.pt", map_location=device))

    # baseline
    baseline = evaluate_latent(model, loader, latent_idx=0, k=400, device=device)

    for k in [1, 2, 5, 10, 20]:
        loss = evaluate_latent(model, loader, latent_idx=0, k=k, device=device)
        print(f"k={k:2d} | ELBO={loss:.4f} | Δ={loss - baseline:.4f}")
