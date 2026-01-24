import torch
import os
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Build Q for ONE latent
# ------------------------------------------------------------
def compute_Q_for_latent(encoder, latent_idx):
    W_L = encoder.block2.w_l
    W_R = encoder.block2.w_r
    w   = encoder.mu.weight[latent_idx]

    Q = torch.einsum("mi,mj,m->ij", W_L, W_R, w)
    return 0.5 * (Q + Q.T)


# ------------------------------------------------------------
# 2. Eigendecomposition (CPU only)
# ------------------------------------------------------------
def decompose_Q(Q):
    Q_cpu = Q.detach().cpu()
    vals, vecs = torch.linalg.eigh(Q_cpu)
    return vals, vecs


# ------------------------------------------------------------
# 3. Decompose VAE (latent-wise)
# ------------------------------------------------------------
def decompose_vae(model):
    encoder = model.encoder
    n_latents = encoder.mu.weight.shape[0]

    all_vals = []
    all_vecs = []

    for latent_idx in range(n_latents):
        Q = compute_Q_for_latent(encoder, latent_idx)
        vals, vecs = decompose_Q(Q)
        all_vals.append(vals)
        all_vecs.append(vecs)

    return torch.stack(all_vals), torch.stack(all_vecs)


# ------------------------------------------------------------
# 4. Plot eigenspectrum (VAE-safe)
# ------------------------------------------------------------
def plot_eigenspectrum_vae_matplotlib(
    model,
    latent_idx,
    eigenvectors=3,
    eigenvalues=20,
    out_dir="eigenspectrum",
):
    os.makedirs(out_dir, exist_ok=True)

    vals, vecs = decompose_vae(model)
    vals = vals[latent_idx]
    vecs = vecs[latent_idx]

    embed_W = model.encoder.embed.weight.detach().cpu()

    negative = torch.arange(eigenvectors)
    positive = -1 - negative

    # -------- eigenvalue spectrum --------
    plt.figure(figsize=(4, 3))
    plt.plot(vals[-eigenvalues:].flip(0).numpy())
    plt.plot(vals[:eigenvalues].numpy())
    plt.axhline(0, color="black", linewidth=0.5)
    plt.title(f"Latent {latent_idx} eigenvalues")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/latent_{latent_idx}_spectrum.png")
    plt.close()

    # -------- eigenvector images --------
    for sign, indices in [("pos", positive), ("neg", negative)]:
        fig, axes = plt.subplots(1, eigenvectors, figsize=(2 * eigenvectors, 2))
        if eigenvectors == 1:
            axes = [axes]

        for ax, idx in zip(axes, indices):
            img = (embed_W.T @ vecs[idx]).view(28, 28)
            ax.imshow(img.flip(0).numpy(), cmap="RdBu")
            ax.axis("off")

        plt.suptitle(f"Latent {latent_idx} {sign} eigenvectors")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/latent_{latent_idx}_{sign}.png")
        plt.close()



# ------------------------------------------------------------
# 5. Run
# ------------------------------------------------------------
if __name__ == "__main__":
    from encoder import VAE

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = VAE().to(device)
    model.load_state_dict(torch.load("decoder_vae.pt", map_location=device))
    model.eval()

    plot_eigenspectrum_vae_matplotlib(model, latent_idx=0)
