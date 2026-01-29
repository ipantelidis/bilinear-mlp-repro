import torch
import matplotlib.pyplot as plt
from conditional import VAE


@torch.no_grad()
def build_encoder_quadratic_forms(model):
    enc = model.encoder

    # Dimensions
    # E : (H, D)
    # Q : (K, H, H)
    E = enc.embed.weight
    L = enc.block1.w_l
    R = enc.block1.w_r
    P = enc.mu.weight
    Ey = enc.y_embed.weight

    # Hidden-space Q_k
    Q = torch.einsum("kh,hi,hj->kij", P, L, R)
    Q = 0.5 * (Q + Q.mT)

    K, H, _ = Q.shape
    D = E.shape[1]
    Y = Ey.shape[0]

    # Pixel-space quadratic term A_k = E^T Q_k E
    A = torch.empty(K, D, D, device=E.device)
    for k in range(K):
        A[k] = E.T @ Q[k] @ E

    # Linear term b_{k,y} = 2 E^T Q_k e_y
    b = torch.empty(K, Y, D, device=E.device)
    for k in range(K):
        for y in range(Y):
            b[k, y] = 2.0 * (E.T @ Q[k] @ Ey[y])

    # Constant term c_{k,y} = e_y^T Q_k e_y
    c = torch.empty(K, Y, device=E.device)
    for k in range(K):
        for y in range(Y):
            c[k, y] = Ey[y] @ Q[k] @ Ey[y]

    return Q, A, b, c


@torch.no_grad()
def mu_from_quadratic(A, b, c, x, y):
    """
    Exact reconstruction of mu(x,y)
    """
    B, D = x.shape
    K = A.shape[0]

    mu = torch.zeros(B, K, device=x.device)

    # Quadratic term
    mu += torch.einsum("bi,kij,bj->bk", x, A, x)

    # Linear + constant terms (explicit, correct)
    for i in range(B):
        yi = y[i].item()
        mu[i] += b[:, yi] @ x[i] + c[:, yi]

    return mu


@torch.no_grad()
def check_exactness(model, device="cpu", n=128):
    model.eval().to(device)

    Q, A, b, c = build_encoder_quadratic_forms(model)

    x = torch.rand(n, 784, device=device)
    y = torch.randint(0, 10, (n,), device=device)

    mu_true, _ = model.encoder(x, y)
    mu_hat = mu_from_quadratic(A, b, c, x, y)

    err = (mu_true - mu_hat).abs().max().item()
    print("max |mu_true - mu_hat| =", err)


@torch.no_grad()
def eigenvectors_for_latent_in_pixel_space(model, k=0):
    _, A, _, _ = build_encoder_quadratic_forms(model)
    Ak = 0.5 * (A[k] + A[k].T)
    vals, vecs = torch.linalg.eigh(Ak)
    return vals, vecs


# ---- RUN ----
device = torch.device("cpu")
model = VAE().to(device)
model.load_state_dict(torch.load("conditional_vae.pt", map_location=device))

check_exactness(model, device=device)

vals0, vecs0 = eigenvectors_for_latent_in_pixel_space(model, k=0)
print(vals0.abs().max().item(), vals0.abs().mean().item())






@torch.no_grad()
def plot_top_eigenvectors(model, k=0, top=9):
    _, A, _, _ = build_encoder_quadratic_forms(model)

    Ak = 0.5 * (A[k] + A[k].T)
    vals, vecs = torch.linalg.eigh(Ak)

    # take largest eigenvalues
    idx = torch.argsort(vals.abs(), descending=True)[:top]
    vecs = vecs[:, idx]

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        v = vecs[:, i].reshape(28, 28).cpu()
        ax.imshow(v, cmap="gray")
        ax.axis("off")
        ax.set_title(f"eig {i}")

    plt.tight_layout()
    plt.show()

plot_top_eigenvectors(model, k=0, top=9)




@torch.no_grad()
def sample_with_truncated_mu(
    model, A, b, c, y, k_trunc, n=9, device="cpu"
):
    model.eval().to(device)

    # sample real inputs just to get x-scale (not used semantically)
    x = torch.rand(n, 784, device=device)

    # build truncated mu
    mu = torch.zeros(n, model.encoder.mu.out_features, device=device)

    for k in range(mu.shape[1]):
        Ak = 0.5 * (A[k] + A[k].T)
        vals, vecs = torch.linalg.eigh(Ak)
        idx = torch.argsort(vals.abs(), descending=True)[:k_trunc]

        proj = x @ vecs[:, idx]
        mu[:, k] = (proj**2 * vals[idx]).sum(dim=1)

        for i in range(n):
            mu[i, k] += b[k, y[i]] @ x[i] + c[k, y[i]]

    # sample z and decode
    z = mu + torch.randn_like(mu)
    imgs = model.decoder(z, y)

    return imgs


from torchvision.utils import save_image

Q, A, b, c = build_encoder_quadratic_forms(model)

y = torch.tensor([0,1,2,3,4,5,6,7,8], device=device)

for k_trunc in [1, 2, 5, 10, 20, 50]:
    imgs = sample_with_truncated_mu(
        model, A, b, c, y, k_trunc, device=device
    )
    save_image(
        imgs.view(-1,1,28,28),
        f"samples_k{k_trunc}.png",
        nrow=3
    )
