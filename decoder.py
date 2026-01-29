# ============================
# bilinear_vae.py
# ============================

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from jaxtyping import Float

# ----------------------------
# Layers (UNCHANGED)
# ----------------------------

class Bilinear(nn.Linear):
    """A bilinear layer with optional gate"""
    def __init__(self, d_in: int, d_out: int, bias=False, gate=None) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.gate = {"relu": nn.ReLU(), "silu": nn.SiLU(),
                     "gelu": nn.GELU(), None: nn.Identity()}[gate]

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        left, right = super().forward(x).chunk(2, dim=-1)
        return self.gate(left) * right

    @property
    def w_l(self):
        return self.weight.chunk(2, dim=0)[0]

    @property
    def w_r(self):
        return self.weight.chunk(2, dim=0)[1]


class Linear(nn.Linear):
    """A linear layer with optional gate"""
    def __init__(self, d_in: int, d_out: int, bias=False, gate=None) -> None:
        super().__init__(d_in, d_out, bias=bias)
        self.gate = {"relu": nn.ReLU(), "silu": nn.SiLU(),
                     "gelu": nn.GELU(), None: nn.Identity()}[gate]

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return self.gate(super().forward(x))


class MLP(nn.Module):
    """MLP supporting bilinear or linear"""
    def __init__(self, d_model: int, d_hidden: int, bias=False, bilinear=True, gate=None):
        super().__init__()
        self.w = (Bilinear if bilinear else Linear)(
            d_model, d_hidden, bias=bias, gate=gate
        )
        self.p = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.p(self.w(x))


class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)


class Norm(nn.Module):
    def __init__(self, norm=True):
        super().__init__()
        self.norm = RMSNorm() if norm else nn.Identity()

    def forward(self, x):
        return self.norm(x)


# ----------------------------
# VAE
# ----------------------------

class VAE(nn.Module):
    def __init__(
        self,
        x_dim: int = 784,
        z_dim: int = 32,
        d_hidden: int = 256,
        n_layers: int = 2,
        bias: bool = False,
        gate=None,
    ):
        super().__init__()

        # Encoder
        enc = []
        for _ in range(n_layers):
            enc += [
                MLP(x_dim, d_hidden, bias=bias, bilinear=True, gate=gate),
                Norm(True),
            ]
        self.encoder = nn.Sequential(*enc)
        self.mu = nn.Linear(x_dim, z_dim, bias=bias)
        self.logvar = nn.Linear(x_dim, z_dim, bias=bias)

        # Decoder (BILINEAR)
        dec = []
        for _ in range(n_layers):
            dec += [
                MLP(z_dim, d_hidden, bias=bias, bilinear=True, gate=gate),
                Norm(True),
            ]
        self.decoder = nn.Sequential(*dec)
        self.out = nn.Linear(z_dim, x_dim, bias=bias)

    def sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward(self, x: Tensor):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.sample(mu, logvar)
        h_dec = self.decoder(z)
        x_hat = self.out(h_dec)
        return x_hat, mu, logvar


# ----------------------------
# Loss
# ----------------------------

def vae_loss(x_hat, x, mu, logvar):
    recon = torch.mean((x_hat - x) ** 2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl, recon, kl


# ----------------------------
# Train
# ----------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # flatten
    ])

    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = VAE(
        x_dim=784,
        z_dim=32,
        d_hidden=256,
        n_layers=2,
        gate=None,          # IMPORTANT: keep None for bilinear
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
    for x, _ in loader:
        x = x.to(device)

        x_hat, mu, logvar = model(x)
        loss, recon, kl = vae_loss(x_hat, x, mu, logvar)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # ---- SAMPLE ----
    sample = sample_from_model(model, device, z_dim=32, n=1)
    sample_img = sample.view(28, 28).cpu()

    print(
        f"epoch {epoch:02d} | loss {loss.item():.4f} "
        f"| recon {recon.item():.4f} | kl {kl.item():.4f}"
    )

    # optional: save
    torch.save(sample_img, f"sample_epoch_{epoch}.pt")



if __name__ == "__main__":
    main()
