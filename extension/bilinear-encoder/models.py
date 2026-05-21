"""
models.py — BilinearVAE architecture.

The encoder uses a bilinear layer (element-wise product of two linear maps)
which makes its entire computation a degree-2 polynomial in the input.
This means for any latent direction μ*, the activation f(x) = μ*^T μ(x)
can be written as a quadratic form x^T Q x, where Q is built analytically
from the weight matrices — see analysis.py.

Architecture:
    Encoder: x(784) ──► Linear(256, no bias) ──► BilinearLayer(512) ──► μ(10), logσ²(10)
    Decoder: z(10)  ──► Linear(256) ──► ReLU ──► Linear(784) ──► Sigmoid

Weight layout (matches saved checkpoints):
    embed.weight          (d_embed, d_input)
    bilinear.weight       (2·d_hidden, d_embed)  — split into w_l / w_r at runtime
    fc_mu.weight          (d_latent, d_hidden)
    fc_logvar.weight/bias (d_latent, d_hidden)
    decoder.fc1.weight/bias
    decoder.fc2.weight/bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearLayer(nn.Module):
    """
    Computes h = (W_L x) ⊙ (W_R x).

    Stores a single weight matrix of shape (2·d_hidden, d_embed).
    The top half is W_L, the bottom half is W_R.
    Properties w_l and w_r provide the two halves for the eigendecomposition.
    """

    def __init__(self, d_embed: int, d_hidden: int):
        super().__init__()
        # Store both halves in one parameter so checkpoint keys stay clean
        self.weight = nn.Parameter(torch.empty(2 * d_hidden, d_embed))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_embed) → h: (batch, d_hidden)
        out = x @ self.weight.T          # (batch, 2·d_hidden)
        left, right = out.chunk(2, dim=1)
        return left * right

    @property
    def w_l(self) -> torch.Tensor:
        """Left projection W_L, shape (d_hidden, d_embed)."""
        return self.weight.chunk(2, dim=0)[0]

    @property
    def w_r(self) -> torch.Tensor:
        """Right projection W_R, shape (d_hidden, d_embed)."""
        return self.weight.chunk(2, dim=0)[1]


class _Decoder(nn.Module):
    """Standard MLP decoder: z → x̂."""

    def __init__(self, d_latent: int, d_hidden: int, d_input: int):
        super().__init__()
        self.fc1 = nn.Linear(d_latent, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_input)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


class BilinearVAE(nn.Module):
    """
    VAE whose encoder contains a single bilinear layer and no other nonlinearity.

    Default dimensions match the Pearce et al. (2025) MNIST classifier:
        d_input=784, d_embed=256, d_hidden=512, d_latent=10
    """

    def __init__(
        self,
        d_input:  int = 784,
        d_embed:  int = 256,
        d_hidden: int = 512,
        d_latent: int = 10,
    ):
        super().__init__()
        self.d_input  = d_input
        self.d_embed  = d_embed
        self.d_hidden = d_hidden
        self.d_latent = d_latent

        # ── Encoder ───────────────────────────────────────────────────────
        # No bias on embed and fc_mu so the encoder is a pure degree-2
        # polynomial in x, preserving the quadratic-form analysis.
        self.embed     = nn.Linear(d_input,  d_embed,  bias=False)
        self.bilinear  = BilinearLayer(d_embed, d_hidden)
        self.fc_mu     = nn.Linear(d_hidden, d_latent, bias=False)
        self.fc_logvar = nn.Linear(d_hidden, d_latent)

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder = _Decoder(d_latent, 256, d_input)

    def encode(self, x: torch.Tensor):
        """Return (μ, logσ²) for input x of shape (batch, d_input)."""
        h = self.embed(x)
        h = self.bilinear(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
