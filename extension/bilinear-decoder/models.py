"""
models.py — DecBilinearVAE architecture.

The decoder uses a bilinear layer (element-wise product of two linear maps)
which makes its entire computation a degree-2 polynomial in the latent code z.
This means for any output direction p*, the activation p* · ŷ(z) can be
written as a quadratic form z^T Q z, where Q is built analytically from the
decoder weights — see analysis.py.

Architecture:
    Encoder: x(784) ──► Linear(256) ──► ReLU ──► Linear(512) ──► ReLU ──► μ(10), logσ²(10)
    Decoder: z(10)  ──► Linear(256, no bias) ──► BilinearLayer(512) ──► Linear(784, no bias) ──► Sigmoid

Weight layout (matches saved checkpoints):
    enc_fc1.weight/bias            (256, 784)
    enc_fc2.weight/bias            (512, 256)
    fc_mu.weight/bias              (10, 512)
    fc_logvar.weight/bias          (10, 512)
    decoder.embed_dec.weight       (256, 10)
    decoder.bilinear_dec.weight    (1024, 256) — split into w_l / w_r at runtime
    decoder.fc_out.weight          (784, 512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearLayer(nn.Module):
    """
    Computes h = (W_L x) ⊙ (W_R x).

    Stores a single weight matrix of shape (2·d_hidden, d_embed).
    Properties w_l and w_r expose the two halves for the eigendecomposition.
    """

    def __init__(self, d_embed: int, d_hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2 * d_hidden, d_embed))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight.T
        left, right = out.chunk(2, dim=1)
        return left * right

    @property
    def w_l(self) -> torch.Tensor:
        return self.weight.chunk(2, dim=0)[0]

    @property
    def w_r(self) -> torch.Tensor:
        return self.weight.chunk(2, dim=0)[1]


class _BilinearDecoder(nn.Module):
    def __init__(self, d_latent: int, d_embed: int, d_hidden: int, d_input: int):
        super().__init__()
        self.embed_dec    = nn.Linear(d_latent, d_embed,  bias=False)
        self.bilinear_dec = BilinearLayer(d_embed, d_hidden)
        self.fc_out       = nn.Linear(d_hidden, d_input,  bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.embed_dec(z)
        h = self.bilinear_dec(h)
        return torch.sigmoid(self.fc_out(h))


class DecBilinearVAE(nn.Module):
    """
    VAE whose decoder contains a single bilinear layer.

    Encoder weights are stored flat on the model (enc_fc1, enc_fc2, fc_mu,
    fc_logvar) to match the saved checkpoints.

    Default dimensions:
        d_input=784, d_embed=256, d_hidden=512, d_latent=10
    """

    def __init__(
        self,
        d_input:   int = 784,
        d_enc1:    int = 256,   # enc_fc1 output
        d_enc2:    int = 512,   # enc_fc2 output
        d_embed:   int = 256,   # decoder embed_dec output
        d_hidden:  int = 512,   # decoder bilinear_dec output
        d_latent:  int = 10,
    ):
        super().__init__()
        self.d_input  = d_input
        self.d_latent = d_latent

        # Encoder (flat on model to match checkpoint keys)
        self.enc_fc1   = nn.Linear(d_input, d_enc1)
        self.enc_fc2   = nn.Linear(d_enc1,  d_enc2)
        self.fc_mu     = nn.Linear(d_enc2,  d_latent)
        self.fc_logvar = nn.Linear(d_enc2,  d_latent)

        # Decoder
        self.decoder = _BilinearDecoder(d_latent, d_embed, d_hidden, d_input)

    def encode(self, x: torch.Tensor):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
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
