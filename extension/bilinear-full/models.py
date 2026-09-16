"""
models.py — FullBilinearVAE_v2: bilinear encoder + bilinear decoder.

Both the encoder and decoder use a bilinear layer, enabling weight-based
eigendecomposition on both sides.  The encoder exposes Q_in (pixel space)
and the decoder exposes Q_dec (latent space).

Linear skip connections are added to both encoder and decoder to fix the
training instabilities of the pure bilinear (v1) model:
  - enc_skip: direct x → hidden path, gives encoder linear capacity
  - decoder.skip: direct z → output path, breaks decoder even-symmetry f(z)=f(−z)

The bilinear components remain fully analytically decomposable via analysis.py.
Q_in captures the quadratic component of encoder sensitivity; Q_dec captures
the quadratic component of decoder sensitivity.

Architecture:
    Encoder: x(784) → Linear(256, no bias) [embed]
                     → BilinearLayer(512)  [bilinear]   ← quadratic path
                     + Linear(512)         [enc_skip]   ← linear path
                     → fc_mu/fc_logvar(10)

    Decoder: z(10) → Linear(256, no bias) [embed_dec]
                   → BilinearLayer(512)   [bilinear_dec] ← quadratic path
                   → Linear(784, no bias) [fc_out]
                   + Linear(784)          [skip]          ← linear path
                   → Sigmoid

Weight layout (matches saved checkpoints):
    embed.weight                (256, 784)
    bilinear.weight             (1024, 256)  — split into w_l/w_r
    enc_skip.weight/bias        (512, 784)
    fc_mu.weight                (10, 512)
    fc_logvar.weight/bias       (10, 512)
    decoder.embed_dec.weight    (256, 10)
    decoder.bilinear_dec.weight (1024, 256)  — split into w_l/w_r
    decoder.fc_out.weight       (784, 512)
    decoder.skip.weight/bias    (784, 10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearLayer(nn.Module):
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


class _BilinearDecoderWithSkip(nn.Module):
    def __init__(self, d_latent: int, d_embed: int, d_hidden: int, d_input: int):
        super().__init__()
        self.embed_dec    = nn.Linear(d_latent, d_embed,  bias=False)
        self.bilinear_dec = BilinearLayer(d_embed, d_hidden)
        self.fc_out       = nn.Linear(d_hidden, d_input,  bias=False)
        self.skip         = nn.Linear(d_latent, d_input)   # breaks even-symmetry

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.embed_dec(z)
        h = self.bilinear_dec(h)
        return torch.sigmoid(self.fc_out(h) + self.skip(z))


class FullBilinearVAE(nn.Module):
    """
    VAE with bilinear encoder and bilinear decoder (+ linear skip connections).

    Default dimensions:
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
        self.d_latent = d_latent

        # Bilinear encoder
        self.embed    = nn.Linear(d_input,  d_embed,  bias=False)
        self.bilinear = BilinearLayer(d_embed, d_hidden)
        self.enc_skip = nn.Linear(d_input, d_hidden)   # linear skip
        self.fc_mu    = nn.Linear(d_hidden, d_latent,  bias=False)
        self.fc_logvar = nn.Linear(d_hidden, d_latent)

        # Bilinear decoder (with linear skip)
        self.decoder = _BilinearDecoderWithSkip(d_latent, d_embed, d_hidden, d_input)

    def encode(self, x: torch.Tensor):
        h = self.bilinear(self.embed(x)) + self.enc_skip(x)
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
