"""
models.py — VAE architectures for the bilinear interpretability extension.

Four models are defined here:

    VanillaVAE     Standard MLP encoder + MLP decoder.
    BilinearVAE    Bilinear encoder + MLP decoder.  ← main extension model

The β parameter (for β-VAE) is NOT stored in the model — it belongs to the
loss function in train.py.  This means:
    VanillaVAE  trained with β=1  →  standard VAE
    VanillaVAE  trained with β>1  →  β-VAE
    BilinearVAE trained with β=1  →  bilinear VAE
    BilinearVAE trained with β>1  →  β-bilinear VAE

All encoders share the same latent dimensionality and decoder, so comparisons
are fair with respect to generative capacity.

Architecture (default dimensions):
    Encoder input  : 784  (flattened 28×28)
    Embedding dim  : 256
    Hidden dim     : 512
    Latent dim     : 16

    BilinearVAE encoder:
        x(784) → Linear(784→256) → BilinearLayer(256→512) → [μ, log σ²](16)

    VanillaVAE encoder:
        x(784) → Linear(784→256) → ReLU → Linear(256→512) → ReLU → [μ, log σ²](16)

    Shared decoder:
        z(16) → Linear(16→256) → ReLU → Linear(256→784) → Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Bilinear layer (self-contained, no dependency on original/)
# ─────────────────────────────────────────────────────────────────────────────

class BilinearLayer(nn.Module):
    """
    Pure bilinear layer: g(x) = (Wx) ⊙ (Vx)

    Weights are stored as a single (2·d_out, d_in) matrix and split into
    left (W) and right (V) halves.  This matches the original paper's
    implementation in original/shared/components.py.

    Properties w_l and w_r expose W and V for the eigendecomposition in
    analysis.py.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Single weight matrix of shape (2·d_out, d_in) — split at forward time
        self.weight = nn.Parameter(torch.empty(2 * d_out, d_in))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_in)
        out = x @ self.weight.T          # (batch, 2·d_out)
        left, right = out.chunk(2, dim=-1)   # each (batch, d_out)
        return left * right              # element-wise product — no nonlinearity

    @property
    def w_l(self) -> torch.Tensor:
        """Left projection W, shape (d_out, d_in)."""
        return self.weight.chunk(2, dim=0)[0]

    @property
    def w_r(self) -> torch.Tensor:
        """Right projection V, shape (d_out, d_in)."""
        return self.weight.chunk(2, dim=0)[1]


# ─────────────────────────────────────────────────────────────────────────────
# Shared decoder (used by all models)
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Standard MLP decoder: z → x̂

    Two linear layers with a ReLU in between and a Sigmoid output so that
    reconstructions are in [0, 1], matching the pixel range.
    """

    def __init__(self, d_latent: int, d_hidden: int, d_input: int):
        super().__init__()
        self.fc1 = nn.Linear(d_latent, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_input)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))


# ─────────────────────────────────────────────────────────────────────────────
# VAE base class (shared encode/decode/forward logic)
# ─────────────────────────────────────────────────────────────────────────────

class _BaseVAE(nn.Module):
    """
    Internal base class that provides reparameterisation, decode, and forward.
    Subclasses only need to implement encode().
    """

    def __init__(self, d_input: int, d_hidden: int, d_latent: int):
        super().__init__()
        self.d_input  = d_input
        self.d_latent = d_latent
        self.decoder  = Decoder(d_latent, d_hidden, d_input)

    def encode(self, x: torch.Tensor):
        raise NotImplementedError

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ q(z|x) during training; return μ at test time."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor):
        """
        Returns:
            recon   : reconstructed input, shape (batch, d_input)
            mu      : posterior mean,      shape (batch, d_latent)
            logvar  : posterior log-variance, shape (batch, d_latent)
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar


# ─────────────────────────────────────────────────────────────────────────────
# Vanilla VAE
# ─────────────────────────────────────────────────────────────────────────────

class VanillaVAE(_BaseVAE):
    """
    Standard VAE with a two-layer MLP encoder.

    Encoder: x → Linear → ReLU → Linear → ReLU → [μ, log σ²]

    Used as the baseline for all comparisons.  Training with β=1 gives a
    standard VAE; β>1 gives a β-VAE (Higgins et al., 2017).
    """

    def __init__(self, d_input: int = 784, d_embed: int = 256,
                 d_hidden: int = 512, d_latent: int = 10):
        super().__init__(d_input, d_embed, d_latent)
        self.enc_fc1   = nn.Linear(d_input,  d_embed)
        self.enc_fc2   = nn.Linear(d_embed,  d_hidden)
        self.fc_mu     = nn.Linear(d_hidden, d_latent)
        self.fc_logvar = nn.Linear(d_hidden, d_latent)

    def encode(self, x: torch.Tensor):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)


# ─────────────────────────────────────────────────────────────────────────────
# Bilinear VAE  ← main extension model
# ─────────────────────────────────────────────────────────────────────────────

class BilinearVAE(_BaseVAE):
    """
    VAE with a bilinear MLP encoder (no element-wise nonlinearity).

    Encoder: x → Linear(embed) → BilinearLayer → [μ, log σ²]

    Because the encoder is bilinear, its computation can be expressed exactly
    as a quadratic form x^T Q x for any fixed latent direction.  This lets us
    decompose the encoder weights directly — without any forward passes —
    to find which input patterns activate each latent coordinate.

    Training with β=1 gives the bilinear VAE introduced in Kefallinou et al. (2025).
    Training with β>1 gives the β-bilinear VAE, our novel architectural contribution.

    Attributes exposed for analysis.py:
        embed.weight   : E,   shape (d_embed,  d_input)   embedding projection
        bilinear.w_l   : W,   shape (d_hidden, d_embed)   left projection
        bilinear.w_r   : V,   shape (d_hidden, d_embed)   right projection
        fc_mu.weight   : P_μ, shape (d_latent, d_hidden)  latent mean projection
    """

    def __init__(self, d_input: int = 784, d_embed: int = 256,
                 d_hidden: int = 512, d_latent: int = 10):
        super().__init__(d_input, d_embed, d_latent)
        # Encoder
        self.embed     = nn.Linear(d_input,  d_embed,  bias=False)
        self.bilinear  = BilinearLayer(d_embed, d_hidden)
        self.fc_mu     = nn.Linear(d_hidden, d_latent, bias=False)
        self.fc_logvar = nn.Linear(d_hidden, d_latent)

    def encode(self, x: torch.Tensor):
        h = self.embed(x)       # linear projection into embedding space
        g = self.bilinear(h)    # (Wh) ⊙ (Vh) — no nonlinearity
        return self.fc_mu(g), self.fc_logvar(g)


# ─────────────────────────────────────────────────────────────────────────────
# Convolutional Bilinear VAE  (for CIFAR-10)
# ─────────────────────────────────────────────────────────────────────────────

class ConvBilinearVAE(nn.Module):
    """
    Convolutional VAE with a bilinear bottleneck, designed for CIFAR-10.

    Architecture
    ────────────
    Encoder (4 strided conv layers, each halving spatial resolution):
        (3, 32, 32)
        → Conv(3→32,   k=4, s=2, p=1) + BN + LeakyReLU → (32,  16, 16)
        → Conv(32→64,  k=4, s=2, p=1) + BN + LeakyReLU → (64,  8,  8)
        → Conv(64→128, k=4, s=2, p=1) + BN + LeakyReLU → (128, 4,  4)
        → Conv(128→256,k=4, s=2, p=1) + BN + LeakyReLU → (256, 2,  2)
        → Flatten → (1024,)
        → BilinearLayer(1024→512)     ← no nonlinearity, the analysable part
        → [fc_mu, fc_logvar](512→d_latent)

    Decoder (mirrors encoder with transposed convolutions):
        z (d_latent,) → Linear(d_latent→1024) + ReLU
        → Reshape (256, 2, 2)
        → ConvTranspose(256→128) + BN + ReLU → (128, 4,  4)
        → ConvTranspose(128→64)  + BN + ReLU → (64,  8,  8)
        → ConvTranspose(64→32)   + BN + ReLU → (32,  16, 16)
        → ConvTranspose(32→3)    + Sigmoid   → (3,   32, 32)

    Design choices vs. the previous shallow version
    ─────────────────────────────────────────────────
    - 4 conv layers (was 3): gives (256, 2, 2) → 1024-dim features.
      Less aggressive than 2048-dim but twice as many filter channels,
      giving more discriminative power per feature.
    - Batch normalisation after each conv: stabilises training on natural
      images, which have much more variance than MNIST.
    - LeakyReLU (slope 0.2): avoids dead neurons which are common in deep
      conv encoders with standard ReLU.
    - d_latent=32 (was 16): CIFAR-10 has 10 visually complex classes that
      require more latent capacity than MNIST digits.
    - MSE reconstruction loss (not BCE): BCE treats pixels as independent
      Bernoulli variables, which works for near-binary MNIST but blurs
      natural images. MSE (Gaussian decoder) is standard for RGB VAEs.

    Attributes exposed for analysis.py:
        bilinear.w_l : W,   shape (512, 1024)  left bilinear weight
        bilinear.w_r : V,   shape (512, 1024)  right bilinear weight
        fc_mu.weight : P_μ, shape (d_latent, 512)
        d_feat       : 1024  (= 256 × 2 × 2)
        feat_shape   : (256, 2, 2) — reshape eigenvectors to this for visualisation
    """

    FEAT_CHANNELS = 256
    FEAT_SIZE     = 2
    D_FEAT        = FEAT_CHANNELS * FEAT_SIZE * FEAT_SIZE   # 1024
    D_HIDDEN      = 512

    def __init__(self, d_latent: int = 32):
        super().__init__()
        self.d_latent   = d_latent
        self.d_feat     = self.D_FEAT
        self.feat_shape = (self.FEAT_CHANNELS, self.FEAT_SIZE, self.FEAT_SIZE)

        # ── Encoder ────────────────────────────────────────────────────────
        def enc_block(c_in, c_out):
            """Conv + BatchNorm + LeakyReLU block."""
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv_encoder = nn.Sequential(
            enc_block(3,   32),   # (3,32,32) → (32,16,16)
            enc_block(32,  64),   # → (64,8,8)
            enc_block(64,  128),  # → (128,4,4)
            enc_block(128, 256),  # → (256,2,2)
        )

        # Bilinear bottleneck — the only weight-analysable part
        self.bilinear  = BilinearLayer(self.D_FEAT, self.D_HIDDEN)
        self.fc_mu     = nn.Linear(self.D_HIDDEN, d_latent, bias=False)
        self.fc_logvar = nn.Linear(self.D_HIDDEN, d_latent)

        # ── Decoder ────────────────────────────────────────────────────────
        def dec_block(c_in, c_out):
            """ConvTranspose + BatchNorm + ReLU block."""
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.dec_fc = nn.Linear(d_latent, self.D_FEAT)
        self.conv_decoder = nn.Sequential(
            dec_block(256, 128),  # (256,2,2) → (128,4,4)
            dec_block(128, 64),   # → (64,8,8)
            dec_block(64,  32),   # → (32,16,16)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),         # → (3,32,32), pixels in [0,1]
        )

    def encode(self, x: torch.Tensor):
        """x: (batch, 3, 32, 32) → (mu, logvar) each (batch, d_latent)"""
        h = self.conv_encoder(x)         # (batch, 256, 2, 2)
        h = h.view(h.size(0), -1)        # (batch, 1024)
        g = self.bilinear(h)             # (batch, 512) — no nonlinearity
        return self.fc_mu(g), self.fc_logvar(g)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, d_latent) → (batch, 3, 32, 32)"""
        h = F.relu(self.dec_fc(z))               # (batch, 1024)
        h = h.view(h.size(0), *self.feat_shape)  # (batch, 256, 2, 2)
        return self.conv_decoder(h)              # (batch, 3, 32, 32)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decode(z)
        return recon, mu, logvar


# ─────────────────────────────────────────────────────────────────────────────
# Simple classifier  (used for interpolation smoothness metric)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleClassifier(nn.Module):
    """
    Lightweight MLP classifier trained on raw pixels.

    Used exclusively as an external probe for the interpolation smoothness
    metric: we decode images along a latent path and measure how smoothly
    the classifier's predicted class transitions from A to B.

    Architecture: 784 → 256 → ReLU → 128 → ReLU → n_classes
    Trains to ~97% accuracy on MNIST in ~5 epochs.
    """

    def __init__(self, d_input: int = 784, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be (batch, d_input) or (batch, 1, 28, 28) — flatten either way
        return self.net(x.view(x.size(0), -1))

    def encode(self, x: torch.Tensor):
        h = self.embed(x)           # linear projection, no activation
        g = self.bilinear(h)        # bilinear: (Wh) ⊙ (Vh), no nonlinearity
        return self.fc_mu(g), self.fc_logvar(g)
