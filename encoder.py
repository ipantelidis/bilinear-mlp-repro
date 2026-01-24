import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

from mlp import Bilinear, Linear



class Encoder(nn.Module):
    def __init__(self, d_input, d_hidden, d_latent):
        super().__init__()
        self.embed = Linear(d_input, d_hidden, bias=False)

        self.block1 = Bilinear(d_hidden, d_hidden, bias=False)
        self.block2 = Bilinear(d_hidden, d_hidden, bias=False)

        self.mu = Linear(d_hidden, d_latent, bias=False)
        self.logvar = Linear(d_hidden, d_latent, bias=False)

    def forward(self, x):
        h = self.embed(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.mu(h), self.logvar(h)



class Decoder(nn.Module):
    def __init__(self, d_latent, d_hidden, d_output):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_output),
        )

    def forward(self, z):
        return self.net(z)



class VAE(nn.Module):
    def __init__(self, d_input=784, d_hidden=400, d_latent=20):
        super().__init__()
        self.encoder = Encoder(d_input, d_hidden, d_latent)
        self.decoder = Decoder(d_latent, d_hidden, d_input)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar



# Likelihood + ELBO
def elbo_loss(x, logits, mu, logvar):
    # Bernoulli likelihood (as used in the paper for MNIST) TRY GAUSSIAN
    recon = F.binary_cross_entropy_with_logits(
        logits, x, reduction="sum"
    ) / x.size(0)

    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ) / x.size(0)

    return recon + kl, recon, kl






def train(model, loader, device, epochs=100):
    opt = Adam(model.parameters(), lr=1e-3)
    os.makedirs("bilinear_samples", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for x, _ in loader:
            x = x.view(x.size(0), -1).to(device)
            logits, mu, logvar = model(x)
            loss, _, _ = elbo_loss(x, logits, mu, logvar)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"epoch {epoch:03d} | loss {total / len(loader):.4f}")

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                z = torch.randn(9, model.encoder.mu.out_features, device=device)
                logits = model.decoder(z)
                x = torch.sigmoid(logits).view(-1, 1, 28, 28)
                save_image(x, f"bilinear_samples/epoch_{epoch:03d}.png", nrow=3)



if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    loader = DataLoader(train_set, batch_size=128, shuffle=True)

    model = VAE().to(device)
    train(model, loader, device, epochs=100)
    torch.save(model.state_dict(), "vae.pt")

