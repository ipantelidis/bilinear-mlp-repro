import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

from mlp import Bilinear, Linear, RMSNorm



class Encoder(nn.Module):
    def __init__(self, d_input, d_hidden, d_latent, bias=False):
        super().__init__()
        self.embed = Linear(d_input, d_hidden, bias=False)
        self.block1 = Bilinear(d_hidden, d_hidden, bias=False)
        self.mu = Linear(d_hidden, d_latent, bias=False)
        
        #self.block1 = Bilinear(d_hidden, d_hidden, bias=False)
        self.logvar = Linear(d_hidden, d_latent, bias=False)
        #self.norm = RMSNorm()       # Normalize after the bilinear to prevent instability

    def forward(self, x):
        h = self.embed(x)
        h = self.block1(h)
        #h = self.norm(h)  
        mu = self.mu(h)

        logvar = self.logvar(h)
        logvar = logvar.clamp(-6, 2)    # constraint variance of posterior (stabulity) 
        return mu, logvar
    

    @torch.no_grad()
    def decompose(self):
        l = self.block1.w_l
        r = self.block1.w_r
        p = self.mu.weight

        #b = torch.einsum("lo,oi,oj->lij", p, l, r)
        b = torch.einsum("ka,ai,aj->kij", p, l, r)
        b = 0.5 * (b + b.mT)

        vals, vecs = torch.linalg.eigh(b)

        

        return vals, vecs




class Decoder(nn.Module):
    def __init__(self, d_latent, d_hidden, d_output):
        super().__init__()
        self.net = nn.Sequential(
            # more layers in decoder to match it can much the quadratic encoder
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
    recon = 0.5 * F.mse_loss(logits, x, reduction="sum")
    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    return recon, kl






def train(model, loader, device, epochs=200):
    opt = Adam(model.parameters(), lr=1e-3)
    os.makedirs("encoder_samples", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0

        total_elbo = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for x, _ in loader:
            x = x.view(x.size(0), -1).to(device)

            logits, mu, logvar = model(x)
            recon, kl = elbo_loss(x, logits, mu, logvar)
            loss = (recon + kl) / x.size(0)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_elbo += loss.item()
            total_recon += recon.item() / x.size(0)
            total_kl += kl.item() / x.size(0)


        print(
            f"epoch {epoch:03d} | "
            f"elbo {total_elbo/len(loader):.3f} | "
            f"recon {total_recon/len(loader):.3f} | "
            f"kl {total_kl/len(loader):.3f}"
        )


        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                z = torch.randn(9, model.encoder.mu.out_features, device=device)
                logits = model.decoder(z)
                x = logits.view(-1, 1, 28, 28)
                save_image(x, f"encoder_samples/epoch_{epoch:03d}.png", nrow=3)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x > 0.5).float()
    ])

    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    loader = DataLoader(train_set, batch_size=128, shuffle=True)

    model = VAE().to(device)
    train(model, loader, device, epochs=200)
    torch.save(model.state_dict(), "encoder_vae.pt")