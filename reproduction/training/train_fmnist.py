import os

import torch
from image.datasets import FMNIST
from image.model import Model

# Ensure output directory exists
out_dir = "reproduction/outputs/models"
os.makedirs(out_dir, exist_ok=True)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
train = FMNIST(train=True, device=device)
test = FMNIST(train=False, device=device)

# Initialize model
model = Model.from_config(
    d_hidden=512,
    n_layer=1,
    bias=False,
    residual=False,
    lr=1e-3,
    wd=1.0,
    epochs=100,
    batch_size=2048,
    seed=42,
)

# Move model to device
model.to(device)

# Train model
history = model.fit(train, test)
print(history.tail())

# Save model
torch.save(model.state_dict(), "reproduction/outputs/models/fmnist_base.pt")
