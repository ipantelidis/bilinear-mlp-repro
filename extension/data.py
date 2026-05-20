"""
data.py — Dataset loading for the bilinear VAE extension.

Provides train and test DataLoaders for three datasets:

    mnist          28×28 grayscale, flattened to (784,)
    fashion_mnist  28×28 grayscale, flattened to (784,)
    cifar10        32×32 RGB, kept as (3, 32, 32) for the conv encoder

All pixel values are in [0, 1] so that binary cross-entropy loss is valid
and eigenfeature visualisations remain interpretable.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ─────────────────────────────────────────────────────────────────────────────
# Dataset configs
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (torchvision class, flatten to 1-D, image shape)
_CONFIGS = {
    "mnist": {
        "cls":     datasets.MNIST,
        "flatten": True,
        "shape":   (784,),
        "channels": 1,
    },
    "fashion_mnist": {
        "cls":     datasets.FashionMNIST,
        "flatten": True,
        "shape":   (784,),
        "channels": 1,
    },
    "cifar10": {
        "cls":     datasets.CIFAR10,
        "flatten": False,           # conv encoder expects (C, H, W)
        "shape":   (3, 32, 32),
        "channels": 3,
    },
}


def get_loaders(dataset: str = "mnist", batch_size: int = 128,
                data_dir: str = "data"):
    """
    Return (train_loader, test_loader) for the requested dataset.

    Args:
        dataset    : "mnist", "fashion_mnist", or "cifar10"
        batch_size : samples per batch
        data_dir   : where datasets are downloaded and cached

    Returns:
        train_loader, test_loader
        config dict with keys: shape, channels, flatten
    """
    if dataset not in _CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {list(_CONFIGS.keys())}"
        )

    cfg = _CONFIGS[dataset]

    # Build transform pipeline
    t = [transforms.ToTensor()]   # converts to [0, 1] float tensor
    if cfg["flatten"]:
        t.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(t)

    train_set = cfg["cls"](data_dir, train=True,  download=True, transform=transform)
    test_set  = cfg["cls"](data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"Dataset  : {dataset}")
    print(f"Shape    : {cfg['shape']}")
    print(f"Train    : {len(train_set):,} samples")
    print(f"Test     : {len(test_set):,} samples")

    return train_loader, test_loader, cfg
