"""
run.py — Entry point for the bilinear VAE extension.

Modes
─────
    train       Train one or all models and save checkpoints.
    metrics     Load checkpoints and compute all quantitative metrics.
    figures     Load checkpoints and generate all figures.
    all         Train + metrics + figures in one go.
    consistency Train the bilinear VAE N times with different seeds and
                measure how consistently it learns the same eigenfeatures.
    beta_sweep  Train the beta-bilinear VAE for beta in {1,2,4,8} and plot
                how spectral structure changes with the KL weight.

Usage examples
──────────────
    # Full MNIST run — all four flat models:
    python extension/run.py --model all --dataset mnist --mode all

    # CIFAR-10 with the convolutional bilinear VAE:
    python extension/run.py --model conv_bilinear_vae --dataset cifar10 --mode all

    # Eigenfeature consistency (5 seeds):
    python extension/run.py --mode consistency --dataset mnist --n_seeds 5

    # Beta sweep:
    python extension/run.py --mode beta_sweep --dataset mnist --betas 1 2 4 8

    # Recompute metrics with FID on existing checkpoints:
    python extension/run.py --mode metrics --model bilinear_vae --fid
"""

import argparse
import json
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data      import get_loaders
from models    import VanillaVAE, BilinearVAE, ConvBilinearVAE, SimpleClassifier
from train     import train, load_checkpoint
from analysis  import (compute_class_means, analyze_direction,
                       analyze_direction_conv, spectrum_stats)
from metrics   import (compute_all, eigenfeature_consistency,
                       interpolation_smoothness, avg_spectrum_stats)
from visualize import (plot_reconstructions, plot_reconstructions_color,
                       plot_eigenspectrum, plot_eigenspectrum_conv,
                       plot_spectra_grid, plot_interpolation,
                       plot_latent_pca, plot_training_curves,
                       plot_beta_sweep, plot_consistency)


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

# Maps CLI name → (ModelClass, is_bilinear, is_conv)
MODEL_REGISTRY = {
    "vanilla_vae":       (VanillaVAE,       False, False),
    "bilinear_vae":      (BilinearVAE,      True,  False),
    "beta_vae":          (VanillaVAE,       False, False),
    "beta_bilinear_vae": (BilinearVAE,      True,  False),
    "conv_bilinear_vae": (ConvBilinearVAE,  True,  True),
}

# Default beta per model
DEFAULT_BETA = {
    "vanilla_vae":       1.0,
    "bilinear_vae":      1.0,
    "beta_vae":          4.0,
    "beta_bilinear_vae": 4.0,
    "conv_bilinear_vae": 1.0,
}

# Models that work on CIFAR-10 (need conv architecture)
CIFAR_MODELS = {"conv_bilinear_vae"}

# Models that work on MNIST/Fashion-MNIST (flat architecture)
FLAT_MODELS  = {"vanilla_vae", "bilinear_vae", "beta_vae", "beta_bilinear_vae"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str):
    cls, _, _ = MODEL_REGISTRY[model_name]
    return cls()


def is_bilinear(model_name: str) -> bool:
    return MODEL_REGISTRY[model_name][1]


def is_conv(model_name: str) -> bool:
    return MODEL_REGISTRY[model_name][2]


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers — all path logic lives here, nowhere else
# ─────────────────────────────────────────────────────────────────────────────

def model_dir(args, model_name: str) -> Path:
    """
    Root directory for one model + dataset pair.

    Example:  checkpoints/mnist/bilinear_vae/
    """
    return Path(args.checkpoint_dir) / args.dataset / model_name


def ckpt_path(args, model_name: str) -> Path:
    """Main checkpoint:  checkpoints/mnist/bilinear_vae/model.pt"""
    return model_dir(args, model_name) / "model.pt"


def metrics_path(args, model_name: str) -> Path:
    """Metrics file:  checkpoints/mnist/bilinear_vae/metrics.json"""
    return model_dir(args, model_name) / "metrics.json"


def log_dir(args) -> Path:
    """All log files:  checkpoints/logs/"""
    return Path(args.checkpoint_dir) / "logs"


def classifier_path(args) -> Path:
    """Auxiliary classifier:  checkpoints/classifiers/mnist.pt"""
    return Path(args.checkpoint_dir) / "classifiers" / f"{args.dataset}.pt"


def fig_dir(args, model_name: str) -> Path:
    """Figures:  figures/bilinear_vae/mnist/"""
    return Path(args.figure_dir) / model_name / args.dataset


def _dataset_defaults(args):
    """
    Return (noise_std, recon_loss) appropriate for the dataset,
    unless the user has overridden them via CLI flags.
    """
    if args.dataset == "cifar10":
        noise_std  = args.noise_std  if args.noise_std  is not None else 0.0
        recon_loss = args.recon_loss if args.recon_loss is not None else "mse"
    else:
        noise_std  = args.noise_std  if args.noise_std  is not None else 0.3
        recon_loss = args.recon_loss if args.recon_loss is not None else "bce"
    return noise_std, recon_loss


def _train_one(args, model_name: str, train_loader, test_loader,
               beta: float = None, ckpt_dir: Path = None,
               run_name: str = "model", seed: int = None):
    """
    Train one model variant. Returns (model, history).

    Args:
        ckpt_dir : directory to save checkpoint (defaults to model_dir)
        run_name : filename prefix for .pt and _history.json
        seed     : if set, calls torch.manual_seed for reproducibility
    """
    if seed is not None:
        torch.manual_seed(seed)

    beta       = beta if beta is not None else DEFAULT_BETA[model_name]
    model      = build_model(model_name)
    ckpt_dir   = ckpt_dir or model_dir(args, model_name)
    noise_std, recon_loss = _dataset_defaults(args)

    history = train(
        model, train_loader, test_loader,
        epochs         = args.epochs,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        beta           = beta,
        noise_std      = noise_std,
        recon_loss     = recon_loss,
        device         = args.device,
        checkpoint_dir = str(ckpt_dir),
        run_name       = run_name,
    )
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary classifier (for interpolation smoothness metric)
# ─────────────────────────────────────────────────────────────────────────────

def get_classifier(args, train_loader) -> SimpleClassifier:
    """Return a trained SimpleClassifier, loading from disk if available."""
    clf_path = classifier_path(args)
    clf_path.parent.mkdir(parents=True, exist_ok=True)
    clf      = SimpleClassifier().to(args.device)

    if clf_path.exists():
        try:
            state = torch.load(clf_path, map_location=args.device, weights_only=True)
            clf.load_state_dict(state)
            clf.eval()
            print(f"  Loaded classifier from {clf_path.name}")
            return clf
        except Exception:
            # File may be corrupt from an interrupted previous run — retrain
            print(f"  Classifier checkpoint corrupt, retraining ...")
            clf_path.unlink(missing_ok=True)

    print("  Training auxiliary classifier (5 epochs) ...")
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    clf.train()
    for _ in range(5):
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad(); loss.backward(); opt.step()

    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in train_loader:
            preds   = clf(x.to(args.device)).argmax(dim=1)
            correct += (preds == y.to(args.device)).sum().item()
            total   += y.size(0)
    print(f"  Classifier accuracy: {correct / total:.4f}")

    torch.save(clf.state_dict(), clf_path)
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Mode: train
# ─────────────────────────────────────────────────────────────────────────────

def mode_train(args, model_name: str, train_loader, test_loader):
    model, history = _train_one(args, model_name, train_loader, test_loader)
    plot_training_curves(history,
                         out_path=str(fig_dir(args, model_name) / "training_curves.png"))
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Mode: metrics
# ─────────────────────────────────────────────────────────────────────────────

def mode_metrics(args, model_name: str, train_loader, test_loader):
    path = ckpt_path(args, model_name)
    if not path.exists():
        print(f"  Checkpoint not found: {path} — skipping {model_name}")
        return

    model = build_model(model_name)
    load_checkpoint(model, str(path), args.device)

    results = compute_all(
        model, train_loader, test_loader,
        device            = args.device,
        bilinear          = is_bilinear(model_name),
        conv              = is_conv(model_name),
        compute_fid_score = args.fid,
    )

    # Interpolation smoothness — only for flat bilinear models
    if is_bilinear(model_name) and not is_conv(model_name):
        clf         = get_classifier(args, train_loader)
        class_means = compute_class_means(model, test_loader, args.device)
        smoothness  = interpolation_smoothness(model, class_means, clf, args.device)
        results["interpolation_smoothness"] = smoothness

    out = metrics_path(args, model_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics saved → {out.parent.name}/{out.name}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Mode: figures
# ─────────────────────────────────────────────────────────────────────────────

def mode_figures(args, model_name: str, train_loader, test_loader,
                 model=None):
    if model is None:
        path = ckpt_path(args, model_name)
        if not path.exists():
            print(f"  Checkpoint not found: {path}")
            return
        model = build_model(model_name)
        load_checkpoint(model, str(path), args.device)

    fdir    = fig_dir(args, model_name)
    conv    = is_conv(model_name)
    bilinr  = is_bilinear(model_name)

    # Reconstructions — grayscale or colour
    if conv:
        plot_reconstructions_color(model, test_loader, args.device,
                                   out_path=str(fdir / "reconstructions.png"))
    else:
        plot_reconstructions(model, test_loader, args.device,
                             out_path=str(fdir / "reconstructions.png"))

    # Latent PCA — works for all models
    plot_latent_pca(model, test_loader, args.device,
                    out_path=str(fdir / "latent_pca.png"))

    # Eigendecomposition figures — bilinear models only
    if bilinr:
        class_means = compute_class_means(model, test_loader, args.device)

        for label, direction in class_means.items():
            if conv:
                # Feature-space eigenvectors — shown as spatial activation maps
                vals, vecs = analyze_direction_conv(model, direction)
                plot_eigenspectrum_conv(
                    vals, vecs,
                    feat_shape = model.feat_shape,
                    title      = f"class {label} — {model_name}",
                    out_path   = str(fdir / "eigenspectrums" / f"class{label}.png"),
                )
            else:
                # Pixel-space eigenvectors — shown as 28×28 images
                vals, vecs = analyze_direction(model, direction)
                plot_eigenspectrum(
                    vals, vecs,
                    title    = f"digit {label} — {model_name}",
                    out_path = str(fdir / "eigenspectrums" / f"digit{label}.png"),
                )

        # Spectra grid and interpolations — flat models only
        # (conv interpolations need separate handling; not implemented yet)
        if not conv:
            plot_spectra_grid(model, test_loader, args.device,
                              out_path=str(fdir / "eigenspectrums" / "spectra_grid.png"))

            for a, b in [(7, 1), (3, 5), (4, 9)]:
                plot_interpolation(
                    model, class_means, a, b,
                    out_path=str(fdir / "interpolations" / f"digit{a}_to_{b}.png"),
                )

    print(f"  Figures saved → {fdir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Mode: eigenfeature consistency
# ─────────────────────────────────────────────────────────────────────────────

def mode_consistency(args, train_loader, test_loader):
    """
    Train BilinearVAE N times with different seeds and measure how
    consistently it learns the same eigenfeatures per class.
    """
    print(f"\nEigenfeature consistency — {args.n_seeds} seeds on {args.dataset}")

    seeds_dir = model_dir(args, "bilinear_vae") / "seeds"
    models = []
    for seed in range(args.n_seeds):
        print(f"\n  Seed {seed} / {args.n_seeds - 1}")
        model, _ = _train_one(
            args, "bilinear_vae", train_loader, test_loader,
            ckpt_dir=seeds_dir, run_name=f"seed{seed}", seed=seed,
        )
        model.eval()
        models.append(model)

    result = eigenfeature_consistency(models, test_loader, args.device)

    # Save consistency results next to the model
    out = model_dir(args, "bilinear_vae") / "consistency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {args.dataset}/bilinear_vae/consistency.json")

    plot_consistency(result,
                     out_path=str(fig_dir(args, "bilinear_vae") / "consistency.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Mode: beta sweep
# ─────────────────────────────────────────────────────────────────────────────

def mode_beta_sweep(args, train_loader, test_loader):
    """
    Train beta-bilinear VAE for each value in --betas and record spectral metrics.
    Tests whether higher beta forces lower-rank interaction matrices.
    """
    import metrics as metrics_module

    print(f"\nBeta sweep — values: {args.betas} on {args.dataset}")
    sweep_dir  = model_dir(args, "beta_bilinear_vae") / "sweep"
    sweep_results = {}

    for beta in args.betas:
        print(f"\n  beta = {beta}")
        model, _ = _train_one(
            args, "beta_bilinear_vae", train_loader, test_loader,
            beta=beta, ckpt_dir=sweep_dir, run_name=f"b{int(beta)}",
        )
        model.eval()

        stats = avg_spectrum_stats(model, test_loader, args.device)
        stats["linear_probe"] = metrics_module.linear_probe(
            model, train_loader, test_loader, args.device
        )
        if args.fid:
            stats["fid"] = metrics_module.compute_fid(model, test_loader, args.device)
        sweep_results[beta] = stats

    # Save sweep results next to the model
    out = model_dir(args, "beta_bilinear_vae") / "beta_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\n  Saved → {args.dataset}/beta_bilinear_vae/beta_sweep.json")

    plot_beta_sweep(sweep_results,
                    out_path=str(fig_dir(args, "beta_bilinear_vae") / "beta_sweep.png"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Bilinear VAE extension.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--mode", default="train",
                   choices=["train", "metrics", "figures", "all",
                            "consistency", "beta_sweep"],
                   help="Pipeline mode (default: train)")
    p.add_argument("--model", default="bilinear_vae",
                   choices=list(MODEL_REGISTRY) + ["all"],
                   help="Model to use (default: bilinear_vae)")
    p.add_argument("--dataset", default="mnist",
                   choices=["mnist", "fashion_mnist", "cifar10"],
                   help="Dataset (default: mnist)")

    # Training
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--beta",         type=float, default=None,
                   help="KL weight (overrides model default)")
    p.add_argument("--noise_std",    type=float, default=None,
                   help="Gaussian input noise std (default: 0.3 for MNIST, 0 for CIFAR-10)")
    p.add_argument("--recon_loss",   default=None, choices=["bce", "mse"],
                   help="Reconstruction loss (default: bce for MNIST, mse for CIFAR-10)")
    p.add_argument("--batch_size",   type=int,   default=128)

    # Consistency mode
    p.add_argument("--n_seeds", type=int, default=5)

    # Beta sweep mode
    p.add_argument("--betas", type=float, nargs="+", default=[1, 2, 4, 8])

    # Metrics options
    p.add_argument("--fid", action="store_true", default=False,
                   help="Compute FID score (slow)")

    # Paths
    p.add_argument("--checkpoint_dir", default="extension/checkpoints")
    p.add_argument("--figure_dir",     default="extension/figures")
    p.add_argument("--data_dir",       default="data")

    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main():
    args = parse_args()

    # Validate: cifar10 only works with conv_bilinear_vae, and vice versa
    if args.dataset == "cifar10" and args.model not in (list(CIFAR_MODELS) + ["all"]):
        print(f"  Warning: {args.model} is not designed for cifar10. "
              f"Use --model conv_bilinear_vae.")
    if args.model in CIFAR_MODELS and args.dataset != "cifar10":
        print(f"  Warning: conv_bilinear_vae expects --dataset cifar10.")

    print(f"\n{'─' * 50}")
    train_loader, test_loader, _ = get_loaders(
        args.dataset, args.batch_size, args.data_dir
    )
    print(f"Device : {args.device}")
    print(f"{'─' * 50}\n")

    # Special modes that don't iterate over models
    if args.mode == "consistency":
        mode_consistency(args, train_loader, test_loader)
        print("\nDone.")
        return

    if args.mode == "beta_sweep":
        mode_beta_sweep(args, train_loader, test_loader)
        print("\nDone.")
        return

    # Standard model loop
    if args.model == "all":
        # For cifar10, only run conv models; for others, only flat models
        if args.dataset == "cifar10":
            models_to_run = list(CIFAR_MODELS)
        else:
            models_to_run = list(FLAT_MODELS)
    else:
        models_to_run = [args.model]

    for model_name in models_to_run:
        print(f"\n{'=' * 50}\nModel : {model_name}\n{'=' * 50}")

        trained_model = None

        if args.mode in ("train", "all"):
            trained_model = mode_train(args, model_name, train_loader, test_loader)

        if args.mode in ("metrics", "all"):
            mode_metrics(args, model_name, train_loader, test_loader)

        if args.mode in ("figures", "all"):
            mode_figures(args, model_name, train_loader, test_loader,
                         model=trained_model)

    print("\nDone.")


if __name__ == "__main__":
    main()
