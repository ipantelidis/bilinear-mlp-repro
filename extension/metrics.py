"""
metrics.py — Quantitative evaluation metrics for the bilinear VAE extension.

All metrics take a trained model and DataLoaders as input and return
plain Python numbers (or dicts of numbers) for easy JSON serialisation.

Metrics
───────
    linear_probe            Logistic regression accuracy on frozen latent μ.
                            Measures how linearly separable classes are in latent space.

    silhouette              Silhouette score of latent μ vectors w.r.t. class labels.
                            Measures cluster compactness and separation.

    reconstruction_mse      Mean squared error between inputs and reconstructions.

    compute_fid             Fréchet Inception Distance between real and generated images.
                            Standard generative model quality metric.

    eigenfeature_consistency
                            Mean cosine similarity of top eigenvectors across N training
                            seeds. Measures whether the learned features are stable.

    interpolation_smoothness
                            Classifier confidence along a latent interpolation path.
                            Measures how semantically smooth the transitions are.

    avg_spectrum_stats      Average effective_rank and PVE across class-mean directions.

    compute_all             Convenience wrapper that runs all metrics and returns a dict.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from torchmetrics.image.fid import FrechetInceptionDistance

from analysis import (compute_class_means, analyze_direction,
                       analyze_direction_conv, spectrum_stats)


# ─────────────────────────────────────────────────────────────────────────────
# Collect latent representations
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_latents(model, loader, device: str = "cpu"):
    """
    Encode all samples and collect (μ, label) pairs.

    Returns:
        z_all : numpy array, shape (N, d_latent)
        y_all : numpy array, shape (N,)
    """
    model.eval()
    zs, ys = [], []

    for x, y in loader:
        mu, _ = model.encode(x.to(device))
        zs.append(mu.cpu().numpy())
        ys.append(y.numpy())

    return np.concatenate(zs), np.concatenate(ys)


# ─────────────────────────────────────────────────────────────────────────────
# Linear probe
# ─────────────────────────────────────────────────────────────────────────────

def linear_probe(model, train_loader, test_loader, device: str = "cpu") -> float:
    """
    Train a logistic regression on frozen latent μ vectors and return test accuracy.

    A high score means the latent space is linearly separable by class —
    a strong indicator of structured representation.

    Returns:
        accuracy in [0, 1]
    """
    z_train, y_train = collect_latents(model, train_loader, device)
    z_test,  y_test  = collect_latents(model, test_loader,  device)

    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(z_train, y_train)
    acc = clf.score(z_test, y_test)

    print(f"  Linear probe accuracy : {acc:.4f}")
    return float(acc)


# ─────────────────────────────────────────────────────────────────────────────
# Silhouette score
# ─────────────────────────────────────────────────────────────────────────────

def silhouette(model, loader, device: str = "cpu",
               max_samples: int = 5000) -> float:
    """
    Silhouette score of latent μ vectors w.r.t. class labels.

    Silhouette ∈ [-1, 1]:
        +1  clusters are compact and well-separated
         0  clusters overlap
        -1  samples assigned to the wrong cluster

    Returns:
        silhouette score as a float
    """
    z_all, y_all = collect_latents(model, loader, device)

    # Subsample if needed (silhouette is O(N²))
    if len(z_all) > max_samples:
        idx   = np.random.choice(len(z_all), max_samples, replace=False)
        z_all = z_all[idx]
        y_all = y_all[idx]

    score = silhouette_score(z_all, y_all)
    print(f"  Silhouette score      : {score:.4f}")
    return float(score)


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction MSE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruction_mse(model, loader, device: str = "cpu") -> float:
    """
    Mean squared error between clean inputs and their reconstructions,
    averaged per pixel.

    Returns:
        MSE in [0, 1]
    """
    model.eval()
    total, n = 0.0, 0

    for x, _ in loader:
        x = x.to(device)
        recon, _, _ = model(x)
        # Flatten all spatial/channel dims before taking the mean so this
        # works correctly for both flat (batch, 784) and conv (batch, 3, 32, 32) inputs.
        total += ((recon - x) ** 2).flatten(1).mean(dim=1).sum().item()
        n     += x.size(0)

    mse = total / n
    print(f"  Reconstruction MSE    : {mse:.6f}")
    return mse


# ─────────────────────────────────────────────────────────────────────────────
# FID — Fréchet Inception Distance
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_fid(model, loader, device: str = "cpu",
                n_samples: int = 5000) -> float:
    """
    Fréchet Inception Distance (FID) between real test images and VAE samples.

    FID measures the distance between the distributions of real and generated
    images in InceptionV3 feature space. Lower is better.

    For grayscale 28×28 images (MNIST / Fashion-MNIST), we:
      1. Upsample to 299×299  (InceptionV3 minimum input size)
      2. Repeat the channel 3 times to simulate RGB

    This preprocessing is applied identically to real and generated images
    so the comparison is fair.

    Args:
        n_samples : number of real/generated pairs to evaluate (more = more accurate)

    Returns:
        FID score as a float (lower = better generative quality)
    """
    model.eval()
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def to_inception_input(x: torch.Tensor) -> torch.Tensor:
        """
        Prepare images for InceptionV3 (expects RGB 299×299 in [0, 1]).

        Handles both:
          - Grayscale flat tensors (batch, 784) from MNIST → reshape, upsample, repeat
          - RGB spatial tensors  (batch, 3, H, W) from CIFAR-10 → just upsample
        """
        if x.ndim == 2:
            # Flat grayscale (MNIST / Fashion-MNIST)
            x = x.view(-1, 1, 28, 28)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            x = x.repeat(1, 3, 1, 1)
        else:
            # Already spatial (CIFAR-10): just upsample
            n_channels = x.shape[1]
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            if n_channels == 1:
                x = x.repeat(1, 3, 1, 1)   # grayscale spatial → RGB
        return x

    n_real = 0
    for x, _ in loader:
        if n_real >= n_samples:
            break
        x = x.to(device)
        fid.update(to_inception_input(x), real=True)
        n_real += x.size(0)

        # Generate the same number of samples from the prior z ~ N(0, I)
        z   = torch.randn(x.size(0), model.d_latent, device=device)
        gen = model.decoder(z)          # shape depends on model (flat or conv)
        fid.update(to_inception_input(gen), real=False)

    score = float(fid.compute())
    print(f"  FID                   : {score:.2f}")
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Eigenfeature consistency
# ─────────────────────────────────────────────────────────────────────────────

def eigenfeature_consistency(models: list, loader, device: str = "cpu") -> dict:
    """
    Measure how consistently the bilinear encoder learns the same eigenfeatures
    across multiple training runs with different random seeds.

    For each digit class, we compute the top positive eigenvector from each
    trained model and measure the mean pairwise cosine similarity. A high
    similarity (close to 1) means the learned features are stable and not
    artefacts of a particular random initialisation.

    This replicates the analysis in Figure 5 of Pearce et al. (2025) but
    for the VAE encoder setting.

    Args:
        models : list of trained BilinearVAE models (one per seed)
        loader : DataLoader used to compute class-mean encodings

    Returns:
        dict with per-class and mean cosine similarities
    """
    n_models = len(models)
    assert n_models >= 2, "Need at least 2 models to measure consistency"

    # Collect top eigenvector per class for each model
    # top_vecs[class_label] = list of eigenvectors, one per model
    top_vecs = {}

    for model in models:
        model.eval()
        class_means = compute_class_means(model, loader, device)
        for label, direction in class_means.items():
            vals, vecs = analyze_direction(model, direction)
            # Top positive eigenvector (largest positive eigenvalue)
            pos_idx = torch.where(vals > 0)[0]
            top_vec = vecs[pos_idx[0]] if len(pos_idx) > 0 else vecs[0]
            top_vecs.setdefault(label, []).append(top_vec)

    # Pairwise cosine similarity for each class
    per_class = {}
    for label, vecs in top_vecs.items():
        sims = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # |cos(u, v)| because eigenvector sign is arbitrary
                sim = float(torch.abs(
                    F.cosine_similarity(vecs[i].unsqueeze(0),
                                        vecs[j].unsqueeze(0))
                ))
                sims.append(sim)
        per_class[label] = float(np.mean(sims))

    mean_sim = float(np.mean(list(per_class.values())))

    print(f"  Eigenfeature consistency (mean cosine sim) : {mean_sim:.4f}")
    for label, sim in per_class.items():
        print(f"    digit {label} : {sim:.4f}")

    return {"mean": mean_sim, "per_class": per_class}


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation smoothness
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def interpolation_smoothness(model, class_means: dict,
                              classifier, device: str = "cpu",
                              n_steps: int = 20,
                              pairs: list = None) -> dict:
    """
    Measure how smoothly the predicted class transitions along a latent
    interpolation path using an external classifier.

    For each digit pair (A, B), we interpolate between their class-mean
    latent encodings, decode each step, and feed the decoded image to a
    pre-trained CNN classifier. We then measure:

        - transition_step : the α at which the dominant class flips from A to B
                            (lower = sharper, more abrupt transition)
        - confidence_auc  : area under the classifier confidence curve for
                            class A along the path (higher = smoother)
        - monotonicity    : fraction of steps where confidence for A
                            decreases monotonically (higher = smoother)

    Args:
        model        : trained VAE
        class_means  : dict {label → mean latent vector}, from compute_class_means()
        classifier   : pre-trained CNN that maps images (1, 28, 28) → class probs
        device       : torch device
        n_steps      : number of interpolation steps
        pairs        : list of (digit_a, digit_b) pairs to evaluate
                       defaults to [(0,1), (3,5), (4,9), (7,1)]

    Returns:
        dict with per-pair and mean smoothness metrics
    """
    if pairs is None:
        pairs = [(0, 1), (3, 5), (4, 9), (7, 1)]

    model.eval()
    classifier.eval()
    alphas  = torch.linspace(0, 1, n_steps)
    results = {}

    for a, b in pairs:
        if a not in class_means or b not in class_means:
            continue

        mu_a = class_means[a].to(device)
        mu_b = class_means[b].to(device)

        conf_a = []   # classifier confidence for class A at each step

        for alpha in alphas:
            z     = (1 - alpha) * mu_a + alpha * mu_b
            img   = model.decoder(z.unsqueeze(0)).view(1, 1, 28, 28)
            probs = classifier(img).softmax(dim=-1)
            conf_a.append(float(probs[0, a]))

        conf_a = np.array(conf_a)

        # Fraction of consecutive steps where confidence for A decreases
        diffs       = np.diff(conf_a)
        monotonicity = float((diffs <= 0).mean())

        # Step where class A confidence drops below 0.5
        below       = np.where(conf_a < 0.5)[0]
        transition  = float(alphas[below[0]]) if len(below) > 0 else 1.0

        results[f"{a}_to_{b}"] = {
            "transition_step": transition,
            "confidence_auc":  float(np.trapz(conf_a, alphas.numpy())),
            "monotonicity":    monotonicity,
        }

    mean_mono = float(np.mean([v["monotonicity"] for v in results.values()]))
    print(f"  Interpolation monotonicity (mean) : {mean_mono:.4f}")

    return {"mean_monotonicity": mean_mono, "per_pair": results}


# ─────────────────────────────────────────────────────────────────────────────
# Spectral metrics (bilinear models only)
# ─────────────────────────────────────────────────────────────────────────────

def avg_spectrum_stats(model, loader, device: str = "cpu",
                       conv: bool = False) -> dict:
    """
    Compute average eigenspectrum statistics across all class-mean directions.

    Only meaningful for bilinear models (VanillaVAE has no interaction matrix).

    Args:
        conv : set True for ConvBilinearVAE so we use the feature-space
               analysis function instead of the pixel-space one.

    Returns:
        dict with mean effective_rank, pve_1, pve_5, pve_10 across all classes
    """
    class_means  = compute_class_means(model, loader, device)
    analyze_fn   = analyze_direction_conv if conv else analyze_direction

    all_stats = []
    for label, direction in class_means.items():
        eigenvalues, _ = analyze_fn(model, direction)
        all_stats.append(spectrum_stats(eigenvalues))

    keys   = all_stats[0].keys()
    result = {k: float(np.mean([s[k] for s in all_stats])) for k in keys}

    print(f"  Avg effective rank    : {result['effective_rank']:.4f}")
    print(f"  Avg PVE@1 / @5 / @10 : "
          f"{result['pve_1']:.3f} / {result['pve_5']:.3f} / {result['pve_10']:.3f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# All metrics at once
# ─────────────────────────────────────────────────────────────────────────────

def compute_all(model, train_loader, test_loader, device: str = "cpu",
                bilinear: bool = True, conv: bool = False,
                compute_fid_score: bool = True) -> dict:
    """
    Run all metrics and return a single dict for JSON serialisation.

    Args:
        model             : trained VAE
        train_loader      : training DataLoader
        test_loader       : test DataLoader
        device            : torch device
        bilinear          : True for bilinear models (enables spectral metrics)
        conv              : True for ConvBilinearVAE (uses feature-space analysis)
        compute_fid_score : whether to compute FID (slow, requires InceptionV3)

    Returns:
        dict of all metric values
    """
    print("\nComputing metrics …")
    results = {}

    results["linear_probe"]       = linear_probe(model, train_loader, test_loader, device)
    results["silhouette"]         = silhouette(model, test_loader, device)
    results["reconstruction_mse"] = reconstruction_mse(model, test_loader, device)

    if compute_fid_score:
        results["fid"] = compute_fid(model, test_loader, device)

    if bilinear:
        results.update(avg_spectrum_stats(model, test_loader, device, conv=conv))

    return results
