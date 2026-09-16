"""
Exp 09 — Encode→Decode Loop

Iterates x_{t+1} = decode(encode(x_t)) starting from different seeds:
  (i)  Real test images (one per class)
  (ii) Random noise
  (iii) Decoder-synthesised image (Q_dec eigvec)

Tracks per iteration:
  - Reconstruction MSE to original x_0
  - Pixel cosine similarity to iteration 1 (convergence measure)
  - Cosine similarity between different starting points at the same t

Hypothesis: if the model has a dominant generative direction, all starting
points should converge toward the same stable fixed point regardless of class.

Also measures the fixed-point image spectrum: does it resemble the near-universal
decoder eigvec (exp02)?

Figure saved:
    figures/mnist/exp09_loop.png
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models   import FullBilinearVAE
from train    import load_checkpoint
from analysis import (get_decoder_interaction_matrix, decompose, mean_lat_norm)
from visualize import save_fig

CKPT   = "checkpoints/mnist/model.pt"
DATA   = "/home/v25/ippa6201/bilinear-mlp-repro/data"
N_ITER = 10


def _cos(a, b):
    return float(torch.dot(a.float(), b.float()) / (a.float().norm() * b.float().norm() + 1e-8))


def _run_loop(model, x0, n_iter):
    """Run the encode→decode loop for n_iter steps. Returns trajectory."""
    traj = [x0.clone()]
    x = x0.clone()
    with torch.no_grad():
        for _ in range(n_iter):
            mu, _ = model.encode(x.unsqueeze(0))
            x = model.decode(mu).squeeze(0).detach()
            traj.append(x.clone())
    return traj  # list of length n_iter+1


def main():
    model = FullBilinearVAE(); load_checkpoint(model, CKPT); model.eval()

    loader = DataLoader(
        datasets.MNIST(DATA, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x.view(-1))])),
        batch_size=512, shuffle=False)

    # Collect one image per class (first occurrence)
    real_imgs = {}
    for x, y in loader:
        for i, lbl in enumerate(y.tolist()):
            if lbl not in real_imgs:
                real_imgs[lbl] = x[i]
        if len(real_imgs) == 10:
            break
    classes = sorted(real_imgs.keys())

    # Decoder synthesis images
    buckets = {}
    for x, y in loader:
        for i, lbl in enumerate(y.tolist()):
            buckets.setdefault(lbl, []).append(x[i])
    mean_imgs = {c: torch.stack(v).mean(0) for c, v in buckets.items()}
    scale = mean_lat_norm(model, loader)

    dec_synth_imgs = {}
    with torch.no_grad():
        for c in classes:
            Q = get_decoder_interaction_matrix(model, mean_imgs[c])
            vals, vecs = decompose(Q)
            pos_idx = (vals > 0).nonzero(as_tuple=True)[0]
            z = vecs[pos_idx[0]] * scale if len(pos_idx) else torch.zeros(model.d_latent)
            dec_synth_imgs[c] = model.decode(z.unsqueeze(0)).squeeze(0).detach()

    # Random noise seeds (fixed)
    torch.manual_seed(0)
    rand_imgs = {c: torch.rand(model.d_input) for c in classes}

    # Run loops for each class and each starting condition
    trajs_real  = {c: _run_loop(model, real_imgs[c],      N_ITER) for c in classes}
    trajs_rand  = {c: _run_loop(model, rand_imgs[c],      N_ITER) for c in classes}
    trajs_synth = {c: _run_loop(model, dec_synth_imgs[c], N_ITER) for c in classes}

    # Convergence metric: cos(x_t, x_{t-1}) per step (class-averaged)
    def mean_step_cos(trajs):
        per_step = []
        for t in range(1, N_ITER + 1):
            cos_vals = [_cos(trajs[c][t], trajs[c][t - 1]) for c in classes]
            per_step.append(np.mean(cos_vals))
        return per_step

    conv_real  = mean_step_cos(trajs_real)
    conv_rand  = mean_step_cos(trajs_rand)
    conv_synth = mean_step_cos(trajs_synth)

    # MSE to original x_0
    def mean_mse_to_x0(trajs):
        per_step = []
        for t in range(1, N_ITER + 1):
            mse_vals = [((trajs[c][t] - trajs[c][0]) ** 2).mean().item() for c in classes]
            per_step.append(np.mean(mse_vals))
        return per_step

    mse_real  = mean_mse_to_x0(trajs_real)
    mse_rand  = mean_mse_to_x0(trajs_rand)
    mse_synth = mean_mse_to_x0(trajs_synth)

    # Cross-class convergence: do all 10 classes converge to the SAME fixed point?
    # Measure mean pairwise cosine between x_t across classes at each step
    def cross_class_cos(trajs):
        per_step = []
        for t in range(N_ITER + 1):
            imgs_t = [trajs[c][t] for c in classes]
            pairs = [(i, j) for i in range(len(classes)) for j in range(i + 1, len(classes))]
            cos_vals = [_cos(imgs_t[a], imgs_t[b]) for a, b in pairs]
            per_step.append(np.mean(cos_vals))
        return per_step

    cross_real  = cross_class_cos(trajs_real)
    cross_rand  = cross_class_cos(trajs_rand)
    cross_synth = cross_class_cos(trajs_synth)

    # Print summary
    print(f"\nEncode→decode loop ({N_ITER} iterations):")
    print(f"\nMean step cosine (convergence speed):")
    for t in range(N_ITER):
        print(f"  step {t+1:>2}: real={conv_real[t]:.4f}  rand={conv_rand[t]:.4f}  synth={conv_synth[t]:.4f}")
    print(f"\nMean cross-class cosine (all classes converge to same point?):")
    for t in [0, 1, 3, 5, 9, N_ITER]:
        if t <= N_ITER:
            print(f"  t={t:>2}: real={cross_real[t]:.4f}  rand={cross_rand[t]:.4f}  synth={cross_synth[t]:.4f}")

    with open("figures/mnist/exp09_results.json", "w") as f:
        json.dump({"n_iter": N_ITER,
                   "step_cos":        {"real": conv_real,  "rand": conv_rand,  "synth": conv_synth},
                   "mse_to_x0":       {"real": mse_real,   "rand": mse_rand,   "synth": mse_synth},
                   "cross_class_cos": {"real": cross_real, "rand": cross_rand, "synth": cross_synth}},
                  f, indent=2)

    # Figure: 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    steps = range(1, N_ITER + 1)

    # Panel 1: step cosine (convergence)
    axes[0, 0].plot(steps, conv_real,  "o-", label="real images",  color="steelblue")
    axes[0, 0].plot(steps, conv_rand,  "s-", label="random noise", color="indianred")
    axes[0, 0].plot(steps, conv_synth, "^-", label="dec synth",    color="seagreen")
    axes[0, 0].set_xlabel("Iteration t"); axes[0, 0].set_ylabel("mean cos(x_t, x_{t-1})")
    axes[0, 0].set_title("Loop convergence speed\n(cos to previous step)", fontsize=10)
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(1.0, color="black", linewidth=0.5, linestyle="--")

    # Panel 2: MSE to x_0
    axes[0, 1].plot(steps, mse_real,  "o-", label="real images",  color="steelblue")
    axes[0, 1].plot(steps, mse_rand,  "s-", label="random noise", color="indianred")
    axes[0, 1].plot(steps, mse_synth, "^-", label="dec synth",    color="seagreen")
    axes[0, 1].set_xlabel("Iteration t"); axes[0, 1].set_ylabel("mean MSE to x_0")
    axes[0, 1].set_title("Drift from initial image\n(MSE to x_0)", fontsize=10)
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: cross-class cosine
    steps0 = range(N_ITER + 1)
    axes[1, 0].plot(steps0, cross_real,  "o-", label="real images",  color="steelblue")
    axes[1, 0].plot(steps0, cross_rand,  "s-", label="random noise", color="indianred")
    axes[1, 0].plot(steps0, cross_synth, "^-", label="dec synth",    color="seagreen")
    axes[1, 0].set_xlabel("Iteration t"); axes[1, 0].set_ylabel("mean pairwise cos (across classes)")
    axes[1, 0].set_title("Cross-class convergence\n(all classes → same fixed point?)", fontsize=10)
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: fixed-point images (real starting points, after 10 iterations)
    n_show = min(5, len(classes))
    inner_axes = [fig.add_axes([0.55 + 0.086 * i, 0.08, 0.075, 0.35]) for i in range(n_show)]
    for i, c in enumerate(classes[:n_show]):
        fp = trajs_real[c][-1].view(28, 28).numpy()
        inner_axes[i].imshow(fp, cmap="gray_r", vmin=0, vmax=1)
        inner_axes[i].axis("off")
        inner_axes[i].set_title(f"d{c}\nt=10", fontsize=7)
    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, 0.95, "Fixed-point images (real→10 iters)\n(d0-d4 shown)",
                    ha="center", va="top", transform=axes[1, 1].transAxes, fontsize=9)

    fig.suptitle(f"Exp 09 — Encode→decode loop ({N_ITER} iterations)\n"
                 f"FullBilinearVAE", fontsize=11, y=1.01)
    fig.tight_layout()
    save_fig(fig, "figures/mnist/exp09_loop.png")


if __name__ == "__main__":
    main()
