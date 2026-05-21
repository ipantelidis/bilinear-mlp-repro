# Bilinear Encoder — Weight-Based Analysis

Extension of Pearce et al. (ICLR 2025) *"Bilinear MLPs Enable Weight-Based
Mechanistic Interpretability"* to the VAE encoder setting.

Pearce et al. showed that a bilinear MLP's computation can be expressed as a
quadratic form in the input, enabling eigendecomposition of the weight matrices
to reveal interpretable input-space features — without any forward passes or
data.  This work applies the same analysis to the **encoder** of a Variational
Autoencoder, a generative model not considered in the original paper.

---

## Core idea

Because the BilinearVAE encoder contains no element-wise nonlinearity, its
output along any latent direction μ\* is a quadratic form in the input:

```
f_{μ*}(x) = x^T Q x
```

where Q is built analytically from the encoder weight matrices.
Eigendecomposing Q gives pixel-space patterns (eigenvectors) that activate
(λ > 0) or suppress (λ < 0) that direction — **derived from weights alone**.

---

## Directory structure

```
bilinear-encoder/
├── models.py           BilinearVAE architecture
├── train.py            Training utilities (loss, epoch, checkpoint I/O)
├── analysis.py         Core: interaction_matrix, decompose, class_means
├── visualize.py        Shared plotting helpers
├── run_all.py          Run all 10 experiments (or a subset)
│
├── checkpoints/
│   ├── mnist/
│   │   ├── model.pt            Trained BilinearVAE on MNIST (30 epochs)
│   │   └── seeds/
│   │       └── seed{0-4}.pt    5 independent seeds (for consistency exp)
│   └── fashion_mnist/
│       └── model.pt            Trained BilinearVAE on Fashion-MNIST (30 epochs)
│
├── figures/
│   ├── mnist/                  Outputs from experiments 01–08 and 10
│   └── fashion_mnist/          Outputs from experiment 09
│
└── experiments/
    ├── exp01_latent_dictionary.py
    ├── exp02_truncation.py
    ├── exp03_cross_class.py
    ├── exp04_semantic_diff.py
    ├── exp05_max_activating.py
    ├── exp06_negative_eigenvectors.py
    ├── exp07_saliency.py
    ├── exp08_eigvec_consistency.py
    ├── exp09_fmnist.py
    └── exp10_adversarial_mask.py
```

Data is read from the shared repository data directory
(`../../../data/`); no local copy is needed.

---

## Running

```bash
cd bilinear-encoder/

# Run all 10 experiments
python run_all.py

# Run a specific subset
python run_all.py 1 5 8

# Run a single experiment directly
python experiments/exp05_max_activating.py
```

All figures are written to `figures/mnist/` or `figures/fashion_mnist/`.

---

## Experiments

| # | Script | What it measures |
|---|--------|-----------------|
| 01 | `exp01_latent_dictionary.py` | Top activating/suppressing eigenvector per latent dimension (μ\* = e_k). Fully data-free. |
| 02 | `exp02_truncation.py` | Pearson r between true encoder activation and rank-k approximation vs. k. Analog of Pearce et al. Fig 5B. |
| 03 | `exp03_cross_class.py` | Pairwise cosine similarity between top eigenvectors across all 10 digit classes. |
| 04 | `exp04_semantic_diff.py` | Eigenvectors for contrastive directions μ\* = mean_A − mean_B for confusable pairs. |
| 05 | `exp05_max_activating.py` | Causal test: encoding the top eigenvector lands in the correct class (9/10). Trained vs. random baseline. |
| 06 | `exp06_negative_eigenvectors.py` | Cross-suppression map: does suppressing class c look like activating class d? |
| 07 | `exp07_saliency.py` | Per-pixel sensitivity s(x) = \|2Qx\| without backpropagation, plus rank-k progression. |
| 08 | `exp08_eigvec_consistency.py` | Cosine similarity of top eigenvectors across 5 training seeds. Analog of Pearce et al. Fig 5A. |
| 09 | `exp09_fmnist.py` | Experiments 01, 05, 03 repeated on Fashion-MNIST. Direct comparison with Pearce et al. Fig 2B. |
| 10 | `exp10_adversarial_mask.py` | Pseudoinverse masks steer encoder toward target class (67% at σ=1 vs. 4% random). Analog of Pearce et al. Section 4.4. |

---

## Key results

| Experiment | Result | Pearce et al. comparison |
|---|---|---|
| Consistency (exp 08) | **0.812 ± 0.060** mean \|cos sim\| across seeds | Classifiers: 0.80–0.90 ✓ replicates |
| Truncation (exp 02) | Rank 10–15 for r > 0.90 | Classifiers: rank 3 — VAE needs more (distributional encoding) |
| Max-activating MNIST (exp 05) | **9/10** trained vs **3/10** random | Novel experiment, no prior analog |
| Max-activating FMNIST (exp 09) | **10/10** trained vs **6/10** random | Novel experiment |
| Adversarial steering (exp 10) | **67%** at σ=1 vs **4%** random (19×) | Analog of Section 4.4 |

---

## Architecture

```
Encoder: x(784) → Linear(256, no bias) → BilinearLayer(512) → μ(10), logσ²(10)
Decoder: z(10)  → Linear(256) → ReLU  → Linear(784) → Sigmoid
```

The `BilinearLayer` stores a single weight matrix of shape `(1024, 256)`.
The top and bottom halves are `W_L` and `W_R`; their product gives the
degree-2 polynomial encoder output that makes the analysis possible.

---

## Reference

Pearce, M., Dooms, T., Rigg, A., Oramas, J., & Sharkey, L. (2025).
*Bilinear MLPs enable weight-based mechanistic interpretability.*
ICLR 2025. [https://github.com/tdooms/bilinear-decomposition](https://github.com/tdooms/bilinear-decomposition)
