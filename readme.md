## Reproduction: Bilinear MLPs enable weight-based mechanistic interpretability

This repository reproduces the main interpretability claims from the paper "Bilinear MLPs enable weight-based mechanistic interpretability" (Pearce, Dooms, Rigg, Oramas & Sharkey, ICLR 2025). All reproduction experiments follow the training setup, datasets, and analysis procedures described by the authors, covering both the vision experiments (Section 4) and the language-model experiments (Section 5).

The paper's contributions are organized below as seven claims, C1–C7. Each claim is a single testable statement about a *different* property of bilinear MLPs, so the claims can be verified (or falsified) independently — this structure is also the skeleton for the accompanying report. A full coverage table of every reproduced figure and appendix follows.

### Repository structure

```
original/       the authors' released library (vendored unmodified)
reproduction/   one self-contained directory per reproduced figure/appendix
  image/          vision experiments  (C1–C4: figs 2–7, appendices A–F, M)
  language/       language experiments (C5–C7: figs 8–9, appendices G–J, N–O)
extension/      bilinear MLPs in VAEs (see the extension section below)
data/           datasets, downloaded automatically on first use
```

---

### C1 — Interpretability: top eigenvectors are human-recognizable features

**Claim.** Eigendecomposing a trained bilinear MLP's interaction matrices — using the weights alone, with no input data — yields top eigenvectors that correspond to human-recognizable visual structure (digit strokes on MNIST, garment edges on Fashion-MNIST).

- **Paper evidence:** Figures 2–3, Appendix A
- **Our code:** `reproduction/image/fig_02/`, `fig_03/`, `appendix_a/`
- **Status:** reproduced — leading eigenfeatures qualitatively match the paper's for both datasets.

---

### C2 — Low rank: a handful of eigenvectors carries the computation

**Claim.** Each class's interaction matrix is effectively low-rank: truncating the model to its few largest-magnitude eigenvectors preserves classification accuracy, consistently across model sizes, and the retained eigenvectors are consistent across training runs.

- **Paper evidence:** Figure 5, Appendices D, F
- **Our code:** `reproduction/image/fig_05/`, `appendix_d/`, `appendix_f/`
- **Status:** reproduced — error curves and cross-seed similarity match; quantitative results serialized to JSON.

---

### C3 — Regularization: input noise cleans features and exposes overfitting

**Claim.** The character of the learned eigenfeatures is controlled by regularization: Gaussian input noise sparsifies and cleans them (revealing overfitting artifacts in unregularized models), while weight decay lowers the effective rank; augmentations reshape features in geometrically predictable ways.

- **Paper evidence:** Figure 4, Appendices B, E
- **Our code:** `reproduction/image/fig_04/`, `appendix_b/`, `appendix_e/`
- **Status:** reproduced — same qualitative feature evolution and sparsity trends.

---

### C4 — Causality: weight-derived masks steer the model

**Claim.** Eigenvectors are causally meaningful, not just descriptive: adversarial masks constructed purely from the weights (via pseudoinverse encoders, with no gradients or per-input optimization) reduce accuracy and induce targeted misclassification far more than matched random masks.

- **Paper evidence:** Figure 7, Appendix M
- **Our code:** `reproduction/image/fig_07/`, `appendix_m/`
- **Status:** reproduced — adversarial masks dominate random baselines at every mask strength (test-set metrics in JSON).

---

### C5 — Circuits: feature interactions in language models are traceable through the weights

**Claim.** In a bilinear transformer, the interactions that construct an SAE output feature from SAE input features can be read off the bilinear tensor directly, recovering an interpretable circuit (the sentiment-negation circuit) whose structure is grounded in the weights rather than in gradient approximations.

- **Paper evidence:** Figure 8, Appendices N, O
- **Our code:** `reproduction/language/fig_08/`, `appendix_n/`, `appendix_o/`
- **Status:** reproduced — AND-gate interaction pattern and eigenvalue outliers match.

---

### C6 — Approximation: language-model features are low-rank too

**Claim.** SAE output features of bilinear transformers are well-approximated by just the top one or two eigenvectors of their interaction matrices — increasingly so the longer the SAE is trained — extending the low-rank finding (C2) from classifiers to language models.

- **Paper evidence:** Figure 9, Appendix H (Fig. 24 / Table 5), Appendix G (SAE quality)
- **Our code:** `reproduction/language/fig_09/`, `appendix_h/`, `appendix_g/`
- **Status:** reproduced, verified per panel — 9A: all three curves rise over the paper's range (deviation: fw-small and fw-medium swap order relative to the paper); 9B: 74% of fw-medium features exceed 0.75 correlation with two eigenvectors (paper: "most"); 9C: 8/9 randomly sampled features lie tightly on the diagonal (r = 0.75–0.96), one dense feature approximates poorly, mirroring the paper's own weakest panel. Fig. 24's training-time trend reproduces (Table 5 comparison in `appendix_h_results.json`). Note: the fw-medium eigendecomposition cache initially in the repo was stale and produced near-zero correlations; it was regenerated from scratch.

---

### C7 — Practicality: bilinear layers are a drop-in replacement

**Claim.** Adopting bilinear MLPs costs little: they match gated activations (ReGLU/SwiGLU) in compute-matched language-model training, and an existing SwiGLU transformer can be converted into a bilinear one by fine-tuning the gate away (annealing β→0) at a modest loss penalty.

- **Paper evidence:** Appendix I (Table 6), Appendix J (Fig. 25)
- **Our code:** `reproduction/language/appendix_i/`, `appendix_j/`
- **Status:** partially reproduced — the core of the claim holds: after 5 epochs the bilinear model trails SwiGLU by only 0.006 nats and matches ReGLU, and TinyLlama's gate anneals away with a modest, shrinking penalty (0.14 nats after 400M tokens vs the paper's 0.05). Absolute losses are not comparable (dataset, see deviations), the constant-time equalization does not manifest when all three variants run at identical wall-clock speed, and fine-tuning needs LR 3e-5 instead of the reported 6e-4.

---

## Full reproduction coverage

Every experiment lives in its own directory under `reproduction/` and is
self-contained: run the script and it produces the figure(s) plus, where
applicable, a JSON file with the quantitative results. Expensive runs cache
trained models / decompositions to disk and skip retraining when the cache
exists (delete the cache file to force a rerun).

### Image experiments (`reproduction/image/`)

| Paper item | Directory | Notes |
|---|---|---|
| Figure 2 (eigenfeatures MNIST + FMNIST) | `fig_02/` | |
| Figure 3 (eigenspectrum, digit 5) | `fig_03/` | |
| Figure 4 (input-noise sweep) | `fig_04/` | noise grid follows the original authors' figure code (std = i·0.2) |
| Figure 5 (consistency & truncation across sizes) | `fig_05/` | results in `fig_05_results.json` |
| Figure 6 (similarity-classifier challenge, biases) | `fig_06/` | |
| Figure 7 (adversarial masks) | `fig_07/` | test-set metrics in `fig_07_results_{1,2}.json` |
| Appendix A (per-digit eigenspectra) | `appendix_a/` | |
| Appendix B (augmentation ablations) | `appendix_b/` | select axis via `APPENDIX_B_AXIS=blur\|rotation\|translation` |
| Appendix C (explainability case study) | `appendix_c/` | |
| Appendix D (HOSVD) | `appendix_d/` | |
| Appendix E (weight decay vs input noise sparsity) | `appendix_e/` | |
| Appendix F (truncation & similarity across sizes) | `appendix_f/` | results in `appendix_f_results.json` |
| Appendix M (adversarial masks vs regularization) | `appendix_m/` | metrics in `adversarial_results_*.json`; reuses saved checkpoints |

### Language experiments (`reproduction/language/`)

| Paper item | Directory | Notes |
|---|---|---|
| Figure 8 (sentiment-negation circuit) | `fig_08/` | |
| Figure 9 (low-rank activation approximations) | `fig_09/` | eigendecompositions cached per model |
| Appendix G (SAE loss added across layers) | `appendix_g/` | computed with the public `tdooms/ts-medium-scope` SAEs |
| Appendix H (SAE training-time effect, Fig. 24 / Table 5) | `appendix_h/` | results in `appendix_h_results.json` |
| Appendix I (Bilinear vs ReGLU vs SwiGLU, Table 6) | `appendix_i/` | trains 3 TinyStories models; `--mode table` builds both table rows |
| Appendix J (SwiGLU→bilinear finetuning, Fig. 25) | `appendix_j/` | fine-tuning LR is 3e-5 (the paper's 6e-4 pretraining LR diverges here) |
| Appendix N (self- vs cross-interactions, Fig. 27) | `appendix_n/` | |
| Appendix O (feature examples) | `appendix_o/` | |

### Not covered

- **Figure 1** is an architectural schematic (nothing to reproduce).
- **Appendices K and L** are mathematical derivations; the bias experiment
  discussed in Appendix L is reproduced as part of `fig_06/`.
- The **toy-model** code in `original/toy/` has no corresponding figure in
  the paper.

### Known deviations

- **The authors' cleaned TinyStories dataset was never published** (and the
  unofficial repository that held most of the original experiment code has
  since been made private). The public
  `tdooms/ts-medium` checkpoint scores ≈2.9 loss on raw
  `roneneldan/TinyStories` validation vs the ≈1.34 the paper reports on
  their own data — we verified this gap is not explained by tokenization,
  punctuation normalization, packing, padding, attention masking, or rotary
  conventions (all checked explicitly). Consequences: absolute losses on
  TinyStories are not comparable to the paper's; Appendix G's absolute
  loss-added values are distorted by this distribution shift (the SAE can
  act as a denoiser), though the qualitative ordering
  (resid_mid > mlp_out at every layer) reproduces, and per-layer
  reconstruction NMSE is reported as a data-robust complement.
- Appendix I trains on the public `roneneldan/TinyStories`; the
  between-activation-function comparison (the actual claim) is internally
  consistent, only absolute losses differ from Table 6.
- Appendix J fine-tunes with LR 3e-5: the paper's Table-3 value (6e-4) is a
  from-scratch pretraining LR and reproducibly diverges when fine-tuning the
  pretrained TinyLlama-1.1B with small batches.
- The training branch of the vendored `Attention` module passes an explicit
  attention mask together with `is_causal=True`, which current PyTorch
  rejects; training scripts route through the (functionally identical)
  inference branch instead.

---

## Extension: Bilinear MLPs in Variational Autoencoders

Beyond the reproduction, the repository extends weight-based interpretability
to a setting the original paper does not consider: generative models trained
without labels. Three VAE families place the bilinear layer in the encoder,
the decoder, or both, and an ordinary MLP VAE serves as the baseline.

### Layout

```
extension/
  bilinear-encoder/   bilinear encoder, MLP decoder      (11 experiments)
  bilinear-decoder/   MLP encoder, bilinear decoder      (18 experiments)
  bilinear-full/      both halves bilinear               (21 experiments)
  vanilla-vae/        MLP baseline checkpoints + trainer
```

Each project is self-contained: `models.py`, `train.py`, `analysis.py`,
`experiments/expNN_*.py`, pre-trained `checkpoints/`, and a `README.md` with
its results. Every experiment prints its headline numbers and writes them to
`figures/<dataset>/expNN_results.json`.

### Main findings

- **Encoder features are causal.** Synthetic inputs built purely from the
  weights are recognized by the encoder for 9–10 of 10 classes on MNIST,
  Fashion-MNIST and KMNIST, far above 10-seed untrained baselines, and
  pseudoinverse keys steer encodings into chosen classes.
- **The decoder fails by default, and the failure has a mechanism.** The
  interaction matrix is linear in the probe, so overlapping class-mean
  targets manufacture a spurious near-universal generative direction.
  Centering the targets is the exact correction and lifts causal generation
  from 3/10 to 9/10 (9.5 ± 0.5 on the fully bilinear model).
- **Recognition is high-rank, generation is rank one.** The corrected
  decoder structure is carried by the single top eigenvector, robustly
  across latent dimensions 10–32.
- **Purely bilinear decoders are exactly even**, `decode(z) ≡ decode(−z)`;
  a linear skip connection breaks the symmetry and fixes training.
- **Honest boundaries.** Gradient baselines on a vanilla VAE match the raw
  synthesis capability (the value of bilinearity is the closed-form, global,
  gradient-free answer), and a hypothesized encoder–decoder eigenvector
  alignment is null even when an explicit alignment loss demonstrably
  optimizes its own objective.

### Running

```bash
cd extension/bilinear-encoder && python run_all.py   # or bilinear-decoder / bilinear-full
python run_all.py 05 11                              # a subset
```

Checkpoints are included, so experiments run without any training. Each
project's `train.py` retrains its models from scratch if desired (existing
checkpoint files are never overwritten).

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ipantelidis/bilinear-mlp-repro.git  
cd bilinear-mlp-repro  
```

### 2. Create and Activate a Virtual Environment (Recommended)

We recommend using a virtual environment to avoid dependency conflicts.

```bash
python3 -m venv venv  
source venv/bin/activate  
```

Alternatively, if you use conda:

```bash
conda create -n bilinear-mlp python=3.10  
conda activate bilinear-mlp  
```

### 3. Install Dependencies

Install all required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt  
```

This installs all dependencies required to reproduce both the original experiments and the VAE extension, including PyTorch, torchvision, NumPy, and plotting utilities.

### 4. Export Original Module Path

Some reproduction scripts rely on the original module layout located in the `original/` directory. To ensure Python resolves these imports correctly, export the `original` directory to your `PYTHONPATH`.

From the repository root, run:

```bash
export PYTHONPATH=$(pwd)/original
```

The language-model scripts import `transformers`, which tries to initialize
TensorFlow if it is installed; with Keras 3 present this fails. Disable the
TensorFlow path (it is never needed here):

```bash
export USE_TF=0
```

After this setup, you can selectively reproduce individual figures, experiments, or appendix results by navigating to the corresponding directory and running the associated scripts. Each experiment is self-contained, allowing targeted reproduction without executing the full pipeline.


