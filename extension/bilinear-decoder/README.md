# Bilinear Decoder Extension

Extension of Pearce et al. (ICLR 2025) to a VAE whose **decoder** contains a bilinear layer.

## Key idea

Because the bilinear decoder is a degree-2 polynomial in the latent code `z`, the output along any direction `p*` can be written as a quadratic form:

```
p* · ŷ(z) = z^T Q_dec z
```

`Q_dec` is built analytically from the decoder weights — no forward pass or training data needed. Eigendecomposing it reveals:
- **Positive eigenvectors**: latent directions the decoder uses to generate the target pattern.
- **Negative eigenvectors**: latent directions the decoder actively suppresses.

## Architecture

```
Encoder:  x(784) → Linear(256) → ReLU → Linear(512) → ReLU → μ(10), logσ²(10)
Decoder:  z(10)  → Linear(256, no bias) → BilinearLayer(512) → Linear(784, no bias) → Sigmoid
```

## Key findings

| Finding | Value |
|---------|-------|
| Decoder cross-class eigvec similarity (MNIST) | **0.842** |
| Decoder cross-class eigvec similarity (Fashion-MNIST) | **0.910** |
| Encoder cross-class eigvec similarity | **0.232** |
| Random DecBilinearVAE cross-class similarity | **0.864** |
| VanillaVAE decoded mean-latent similarity | 0.565 |
| Rank of generative subspace per class | 2–3 positive eigenvectors |
| Seed consistency (pixel-space cosine) | **0.993** |
| Causal accuracy (generate→encode→classify) | 3/10 (MNIST), 1/10 (Fashion-MNIST) |
| Synthesis quality penalty vs. VAE reconstruction | **5.8×** MSE |
| Mass ratio vs. reconstruction MSE correlation | r = −0.697, p = 0.025 |

The decoder has a **near-universal generative direction** — the top positive eigenvector of `Q_dec` is nearly identical across all digit/class directions (cosine ≈ 0.84–0.91). This contrasts sharply with the encoder (0.232) and means the decoder synthesises visually similar images regardless of the target class.

Crucially (Exp 12), a randomly initialised DecBilinearVAE shows *higher* cross-class similarity (0.864) than the trained model, while VanillaVAE shows only 0.565. This establishes that the near-universal direction is a **mathematical property of the bilinear Q_dec geometry**, not something learned during training, and is specific to the bilinear decoder architecture.

## Structure

```
bilinear-decoder/
├── models.py           # DecBilinearVAE (standard encoder + bilinear decoder)
├── train.py            # ELBO loss, training loop, checkpoint utilities
├── analysis.py         # get_decoder_interaction_matrix, decompose, class means
├── visualize.py        # shared plotting helpers
├── run_all.py          # run all experiments: python run_all.py [1..11]
├── checkpoints/
│   ├── mnist/
│   │   ├── model.pt         # main checkpoint (epoch 28)
│   │   └── seeds/           # seed{0..4}.pt (5 independent runs)
│   └── fashion_mnist/
│       └── model.pt         # epoch 20
├── figures/
│   ├── mnist/
│   └── fashion_mnist/
└── experiments/
    ├── exp01_synthesis.py       # D1: analytical synthesis per class
    ├── exp02_pixel_fields.py    # D2: per-pixel generative fields
    ├── exp03_causal_gen.py      # D3: generate→encode→classify loop + PCA
    ├── exp04_eigenspectrum.py   # D4: full eigenvalue spectrum per class
    ├── exp05_consistency.py     # D5: synthesis consistency across 5 seeds
    ├── exp06_generative_basis.py# D6: all positive eigvecs decoded per class
    ├── exp07_suppressor_map.py  # D7: decoded negative eigvecs + cross-suppression
    ├── exp08_mass_ratio.py      # D8: pos/neg mass ratio + spatial map
    ├── exp09_crossclass.py      # E1: encoder vs decoder cross-class + interpolation
    ├── exp10_synthesis_quality.py# E2+: quality ratio + mass-MSE scatter
    ├── exp11_fmnist.py          # Fashion-MNIST: synthesis + causal + cross-class
    └── exp12_decoder_control.py # Control: trained vs random vs VanillaVAE
```

## Usage

```bash
cd extension/bilinear-decoder
python run_all.py           # all 11 experiments
python run_all.py 1 9 11    # specific experiments
```

## Training

The checkpoints are pre-trained. To retrain from scratch:

```bash
cd /path/to/bilinear-mlp-repro
python extension_full/run.py --mode train --model dec_bilinear_vae --dataset mnist --epochs 30 --device cuda:0
python extension_full/run.py --mode train --model dec_bilinear_vae --dataset fashion_mnist --epochs 30 --device cuda:0
```

## Reference

Pearce, T. et al. *Bilinear MLPs enable weight-based mechanistic interpretability*. ICLR 2025.
