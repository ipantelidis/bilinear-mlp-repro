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
| Decoder cross-class eigvec similarity, raw targets (MNIST) | **0.842** |
| Decoder cross-class eigvec similarity, raw targets (Fashion-MNIST) | **0.910** |
| Decoder cross-class eigvec similarity, centered targets (MNIST, exp13) | **0.429** |
| Encoder cross-class eigvec similarity (class-mean μ encodings, exp09) | **0.350** |
| Random (untrained) DecBilinearVAE cross-class similarity | **0.646** |
| VanillaVAE decoded mean-latent similarity | 0.565 |
| Rank of generative subspace per class | 2–3 positive eigenvectors |
| Seed consistency (pixel-space cosine) | **0.993** |
| Causal accuracy (generate→encode→classify) | 3/10 (MNIST), 1/10 (Fashion-MNIST) |
| Synthesis quality penalty vs. VAE reconstruction (raw targets) | **5.8×** MSE (centered: 2.6×) |
| Mass ratio vs. reconstruction MSE correlation | r = −0.697, p = 0.025 — **fragile**, see below |

With raw targets the decoder has a **near-universal generative direction** — the top positive eigenvector of `Q_dec` is nearly identical across all digit/class directions (cosine ≈ 0.84–0.91). Exp 09 compares like-for-like: decoder raw targets 0.842 vs decoder centered targets 0.429 vs encoder with class-mean latent encodings 0.350 — with centered targets the decoder is nearly as class-discriminative as the encoder, so the raw overlap is mostly a target artefact (see exp 13). (Earlier versions of exp09 used latent basis directions μ\* = e_k for the encoder, giving 0.232, but mislabeled them as digit classes; the encoder side now uses class-mean encodings.)

**Mass-ratio correlation is fragile (exp10):** the Pearson r = −0.697 (p = 0.025) hinges on outlier class d1 (mass ratio 0.243 vs ≈ 0.09–0.13 for all other classes). Spearman rank correlation is not significant (ρ = −0.479, p = 0.162), and leave-one-out Pearson spans r ∈ [−0.773, −0.324]; dropping d1 alone gives r = −0.324 (p = 0.395). The correlation should not be treated as established.

Exp 12 shows a randomly initialised DecBilinearVAE already has high cross-class similarity (**0.646**) — well above the VanillaVAE decoded-mean-latent baseline (0.565) — and training *amplifies* it to 0.842. The near-universal direction is therefore **seeded by the bilinear Q_dec geometry at initialisation and reinforced by training**, rather than purely learned (as for a vanilla decoder) or purely architectural.

## The resolution (exp 13): the universal direction is mostly a target artefact

`Q_dec` is linear in the target direction `p*`, and raw class-mean images overlap
heavily (pairwise cosine 0.742 on MNIST, 0.771 on Fashion-MNIST) — so their
interaction matrices, and hence their top eigenvectors, inherit that overlap.
Repeating every analysis with **centered targets** `p*_c − mean(p*)` removes the
shared-ink component and flips the conclusions:

| Metric (MNIST) | Raw targets | Centered targets |
|---|---|---|
| Target pairwise cosine | 0.742 | 0.240 |
| Top-eigvec cross-class cosine | 0.842 | **0.429** (random model: 0.289) |
| Causal generation (synthesize→encode→classify) | 3/10 | **9/10** (random model: 1/10) |
| Synthesis MSE to class mean | 0.088 | **0.040** (VAE reconstruction: 0.016) |
| MSE penalty vs. VAE reconstruction | 5.8× | **2.6×** |

Fashion-MNIST: causal 2/10 → **6/10**; synthesis MSE 0.189 → **0.063** (VAE
reconstruction 0.013; penalty 19.3× → 5.1×). With centered targets the decoder
synthesises recognisable, class-specific images from weight eigenvectors alone
(`exp13_synthesis_centered.png`) — weight-based synthesis *works* once the
question is contrastive ("what distinguishes class c from the average image")
rather than absolute ("what generates class c's ink"). The remaining
similarity above the random-model baseline (0.43 vs 0.29) is the genuine
architectural component. Two other facts to note alongside: the decoder is
exactly even (`decode(z) ≡ decode(−z)`, max diff 0.0 — no biases anywhere), and
the raw-target analyses above are retained as the cautionary baseline.

## Beyond centering (exp 14): nothing beats it — negative result

Three alternative contrastive constructions were compared against simple
centering on MNIST:

| Method | Causal | Cross-class cos | Synth MSE |
|---|---|---|---|
| raw (baseline) | 3/10 | 0.842 | 0.088 |
| **centered (exp13)** | **9/10** | 0.429 | **0.040** |
| pairwise contrast (mean_c − mean_nearest-confusable) | 7/10 | **0.365** | 0.046 |
| deflation (project Q̄'s top +eigvec out of Q_c) | 3/10 | 0.890 | 0.094 |
| generalized eigenproblem (Q_c vs Q̄₊ = \|Q̄\| + εI, best ε=0.01) | 4/10 | 0.572 | 0.100 |

**None of the alternatives beats simple centering.** Pairwise contrast is the
runner-up — its directions overlap even less (0.365) but causal accuracy and
MSE are worse. The explanation is exact: `Q_dec` is *linear* in `p*`, so
`Q(p_c − g) = Q_c − Q̄` (verified to ~1e-4) — centering already performs
*full-matrix* deflation of the shared component. `Q̄` is far from rank-1 (top
|λ| eigenvalue share 0.39, dominant eigenvalues negative), so projecting out
one or a few of its eigenvectors removes only a fraction of the shared
structure while distorting the eigenspace (a depth sweep k = 1..6 makes things
monotonically worse, down to 1/10). Note also that the one-vs-rest contrast
`mean_c − mean_{c'≠c}` equals `(10/9)(mean_c − g)` — direction-wise it *is*
centering. Centering is not just sufficient; among target-linear
constructions, it is essentially the right answer.

### Latent-dimension sweep (exp 16): rank-1 and centering are not artifacts of d=10

The strongest predictable objection — that with d_latent = 10 ≈ #classes the
rank-1 structure and the centering fix are artifacts of a cramped latent
space — is tested directly: 3 fresh seeds at each of d_latent ∈ {10, 20, 32},
identical recipe, full battery per model.

| d_latent | causal raw | causal centered | eigvec cos raw → centered (random) | top-λ share of + mass | best trunc. k |
|---|---|---|---|---|---|
| 10 | 4.0 ± 0.0 | 8.7 ± 0.6 | 0.833 → 0.426 (0.250) | 0.696 | 1 |
| 20 | 6.3 ± 1.2 | **9.0 ± 0.0** | 0.880 → 0.439 (0.202) | 0.696 | 1 |
| 32 | 6.0 ± 1.0 | **9.0 ± 0.0** | 0.879 → 0.434 (0.163) | 0.690 | 1 |

Everything survives — and sharpens — with dimension: centered causal accuracy
is exact 9/10 at d=20 and d=32; k=1 truncation remains optimal at every
dimension (extra eigenvectors never help, even with ~16 positive eigenvalues
available at d=32); the top eigenvalue's share of positive mass is strikingly
constant (~0.69); and since the random-model baseline falls with dimension
while the centered similarity stays ~0.43, the learned class structure grows
*more* distinguishable from chance as the latent space widens.

### The gradient baseline (exp 17): what does the bilinear constraint actually buy?

A fair skeptic asks whether the centered-target results need the bilinear
decoder at all: ordinary gradient machinery can search for latent directions
on *any* decoder. Exp 17 runs the same protocol (centered targets
`p*_c − mean(p*)`, per-model norm budget = mean latent norm, causal test =
decode → encode → nearest class-mean latent) with gradient-based direction
finders on a standard MLP-decoder VAE (VanillaVAE from `../vanilla-vae/`,
same encoder as DecBilinearVAE; decoder z(10) → Linear(256) → ReLU →
Linear(784) → Sigmoid):

| Method (MNIST, centered targets) | Causal | Synth MSE | Needs gradients | Globally valid? |
|---|---|---|---|---|
| weight-eig (DecBilinearVAE, exp13 reference) | 9/10 | 0.040 | no — closed form from weights | **yes** — exact quadratic form over all of latent space; full generative + suppressor eigenbasis in one decomposition |
| grad-ascent (DecBilinearVAE) | **10/10** | 0.039 | yes (8 restarts × 300 Adam steps/class) | no — one local optimum per run |
| grad-ascent (VanillaVAE) | **10/10** (all of main + 3 seeds) | 0.036 | yes (8 restarts × 300 Adam steps/class) | no — one local optimum per run |
| jacobian 1-step, `J^T p*` at z=0 (VanillaVAE) | **10/10** (all of main + 3 seeds) | 0.032 | yes (one backward pass) | no — linearization at z=0 |
| jacobian 1-step (DecBilinearVAE) | — | — | — | degenerate: the bias-free quadratic decoder has `J^T p* ≡ 0` at z=0 |
| random-init controls (all three procedures) | 1/10 | 0.179 | — | — |

(VAE reconstruction anchors: bilinear 0.016, vanilla 0.012. All methods use
data only to define the targets and the norm budget; the gradient methods
additionally need forward/backward passes through the decoder.)

**Gradient-based baselines match — in fact slightly exceed — the weight-based
protocol on causal accuracy and MSE.** Even the one-backward-pass Jacobian
direction on a plain VAE gets 10/10, robust across 4 training seeds. The
bilinear constraint therefore does **not** buy raw capability at this task:
per-class generative directions are easy to find with autograd on any trained
decoder. What it buys is the *form* of the answer: `Q_dec` is data-free at
extraction time, gradient-free, exact everywhere in latent space (not a local
optimum or a linearization at one point), and delivers the entire spectrum —
generative and suppressor directions and their eigenvalue mass — in a single
decomposition. The honest framing of the decoder result is interpretability /
global validity, not synthesis power. Two sanity checks bracket this: gradient
ascent on the bilinear model itself reaches 10/10 (it can exploit the same
structure plus the output sigmoid that `Q_dec` ignores), and the one-step
Jacobian fails *structurally* on the bilinear decoder (no linear term at the
origin) — the weight-based eigendecomposition is precisely the second-order
object that replaces it.

## Key-findings table rows (optional additions)

| Finding | Value |
|---------|-------|
| Gradient ascent on VanillaVAE, causal (centered, exp17) | **10/10** (4/4 seeds; weight-based on bilinear: 9/10) |
| One-step Jacobian `J^T p*` on VanillaVAE, causal (exp17) | **10/10** (4/4 seeds) |

### KMNIST (exp 18): the artifact-and-fix story generalises to a third dataset

| Metric (KMNIST) | Raw targets | Centered targets |
|---|---|---|
| Target pairwise cosine | 0.821 | 0.237 |
| Top-eigvec cross-class cosine | 0.833 (random model: 0.921) | **0.460** (random model: 0.268) |
| Causal generation (synthesize→encode→classify) | 1/10 | **7/10** (random model: 1/10) |
| Synthesis MSE to class mean | 0.141 | **0.054** (VAE reconstruction: 0.021) |
| MSE penalty vs. VAE reconstruction | 7.0× | **2.5×** |

Cross-dataset centered picture: MNIST 0.429, FMNIST 0.402, KMNIST 0.460, all vs
random ≈ 0.27–0.30. The three centered misses (ma→ki, re→ki, wo→su) sit in a
persistent ki/su/ma/re/wo cluster that stays 0.7–0.98 similar even after
centering: contrastive targets separate classes only up to genuinely shared
visual components.

## GANSpace control (exp 15)

The earlier working-directory claim that PCA over latent codes "converges to"
the analytical directions was an artefact of a degenerate protocol (identical
first-n samples every repeat, std = 0) and a saturated metric (best-match
|cos| against up to 10 directions in a 10-d space has a high chance floor).
Redone with true random subsets and a random-orthonormal-basis floor: the top
PCA directions match the centered analytical eigenvectors **no better than
chance** (m=1: 0.22 vs floor 0.26; m=3: 0.34 vs floor 0.43) at every sample
count; only the complete 10-direction basis edges marginally above its floor
(0.70 vs 0.61 ± 0.04). PCA finds data-variance directions, Q_dec finds
generative-output directions — the methods are complementary, not redundant.

## Structure

```
bilinear-decoder/
├── models.py           # DecBilinearVAE (standard encoder + bilinear decoder)
├── train.py            # ELBO loss, training loop, checkpoint utilities
├── analysis.py         # get_decoder_interaction_matrix, decompose, class means
├── visualize.py        # shared plotting helpers
├── run_all.py          # run all experiments: python run_all.py [1..14]
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
    ├── exp12_decoder_control.py # Control: trained vs random vs VanillaVAE
    ├── exp13_contrastive_targets.py # Centered p*: synthesis + causal rescue (see above)
    └── exp14_beyond_centering.py    # Pairwise / deflation / generalized vs centering (negative)
```

Experiments 02, 04, 06–10, 13, 14 also write their quantitative results to
`figures/expNN_results.json`.

## Usage

```bash
cd extension/bilinear-decoder
python run_all.py           # all 14 experiments
python run_all.py 1 9 14    # specific experiments
```

## Training

The checkpoints are pre-trained. To retrain from scratch (same recipe:
epochs 30, lr 1e-3, weight decay 0.01, β=1, input noise 0.3, batch 128,
AdamW + cosine LR, best-test checkpointing; existing files are never
overwritten):

```bash
cd extension/bilinear-decoder
python train.py --dataset mnist                        # checkpoints/mnist/model.pt
python train.py --dataset fashion_mnist                # checkpoints/fashion_mnist/model.pt
python train.py --dataset kmnist                       # checkpoints/kmnist/model.pt
python train.py --dataset mnist --seed 0               # checkpoints/mnist/seeds/seed0.pt (0–4)
python train.py --dataset mnist --d-latent 20 --seed 0 # checkpoints/mnist/latent_sweep/d20_seed0.pt
```

## Reference

Pearce, T. et al. *Bilinear MLPs enable weight-based mechanistic interpretability*. ICLR 2025.
