# Full Bilinear VAE

Extension of Pearce et al. (ICLR 2025) to a VAE in which **both** the encoder
and the decoder contain a bilinear layer, so that weight-based
eigendecomposition applies on both sides of the latent space simultaneously.

## Model variants

| Variant | Description | Outcome |
|---|---|---|
| v1 | pure bilinear encoder + decoder, no biases | fails: `decode(z) ≡ decode(−z)` exactly (exp 11), recon MSE 0.064 |
| v2 | + linear skip connection in the decoder, KL annealing | works: recon MSE 0.028; **main checkpoint + 5 seeds used everywhere** |
| v3 | v2 + explicit encoder–decoder alignment loss | the loss as implemented is detached from the graph (zero gradient) — the shipped v3 checkpoint is effectively another v2 run; even a repaired loss leaves the corrected alignment metric at null (exp 19) |

Architecture (v2):

```
Encoder: x(784) → Linear(256, no bias) → Bilinear(512) → μ(10), logσ²(10)
Decoder: z(10)  → Linear(256, no bias) → Bilinear(512) → Linear(784, no bias) (+ linear skip) → Sigmoid
```

## Key results

| Finding | Value | Exp |
|---|---|---|
| Even-symmetry pathology of the pure bilinear decoder, proved exactly | max\|f(z)−f(−z)\| = 0.0 | 11 |
| Skip connection fixes it | recon MSE 0.064 → 0.028 | 11 |
| Causal generation (raw class-mean targets), 6 checkpoints | 4.7 ± 0.7 /10 | 18 (cf. 15) |
| **Causal generation with centered targets** | **9.5 ± 0.5 /10** | 18 |
| Decoder cross-class eigvec cos, raw → centered targets | 0.83 → 0.40 | 18 |
| Universal direction over training (raw targets) | 0.875 → 0.816 (declines) | 13 |
| Encoder cross-class cos over training | 0.40 → 0.22 (more discriminative) | 13 |
| Latent enc↔dec alignment (corrected measure) | **null**: 0.007 vs \|random\| 0.220; 5-seed 0.030 ± 0.141 | 07, 15 |
| Alignment loss (v3) cannot help: shipped implementation has zero gradient | v3's `alignment_loss` detaches both Jacobians (grad_fn=None); shipped v3 ≡ v2 (Jacobian cos 0.375 vs 0.376) | 19 |
| Alignment stays null even with a repaired, working loss | corrected metric: v2 0.03 ± 0.13, repaired λ=1 −0.14 ± 0.15, repaired λ=1000 −0.06 ± 0.12 — all inside the \|random\| band 0.26 ± 0.06, while λ=1000 lifts the loss's own Jacobian objective 0.376 → 0.465 | 19 |
| Causal decomposition (raw targets, single seed) | dec-only 3/10 → +full enc 5/10 → full 6/10 | 08 |
| Disentanglement (MIG proxy) | VanillaVAE 0.126 > full 0.097 > dec 0.091 > enc 0.061 | 12 |
| Truncated Q_dec reconstruction (raw targets) | no rank-2 plateau; full decode ≈ 8× better than any truncation | 16 |
| Truncated reconstruction, centered targets | still no plateau: k=1 (top +eigvec) already saturates — MSE 0.028 ± 0.005 vs full decode 0.0074; extra eigvecs add nothing (MSE flat) and hurt causal landing under the \|λ\| rule | 20 |
| Centered k=1 causal landing (6 ckpts) | 9.5 ± 0.5 /10 (reproduces exp 18); raw k=1: 4.7 ± 0.7 | 20 |
| FashionMNIST generalisation | NUC 0.869 (MNIST 0.808); causal 2/10 raw targets (MNIST 6/10); dec/enc seed consistency 0.962/0.333 (MNIST 0.941/0.326); latent alignment 0.424 on the main ckpt — resolved as a fluke by exp 21 | 17 |
| FMNIST alignment anomaly resolved | exp 17's 0.424 is the extreme of a wide seed spread: 6-ckpt mean 0.045 ± 0.233 (range −0.245 … +0.424) vs \|random\| 0.256 ± 0.058 — **null**, same as MNIST; per-class values flip sign across seeds | 21 |

The headline story: the pure bilinear decoder has a provable sign-symmetry
degeneracy that a linear skip removes; with the skip, weight-based synthesis
from **centered** (contrastive) targets works almost perfectly (9.5/10),
while the previously reported failures (≈4/10) were largely an artefact of
overlapping raw class-mean targets (see bilinear-decoder exp 13 for the full
diagnosis). Even so, the extractable causal structure is rank-1 per class:
truncation curves are flat beyond k=1 with either raw or centered targets, and
full decoding stays ≈4–8× more accurate than any weight-only synthesis
(exp 16/20). The hypothesised encoder–decoder eigenvector alignment, by
contrast, is a null result under its corrected measurement and stays null even
when a (repaired) explicit alignment loss demonstrably optimises its own
Jacobian objective during training (exp 19); it should be reported as such.

## Experiments

01 encoder eigvecs · 02 decoder eigvecs · 03 pixel-space alignment
(superseded by 07) · 04 causal generation · 05 4-way model comparison ·
06 seed consistency · 07 latent alignment (corrected; null) · 08 causal
decomposition · 09 encode–decode loop · 10 eigenspectra · 11 v1 failure
diagnosis · 12 disentanglement · 13 training dynamics · 14 alignment
interpolation · 15 5-seed significance · 16 truncated reconstruction ·
17 Fashion-MNIST summary · 18 contrastive targets · 19 alignment settled
(v2/v3/repaired loss; null) · 20 truncation with centered targets ·
21 FMNIST alignment anomaly (fluke; null)

Experiments print their headline numbers and also write them to
`figures/<dataset>/expNN_results.json`.

```bash
cd extension/bilinear-full
python run_all.py            # all experiments
python run_all.py 11 18      # a subset
```

## Checkpoints

`checkpoints/mnist/{model.pt, seeds/seed0..4.pt, epochs/epoch01..20.pt}` and
`checkpoints/fashion_mnist/{model.pt, seeds/seed0..4.pt}` (train with
`train.py`, `train_epochs.py`, `train_fmnist.py`; existing files are skipped).
Exp 19 reads its v3 checkpoints from `checkpoints/mnist/v3/` (shipped.pt plus
the repaired-loss `lam1_seed*` / `lam1000_seed*` seeds, trained by
`extension_full/proto_full_exp19_train.py`).

## Reference

Pearce, M., Dooms, T., Rigg, A., Oramas, J., & Sharkey, L. (2025).
*Bilinear MLPs enable weight-based mechanistic interpretability.* ICLR 2025.
