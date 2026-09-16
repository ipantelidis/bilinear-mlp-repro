# The Bilinear-VAE Extension: Complete Experiment Reference

This document describes every experiment in the extension — what it does, how
it does it, why it exists, and what it established — across the three
sub-projects (`bilinear-encoder`, `bilinear-decoder`, `bilinear-full`), plus
the pipeline-level results and the discarded precursors. It is the source
document for writing the manuscript's extension section. Every number quoted
here is serialized in a `figures/**/expNN_results.json` next to its figure.

**Status tags** used throughout:
- **HEADLINE** — belongs in the main text of the paper.
- **SUPPORTING** — one paragraph or table row in the main text.
- **APPENDIX** — reproduced and solid, but secondary.
- **HONEST CAVEAT** — a negative/fragile finding that must be stated, briefly.
- **SUPERSEDED** — kept for the record; a later experiment replaces it.

---

## 0. Shared machinery, architectures, and conventions

### 0.1 The mathematical object

A bilinear layer computes `h = (W_L x) ⊙ (W_R x)` with no element-wise
nonlinearity, so any scalar readout of the layer is an exact quadratic form in
its input. Both sides of the VAE use the same recipe with different geometry:

- **Encoder** (`bilinear-encoder`): for a latent direction μ\* ∈ ℝ¹⁰, the
  activation μ\*·μ(x) equals xᵀ Q x with **Q ∈ ℝ⁷⁸⁴ˣ⁷⁸⁴** built from the
  weights: u = P_μᵀ μ\*, Q_embed = ½(W_Lᵀ diag(u) W_R + sym), Q = Eᵀ Q_embed E.
  Eigenvectors are pixel-space patterns; λ>0 activates the direction, λ<0
  suppresses it.
- **Decoder** (`bilinear-decoder`, `bilinear-full`): for a pixel-space target
  p\* ∈ ℝ⁷⁸⁴, the pre-sigmoid output satisfies p\*·logits(z) = zᵀ Q_dec z with
  **Q_dec ∈ ℝᵈˣᵈ** (d = latent dim): u = P_outᵀ p\*, Q_embed as above,
  Q_dec = E_decᵀ Q_embed E_dec. Eigenvectors are latent directions.

Both are symmetrized (½(Q+Qᵀ)) before `eigh`. Crucially, **Q is linear in the
probe direction** (μ\* or p\*) — the single most consequential fact in the
whole extension (see decoder exp 13/14).

### 0.2 Architectures and checkpoints

| Model | Architecture | Checkpoints |
|---|---|---|
| BilinearVAE (encoder project) | x(784) → Linear(256, no bias) → **Bilinear(512)** → μ, logσ² (10); MLP decoder | MNIST (30 ep, best ep 11 — the encoder VAE overfits, disclosed), 5 seeds; FMNIST; KMNIST (best ELBO 252.07, ep 11) |
| DecBilinearVAE (decoder project) | MLP encoder (784→256→512→μ/logσ² 10); z(10) → Linear(256, no bias) → **Bilinear(512)** → Linear(784, no bias) → Sigmoid | MNIST (ep 28), 5 seeds; FMNIST (ep 20); KMNIST (ELBO 284.47, ep 4); latent sweep d∈{10,20,32}×3 seeds |
| FullBilinearVAE (full project) | both sides bilinear; **v1** pure (fails, exp 11), **v2** + linear skip in decoder + KL annealing (the workhorse), **v3** + alignment loss (see exp 19) | MNIST main + 5 seeds + 20 per-epoch ckpts; FMNIST main + 5 seeds (seed4 retrained with BCE clamp + grad clipping after divergence); v3 + repaired-v3 seeds in `extension_full/` |

A structural fact discovered along the way: **every bias-free bilinear decoder
is exactly even** — decode(z) ≡ decode(−z), verified to machine zero for both
DecBilinearVAE and FullBilinearVAE-v1 — because every term in the output is a
product of exactly two z-linear factors. For DecBilinearVAE the (flexible MLP)
encoder adapts around this ℤ₂ quotient; for the pure full model it is fatal
(exp 11). Eigenvector sign ambiguity is therefore harmless on the decoder side.

### 0.3 Protocol conventions (used consistently, stated once)

- **Class-mean directions**: encoder analyses use μ\* = mean posterior μ over
  a class's test images; decoder analyses use p\* = the class-mean image.
- **Centered targets** (decoder): p\*_c − (1/10)Σ_c' p\*_c' — the corrected
  protocol established by exp 13.
- **Causal generation test** (decoder side): take the top positive eigenvector
  of Q_dec, scale to the model's mean latent norm, decode, re-encode, classify
  by nearest class-mean latent (cosine). /10 score.
- **Causal max-activating test** (encoder side): scale the top eigenvector to
  the mean image norm, encode it, classify by nearest class-mean latent. /10.
- **Random baselines**: untrained same-architecture models. After a
  reproducibility fix, encoder-side baselines are reported over **10 seeded
  inits (mean ± std + the counts list)** — single draws had varied 1–6/10 and
  earlier quotes of "2/10" understated the baseline. Decoder-side chance-floor
  baselines (exp 15) use 200 random orthonormal bases.
- Every experiment writes its figures plus a results JSON; figures use
  label-preserving axes (no `axis("off")`) and labeled rows.

---

## PART I — The encoder study (`bilinear-encoder`, 11 experiments)

**The question:** does weight-based eigendecomposition survive the move from
classifiers (where every output has a label) to an *unsupervised generative*
model's encoder — and are the recovered features causal?

### Exp 01 — Latent dictionary
**What:** sets μ\* = e_k for each of the 10 latent coordinates and shows the
top activating and top suppressing eigenvector per coordinate — a complete
"what does each latent neuron respond to" dictionary computed from weights
alone, zero forward passes.
**How:** Q per basis direction → eigh → project top ±eigenvectors to pixels.
**Why:** the most basic transfer test; the original paper's Figure-2 analogue
for a model with no labels.
**Achieves:** ten visually distinct, structured receptive fields — the encoder
learned specialized detectors, not a degenerate code. Consistent asymmetry:
top negative eigenvalues are 1.5–2.5× larger than positive ones (λ₊ 0.10–0.21
vs λ₋ 0.17–0.30) — the encoder represents "what this dimension is *not*" more
sharply than what it is. *Qualitative.* — **SUPPORTING**

### Exp 02 — Truncation: how many eigenvectors carry the activation?
**What:** correlates the true encoder activation μ\*·μ(x) with its rank-k
eigen-approximation Σ_{i≤k} λ_i (v_iᵀx)², per digit class, k = 1..20.
**How:** class-mean directions; Pearson r over the test set.
**Why:** the paper's classifiers are rank-3 objects; is a generative encoder?
**Achieves:** r > 0.90 requires k ≈ 8–13 for most classes; digit 8 (the most
complex) needs ~18–20; r at k=1 ranges 0.0–0.55. **The VAE encoder is
intrinsically higher-rank than a classifier** — it must encode distributional
geometry, not one decision boundary. One half of the paper's "rank duality"
result (the other half is decoder exp 20/16). — **HEADLINE (as part of the
rank-duality result)**

### Exp 03 — Cross-class similarity of top eigenvectors
**What:** the 10×10 matrix of |cos| between the top eigenvectors of all
class-mean directions.
**How:** class means from test encodings; top positive eigenvector each; |cos|.
**Why:** if the encoder organizes features around visual structure, visually
confusable digits should share eigenvectors — with **no label ever seen**.
**Achieves:** the matrix reproduces human visual confusability: (4,9)=0.81,
(0,6)=0.61, (1,7)=0.57, (3,5)=0.44, with the dissimilar pair (3,8)=0.21 as a
built-in control; off-diagonal mean 0.350. ELBO training alone internalized
the visual taxonomy of MNIST into the weights. — **SUPPORTING (showcase
figure candidate)**

### Exp 04 — Semantic difference directions
**What:** μ\* = μ_A − μ_B for the four most confusable pairs; positive
eigenvectors show what makes A an A, negative what makes B a B.
**How:** normalized difference directions → Q → eigh → top ±4 each.
**Why:** contrastive interpretability — the encoder-side ancestor of the idea
that later becomes decisive on the decoder side (centering).
**Achieves:** interpretable contrastive features plus meaningful eigenvalue
asymmetries — "1-ness" is concentrated (λ = +0.224) while "7-ness" is diffuse
(λ = −0.118); for (0,6) and (3,5) the suppressive side dominates. —
**APPENDIX**

### Exp 05 — Maximally activating input test (the causal experiment)
**What:** is the top eigenvector *causally* what the weights say it is? Scale
it to a realistic image norm, feed it through the actual encoder, and check
whether it lands nearest its own class mean.
**How:** per class: α·v₁ with α = mean image L2 norm → encode → nearest
class-mean latent by cosine. Control: identical procedure on untrained models —
now over 10 seeded inits.
**Why:** upgrades every "looks interpretable" claim to a falsifiable causal
one; no analogue exists in the original paper.
**Achieves:** **9/10 correct** (only 8→1 fails — consistent with digit 8 being
hardest in exp 02 and its high (8,1) similarity), synthetic encodings land
well inside class neighborhoods (cos to true mean 0.62–0.86). Random-init
baseline: **4.4 ± 1.6** (counts [6,3,6,3,7,5,4,2,3,5]; max draw 7) — the
trained score beats every draw. NOTE: earlier documents quoted "2/10 random";
that was a single lucky draw and is superseded. — **HEADLINE #1**

### Exp 06 — Cross-suppression map
**What:** does the pattern that *suppresses* class c look like the pattern
that *activates* its nearest confusable class d?
**How:** |cos(top-negative eigvec of c, top-positive eigvec of d)|, 10×10.
**Why:** tests whether the suppressive spectrum encodes implicit inter-class
boundaries.
**Achieves:** strongest pairs align with confusability — (2,9)=0.72,
(6,7)=0.67, (3,6)=0.66, (7,6)=0.64 — and digit 1 suppresses like nothing else
(≤0.2): it is visually isolated and the weights know it. — **APPENDIX**

### Exp 07 — Weight-based saliency without backpropagation
**What:** per-pixel sensitivity s(x) = |2Qx| — a saliency map that is one
matrix–vector product, no autograd; plus its rank-k progression.
**How:** precomputed Q per class; full map vs rank-1/3/10 approximations on
real test digits.
**Why:** a concrete practical payoff of having Q in closed form: one matrix
serves every image, where a standard VAE needs backprop per image.
**Achieves:** maps highlight the class-defining strokes; rank-3 already
reproduces the full map, converged by rank-10 — the spatial counterpart of
exp 02. — **SUPPORTING / APPENDIX**

### Exp 08 — Eigenvector consistency across training runs
**What:** are the eigenvectors properties of the task or of the seed?
**How:** 5 independently trained models; mean pairwise |cos| of eigenvectors
matched by rank, per class.
**Why:** direct analogue of the original paper's Fig 5A; also delimits how
deep into the spectrum the features are meaningful.
**Achieves:** rank-1 consistency **0.812 ± 0.060** (per class 0.68–0.90),
squarely inside the original's 0.80–0.90 classifier range; decays to ~0.2 by
rank 5+. Only the top of the spectrum is seed-stable — stated honestly. —
**SUPPORTING**

### Exp 09 — Fashion-MNIST
**What:** exps 01/03/05 repeated on FMNIST.
**Achieves:** causal test **10/10** vs random-init **3.5 ± 1.9** (counts
[2,1,5,1,2,6,3,6,5,4]); cross-class off-diagonal mean 0.275; garment-edge
dictionaries. First generality axis. — **SUPPORTING**

### Exp 10 — Adversarial steering via pseudoinverse masks
**What:** the encoder-side analogue of the original paper's §4.4: input-space
masks, built purely from the weights, that steer the encoder toward a chosen
class.
**How:** stack the top-10 eigenvectors per class; the rows of the pseudoinverse
act as "keys" that activate a chosen eigenvector with specificity
(v_j·(V⁺)_i = δ_ij); add a scaled key to test images; measure the rate of
landing in the target class vs a random-mask control across scales.
**Why:** second, independent causal validation — and the pseudoinverse detail
matters: the naive precursor (adding ε·v₁ directly) performed *below chance*
(documented in the working directory: nearest-class accuracy 0.029 vs 0.10
chance), exactly mirroring the original paper's reasoning for pseudoinverse
keys.
**Achieves:** steering rate 0.216 / **0.677** / 0.926 / 0.948 at σ = 0.5/1/2/4
vs random 0.029/0.035/0.046/0.083 — **19× at σ=1**; hardest cases: digit 9 at
σ=1 (0.267), digit 3 saturates at 0.53. — **HEADLINE #1 (co-evidence)**

### Exp 12 — KMNIST
**What:** exps 01/03/05 on KMNIST (10 Hiragana; higher intra-class
variability).
**Achieves:** causal **9/10** (only su→ki) vs random **3.5 ± 1.6** (counts
[4,1,3,4,1,6,5,2,4,5]); cross-class off-diagonal mean 0.280 with top pairs
(ki,re)=0.839, (ki,su)=0.755, (su,re)=0.678 — the one causal miss sits exactly
in the most-entangled character cluster. Third dataset, same story. —
**SUPPORTING**

**Part I verdict:** the encoder transfer is a clean success with two novel
quantitative findings (higher intrinsic rank; the causal 9–10/10 results) and
consistent behavior across three datasets.

---

## PART II — The decoder study (`bilinear-decoder`, 18 experiments)

**The question:** does the same machinery work on the *output* side — can the
decoder's weights tell us how images are generated? The answer became the
paper's centerpiece: a naive transfer fails for an identifiable mathematical
reason, and the corrected protocol works almost perfectly.

### Exp 01 — Analytical synthesis (the motivating picture)
**What:** for each class, decode the top positive and top negative eigenvector
of Q_dec(class-mean image), next to the class mean itself.
**Achieves:** the failure made visible: every class's "+eig" decodes to nearly
the same fat "8"-like blob, every "−eig" to a "1"-like shape, regardless of
target class. This figure motivates everything that follows. — **SUPPORTING
(as the "before" picture)**

### Exp 02 — Per-pixel generative fields
**What:** p\* = e_i for single pixels on a 7×7 grid; decoded fields and the
effective rank of each pixel's Q.
**Achieves:** smooth, localized generative fields; effective rank mean 4.74
(range 2.0–6.6) across pixels. — **APPENDIX**

### Exp 03 — Causal generation, raw targets (the original negative)
**What:** the decoder-side causal test with raw class-mean targets.
**Achieves:** **3/10**; landings [0,8,8,3,8,8,0,8,8,8] — everything collapses
toward digit 8; a random-init model decodes to uniform gray and scores 1/10.
— **SUPPORTING (the "before" number)**

### Exp 04 — Eigenspectra
**What:** all 10 eigenvalues of Q_dec per class. **Achieves:** 2–3 positive
eigenvalues per class (n₊ = [2,3,2,2,2,2,2,2,2,2]); few dominant. —
**APPENDIX**

### Exp 05 — Synthesis consistency across seeds
**What:** pixel-space cosine of the synthesized image across 5 seeds.
**Achieves:** **0.9928** mean pairwise — the synthesis is highly seed-stable.
**Honest caveat:** the *latent-space* eigenvectors themselves are only ~0.21
consistent across seeds; the pixel-level stability arises through the
decoder's smoothing. Both numbers must be reported together. — **SUPPORTING +
HONEST CAVEAT**

### Exp 06 — Generative basis
**What:** decodes *all* positive eigenvectors per class. **Achieves:** only
2–3 exist and beyond the first they add little variety. — **APPENDIX**

### Exp 07 — Suppressor map
**What:** decodes top negative eigenvectors; cross-suppression matrix.
**Achieves:** honest null: the off-diagonal mean (0.559) slightly *exceeds*
the diagonal (0.517) — decoder suppressors are not class-specific, consistent
with the universal-direction picture. — **APPENDIX / HONEST CAVEAT**

### Exp 08 — Positive/negative mass ratio
**What:** per-class ratio of positive to total |eigenvalue| mass + a 784-pixel
spatial map of the ratio. **Achieves:** descriptive structure (spatial mean
0.169, max 0.899). — **APPENDIX**

### Exp 09 — Encoder vs decoder cross-class similarity (corrected)
**What:** the side-by-side that quantifies how much less class-specific the
decoder's top eigenvectors are than the encoder's.
**How:** after a protocol fix (the original panel used latent *basis*
directions mislabeled as digits), all three panels now use class-mean
directions: decoder raw / decoder centered / encoder.
**Achieves:** decoder raw **0.842** vs decoder centered **0.429** vs encoder
**0.350** (encoder range 0.08–0.81). — **SUPPORTING**

### Exp 10 — Synthesis quality and the mass–MSE correlation
**What:** (a) MSE of weight-based synthesis vs actual VAE reconstruction;
(b) does a class's positive-mass ratio predict its reconstruction error?
**Achieves:** (a) raw-target synthesis costs **5.8×** the reconstruction MSE
(3.7–8.8× per class) — the quantitative form of the failure. (b) Pearson
r = −0.697 (p = 0.025) **but** Spearman ρ = −0.479 (p = 0.16, n.s.) and
leave-one-out shows the result hinges on outlier class 1 (drop it:
r = −0.324, p = 0.40). **The correlation is not established and should not be
claimed.** — (a) **SUPPORTING**; (b) **HONEST CAVEAT / CUT from claims**

### Exp 11 — Fashion-MNIST battery
**Achieves:** universality even stronger (0.910); causal raw 2/10. The failure
is not MNIST-specific. — **SUPPORTING**

### Exp 12 — Is the universal direction architectural or learned?
**What:** the control that disambiguates: trained DecBilinearVAE vs *untrained*
DecBilinearVAE vs VanillaVAE.
**Achieves:** trained 0.842, random-init **0.646**, VanillaVAE decoded-means
0.565 (DecBilinear decoded-means 0.625). The direction is **seeded by the
bilinear geometry at initialization and amplified by training** — neither
purely learned nor purely architectural. (An earlier README claimed
random = 0.864 > trained; that was a transcription error, corrected.) —
**SUPPORTING**

### Exp 13 — Contrastive (centered) targets: the fix ★
**What:** the pivotal experiment. Diagnosis: Q_dec is linear in p\*, and raw
class-mean images share most of their ink — so their interaction matrices, and
hence top eigenvectors, inherit that overlap. The "universal direction" is
largely the shared component *of the question being asked*. Fix: center the
targets, p\*_c − global mean, asking "what distinguishes class c from the
average image".
**How:** full battery, MNIST + FMNIST, with random-init controls.
**Achieves (MNIST / FMNIST):** target overlap 0.742→0.240 / 0.771→0.436;
top-eigvec cross-class 0.842→**0.429** (random 0.289) / 0.910→**0.402**
(random 0.301); causal generation **3/10→9/10** / **2/10→6/10** (random 1/10);
synthesis MSE 0.088→**0.040** (recon 0.016; penalty 5.8×→**2.6×**) /
0.189→**0.063** (recon 0.013; 19.3×→**5.1×**). The synthesis grid — universal
blobs above, recognizable class-specific digits below — is the single most
compelling figure of the extension. — **HEADLINE #2 (the centerpiece)**

### Exp 14 — Beyond centering: why the fix is canonical ★
**What:** could something smarter beat centering? Tests pairwise-contrast
targets (vs nearest confusable), eigenvector deflation of the shared matrix,
and a generalized eigenproblem (maximize class output relative to a PSD
surrogate of the average).
**Achieves:** nothing beats centering — centered 9/10 (cos 0.429,
MSE 0.040) vs pairwise 7/10 (0.365, 0.046), deflation 3/10 (0.890 — *worse*
than raw!), generalized 4/10 (0.572, 0.100). And the explanation is exact:
**Q(p−g) = Q(p) − Q(g)** (verified, max deviation 9.2e-5) — centering already
performs *full-matrix* deflation of the shared component, while rank-k
eigenvector deflation cannot reproduce Q̄ (its top-|λ| share is only 0.39, and
its dominant eigenvalues are negative; a deflation depth sweep k=1..6
monotonically degrades to 1/10). Bonus identity: one-vs-rest targets equal
(10/9)·centered, so that family collapses to centering too. Centering is not
a trick; it is the canonical correction. — **HEADLINE #2 (the proof half)**

### Exp 15 — GANSpace/PCA control
**What:** does PCA over latent codes (the GANSpace recipe) find the same
directions as the weight-based eigendecomposition? A working-directory
predecessor claimed convergence at ~0.65–0.70 — but it reused the identical
first-n samples every "repeat" (std exactly 0) and its metric (best-match
|cos| against up to 10 directions in a 10-d space) has a high chance floor.
**How:** true random subsets (5 repeats × 8 sample counts, 10→5000) and a
200-draw random-orthonormal-basis floor.
**Achieves:** PCA's top directions match the centered analytical eigenvectors
**no better than chance** (m=1: ~0.22 vs floor 0.264±0.090; m=3: ~0.34 vs
0.431±0.075) at every sample count; only the complete 10-direction basis edges
marginally above its floor (0.70 vs 0.608±0.035). PCA finds data-variance
directions; Q_dec finds generative-output directions. **Complementary, not
redundant** — no amount of sampling makes PCA reproduce the weight-based
directions. — **SUPPORTING**

### Exp 16 — Latent-dimension sweep: robustness of rank-1 and centering ★
**What:** the strongest predictable reviewer objection — "d_latent = 10 ≈
number of classes; your rank-1 and centering findings are artifacts of a
cramped latent space" — tested head-on.
**How:** 3 fresh seeds at each of d ∈ {10, 20, 32}, identical recipe; full
battery + truncation per model.
**Achieves:**

| d | causal raw | causal centered | eigvec raw→centered (random) | top-λ share | best k |
|---|---|---|---|---|---|
| 10 | 4.0 ± 0.0 | 8.7 ± 0.6 | 0.833 → 0.426 (0.250) | 0.696 | 1 |
| 20 | 6.3 ± 1.2 | **9.0 ± 0.0** | 0.880 → 0.439 (0.202) | 0.696 | 1 |
| 32 | 6.0 ± 1.0 | **9.0 ± 0.0** | 0.879 → 0.434 (0.163) | 0.690 | 1 |

Everything survives and sharpens: k=1 truncation stays optimal even with ~16
positive eigenvalues available at d=32; the top eigenvalue's share of positive
mass is strikingly constant (~0.69); and since the random baseline falls with
dimension while centered similarity stays ~0.43, the learned structure grows
*more* distinguishable from chance as the latent widens. — **HEADLINE #3
(robustness half of rank duality)**

### Exp 17 — The gradient (Jacobian) baseline: what does bilinearity buy? ★
**What:** the fair skeptic's question — ordinary autograd can search for
latent directions on *any* decoder; is the bilinear constraint needed at all?
**How:** identical protocol (centered targets, per-model norm budget, causal
test) with: gradient ascent on a VanillaVAE (Adam, lr 0.05, 300 steps, 8
seeded restarts; main + 3 fresh seeds), the one-backward-pass Jacobian
direction Jᵀp\* at z=0, gradient ascent on the DecBilinearVAE itself
(ablation), and random-init controls.
**Achieves:** gradient methods **match and slightly exceed** weight-based
capability: weight-eig 9/10 (MSE 0.040); bilinear grad-ascent 10/10 (0.039);
vanilla grad-ascent 10/10 on all four models (0.036); vanilla one-step
Jacobian 10/10 on all four (0.032, the best). Random controls 1/10 (0.179).
Two structural brackets sharpen the interpretation: gradient ascent on the
bilinear model exploits the same structure plus the sigmoid the quadratic form
ignores; and the one-step Jacobian **fails structurally on the bilinear
decoder — Jᵀp\* ≡ 0 exactly at z=0** (bias-free quadratic ⇒ no linear term):
the weight-based eigendecomposition is precisely the second-order object that
replaces a first-order method where it cannot operate. **The honest framing of
the whole decoder study: the bilinear win is the *form* of the answer —
closed-form, gradient-free, globally valid, the full generative + suppressor
spectrum in one decomposition — not raw synthesis capability.** — **HEADLINE
(reframes #1/#2's value proposition)**

### Exp 18 — KMNIST battery
**What:** the full raw-vs-centered story on a third dataset.
**Achieves:** targets 0.821→0.237; eigenvectors 0.833→**0.460** (random-init
raw 0.921 — above trained, further confirming the artifact reading; centered
random 0.268); causal **1/10→7/10** (random 1/10); MSE 0.141→0.054 (recon
0.021; 7.0×→2.5×). The three centered misses (ma→ki, re→ki, wo→su) all fall in
a persistent ki/su/ma/re/wo cluster that stays 0.7–0.98 similar after
centering: **contrastive targets separate classes only up to genuinely shared
visual components** — an informative, honest limit of the method. Centered
cross-dataset picture: MNIST 0.429, FMNIST 0.402, KMNIST 0.460, all vs random
≈ 0.27–0.30. — **SUPPORTING (+ the limitation sentence for the main text)**

**Part II verdict:** the decoder chapter is the paper's core: pitfall →
mechanism → provably-canonical fix → 9/10 generation → robustness in latent
dimension and datasets → honest positioning against gradient baselines.

---

## PART III — The full bilinear VAE (`bilinear-full`, 21 experiments)

**The questions:** what happens when *both* sides are bilinear — does the pure
model even train, do the two analyses reinforce each other, and do encoder and
decoder eigenstructures align?

### Exp 11 — The v1 failure: an exact symmetry pathology ★
**What:** the pure bilinear decoder (no biases) satisfies decode(z) ≡
decode(−z) *identically* — the latent space collapses to a ℤ₂ quotient and
sign information is unusable.
**How:** algebraic argument (every output term is a product of two z-linear
factors) plus numerical verification: max |f(z) − f(−z)| = **0.00e+00**; plus
a latent-usage diagnosis and training-curve comparison.
**Achieves:** v1 recon MSE 0.0638; adding a **linear skip connection** breaks
the symmetry: v2 = 0.0281. A clean proposition + diagnosis + minimal fix —
the theory nugget of the paper. — **HEADLINE #4**

### Exps 01–02 — Eigenstructure of the full model
**What:** does each side keep its character when both are bilinear?
**Achieves:** the decoder's raw-target universality persists (0.808); the
full-model encoder is somewhat *less* class-discriminative than the
encoder-only model (cross-class 0.364 vs 0.35-encoder-only baseline
comparison; see exp 15 for seeds: 0.342 ± 0.024). — **SUPPORTING**

### Exp 03 — Pixel-space alignment (first attempt)
**SUPERSEDED** by exp 07: it compared a signed pixel filter with a [0,1]
image — a flawed measurement kept only for the record.

### Exp 04 / Exp 08 — Causal generation and its decomposition
**What:** raw-target causal test on the full model, then component swapping to
localize the improvement over the decoder-only 3/10.
**Achieves:** single main checkpoint: dec-only 3/10 → decoder + full-model
encoder 5/10 → full model 6/10 (+2 from the bilinear encoder, +1 joint).
**Seed honesty (exp 15):** the 6/10 was the best seed; the 5-seed mean is
**4.2 ± 0.4** — the number the paper must quote. With centered targets all of
this is superseded by exp 18's 9.5 ± 0.5. — **SUPPORTING**

### Exp 05 — Four-way model comparison
**What:** VanillaVAE vs BilinearVAE(enc) vs DecBilinearVAE vs FullBilinearVAE
on reconstruction/ELBO per class; pipeline-level latent metrics.
**Achieves (pipeline table, MNIST):** linear-probe accuracy Vanilla 0.8903,
Bilinear-enc 0.8456, **DecBilinear 0.9075 (best)**, Full-v2 0.8874;
silhouettes 0.117 / 0.091 / **0.147** / 0.129; recon MSE 0.0173 / 0.0347 /
0.0187 / 0.0282. **DecBilinearVAE beats the VanillaVAE on latent structure**
despite its interpretability pathology — the practicality note. —
**SUPPORTING**

### Exp 06 — Consistency across seeds
**Achieves:** decoder synthesis 0.941, encoder eigenvectors 0.326, alignment
per-class boxplots straddle zero (foreshadowing exp 07). — **APPENDIX**

### Exp 07 — Latent-space encoder↔decoder alignment (the corrected null)
**What:** the hypothesis behind "v3": encoder and decoder eigenvectors should
align in latent space.
**Achieves:** mean cosine **0.007** against an *empirical* |random| baseline
of **0.220** — at/below chance; per-class values swing −0.74 to +0.52. —
**HEADLINE #5 (with exps 15/19/21)**

### Exp 09 — Encode–decode loop dynamics
**What:** iterate x ← decode(encode(x)) from images, noise, and synthesized
eigenvector images; observe the attractors. — **APPENDIX**

### Exp 10 — Full eigenspectra (Q_dec complete; Q_in top-20). — **APPENDIX**

### Exp 12 — Disentanglement (MIG proxy)
**Achieves:** honest negative: VanillaVAE 0.126 > Full 0.097 > DecBilinear
0.091 > Bilinear-enc 0.061 — every bilinear variant is *less* disentangled
than the baseline. One caveat sentence in the paper; kills any
"disentanglement pressure" speculation. — **HONEST CAVEAT**

### Exp 13 — Training dynamics
**What:** re-analyzes 20 per-epoch checkpoints.
**Achieves:** under joint training the raw-target universality *declines*
(0.875 → 0.816) while the encoder grows steadily more discriminative
(cross-class 0.398 → 0.220) as ELBO falls — training works against the
architectural seed, not for it (complementing decoder exp 12). —
**SUPPORTING**

### Exp 14 — Alignment interpolation (follow-up to the null; figure rebuilt
with proper gridspec). — **APPENDIX**

### Exp 15 — Five-seed significance
**What:** bootstrap CIs for the four headline quantities.
**Achieves:** universality 0.826 ± 0.011; encoder cross-class 0.342 ± 0.024;
causal (raw) **4.2 ± 0.4**; alignment 0.030 ± 0.141 (CI spans zero). The
rigor layer under Part III. — **SUPPORTING**

### Exp 16 — Truncated reconstruction, raw targets
**Achieves:** no low-rank plateau: k=1..10 MSE only 0.062→0.054 while the
full decode achieves 0.0071 (~8× better). Raw targets hide the structure that
exp 20 reveals. — **SUPPORTING (as the "before" of exp 20)**

### Exp 17 — Fashion-MNIST summary
**Achieves:** universality 0.869; causal 2/10 (raw); seed consistency decoder
0.962 / encoder 0.333; and one flagged anomaly — single-checkpoint alignment
0.424, resolved by exp 21. FMNIST seed4 initially diverged (BCE assert);
retrained cleanly after adding output clamping + gradient clipping (best test
254.28). — **SUPPORTING**

### Exp 18 — Contrastive targets on the full model ★
**Achieves:** across all 6 MNIST checkpoints: causal **4.7 ± 0.7 → 9.5 ± 0.5**;
decoder cross-class 0.83 → 0.40. The centering fix is architecture-general.
— **HEADLINE #2 (seed-level extension)**

### Exp 19 — The alignment question, settled ★
**What:** the definitive test of the alignment hypothesis, with a built-in
manipulation check.
**How + discovery:** inspection of the shipped v3 revealed its alignment loss
is **detached from the computation graph** (both Jacobians computed under
detach/no_grad ⇒ zero gradient) — the shipped "v3" never actually trained
with an alignment loss (its own objective: 0.375 vs v2's 0.376). The loss was
repaired (differentiable finite differences) and 6 fresh seeds trained at
λ = 1 and λ = 1000.
**Achieves:** the repaired λ=1000 loss demonstrably optimizes its own Jacobian
objective (0.376 → **0.465**, +24%) — and the corrected weight-basis alignment
metric *still* stays inside the noise band for every group: v2 0.026 ± 0.129,
v3-shipped −0.043, repaired λ=1 −0.142 ± 0.147, repaired λ=1000 −0.061 ± 0.116,
vs |random| 0.258 ± 0.056. A textbook null with a positive control — plus a
cautionary tale about silent gradient detachment worth a sentence of its own.
— **HEADLINE #5**

### Exp 20 — Truncation with centered targets: the structure is rank-1 ★
**What:** does centering reveal the low-rank structure that raw targets (exp
16) hid?
**Achieves:** sharper than "low-rank" — **rank-1 per class**: the single top
centered eigenvector already attains MSE 0.0276 ± 0.0050 (raw k=1:
0.0629 ± 0.0070; full decode 0.0074) and causal 9.5 ± 0.5 (exactly reproducing
exp 18); curves are flat beyond k=1, and under the |λ| ordering extra
eigenvectors actively hurt (causal 4→2). Combined with exp 16 (encoder needs
rank ~10–15), this is the paper's **rank duality**: high-rank recognition,
rank-1 generation. — **HEADLINE #3**

### Exp 21 — The FMNIST alignment anomaly, resolved
**What:** exp 17's single-checkpoint alignment of 0.424 (above random) tested
across all 5 FMNIST seeds with an FMNIST-specific random baseline.
**Achieves:** a fluke: per-checkpoint values [+0.424, +0.222, +0.124, −0.154,
−0.245, −0.100], mean **0.045 ± 0.233** vs |random| 0.256 ± 0.058; per-class
cosines flip sign across seeds for the same class (Shirt: +0.72 → −0.65).
FMNIST is the same null as MNIST, with roughly double the variance. —
**SUPPORTING (closes #5)**

**Part III verdict:** one theory gem (exp 11), one exemplary null (exp
07/15/19/21), the centering fix confirmed at seed level (exp 18/20), and the
honest caveats (exp 12, exp 15's 4.2 ± 0.4) all on record.

---

## PART IV — Pipeline-level results, precursors, and discarded material

These are not standalone experiments but belong in the record (and some in the
paper's appendix or limitations).

- **β-sweep** (encoder pipeline): β = 1→8 lowers effective rank 79.3→64.3,
  raises PVE@10 0.316→0.396, costs linear-probe 0.870→0.847 — stronger priors
  compress the code as expected. — APPENDIX at most.
- **CIFAR-10 attempt** (conv bilinear VAE, 100 epochs): linear probe 0.362,
  silhouette −0.052 — the method as instantiated does not scale to natural
  images; reported as a limitation, not silently dropped.
- **Additive adversarial precursor** (encoder): adding ε·v₁ directly performs
  *below chance* (0.029 vs 0.10) — the failure that motivated pseudoinverse
  keys in exp 10. One sentence of honest history.
- **Original GANSpace efficiency curve**: superseded by decoder exp 15 after
  two flaws were found (identical samples per repeat → std = 0; saturated
  metric with no chance floor).
- **Integrated-gradients prototype** (Jan 2026, pre-project): compared IG
  attributions to eigenvectors (mean |cos| ≈ 0.25); abandoned — decoder exp 17
  now supersedes it as the proper gradient-vs-weight comparison.
- **Old report claims corrected during the audit:** the decoder README's
  "random = 0.864 ⇒ purely architectural" (actual 0.646 ⇒ seeded + amplified);
  the mislabeled encoder panel in exp 09; the "2/10 random" causal baselines
  (actual 4.4 ± 1.6 / 3.5 ± 1.9 / 3.5 ± 1.6 over 10 seeded inits); the v3
  alignment claims (exp 19); the rank-2 subspace claim (exps 16/20).

---

## PART V — The narrative arc and suggested tiering

The experiments, read in order, tell one story:

1. **Recognition works and is causal** (encoder: exps 02/03/05/08/09/10/12) —
   weight-based features in an unsupervised encoder are real, higher-rank than
   classifiers, and steerable.
2. **Generation naively fails — for a provable reason** (decoder: 01/03 →
   12/13/14) — Q's linearity in the target makes overlapping targets the
   culprit; centering is the canonical fix; 9/10 generation follows.
3. **The corrected structure is rank-1, robustly** (16/20 + encoder 02) —
   rank duality across sides, surviving latent dimensions 10→32 and three
   datasets (13/18, encoder 09/12).
4. **The theory nugget** (full exp 11) — the exact even-symmetry pathology and
   its one-line fix.
5. **The honest boundaries** (17, 15, 07/19/21, 12) — gradient baselines match
   capability (bilinearity buys form, not power); PCA finds different
   directions; encoder–decoder alignment is a controlled null; bilinear VAEs
   are not more disentangled.

**Main-text figure/table shortlist (≈7):** exp13 synthesis grid (the money
shot) · exp14 method-comparison bars · exp16 sweep table · exp05+exp10
encoder causal/steering panel · exp11 symmetry figure · exp19 two-panel null ·
exp17 method table. **Everything else:** appendix or repository.
