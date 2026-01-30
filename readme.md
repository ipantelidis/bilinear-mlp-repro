## Reproduction: Bilinear MLPs enable weight-based mechanistic interpretability

This repository reproduces the main interpretability claims from the paper "Bilinear MLPs enable weight-based mechanistic interpretability". All reproduction experiments follow the training setup, datasets, and analysis procedures described by the authors and focus on vision-based models.

Each claim below corresponds to a specific set of experiments and figures in the original paper, with direct pointers to the relevant code.

---

### C1 — Interpretable Eigenfeatures

**Claim**  
Bilinear MLPs trained on image classification tasks learn dominant eigenvectors of their bilinear interaction matrices that correspond to human-recognizable visual structure.

**What we reproduce**
- Bilinear MLPs trained on **MNIST** and **Fashion-MNIST**
- Extraction of class-specific bilinear interaction matrices
- Eigendecomposition of interaction matrices
- Projection of leading eigenvectors back into input (pixel) space

**Observed results**
- Leading eigenfeatures align with meaningful image structure:
  - Digit strokes and components for MNIST
  - Garment contours and edges for Fashion-MNIST
- Positive and negative regions correspond to opposing contributions to the class score
- Qualitative patterns closely match those reported in the original work

**Corresponds to**  
Figure 2 of the original paper

**Code location**  
`reproduction/image/fig_02/fig_02.py`

---

### C2 — Low-Rank Dominance of Bilinear Interactions

**Claim**  
For each output class, the bilinear interaction is dominated by a small number of eigenvalues with large magnitude, indicating an effectively low-rank computation.

**What we reproduce**
- Computation of eigenvalue spectra for class-specific interaction matrices
- Sorting eigenvalues by absolute magnitude
- Visualization of spectra on a logarithmic scale

**Observed results**
- A sharp drop-off after the leading eigenvalues
- A long tail of near-zero eigenvalues
- Consistent low-rank structure across:
  - Multiple classes
  - MNIST and Fashion-MNIST

**Corresponds to**  
Figure 3 of the original paper

**Code location**  
`reproduction/image/fig_03/fig_03.py`

---

### C3 — Robustness Under Input Noise Regularization

**Claim**  
Training bilinear MLPs with additive Gaussian input noise preserves interpretable interaction structure even when the input distribution is substantially corrupted.

**What we reproduce**
- Training bilinear MLPs with varying levels of Gaussian input noise
- Extraction of dominant eigenvectors at each noise level
- Visualization of how eigenfeatures evolve as noise increases

**Observed results**
- Dominant eigenfeatures remain visually structured and class-aligned
- Increasing noise leads to smoother, more spatially coherent patterns
- Eigenfeatures do not collapse into noise even at high corruption levels
- Test accuracy degrades gracefully, consistent with the original paper

**Corresponds to**  
Figure 4 of the original paper

**Code location**  
`reproduction/image/fig_04/fig_04.py`

---

### C4 — Eigenfeature-Based Adversarial Masks

**Claim**  
Masks constructed from dominant eigenfeatures selectively suppress decision-relevant regions of the input and are significantly more effective than random masks.

**What we reproduce**
- Construction of adversarial masks directly from leading eigenvectors
- Application of masks to test images from the corresponding class
- Comparison with random masks of equal sparsity
- Measurement of classification accuracy and misclassification rates

**Observed results**
- Eigenfeature-based masks cause substantially higher misclassification
- Random masks have minimal effect on model performance
- Masks are derived purely from learned weights:
  - No gradients
  - No input-specific optimization
- Even very sparse masks (active on a small fraction of pixels) are effective

**Corresponds to**  
Figure 7 of the original paper

**Code location**  
`reproduction/image/fig_07/fig_07.py`

---

## Extension: Bilinear MLPs in Variational Autoencoders (VAE)

Beyond reproducing the original claims, this repository introduces a **novel extension** by integrating bilinear MLPs into the **encoder of a variational autoencoder (VAE)**.

This extension investigates whether weight-based interpretability enabled by bilinear interactions can be preserved in a **generative, unsupervised setting**.

---

### Motivation

- The original paper focuses exclusively on supervised classification
- We ask whether bilinear interaction structure:
  - Remains low-rank
  - Produces interpretable eigenfeatures
  - Organizes latent space in a semantically meaningful way
- Crucially, this is studied **without using class labels during training**

---

### What We Show

- Replacing a standard MLP encoder with a bilinear MLP encoder:
  - Preserves reconstruction quality
  - Does not destabilize VAE training
- Weight-based eigenanalysis of the encoder reveals:
  - Structured input-space eigenfeatures
  - Low-rank interaction structure in latent mappings
- Class-conditional latent analyses reveal:
  - Digit-aligned eigenfeatures
  - Semantically meaningful organization of latent space
- Eigenfeature-guided interpolations enable:
  - Smooth transitions between digit classes
  - Interpretable control directions in latent space

---

### Analyses Included

- Reconstruction quality of bilinear-encoder VAE
- Eigenvalue spectra of latent interaction matrices
- Eigenfeature visualizations from:
  - Individual latent dimensions
  - Class-mean latent encodings
- Eigenfeature-based interpolation between digit classes
- Comparison of interaction structure in:
  - Latent means (μ)
  - Latent log-variances (log σ²)

---

### Code location

`extension/bilinear_vae.ipynb`

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

After installation, you can choose which individual figure, experiment, or appendix result to reproduce by running the corresponding scripts. Each experiment is self-contained, allowing selective reproduction without executing the full pipeline.



