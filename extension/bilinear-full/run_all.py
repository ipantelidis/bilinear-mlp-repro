"""
run_all.py — Run all 21 bilinear-full experiments.

Usage (from the bilinear-full/ directory):
    python run_all.py            # run all
    python run_all.py 1 3        # run specific experiments
"""

import sys, traceback, runpy, os
from pathlib import Path

EXPERIMENTS = {
    1:  "experiments.exp01_encoder_eigenvecs",
    2:  "experiments.exp02_decoder_eigenvecs",
    3:  "experiments.exp03_alignment",
    4:  "experiments.exp04_causal_gen",
    5:  "experiments.exp05_comparison",
    6:  "experiments.exp06_consistency",
    7:  "experiments.exp07_latent_alignment",
    8:  "experiments.exp08_causal_decomposition",
    9:  "experiments.exp09_loop",
    10: "experiments.exp10_eigenspectrum",
    11: "experiments.exp11_v1_failure",
    12: "experiments.exp12_disentanglement",
    13: "experiments.exp13_training_dynamics",
    14: "experiments.exp14_alignment_interp",
    15: "experiments.exp15_significance",
    16: "experiments.exp16_truncated_recon",
    17: "experiments.exp17_fmnist_summary",
    18: "experiments.exp18_contrastive_targets",
    19: "experiments.exp19_alignment_settled",
    20: "experiments.exp20_truncation_centered",
    21: "experiments.exp21_fmnist_alignment",
}

DESCRIPTIONS = {
    1:  "Encoder eigenvectors      (class-discriminativeness with bilinear decoder)",
    2:  "Decoder eigenvectors      (near-universal direction with bilinear encoder)",
    3:  "Encoder–decoder alignment (pixel-space, flawed — see exp07)",
    4:  "Causal generation test    (does bilinear encoder improve accuracy?)",
    5:  "Four-way model comparison (Vanilla / Enc / Dec / Full)",
    6:  "Cross-seed consistency    (decoder, encoder, and alignment stability)",
    7:  "Latent-space alignment    (corrected enc↔dec measure in ℝ¹⁰)",
    8:  "Causal accuracy decomp.   (isolate encoder vs joint-training contribution)",
    9:  "Encode→decode loop        (fixed-point / convergence analysis)",
    10: "Q_dec eigenspectrum       (full spectrum per class, rank structure)",
    11: "V1 failure analysis       (even-symmetry f(z)=f(-z) collapse)",
    12: "Disentanglement / MIG     (latent-class correlation, 4-way comparison)",
    13: "Training dynamics         (spectral properties vs epoch, needs train_epochs.py)",
    14: "Alignment interpretation  (decode z_dec → re-encode → class check)",
    15: "Statistical significance  (bootstrap 95% CI over 5 seeds)",
    16: "Truncated reconstruction  (rank-k subspace test)",
    17: "FashionMNIST summary      (key metrics, needs train_fmnist.py)",
    18: "Contrastive targets       (centered p*: synthesis, cross-class, causal, per seed)",
    19: "Alignment settled         (corrected metric: v2 seeds, v3, repaired align loss)",
    20: "Truncation, centered      (rank-k test with centered targets, exp16 revisited)",
    21: "FMNIST alignment anomaly  (exp17's 0.424 across all seeds vs |random| baseline)",
}


def run_experiment(n):
    print(f"\n{'='*60}\n  Exp {n:02d}: {DESCRIPTIONS[n]}\n{'='*60}")
    try:
        runpy.run_module(EXPERIMENTS[n], run_name="__main__", alter_sys=True)
        return True
    except Exception:
        print(f"\n  [FAILED] Exp {n:02d}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    to_run = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(EXPERIMENTS)
    results = {n: run_experiment(n) for n in to_run if n in EXPERIMENTS}

    print(f"\n{'='*60}\n  Summary\n{'='*60}")
    for n, ok in sorted(results.items()):
        print(f"  Exp {n:02d}: {'✓' if ok else '✗ FAILED'}  — {DESCRIPTIONS[n]}")
    print(f"\n  {sum(results.values())}/{len(results)} passed")
