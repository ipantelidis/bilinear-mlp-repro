"""
run_all.py — Run all 11 decoder analysis experiments in sequence.

Usage (from the bilinear-decoder/ directory):
    python run_all.py            # run all experiments
    python run_all.py 1 3 9      # run only experiments 1, 3, 9

All outputs go to figures/mnist/ and figures/fashion_mnist/.
"""

import sys
import traceback
import runpy
from pathlib import Path


EXPERIMENTS = {
    1:  "experiments.exp01_synthesis",
    2:  "experiments.exp02_pixel_fields",
    3:  "experiments.exp03_causal_gen",
    4:  "experiments.exp04_eigenspectrum",
    5:  "experiments.exp05_consistency",
    6:  "experiments.exp06_generative_basis",
    7:  "experiments.exp07_suppressor_map",
    8:  "experiments.exp08_mass_ratio",
    9:  "experiments.exp09_crossclass",
    10: "experiments.exp10_synthesis_quality",
    11: "experiments.exp11_fmnist",
    12: "experiments.exp12_decoder_control",
}

DESCRIPTIONS = {
    1:  "Analytical synthesis        (p*=mean_c, decode top +eigvec)",
    2:  "Pixel generative fields     (p*=e_i, 10×10 spatial grid)",
    3:  "Causal generation test      (generate→encode→classify loop)",
    4:  "Decoder eigenspectrum       (rank-2 generative subspace)",
    5:  "Seed consistency            (pixel-space cosine across 5 seeds)",
    6:  "Complete generative basis   (all positive eigvecs per class)",
    7:  "Suppressor map              (decoded negative eigvecs)",
    8:  "Mass ratio                  (pos/neg eigenvalue mass per class)",
    9:  "Cross-class comparison      (encoder vs decoder similarity)",
    10: "Synthesis quality           (weight-based vs VAE reconstruction)",
    11: "Fashion-MNIST extension     (synthesis + causal + cross-class)",
    12: "Decoder control             (trained vs random vs VanillaVAE)",
}


def run_experiment(n: int) -> bool:
    module_path = EXPERIMENTS[n]
    print(f"\n{'='*60}")
    print(f"  Exp {n:02d}: {DESCRIPTIONS[n]}")
    print(f"{'='*60}")
    try:
        runpy.run_module(module_path, run_name="__main__", alter_sys=True)
        return True
    except Exception:
        print(f"\n  [FAILED] Exp {n:02d}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        to_run = [int(a) for a in sys.argv[1:] if a.isdigit()]
    else:
        to_run = list(EXPERIMENTS.keys())

    import os
    os.chdir(Path(__file__).parent)

    results = {}
    for n in to_run:
        if n not in EXPERIMENTS:
            print(f"  Unknown experiment {n}, skipping.")
            continue
        results[n] = run_experiment(n)

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    passed = sum(results.values())
    total  = len(results)
    for n, ok in sorted(results.items()):
        status = "✓" if ok else "✗ FAILED"
        print(f"  Exp {n:02d}: {status}  — {DESCRIPTIONS[n]}")
    print(f"\n  {passed}/{total} passed")
