"""
run_all.py — Run all 10 encoder analysis experiments in sequence.

Usage (from the extension/ directory):
    python run_all.py            # run all experiments
    python run_all.py 1 3 5      # run only experiments 1, 3, 5

All outputs go to figures/mnist/ and figures/fashion_mnist/.
"""

import sys
import importlib
import traceback
from pathlib import Path


EXPERIMENTS = {
    1:  "experiments.exp01_latent_dictionary",
    2:  "experiments.exp02_truncation",
    3:  "experiments.exp03_cross_class",
    4:  "experiments.exp04_semantic_diff",
    5:  "experiments.exp05_max_activating",
    6:  "experiments.exp06_negative_eigenvectors",
    7:  "experiments.exp07_saliency",
    8:  "experiments.exp08_eigvec_consistency",
    9:  "experiments.exp09_fmnist",
    10: "experiments.exp10_adversarial_mask",
}

DESCRIPTIONS = {
    1:  "Latent dictionary           (μ* = e_k, weights only)",
    2:  "Truncation                  (rank vs. Pearson r)",
    3:  "Cross-class similarity      (10×10 cosine matrix)",
    4:  "Semantic difference         (mean_A − mean_B directions)",
    5:  "Maximally activating test   (9/10 trained vs 1/10 random)",
    6:  "Negative eigenvectors       (cross-suppression map)",
    7:  "Saliency maps               (|2Qx|, no backprop)",
    8:  "Eigenvector consistency     (analog of Pearce et al. Fig 5A)",
    9:  "Fashion-MNIST extension     (latent dict + causal + cross-class)",
    10: "Adversarial mask            (pseudoinverse steering)",
}


def run_experiment(n: int) -> bool:
    module_path = EXPERIMENTS[n]
    print(f"\n{'='*60}")
    print(f"  Exp {n:02d}: {DESCRIPTIONS[n]}")
    print(f"{'='*60}")
    try:
        mod = importlib.import_module(module_path)
        # Each experiment script runs its logic in __main__ — we call it directly
        # by re-running the module as __main__ via runpy
        import runpy
        runpy.run_module(module_path, run_name="__main__", alter_sys=True)
        return True
    except Exception:
        print(f"\n  [FAILED] Exp {n:02d}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Which experiments to run
    if len(sys.argv) > 1:
        to_run = [int(a) for a in sys.argv[1:] if a.isdigit()]
    else:
        to_run = list(EXPERIMENTS.keys())

    # Must run from extension/ directory so relative paths resolve
    script_dir = Path(__file__).parent
    import os; os.chdir(script_dir)

    results = {}
    for n in to_run:
        if n not in EXPERIMENTS:
            print(f"  Unknown experiment {n}, skipping.")
            continue
        results[n] = run_experiment(n)

    # Summary
    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    passed = sum(results.values())
    total  = len(results)
    for n, ok in sorted(results.items()):
        status = "✓" if ok else "✗ FAILED"
        print(f"  Exp {n:02d}: {status}  — {DESCRIPTIONS[n]}")
    print(f"\n  {passed}/{total} passed")
