#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh — Run every experiment for the bilinear VAE extension.
#
# Launches 5 jobs in parallel across 5 GPUs then waits for all of them.
# All output is saved to extension/checkpoints/logs/.
#
# Usage:
#   bash extension/run_all.sh            # full run (recommended)
#   EPOCHS=3 bash extension/run_all.sh   # quick smoke test (3 epochs each)
#
# GPU layout:
#   GPU 0 — MNIST:          vanilla_vae, bilinear_vae, beta_vae, beta_bilinear_vae
#   GPU 1 — Fashion-MNIST:  same four models
#   GPU 2 — Consistency:    bilinear_vae × 5 seeds on MNIST
#   GPU 3 — Beta sweep:     beta_bilinear_vae, β ∈ {1, 2, 4, 8} on MNIST
#   GPU 4 — CIFAR-10:       conv_bilinear_vae (100 epochs, MSE loss)
# ─────────────────────────────────────────────────────────────────────────────

set -e

PYTHON=~/.venv/bin/python
SCRIPT=extension/run.py
LOG_DIR=extension/checkpoints/logs
EPOCHS=${EPOCHS:-30}          # override with EPOCHS=3 for a quick test
CIFAR_EPOCHS=${CIFAR_EPOCHS:-100}

mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  Bilinear VAE — full experiment suite"
echo "  Flat models  : ${EPOCHS} epochs each"
echo "  CIFAR-10     : ${CIFAR_EPOCHS} epochs"
echo "  Logs         : ${LOG_DIR}/"
echo "================================================================"
echo ""

# ── GPU 0: MNIST flat models ─────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON $SCRIPT \
    --model all --dataset mnist \
    --mode all --epochs $EPOCHS \
    > "$LOG_DIR/mnist.log" 2>&1 &
PID_MNIST=$!
echo "  GPU 0  MNIST flat models        PID $PID_MNIST"

# ── GPU 1: Fashion-MNIST flat models ─────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON $SCRIPT \
    --model all --dataset fashion_mnist \
    --mode all --epochs $EPOCHS \
    > "$LOG_DIR/fashion_mnist.log" 2>&1 &
PID_FMNIST=$!
echo "  GPU 1  Fashion-MNIST flat models PID $PID_FMNIST"

# ── GPU 2: Eigenfeature consistency (5 seeds) ────────────────────────────────
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON $SCRIPT \
    --mode consistency --dataset mnist \
    --n_seeds 5 --epochs $EPOCHS \
    > "$LOG_DIR/consistency.log" 2>&1 &
PID_CONS=$!
echo "  GPU 2  Consistency (5 seeds)     PID $PID_CONS"

# ── GPU 3: Beta sweep ────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON $SCRIPT \
    --mode beta_sweep --dataset mnist \
    --betas 1 2 4 8 --epochs $EPOCHS \
    > "$LOG_DIR/beta_sweep.log" 2>&1 &
PID_SWEEP=$!
echo "  GPU 3  Beta sweep (β 1,2,4,8)   PID $PID_SWEEP"

# ── GPU 4: CIFAR-10 ──────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON $SCRIPT \
    --model conv_bilinear_vae --dataset cifar10 \
    --mode all --epochs $CIFAR_EPOCHS \
    > "$LOG_DIR/cifar10.log" 2>&1 &
PID_CIFAR=$!
echo "  GPU 4  CIFAR-10 (100 epochs)    PID $PID_CIFAR"

echo ""
echo "All jobs launched. Waiting for completion ..."
echo "(Monitor with: tail -f extension/checkpoints/logs/<name>.log)"
echo ""

# ── Wait for all jobs ─────────────────────────────────────────────────────────
wait $PID_MNIST  && echo "  [done] MNIST flat models"        || echo "  [FAILED] MNIST flat models"
wait $PID_FMNIST && echo "  [done] Fashion-MNIST flat models" || echo "  [FAILED] Fashion-MNIST"
wait $PID_CONS   && echo "  [done] Consistency"               || echo "  [FAILED] Consistency"
wait $PID_SWEEP  && echo "  [done] Beta sweep"                || echo "  [FAILED] Beta sweep"
wait $PID_CIFAR  && echo "  [done] CIFAR-10"                  || echo "  [FAILED] CIFAR-10"

echo ""
echo "================================================================"
echo "  All experiments complete."
echo "  Checkpoints : extension/checkpoints/{dataset}/{model}/"
echo "  Figures     : extension/figures/{model}/{dataset}/"
echo "================================================================"
