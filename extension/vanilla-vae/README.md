# Vanilla VAE baseline

The ordinary MLP VAE used as the baseline throughout the extension:

```
Encoder: x(784) → Linear(256) → ReLU → Linear(512) → ReLU → μ(10), logσ²(10)
Decoder: z(10)  → Linear(256) → ReLU → Linear(784) → Sigmoid
```

It is consumed by:

- `bilinear-decoder` exp 12 (trained-decoder control) and exp 17 (gradient /
  Jacobian baselines),
- `bilinear-full` exp 05 (4-way model comparison) and exp 12 (disentanglement).

Those experiments define their own minimal loader for the checkpoint format,
so this directory only provides the weights and the training script.

## Checkpoints

`checkpoints/mnist/model.pt` (main, seed 0) and
`checkpoints/mnist/seeds/seed{1,2,3}.pt`, each with its training history
alongside. Format: `{"model_state": state_dict, "epoch": int, "test_loss": float}`.

## Training

Same recipe as the bilinear models (AdamW lr 1e-3, weight decay 0.01, cosine
annealing, batch 128, 30 epochs, input noise 0.3, BCE reconstruction, β = 1,
best-validation checkpointing):

```bash
cd extension/vanilla-vae
python train.py            # main checkpoint
python train.py --seed 1   # seed runs
```

Existing checkpoint files are never overwritten.
