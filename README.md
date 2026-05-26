# AEVB Reproduction — Kingma & Welling (2013)

Reproduction of *Auto-Encoding Variational Bayes* (VAE) for the ASI course at EURECOM.

## Datasets
- **MNIST** (binarized, Bernoulli decoder) — 784 dims
- **Frey Face** (Gaussian decoder) — 560 dims

## Reproduced figures
- **Fig 2** — ELBO curves (train/test) for Nz ∈ {2, 5, 10, 20}
- **Fig 4** — 2D latent manifold
- **Fig 5** — Random samples from generative models

## Structure
```
vae_kaggle.ipynb       # Training on Kaggle (GPU T4)
vae_local.ipynb        # Local training + plots
Reproduction_of_VAE.pdf # Final report PDF
results/
  checkpoints/         # Saved model weights (.pt)
```

## Key details
- Optimizer: Adagrad, lr=0.01, weight_decay=1e-4
- Architecture: 1 hidden layer (500 units, tanh), latent dims Nz ∈ {2, 5, 10, 20}
- Frey Face: 1500 train / 465 test
- MNIST: 60k train / 10k test, batch=500, preloaded to GPU
