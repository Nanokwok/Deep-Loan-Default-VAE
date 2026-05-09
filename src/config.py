"""
Central configuration for the Credit Card Fraud VAE project.
Edit values here rather than hunting through individual scripts.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR    = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROC_DIR   = os.path.join(ROOT_DIR, "data", "processed")
REPORTS_DIR     = os.path.join(ROOT_DIR, "reports")
FIGURES_DIR     = os.path.join(REPORTS_DIR, "figures")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
RAW_CSV         = os.path.join(DATA_RAW_DIR, "creditcard.csv")

# ── Target column & labels ─────────────────────────────────────────────────────
TARGET_COL      = "Class"   # 0 = Normal, 1 = Fraud

# ── Model Architecture ────────────────────────────────────
INPUT_DIM       = 30
LATENT_DIM      = 4
ENCODER_DIMS    = [32, 16]
DECODER_DIMS    = [16, 32]

# ── VAE Hyperparameters ──────────────────────────────────────────
BETA            = 0.005
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 512
NUM_EPOCHS      = 200
RANDOM_SEED     = 42

# ── Early stopping & LR scheduler ────────────────────────────────
# PATIENCE     : stop if Val AUPRC does not improve for this many epochs.
#                Rule of thumb: ~10% of NUM_EPOCHS.  200 epochs → 20.
# LR_PATIENCE  : halve LR after this many non-improving epochs.
#                Should fire before early stopping: ~PATIENCE // 2.
PATIENCE        = 20
LR_PATIENCE     = 7

# ── KL Annealing ─────────────────────────────────────────────────
# β ramps linearly from 0 → BETA over the first KL_ANNEAL_EPOCHS epochs.
# This lets the VAE focus on reconstruction quality first, then gradually
# tighten the latent space. 50 epochs gives a slow, stable warm-up for
# NUM_EPOCHS=200.
KL_ANNEAL_EPOCHS: int = 50

# ── Denoising ─────────────────────────────────────────────────────
# Feature-specific Gaussian noise added to input during training only.
# The per-feature sigma is NOISE_STD / (w_i / mean(w)), so high-weight
# (discriminative) features receive LESS noise — their fraud signal is
# preserved — while low-weight features receive MORE noise, forcing the
# model to learn their underlying structure rather than memorise them.
# Base std of 0.02 ≈ 2% of a unit-std feature (down from 0.05).
NOISE_STD: float = 0.02

# ── Activation function ───────────────────────────────────────────
# LeakyReLU prevents dying neurons on the many negative values in V1-V28.
# negative_slope=0.01 lets a small gradient flow for x < 0.
LEAKY_RELU_SLOPE: float = 0.01

# ── Feature-wise reconstruction weights ──────────────────────────
# Derived from EDA: |mean_fraud − mean_normal| on the val set.
# Higher weight → loss spikes harder when this feature deviates.
# Features not listed default to 1.0.
# Scale guide:  |Δμ| > 6 → 3.0 | |Δμ| 4-6 → 2.0 | |Δμ| 2-4 → 1.5
FEATURE_WEIGHTS: dict[str, float] = {
    "V3":  3.0,   # |Δμ| = 7.05  — strongest fraud signal
    "V14": 3.0,   # |Δμ| = 6.98
    "V17": 3.0,   # |Δμ| = 6.68
    "V12": 2.5,   # |Δμ| = 6.28
    "V10": 2.5,   # |Δμ| = 5.68
    "V7":  2.0,   # |Δμ| = 5.57
    "V11": 2.0,   # |Δμ| = 3.80
    "V4":  2.0,   # |Δμ| = 4.54
    "V16": 2.0,   # |Δμ| = 4.14
    "V1":  1.5,   # |Δμ| = 4.77
}
