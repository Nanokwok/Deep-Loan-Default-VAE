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
ENCODER_DIMS    = [20, 12]
DECODER_DIMS    = [12, 20]

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
