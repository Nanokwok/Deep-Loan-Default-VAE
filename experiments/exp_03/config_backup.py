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

# ── Model dimensions ───────────────────────────────────────────────────────────
INPUT_DIM       = 30        # Time + V1-V28 + Amount
LATENT_DIM      = 2
ENCODER_DIMS    = [16, 8, 4]
DECODER_DIMS    = [4, 8, 16]

# ── VAE Hyperparameters ────────────────────────────────────────────────────────
BETA            = 0.001
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 256
NUM_EPOCHS      = 100
RANDOM_SEED     = 42
