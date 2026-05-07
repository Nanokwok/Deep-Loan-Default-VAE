"""
Central configuration for the Deep Loan Default VAE project.
Edit values here rather than hunting through individual scripts.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR    = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROC_DIR   = os.path.join(ROOT_DIR, "data", "processed")
REPORTS_DIR     = os.path.join(ROOT_DIR, "reports")
FIGURES_DIR     = os.path.join(REPORTS_DIR, "figures")

RAW_CSV         = os.path.join(DATA_RAW_DIR, "accepted_2007_to_2018Q4.csv")

# ── Target column & labels ─────────────────────────────────────────────────────
TARGET_COL      = "loan_status"
NORMAL_LABEL    = "Fully Paid"
ANOMALY_LABEL   = "Charged Off"

# ── Features available at application time (no data leakage) ──────────────────
APPLICATION_FEATURES = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "purpose", "addr_state",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "initial_list_status", "application_type",
]

# ── VAE Hyperparameters (tuned in Phase 4) ────────────────────────────────────
LATENT_DIM      = 16
ENCODER_DIMS    = [256, 128, 64]   # hidden layer sizes (encoder)
DECODER_DIMS    = [64, 128, 256]   # mirror of encoder
BETA            = 1.0              # KL weight in β-VAE loss
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 512
NUM_EPOCHS      = 50
RANDOM_SEED     = 42
