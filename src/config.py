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
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
# Override in Colab to persist to Drive across VM resets, e.g.:
#   import src.config as cfg
#   cfg.EXPERIMENTS_DIR = "/content/drive/MyDrive/Loan_VAE_Project"

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

# ── VAE Hyperparameters ────────────────────────────────────
LATENT_DIM      = 16
ENCODER_DIMS    = [64, 32, 16]   # hidden layer sizes (encoder)
DECODER_DIMS    = [16, 32, 64]   # mirror of encoder
BETA            = 0.01
LEARNING_RATE   = 1e-3
BATCH_SIZE      = 2048
NUM_EPOCHS      = 300
RANDOM_SEED     = 42

# ── Phase 4 Design Warnings ───────────────────────────────────────────────────
#
# [1] β-VAE RISK WITH TABULAR DATA
#     The KL term forces the latent space toward N(0,1). For images this helps
#     generalization; for tabular data it can erase meaningful cluster structure
#     (e.g. grade A vs grade G customers collapse to the same latent mean).
#     Symptom: reconstruction error distributions for Normal and Anomaly overlap
#              heavily → model loses discriminative power.
#     Mitigation: start with BETA = 0.1–0.5; increase only if the model
#                 over-fits (val reconstruction error diverges from train).
#     Diagnostic: after training, plot latent μ distributions for Fully Paid vs
#                 Charged Off — if indistinguishable, lower BETA.
#
# [2] MSE LOSS WEIGHTING WITH ONE-HOT COLUMNS
#     MSE treats all 53 dimensions equally. A wrong bit in purpose_medical
#     (base rate < 5 %) contributes the same gradient as a $10 k error in
#     annual_inc. For Phase 4 evaluation, consider:
#       a) Weighted MSE: assign higher weight to continuous/ordinal features.
#       b) Hybrid loss: MSE on numeric columns + BCE on binary OHE columns.
#       c) Post-hoc: report reconstruction error computed on numeric-only dims
#          as a secondary metric alongside the full-dim score.
#
# [3] FEATURE INTERACTION LIMITATION vs. SUPERVISED BASELINES
#     A vanilla VAE (linear → ReLU → linear) does not explicitly model
#     cross-feature interactions (e.g. high DTI + low FICO + high grade).
#     XGBoost will likely outperform on standard Precision/Recall/F1.
#     Prepared counter-argument for project defence:
#       "VAE is trained on Fully Paid data only and therefore can detect
#        novel default patterns not present in any historical Charged Off
#        labels — supervised models are blind to these unseen anomaly types.
#        This is the core value proposition of semi-supervised anomaly
#        detection in non-stationary credit risk environments."
#
# [4] LATENT SPACE VISUALIZATION (recommended extra-mile section)
#     After training, reduce latent μ vectors from LATENT_DIM to 2D using
#     t-SNE or UMAP and colour points by label (0 = Fully Paid, 1 = Charged Off).
#     Expected result: Fully Paid points form a tight central cluster; Charged Off
#     points scatter toward the periphery or form a secondary cluster.
#     If the clusters are indistinguishable → lower BETA or increase LATENT_DIM.
