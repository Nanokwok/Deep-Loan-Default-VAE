"""
Preprocessing pipeline — Credit Card Fraud Detection (semi-supervised VAE).

Pipeline:
  1. Load creditcard.csv
  2. Semi-supervised split:
       X_train — 80% of Class 0 (Normal) only
       X_val   — 20% of Class 0  +  100% of Class 1 (Fraud)
  3. Two-stage scaling (fit on X_train only, no val leakage):

       Stage 1 — RobustScaler on Time & Amount:
         Removes the influence of extreme outliers (IQR-based centering).
         Amount has a heavy right tail (max ~25k vs median ~22);
         Time is monotonically increasing over 48 h, creating an outlier-like
         structure for StandardScaler alone.

       Stage 2 — StandardScaler on all 30 features:
         Forces every column to μ≈0, σ≈1 — exactly the space the VAE decoder
         is trained to reconstruct (MSE targets are ~N(0,1)).
         V1–V28 are already PCA-normalised, so this stage is near-identity for
         them but guarantees consistency and removes residual scale differences.

       Net effect: Time & Amount: RobustScaler → StandardScaler
                   V1–V28:                        StandardScaler

  4. Save to data/processed/

Outputs:
  X_train.npy          (N_train, 30) float32 — normal only
  X_val.npy            (N_val,   30) float32 — mixed
  y_train.npy          (N_train,)    int8    — all zeros
  y_val.npy            (N_val,)      int8    — 0/1
  feature_columns.json — ordered list of 30 feature names
  scaler.pkl           — {'robust': RobustScaler, 'standard': StandardScaler}
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

import src.config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

VAL_SIZE:        float = 0.20
RANDOM_SEED:     int   = cfg.RANDOM_SEED
CLIP_THRESHOLD:  float = 5.0   # post-standardisation clip; >5σ is <0.0001% of N(0,1)

# Feature columns in output order (Class column is dropped)
FEATURE_COLS: list[str] = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)   # len == 30


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load creditcard.csv and verify expected columns."""
    log.info("Reading %s ...", csv_path)
    df = pd.read_csv(csv_path)
    log.info(
        "Loaded  rows=%d  Class distribution: 0=%d  1=%d",
        len(df),
        (df[cfg.TARGET_COL] == 0).sum(),
        (df[cfg.TARGET_COL] == 1).sum(),
    )
    missing = [c for c in FEATURE_COLS + [cfg.TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def split_semi_supervised(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    X_train : 80% of Class 0 rows only.
    X_val   : 20% of Class 0 rows  +  ALL Class 1 rows.
    Scalers must be fit on X_train only (no val/fraud leakage).
    """
    normal  = df[df[cfg.TARGET_COL] == 0][FEATURE_COLS].reset_index(drop=True)
    fraud   = df[df[cfg.TARGET_COL] == 1][FEATURE_COLS].reset_index(drop=True)

    X_tr, X_val_normal = train_test_split(
        normal, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )

    X_val   = pd.concat([X_val_normal, fraud], ignore_index=True)
    y_train = pd.Series(np.zeros(len(X_tr),      dtype=np.int8), name=cfg.TARGET_COL)
    y_val   = pd.Series(
        np.concatenate([
            np.zeros(len(X_val_normal), dtype=np.int8),
            np.ones(len(fraud),         dtype=np.int8),
        ]),
        name=cfg.TARGET_COL,
    )

    log.info(
        "Split  train(normal)=%d  val_normal=%d  val_fraud=%d  "
        "val_fraud_rate=%.2f%%",
        len(X_tr), len(X_val_normal), len(fraud),
        len(fraud) / len(X_val) * 100,
    )
    return (
        X_tr.reset_index(drop=True),
        X_val,
        y_train,
        y_val,
    )


_OUTLIER_COLS: list[str] = ["Time", "Amount"]   # heavy-tailed; need Stage 1


def scale_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """
    Two-stage scaling fit on X_train only.

    Stage 1: RobustScaler on Time & Amount — neutralises outlier pull.
    Stage 2: StandardScaler on all 30 features — maps everything to μ≈0, σ≈1.

    Returns float32 arrays in FEATURE_COLS order and a scalers dict.
    """
    X_train = X_train.copy()
    X_val   = X_val.copy()

    # Stage 1: RobustScaler on outlier-prone columns
    robust_scaler = RobustScaler()
    X_train[_OUTLIER_COLS] = robust_scaler.fit_transform(X_train[_OUTLIER_COLS])
    X_val[_OUTLIER_COLS]   = robust_scaler.transform(X_val[_OUTLIER_COLS])

    # Stage 2: StandardScaler on ALL features (incl. V1-V28 for consistency)
    standard_scaler = StandardScaler()
    X_train[FEATURE_COLS] = standard_scaler.fit_transform(X_train[FEATURE_COLS])
    X_val[FEATURE_COLS]   = standard_scaler.transform(X_val[FEATURE_COLS])

    # Stage 3: clip to [-CLIP_THRESHOLD, +CLIP_THRESHOLD]
    # PCA features (V1-V28) can have isolated extreme values (e.g. V28 > 100σ)
    # that blow up MSE loss and destabilise VAE training.
    # Clipping at ±5σ removes <0.0001% of a true Gaussian and is applied
    # identically to both splits (no leakage — no fit required).
    tr_arr = X_train[FEATURE_COLS].to_numpy(dtype=np.float32)
    va_arr = X_val[FEATURE_COLS].to_numpy(dtype=np.float32)
    tr_arr = np.clip(tr_arr, -CLIP_THRESHOLD, CLIP_THRESHOLD)
    va_arr = np.clip(va_arr, -CLIP_THRESHOLD, CLIP_THRESHOLD)

    log.info(
        "Scaling complete (train)  global_mean=%.4f  global_std=%.4f  "
        "max_abs=%.2f  (clipped at ±%.1f)",
        tr_arr.mean(), tr_arr.std(), np.abs(tr_arr).max(), CLIP_THRESHOLD,
    )

    return (
        tr_arr,
        va_arr,
        {"robust": robust_scaler, "standard": standard_scaler},
    )


def save_artifacts(
    proc_dir: str | Path,
    X_train:  np.ndarray,
    X_val:    np.ndarray,
    y_train:  pd.Series,
    y_val:    pd.Series,
    scalers:  dict[str, object],
) -> None:
    out = Path(proc_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "X_train.npy", X_train)
    np.save(out / "X_val.npy",   X_val)
    np.save(out / "y_train.npy", y_train.to_numpy())
    np.save(out / "y_val.npy",   y_val.to_numpy())

    with open(out / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scalers, f)

    log.info("Saved to %s:", out)
    log.info("  X_train.npy  %s  (RobustScaler → StandardScaler)", X_train.shape)
    log.info("  X_val.npy    %s", X_val.shape)
    log.info("  y_train.npy  all-zeros  n=%d", len(y_train))
    log.info("  y_val.npy    fraud_rate=%.2f%%", y_val.mean() * 100)
    log.info("  scaler.pkl   keys: robust, standard")


# ── Entry point ────────────────────────────────────────────────────────────────

def preprocess() -> None:
    df = load_data(cfg.RAW_CSV)

    X_train, X_val, y_train, y_val = split_semi_supervised(df)

    X_train_sc, X_val_sc, scalers = scale_features(X_train, X_val)

    save_artifacts(cfg.DATA_PROC_DIR, X_train_sc, X_val_sc, y_train, y_val, scalers)
    log.info("Preprocessing complete.")


if __name__ == "__main__":
    preprocess()
