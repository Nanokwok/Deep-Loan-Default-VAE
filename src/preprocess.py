"""
Preprocessing pipeline — Credit Card Fraud Detection (semi-supervised VAE).

Pipeline:
  1. Load creditcard.csv
  2. 3-way semi-supervised split (fit scalers on X_train only):
       Normal (Class 0) → 70% Train / 15% Val / 15% Test
       Fraud  (Class 1) →           50% Val   / 50% Test

       X_train : Normal (Train) only
       X_val   : Normal (Val)   + Fraud (Val)
       X_test  : Normal (Test)  + Fraud (Test)

  3. Two-stage scaling (fit on X_train only, no val/test leakage):

       Stage 1 — RobustScaler on Time & Amount:
         Removes the influence of extreme outliers (IQR-based centering).
         Amount has a heavy right tail (max ~25k vs median ~22);
         Time is monotonically increasing over 48 h.

       Stage 2 — StandardScaler on all 30 features:
         Forces every column to μ≈0, σ≈1 — the target space for MSE-based
         VAE reconstruction. V1–V28 are already PCA-normalised so this stage
         is near-identity for them but guarantees consistency.

       Stage 3 — Clip to ±CLIP_THRESHOLD (default 5.0):
         Caps isolated PCA outliers (e.g. V28 > 100σ) that would otherwise
         dominate the MSE loss. Applied identically to all splits (no fit).

       Net effect: Time & Amount → RobustScaler → StandardScaler → clip
                   V1–V28       →                 StandardScaler → clip

  4. Save to data/processed/

Outputs:
  X_train.npy          (N_train, 30) float32 — normal only
  X_val.npy            (N_val,   30) float32 — normal + fraud (val)
  X_test.npy           (N_test,  30) float32 — normal + fraud (test)
  y_train.npy          (N_train,)    int8    — all zeros
  y_val.npy            (N_val,)      int8    — 0/1
  y_test.npy           (N_test,)     int8    — 0/1
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

NORMAL_VAL_SIZE:  float = 0.15   # fraction of normal rows → val
NORMAL_TEST_SIZE: float = 0.15   # fraction of normal rows → test  (remainder → train)
FRAUD_TEST_SIZE:  float = 0.50   # fraction of fraud rows → test   (remainder → val)
RANDOM_SEED:      int   = cfg.RANDOM_SEED
CLIP_THRESHOLD:   float = 5.0    # post-standardisation clip; >5σ is <0.0001% of N(0,1)

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
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series,    pd.Series,    pd.Series,
]:
    """
    3-way semi-supervised split.

    Normal (Class 0): 70% train / 15% val / 15% test.
    Fraud  (Class 1): 50% val  / 50% test.

    Returns:
        X_train, X_val, X_test  — feature DataFrames
        y_train, y_val, y_test  — int8 label Series
    """
    normal = df[df[cfg.TARGET_COL] == 0][FEATURE_COLS].reset_index(drop=True)
    fraud  = df[df[cfg.TARGET_COL] == 1][FEATURE_COLS].reset_index(drop=True)

    # ── Normal split: 70 / 15 / 15 ───────────────────────────────────────────
    # Step 1: carve out 30% (val+test) from normal
    X_tr, X_normal_remaining = train_test_split(
        normal,
        test_size=NORMAL_VAL_SIZE + NORMAL_TEST_SIZE,  # 0.30
        random_state=RANDOM_SEED,
    )
    # Step 2: split the 30% evenly into val and test (0.15 / 0.30 = 0.5 each)
    X_val_normal, X_test_normal = train_test_split(
        X_normal_remaining,
        test_size=0.5,
        random_state=RANDOM_SEED,
    )

    # ── Fraud split: 50 / 50 ─────────────────────────────────────────────────
    X_val_fraud, X_test_fraud = train_test_split(
        fraud,
        test_size=FRAUD_TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    # ── Assemble val and test ─────────────────────────────────────────────────
    X_val  = pd.concat([X_val_normal,  X_val_fraud],  ignore_index=True)
    X_test = pd.concat([X_test_normal, X_test_fraud], ignore_index=True)

    y_train = pd.Series(np.zeros(len(X_tr), dtype=np.int8), name=cfg.TARGET_COL)
    y_val   = pd.Series(
        np.concatenate([
            np.zeros(len(X_val_normal),  dtype=np.int8),
            np.ones(len(X_val_fraud),    dtype=np.int8),
        ]),
        name=cfg.TARGET_COL,
    )
    y_test  = pd.Series(
        np.concatenate([
            np.zeros(len(X_test_normal), dtype=np.int8),
            np.ones(len(X_test_fraud),   dtype=np.int8),
        ]),
        name=cfg.TARGET_COL,
    )

    log.info(
        "Split  train_normal=%d  "
        "val_normal=%d  val_fraud=%d  val_fraud_rate=%.2f%%  "
        "test_normal=%d  test_fraud=%d  test_fraud_rate=%.2f%%",
        len(X_tr),
        len(X_val_normal),  len(X_val_fraud),  y_val.mean()  * 100,
        len(X_test_normal), len(X_test_fraud), y_test.mean() * 100,
    )
    return (
        X_tr.reset_index(drop=True),
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


_OUTLIER_COLS: list[str] = ["Time", "Amount"]   # heavy-tailed; need Stage 1


def scale_features(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """
    Three-stage scaling fit on X_train only (no leakage into val/test).

    Stage 1: RobustScaler on Time & Amount — neutralises outlier pull.
    Stage 2: StandardScaler on all 30 features — maps everything to μ≈0, σ≈1.
    Stage 3: Clip to ±CLIP_THRESHOLD — caps residual PCA extremes.

    Returns float32 arrays for train/val/test and a scalers dict.
    """
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    # Stage 1: RobustScaler on outlier-prone columns
    robust_scaler = RobustScaler()
    X_train[_OUTLIER_COLS] = robust_scaler.fit_transform(X_train[_OUTLIER_COLS])
    X_val[_OUTLIER_COLS]   = robust_scaler.transform(X_val[_OUTLIER_COLS])
    X_test[_OUTLIER_COLS]  = robust_scaler.transform(X_test[_OUTLIER_COLS])

    # Stage 2: StandardScaler on ALL features (incl. V1-V28 for consistency)
    standard_scaler = StandardScaler()
    X_train[FEATURE_COLS] = standard_scaler.fit_transform(X_train[FEATURE_COLS])
    X_val[FEATURE_COLS]   = standard_scaler.transform(X_val[FEATURE_COLS])
    X_test[FEATURE_COLS]  = standard_scaler.transform(X_test[FEATURE_COLS])

    # Stage 3: clip to [-CLIP_THRESHOLD, +CLIP_THRESHOLD]
    tr_arr = np.clip(X_train[FEATURE_COLS].to_numpy(dtype=np.float32), -CLIP_THRESHOLD, CLIP_THRESHOLD)
    va_arr = np.clip(X_val[FEATURE_COLS].to_numpy(dtype=np.float32),   -CLIP_THRESHOLD, CLIP_THRESHOLD)
    te_arr = np.clip(X_test[FEATURE_COLS].to_numpy(dtype=np.float32),  -CLIP_THRESHOLD, CLIP_THRESHOLD)

    log.info(
        "Scaling complete (train)  global_mean=%.4f  global_std=%.4f  "
        "max_abs=%.2f  (clipped at ±%.1f)",
        tr_arr.mean(), tr_arr.std(), np.abs(tr_arr).max(), CLIP_THRESHOLD,
    )

    return (
        tr_arr,
        va_arr,
        te_arr,
        {"robust": robust_scaler, "standard": standard_scaler},
    )


def save_artifacts(
    proc_dir: str | Path,
    X_train:  np.ndarray,
    X_val:    np.ndarray,
    X_test:   np.ndarray,
    y_train:  pd.Series,
    y_val:    pd.Series,
    y_test:   pd.Series,
    scalers:  dict[str, object],
) -> None:
    out = Path(proc_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "X_train.npy", X_train)
    np.save(out / "X_val.npy",   X_val)
    np.save(out / "X_test.npy",  X_test)
    np.save(out / "y_train.npy", y_train.to_numpy())
    np.save(out / "y_val.npy",   y_val.to_numpy())
    np.save(out / "y_test.npy",  y_test.to_numpy())

    with open(out / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scalers, f)

    log.info("Saved to %s:", out)
    log.info("  X_train.npy  %s  float32  (normal only)", X_train.shape)
    log.info("  X_val.npy    %s  float32  fraud_rate=%.2f%%", X_val.shape,  y_val.mean()  * 100)
    log.info("  X_test.npy   %s  float32  fraud_rate=%.2f%%", X_test.shape, y_test.mean() * 100)
    log.info("  scaler.pkl   keys: robust, standard")


# ── Entry point ────────────────────────────────────────────────────────────────

def preprocess() -> None:
    df = load_data(cfg.RAW_CSV)

    X_train, X_val, X_test, y_train, y_val, y_test = split_semi_supervised(df)

    X_train_sc, X_val_sc, X_test_sc, scalers = scale_features(X_train, X_val, X_test)

    save_artifacts(
        cfg.DATA_PROC_DIR,
        X_train_sc, X_val_sc, X_test_sc,
        y_train, y_val, y_test,
        scalers,
    )
    log.info("Preprocessing complete.")


if __name__ == "__main__":
    preprocess()
