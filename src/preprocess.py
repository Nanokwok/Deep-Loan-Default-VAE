"""
Preprocessing pipeline for the Deep Loan Default VAE project.

Steps:
  1. Load raw CSV (accepted_2007_to_2018Q4.csv)
  2. Select only APPLICATION_FEATURES + TARGET_COL
  3. Filter rows to "Fully Paid" (0) and "Charged Off" (1)
  4. Handle missing values (NaNs) with per-column median (fit on train only)
  5. Apply log1p to right-skewed columns (annual_inc, dti, revol_util)
     — compresses extreme tails; log1p(0) = 0, safe for all non-negative data
  6. Winsorize all columns at p1/p99 (bounds fitted on train only)
     — clips any remaining extreme values after log compression
  7. Scale features with StandardScaler (fit on train only)
  8. Semi-supervised split:
       train  — Fully Paid only (VAE trains on normal class)
       val    — stratified mix of both classes
  9. Save arrays + feature_columns.json to DATA_PROC_DIR

Outputs (DATA_PROC_DIR/):
  X_train.npy        — (N_train, 6) float32, normal only
  y_train.npy        — (N_train,)   all zeros
  X_val.npy          — (N_val,   6) float32, mixed
  y_val.npy          — (N_val,)   0/1
  feature_columns.json
  scaler.pkl         — fitted StandardScaler (for inference)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import src.config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

LABEL_MAP: dict[str, int] = {
    cfg.NORMAL_LABEL:  0,   # "Fully Paid"
    cfg.ANOMALY_LABEL: 1,   # "Charged Off"
}

# Columns with severe right skew (confirmed empirically: max|z| > 6 after StandardScaler).
# log1p compresses the long tail while preserving zero (log1p(0) = 0).
# All three are non-negative by definition, so log1p is always safe.
LOG1P_COLS: list[str] = ["annual_inc", "dti", "revol_util"]

VAL_SIZE:    float = 0.20   # 80 / 20 split; train keeps normal rows only
RANDOM_SEED: int   = cfg.RANDOM_SEED


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def load_and_filter(csv_path: str | Path) -> pd.DataFrame:
    """
    Read raw CSV, keep only the 6 financial features + target column,
    and filter to the two target classes.
    """
    cols_needed = cfg.APPLICATION_FEATURES + [cfg.TARGET_COL]

    log.info("Reading %s ...", csv_path)
    df = pd.read_csv(
        csv_path,
        usecols=cols_needed,
        low_memory=False,
    )
    log.info("Raw rows: %d", len(df))

    # Strip accidental leading/trailing whitespace from the target column
    df[cfg.TARGET_COL] = df[cfg.TARGET_COL].astype(str).str.strip()

    # Keep only the two classes we care about
    df = df[df[cfg.TARGET_COL].isin(LABEL_MAP)].copy()
    df[cfg.TARGET_COL] = df[cfg.TARGET_COL].map(LABEL_MAP)

    log.info(
        "After class filter: %d rows  |  Fully Paid=%d  Charged Off=%d",
        len(df),
        (df[cfg.TARGET_COL] == 0).sum(),
        (df[cfg.TARGET_COL] == 1).sum(),
    )
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Semi-supervised split:
      - train : normal (Fully Paid) rows only
      - val   : stratified 20% of the full dataset (both classes)

    The validation set is carved out first so its rows never contaminate
    the scaler / imputer fit.
    """
    X = df[cfg.APPLICATION_FEATURES]
    y = df[cfg.TARGET_COL]

    X_train_pool, X_val, y_train_pool, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # Train on normal class only (semi-supervised)
    normal_mask = y_train_pool == 0
    X_train = X_train_pool[normal_mask]
    y_train = y_train_pool[normal_mask]

    log.info(
        "Split  train(normal)=%d  val(mixed)=%d  val_anomaly_rate=%.1f%%",
        len(X_train),
        len(X_val),
        y_val.mean() * 100,
    )
    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
    )


def impute(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill NaNs with per-column median. Median fitted on train only."""
    medians = X_train.median()
    missing_train = int(X_train.isnull().sum().sum())
    missing_val   = int(X_val.isnull().sum().sum())
    if missing_train or missing_val:
        log.info(
            "Imputing NaNs  train=%d  val=%d  (median from train)",
            missing_train, missing_val,
        )
    X_train = X_train.fillna(medians)
    X_val   = X_val.fillna(medians)
    return X_train, X_val


def apply_log1p(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply np.log1p to LOG1P_COLS in both splits.
    Only transforms columns that are present in APPLICATION_FEATURES.
    Parameterless — no fit needed, applied identically to train and val.
    """
    cols = [c for c in LOG1P_COLS if c in X_train.columns]
    if cols:
        X_train = X_train.copy()
        X_val   = X_val.copy()
        X_train[cols] = np.log1p(X_train[cols])
        X_val[cols]   = np.log1p(X_val[cols])
        log.info("log1p applied to: %s", cols)
    return X_train, X_val


def winsorize(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clip each column to [p1, p99] computed on X_train.
    Eliminates extreme outliers that survive log1p (e.g. annual_inc > $10M).
    Bounds are fitted on train only — no leakage into val.
    """
    lo = np.percentile(X_train, lower_pct, axis=0)   # shape (n_features,)
    hi = np.percentile(X_train, upper_pct, axis=0)
    X_train = X_train.clip(lower=lo, upper=hi, axis=1)
    X_val   = X_val.clip(lower=lo,   upper=hi, axis=1)
    log.info(
        "Winsorized at p%.0f/p%.0f (fitted on train)",
        lower_pct, upper_pct,
    )
    return X_train, X_val


def scale(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """StandardScaler fitted on train only. Returns float32 arrays."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_val_sc   = scaler.transform(X_val).astype(np.float32)
    log.info(
        "Scaled  train_mean~0: %s  train_std~1: %s",
        np.allclose(X_train_sc.mean(axis=0), 0, atol=1e-5),
        np.allclose(X_train_sc.std(axis=0),  1, atol=1e-2),
    )
    return X_train_sc, X_val_sc, scaler


def save_artifacts(
    proc_dir: str | Path,
    X_train: np.ndarray,
    X_val:   np.ndarray,
    y_train: pd.Series,
    y_val:   pd.Series,
    scaler:  StandardScaler,
) -> None:
    """Write all processed arrays and the scaler to proc_dir."""
    out = Path(proc_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "X_train.npy", X_train)
    np.save(out / "X_val.npy",   X_val)
    np.save(out / "y_train.npy", y_train.to_numpy().astype(np.int8))
    np.save(out / "y_val.npy",   y_val.to_numpy().astype(np.int8))

    with open(out / "feature_columns.json", "w") as f:
        json.dump(cfg.APPLICATION_FEATURES, f, indent=2)

    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    log.info("Saved to %s:", out)
    log.info("  X_train.npy  %s", X_train.shape)
    log.info("  X_val.npy    %s", X_val.shape)
    log.info("  y_train.npy  %s", y_train.shape)
    log.info("  y_val.npy    %s", y_val.shape)
    log.info("  feature_columns.json  %s", cfg.APPLICATION_FEATURES)
    log.info("  scaler.pkl")


# ── Main entry point ───────────────────────────────────────────────────────────

def preprocess() -> None:
    """Run the full preprocessing pipeline end-to-end."""
    df = load_and_filter(cfg.RAW_CSV)

    X_train, X_val, y_train, y_val = split_data(df)

    X_train, X_val = impute(X_train, X_val)

    X_train, X_val = apply_log1p(X_train, X_val)

    X_train, X_val = winsorize(X_train, X_val)

    X_train_sc, X_val_sc, scaler = scale(X_train, X_val)

    save_artifacts(
        cfg.DATA_PROC_DIR,
        X_train_sc, X_val_sc,
        y_train, y_val,
        scaler,
    )
    log.info("Preprocessing complete.")


if __name__ == "__main__":
    preprocess()
