"""
Data Preprocessing
Pipeline:
  1. Load only application-time features (no post-approval leakage).
  2. Filter to Fully Paid / Charged Off only.
  3. Parse messy string columns (term, emp_length).
  4. Impute missing values (median for numeric, mode for nominal).
  5. Ordinal-encode features with a true natural order (grade, sub_grade).
     One-Hot-Encode all nominal categoricals (no dummy drop — VAE must
     reconstruct every column; false ordinal distances corrupt latent space).
  6. Semi-supervised split: train = Fully Paid 80 %, test = Fully Paid 20 % + ALL Charged Off.
  7. Fit StandardScaler on train only; transform both splits.
  8. Persist artifacts to data/processed/.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    APPLICATION_FEATURES,
    ANOMALY_LABEL,
    DATA_PROC_DIR,
    NORMAL_LABEL,
    RANDOM_SEED,
    RAW_CSV,
    TARGET_COL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TRAIN_NORMAL_FRAC: float = 0.80

# Natural ordinal mappings — only used where order is semantically real
GRADE_MAP: dict[str, int] = {g: i + 1 for i, g in enumerate("ABCDEFG")}
SUBGRADE_MAP: dict[str, int] = {
    f"{g}{n}": i * 5 + n
    for i, g in enumerate("ABCDEFG")
    for n in range(1, 6)
}

# Columns with NO natural order: encode with OHE, not ordinal.
# False ordinal distances (purpose=2 vs purpose=1) corrupt VAE reconstruction loss.
NOMINAL_COLS: list[str] = [
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "initial_list_status",
    "application_type",
]

# Purely numeric after parsing (no encoding needed)
NUMERIC_COLS: list[str] = [
    c for c in APPLICATION_FEATURES if c not in NOMINAL_COLS + ["grade", "sub_grade"]
]


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_term(series: pd.Series) -> pd.Series:
    """' 36 months' → 36."""
    return pd.to_numeric(series.str.extract(r"(\d+)")[0], errors="coerce")


def _parse_emp_length(series: pd.Series) -> pd.Series:
    """'10+ years' → 10, '< 1 year' → 0, 'n/a' / NaN → NaN."""
    s = series.str.lower().str.strip()
    s = s.replace({"< 1 year": "0", "10+ years": "10", "n/a": np.nan})
    return pd.to_numeric(s.str.extract(r"^(\d+)")[0], errors="coerce")


# ── Pipeline steps ────────────────────────────────────────────────────────────

def load_and_filter(path: str | Path) -> pd.DataFrame:
    """Read only TARGET_COL + APPLICATION_FEATURES; keep Fully Paid / Charged Off."""
    usecols = [TARGET_COL] + APPLICATION_FEATURES
    log.info("Reading CSV (usecols only) …")
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    log.info("Raw rows: %d", len(df))

    df = df[df[TARGET_COL].isin([NORMAL_LABEL, ANOMALY_LABEL])].copy()
    log.info(
        "After status filter → %d rows  |  %s: %d  |  %s: %d",
        len(df),
        NORMAL_LABEL, (df[TARGET_COL] == NORMAL_LABEL).sum(),
        ANOMALY_LABEL, (df[TARGET_COL] == ANOMALY_LABEL).sum(),
    )
    return df


def parse_string_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["term"] = _parse_term(df["term"])
    df["emp_length"] = _parse_emp_length(df["emp_length"])
    return df


def impute(
    df: pd.DataFrame,
    medians: dict[str, float] | None = None,
    modes: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, str]]:
    """
    Median-impute numeric cols (fit on train only).
    Mode-impute nominal cols (fit on train only).
    Pass pre-computed dicts at test time to prevent leakage.
    """
    df = df.copy()

    # Numeric: grade + sub_grade are int after map; include them + NUMERIC_COLS
    num_targets = NUMERIC_COLS + ["grade", "sub_grade"]

    if medians is None:
        medians = {c: float(df[c].median()) for c in num_targets if c in df.columns}
    for col, val in medians.items():
        df[col] = df[col].fillna(val)

    if modes is None:
        modes = {
            c: str(df[c].mode(dropna=True).iloc[0])
            for c in NOMINAL_COLS if c in df.columns and df[c].notna().any()
        }
    for col, val in modes.items():
        df[col] = df[col].fillna(val)

    return df, medians, modes


def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit ordinal maps for grade / sub_grade (natural credit-risk order)."""
    df = df.copy()
    df["grade"] = df["grade"].map(GRADE_MAP)
    df["sub_grade"] = df["sub_grade"].map(SUBGRADE_MAP)
    return df


def fit_ohe(df: pd.DataFrame) -> OneHotEncoder:
    """
    Fit OHE on NOMINAL_COLS using the full filtered set for complete vocabulary.
    drop=None — keep ALL dummy columns so the VAE can reconstruct every category.
    handle_unknown='ignore' — unseen categories at inference time → all-zero vector.
    """
    enc = OneHotEncoder(
        drop=None,
        sparse_output=False,
        handle_unknown="ignore",
        dtype=np.float32,
    )
    enc.fit(df[NOMINAL_COLS])
    return enc


def apply_ohe(df: pd.DataFrame, enc: OneHotEncoder) -> pd.DataFrame:
    """Replace NOMINAL_COLS with OHE columns; return new DataFrame."""
    ohe_array = enc.transform(df[NOMINAL_COLS])
    ohe_cols = enc.get_feature_names_out(NOMINAL_COLS).tolist()
    ohe_df = pd.DataFrame(ohe_array, columns=ohe_cols, index=df.index)

    non_nominal = [c for c in df.columns if c not in NOMINAL_COLS]
    return pd.concat([df[non_nominal].reset_index(drop=True),
                      ohe_df.reset_index(drop=True)], axis=1)


def semi_supervised_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    train : 80 % of Fully Paid  (VAE trains on normal only, no labels)
    test  : 20 % of Fully Paid  +  ALL Charged Off  +  integer labels (0/1)
    """
    normal_df  = df[df[TARGET_COL] == NORMAL_LABEL]
    anomaly_df = df[df[TARGET_COL] == ANOMALY_LABEL]

    train_normal, test_normal = train_test_split(
        normal_df, train_size=TRAIN_NORMAL_FRAC,
        random_state=RANDOM_SEED, shuffle=True,
    )
    test_df = pd.concat([test_normal, anomaly_df], ignore_index=True)
    test_labels = np.array(
        [0] * len(test_normal) + [1] * len(anomaly_df), dtype=np.int8
    )
    log.info(
        "Split → train_normal: %d  |  test_normal: %d  |  test_anomaly: %d",
        len(train_normal), len(test_normal), len(anomaly_df),
    )
    feature_cols = [c for c in APPLICATION_FEATURES if c != TARGET_COL]
    return (
        train_normal[feature_cols].reset_index(drop=True),
        test_df[feature_cols].reset_index(drop=True),
        test_labels,
    )


def scale(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """
    Fit StandardScaler on train only; transform both.
    OHE binary columns are also scaled — this centres them and gives the VAE
    uniform loss weighting across all input dimensions.
    """
    feature_names = train_X.columns.tolist()
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(train_X.to_numpy(dtype=np.float32))
    test_arr  = scaler.transform(test_X.to_numpy(dtype=np.float32))
    return train_arr, test_arr, scaler, feature_names


def save_artifacts(
    train_arr: np.ndarray,
    test_arr: np.ndarray,
    test_labels: np.ndarray,
    scaler: StandardScaler,
    ohe: OneHotEncoder,
    medians: dict[str, float],
    modes: dict[str, str],
    feature_names: list[str],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "train_features.npy", train_arr)
    np.save(out / "test_features.npy",  test_arr)
    np.save(out / "test_labels.npy",    test_labels)

    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out / "ohe_encoder.pkl", "wb") as f:
        pickle.dump(ohe, f)

    with open(out / "imputation_medians.json", "w") as f:
        json.dump(medians, f, indent=2)
    with open(out / "imputation_modes.json", "w") as f:
        json.dump(modes, f, indent=2)

    # Final ordered feature name list (post-OHE expansion) for the VAE input layer
    with open(out / "feature_columns.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    log.info("Artifacts saved to %s", out.resolve())
    log.info("  train_features : %s  dtype=%s", train_arr.shape, train_arr.dtype)
    log.info("  test_features  : %s  dtype=%s", test_arr.shape, test_arr.dtype)
    log.info("  test_labels    : %s  (0=normal, 1=anomaly)", test_labels.shape)
    log.info("  input_dim      : %d", train_arr.shape[1])


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    # 1. Load + filter
    df = load_and_filter(RAW_CSV)

    # 2. Parse string-encoded numerics
    df = parse_string_fields(df)

    # 3. Apply explicit ordinal maps (grade, sub_grade) on full set — no stats
    df = encode_ordinal(df)

    # 4. Fit OHE vocabulary on full filtered set — vocabulary discovery, not statistics
    ohe = fit_ohe(df)

    # 5. Split BEFORE fitting any statistics (median, mode, scaler)
    train_raw, test_raw, test_labels = semi_supervised_split(df)

    # 6. Fit imputation on train only; apply to both
    train_raw, medians, modes = impute(train_raw)
    test_raw, _, _             = impute(test_raw, medians=medians, modes=modes)

    # 7. Apply OHE (transform only — encoder already fitted on full vocab)
    train_raw = apply_ohe(train_raw, ohe)
    test_raw  = apply_ohe(test_raw,  ohe)

    # 8. Fit scaler on train; transform both
    train_arr, test_arr, scaler, feature_names = scale(train_raw, test_raw)

    # Sanity checks
    assert not np.isnan(train_arr).any(),  "NaN in train_features"
    assert not np.isnan(test_arr).any(),   "NaN in test_features"
    assert not np.isinf(train_arr).any(),  "Inf in train_features"
    log.info("Sanity checks passed.")

    save_artifacts(
        train_arr, test_arr, test_labels,
        scaler, ohe, medians, modes, feature_names,
        DATA_PROC_DIR,
    )


if __name__ == "__main__":
    run()
