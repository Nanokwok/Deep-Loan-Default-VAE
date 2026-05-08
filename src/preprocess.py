"""
Data Preprocessing
Pipeline (all statistical fits on train split only — zero leakage):
  1.  Load application-time features only (no post-approval columns).
  2.  Filter to Fully Paid / Charged Off.
  3.  Parse string-encoded numerics (term, emp_length).
  4.  Ordinal-encode grade / sub_grade (natural credit-risk order).
  5.  Three-way split  →  Train 70 % / Val 15 % / Test 15 %
        · Train : Fully Paid only  (VAE trains unsupervised on normal class)
        · Val   : Fully Paid + Charged Off at natural default rate
                  (used to tune the reconstruction-error threshold / find best F1)
        · Test  : Fully Paid + Charged Off at natural default rate
                  (held-out; touched only for final reported metrics)
  6.  Impute: median for numeric, mode for nominal (fit on train only).
  6b. Log1p: log(1+x) on right-skewed numeric cols (parameterless; applied to
       all splits identically). Compresses heavy tails of annual_inc, revol_bal,
       dti, etc. before StandardScaler — brings them within ~3–5 σ.
  6c. Winsorize: clip at [p1, p99] in log-space (fit on train only).
       Safety net for any residual outliers after log transform.
  7.  Frequency-bin nominal cols (fit on train only):
        keep Top-N most frequent categories per column,
        map all others → 'other'  (caps OHE width at ≤ N+1 dummies per group).
  8.  OneHotEncode nominal cols (fit on train only, drop=None).
  9.  Drop dead OHE columns (< 1 % activation in training, fit on train only).
        Must run BEFORE StandardScaler — see remove_dead_columns docstring.
  10. StandardScaler (fit on train only).
  11. Save all artifacts to data/processed/.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.config import (   # noqa: E402
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

# ── Tunable constants ─────────────────────────────────────────────────────────

# 70 / 15 / 15 split on normal (Fully Paid) rows.
# Val and test each receive a natural-rate subsample of Charged Off so their
# class distribution matches the real-world lending portfolio (~20 % default).
TRAIN_NORMAL_FRAC: float = 0.70
VAL_NORMAL_FRAC:   float = 0.15
# TEST_NORMAL_FRAC is implicit: 1 - TRAIN - VAL = 0.15

# Target anomaly fraction in val and test sets.
# None → auto-computed from the raw dataset (recommended).
# float → override (e.g. 0.20 for an explicit 20 % target).
TARGET_ANOMALY_RATE: float | None = None

# OHE columns active in fewer than this fraction of training rows are dropped.
# Bernoulli variance identity: Var = p*(1-p) → threshold ≈ 0.0099.
# Must match the EDA notebook's DEAD_COL_THRESH.
DEAD_COL_ACTIVATION_THRESHOLD: float = 0.01

# Percentile bounds for winsorization (fit on train only).
# Clips extreme outliers before StandardScaler so no feature exceeds ~5 σ.
WINSORIZE_LOWER: float = 0.01   # 1st  percentile
WINSORIZE_UPPER: float = 0.99   # 99th percentile

# Right-skewed numeric columns that receive log(1+x) before StandardScaler.
# All must be non-negative after imputation (any negatives are clipped to 0 first).
LOG1P_COLS: list[str] = [
    "annual_inc",      # income — extreme right skew ($5 M outliers)
    "revol_bal",       # revolving balance — zero-inflated + heavy tail
    "dti",             # debt-to-income ratio — right-skewed
    "loan_amnt",       # loan amount
    "funded_amnt",     # funded amount
    "installment",     # monthly payment
    "delinq_2yrs",     # count — zero-inflated
    "inq_last_6mths",  # count — zero-inflated
    "open_acc",        # count
    "pub_rec",         # count — zero-inflated
    "revol_util",      # 0–100 % utilization — right-skewed
    "total_acc",       # count
]

# Explicit ordinal maps — only where order is semantically real
GRADE_MAP: dict[str, int] = {g: i + 1 for i, g in enumerate("ABCDEFG")}
SUBGRADE_MAP: dict[str, int] = {
    f"{g}{n}": i * 5 + n
    for i, g in enumerate("ABCDEFG")
    for n in range(1, 6)
}

# Nominal categoricals → OneHotEncoder (false ordinal distances corrupt VAE loss)
NOMINAL_COLS: list[str] = [
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "initial_list_status",
    "application_type",
]

# Max categories kept per nominal column; rest → 'other'.
TOP_N_CATEGORIES: dict[str, int] = {
    "home_ownership":      10,
    "verification_status": 10,
    "purpose":             13,
    "addr_state":          15,
    "initial_list_status": 10,
    "application_type":    10,
}

NUMERIC_COLS: list[str] = [
    c for c in APPLICATION_FEATURES
    if c not in NOMINAL_COLS + ["grade", "sub_grade"]
]


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_term(series: pd.Series) -> pd.Series:
    """' 36 months' → 36."""
    return pd.to_numeric(series.str.extract(r"(\d+)")[0], errors="coerce")


def _parse_emp_length(series: pd.Series) -> pd.Series:
    """'10+ years' → 10  |  '< 1 year' → 0  |  'n/a' / NaN → NaN."""
    s = series.str.lower().str.strip()
    s = s.replace({"< 1 year": "0", "10+ years": "10", "n/a": np.nan})
    return pd.to_numeric(s.str.extract(r"^(\d+)")[0], errors="coerce")


# ── Pipeline steps ────────────────────────────────────────────────────────────

def load_and_filter(path: str | Path) -> pd.DataFrame:
    """Load TARGET_COL + APPLICATION_FEATURES; keep Fully Paid / Charged Off only."""
    usecols = [TARGET_COL] + APPLICATION_FEATURES
    log.info("Reading CSV (usecols only) …")
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    log.info("Raw rows: %d", len(df))
    df = df[df[TARGET_COL].isin([NORMAL_LABEL, ANOMALY_LABEL])].copy()
    log.info(
        "After status filter → %d rows  |  %s: %d  |  %s: %d",
        len(df),
        NORMAL_LABEL,  (df[TARGET_COL] == NORMAL_LABEL).sum(),
        ANOMALY_LABEL, (df[TARGET_COL] == ANOMALY_LABEL).sum(),
    )
    return df


def parse_string_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["term"]       = _parse_term(df["term"])
    df["emp_length"] = _parse_emp_length(df["emp_length"])
    return df


def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Map grade / sub_grade to integers (natural credit-risk order)."""
    df = df.copy()
    df["grade"]     = df["grade"].map(GRADE_MAP)
    df["sub_grade"] = df["sub_grade"].map(SUBGRADE_MAP)
    return df


def semi_supervised_split(df: pd.DataFrame) -> tuple[
    pd.DataFrame,               # train_X  — Fully Paid only (VAE training)
    pd.DataFrame, np.ndarray,   # val_X,   val_labels
    pd.DataFrame, np.ndarray,   # test_X,  test_labels
]:
    """
    Three-way split: Train 70 % / Val 15 % / Test 15 %.

    Why a separate Validation set?
    ───────────────────────────────
    The VAE produces a reconstruction error for each sample. To turn that
    continuous score into a binary default/no-default prediction, we need to
    pick a decision threshold (e.g. the value that maximises F1-Score).

    · Finding that threshold on TEST data would be data snooping — the test
      set must remain completely untouched until the very final evaluation.
    · Val set is used exclusively for threshold search and early stopping.
    · Test set is used exactly once to report the final metrics.

    Class-distribution fix (carried forward from previous revision):
    ────────────────────────────────────────────────────────────────
    Val and test each receive a Charged Off subsample sized to match the
    natural ~20 % portfolio default rate, not ALL anomalies.
    Val and test anomaly pools are drawn WITHOUT replacement from the total
    Charged Off population so there is zero overlap between the two sets.
    """
    normal_df  = df[df[TARGET_COL] == NORMAL_LABEL]
    anomaly_df = df[df[TARGET_COL] == ANOMALY_LABEL]

    n_normal_total  = len(normal_df)
    n_anomaly_total = len(anomaly_df)
    natural_rate    = n_anomaly_total / (n_normal_total + n_anomaly_total)
    target_rate     = TARGET_ANOMALY_RATE if TARGET_ANOMALY_RATE is not None \
                      else natural_rate

    log.info(
        "Natural default rate: %.2f %%  |  target val/test anomaly rate: %.2f %%",
        natural_rate * 100, target_rate * 100,
    )

    # ── Split normal rows 70 / 15 / 15 ───────────────────────────────────────
    train_normal, remaining_normal = train_test_split(
        normal_df,
        train_size=TRAIN_NORMAL_FRAC,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    # Remaining 30 % split evenly → val 15 %, test 15 %
    val_normal, test_normal = train_test_split(
        remaining_normal,
        test_size=0.5,          # 50 % of the 30 % remainder = 15 % of total
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    # ── Allocate Charged Off: val pool first, then test pool (no overlap) ────
    def _sample_anomalies(
        pool: pd.DataFrame,
        n_normal: int,
        rate: float,
        label: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Draw a rate-matched anomaly sample; return (sample, remaining_pool)."""
        n_want = int(round(n_normal * rate / (1.0 - rate)))
        n_want = min(n_want, len(pool))
        sample    = pool.sample(n=n_want, random_state=RANDOM_SEED)
        remaining = pool.drop(sample.index)
        log.info(
            "  %-5s anomaly sample: %d  (pool remaining: %d)",
            label, n_want, len(remaining),
        )
        return sample, remaining

    val_anomaly,  pool_after_val  = _sample_anomalies(
        anomaly_df, len(val_normal),  target_rate, "val"
    )
    test_anomaly, pool_holdout    = _sample_anomalies(
        pool_after_val, len(test_normal), target_rate, "test"
    )

    # ── Assemble splits ────────────────────────────────────────────────────────
    def _build_split(
        normal_part: pd.DataFrame,
        anomaly_part: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        combined = pd.concat([normal_part, anomaly_part], ignore_index=True)
        labels   = np.array(
            [0] * len(normal_part) + [1] * len(anomaly_part), dtype=np.int8
        )
        # Shuffle so val/test are not ordered [all normals … all anomalies]
        rng  = np.random.default_rng(RANDOM_SEED)
        perm = rng.permutation(len(combined))
        return (
            combined.iloc[perm][APPLICATION_FEATURES].reset_index(drop=True),
            labels[perm],
        )

    val_df,  val_labels  = _build_split(val_normal,  val_anomaly)
    test_df, test_labels = _build_split(test_normal, test_anomaly)

    val_rate  = len(val_anomaly)  / len(val_df)
    test_rate = len(test_anomaly) / len(test_df)

    log.info(
        "Split summary\n"
        "  train  : %8d rows  (Fully Paid only)\n"
        "  val    : %8d rows  (%.1f %% anomaly)\n"
        "  test   : %8d rows  (%.1f %% anomaly)\n"
        "  holdout: %8d Charged Off rows unused (available for stress-testing)",
        len(train_normal),
        len(val_df),  val_rate  * 100,
        len(test_df), test_rate * 100,
        len(pool_holdout),
    )

    return (
        train_normal[APPLICATION_FEATURES].reset_index(drop=True),
        val_df,  val_labels,
        test_df, test_labels,
    )


def impute(
    df: pd.DataFrame,
    medians: dict[str, float] | None = None,
    modes:   dict[str, str]   | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, str]]:
    """
    Median-impute numeric / ordinal columns; mode-impute nominal columns.
    medians / modes = None → compute from df (train-time only).
    Pass pre-computed dicts at val/test time to prevent leakage.
    """
    df = df.copy()
    num_targets = NUMERIC_COLS + ["grade", "sub_grade"]

    if medians is None:
        medians = {c: float(df[c].median()) for c in num_targets if c in df.columns}
    for col, val in medians.items():
        df[col] = df[col].fillna(val)

    if modes is None:
        modes = {
            c: str(df[c].mode(dropna=True).iloc[0])
            for c in NOMINAL_COLS
            if c in df.columns and df[c].notna().any()
        }
    for col, val in modes.items():
        df[col] = df[col].fillna(val)

    return df, medians, modes



def apply_log1p(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log(1+x) to right-skewed numeric columns.

    Parameterless — no train/val/test distinction needed.
    Negative values are clipped to 0 before the transform
    (guards against rare data-quality issues in dti / revol_util).
    Columns transformed are listed in LOG1P_COLS.
    """
    df = df.copy()
    for col in LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df


def fit_winsorizer(
    df: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """
    Compute per-column [p_lower, p_upper] clip bounds on all numeric columns.
    Called once on train only; bounds reused for val and test.
    """
    bounds: dict[str, tuple[float, float]] = {}
    cols = NUMERIC_COLS + ["grade", "sub_grade"]
    for col in cols:
        if col in df.columns:
            lo = float(df[col].quantile(WINSORIZE_LOWER))
            hi = float(df[col].quantile(WINSORIZE_UPPER))
            bounds[col] = (lo, hi)
    log.info(
        "Winsorizer fitted on %d columns  (p%.0f / p%.0f).",
        len(bounds), WINSORIZE_LOWER * 100, WINSORIZE_UPPER * 100,
    )
    return bounds


def apply_winsorizer(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Clip each numeric column to its pre-fitted [p_lower, p_upper] bounds."""
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def fit_frequency_binner(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Record Top-N most frequent categories per nominal column (train-time).
    Categories outside the top-N are collapsed to 'other' at transform time.
    """
    binner: dict[str, list[str]] = {}
    log.info("Fitting frequency binner on train split:")
    for col in NOMINAL_COLS:
        n      = TOP_N_CATEGORIES.get(col, 15)
        counts = df[col].value_counts(dropna=True)
        top    = counts.nlargest(n).index.tolist()
        binner[col] = top
        n_total = counts.shape[0]
        log.info(
            "  %-26s : keep %2d / %2d categories → max %2d OHE dummies",
            col, len(top), n_total, len(top) + (1 if len(top) < n_total else 0),
        )
    return binner


def apply_frequency_binner(
    df: pd.DataFrame,
    binner: dict[str, list[str]],
) -> pd.DataFrame:
    """Replace infrequent categories with 'other' using a pre-fitted binner."""
    df = df.copy()
    for col, kept in binner.items():
        df[col] = df[col].where(df[col].isin(set(kept)), other="other")
    return df


def fit_ohe(df: pd.DataFrame) -> OneHotEncoder:
    """
    Fit OHE on NOMINAL_COLS of df (train-time only).
    drop=None  — keep all dummies; VAE reconstructs every output dimension.
    handle_unknown='ignore' — unseen categories at inference → all-zero row.
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
    """Replace NOMINAL_COLS with OHE-expanded columns."""
    ohe_array    = enc.transform(df[NOMINAL_COLS])
    ohe_cols     = enc.get_feature_names_out(NOMINAL_COLS).tolist()
    ohe_df       = pd.DataFrame(ohe_array, columns=ohe_cols, index=df.index)
    non_nominal  = [c for c in df.columns if c not in NOMINAL_COLS]
    return pd.concat(
        [df[non_nominal].reset_index(drop=True), ohe_df.reset_index(drop=True)],
        axis=1,
    )


def remove_dead_columns(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, VarianceThreshold]:
    """
    Drop OHE columns active in < DEAD_COL_ACTIVATION_THRESHOLD of training rows.

    Must run on raw 0/1 OHE values BEFORE StandardScaler.
    After scaling every non-constant column has variance = 1.0, making it
    impossible to distinguish a 0.1 %-active column from a 50 %-active one.

    Bernoulli variance: Var(p) = p*(1-p).
    threshold = 0.01 * 0.99 ≈ 0.0099 → drops columns where activation < 1 %.

    Selector fitted on train only; same mask applied to val and test.
    """
    p             = DEAD_COL_ACTIVATION_THRESHOLD
    var_threshold = p * (1.0 - p)

    selector       = VarianceThreshold(threshold=var_threshold)
    train_filtered = selector.fit_transform(train_df.to_numpy(dtype=np.float32))
    val_filtered   = selector.transform(val_df.to_numpy(dtype=np.float32))
    test_filtered  = selector.transform(test_df.to_numpy(dtype=np.float32))

    support       = selector.get_support()
    all_names     = train_df.columns.tolist()
    kept_names    = [n for n, k in zip(all_names, support) if k]
    dropped_names = [n for n, k in zip(all_names, support) if not k]

    if dropped_names:
        log.info(
            "Dead-column removal: %d columns with < %.0f %% training activation:",
            len(dropped_names), p * 100,
        )
        for name in dropped_names:
            log.info("  dropped → %s", name)
    else:
        log.info("Dead-column removal: no dead columns found.")
    log.info(
        "Dead-column removal: %d → %d columns  (removed %d)",
        len(all_names), len(kept_names), len(dropped_names),
    )

    return (
        pd.DataFrame(train_filtered, columns=kept_names),
        pd.DataFrame(val_filtered,   columns=kept_names),
        pd.DataFrame(test_filtered,  columns=kept_names),
        selector,
    )


def scale(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """
    Fit StandardScaler on train only; transform train, val, and test.
    Scaling OHE binary columns gives the VAE uniform loss weighting across dims.
    """
    feature_names = train_df.columns.tolist()
    scaler    = StandardScaler()
    train_arr = scaler.fit_transform(train_df.to_numpy(dtype=np.float32))
    val_arr   = scaler.transform(val_df.to_numpy(dtype=np.float32))
    test_arr  = scaler.transform(test_df.to_numpy(dtype=np.float32))
    return train_arr, val_arr, test_arr, scaler, feature_names


def save_artifacts(
    train_arr:   np.ndarray,
    val_arr:     np.ndarray,
    val_labels:  np.ndarray,
    test_arr:    np.ndarray,
    test_labels: np.ndarray,
    scaler:      StandardScaler,
    ohe:         OneHotEncoder,
    binner:      dict[str, list[str]],
    vt_selector:       VarianceThreshold,
    winsorizer_bounds: dict[str, tuple[float, float]],
    medians:           dict[str, float],
    modes:             dict[str, str],
    feature_names: list[str],
    out_dir:     str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "train_features.npy", train_arr)
    np.save(out / "val_features.npy",   val_arr)
    np.save(out / "val_labels.npy",     val_labels)
    np.save(out / "test_features.npy",  test_arr)
    np.save(out / "test_labels.npy",    test_labels)

    for fname, obj in [
        ("scaler.pkl",            scaler),
        ("ohe_encoder.pkl",       ohe),
        ("frequency_binner.pkl",  binner),
        ("variance_selector.pkl", vt_selector),
    ]:
        with open(out / fname, "wb") as f:
            pickle.dump(obj, f)

    with open(out / "log1p_cols.json", "w") as f:
        json.dump(LOG1P_COLS, f, indent=2)
    with open(out / "winsorizer_bounds.json", "w") as f:
        json.dump({k: list(v) for k, v in winsorizer_bounds.items()}, f, indent=2)
    with open(out / "imputation_medians.json", "w") as f:
        json.dump(medians, f, indent=2)
    with open(out / "imputation_modes.json", "w") as f:
        json.dump(modes, f, indent=2)
    with open(out / "feature_columns.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    log.info("Artifacts saved → %s", out.resolve())
    log.info(
        "Final shapes\n"
        "  train : %s\n"
        "  val   : %s  labels: %s\n"
        "  test  : %s  labels: %s",
        train_arr.shape,
        val_arr.shape,  val_labels.shape,
        test_arr.shape, test_labels.shape,
    )
    log.info("=" * 50)
    log.info("input_dim (VAE encoder input): %d", train_arr.shape[1])
    log.info("=" * 50)


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    # ── 1–4: Load, parse, encode ──────────────────────────────────────────────
    df = load_and_filter(RAW_CSV)
    df = parse_string_fields(df)
    df = encode_ordinal(df)

    # ── 5: Three-way split — ALL subsequent fits use train only ───────────────
    train_raw, val_raw, val_labels, test_raw, test_labels = semi_supervised_split(df)

    # ── 6: Impute (fit on train) ──────────────────────────────────────────────
    train_raw, medians, modes = impute(train_raw)
    val_raw,   _,       _     = impute(val_raw,  medians=medians, modes=modes)
    test_raw,  _,       _     = impute(test_raw, medians=medians, modes=modes)

    # ── 6b: Log1p transform (parameterless — applied identically to all splits) ──
    train_raw = apply_log1p(train_raw)
    val_raw   = apply_log1p(val_raw)
    test_raw  = apply_log1p(test_raw)

    # ── 6c: Winsorize in log-space (fit on train) ─────────────────────────────
    win_bounds = fit_winsorizer(train_raw)
    train_raw  = apply_winsorizer(train_raw, win_bounds)
    val_raw    = apply_winsorizer(val_raw,   win_bounds)
    test_raw   = apply_winsorizer(test_raw,  win_bounds)

    # ── 7: Frequency binning (fit on train) ───────────────────────────────────
    binner    = fit_frequency_binner(train_raw)
    train_raw = apply_frequency_binner(train_raw, binner)
    val_raw   = apply_frequency_binner(val_raw,   binner)
    test_raw  = apply_frequency_binner(test_raw,  binner)

    # ── 8: OHE (fit on train) ─────────────────────────────────────────────────
    ohe       = fit_ohe(train_raw)
    train_ohe = apply_ohe(train_raw, ohe)
    val_ohe   = apply_ohe(val_raw,   ohe)
    test_ohe  = apply_ohe(test_raw,  ohe)
    log.info(
        "Post-OHE shape: train=%s  val=%s  test=%s",
        train_ohe.shape, val_ohe.shape, test_ohe.shape,
    )

    # ── 9: Drop dead OHE columns BEFORE scaling (fit on train) ───────────────
    train_ohe, val_ohe, test_ohe, vt_selector = remove_dead_columns(
        train_ohe, val_ohe, test_ohe
    )

    # ── 10: Scale (fit on train) ──────────────────────────────────────────────
    train_arr, val_arr, test_arr, scaler, feature_names = scale(
        train_ohe, val_ohe, test_ohe
    )

    # ── Sanity checks ─────────────────────────────────────────────────────────
    for name, arr in [("train", train_arr), ("val", val_arr), ("test", test_arr)]:
        assert not np.isnan(arr).any(), f"NaN found in {name}_features"
        assert not np.isinf(arr).any(), f"Inf found in {name}_features"
        assert arr.shape[1] == train_arr.shape[1], f"{name}/train column mismatch"
    log.info("Sanity checks passed.")

    # ── Persist ───────────────────────────────────────────────────────────────
    save_artifacts(
        train_arr,
        val_arr,  val_labels,
        test_arr, test_labels,
        scaler, ohe, binner, vt_selector,
        win_bounds,
        medians, modes, feature_names,
        DATA_PROC_DIR,
    )


if __name__ == "__main__":
    run()
