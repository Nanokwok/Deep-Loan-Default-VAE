"""
Threshold Engineering — Credit Card Fraud VAE.

After training, the VAE produces a continuous anomaly score (weighted
reconstruction error) for every transaction.  Turning that score into a
binary decision ("block the card" / "allow") requires choosing a threshold.

This module provides:

  compute_anomaly_scores()    — run a checkpoint on val or test data
  threshold_analysis()        — sweep thresholds, compute P/R/F1/cost at each
  find_optimal_thresholds()   — return the three decision-optimal cut-offs
  plot_threshold_curves()     — 3-panel figure saved to exp_dir
  evaluate_at_threshold()     — full confusion-matrix report at one cut-off
  run_evaluation()            — end-to-end: load → score → analyse → plot → save

Cost matrix (configurable):
    COST_FN  — letting a fraudster through   (default 1 000 ฿)
    COST_FP  — blocking a legitimate customer (default   100 ฿)

Three optimal thresholds are found and reported:
    max_f1       — maximises F1-score (harmonic mean of precision & recall)
    min_cost     — minimises expected cost per transaction (using cost matrix)
    max_recall90 — highest threshold that still achieves recall ≥ 0.90
                   (use when catching fraud is paramount)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

log = logging.getLogger(__name__)


# ── Types ─────────────────────────────────────────────────────────────────────

class ThresholdRow(TypedDict):
    threshold:  float
    precision:  float
    recall:     float
    f1:         float
    tp:         int
    fp:         int
    fn:         int
    tn:         int
    cost:       float   # total cost at this threshold


class OptimalThresholds(TypedDict):
    max_f1:         float
    max_f1_score:   float
    min_cost:       float
    min_cost_value: float
    recall90:       float   # highest threshold with recall ≥ 0.90 (or -1)


# ── Score computation ──────────────────────────────────────────────────────────

def compute_anomaly_scores(
    ckpt_path:       str | Path,
    X:               np.ndarray,
    device:          torch.device,
    feature_weights: torch.Tensor | None = None,
) -> np.ndarray:
    """
    Load a saved checkpoint and return per-sample weighted reconstruction error.

    Args:
        ckpt_path       : path to best_model.pth
        X               : scaled feature array  (N, 30) float32
        device          : torch device
        feature_weights : (30,) weight tensor used during training, or None

    Returns:
        scores : (N,) float32 anomaly scores — higher = more anomalous
    """
    from src.model import BetaVAE
    import src.config as cfg

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    input_dim  = ckpt.get("input_dim",  cfg.INPUT_DIM)
    latent_dim = ckpt.get("latent_dim", cfg.LATENT_DIM)

    # Temporarily patch LATENT_DIM if checkpoint used a different value
    _orig = cfg.LATENT_DIM
    cfg.LATENT_DIM = latent_dim
    model = BetaVAE(input_dim=input_dim).to(device)
    cfg.LATENT_DIM = _orig

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
    with torch.no_grad():
        x_hat, _, _ = model(x_tensor)
        sq = (x_tensor - x_hat) ** 2
        if feature_weights is not None:
            sq = sq * feature_weights.unsqueeze(0)
        scores = sq.mean(dim=1).cpu().numpy()

    log.info(
        "Scores  n=%d  mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        len(scores), scores.mean(), scores.std(), scores.min(), scores.max(),
    )
    return scores


# ── Threshold sweep ────────────────────────────────────────────────────────────

def threshold_analysis(
    scores:   np.ndarray,
    y_true:   np.ndarray,
    n_points: int   = 500,
    cost_fn:  float = 1_000.0,
    cost_fp:  float =   100.0,
) -> list[ThresholdRow]:
    """
    Sweep n_points thresholds between score min and max.
    Return a list of ThresholdRow dicts (one per threshold).

    Cost model:
        total_cost = FN_count × cost_fn + FP_count × cost_fp
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_points)
    rows: list[ThresholdRow] = []

    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        cost = fp * cost_fp + fn * cost_fn

        rows.append(ThresholdRow(
            threshold=float(t),
            precision=prec,
            recall=rec,
            f1=f1,
            tp=tp, fp=fp, fn=fn, tn=tn,
            cost=cost,
        ))

    return rows


# ── Optimal thresholds ────────────────────────────────────────────────────────

def find_optimal_thresholds(
    rows:       list[ThresholdRow],
    min_recall: float = 0.90,
) -> OptimalThresholds:
    """
    Extract three decision-optimal thresholds from a threshold sweep.

    max_f1       — maximises F1 (balanced precision/recall)
    min_cost     — minimises total cost (weighted false positives + negatives)
    recall90     — highest threshold that still achieves recall ≥ min_recall
                   (catches most fraud; use when regulatory mandate applies)
    """
    best_f1_row   = max(rows, key=lambda r: r["f1"])
    best_cost_row = min(rows, key=lambda r: r["cost"])

    recall_ok = [r for r in rows if r["recall"] >= min_recall]
    recall90_thresh = max(r["threshold"] for r in recall_ok) if recall_ok else -1.0

    result = OptimalThresholds(
        max_f1         = best_f1_row["threshold"],
        max_f1_score   = best_f1_row["f1"],
        min_cost       = best_cost_row["threshold"],
        min_cost_value = best_cost_row["cost"],
        recall90       = recall90_thresh,
    )

    log.info("─" * 55)
    log.info("Optimal thresholds:")
    log.info("  Max F1      threshold=%.4f  F1=%.4f  P=%.4f  R=%.4f",
             result["max_f1"], best_f1_row["f1"],
             best_f1_row["precision"], best_f1_row["recall"])
    log.info("  Min cost    threshold=%.4f  cost=%.0f  FP=%d  FN=%d",
             result["min_cost"], best_cost_row["cost"],
             best_cost_row["fp"], best_cost_row["fn"])
    log.info("  Recall≥%.2f threshold=%.4f", min_recall, result["recall90"])
    log.info("─" * 55)

    return result


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_threshold_curves(
    rows:      list[ThresholdRow],
    opt:       OptimalThresholds,
    scores:    np.ndarray,
    y_true:    np.ndarray,
    out_path:  str | Path,
    cost_fn:   float = 1_000.0,
    cost_fp:   float =   100.0,
) -> None:
    """
    3-panel figure:
      Panel 1 — Precision / Recall / F1 vs Threshold
      Panel 2 — Total cost vs Threshold (cost matrix)
      Panel 3 — Precision-Recall curve (model quality reference)
    Saved as PNG to out_path.
    """
    thresholds = [r["threshold"] for r in rows]
    precisions = [r["precision"] for r in rows]
    recalls    = [r["recall"]    for r in rows]
    f1s        = [r["f1"]        for r in rows]
    costs      = [r["cost"]      for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: P / R / F1 vs Threshold ──────────────────────────────────────
    ax = axes[0]
    ax.plot(thresholds, precisions, color="steelblue",  label="Precision")
    ax.plot(thresholds, recalls,    color="crimson",    label="Recall")
    ax.plot(thresholds, f1s,        color="darkorange", label="F1", linewidth=2)

    for label, t, color in [
        ("Max F1",    opt["max_f1"],    "darkorange"),
        ("Min Cost",  opt["min_cost"],  "green"),
        ("Recall≥90%",opt["recall90"], "purple"),
    ]:
        if t >= 0:
            ax.axvline(t, color=color, linestyle="--", linewidth=1.2, label=label)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 2: Cost vs Threshold ─────────────────────────────────────────────
    ax = axes[1]
    ax.plot(thresholds, costs, color="firebrick", linewidth=2)
    ax.axvline(opt["min_cost"], color="green", linestyle="--", linewidth=1.2,
               label=f"Min cost = {opt['min_cost_value']:,.0f} ฿")

    # Annotate cost formula
    ax.text(0.98, 0.97,
            f"FP cost = {cost_fp:,.0f} ฿\nFN cost = {cost_fn:,.0f} ฿",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total cost (฿)")
    ax.set_title("Cost-Matrix Optimisation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 3: PR curve ──────────────────────────────────────────────────────
    ax = axes[2]
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    auroc = roc_auc_score(y_true, scores)
    baseline = y_true.mean()

    ax.plot(rec_curve, prec_curve, color="steelblue", linewidth=2,
            label=f"VAE  AUPRC={auprc:.4f}")
    ax.axhline(baseline, color="grey", linestyle="--",
               label=f"Random  AUPRC={baseline:.4f}")

    # Mark the three operating points on the PR curve
    for label, t, color in [
        ("Max F1",    opt["max_f1"],    "darkorange"),
        ("Min Cost",  opt["min_cost"],  "green"),
    ]:
        if t >= 0:
            idx = int(np.argmin(np.abs([r["threshold"] for r in rows] -
                                        np.full(len(rows), t))))
            ax.scatter(rows[idx]["recall"], rows[idx]["precision"],
                       color=color, zorder=5, s=80, label=label)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve  (AUROC={auroc:.4f})")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle("Threshold Engineering — Fraud VAE", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out_path)


# ── Per-threshold report ──────────────────────────────────────────────────────

def evaluate_at_threshold(
    scores:    np.ndarray,
    y_true:    np.ndarray,
    threshold: float,
    cost_fn:   float = 1_000.0,
    cost_fp:   float =   100.0,
    label:     str   = "Evaluation",
) -> dict:
    """Print and return a full confusion-matrix report at one threshold."""
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    prec  = tp / (tp + fp + 1e-9)
    rec   = tp / (tp + fn + 1e-9)
    f1    = 2 * prec * rec / (prec + rec + 1e-9)
    cost  = fp * cost_fp + fn * cost_fn
    fpr   = fp / (fp + tn + 1e-9)

    print(f"\n{'─'*50}")
    print(f"  {label}  |  threshold = {threshold:.4f}")
    print(f"{'─'*50}")
    print(f"  Confusion matrix:")
    print(f"    TP={tp:>6,}   FP={fp:>6,}")
    print(f"    FN={fn:>6,}   TN={tn:>6,}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}   (fraud caught rate)")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  FPR       : {fpr:.4f}   (normal customers wrongly blocked)")
    print(f"  Total cost: {cost:>12,.0f} ฿")
    print(f"    FP cost : {fp * cost_fp:>12,.0f} ฿  ({fp} customers blocked)")
    print(f"    FN cost : {fn * cost_fn:>12,.0f} ฿  ({fn} frauds missed)")
    print(f"{'─'*50}")

    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=rec, f1=f1, fpr=fpr, cost=cost)


# ── Feature-wise reconstruction error ────────────────────────────────────────

def compute_per_feature_errors(
    ckpt_path:       str | Path,
    X:               np.ndarray,
    device:          torch.device,
    feature_weights: torch.Tensor | None = None,
) -> np.ndarray:
    """
    Return per-sample, per-feature weighted squared reconstruction error.
    Shape: (N, input_dim)
    """
    from src.model import BetaVAE
    import src.config as cfg

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=True)
    input_dim  = ckpt.get("input_dim",  cfg.INPUT_DIM)
    latent_dim = ckpt.get("latent_dim", cfg.LATENT_DIM)

    _orig = cfg.LATENT_DIM
    cfg.LATENT_DIM = latent_dim
    model = BetaVAE(input_dim=input_dim).to(device)
    cfg.LATENT_DIM = _orig

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_t = torch.from_numpy(X.astype(np.float32)).to(device)
    with torch.no_grad():
        x_hat, _, _ = model(x_t)
        sq = (x_t - x_hat) ** 2
        if feature_weights is not None:
            sq = sq * feature_weights.unsqueeze(0)
    return sq.cpu().numpy()   # (N, input_dim)


def plot_feature_reconstruction_error(
    per_feat_errors: np.ndarray,
    y_true:          np.ndarray,
    feature_cols:    list[str],
    out_path:        str | Path,
) -> None:
    """
    Grouped bar chart: mean per-feature weighted MSE for Normal vs Fraud.
    Shows which features are responsible for driving the anomaly score.
    """
    mean_normal = per_feat_errors[y_true == 0].mean(axis=0)
    mean_fraud  = per_feat_errors[y_true == 1].mean(axis=0)
    # Sort by fraud − normal delta (most discriminative first)
    delta = mean_fraud - mean_normal
    order = np.argsort(delta)[::-1]

    n = len(feature_cols)
    x = np.arange(n)
    w = 0.4

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(x - w/2, mean_normal[order], w,
           color="steelblue", alpha=0.85, label="Normal")
    ax.bar(x + w/2, mean_fraud[order],  w,
           color="crimson",   alpha=0.85, label="Fraud")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [feature_cols[i] for i in order], rotation=45, ha="right", fontsize=8
    )
    ax.set_ylabel("Mean weighted squared error")
    ax.set_title(
        "Per-Feature Reconstruction Error — Normal vs Fraud\n"
        "(sorted by fraud−normal gap; features at left drive anomaly score most)"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out_path)


# ── Latent space ──────────────────────────────────────────────────────────────

def compute_latent_mu(
    ckpt_path: str | Path,
    X:         np.ndarray,
    device:    torch.device,
) -> np.ndarray:
    """
    Encode X and return the posterior mean μ.  Shape: (N, latent_dim).
    """
    from src.model import BetaVAE
    import src.config as cfg

    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=True)
    input_dim  = ckpt.get("input_dim",  cfg.INPUT_DIM)
    latent_dim = ckpt.get("latent_dim", cfg.LATENT_DIM)

    _orig = cfg.LATENT_DIM
    cfg.LATENT_DIM = latent_dim
    model = BetaVAE(input_dim=input_dim).to(device)
    cfg.LATENT_DIM = _orig

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_t = torch.from_numpy(X.astype(np.float32)).to(device)
    with torch.no_grad():
        mu, _ = model.encode(x_t)
    return mu.cpu().numpy()


def plot_latent_tsne(
    mu:           np.ndarray,
    y_true:       np.ndarray,
    out_path:     str | Path,
    n_samples:    int = 5_000,
    random_state: int = 42,
) -> None:
    """
    T-SNE of the VAE latent space (μ vectors) coloured by class.
    Clear cluster separation → encoder has learned fraud-discriminative structure.
    Stratified subsample keeps fraud points visible despite class imbalance.
    """
    from sklearn.manifold import TSNE

    rng        = np.random.default_rng(random_state)
    idx_normal = np.where(y_true == 0)[0]
    idx_fraud  = np.where(y_true == 1)[0]
    n_fraud    = min(len(idx_fraud),  n_samples // 2)
    n_normal   = min(len(idx_normal), n_samples - n_fraud)

    idx = np.concatenate([
        rng.choice(idx_normal, n_normal, replace=False),
        rng.choice(idx_fraud,  n_fraud,  replace=False),
    ])
    rng.shuffle(idx)

    mu_sub = mu[idx]
    y_sub  = y_true[idx]

    log.info("T-SNE on %d points (latent_dim=%d) …", len(idx), mu.shape[1])
    z2d = TSNE(
        n_components=2, perplexity=30, n_iter=1_000,
        random_state=random_state,
    ).fit_transform(mu_sub)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls, color, label in [
        (0, "steelblue", f"Normal  n={int((y_sub==0).sum()):,}"),
        (1, "crimson",   f"Fraud   n={int((y_sub==1).sum()):,}"),
    ]:
        mask = y_sub == cls
        ax.scatter(
            z2d[mask, 0], z2d[mask, 1],
            c=color, alpha=0.45 if cls == 0 else 0.8,
            s=6 if cls == 0 else 25,
            label=label, linewidths=0,
        )

    ax.set_title("Latent Space T-SNE — VAE Encoder (μ)\n"
                 "Clear separation → encoder learned fraud-discriminative structure",
                 fontsize=12)
    ax.set_xlabel("T-SNE dim 1")
    ax.set_ylabel("T-SNE dim 2")
    ax.legend(markerscale=3, fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out_path)


# ── Amount vs score scatter ───────────────────────────────────────────────────

def plot_score_vs_amount(
    scores:          np.ndarray,
    y_true:          np.ndarray,
    amount_scaled:   np.ndarray,
    out_path:        str | Path,
    threshold:       float | None = None,
    threshold_label: str   = "Threshold",
) -> None:
    """
    Scatter: normalised Amount (x) vs anomaly score (y), coloured by class.
    Diagnoses amount-bias: do low-value frauds escape the model?
    """
    clip_score = float(np.percentile(scores, 99))

    fig, ax = plt.subplots(figsize=(10, 6))
    for cls, color, alpha, s, lbl in [
        (0, "steelblue", 0.25,  6, "Normal"),
        (1, "crimson",   0.80, 25, "Fraud"),
    ]:
        mask = y_true == cls
        ax.scatter(
            amount_scaled[mask],
            scores[mask].clip(max=clip_score),
            c=color, alpha=alpha, s=s,
            label=f"{lbl}  n={mask.sum():,}", linewidths=0,
        )

    if threshold is not None:
        ax.axhline(
            threshold, color="darkorange", linestyle="--",
            linewidth=1.8, label=f"{threshold_label} = {threshold:.4f}",
        )

    ax.set_xlabel("Amount  (StandardScaler-normalised)")
    ax.set_ylabel(f"Anomaly score  (clipped at 99th pct ≈ {clip_score:.3f})")
    ax.set_title(
        "Anomaly Score vs Transaction Amount\n"
        "Points below threshold are passed through — check if small-amount fraud escapes"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out_path)


# ── Visual confusion matrix ───────────────────────────────────────────────────

def plot_confusion_matrix_heatmap(
    scores:    np.ndarray,
    y_true:    np.ndarray,
    threshold: float,
    out_path:  str | Path,
    label:     str   = "Threshold",
    cost_fn:   float = 1_000.0,
    cost_fp:   float =   100.0,
) -> None:
    """
    2×2 visual confusion matrix heatmap.
    Each cell annotated with count, role (TP/FP/…), and monetary cost.
    """
    y_pred = (scores >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    matrix  = np.array([[tn, fp], [fn, tp]], dtype=float)
    bg_norm = matrix / (matrix.max() + 1e-9)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(bg_norm, cmap="Blues", vmin=0, vmax=1.2)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Normal", "Predicted: Fraud"], fontsize=11)
    ax.set_yticklabels(["Actual: Normal", "Actual: Fraud"],       fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label",    fontsize=12)

    cell_text = [
        [f"TN\n{tn:,}\nCorrectly allowed",
         f"FP\n{fp:,}\nWrongly blocked\n−{fp*cost_fp:,.0f} ฿"],
        [f"FN\n{fn:,}\nFraud missed\n−{fn*cost_fn:,.0f} ฿",
         f"TP\n{tp:,}\nFraud caught"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cell_text[i][j],
                    ha="center", va="center", fontsize=10, fontweight="bold")

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    cost = fp * cost_fp + fn * cost_fn

    ax.set_title(
        f"Confusion Matrix — {label}  (threshold = {threshold:.4f})\n"
        f"Precision = {prec:.3f}   Recall = {rec:.3f}   F1 = {f1:.3f}   "
        f"Total cost = {cost:,.0f} ฿",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, label="Relative count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out_path)


# ── End-to-end runner ─────────────────────────────────────────────────────────

def run_evaluation(
    exp_dir:  str | Path,
    split:    str   = "val",
    cost_fn:  float = 1_000.0,
    cost_fp:  float =   100.0,
) -> dict:
    """
    Full threshold-engineering pipeline for a completed experiment.

    Args:
        exp_dir : path to an experiment folder (exp_01/, exp_02/, …)
        split   : 'val' or 'test'
        cost_fn : cost of missing a fraud (false negative)
        cost_fp : cost of blocking a legit customer (false positive)

    Returns:
        dict with optimal thresholds and per-threshold metrics at each.
    """
    import src.config as cfg
    from src.train import _build_feature_weights, _FEATURE_COLS

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    exp_dir  = Path(exp_dir)
    proc_dir = Path(cfg.DATA_PROC_DIR)
    ckpt     = exp_dir / "best_model.pth"

    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    # ── Load split ────────────────────────────────────────────────────────────
    X      = np.load(proc_dir / f"X_{split}.npy").astype(np.float32)
    y_true = np.load(proc_dir / f"y_{split}.npy").astype(np.int32)
    log.info("Loaded %s  shape=%s  fraud_rate=%.2f%%", split, X.shape, y_true.mean()*100)

    # ── Feature weights ───────────────────────────────────────────────────────
    feat_w = _build_feature_weights(device)

    # ── Anomaly scores ────────────────────────────────────────────────────────
    scores = compute_anomaly_scores(ckpt, X, device, feature_weights=feat_w)

    log.info("AUROC=%.4f  AUPRC=%.4f",
             roc_auc_score(y_true, scores),
             average_precision_score(y_true, scores))

    # ── Threshold sweep ───────────────────────────────────────────────────────
    rows = threshold_analysis(scores, y_true, n_points=500,
                              cost_fn=cost_fn, cost_fp=cost_fp)
    opt  = find_optimal_thresholds(rows)

    # ── Print reports at each optimal threshold ───────────────────────────────
    results = {}
    for name, t in [
        ("Max F1",      opt["max_f1"]),
        ("Min Cost",    opt["min_cost"]),
        ("Recall ≥ 90%", opt["recall90"]),
    ]:
        if t >= 0:
            results[name] = evaluate_at_threshold(
                scores, y_true, t, cost_fn=cost_fn, cost_fp=cost_fp, label=name
            )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_path = exp_dir / f"threshold_curves_{split}.png"
    plot_threshold_curves(rows, opt, scores, y_true, plot_path,
                          cost_fn=cost_fn, cost_fp=cost_fp)

    # ── Save thresholds ───────────────────────────────────────────────────────
    out = {
        "split"              : split,
        "cost_fn"            : cost_fn,
        "cost_fp"            : cost_fp,
        "optimal_thresholds" : opt,
        "reports"            : results,
    }
    json_path = exp_dir / f"thresholds_{split}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Thresholds saved → %s", json_path)

    return out


if __name__ == "__main__":
    import argparse, sys
    from src.config import EXPERIMENTS_DIR

    parser = argparse.ArgumentParser(description="Threshold engineering for fraud VAE")
    parser.add_argument("--exp",      default=None,  help="Experiment folder (default: latest)")
    parser.add_argument("--split",    default="val",  choices=["val", "test"])
    parser.add_argument("--cost-fn",  type=float, default=1_000.0)
    parser.add_argument("--cost-fp",  type=float, default=100.0)
    args = parser.parse_args()

    if args.exp:
        exp_dir = Path(args.exp)
    else:
        # Auto-select the latest experiment
        exp_base = Path(EXPERIMENTS_DIR)
        dirs = sorted(d for d in exp_base.iterdir()
                      if d.is_dir() and d.name.startswith("exp_"))
        if not dirs:
            print("No experiment directories found.", file=sys.stderr)
            sys.exit(1)
        exp_dir = dirs[-1]
        print(f"Using latest experiment: {exp_dir}")

    run_evaluation(exp_dir, split=args.split,
                   cost_fn=args.cost_fn, cost_fp=args.cost_fp)
