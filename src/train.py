"""
VAE Training Loop — Credit Card Fraud Detection (Semi-Supervised Anomaly Detection).

Key design decisions vs. standard VAE training:
  - Early stopping & checkpointing on Val AUPRC, not Val Loss.
    AUPRC is the correct metric for extreme class imbalance (0.17% fraud).
    A model can achieve low reconstruction loss while completely failing to
    separate fraud from normal — AUPRC catches this, Val Loss does not.
  - Validation uses the FULL val set (normal + fraud) to compute anomaly scores.
    Training uses ONLY normal samples, so the model never sees fraud labels.
  - Per-sample reconstruction error (mean over 30 features) is the anomaly score.
    Higher error → more anomalous → flagged as potential fraud.
"""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive — works on Colab & headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    BATCH_SIZE,
    BETA,
    DATA_PROC_DIR,
    EXPERIMENTS_DIR,
    FEATURE_WEIGHTS,
    INPUT_DIM,
    KL_ANNEAL_EPOCHS,
    LATENT_DIM,
    LEARNING_RATE,
    LR_PATIENCE,
    NOISE_STD,
    NUM_EPOCHS,
    PATIENCE,
    RANDOM_SEED,
)
from src.model import BetaVAE, vae_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_ROOT          = Path(__file__).resolve().parents[1]
MODELS_DIR     = _ROOT / "models"
CANONICAL_CKPT = MODELS_DIR / "best_vae.pth"

# All tunable constants come from src/config.py — edit them there, not here.
# PATIENCE         → early stopping on Val AUPRC
# LR_PATIENCE      → ReduceLROnPlateau patience
# KL_ANNEAL_EPOCHS → β warm-up length (epochs 1 → KL_ANNEAL_EPOCHS)
# NOISE_STD        → denoising Gaussian σ (0 = disabled)

# Column order must match FEATURE_COLS in preprocess.py
_FEATURE_COLS: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _build_feature_weights(device: torch.device) -> torch.Tensor:
    """
    Build a (INPUT_DIM,) float32 weight tensor from FEATURE_WEIGHTS config.
    Features absent from the dict default to 1.0.
    Constructed once before training and moved to device.
    """
    weights = torch.ones(INPUT_DIM, dtype=torch.float32)
    for feat, w in FEATURE_WEIGHTS.items():
        if feat in _FEATURE_COLS:
            weights[_FEATURE_COLS.index(feat)] = w
        else:
            log.warning("FEATURE_WEIGHTS key '%s' not in feature columns — ignored", feat)
    log.info(
        "Feature weights  non-unit=%d  max=%.1f  weighted_features=%s",
        int((weights != 1.0).sum()),
        weights.max().item(),
        [f for f, w in FEATURE_WEIGHTS.items() if f in _FEATURE_COLS],
    )
    return weights.to(device)


def _build_noise_sigma(
    feature_weights: torch.Tensor,
    base_std: float,
) -> torch.Tensor | None:
    """
    Per-feature noise std, inversely proportional to reconstruction weight.

    Formula:  σ_d = base_std / (w_d / mean(w))

    High-weight features (large w_d, strong fraud signal) → small σ_d.
      → Their discriminative structure is preserved during training.
    Low-weight features (w_d ≈ 1.0) → σ_d ≈ base_std.
      → Model is forced to learn their underlying structure.

    Mean noise across features ≈ base_std (conservation property).
    Returns None when base_std == 0 (denoising disabled).
    """
    if base_std == 0.0:
        return None
    w_norm   = feature_weights / feature_weights.mean()   # normalise weights
    sigma    = base_std / w_norm                          # invert: high-w → low σ
    log.info(
        "Noise sigma  base_std=%.3f  min_sigma=%.4f (high-weight feats)  "
        "max_sigma=%.4f (low-weight feats)",
        base_std, sigma.min().item(), sigma.max().item(),
    )
    return sigma


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log.info("Device: CPU")
    return device


# ── Experiment directory ──────────────────────────────────────────────────────

def _next_exp_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        d for d in base.iterdir()
        if d.is_dir() and d.name.startswith("exp_")
    )
    n = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    exp_dir = base / f"exp_{n:02d}"
    exp_dir.mkdir()
    return exp_dir


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(proc_dir: str | Path) -> tuple[DataLoader, torch.Tensor, np.ndarray]:
    """
    Returns:
        train_loader  — batched X_train (normal only), for gradient updates
        X_val_tensor  — full val set as a single tensor, for anomaly scoring
        y_val         — numpy int array of ground-truth labels (0/1)
    """
    proc = Path(proc_dir)

    X_train = np.load(proc / "X_train.npy").astype(np.float32)
    X_val   = np.load(proc / "X_val.npy").astype(np.float32)
    y_val   = np.load(proc / "y_val.npy").astype(np.int32)

    log.info(
        "Data  train(normal)=%d  val_total=%d  val_fraud=%d  input_dim=%d",
        len(X_train), len(X_val), int(y_val.sum()), X_train.shape[1],
    )

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        drop_last=True,
        num_workers=0,      # 0 avoids Colab/multiprocessing deadlocks
        pin_memory=True,
    )
    X_val_tensor = torch.from_numpy(X_val)

    return train_loader, X_val_tensor, y_val


# ── Training epoch ────────────────────────────────────────────────────────────

def _train_epoch(
    model:           BetaVAE,
    loader:          DataLoader,
    optimizer:       optim.Optimizer,
    device:          torch.device,
    beta:            float = BETA,
    feature_weights: torch.Tensor | None = None,
    noise_sigma:     torch.Tensor | None = None,
) -> float:
    """
    One gradient-update pass over the training set.

    Denoising VAE: if noise_sigma is not None, feature-specific Gaussian
    noise is added to x before encoding.  High-weight (discriminative)
    features receive less noise; low-weight features receive more.
    The reconstruction target always remains the original clean x.

    Returns mean total loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for (x,) in loader:
        x = x.to(device, non_blocking=True)

        # ── Feature-specific denoising: corrupt input, reconstruct clean ───
        if noise_sigma is not None:
            # noise_sigma: (INPUT_DIM,) — broadcast over batch
            x_in = x + torch.randn_like(x) * noise_sigma.unsqueeze(0)
        else:
            x_in = x

        optimizer.zero_grad()
        x_hat, mu, log_var = model(x_in)               # encode noisy
        loss, _, _ = vae_loss(x, x_hat, mu, log_var,   # target = clean x
                              beta=beta, feature_weights=feature_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# ── Validation epoch ──────────────────────────────────────────────────────────

def _val_epoch(
    model:           BetaVAE,
    X_val_tensor:    torch.Tensor,
    y_val:           np.ndarray,
    device:          torch.device,
    feature_weights: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """
    Full-val pass in eval mode.

    Returns:
        val_loss  — weighted β-ELBO loss over normal+fraud samples
        auroc     — roc_auc_score(y_val, anomaly_scores)
        auprc     — average_precision_score(y_val, anomaly_scores)

    Anomaly score = per-sample weighted reconstruction error (same weights as
    training loss), so the score directly reflects the loss landscape.
    """
    model.eval()
    with torch.no_grad():
        X_val = X_val_tensor.to(device)
        x_hat, mu, log_var = model(X_val)

        loss, _, _ = vae_loss(X_val, x_hat, mu, log_var,
                              feature_weights=feature_weights)

        # Weighted per-sample anomaly score — mirrors the training objective
        sq_err = (X_val - x_hat) ** 2
        if feature_weights is not None:
            sq_err = sq_err * feature_weights.unsqueeze(0)
        anomaly_scores = sq_err.mean(dim=1).cpu().numpy()

    auroc = float(roc_auc_score(y_val, anomaly_scores))
    auprc = float(average_precision_score(y_val, anomaly_scores))

    return float(loss.item()), auroc, auprc


# ── Post-training plots ───────────────────────────────────────────────────────

def _plot_training_curves(exp_dir: Path, log_csv: Path) -> None:
    """3-panel figure: Loss / AUROC / AUPRC saved to exp_dir/training_curves.png."""
    rows: list[dict] = []
    with open(log_csv) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    if not rows:
        return

    epochs   = [r["epoch"]    for r in rows]
    best_ep  = max(rows, key=lambda r: r["val_auprc"])["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    ax = axes[0]
    ax.plot(epochs, [r["train_loss"] for r in rows], color="steelblue", label="Train")
    ax.plot(epochs, [r["val_loss"]   for r in rows], color="crimson", linestyle="--", label="Val")
    ax.axvline(best_ep, color="orange", linewidth=1.2, linestyle=":", label=f"best ep {int(best_ep)}")
    ax.set_xlabel("Epoch"); ax.set_title("β-ELBO Loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # AUROC
    ax = axes[1]
    ax.plot(epochs, [r["val_auroc"] for r in rows], color="steelblue", label="Val AUROC")
    ax.axvline(best_ep, color="orange", linewidth=1.2, linestyle=":")
    ax.set_xlabel("Epoch"); ax.set_title("Val AUROC"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # AUPRC
    ax = axes[2]
    ax.plot(epochs, [r["val_auprc"] for r in rows], color="crimson", label="Val AUPRC")
    ax.axvline(best_ep, color="orange", linewidth=1.2, linestyle=":", label=f"best ep {int(best_ep)}")
    ax.set_xlabel("Epoch"); ax.set_title("Val AUPRC  (early-stopping metric)"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    best = max(rows, key=lambda r: r["val_auprc"])
    plt.suptitle(
        f"β-VAE Training  (β={BETA}  anneal={KL_ANNEAL_EPOCHS}ep  best_ep={int(best_ep)}"
        f"  AUPRC={best['val_auprc']:.4f}  AUROC={best['val_auroc']:.4f})",
        fontsize=11,
    )
    plt.tight_layout()
    out = exp_dir / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out)


def _plot_anomaly_scores(
    exp_dir:         Path,
    model:           BetaVAE,
    X_val_tensor:    torch.Tensor,
    y_val:           np.ndarray,
    device:          torch.device,
    feature_weights: torch.Tensor | None = None,
) -> None:
    """Weighted reconstruction error histogram (Normal vs Fraud) saved to exp_dir/anomaly_scores.png."""
    model.eval()
    with torch.no_grad():
        X_val  = X_val_tensor.to(device)
        x_hat, _, _ = model(X_val)
        sq_err = (X_val - x_hat) ** 2
        if feature_weights is not None:
            sq_err = sq_err * feature_weights.unsqueeze(0)
        scores = sq_err.mean(dim=1).cpu().numpy()

    normal_s = scores[y_val == 0]
    fraud_s  = scores[y_val == 1]
    sep      = fraud_s.mean() / (normal_s.mean() + 1e-9)
    auprc    = float(average_precision_score(y_val, scores))
    auroc    = float(roc_auc_score(y_val, scores))

    log.info(
        "Anomaly scores  normal_mean=%.4f  fraud_mean=%.4f  sep=%.2f×  AUPRC=%.4f",
        normal_s.mean(), fraud_s.mean(), sep, auprc,
    )

    clip = float(np.percentile(scores, 99))
    bins = np.linspace(0, clip, 80)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(normal_s.clip(max=clip), bins=bins, alpha=0.6, density=True,
            color="steelblue", label=f"Normal  (n={len(normal_s):,})")
    ax.hist(fraud_s.clip(max=clip),  bins=bins, alpha=0.6, density=True,
            color="crimson",   label=f"Fraud   (n={len(fraud_s):,})")
    ax.set_xlabel("Reconstruction Error (anomaly score)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Anomaly Score Distribution — Val Set\n"
        f"Separation={sep:.2f}×  |  AUROC={auroc:.4f}  |  AUPRC={auprc:.4f}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = exp_dir / "anomaly_scores.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out)


# ── Main training loop ────────────────────────────────────────────────────────

def train() -> Path:
    """
    Run one full training experiment.
    Returns the experiment directory Path for downstream Colab cells.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    exp_dir = _next_exp_dir(Path(EXPERIMENTS_DIR))
    shutil.copy(Path(__file__).parent / "config.py", exp_dir / "config_backup.py")
    log.info("Experiment: %s", exp_dir)

    device = get_device()
    train_loader, X_val_tensor, y_val = load_data(DATA_PROC_DIR)

    model     = BetaVAE(input_dim=INPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",   # maximise AUPRC
        factor=0.5, patience=LR_PATIENCE,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "BetaVAE  input_dim=%d  latent_dim=%d  β=%.4f  anneal_epochs=%d  params=%d",
        INPUT_DIM, LATENT_DIM, BETA, KL_ANNEAL_EPOCHS, n_params,
    )

    feature_weights = _build_feature_weights(device)
    noise_sigma     = _build_noise_sigma(feature_weights, NOISE_STD)

    # ── CSV history ───────────────────────────────────────────────────────────
    log_csv  = exp_dir / "loss_history.csv"
    csv_file = open(log_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_auroc", "val_auprc", "lr", "beta"])

    exp_ckpt          = exp_dir / "best_model.pth"
    best_auprc        = -1.0
    patience_counter  = 0

    log.info("=" * 65)
    log.info(
        "Training  epochs=%d  patience=%d  lr_patience=%d  "
        "kl_anneal=%dep  noise_std=%.3f  metric=AUPRC",
        NUM_EPOCHS, PATIENCE, LR_PATIENCE, KL_ANNEAL_EPOCHS, NOISE_STD,
    )
    log.info("=" * 65)

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            # KL annealing: linearly ramp β from 0 → BETA over KL_ANNEAL_EPOCHS
            # Prevents posterior collapse early in training when the encoder
            # hasn't learned meaningful structure yet.
            annealed_beta           = min(BETA, BETA * epoch / KL_ANNEAL_EPOCHS)
            train_loss              = _train_epoch(model, train_loader, optimizer, device,
                                                  beta=annealed_beta,
                                                  feature_weights=feature_weights,
                                                  noise_sigma=noise_sigma)
            val_loss, auroc, auprc  = _val_epoch(model, X_val_tensor, y_val, device,
                                                 feature_weights=feature_weights)
            current_lr              = optimizer.param_groups[0]["lr"]
            scheduler.step(auprc)

            print(
                f"Epoch [{epoch:>3}/{NUM_EPOCHS}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUROC: {auroc:.4f} | "
                f"Val AUPRC: {auprc:.4f} | "
                f"β: {annealed_beta:.4f}"
            )

            csv_writer.writerow([
                epoch,
                f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{auroc:.6f}",      f"{auprc:.6f}",
                f"{current_lr:.2e}", f"{annealed_beta:.6f}",
            ])
            csv_file.flush()

            # ── Checkpoint on AUPRC improvement ──────────────────────────────
            if auprc > best_auprc:
                best_auprc       = auprc
                patience_counter = 0
                ckpt = {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_auprc":   best_auprc,
                    "val_auroc":   auroc,
                    "input_dim":   INPUT_DIM,
                    "latent_dim":  LATENT_DIM,
                    "beta":        BETA,
                }
                torch.save(ckpt, exp_ckpt)
                shutil.copy(exp_ckpt, CANONICAL_CKPT)
                log.info("  [+] AUPRC improved → %.4f  checkpoint saved", best_auprc)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    log.info("Early stopping at epoch %d  (best AUPRC=%.4f)", epoch, best_auprc)
                    break

    finally:
        csv_file.close()

    # ── Post-training plots ───────────────────────────────────────────────────
    log.info("Generating plots...")
    _plot_training_curves(exp_dir, log_csv)

    # Reload best checkpoint so plots use the best weights, not last epoch
    best_ckpt = torch.load(exp_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state"])
    _plot_anomaly_scores(exp_dir, model, X_val_tensor, y_val, device,
                        feature_weights=feature_weights)

    log.info("=" * 65)
    log.info("Done.  best_val_auprc=%.4f  exp=%s", best_auprc, exp_dir.name)
    log.info("  Artefacts in %s:", exp_dir)
    log.info("    config_backup.py  best_model.pth  loss_history.csv")
    log.info("    training_curves.png  anomaly_scores.png")
    log.info("=" * 65)

    return exp_dir


if __name__ == "__main__":
    train()
