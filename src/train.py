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
    INPUT_DIM,
    LATENT_DIM,
    LEARNING_RATE,
    LR_PATIENCE,
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

# Patience values come from src/config.py — edit them there, not here.
# PATIENCE     → early stopping on Val AUPRC (imported as PATIENCE)
# LR_PATIENCE  → ReduceLROnPlateau patience  (imported as LR_PATIENCE)
KL_ANNEAL_EPOCHS: int = 10   # ramp β from 0 → BETA over first N epochs


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
    model:     BetaVAE,
    loader:    DataLoader,
    optimizer: optim.Optimizer,
    device:    torch.device,
    beta:      float = BETA,
) -> float:
    """One gradient-update pass over the training set. Returns mean total loss."""
    model.train()
    total_loss = 0.0

    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad()
        x_hat, mu, log_var = model(x)
        loss, _, _ = vae_loss(x, x_hat, mu, log_var, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# ── Validation epoch ──────────────────────────────────────────────────────────

def _val_epoch(
    model:        BetaVAE,
    X_val_tensor: torch.Tensor,
    y_val:        np.ndarray,
    device:       torch.device,
) -> tuple[float, float, float]:
    """
    Full-val pass in eval mode.

    Returns:
        val_loss  — mean β-ELBO loss over normal+fraud samples
        auroc     — roc_auc_score(y_val, anomaly_scores)
        auprc     — average_precision_score(y_val, anomaly_scores)

    Anomaly score = per-sample mean squared reconstruction error.
    Higher score → more anomalous → predicted fraud.
    Computed in eval mode so reparameterise() returns μ (deterministic).
    """
    model.eval()
    with torch.no_grad():
        X_val = X_val_tensor.to(device)
        x_hat, mu, log_var = model(X_val)

        # β-ELBO on full val set (monitoring purposes only — not used for ES)
        loss, _, _ = vae_loss(X_val, x_hat, mu, log_var)

        # Per-sample reconstruction error — the anomaly score
        # Shape: (N,) — mean over INPUT_DIM features per sample
        anomaly_scores = ((X_val - x_hat) ** 2).mean(dim=1).cpu().numpy()

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
    exp_dir:      Path,
    model:        BetaVAE,
    X_val_tensor: torch.Tensor,
    y_val:        np.ndarray,
    device:       torch.device,
) -> None:
    """Reconstruction error histogram (Normal vs Fraud) saved to exp_dir/anomaly_scores.png."""
    model.eval()
    with torch.no_grad():
        X_val  = X_val_tensor.to(device)
        x_hat, _, _ = model(X_val)
        scores = ((X_val - x_hat) ** 2).mean(dim=1).cpu().numpy()

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

    # ── CSV history ───────────────────────────────────────────────────────────
    log_csv  = exp_dir / "loss_history.csv"
    csv_file = open(log_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_auroc", "val_auprc", "lr", "beta"])

    exp_ckpt          = exp_dir / "best_model.pth"
    best_auprc        = -1.0
    patience_counter  = 0

    log.info("=" * 65)
    log.info("Training  epochs=%d  patience=%d  lr_patience=%d  metric=AUPRC",
             NUM_EPOCHS, PATIENCE, LR_PATIENCE)
    log.info("=" * 65)

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            # KL annealing: linearly ramp β from 0 → BETA over KL_ANNEAL_EPOCHS
            # Prevents posterior collapse early in training when the encoder
            # hasn't learned meaningful structure yet.
            annealed_beta           = min(BETA, BETA * epoch / KL_ANNEAL_EPOCHS)
            train_loss              = _train_epoch(model, train_loader, optimizer, device, beta=annealed_beta)
            val_loss, auroc, auprc  = _val_epoch(model, X_val_tensor, y_val, device)
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
    _plot_anomaly_scores(exp_dir, model, X_val_tensor, y_val, device)

    log.info("=" * 65)
    log.info("Done.  best_val_auprc=%.4f  exp=%s", best_auprc, exp_dir.name)
    log.info("  Artefacts in %s:", exp_dir)
    log.info("    config_backup.py  best_model.pth  loss_history.csv")
    log.info("    training_curves.png  anomaly_scores.png")
    log.info("=" * 65)

    return exp_dir


if __name__ == "__main__":
    train()
