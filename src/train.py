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
    NUM_EPOCHS,
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

EARLY_STOP_PATIENCE:   int = 10
LR_SCHEDULER_PATIENCE: int = 5


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
        num_workers=2,
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
) -> float:
    """One gradient-update pass over the training set. Returns mean total loss."""
    model.train()
    total_loss = 0.0

    for (x,) in loader:
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad()
        x_hat, mu, log_var = model(x)
        loss, _, _ = vae_loss(x, x_hat, mu, log_var)
        loss.backward()
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
        factor=0.5, patience=LR_SCHEDULER_PATIENCE,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "BetaVAE  input_dim=%d  latent_dim=%d  β=%.4f  params=%d",
        INPUT_DIM, LATENT_DIM, BETA, n_params,
    )

    # ── CSV history ───────────────────────────────────────────────────────────
    log_csv  = exp_dir / "loss_history.csv"
    csv_file = open(log_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_auroc", "val_auprc", "lr"])

    exp_ckpt          = exp_dir / "best_model.pth"
    best_auprc        = -1.0
    patience_counter  = 0

    log.info("=" * 65)
    log.info("Training  epochs=%d  patience=%d  metric=AUPRC",
             NUM_EPOCHS, EARLY_STOP_PATIENCE)
    log.info("=" * 65)

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss              = _train_epoch(model, train_loader, optimizer, device)
            val_loss, auroc, auprc  = _val_epoch(model, X_val_tensor, y_val, device)
            current_lr              = optimizer.param_groups[0]["lr"]
            scheduler.step(auprc)

            print(
                f"Epoch [{epoch:>3}/{NUM_EPOCHS}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUROC: {auroc:.4f} | "
                f"Val AUPRC: {auprc:.4f}"
            )

            csv_writer.writerow([
                epoch,
                f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{auroc:.6f}",      f"{auprc:.6f}",
                f"{current_lr:.2e}",
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
                if patience_counter >= EARLY_STOP_PATIENCE:
                    log.info("Early stopping at epoch %d  (best AUPRC=%.4f)", epoch, best_auprc)
                    break

    finally:
        csv_file.close()

    log.info("=" * 65)
    log.info("Done.  best_val_auprc=%.4f  exp=%s", best_auprc, exp_dir.name)
    log.info("=" * 65)

    return exp_dir


if __name__ == "__main__":
    train()
