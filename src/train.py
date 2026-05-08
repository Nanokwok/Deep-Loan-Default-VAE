"""
Phase 3 — VAE Training with Experiment Tracking

Each run creates an auto-incremented experiment directory:

  experiments/
    exp_01/
      config_backup.py        ← exact copy of src/config.py used for this run
      best_model.pth          ← best checkpoint (lowest val normal-only loss)
      loss_history.csv        ← per-epoch train/val losses (flushed every epoch)
      loss_curves.png         ← 3-panel plot: Total / Recon / KL
      reconstruction_plot.png ← anomaly score histogram: Normal vs Charged Off
    exp_02/
      ...

models/best_vae.pth is also updated after every run (symlink-style copy)
so downstream scripts (Phase 4) always find the latest best model there.

To redirect experiments to Google Drive in Colab (persists across VM resets):
    import src.config as cfg
    cfg.EXPERIMENTS_DIR = '/content/drive/MyDrive/Loan_VAE_Project'
  Then run:
    from src.train import train; train()
"""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on Colab & headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    BATCH_SIZE,
    BETA,
    DATA_PROC_DIR,
    EXPERIMENTS_DIR,
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

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT           = Path(__file__).resolve().parents[1]
MODELS_DIR      = _ROOT / "models"
CANONICAL_CKPT  = MODELS_DIR / "best_vae.pth"   # always points to latest best

# ── Training constants ────────────────────────────────────────────────────────
EARLY_STOP_PATIENCE:  int = 10
LR_SCHEDULER_PATIENCE: int = 3


# ── Experiment directory management ──────────────────────────────────────────

def _next_exp_dir(base: Path) -> Path:
    """
    Return the next auto-incremented experiment directory.

    Scans base/ for existing exp_NN folders and returns exp_(N+1).
    Creates the directory immediately so concurrent runs don't collide.
    """
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        d for d in base.iterdir()
        if d.is_dir() and d.name.startswith("exp_")
    )
    n = int(existing[-1].name.split("_")[1]) + 1 if existing else 1
    exp_dir = base / f"exp_{n:02d}"
    exp_dir.mkdir()
    return exp_dir


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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(
    proc_dir: str | Path,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Returns (train_loader, val_normal_loader, input_dim).

    val_loader contains NORMAL rows only — using anomaly rows for early
    stopping would penalise the model for correctly ignoring anomalies.
    """
    proc = Path(proc_dir)

    train_arr  = np.load(proc / "train_features.npy").astype(np.float32)
    val_arr    = np.load(proc / "val_features.npy").astype(np.float32)
    val_labels = np.load(proc / "val_labels.npy")
    val_normal = val_arr[val_labels == 0]

    log.info(
        "Data  train=%d  val_total=%d  val_normal=%d  input_dim=%d",
        len(train_arr), len(val_arr), len(val_normal), train_arr.shape[1],
    )

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_arr)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_normal)),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader, train_arr.shape[1]


# ── Epoch passes ──────────────────────────────────────────────────────────────

def _run_epoch(
    model:     BetaVAE,
    loader:    DataLoader,
    device:    torch.device,
    optimizer: optim.Optimizer | None,
) -> tuple[float, float, float]:
    """
    One forward pass over loader.  optimizer=None → eval mode, no gradients.
    Returns per-sample averages of (total, recon, kl).
    """
    is_train = optimizer is not None
    model.train(is_train)
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    sum_total = sum_recon = sum_kl = 0.0
    with ctx:
        for (x,) in loader:
            x = x.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad()
            x_hat, mu, log_var   = model(x)
            total, recon, kl     = vae_loss(x, x_hat, mu, log_var)
            if is_train:
                total.backward()
                optimizer.step()
            n = x.size(0)
            sum_total += total.item() * n
            sum_recon += recon.item() * n
            sum_kl    += kl.item()    * n

    n_total = len(loader.dataset)
    return sum_total / n_total, sum_recon / n_total, sum_kl / n_total


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_loss_curves(exp_dir: Path, log_csv: Path) -> None:
    """3-panel loss curve saved to exp_dir/loss_curves.png."""
    df_rows: list[dict] = []
    with open(log_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            df_rows.append({k: float(v) for k, v in row.items()})

    if not df_rows:
        return

    epochs    = [r["epoch"]       for r in df_rows]
    best_ep   = min(df_rows, key=lambda r: r["val_total"])["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [
        ("train_total", "val_total",  "Total Loss  (recon + β·KL)"),
        ("train_recon", "val_recon",  "Reconstruction Loss (MSE)"),
        ("train_kl",    "val_kl",     "KL Divergence"),
    ]
    for ax, (tr_key, vl_key, title) in zip(axes, pairs):
        ax.plot(epochs, [r[tr_key] for r in df_rows],
                color="steelblue", label="train")
        ax.plot(epochs, [r[vl_key] for r in df_rows],
                color="crimson", linestyle="--", label="val (normal)")
        ax.axvline(best_ep, color="orange", linewidth=1.2,
                   linestyle=":", label=f"best ep {int(best_ep)}")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    best_val = min(r["val_total"] for r in df_rows)
    plt.suptitle(
        f"β-VAE Training  (β={BETA}  best_val={best_val:.4f}  ep={int(best_ep)})",
        fontsize=12,
    )
    plt.tight_layout()
    out = exp_dir / "loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out)


def _plot_reconstruction(
    exp_dir:  Path,
    model:    BetaVAE,
    device:   torch.device,
    proc_dir: str | Path,
) -> None:
    """
    Anomaly score histogram saved to exp_dir/reconstruction_plot.png.

    Loads val_features + val_labels, runs inference with best model (eval,
    deterministic z=μ), plots per-sample MSE distribution split by label.
    """
    proc = Path(proc_dir)
    val_t   = torch.from_numpy(
        np.load(proc / "val_features.npy").astype(np.float32)
    ).to(device)
    val_lbl = np.load(proc / "val_labels.npy")

    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(val_t)
        errors = ((val_t - x_hat) ** 2).mean(dim=1).cpu().numpy()

    normal_e  = errors[val_lbl == 0]
    anomaly_e = errors[val_lbl == 1]

    sep = anomaly_e.mean() / (normal_e.mean() + 1e-9)
    log.info(
        "Anomaly score  normal_mean=%.4f  anomaly_mean=%.4f  separation=%.2f×",
        normal_e.mean(), anomaly_e.mean(), sep,
    )

    clip     = float(np.percentile(errors, 99))
    bins     = np.linspace(0, clip, 80)
    fig, ax  = plt.subplots(figsize=(10, 4))
    ax.hist(normal_e.clip(max=clip),  bins=bins, alpha=0.6,
            density=True, color="steelblue", label=f"Fully Paid (n={len(normal_e):,})")
    ax.hist(anomaly_e.clip(max=clip), bins=bins, alpha=0.6,
            density=True, color="crimson",   label=f"Charged Off (n={len(anomaly_e):,})")
    ax.set_xlabel("Reconstruction Error (mean squared per feature)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Anomaly Score Distribution — Val Set  "
        f"(separation {sep:.2f}×,  β={BETA})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = exp_dir / "reconstruction_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", out)


# ── Main training loop ────────────────────────────────────────────────────────

def train() -> Path:
    """
    Run one full training experiment.

    Returns the experiment directory path (e.g. experiments/exp_03/)
    so Colab cells can display / copy artefacts without hard-coding paths.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── Create experiment directory first ─────────────────────────────────────
    exp_dir = _next_exp_dir(Path(EXPERIMENTS_DIR))
    log.info("Experiment dir: %s", exp_dir)

    # Save config snapshot immediately — captures exactly what was used,
    # even if training is interrupted before completion.
    config_src = Path(__file__).parent / "config.py"
    shutil.copy(config_src, exp_dir / "config_backup.py")
    log.info("Config backed up → %s/config_backup.py", exp_dir.name)

    # ── Setup ─────────────────────────────────────────────────────────────────
    device                          = get_device()
    train_loader, val_loader, input_dim = load_data(DATA_PROC_DIR)

    model     = BetaVAE(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_SCHEDULER_PATIENCE,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "BetaVAE  input_dim=%d  latent_dim=%d  β=%.3f  params=%d",
        input_dim, LATENT_DIM, BETA, n_params,
    )

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_csv  = exp_dir / "loss_history.csv"
    csv_file = open(log_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch",
        "train_total", "train_recon", "train_kl",
        "val_total",   "val_recon",   "val_kl",
        "lr",
    ])

    exp_ckpt         = exp_dir / "best_model.pth"
    best_val_loss    = float("inf")
    patience_counter = 0

    log.info("=" * 70)
    log.info("Training  max_epochs=%d  early_stop_patience=%d",
             NUM_EPOCHS, EARLY_STOP_PATIENCE)
    log.info("=" * 70)

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            tr_total, tr_recon, tr_kl = _run_epoch(
                model, train_loader, device, optimizer
            )
            vl_total, vl_recon, vl_kl = _run_epoch(
                model, val_loader, device, None      # eval mode
            )
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(vl_total)

            log.info(
                "Epoch %3d/%d | "
                "train [tot=%.4f  rec=%.4f  kl=%.4f]  "
                "val   [tot=%.4f  rec=%.4f  kl=%.4f]  lr=%.2e",
                epoch, NUM_EPOCHS,
                tr_total, tr_recon, tr_kl,
                vl_total, vl_recon, vl_kl,
                current_lr,
            )

            csv_writer.writerow([
                epoch,
                f"{tr_total:.6f}", f"{tr_recon:.6f}", f"{tr_kl:.6f}",
                f"{vl_total:.6f}", f"{vl_recon:.6f}", f"{vl_kl:.6f}",
                f"{current_lr:.2e}",
            ])
            csv_file.flush()

            # ── Checkpoint ────────────────────────────────────────────────────
            if vl_total < best_val_loss:
                best_val_loss    = vl_total
                patience_counter = 0
                ckpt = {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_loss":    best_val_loss,
                    "input_dim":   input_dim,
                    "beta":        BETA,
                    "latent_dim":  LATENT_DIM,
                    "exp_dir":     str(exp_dir),
                }
                torch.save(ckpt, exp_ckpt)
                shutil.copy(exp_ckpt, CANONICAL_CKPT)  # keep models/best_vae.pth current
                log.info("  ✓ checkpoint saved  val_loss=%.4f", best_val_loss)
            else:
                patience_counter += 1
                log.info(
                    "  patience %d/%d  (best=%.4f)",
                    patience_counter, EARLY_STOP_PATIENCE, best_val_loss,
                )
                if patience_counter >= EARLY_STOP_PATIENCE:
                    log.info("Early stopping at epoch %d.", epoch)
                    break

    finally:
        csv_file.close()

    # ── Post-training plots ───────────────────────────────────────────────────
    log.info("Generating plots...")
    _plot_loss_curves(exp_dir, log_csv)

    # Reload best checkpoint for reconstruction plot
    best_ckpt = torch.load(exp_ckpt, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    _plot_reconstruction(exp_dir, model, device, DATA_PROC_DIR)

    log.info("=" * 70)
    log.info("Done.  best_val=%.4f  exp=%s", best_val_loss, exp_dir)
    log.info("  %s/best_model.pth", exp_dir.name)
    log.info("  %s/loss_history.csv", exp_dir.name)
    log.info("  %s/loss_curves.png", exp_dir.name)
    log.info("  %s/reconstruction_plot.png", exp_dir.name)
    log.info("  %s  (canonical link)", CANONICAL_CKPT)
    log.info("=" * 70)

    return exp_dir


if __name__ == "__main__":
    train()
