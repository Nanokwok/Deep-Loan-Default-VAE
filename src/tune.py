"""
Optuna hyperparameter search for the β-VAE fraud detector.

Search space:
    beta        — log-uniform [1e-4, 1e-2]   KL weight
    latent_dim  — int         [2, 16]         latent space size
    noise_std   — uniform     [0.01, 0.08]    denoising noise base σ

Fixed during search (set in config.py):
    ENCODER_DIMS, DECODER_DIMS, LEARNING_RATE, BATCH_SIZE, FEATURE_WEIGHTS

Strategy:
    MedianPruner — prune trials whose intermediate AUPRC falls below the
    median of completed trials at the same epoch. Saves ~60% wall-clock time.

Usage:
    python -m src.tune                      # 50 trials, 40 epochs each
    python -m src.tune --n-trials 100       # more trials
    python -m src.tune --n-epochs 60        # longer per trial

Results saved to:  experiments/tune_results.json
                   experiments/tune_study.db   (resumable SQLite storage)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_data(
    proc_dir: Path,
    batch_size: int,
    random_seed: int,
) -> tuple[DataLoader, torch.Tensor, np.ndarray]:
    X_train = np.load(proc_dir / "X_train.npy").astype(np.float32)
    X_val   = np.load(proc_dir / "X_val.npy").astype(np.float32)
    y_val   = np.load(proc_dir / "y_val.npy").astype(np.int32)

    g = torch.Generator()
    g.manual_seed(random_seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    return loader, torch.from_numpy(X_val), y_val


def _build_weights(
    feature_weights_cfg: dict[str, float],
    feature_cols: list[str],
    input_dim: int,
    device: torch.device,
) -> torch.Tensor:
    w = torch.ones(input_dim, dtype=torch.float32)
    for feat, val in feature_weights_cfg.items():
        if feat in feature_cols:
            w[feature_cols.index(feat)] = val
    return w.to(device)


def _noise_sigma(
    feature_weights: torch.Tensor,
    base_std: float,
) -> torch.Tensor | None:
    if base_std == 0.0:
        return None
    w_norm = feature_weights / feature_weights.mean()
    return base_std / w_norm


def _val_auprc(
    model:           "BetaVAE",
    X_val:           torch.Tensor,
    y_val:           np.ndarray,
    device:          torch.device,
    feature_weights: torch.Tensor | None,
) -> float:
    from src.model import vae_loss
    model.eval()
    with torch.no_grad():
        xv    = X_val.to(device)
        x_hat, mu, log_var = model(xv)
        sq    = (xv - x_hat) ** 2
        if feature_weights is not None:
            sq = sq * feature_weights.unsqueeze(0)
        scores = sq.mean(dim=1).cpu().numpy()
    return float(average_precision_score(y_val, scores))


# ── Objective ─────────────────────────────────────────────────────────────────

def _objective(
    trial,
    proc_dir:    Path,
    device:      torch.device,
    n_epochs:    int,
    anneal_ep:   int,
) -> float:
    import optuna
    from src.config import (
        BATCH_SIZE, ENCODER_DIMS, DECODER_DIMS, FEATURE_WEIGHTS,
        INPUT_DIM, LEARNING_RATE, RANDOM_SEED,
    )
    from src.model import BetaVAE, vae_loss

    # ── Sample hyperparameters ────────────────────────────────────────────────
    beta       = trial.suggest_float("beta",       1e-4, 1e-2, log=True)
    latent_dim = trial.suggest_int("latent_dim",   2, 16)
    noise_std  = trial.suggest_float("noise_std",  0.01, 0.08)

    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    # ── Data ──────────────────────────────────────────────────────────────────
    loader, X_val, y_val = _load_data(proc_dir, BATCH_SIZE, RANDOM_SEED)

    # ── Model — override latent_dim only; architecture from config ────────────
    # Temporarily patch LATENT_DIM for BetaVAE constructor
    import src.config as _cfg
    _orig_latent = _cfg.LATENT_DIM
    _cfg.LATENT_DIM = latent_dim

    model = BetaVAE(input_dim=INPUT_DIM).to(device)
    _cfg.LATENT_DIM = _orig_latent   # restore immediately after construction

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    feat_w    = _build_weights(FEATURE_WEIGHTS, feature_cols, INPUT_DIM, device)
    noise_sig = _noise_sigma(feat_w, noise_std)

    best_auprc = -1.0

    for epoch in range(1, n_epochs + 1):
        # KL annealing
        annealed_beta = min(beta, beta * epoch / anneal_ep)

        # ── Train one epoch ───────────────────────────────────────────────────
        model.train()
        for (x,) in loader:
            x = x.to(device, non_blocking=True)
            x_in = (x + torch.randn_like(x) * noise_sig.unsqueeze(0)
                    if noise_sig is not None else x)
            optimizer.zero_grad()
            x_hat, mu, log_var = model(x_in)
            loss, _, _ = vae_loss(x, x_hat, mu, log_var,
                                  beta=annealed_beta, feature_weights=feat_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ── Validate ──────────────────────────────────────────────────────────
        auprc = _val_auprc(model, X_val, y_val, device, feat_w)
        best_auprc = max(best_auprc, auprc)

        # Report for MedianPruner
        trial.report(auprc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_auprc


# ── Main ──────────────────────────────────────────────────────────────────────

def tune(n_trials: int = 50, n_epochs: int = 40) -> dict:
    """
    Run Optuna study and return best hyperparameters.
    Results are persisted to experiments/tune_results.json.
    """
    import optuna
    from src.config import DATA_PROC_DIR, EXPERIMENTS_DIR, KL_ANNEAL_EPOCHS

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    device   = torch.device("cuda" if torch.cuda.is_available() else
                            "mps"  if torch.backends.mps.is_available() else "cpu")
    proc_dir = Path(DATA_PROC_DIR)
    exp_dir  = Path(EXPERIMENTS_DIR)
    exp_dir.mkdir(parents=True, exist_ok=True)

    db_path = exp_dir / "tune_study.db"

    log.info("Optuna search  n_trials=%d  n_epochs=%d  device=%s", n_trials, n_epochs, device)
    log.info("Search space:  beta=[1e-4,1e-2]  latent_dim=[2,16]  noise_std=[0.01,0.08]")
    log.info("Storage: %s", db_path)

    study = optuna.create_study(
        direction="maximize",
        study_name="fraud_vae",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,    # wait for 5 complete trials before pruning
            n_warmup_steps=10,     # don't prune before epoch 10
            interval_steps=1,
        ),
    )

    study.optimize(
        lambda trial: _objective(
            trial,
            proc_dir  = proc_dir,
            device    = device,
            n_epochs  = n_epochs,
            anneal_ep = min(KL_ANNEAL_EPOCHS, n_epochs),
        ),
        n_trials         = n_trials,
        show_progress_bar= True,
    )

    best = {
        "best_auprc" : study.best_value,
        "best_params": study.best_params,
        "n_trials"   : len(study.trials),
        "n_epochs"   : n_epochs,
    }

    out = exp_dir / "tune_results.json"
    with open(out, "w") as f:
        json.dump(best, f, indent=2)

    log.info("=" * 60)
    log.info("Best AUPRC : %.4f", best["best_auprc"])
    log.info("Best params:")
    for k, v in best["best_params"].items():
        log.info("  %-15s = %s", k, v)
    log.info("Results saved → %s", out)
    log.info("=" * 60)

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna VAE hyperparameter search")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=40,
                        help="Epochs per trial (shorter = faster search)")
    args = parser.parse_args()
    tune(n_trials=args.n_trials, n_epochs=args.n_epochs)
