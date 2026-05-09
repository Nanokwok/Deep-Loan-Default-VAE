"""
β-VAE for semi-supervised anomaly detection on tabular loan data.

Architecture  (input_dim = 53):
  Encoder : 53 → 48 → 32 → 16  →  μ (16)  &  log σ² (16)
  Latent  :                    z = μ + ε · σ   (reparameterisation trick)
  Decoder : 16 → 16 → 32 → 48 → 53

Loss — β-ELBO:
  L = MSE_recon  +  β · KL(q(z|x) ∥ N(0,I))

  Reconstruction term (MSE):
      Summed over all input dimensions per sample, averaged over the batch.
      Using 'sum per sample' keeps the loss magnitude proportional to input_dim
      (as opposed to F.mse_loss reduction='mean' which divides by N×D and
      inadvertently rescales the KL term by 1/D).

  KL Divergence (closed-form Gaussian):
      KL = -½ · Σ_d (1 + log σ²_d - μ²_d - σ²_d)
      Summed over latent dims, averaged over batch — same normalisation as recon.

  β weighting:
      β = 1.0  →  standard VAE; full KL pressure toward N(0,I).
      β < 1.0  →  relaxed prior; encoder retains more discriminative structure.
      β > 1.0  →  β-VAE disentanglement (not appropriate here).

      For tabular credit data the config default is β = 0.1.
      Strong KL pressure collapses grade / income clusters that the encoder
      needs to distinguish normal loans from anomalous ones.
      If after training the reconstruction-error distributions of Fully Paid
      and Charged Off rows heavily overlap → lower β further.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BETA, DECODER_DIMS, ENCODER_DIMS, LATENT_DIM


class BetaVAE(nn.Module):
    """
    β-Variational Autoencoder trained on the normal class only.

    At inference time the anomaly score for a sample x is its
    reconstruction error  ‖x − x̂‖².  Samples with unusually high
    reconstruction error are flagged as anomalous (defaulted loans).
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.encoder    = self._build_encoder(input_dim)
        self.fc_mu      = nn.Linear(ENCODER_DIMS[-1], LATENT_DIM)
        self.fc_log_var = nn.Linear(ENCODER_DIMS[-1], LATENT_DIM)
        self.decoder    = self._build_decoder(input_dim)

    # ── Architecture builders ──────────────────────────────────────────────────

    @staticmethod
    def _build_encoder(input_dim: int, dropout_rate: float = 0.2) -> nn.Sequential:
        """Funnel: input_dim → ENCODER_DIMS[0] → … → ENCODER_DIMS[-1]."""
        layers: list[nn.Module] = []
        in_size = input_dim
        for out_size in ENCODER_DIMS:
            layers += [
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),   # stabilises training on tabular data
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            ]
            in_size = out_size
        return nn.Sequential(*layers)

    @staticmethod
    def _build_decoder(input_dim: int, dropout_rate: float = 0.2) -> nn.Sequential:
        """
        Mirror funnel: LATENT_DIM → DECODER_DIMS[0] → … → input_dim.
        No activation on the final layer — output is in the same space as the
        StandardScaler-normalised input, so raw values are the MSE targets.
        """
        layers: list[nn.Module] = []
        in_size = LATENT_DIM
        for out_size in DECODER_DIMS:
            layers += [
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            ]
            in_size = out_size
        layers.append(nn.Linear(in_size, input_dim))   # linear output
        return nn.Sequential(*layers)

    # ── Forward pass components ────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x → (μ, log σ²) via the encoder network."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterise(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterisation trick:  z = μ + ε · σ,   ε ~ N(0, I).

        Keeps the sampling operation differentiable so gradients flow back
        through μ and log σ² during training.

        At inference (model.eval()) we skip the noise and return μ directly.
        This gives a deterministic, lower-variance reconstruction error — which
        is exactly the anomaly score used in Phase 4 evaluation.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)    # σ = exp(log σ² / 2)
            eps = torch.randn_like(std)       # ε ~ N(0, I), same shape as σ
            return mu + eps * std
        return mu                             # eval: deterministic mean only

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z → x̂ via the decoder network."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            x_hat   : reconstructed input  (batch, input_dim)
            mu      : latent mean           (batch, latent_dim)
            log_var : latent log-variance   (batch, latent_dim)

        All three values are required by vae_loss().
        """
        mu, log_var = self.encode(x)
        z           = self.reparameterise(mu, log_var)
        x_hat       = self.decode(z)
        return x_hat, mu, log_var


# ── Loss function ──────────────────────────────────────────────────────────────

def vae_loss(
    x:       torch.Tensor,
    x_hat:   torch.Tensor,
    mu:      torch.Tensor,
    log_var: torch.Tensor,
    beta:    float = BETA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    β-ELBO loss.

    Args:
        x        : original input           (batch, input_dim)
        x_hat    : reconstructed input      (batch, input_dim)
        mu       : latent mean              (batch, latent_dim)
        log_var  : latent log-variance      (batch, latent_dim)
        beta     : KL weight (default from config)

    Returns:
        total_loss : scalar for .backward()
        recon_loss : scalar for logging (reconstruction component)
        kl_loss    : scalar for logging (KL component, before β scaling)

    Loss formulation:
        recon_loss = (1/N) · Σ_i  ‖x_i − x̂_i‖²          (sum over D, mean over N)
        kl_loss    = (1/N) · Σ_i  −½ Σ_d (1 + log σ²_id − μ²_id − σ²_id)
        total_loss = recon_loss + β · kl_loss

    Note on normalisation:
        Both terms use 'sum over dims, mean over batch'.
        This ensures the β scale remains consistent regardless of input_dim
        or latent_dim — β = 0.1 means the KL contributes 10 % of the loss
        in expectation at the start of training when recon ≈ KL.
    """
    # ── Reconstruction loss ────────────────────────────────────────────────────
    # F.mse_loss with reduction='sum' gives Σ_i Σ_d (x_id - x̂_id)²
    # Dividing by batch size N gives per-sample average (sum over dims).
    recon_loss = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)

    # ── KL Divergence ──────────────────────────────────────────────────────────
    # Closed-form KL between diagonal Gaussian q(z|x) and standard prior p(z):
    #   KL = -½ Σ_d (1 + log σ²_d - μ²_d - exp(log σ²_d))
    # torch.sum(…, dim=1) → sum over latent dims per sample  (batch,)
    # .mean()              → average over the batch            scalar
    # Clamp log_var to prevent exp() overflow (numerical stability).
    # [-4, 15] keeps σ in [~0.02, ~3000] — well beyond any legitimate range.
    log_var_clamped = log_var.clamp(-4, 15)
    kl_loss = -0.5 * torch.sum(
        1.0 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(),
        dim=1,
    ).mean()

    # ── β-weighted total ───────────────────────────────────────────────────────
    # β < 1 relaxes the Gaussian prior constraint.
    # Logged separately so we can track each component across epochs.
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
