"""
β-VAE for semi-supervised anomaly detection on tabular loan data.

Architecture  (input_dim = 53):
  Encoder : 53 → 48 → 32 → 16  →  μ (16)  &  log σ² (16)
  Latent  :                    z = μ + ε · σ   (reparameterisation trick)
  Decoder : 16 → 16 → 32 → 48 → 53
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
    """
    # ── Reconstruction loss ────────────────────────────────────────────────────
    recon_loss = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)

    # ── KL Divergence ──────────────────────────────────────────────────────────
    log_var_clamped = log_var.clamp(-4, 15)
    kl_loss = -0.5 * torch.sum(
        1.0 + log_var_clamped - mu.pow(2) - log_var_clamped.exp(),
        dim=1,
    ).mean()

    # ── β-weighted total ───────────────────────────────────────────────────────
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
