"""
src/models/linear.py
Modelos lineales para forecasting: DLinear y NLinear.

Referencia:
  Zeng et al. (2023) "Are Transformers Effective for Time Series Forecasting?"
  https://arxiv.org/abs/2205.13504

DLinear: descompone la serie en tendencia (moving average) y residuo,
         aplica una capa lineal independiente a cada componente y suma.

NLinear:  substrae el último valor de la ventana (normalización de instancia
          simplificada), aplica una capa lineal y suma de vuelta el último valor.
          Muy efectivo ante distributional shift.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# DLinear
# ─────────────────────────────────────────────────────────────────────────────

class MovingAvg(nn.Module):
    """Media móvil causal con kernel de tamaño `kernel_size`."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, L, n_features)
        Returns:
            tendencia (batch, L, n_features) — padding al inicio para mantener T
        """
        # padding al inicio (causal)
        front = x[:, :1, :].repeat(1, self.kernel_size - 1, 1)
        x_padded = torch.cat([front, x], dim=1)          # (B, L+k-1, F)
        x_t = self.avg(x_padded.permute(0, 2, 1))        # (B, F, L)
        return x_t.permute(0, 2, 1)                      # (B, L, F)


class DLinear(nn.Module):
    """
    Descomposición lineal (DLinear).

    Pasos:
      1. trend    = MovingAvg(x)
      2. residual = x - trend
      3. ŷ = Linear_trend(flatten(trend)) + Linear_res(flatten(residual))

    Args:
        input_len   : L (longitud de ventana de entrada).
        n_features  : número de features (usa solo la primera = serie objetivo).
        output_len  : H.
        kernel_size : tamaño del kernel para media móvil.
        individual  : si True, una capa lineal por feature (más parámetros).
    """

    def __init__(
        self,
        input_len: int = 168,
        n_features: int = 1,
        output_len: int = 24,
        kernel_size: int = 25,
        individual: bool = False,
    ) -> None:
        super().__init__()
        self.individual = individual
        self.n_features = n_features
        self.moving_avg = MovingAvg(kernel_size)

        if individual:
            self.linear_trend = nn.ModuleList(
                [nn.Linear(input_len, output_len) for _ in range(n_features)]
            )
            self.linear_res = nn.ModuleList(
                [nn.Linear(input_len, output_len) for _ in range(n_features)]
            )
        else:
            self.linear_trend = nn.Linear(input_len, output_len)
            self.linear_res   = nn.Linear(input_len, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, L, n_features)
        Returns:
            (batch, H)  — predicción solo de la primera feature (serie objetivo)
        """
        trend    = self.moving_avg(x)          # (B, L, F)
        residual = x - trend                   # (B, L, F)

        if self.individual:
            out_trend = torch.stack(
                [self.linear_trend[i](trend[:, :, i]) for i in range(self.n_features)],
                dim=-1,
            )  # (B, H, F)
            out_res = torch.stack(
                [self.linear_res[i](residual[:, :, i]) for i in range(self.n_features)],
                dim=-1,
            )
            out = (out_trend + out_res)[:, :, 0]   # solo primera feature
        else:
            # Usar solo el canal de la serie objetivo (índice 0)
            out = self.linear_trend(trend[:, :, 0]) + self.linear_res(residual[:, :, 0])

        return out   # (B, H)


# ─────────────────────────────────────────────────────────────────────────────
# NLinear
# ─────────────────────────────────────────────────────────────────────────────

class NLinear(nn.Module):
    """
    Normalized Linear (NLinear).

    Normalización por substracción del último valor observado antes de la
    proyección lineal; se suma de vuelta al final.  Muy robusto ante
    distributional shift (cambios de nivel).

    Args:
        input_len  : L.
        output_len : H.
    """

    def __init__(self, input_len: int = 168, output_len: int = 24) -> None:
        super().__init__()
        self.linear = nn.Linear(input_len, output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, L, n_features) — usa solo x[:, :, 0]
        Returns:
            (batch, H)
        """
        series = x[:, :, 0]               # (B, L) — canal objetivo
        last   = series[:, -1:]           # (B, 1) — último valor observado
        series_norm = series - last       # substracción de instancia
        out = self.linear(series_norm)    # (B, H)
        return out + last                 # restaurar nivel
