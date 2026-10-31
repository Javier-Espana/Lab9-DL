"""
src/models/lstm.py
LSTM / GRU para forecasting con tres estrategias multi-paso:
  1. Autorregresiva    : predice 1 paso y realimenta la predicción (H veces).
  2. Multi-salida      : una capa densa devuelve los H pasos de una vez.
  3. Encoder-Decoder   : encoder LSTM procesa historia; decoder LSTM genera
                         paso a paso usando las covariables de calendario futuras.

También incluye RevIN (Reversible Instance Normalization) opcional.

Referencia RevIN:
  Kim et al. (2021) "Reversible Instance Normalization for Accurate
  Time-Series Forecasting against Distribution Shift"
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# RevIN
# ─────────────────────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    Normaliza la instancia antes de la red y la desnormaliza a la salida,
    lo que ayuda ante cambios de nivel (non-stationarity).

    Args:
        n_features: número de features a normalizar (normalmente 1, la serie).
        eps: estabilidad numérica.
        affine: si True aprende parámetros γ y β.
    """

    def __init__(self, n_features: int = 1, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(n_features))
            self.beta  = nn.Parameter(torch.zeros(n_features))
        else:
            self.gamma = self.beta = None

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, L, n_features)
        Returns:
            x_norm, mean, std
        """
        mean = x.mean(dim=1, keepdim=True)
        std  = x.std(dim=1, keepdim=True) + self.eps
        x_norm = (x - mean) / std
        if self.gamma is not None:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm, mean, std

    def denormalize(
        self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            y   : (batch, H) o (batch, H, n_features)
            mean: (batch, 1, n_features)
            std : (batch, 1, n_features)
        """
        if self.gamma is not None:
            y = (y - self.beta) / (self.gamma + self.eps)
        return y * std[:, 0, :] + mean[:, 0, :]


# ─────────────────────────────────────────────────────────────────────────────
# LSTM / GRU base (estrategias 1 y 2)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """
    LSTM (o GRU) con dos modos de predicción:
      - strategy="multi_output"  : proyección directa de H pasos (sin autoregresión).
      - strategy="autoregressive": genera 1 paso y realimenta la predicción.

    Args:
        input_size  : número de features de entrada por paso de tiempo.
        hidden_size : dimensión del estado oculto.
        num_layers  : capas apiladas del LSTM/GRU.
        output_len  : H (horizonte).
        dropout     : dropout entre capas (solo activo si num_layers > 1).
        use_gru     : si True usa GRU en vez de LSTM.
        strategy    : "multi_output" | "autoregressive".
        use_revin   : si True aplica RevIN sobre la primera feature.
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 256,
        num_layers: int = 2,
        output_len: int = 24,
        dropout: float = 0.2,
        use_gru: bool = False,
        strategy: str = "multi_output",
        use_revin: bool = False,
    ) -> None:
        super().__init__()
        assert strategy in ("multi_output", "autoregressive"), \
            "strategy debe ser 'multi_output' o 'autoregressive'"

        self.strategy   = strategy
        self.output_len = output_len
        self.use_revin  = use_revin

        if use_revin:
            self.revin = RevIN(n_features=1)

        rnn_cls = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if strategy == "multi_output":
            self.head = nn.Linear(hidden_size, output_len)
        else:
            # En autoregresiva necesitamos un paso a la vez
            self.head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        future_cal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x         : (batch, L, input_size)
            future_cal: (batch, H, cal_features) — covariables futuras (solo
                        usadas en la estrategia autoregresiva si se pasan).
        Returns:
            (batch, H)
        """
        if self.use_revin:
            x_series = x[:, :, :1]
            x_norm, mean_, std_ = self.revin.normalize(x_series)
            x = torch.cat([x_norm, x[:, :, 1:]], dim=-1)

        _, hidden = self.rnn(x)   # procesa toda la ventana

        if self.strategy == "multi_output":
            # hidden[0]: (num_layers, batch, hidden) → tomamos última capa
            h_last = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
            out = self.head(h_last)   # (batch, H)

        else:  # autoregresiva
            h_last = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
            # Estado oculto completo para continuar generando
            h_state = hidden

            preds = []
            # El primer input del decoder es la última observación de x
            dec_in = x[:, -1:, :]   # (batch, 1, input_size)

            for t in range(self.output_len):
                out_t, h_state = self.rnn(dec_in, h_state)
                pred_t = self.head(out_t[:, -1, :])   # (batch, 1)
                preds.append(pred_t)

                # Realimentar predicción; si hay covariables futuras, usarlas
                if future_cal is not None:
                    cal_t = future_cal[:, t:t+1, :]  # (batch, 1, cal_features)
                    dec_in = torch.cat([pred_t.unsqueeze(1), cal_t], dim=-1)
                else:
                    dec_in = pred_t.unsqueeze(1)
                    if dec_in.shape[-1] < x.shape[-1]:
                        pad = torch.zeros(
                            dec_in.shape[0], 1, x.shape[-1] - 1, device=x.device
                        )
                        dec_in = torch.cat([dec_in, pad], dim=-1)

            out = torch.cat(preds, dim=1)   # (batch, H)

        if self.use_revin:
            # Desnormalizar: mean_ y std_ son (batch, 1, 1)
            out = out * std_[:, 0, :] + mean_[:, 0, :]

        return out
