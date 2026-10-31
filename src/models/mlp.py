"""
src/models/mlp.py
MLP multi-salida para forecasting de series de tiempo.

Arquitectura:
  Input  → Flatten(L * n_features)
         → [Linear → BatchNorm → ReLU → Dropout] × n_layers
         → Linear(hidden, H)
  Output : H valores simultáneos (estrategia multi-salida directa, sin autoregresión).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPForecaster(nn.Module):
    """
    MLP multi-salida: recibe una ventana de historia y predice H pasos
    de forma simultánea (sin autoregresión).

    Args:
        input_len   : longitud de la ventana de entrada L.
        n_features  : número de features por paso de tiempo (1 + covariables).
        hidden_size : tamaño de cada capa oculta.
        n_layers    : número de capas ocultas.
        output_len  : horizonte H.
        dropout     : tasa de dropout.
    """

    def __init__(
        self,
        input_len: int = 168,
        n_features: int = 8,
        hidden_size: int = 256,
        n_layers: int = 3,
        output_len: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = input_len * n_features

        layers: list[nn.Module] = [nn.Flatten()]
        for i in range(n_layers):
            layers += [
                nn.Linear(in_dim if i == 0 else hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden_size, output_len))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, L, n_features)
        Returns:
            (batch, H)
        """
        return self.net(x)
