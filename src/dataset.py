"""
src/dataset.py
Dataset de PyTorch para ventanas deslizantes sobre el panel de series
de consumo eléctrico (ElectricityLoadDiagrams20112014).

Diseño:
  - Entrada  X: (batch, L, n_features)  — L horas de historia + covariables
  - Objetivo y: (batch, H)              — H horas a pronosticar
  - El scaler (z-score por serie) se ajusta SOLO con estadísticos de entrenamiento.
  - Las covariables de calendario son "conocidas a futuro" (sin fuga).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils import portugal_holidays


# ─────────────────────────────────────────────────────────────────────────────
# Encoder de covariables de calendario (seno/coseno)
# ─────────────────────────────────────────────────────────────────────────────

def _sincos(value: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Codificación seno-coseno para una variable cíclica."""
    angle = 2 * np.pi * value / period
    return np.sin(angle), np.cos(angle)


def build_calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Construye las covariables de calendario conocidas a futuro.

    Codificación seno/coseno para capturar ciclicidad:
        - Hora del día   (período 24)
        - Día de la semana (período 7)
        - Mes del año    (período 12)
    Más una feature binaria de feriado.

    Returns:
        array de shape (len(index), 7):
        [sin_hora, cos_hora, sin_dow, cos_dow, sin_mes, cos_mes, es_feriado]
    """
    holidays = portugal_holidays(years=range(index.year.min(), index.year.max() + 1))
    holiday_flag = np.array(
        [1.0 if d.date() in holidays else 0.0 for d in index], dtype=np.float32
    )

    sin_h, cos_h = _sincos(index.hour.values.astype(np.float32), 24.0)
    sin_d, cos_d = _sincos(index.dayofweek.values.astype(np.float32), 7.0)
    sin_m, cos_m = _sincos((index.month.values - 1).astype(np.float32), 12.0)

    return np.stack(
        [sin_h, cos_h, sin_d, cos_d, sin_m, cos_m, holiday_flag], axis=1
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset principal
# ─────────────────────────────────────────────────────────────────────────────

class ElectricityDataset(Dataset):
    """
    Dataset de ventanas deslizantes para el panel de electricidad.

    Args:
        panel      : DataFrame (T x N_clientes) con índice DatetimeIndex
                     en frecuencia horaria, valores en kWh.
        L          : longitud de la ventana de entrada (horas de historia).
        H          : horizonte de pronóstico (horas).
        stride     : paso entre ventanas (horas). Usar 1 para máxima densidad,
                     24 para step diario.
        mean_      : Media por cliente (shape N,).  Si None se calcula internamente.
        std_       : Desviación std por cliente (shape N,). Si None se calcula.
        use_covariates: Si True, concatena covariables de calendario a X.

    Raises:
        ValueError: Si mean_/std_ son None simultáneamente (en val/test hay que
                    pasar los estadísticos de entrenamiento para evitar fuga).
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        L: int = 168,
        H: int = 24,
        stride: int = 1,
        mean_: np.ndarray | None = None,
        std_: np.ndarray | None = None,
        use_covariates: bool = True,
    ) -> None:
        super().__init__()
        self.L = L
        self.H = H
        self.use_covariates = use_covariates

        values = panel.values.astype(np.float32)   # (T, N)
        T, N = values.shape
        self.T = T
        self.N = N

        # ── Normalización por serie (z-score) ─────────────────────────────────
        # Se calculan SIEMPRE con datos de entrenamiento; val/test reciben
        # mean_ y std_ desde fuera para evitar fuga de información.
        if mean_ is None:
            self.mean_ = values.mean(axis=0, keepdims=True)   # (1, N)
        else:
            self.mean_ = mean_.reshape(1, -1).astype(np.float32)

        if std_ is None:
            self.std_ = values.std(axis=0, keepdims=True) + 1e-8
        else:
            self.std_ = std_.reshape(1, -1).astype(np.float32) + 1e-8

        self.values_norm = (values - self.mean_) / self.std_   # (T, N)

        # ── Covariables de calendario ─────────────────────────────────────────
        if use_covariates:
            self.cal = build_calendar_features(panel.index)   # (T, 7)
        else:
            self.cal = None

        # ── Índices de ventanas ───────────────────────────────────────────────
        # Cada ventana comienza en t y necesita [t, t+L+H)
        max_t = T - L - H
        self.starts = list(range(0, max_t + 1, stride))   # paso diario o por hora

    # ── Interfaz Dataset ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        # Una muestra por (ventana, cliente)
        return len(self.starts) * self.N

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            X: (L, n_features) — historia normalizada [+ covariables de calendario]
            y: (H,)            — target normalizado del cliente
        """
        win_idx, client_idx = divmod(idx, self.N)
        t = self.starts[win_idx]

        # Historia del cliente: (L,)
        x_series = self.values_norm[t : t + self.L, client_idx]

        if self.use_covariates:
            # Covariables del horizonte de ENTRADA: (L, 7)
            x_cal = self.cal[t : t + self.L]
            X = np.concatenate(
                [x_series[:, None], x_cal], axis=1
            )  # (L, 1+7) = (L, 8)
        else:
            X = x_series[:, None]  # (L, 1)

        # Target: (H,)
        y = self.values_norm[t + self.L : t + self.L + self.H, client_idx]

        return torch.from_numpy(X), torch.from_numpy(y)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def inverse_transform(
        self, y_norm: np.ndarray, client_indices: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Desnormaliza predicciones.

        Args:
            y_norm        : array (..., H) en escala normalizada.
            client_indices: índices de los clientes correspondientes.
                            Si None se asume broadcast sobre todos los clientes.
        """
        if client_indices is not None:
            mean = self.mean_[0, client_indices]
            std  = self.std_[0, client_indices]
            return y_norm * std[..., None] + mean[..., None]
        return y_norm * self.std_ + self.mean_

    @property
    def n_features(self) -> int:
        """Número de features de entrada (1 serie + 7 covariables si aplica)."""
        return 8 if self.use_covariates else 1

    def window_count_info(self) -> dict:
        """Reporta cuántas ventanas produce el panel."""
        return {
            "n_windows": len(self.starts),
            "n_clients": self.N,
            "total_samples": len(self),
            "L": self.L,
            "H": self.H,
        }
