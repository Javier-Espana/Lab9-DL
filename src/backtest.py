"""
src/backtest.py
Protocolo de backtesting con origen móvil (rolling-origin) para todos los modelos.

Implementado UNA sola vez; todos los modelos reciben la misma función backtest().

Diseño:
  - Prueba: 2014-10-01 a 2014-12-31, 92 orígenes.
  - Cada origen avanza 24 h.
  - En cada origen el modelo recibe datos hasta las 23:00 del día anterior
    y pronostica las 24 h siguientes.
  - Los pesos se entrenan una vez (hasta 2014-09-30); los modelos de estadística
    y LightGBM pueden re-ajustarse con ventana expandible si se indica.
"""
from __future__ import annotations

import time
from typing import Callable, Protocol, Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.metrics import evaluate_all, mae_per_horizon, pinball_loss, coverage, mae, rmse, wape, mape, mase


# ─────────────────────────────────────────────────────────────────────────────
# Protocolo de modelo (duck-typing)
# ─────────────────────────────────────────────────────────────────────────────

class ForecastModel(Protocol):
    """
    Interfaz mínima que cualquier modelo debe cumplir para ser evaluado
    por la función backtest().
    """

    def predict(
        self,
        history: pd.DataFrame,
        horizon: int,
        origin_time: pd.Timestamp,
    ) -> np.ndarray:
        """
        Genera pronósticos para las `horizon` horas siguientes a origin_time.

        Args:
            history     : DataFrame (T x N_clientes), datos disponibles
                          hasta (origin_time - 1 hora) inclusive.
            horizon     : número de pasos a pronosticar (H = 24).
            origin_time : instante en que se emite el pronóstico (00:00).

        Returns:
            array de shape (N_clientes, horizon) en kWh (escala original).
        """
        ...

    @property
    def name(self) -> str:
        """Nombre del modelo (para la tabla comparativa)."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def backtest(
    model: ForecastModel,
    panel: pd.DataFrame,
    y_train: np.ndarray,
    H: int = 24,
    test_start: str = "2014-10-01",
    test_end: str = "2014-12-31",
    seasonality: int = 24,
    return_predictions: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Backtesting con origen móvil diario sobre el período de prueba.

    Args:
        model            : objeto con interfaz ForecastModel.
        panel            : DataFrame completo (T x N) con frecuencia horaria.
        y_train          : array (N, T_train) con la serie de entrenamiento
                           en kWh original, para calcular MASE.
        H                : horizonte (24 h).
        test_start       : primer día de prueba (emite pronóstico a las 00:00).
        test_end         : último día de prueba.
        seasonality      : período del naive seasonal para MASE.
        return_predictions: si True devuelve DataFrame con (unique_id, ds, y_hat).
        verbose          : si True imprime progreso.

    Returns:
        dict con:
          - metrics      : dict {MAE, RMSE, WAPE, MAPE, MASE}
          - mae_horizon  : array (H,) — MAE por paso del horizonte
          - predictions  : DataFrame (unique_id, ds, modelo, y_hat) o None
          - inference_time: segundos totales de inferencia
    """
    origins = pd.date_range(start=test_start, end=test_end, freq="D")
    n_origins = len(origins)

    all_true  = []
    all_pred  = []
    pred_rows = []

    t_inference_start = time.perf_counter()

    for i, origin in enumerate(origins):
        cutoff = origin - pd.Timedelta(hours=1)
        history = panel.loc[:cutoff]

        preds = model.predict(history, H, origin)  # (N, H)

        future_idx = pd.date_range(start=origin, periods=H, freq="h")
        valid_idx = future_idx.intersection(panel.index)
        true_vals = panel.loc[valid_idx].values.T   # (N, len(valid_idx))

        if len(valid_idx) < H:
            true_full = np.full((preds.shape[0], H), np.nan)
            true_full[:, : len(valid_idx)] = true_vals
            true_vals = true_full

        all_true.append(true_vals)
        all_pred.append(preds)

        if return_predictions:
            clients = panel.columns
            for c_idx, client in enumerate(clients):
                for h_idx, ts in enumerate(future_idx):
                    pred_rows.append(
                        {
                            "unique_id": client,
                            "ds": ts,
                            "modelo": getattr(model, "name", "model"),
                            "y_hat": float(preds[c_idx, h_idx]),
                        }
                    )

        if verbose and (i % 20 == 0 or i == n_origins - 1):
            print(f"  Origen {i+1:3d}/{n_origins} — {origin.date()}", end="\r")

    inference_time = time.perf_counter() - t_inference_start
    if verbose:
        print()

    y_true_arr = np.concatenate(all_true, axis=1).reshape(-1, H)
    y_pred_arr = np.concatenate(all_pred, axis=1).reshape(-1, H)

    valid_mask = ~np.isnan(y_true_arr).any(axis=1)
    y_true_arr = y_true_arr[valid_mask]
    y_pred_arr = y_pred_arr[valid_mask]

    metrics = evaluate_all(
        y_true_arr, y_pred_arr,
        y_train=y_train.reshape(-1),
        seasonality=seasonality,
        model_name=getattr(model, "name", "model"),
    )
    metrics["inference_time_s"] = round(inference_time, 2)

    mae_h = mae_per_horizon(y_true_arr, y_pred_arr)
    predictions_df = pd.DataFrame(pred_rows) if return_predictions else None

    return {
        "metrics": metrics,
        "mae_horizon": mae_h,
        "predictions": predictions_df,
        "inference_time": inference_time,
    }


def backtest_with_refit(
    model_factory: Callable[[], ForecastModel],
    panel: pd.DataFrame,
    y_train: np.ndarray,
    H: int = 24,
    test_start: str = "2014-10-01",
    test_end: str = "2014-12-31",
    seasonality: int = 24,
    verbose: bool = True,
) -> dict:
    """
    Backtesting con origen móvil reentrenando al inicio de cada mes.
    """
    origins = pd.date_range(start=test_start, end=test_end, freq="D")
    months  = sorted(set(o.to_period("M") for o in origins))

    all_results = []
    for month in months:
        if verbose:
            print(f"  Reentrenando para {month}…")
        cutoff = month.to_timestamp() - pd.Timedelta(hours=1)
        panel_train = panel.loc[:cutoff]

        model = model_factory()
        month_origins = [o for o in origins if o.to_period("M") == month]
        res = backtest(
            model, panel,
            y_train=panel_train.values.T,
            H=H, test_start=str(month_origins[0].date()),
            test_end=str(month_origins[-1].date()),
            seasonality=seasonality, verbose=verbose,
        )
        all_results.append(res)

    all_preds = pd.concat(
        [r["predictions"] for r in all_results if r["predictions"] is not None]
    )
    return {"monthly_results": all_results, "predictions": all_preds}
