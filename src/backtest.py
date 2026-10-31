"""
backtest.py - Walk-forward validation (backtesting) for time-series forecasting models.

Supports monthly retraining and returns a comprehensive dictionary of metrics.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (robust to near-zero values)."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error.

    WARNING: unstable when y_true contains values near zero.
    Use WAPE as an alternative in those cases.
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))))


def mase(y_true: np.ndarray, y_pred: np.ndarray,
         y_train: np.ndarray, seasonality: int = 24) -> float:
    """Mean Absolute Scaled Error.

    Scales MAE by the in-sample naive seasonal forecast error.
    """
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def mae_by_horizon(y_true: np.ndarray, y_pred: np.ndarray,
                   horizon: int = 24) -> np.ndarray:
    """Return MAE for each forecast step h = 1 .. horizon.

    Expects y_true and y_pred shaped (n_windows, horizon).
    """
    errors = np.abs(y_true - y_pred)   # (n_windows, horizon)
    return errors.mean(axis=0)         # (horizon,)


def mae_by_hour(timestamps: pd.DatetimeIndex,
                y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    """Return MAE grouped by hour of day (0-23).

    Parameters
    ----------
    timestamps : DatetimeIndex of length n_windows * horizon (flat).
    y_true, y_pred : flat arrays of the same length.
    """
    df = pd.DataFrame({"hour": timestamps.hour,
                       "error": np.abs(y_true - y_pred)})
    return df.groupby("hour")["error"].mean()


# ---------------------------------------------------------------------------
# Core backtest function
# ---------------------------------------------------------------------------

def backtest(
    model: Any,
    dataset: pd.DataFrame,
    config: Optional[Dict] = None,
) -> Dict:
    """Walk-forward (expanding-window) backtesting with optional monthly retraining.

    Parameters
    ----------
    model : object
        Must implement:
          - ``fit(X_train, y_train)``   – trains the model on a training slice.
          - ``predict(X)``              – returns a 1-D array of point forecasts.
        A ``seasonality`` attribute (int, default 24) is read if present.

    dataset : pd.DataFrame
        Must contain:
          - A DatetimeIndex (hourly resolution assumed).
          - A column ``"y"``  with the target variable (energy consumption).
          - Optionally other columns used as features by the model.

    config : dict, optional
        Keys
        ----
        horizon         int  (default 24)    – forecast horizon in steps.
        initial_train   int  (default 8760)  – minimum training observations.
        step            int  (default 24)    – steps to advance window each iter.
        monthly_retrain bool (default True)  – re-fit at the start of each month.
        seasonality     int  (default 24)    – used for MASE denominator.
        verbose         bool (default False).

    Returns
    -------
    dict with keys:
        metrics      – dict: mae, rmse, wape, mape, mase (scalars).
        mae_horizon  – np.ndarray shape (horizon,).
        mae_hour     – pd.Series indexed 0-23.
        predictions  – pd.DataFrame with columns y_true, y_pred (DatetimeIndex).
        windows      – list of dicts (train_end, test_start, test_end) per window.
    """
    cfg = config or {}
    horizon: int           = cfg.get("horizon", 24)
    initial_train: int     = cfg.get("initial_train", 8760)
    step: int              = cfg.get("step", 24)
    monthly_retrain: bool  = cfg.get("monthly_retrain", True)
    seasonality: int       = cfg.get("seasonality",
                                     getattr(model, "seasonality", 24))
    verbose: bool          = cfg.get("verbose", False)

    y = dataset["y"].values
    feature_cols = [c for c in dataset.columns if c != "y"]
    X = dataset[feature_cols].values if feature_cols else None
    timestamps = dataset.index

    all_true, all_pred, all_ts = [], [], []
    windows = []

    current_month: Optional[int] = None
    model_fitted = False

    t = initial_train
    while t + horizon <= len(y):
        train_slice = slice(0, t)
        test_slice  = slice(t, t + horizon)

        y_train = y[train_slice]
        y_test  = y[test_slice]
        ts_test = timestamps[test_slice]

        # Monthly retraining logic
        month_now = ts_test[0].month if hasattr(ts_test[0], "month") else None
        needs_fit = (
            not model_fitted
            or (monthly_retrain and month_now != current_month)
        )

        if needs_fit:
            if verbose:
                print(f"  [backtest] Fitting at t={t}, month={month_now}")
            X_train = X[train_slice] if X is not None else y_train
            model.fit(X_train, y_train)
            model_fitted = True
            current_month = month_now

        X_test = X[test_slice] if X is not None else y[train_slice]
        y_hat = np.asarray(model.predict(X_test)).ravel()[:horizon]

        all_true.append(y_test)
        all_pred.append(y_hat)
        all_ts.append(ts_test)

        windows.append({
            "train_end":  timestamps[t - 1],
            "test_start": ts_test[0],
            "test_end":   ts_test[-1],
        })

        t += step

    # --- flatten ---
    y_true_flat = np.concatenate(all_true)
    y_pred_flat = np.concatenate(all_pred)
    ts_flat     = timestamps[0:0].append([idx for idx in all_ts])

    # --- scalar metrics ---
    y_train_full = y[:initial_train]
    metrics = {
        "mae":  mae(y_true_flat, y_pred_flat),
        "rmse": rmse(y_true_flat, y_pred_flat),
        "wape": wape(y_true_flat, y_pred_flat),
        "mape": mape(y_true_flat, y_pred_flat),
        "mase": mase(y_true_flat, y_pred_flat, y_train_full, seasonality),
    }

    # --- MAE by horizon ---
    mat_true = np.array(all_true)   # (n_windows, horizon)
    mat_pred = np.array(all_pred)
    mae_h    = mae_by_horizon(mat_true, mat_pred, horizon)

    # --- MAE by hour ---
    mae_h_of_day = mae_by_hour(ts_flat, y_true_flat, y_pred_flat)

    # --- predictions DataFrame ---
    preds_df = pd.DataFrame({
        "timestamp": ts_flat,
        "y_true":    y_true_flat,
        "y_pred":    y_pred_flat,
    }).set_index("timestamp")

    if verbose:
        print("\n=== Backtest results ===")
        for k, v in metrics.items():
            print(f"  {k.upper():6s}: {v:.4f}")

    return {
        "metrics":     metrics,
        "mae_horizon": mae_h,
        "mae_hour":    mae_h_of_day,
        "predictions": preds_df,
        "windows":     windows,
    }
