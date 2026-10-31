"""
src/metrics.py
Métricas de evaluación para forecasting de series de tiempo.

Todas operan en la escala original (kWh) y esperan arrays 2D:
    y_true : (n_samples, H)
    y_pred : (n_samples, H)
o 1D cuando se llaman por paso del horizonte.
"""
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Métricas básicas
# ─────────────────────────────────────────────────────────────────────────────

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (kWh)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (kWh)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error (%).
    ⚠️  Diverge cuando y_true ≈ 0 (clientes con consumo cercano a cero).
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error = Σ|error| / Σ|real|.
    Robusto ante valores cercanos a cero.
    """
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8))


# ─────────────────────────────────────────────────────────────────────────────
# MASE
# ─────────────────────────────────────────────────────────────────────────────

def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 24,
) -> float:
    """
    Mean Absolute Scaled Error.

    Se normaliza por el MAE del seasonal naive en el conjunto de entrenamiento:
        denom = mean(|y_train[t] - y_train[t - seasonality]|)

    Args:
        y_true  : valores reales de prueba, shape (n, H).
        y_pred  : predicciones, shape (n, H).
        y_train : serie de entrenamiento usada para calcular la escala,
                  shape (T,) o (n_series, T).
        seasonality: período estacional (24 para diario, 168 para semanal).

    Returns:
        MASE escalar.
    """
    errors = np.abs(y_true - y_pred)
    numerator = np.mean(errors)

    # Calcular denominador con la serie de entrenamiento
    y_train = np.asarray(y_train)
    if y_train.ndim == 1:
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    else:
        naive_errors = np.abs(
            y_train[:, seasonality:] - y_train[:, :-seasonality]
        )
    denominator = np.mean(naive_errors) + 1e-8

    return float(numerator / denominator)


# ─────────────────────────────────────────────────────────────────────────────
# Pinball Loss (cuantiles)
# ─────────────────────────────────────────────────────────────────────────────

def pinball_loss(
    y_true: np.ndarray,
    y_quantile: np.ndarray,
    q: float,
) -> float:
    """
    Pinball (quantile) loss para el cuantil q ∈ (0, 1).

    L_q(y, ŷ) = q · max(y - ŷ, 0) + (1 - q) · max(ŷ - y, 0)

    Interpretación: minimizar pinball loss en q equivale a producir el
    cuantil q del error de predicción.
    """
    errors = y_true - y_quantile
    return float(np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors)))


def coverage(
    y_true: np.ndarray,
    y_low: np.ndarray,
    y_high: np.ndarray,
) -> float:
    """Fracción de observaciones dentro del intervalo [y_low, y_high]."""
    return float(np.mean((y_true >= y_low) & (y_true <= y_high)))


# ─────────────────────────────────────────────────────────────────────────────
# Resumen completo
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 24,
    model_name: str = "model",
) -> dict:
    """
    Calcula MAE, RMSE, WAPE, MAPE y MASE de una vez.

    Returns:
        dict con claves: model, MAE, RMSE, WAPE, MAPE, MASE.
    """
    return {
        "model": model_name,
        "MAE":   mae(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "WAPE":  wape(y_true, y_pred),
        "MAPE":  mape(y_true, y_pred),
        "MASE":  mase(y_true, y_pred, y_train, seasonality),
    }


def mae_per_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    MAE por paso del horizonte.

    Args:
        y_true, y_pred: (n_samples, H)

    Returns:
        array de shape (H,).
    """
    return np.mean(np.abs(y_true - y_pred), axis=0)
