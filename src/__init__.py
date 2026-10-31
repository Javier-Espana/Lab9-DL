"""
src/__init__.py
Paquete de utilidades para el Lab 9 – CC3092 Series de Tiempo.
"""
from src.utils   import set_seed, get_device, portugal_holidays, SEED
from src.metrics import (
    mae, rmse, mape, wape, mase,
    pinball_loss, coverage,
    evaluate_all, mae_per_horizon,
)
from src.dataset  import ElectricityDataset, build_calendar_features
from src.backtest import backtest, backtest_with_refit

__all__ = [
    # utils
    "set_seed", "get_device", "portugal_holidays", "SEED",
    # metrics
    "mae", "rmse", "mape", "wape", "mase",
    "pinball_loss", "coverage", "evaluate_all", "mae_per_horizon",
    # dataset
    "ElectricityDataset", "build_calendar_features",
    # backtest
    "backtest", "backtest_with_refit",
]
