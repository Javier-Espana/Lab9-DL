"""
src/utils.py
Utilidades globales: semilla fija, helpers de tiempo y formato.
"""
import random
import numpy as np
import torch


SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fija todas las semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Devuelve CUDA si está disponible, si no MPS (Apple), si no CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def portugal_holidays(years):
    """
    Devuelve un set de fechas (datetime.date) con los feriados nacionales
    de Portugal para los años indicados.

    Usa la librería `holidays`.  Si no está instalada cae a lista manual
    con los feriados fijos más comunes.
    """
    try:
        import holidays as hol
        dates = set()
        for y in years:
            dates.update(hol.Portugal(years=y).keys())
        return dates
    except ImportError:
        import datetime
        dates = set()
        for y in years:
            dates.update([
                datetime.date(y, 1, 1),   # Ano Novo
                datetime.date(y, 4, 25),  # Dia da Liberdade
                datetime.date(y, 5, 1),   # Dia do Trabalhador
                datetime.date(y, 6, 10),  # Dia de Portugal
                datetime.date(y, 8, 15),  # Assunção de Nossa Senhora
                datetime.date(y, 10, 5),  # Implantação da República
                datetime.date(y, 11, 1),  # Todos os Santos
                datetime.date(y, 12, 1),  # Restauração da Independência
                datetime.date(y, 12, 8),  # Imaculada Conceição
                datetime.date(y, 12, 25), # Natal
            ])
        return dates
