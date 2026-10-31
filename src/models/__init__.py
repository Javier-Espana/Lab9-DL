"""
src/models/__init__.py
Registro de todos los modelos propios del laboratorio.
"""
from src.models.mlp     import MLPForecaster
from src.models.linear  import DLinear, NLinear
from src.models.lstm    import LSTMForecaster, RevIN
from src.models.seq2seq import Seq2SeqForecaster

__all__ = [
    "MLPForecaster",
    "DLinear",
    "NLinear",
    "LSTMForecaster",
    "RevIN",
    "Seq2SeqForecaster",
]
