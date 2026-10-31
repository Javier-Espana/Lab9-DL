# CC3092 – Lab 9: Series de Tiempo y Forecasting

## Descripción
Pronóstico del consumo eléctrico horario (dataset **ElectricityLoadDiagrams20112014** del UCI ML Repository) para 370 clientes de Portugal (2011–2014), usando múltiples familias de modelos: baselines estadísticos, árboles, redes lineales, recurrentes, convolucionales, transformers, probabilísticos y fundacionales (Chronos zero-shot).

## Estructura del proyecto

```
Lab9-DL/
├── docs/                        # Enunciado del laboratorio
├── data/
│   ├── raw/                     # Datos crudos (excluidos del repo, ver .gitignore)
│   └── processed/               # Series horarias limpias (.parquet)
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Sección 2: Carga, limpieza, Dataset, auditoría
│   ├── 02_research.ipynb              # Sección 3: Investigación de herramientas
│   ├── 03_backtest_protocol.ipynb     # Sección 4: Protocolo y métricas
│   ├── 04_models_baselines.ipynb      # Sección 5: Baselines (Naive, MSTL, AutoETS)
│   ├── 05_models_lightgbm.ipynb       # Sección 5: LightGBM global
│   ├── 06_models_mlp_linear.ipynb     # Sección 5: MLP, DLinear, NLinear
│   ├── 07_models_lstm_seq2seq.ipynb   # Sección 5: LSTM/GRU + Seq2Seq (5.1, 5.2, 5.3)
│   ├── 08_models_nhits_tcn.ipynb      # Sección 5: N-HiTS, TCN (NeuralForecast)
│   ├── 09_models_transformers.ipynb   # Sección 5: PatchTST, TFT/iTransformer
│   ├── 10_models_probabilistic.ipynb  # Sección 5: DeepAR + Chronos zero-shot
│   └── 11_comparison_discussion.ipynb # Secciones 6, 7, 8: Tablas, gráficas, discusión
├── src/
│   ├── __init__.py
│   ├── dataset.py               # ElectricityDataset (PyTorch)
│   ├── metrics.py               # MAE, RMSE, WAPE, MASE, MAPE, Pinball
│   ├── backtest.py              # Función backtest(modelo)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py               # MLP multi-salida
│   │   ├── linear.py            # DLinear, NLinear
│   │   ├── lstm.py              # LSTM/GRU (3 estrategias multi-paso)
│   │   └── seq2seq.py           # Encoder-Decoder con covariables
│   └── utils.py                 # Semilla global, helpers
├── models/
│   └── checkpoints/             # Pesos entrenados (.pt) – en .gitignore los intermedios
├── outputs/
│   ├── predictions.csv          # unique_id, ds, modelo, y_hat (entregable final)
│   └── figures/                 # Gráficas guardadas (.png / .pdf)
├── requirements.txt
└── README.md
```

## Splits del problema

| Conjunto      | Rango                       |
|---------------|-----------------------------|
| Entrenamiento | 2012-01-01 – 2014-06-30     |
| Validación    | 2014-07-01 – 2014-09-30     |
| Prueba        | 2014-10-01 – 2014-12-31 (92 orígenes, backtesting diario) |

**L = 168 h** (1 semana de historia), **H = 24 h** (day-ahead).

## Modelos implementados

| Familia          | Modelos                                      | Librería         |
|------------------|----------------------------------------------|------------------|
| Baselines        | Naive, Seasonal Naive (24h/168h), MSTL/AutoETS | StatsForecast  |
| Árboles          | LightGBM global (rezagos + calendario)       | LightGBM         |
| MLP y lineales   | MLP multi-salida, DLinear, NLinear           | PyTorch          |
| Recurrentes      | LSTM/GRU (3 estrategias multi-paso)          | PyTorch          |
| Encoder-Decoder  | Seq2Seq con covariables en el decoder        | PyTorch          |
| Residuales/Conv  | N-HiTS, TCN                                  | NeuralForecast   |
| Atención         | PatchTST + TFT o iTransformer               | NeuralForecast   |
| Probabilísticos  | DeepAR                                       | NeuralForecast   |
| Fundacional      | Chronos-Bolt (zero-shot)                     | chronos-forecasting |

## Cómo reproducir

```bash
# 1. Crear entorno
python -m venv venv && source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar notebooks en orden (01 → 11)
jupyter lab
```

> **Nota:** Los datos se descargan automáticamente en el notebook 01. Se necesita ~1 GB de disco.

## Entrega
- **Fecha límite:** Domingo 1 de noviembre, 2026, 23:59
- **Entregable:** PDF (máx. 5 páginas) + este repositorio Git
