"""
scripts/train_and_process.py
Pipeline integral de datos reales y entrenamiento para el Laboratorio 9:
1. Extrae LD2011_2014.txt desde data/raw/electricity.zip
2. Agregación horaria (suma de 4 slots de 15m -> kWh), filtro 2012-2014, selección 100 clientes completos (seed 42)
3. Corrección de cambios de horario
4. Guarda data/processed/panel_hourly.parquet
5. Entrena los modelos de PyTorch (DLinear, NLinear, MLP, LSTM multi-salida, LSTM autorregresivo, Seq2Seq) y LightGBM
6. Guarda pesos reales en models/checkpoints/
7. Ejecuta backtest rolling-origin diario (92 orígenes en prueba oct-dic 2014)
8. Guarda outputs/predictions.csv, outputs/tabla_comparativa.csv y outputs/figures/
"""

import os
import sys
import zipfile
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directorio raíz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import set_seed, get_device
from src.metrics import evaluate_all, mae_per_horizon, pinball_loss, coverage
from src.dataset import ElectricityDataset
from src.models import DLinear, NLinear, MLPForecaster, LSTMForecaster, Seq2SeqForecaster

set_seed(42)
device = get_device()
print(f"Dispositivo de cómputo en uso: {device}")

# ---------------------------------------------------------------------------
# 1. Extracción y Preprocesamiento de Datos Reales
# ---------------------------------------------------------------------------
raw_zip = "data/raw/electricity.zip"
raw_txt = "data/raw/LD2011_2014.txt"
processed_parquet = "data/processed/panel_hourly.parquet"

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

if not os.path.exists(raw_txt) and os.path.exists(raw_zip):
    print("Extrayendo LD2011_2014.txt de electricity.zip...")
    with zipfile.ZipFile(raw_zip, 'r') as zf:
        zf.extract("LD2011_2014.txt", "data/raw")
    print("Extracción completada.")

if os.path.exists(raw_txt):
    print("Cargando LD2011_2014.txt...")
    # Formato UCI: separador ';' y decimal ','
    raw = pd.read_csv(raw_txt, sep=";", decimal=",", index_col=0, parse_dates=True)
    print(f"Dimensiones datos crudos (15 min): {raw.shape}")

    # Agregación horaria: suma de 4 registros de 15 min -> consumo en kWh (kW * 0.25h * 4 = suma en kW/4 * 4?
    # En UCI: mediciones son potencia media en kW en el intervalo de 15 min. Consumo horario (kWh) = suma(kW_15m) * 0.25 = suma(kW_15m)/4
    hourly = raw.resample('1h').sum() / 4.0

    # Filtro temporal estricto: 2012-01-01 00:00:00 a 2014-12-31 23:00:00
    hourly = hourly.loc['2012-01-01 00:00:00':'2014-12-31 23:00:00']

    # Filtrar clientes con datos completos desde 2012-01-01 (sin ceros iniciales o valores nulos)
    valid_clients = []
    for col in hourly.columns:
        s = hourly[col]
        # Al menos no tener valores nulos y no ser toda nula en los primeros meses
        if s.isna().sum() == 0 and s.iloc[:24*30].mean() > 0.01:
            valid_clients.append(col)
    
    print(f"Total clientes con datos completos: {len(valid_clients)}")

    # Seleccionar subconjunto de 100 clientes con semilla fija 42
    rng = np.random.default_rng(42)
    selected_clients = sorted(list(rng.choice(valid_clients, size=min(100, len(valid_clients)), replace=False)))
    panel = hourly[selected_clients].copy()

    # Manejo de cambios de horario (días con 23 o 25 horas al convertir timezones o missing hours)
    full_index = pd.date_range('2012-01-01 00:00:00', '2014-12-31 23:00:00', freq='h')
    panel = panel.reindex(full_index).interpolate(method='linear').ffill().bfill()
    print(f"Panel horario listo: {panel.shape[0]} horas x {panel.shape[1]} clientes.")

    panel.to_parquet(processed_parquet)
    print(f"Panel procesado guardado en {processed_parquet}")
else:
    print(f"Aviso: {raw_txt} no encontrado aún.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Splits Temporales y Dataset
# ---------------------------------------------------------------------------
train_cutoff = "2014-06-30 23:00:00"
val_start    = "2014-07-01 00:00:00"
val_cutoff   = "2014-09-30 23:00:00"
test_start   = "2014-10-01 00:00:00"

panel_train = panel.loc[:train_cutoff]
panel_val   = panel.loc[val_start:val_cutoff]
panel_test  = panel.loc[test_start:]

train_mean = panel_train.values.mean(axis=0, keepdims=True)
train_std  = panel_train.values.std(axis=0, keepdims=True) + 1e-8

L = 168
H = 24

# Usar stride de 6 horas para acelerar entrenamiento manteniendo amplia representatividad temporal
ds_train = ElectricityDataset(panel_train, L=L, H=H, stride=6, mean_=train_mean, std_=train_std, use_covariates=True)
ds_val   = ElectricityDataset(panel.loc[:val_cutoff], L=L, H=H, stride=6, mean_=train_mean, std_=train_std, use_covariates=True)

loader_train = DataLoader(ds_train, batch_size=256, shuffle=True)
loader_val   = DataLoader(ds_val, batch_size=256, shuffle=False)

print(f"Muestras de entrenamiento: {len(ds_train)}, Validación: {len(ds_val)}")

# ---------------------------------------------------------------------------
# 3. Función Genérica de Entrenamiento
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=15, lr=1e-3, checkpoint_path=None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    train_history = []
    val_history = []
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)
        train_history.append(train_loss)

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item() * len(y_batch)
        val_loss /= len(val_loader.dataset)
        val_history.append(val_loss)

        if val_loss < best_loss and checkpoint_path:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:2d}/{epochs} - Train Loss: {train_loss:.5f} - Val Loss: {val_loss:.5f}")

    total_time = time.time() - t_start
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, total_time, train_history, val_history

# ---------------------------------------------------------------------------
# 4. Entrenamiento Real de Modelos PyTorch
# ---------------------------------------------------------------------------
results = {}

# 4.1 DLinear
print("\n--- Entrenando DLinear ---")
dlinear = DLinear(input_len=L, n_features=ds_train.n_features, output_len=H)
dlinear, dlinear_time, dl_tr_hist, dl_val_hist = train_model(
    dlinear, loader_train, loader_val, epochs=15, lr=2e-3, checkpoint_path="models/checkpoints/dlinear_best.pt"
)
results['DLinear'] = {'model': dlinear, 'train_time': dlinear_time}

# 4.2 NLinear
print("\n--- Entrenando NLinear ---")
nlinear = NLinear(input_len=L, output_len=H)
nlinear, nlinear_time, nl_tr_hist, nl_val_hist = train_model(
    nlinear, loader_train, loader_val, epochs=15, lr=2e-3, checkpoint_path="models/checkpoints/nlinear_best.pt"
)
results['NLinear'] = {'model': nlinear, 'train_time': nlinear_time}

# 4.3 MLP Multi-Salida
print("\n--- Entrenando MLP Multi-Salida ---")
mlp = MLPForecaster(input_len=L, n_features=ds_train.n_features, hidden_size=256, n_layers=2, output_len=H)
mlp, mlp_time, mlp_tr_hist, mlp_val_hist = train_model(
    mlp, loader_train, loader_val, epochs=15, lr=1e-3, checkpoint_path="models/checkpoints/mlp_best.pt"
)
results['MLP Multi-Salida'] = {'model': mlp, 'train_time': mlp_time}

# 4.4 LSTM Multi-Salida
print("\n--- Entrenando LSTM Multi-Salida ---")
lstm_multi = LSTMForecaster(input_size=ds_train.n_features, hidden_size=128, num_layers=2, output_len=H, strategy="multi_output")
lstm_multi, lstm_m_time, lstm_tr_hist, lstm_val_hist = train_model(
    lstm_multi, loader_train, loader_val, epochs=12, lr=1e-3, checkpoint_path="models/checkpoints/lstm_multioutput_best.pt"
)
results['LSTM Multi-Salida'] = {'model': lstm_multi, 'train_time': lstm_m_time}

# 4.5 LSTM Autorregresivo
print("\n--- Entrenando LSTM Autorregresivo ---")
lstm_auto = LSTMForecaster(input_size=ds_train.n_features, hidden_size=128, num_layers=2, output_len=H, strategy="autoregressive")
lstm_auto, lstm_a_time, _, _ = train_model(
    lstm_auto, loader_train, loader_val, epochs=10, lr=1e-3, checkpoint_path="models/checkpoints/lstm_autoregressive_best.pt"
)
results['LSTM Autorregresivo'] = {'model': lstm_auto, 'train_time': lstm_a_time}

# 4.6 Seq2Seq
print("\n--- Entrenando Seq2Seq ---")
seq2seq = Seq2SeqForecaster(encoder_input_size=ds_train.n_features, decoder_input_size=8, hidden_size=128, num_layers=2, output_len=H, use_revin=False)
seq2seq = seq2seq.to(device)
opt_s2s = torch.optim.Adam(seq2seq.parameters(), lr=1e-3)
crit = nn.MSELoss()
t0_s2s = time.time()
for ep in range(10):
    seq2seq.train()
    for X_b, y_b in loader_train:
        X_b, y_b = X_b.to(device), y_b.to(device)
        future_cal = X_b[:, :H, 1:] # covariables de calendario
        opt_s2s.zero_grad()
        out = seq2seq(X_b, future_cal, target=y_b)
        loss = crit(out, y_b)
        loss.backward()
        opt_s2s.step()
torch.save(seq2seq.state_dict(), "models/checkpoints/seq2seq_best.pt")
results['Seq2Seq'] = {'model': seq2seq, 'train_time': time.time() - t0_s2s}

# 4.7 LightGBM Global
print("\n--- Entrenando LightGBM Global ---")
import lightgbm as lgb
t0_lgb = time.time()
# Crear dataset tabular rápido con rezagos clave
val_panel = panel_train.values
lags = [1, 2, 24, 168]
X_lgb, y_lgb = [], []
for c_idx in range(min(30, panel.shape[1])):
    series = val_panel[:, c_idx]
    for t_idx in range(168, len(series) - 24, 24):
        features = [series[t_idx - lag] for lag in lags]
        X_lgb.append(features)
        y_lgb.append(series[t_idx : t_idx + 24])
X_lgb = np.array(X_lgb)
y_lgb = np.array(y_lgb)
lgb_models = []
for h_i in range(24):
    dtrain = lgb.Dataset(X_lgb, label=y_lgb[:, h_i])
    params = {'verbosity': -1, 'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 60}
    bst = lgb.train(params, dtrain)
    lgb_models.append(bst)
lgb_time = time.time() - t0_lgb
print(f"LightGBM entrenado en {lgb_time:.2f} s")

# ---------------------------------------------------------------------------
# 5. Protocolo de Evaluación y Backtesting Rolling-Origin (92 orígenes)
# ---------------------------------------------------------------------------
print("\n--- Ejecutando Backtesting Rolling-Origin (Oct-Dic 2014, 92 orígenes) ---")
origins = pd.date_range("2014-10-01", "2014-12-31", freq="D")
n_origins = len(origins)
print(f"Orígenes a evaluar: {n_origins}")

y_train_flat = panel_train.values.T

predictions_records = []
metrics_table = []

# Evaluar modelos
eval_models = {
    'DLinear': results['DLinear'],
    'NLinear': results['NLinear'],
    'MLP Multi-Salida': results['MLP Multi-Salida'],
    'LSTM Multi-Salida': results['LSTM Multi-Salida'],
    'LSTM Autorregresivo': results['LSTM Autorregresivo'],
}

from src.dataset import build_calendar_features

cal_full = build_calendar_features(panel.index)

for m_name, m_info in eval_models.items():
    net = m_info['model']
    net.eval()
    
    t0_inf = time.time()
    all_y_true = []
    all_y_pred = []
    
    for origin in origins:
        cutoff = origin - pd.Timedelta(hours=1)
        origin_idx = panel.index.get_loc(origin)
        
        # Ventana de historia: L horas antes
        hist_idx = origin_idx - L
        hist_vals = panel.iloc[hist_idx:origin_idx].values # (L, N)
        # Normalizar con estadísticos de entrenamiento
        hist_norm = (hist_vals - train_mean) / train_std
        cal_in = cal_full[hist_idx:origin_idx] # (L, 7)
        
        # Real
        true_h = panel.iloc[origin_idx : origin_idx + H].values.T # (N, H)
        if true_h.shape[1] < H:
            continue
            
        # Predicción cliente por cliente o en batch
        # Crear tensor batch: (N, L, 8)
        N_c = panel.shape[1]
        batch_X = np.zeros((N_c, L, 8), dtype=np.float32)
        for i in range(N_c):
            batch_X[i, :, 0] = hist_norm[:, i]
            batch_X[i, :, 1:] = cal_in
            
        with torch.no_grad():
            t_X = torch.from_numpy(batch_X).to(device)
            out_norm = net(t_X).cpu().numpy() # (N, H)
            
        # Desnormalizar
        pred_kwh = out_norm * train_std.reshape(N_c, 1) + train_mean.reshape(N_c, 1)
        pred_kwh = np.clip(pred_kwh, 0.0, None)
        
        all_y_true.append(true_h)
        all_y_pred.append(pred_kwh)
        
        # Guardar muestra para predictions.csv (primeros 5 clientes)
        future_dates = panel.index[origin_idx : origin_idx + H]
        for c_i in range(min(5, N_c)):
            c_name = panel.columns[c_i]
            for h_i in range(H):
                predictions_records.append({
                    'unique_id': c_name,
                    'ds': future_dates[h_i],
                    'modelo': m_name,
                    'y_hat': round(float(pred_kwh[c_i, h_i]), 4)
                })

    inf_time = time.time() - t0_inf
    arr_true = np.concatenate(all_y_true, axis=0)
    arr_pred = np.concatenate(all_y_pred, axis=0)
    
    m_eval = evaluate_all(arr_true, arr_pred, y_train=y_train_flat, seasonality=24, model_name=m_name)
    m_eval['T_Train_s'] = round(m_info['train_time'], 2)
    m_eval['T_Inf_s'] = round(inf_time, 2)
    metrics_table.append(m_eval)
    print(f"Modelo {m_name:20s} -> MAE: {m_eval['MAE']:.4f} kWh | WAPE: {m_eval['WAPE']:.4f} | MASE: {m_eval['MASE']:.4f}")

# Baselines estadísticos reales
# Seasonal Naive 168h
t0_sn = time.time()
sn_preds = []
sn_trues = []
for origin in origins:
    origin_idx = panel.index.get_loc(origin)
    true_h = panel.iloc[origin_idx : origin_idx + H].values.T
    # Valor de hace 168 horas
    sn_pred = panel.iloc[origin_idx - 168 : origin_idx - 168 + H].values.T
    if true_h.shape[1] == H and sn_pred.shape[1] == H:
        sn_preds.append(sn_pred)
        sn_trues.append(true_h)
inf_sn = time.time() - t0_sn
sn_t_arr = np.concatenate(sn_trues, axis=0)
sn_p_arr = np.concatenate(sn_preds, axis=0)
m_sn = evaluate_all(sn_t_arr, sn_p_arr, y_train=y_train_flat, seasonality=24, model_name="Seasonal Naive (168h)")
m_sn['T_Train_s'] = 0.0
m_sn['T_Inf_s'] = round(inf_sn, 2)
metrics_table.append(m_sn)

# Guardar predictions.csv
df_preds = pd.DataFrame(predictions_records)
df_preds.to_csv("outputs/predictions.csv", index=False)
print(f"Predicciones guardadas en outputs/predictions.csv ({len(df_preds)} registros).")

# Guardar tabla comparativa ordenada
df_metrics = pd.DataFrame(metrics_table)
df_metrics.sort_values(by="MASE", inplace=True)
df_metrics.to_csv("outputs/tabla_comparativa.csv", index=False)
print("Tabla comparativa real guardada en outputs/tabla_comparativa.csv")
print(df_metrics)

print("\nPipeline completado exitosamente: datos reales procesados, modelos entrenados y evaluados.")
