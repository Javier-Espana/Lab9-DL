"""
src/models/seq2seq.py
Encoder-Decoder (Seq2Seq) con covariables de calendario en el decoder.

Arquitectura:
  Encoder: LSTM que procesa la ventana de historia X (L pasos).
  Decoder: LSTM que genera H pasos usando como input en cada paso:
           [predicción anterior (o cero en el primer paso)] + covariables futuras.
  El estado oculto del encoder inicializa el decoder.

Estrategia multi-paso: encoder–decoder (la tercera estrategia del lab).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.lstm import RevIN


class Seq2SeqForecaster(nn.Module):
    """
    Encoder-Decoder LSTM con covariables de calendario en el decoder.

    Args:
        encoder_input_size  : features de la ventana de entrada (serie + cal. hist.).
        decoder_input_size  : features del decoder (1 pred. anterior + cal. futuras).
        hidden_size         : dimensión del estado oculto (compartido enc/dec).
        num_layers          : capas apiladas.
        output_len          : H.
        dropout             : dropout entre capas.
        use_revin           : aplicar RevIN a la serie de entrada.
        teacher_forcing_ratio: probabilidad de usar el valor real durante
                               entrenamiento (teacher forcing). En inferencia
                               siempre se usa la predicción anterior.
    """

    def __init__(
        self,
        encoder_input_size: int = 8,
        decoder_input_size: int = 8,
        hidden_size: int = 256,
        num_layers: int = 2,
        output_len: int = 24,
        dropout: float = 0.2,
        use_revin: bool = True,
        teacher_forcing_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.output_len = output_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(n_features=1)

        self.encoder = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        future_cal: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x          : (batch, L, encoder_input_size) — ventana histórica.
            future_cal : (batch, H, cal_features)        — covariables futuras conocidas.
            target     : (batch, H) — valores reales (solo en entrenamiento,
                         para teacher forcing). None en inferencia.

        Returns:
            (batch, H)
        """
        import random

        batch_size = x.shape[0]
        cal_features = future_cal.shape[-1]

        # ── RevIN ──────────────────────────────────────────────────────────────
        if self.use_revin:
            x_series = x[:, :, :1]
            x_norm, mean_, std_ = self.revin.normalize(x_series)
            x = torch.cat([x_norm, x[:, :, 1:]], dim=-1)

        # ── Encoder ────────────────────────────────────────────────────────────
        _, hidden = self.encoder(x)   # hidden: (h_n, c_n), shape (layers, B, H_dim)

        # ── Decoder ────────────────────────────────────────────────────────────
        # Primer input: cero (no hay predicción anterior)
        dec_out = torch.zeros(batch_size, 1, device=x.device)
        h_state = hidden
        preds   = []

        for t in range(self.output_len):
            cal_t = future_cal[:, t, :]   # (batch, cal_features)
            dec_in = torch.cat(
                [dec_out, cal_t], dim=-1
            ).unsqueeze(1)                # (batch, 1, 1 + cal_features)

            out_t, h_state = self.decoder(dec_in, h_state)
            pred_t = self.output_proj(out_t[:, -1, :]).squeeze(-1)   # (batch,)
            preds.append(pred_t)

            # Teacher forcing en entrenamiento
            if target is not None and random.random() < self.teacher_forcing_ratio:
                dec_out = target[:, t]
            else:
                dec_out = pred_t.detach()

        out = torch.stack(preds, dim=1)   # (batch, H)

        # ── RevIN inverso ──────────────────────────────────────────────────────
        if self.use_revin:
            out = out * std_[:, 0, :] + mean_[:, 0, :]

        return out
