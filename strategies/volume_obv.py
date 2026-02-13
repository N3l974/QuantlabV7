"""Volume Profile + OBV Strategy."""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


def ema(data: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


class VolumeOBV(BaseStrategy):
    name = "Volume Profile + OBV"
    strategy_type = "volume"
    default_params = {
        "obv_ema_period": 20,
        "volume_spike_threshold": 2.0,
        "confirmation_period": 3,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.02,
        "atr_sl_mult": 0.0,
        "atr_tp_mult": 0.0,
        "atr_period": 14,
        "trailing_atr_mult": 0.0,
        "max_holding_bars": 0,
        "breakeven_trigger_pct": 0.0,
    }

    def generate_signals(self, data: pd.DataFrame, params: dict) -> np.ndarray:
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        volume = data["volume"].values.astype(np.float64)
        n = len(close)

        obv_ema_p = int(params["obv_ema_period"])
        spike_thresh = params["volume_spike_threshold"]
        confirm_p = int(params["confirmation_period"])
        sl = params["stop_loss_pct"]
        tp = params["take_profit_pct"]
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        atr_period = int(params.get("atr_period", 14))
        high = data["high"].values.astype(np.float64) if "high" in data.columns else close
        low_arr = data["low"].values.astype(np.float64) if "low" in data.columns else close
        atr = self.compute_atr(high, low_arr, close, atr_period)

        # Compute OBV
        obv = np.zeros(n)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        obv_ema_line = ema(obv, obv_ema_p)

        # Compute volume SMA for spike detection
        vol_sma = np.full(n, np.nan)
        for i in range(obv_ema_p - 1, n):
            vol_sma[i] = np.mean(volume[i - obv_ema_p + 1 : i + 1])

        warmup = obv_ema_p + confirm_p + 1
        signals = np.zeros(n)
        position = 0
        entry_price = 0.0

        for i in range(warmup, n):
            if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
                signals[i] = position
                continue

            volume_spike = volume[i] > spike_thresh * vol_sma[i]
            obv_bullish = obv[i] > obv_ema_line[i]
            obv_bearish = obv[i] < obv_ema_line[i]

            # Price confirmation: close trending in same direction for confirm_p candles
            price_up = all(close[i - j] > close[i - j - 1] for j in range(confirm_p))
            price_down = all(close[i - j] < close[i - j - 1] for j in range(confirm_p))

            if position == 0:
                if volume_spike and obv_bullish and price_up:
                    position = 1
                    entry_price = close[i]
                elif volume_spike and obv_bearish and price_down:
                    position = -1
                    entry_price = close[i]
            elif position == 1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (close[i] - entry_price) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or obv[i] < obv_ema_line[i]:
                    position = 0
            elif position == -1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (entry_price - close[i]) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or obv[i] > obv_ema_line[i]:
                    position = 0

            signals[i] = position

        return self._apply_advanced_exits(signals, data, params)

    def generate_signals_v5(self, data: pd.DataFrame, params: dict):
        """V5 API: returns (signals, sl_distances) for risk-based sizing."""
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64) if "high" in data.columns else close
        low_arr = data["low"].values.astype(np.float64) if "low" in data.columns else close
        n = len(close)
        atr_period = int(params.get("atr_period", 14))
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr = self.compute_atr(high, low_arr, close, atr_period)
        signals = self.generate_signals(data, params)
        sl_distances = np.full(n, np.nan)
        for i in range(atr_period + 1, n):
            if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                sl_distances[i] = atr_sl * atr[i] / close[i]
            elif float(params["stop_loss_pct"]) > 0:
                sl_distances[i] = float(params["stop_loss_pct"])
        return signals, sl_distances
