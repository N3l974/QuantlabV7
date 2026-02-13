"""VWAP Deviation Mean-Reversion Strategy."""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class VWAPDeviation(BaseStrategy):
    name = "VWAP Deviation"
    strategy_type = "mean_reversion"
    default_params = {
        "vwap_period": 20,
        "deviation_threshold": 1.5,
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

        period = int(params["vwap_period"])
        threshold = params["deviation_threshold"]
        sl = params["stop_loss_pct"]
        tp = params["take_profit_pct"]
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        atr_period = int(params.get("atr_period", 14))
        high_arr = data["high"].values.astype(np.float64)
        low_arr = data["low"].values.astype(np.float64)
        atr = self.compute_atr(high_arr, low_arr, close, atr_period)

        # Compute rolling VWAP
        typical_price = (data["high"].values + data["low"].values + close) / 3.0
        tp_vol = typical_price * volume

        vwap = np.full(n, np.nan)
        vwap_std = np.full(n, np.nan)

        for i in range(period - 1, n):
            window_tpv = tp_vol[i - period + 1 : i + 1]
            window_vol = volume[i - period + 1 : i + 1]
            vol_sum = np.sum(window_vol)
            if vol_sum > 0:
                vwap[i] = np.sum(window_tpv) / vol_sum
            else:
                vwap[i] = close[i]
            window_close = close[i - period + 1 : i + 1]
            vwap_std[i] = np.std(window_close - vwap[i], ddof=1) if not np.isnan(vwap[i]) else 0

        signals = np.zeros(n)
        position = 0
        entry_price = 0.0

        for i in range(period, n):
            if np.isnan(vwap[i]) or vwap_std[i] == 0:
                signals[i] = position
                continue

            deviation = (close[i] - vwap[i]) / vwap_std[i] if vwap_std[i] > 0 else 0

            if position == 0:
                if deviation < -threshold:
                    position = 1  # Price below VWAP -> buy
                    entry_price = close[i]
                elif deviation > threshold:
                    position = -1  # Price above VWAP -> sell
                    entry_price = close[i]
            elif position == 1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (close[i] - entry_price) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or deviation > 0:
                    position = 0
            elif position == -1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (entry_price - close[i]) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or deviation < 0:
                    position = 0

            signals[i] = position

        return self._apply_advanced_exits(signals, data, params)

    def generate_signals_v5(self, data: pd.DataFrame, params: dict):
        """V5 API: returns (signals, sl_distances) for risk-based sizing."""
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        high_arr = data["high"].values.astype(np.float64)
        low_arr = data["low"].values.astype(np.float64)
        n = len(close)
        atr_period = int(params.get("atr_period", 14))
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr = self.compute_atr(high_arr, low_arr, close, atr_period)
        signals = self.generate_signals(data, params)
        sl_distances = np.full(n, np.nan)
        for i in range(atr_period + 1, n):
            if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                sl_distances[i] = atr_sl * atr[i] / close[i]
            elif float(params["stop_loss_pct"]) > 0:
                sl_distances[i] = float(params["stop_loss_pct"])
        return signals, sl_distances
