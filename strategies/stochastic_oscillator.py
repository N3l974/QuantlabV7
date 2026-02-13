"""Stochastic Oscillator Strategy."""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class StochasticOscillator(BaseStrategy):
    name = "Stochastic Oscillator"
    strategy_type = "mean_reversion"
    default_params = {
        "k_period": 14,
        "d_period": 3,
        "oversold": 20,
        "overbought": 80,
        "zone_buffer": 20,
        "stop_loss_pct": 0.015,
        "take_profit_pct": 0.03,
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
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(close)

        k_period = int(params["k_period"])
        d_period = int(params["d_period"])
        oversold = params["oversold"]
        overbought = params["overbought"]
        zone_buffer = params.get("zone_buffer", 20)
        sl = params["stop_loss_pct"]
        tp = params["take_profit_pct"]
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        atr_period = int(params.get("atr_period", 14))
        atr = self.compute_atr(high, low, close, atr_period)

        # Compute %K
        k_values = np.full(n, np.nan)
        for i in range(max(k_period - 1, 0), n):
            start = max(0, i - k_period + 1)
            if start >= i + 1:
                continue
            highest = np.max(high[start : i + 1])
            lowest = np.min(low[start : i + 1])
            if highest != lowest:
                k_values[i] = 100.0 * (close[i] - lowest) / (highest - lowest)
            else:
                k_values[i] = 50.0

        # Compute %D (SMA of %K)
        d_values = np.full(n, np.nan)
        for i in range(k_period - 1 + d_period - 1, n):
            window = k_values[i - d_period + 1 : i + 1]
            if not np.any(np.isnan(window)):
                d_values[i] = np.mean(window)

        warmup = k_period + d_period
        signals = np.zeros(n)
        position = 0
        entry_price = 0.0

        for i in range(warmup, n):
            if np.isnan(k_values[i]) or np.isnan(d_values[i]):
                signals[i] = position
                continue

            if position == 0:
                # Bullish: %K crosses above %D in oversold zone
                if (k_values[i] > d_values[i] and k_values[i - 1] <= d_values[i - 1]
                        and k_values[i] < oversold + zone_buffer):
                    position = 1
                    entry_price = close[i]
                # Bearish: %K crosses below %D in overbought zone
                elif (k_values[i] < d_values[i] and k_values[i - 1] >= d_values[i - 1]
                      and k_values[i] > overbought - zone_buffer):
                    position = -1
                    entry_price = close[i]
            elif position == 1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (close[i] - entry_price) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or k_values[i] > overbought:
                    position = 0
            elif position == -1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (entry_price - close[i]) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or k_values[i] < oversold:
                    position = 0

            signals[i] = position

        return self._apply_advanced_exits(signals, data, params)

    def generate_signals_v5(self, data: pd.DataFrame, params: dict):
        """V5 API: returns (signals, sl_distances) for risk-based sizing."""
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(close)
        atr_period = int(params.get("atr_period", 14))
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr = self.compute_atr(high, low, close, atr_period)
        signals = self.generate_signals(data, params)
        sl_distances = np.full(n, np.nan)
        for i in range(atr_period + 1, n):
            if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                sl_distances[i] = atr_sl * atr[i] / close[i]
            elif float(params["stop_loss_pct"]) > 0:
                sl_distances[i] = float(params["stop_loss_pct"])
        return signals, sl_distances
