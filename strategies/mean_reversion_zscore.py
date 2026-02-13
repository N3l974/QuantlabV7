"""
Z-Score Mean Reversion Strategy — Statistical mean reversion.

Computes rolling z-score of price vs moving average.
Enters when z-score is extreme (oversold/overbought), exits at mean.
Pure statistical approach complementary to RSI-based mean reversion.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class MeanReversionZScore(BaseStrategy):
    name = "Z-Score Mean Reversion"
    strategy_type = "statistical"
    default_params = {
        "lookback": 50,
        "entry_zscore": 2.0,
        "exit_zscore": 0.5,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.05,
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
        n = len(close)

        lookback = int(params["lookback"])
        entry_z = float(params["entry_zscore"])
        exit_z = float(params["exit_zscore"])
        sl_pct = float(params["stop_loss_pct"])
        tp_pct = float(params["take_profit_pct"])
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        atr_period = int(params.get("atr_period", 14))
        high = data["high"].values.astype(np.float64) if "high" in data.columns else close
        low_arr = data["low"].values.astype(np.float64) if "low" in data.columns else close
        atr = self.compute_atr(high, low_arr, close, atr_period)

        # Rolling mean and std
        rolling_mean = np.full(n, np.nan)
        rolling_std = np.full(n, np.nan)
        for i in range(lookback - 1, n):
            window = close[i - lookback + 1:i + 1]
            rolling_mean[i] = np.mean(window)
            rolling_std[i] = np.std(window, ddof=1)

        # Z-score
        zscore = np.full(n, 0.0)
        for i in range(lookback - 1, n):
            if rolling_std[i] > 0:
                zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]

        signals = np.zeros(n, dtype=np.float64)
        position = 0.0
        entry_price = 0.0

        for i in range(lookback, n):
            # SL/TP
            if position != 0 and entry_price > 0:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp_pct
                else:
                    sl_d, tp_d = sl_pct, tp_pct
                pnl = position * (close[i] - entry_price) / entry_price
                if pnl <= -sl_d or pnl >= tp_d:
                    position = 0.0
                    entry_price = 0.0
                    signals[i] = 0.0
                    continue

            if position == 0:
                if zscore[i] < -entry_z:
                    position = 1.0  # Oversold → long
                    entry_price = close[i]
                elif zscore[i] > entry_z:
                    position = -1.0  # Overbought → short
                    entry_price = close[i]
            elif position == 1.0 and zscore[i] > -exit_z:
                position = 0.0
                entry_price = 0.0
            elif position == -1.0 and zscore[i] < exit_z:
                position = 0.0
                entry_price = 0.0

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
