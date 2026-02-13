"""MACD Crossover Strategy."""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA using vectorized approach."""
    alpha = 2.0 / (period + 1)
    result = np.zeros_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


class MACDCrossover(BaseStrategy):
    name = "MACD Crossover"
    strategy_type = "trend_following"
    default_params = {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
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

        fast_p = int(params["fast_period"])
        slow_p = int(params["slow_period"])
        signal_p = int(params["signal_period"])
        sl = params["stop_loss_pct"]
        tp = params["take_profit_pct"]
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        atr_period = int(params.get("atr_period", 14))

        high = data["high"].values.astype(np.float64) if "high" in data.columns else close
        low = data["low"].values.astype(np.float64) if "low" in data.columns else close
        atr = self.compute_atr(high, low, close, atr_period)

        fast_ema = ema(close, fast_p)
        slow_ema = ema(close, slow_p)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, signal_p)
        histogram = macd_line - signal_line

        warmup = slow_p + signal_p
        signals = np.zeros(n)
        position = 0
        entry_price = 0.0

        for i in range(warmup, n):
            if position == 0:
                if histogram[i] > 0 and histogram[i - 1] <= 0:
                    position = 1
                    entry_price = close[i]
                elif histogram[i] < 0 and histogram[i - 1] >= 0:
                    position = -1
                    entry_price = close[i]
            elif position == 1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (close[i] - entry_price) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or (histogram[i] < 0 and histogram[i - 1] >= 0):
                    position = 0
            elif position == -1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (entry_price - close[i]) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or (histogram[i] > 0 and histogram[i - 1] <= 0):
                    position = 0

            signals[i] = position

        return self._apply_advanced_exits(signals, data, params)

    def generate_signals_v5(self, data: pd.DataFrame, params: dict):
        """V5 API: returns (signals, sl_distances) for risk-based sizing."""
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64) if "high" in data.columns else close
        low = data["low"].values.astype(np.float64) if "low" in data.columns else close
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
