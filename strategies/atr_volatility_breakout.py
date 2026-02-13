"""ATR Volatility Breakout Strategy."""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class ATRVolatilityBreakout(BaseStrategy):
    name = "ATR Volatility Breakout"
    strategy_type = "volatility"
    default_params = {
        "atr_period": 14,
        "atr_multiplier": 2.0,
        "lookback_period": 20,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "atr_sl_mult": 0.0,
        "atr_tp_mult": 0.0,
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

        atr_period = int(params["atr_period"])
        atr_mult = params["atr_multiplier"]
        lookback = int(params["lookback_period"])
        sl = params["stop_loss_pct"]
        tp = params["take_profit_pct"]
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))

        # Compute ATR
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        atr = np.full(n, np.nan)
        atr[atr_period] = np.mean(tr[1 : atr_period + 1])
        for i in range(atr_period + 1, n):
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

        # Compute reference level (SMA of close over lookback)
        ref = np.full(n, np.nan)
        for i in range(lookback - 1, n):
            ref[i] = np.mean(close[i - lookback + 1 : i + 1])

        warmup = max(atr_period, lookback) + 1
        signals = np.zeros(n)
        position = 0
        entry_price = 0.0

        for i in range(warmup, n):
            if np.isnan(atr[i]) or np.isnan(ref[i]):
                signals[i] = position
                continue

            upper_band = ref[i] + atr_mult * atr[i]
            lower_band = ref[i] - atr_mult * atr[i]

            if position == 0:
                if close[i] > upper_band:
                    position = 1
                    entry_price = close[i]
                elif close[i] < lower_band:
                    position = -1
                    entry_price = close[i]
            elif position == 1:
                if atr_sl > 0 and not np.isnan(atr[i]) and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (close[i] - entry_price) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or close[i] < ref[i]:
                    position = 0
            elif position == -1:
                if atr_sl > 0 and not np.isnan(atr[i]) and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (entry_price - close[i]) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or close[i] > ref[i]:
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
        atr_period = int(params["atr_period"])
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
