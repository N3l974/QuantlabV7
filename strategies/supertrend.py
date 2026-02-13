"""
SuperTrend Strategy — ATR-based trend following with dynamic trailing stop.

SuperTrend flips between bullish/bearish based on ATR bands around price.
Very popular in crypto for its simplicity and effectiveness on trending markets.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class SuperTrend(BaseStrategy):
    name = "SuperTrend"
    strategy_type = "trend_following"
    default_params = {
        "atr_period": 10,
        "multiplier": 3.0,
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.08,
        "atr_sl_mult": 0.0,
        "atr_tp_mult": 0.0,
        "trailing_atr_mult": 0.0,
        "max_holding_bars": 0,
        "breakeven_trigger_pct": 0.0,
    }

    def _compute_supertrend(self, close, high, low, atr_period, multiplier):
        """Compute SuperTrend indicator, returns (supertrend_dir, atr, warmup)."""
        n = len(close)
        atr = self.compute_atr(high, low, close, atr_period)

        hl2 = (high + low) / 2.0
        upper_band = np.zeros(n)
        lower_band = np.zeros(n)
        supertrend = np.zeros(n)

        warmup = atr_period + 1
        for i in range(warmup, n):
            basic_upper = hl2[i] + multiplier * atr[i]
            basic_lower = hl2[i] - multiplier * atr[i]

            if basic_upper < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
                upper_band[i] = basic_upper
            else:
                upper_band[i] = upper_band[i - 1]

            if basic_lower > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
                lower_band[i] = basic_lower
            else:
                lower_band[i] = lower_band[i - 1]

            if supertrend[i - 1] == 1:
                if close[i] < lower_band[i]:
                    supertrend[i] = -1
                else:
                    supertrend[i] = 1
            else:
                if close[i] > upper_band[i]:
                    supertrend[i] = 1
                else:
                    supertrend[i] = -1

        return supertrend, atr, warmup

    def generate_signals(self, data: pd.DataFrame, params: dict) -> np.ndarray:
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(close)

        atr_period = int(params["atr_period"])
        multiplier = float(params["multiplier"])
        sl_pct = float(params["stop_loss_pct"])
        tp_pct = float(params["take_profit_pct"])
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))

        supertrend, atr, warmup = self._compute_supertrend(close, high, low, atr_period, multiplier)

        signals = np.zeros(n, dtype=np.float64)
        position = 0.0
        entry_price = 0.0

        for i in range(warmup, n):
            if position != 0 and entry_price > 0:
                # ATR-based SL/TP if configured, else percentage
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_dist = atr_sl * atr[i] / close[i]
                    tp_dist = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp_pct
                else:
                    sl_dist = sl_pct
                    tp_dist = tp_pct

                pnl = position * (close[i] - entry_price) / entry_price
                if pnl <= -sl_dist or pnl >= tp_dist:
                    position = 0.0
                    entry_price = 0.0
                    signals[i] = 0.0
                    continue

            new_pos = float(supertrend[i])
            if new_pos != position:
                position = new_pos
                entry_price = close[i]

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

        _, atr, _ = self._compute_supertrend(close, high, low, atr_period, float(params["multiplier"]))

        signals = self.generate_signals(data, params)

        # Compute per-bar SL distance as fraction of price
        sl_distances = np.full(n, np.nan)
        for i in range(atr_period + 1, n):
            if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                sl_distances[i] = atr_sl * atr[i] / close[i]
            elif float(params["stop_loss_pct"]) > 0:
                sl_distances[i] = float(params["stop_loss_pct"])

        return signals, sl_distances
