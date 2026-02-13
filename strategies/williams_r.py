"""
Williams %R Strategy — Momentum oscillator for overbought/oversold detection.

Williams %R ranges from -100 to 0. Values near -100 = oversold, near 0 = overbought.
Different math from Stochastic but complementary signal generation.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class WilliamsR(BaseStrategy):
    name = "Williams %R"
    strategy_type = "momentum"
    default_params = {
        "period": 14,
        "oversold_threshold": 80,
        "overbought_threshold": 20,
        "confirmation_period": 3,
        "stop_loss_pct": 0.025,
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
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(close)

        period = int(params["period"])
        oversold = -float(params["oversold_threshold"])
        overbought = -float(params["overbought_threshold"])
        confirm = int(params["confirmation_period"])
        sl_pct = float(params["stop_loss_pct"])
        tp_pct = float(params["take_profit_pct"])
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        atr_period_val = int(params.get("atr_period", 14))
        atr = self.compute_atr(high, low, close, atr_period_val)

        # Williams %R
        wr = np.full(n, -50.0)
        for i in range(period - 1, n):
            highest = np.max(high[i - period + 1:i + 1])
            lowest = np.min(low[i - period + 1:i + 1])
            if highest != lowest:
                wr[i] = -100.0 * (highest - close[i]) / (highest - lowest)

        signals = np.zeros(n, dtype=np.float64)
        position = 0.0
        entry_price = 0.0

        for i in range(period + confirm, n):
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

            # Confirmation: %R must be in zone for 'confirm' consecutive bars
            if position == 0:
                # Check previous bars were in zone (j=1..confirm), current bar exits
                oversold_confirmed = all(wr[i - j] < oversold for j in range(1, confirm + 1))
                overbought_confirmed = all(wr[i - j] > overbought for j in range(1, confirm + 1))

                if oversold_confirmed and wr[i] >= oversold:
                    # Exiting oversold → long
                    position = 1.0
                    entry_price = close[i]
                elif overbought_confirmed and wr[i] <= overbought:
                    # Exiting overbought → short
                    position = -1.0
                    entry_price = close[i]
            elif position == 1.0 and wr[i] > overbought:
                position = 0.0
                entry_price = 0.0
            elif position == -1.0 and wr[i] < oversold:
                position = 0.0
                entry_price = 0.0

            signals[i] = position

        return self._apply_advanced_exits(signals, data, params)

    def generate_signals_v5(self, data: pd.DataFrame, params: dict):
        """V5 API: returns (signals, sl_distances) for risk-based sizing."""
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(close)
        atr_period_val = int(params.get("atr_period", 14))
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr = self.compute_atr(high, low, close, atr_period_val)
        signals = self.generate_signals(data, params)
        sl_distances = np.full(n, np.nan)
        for i in range(atr_period_val + 1, n):
            if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                sl_distances[i] = atr_sl * atr[i] / close[i]
            elif float(params["stop_loss_pct"]) > 0:
                sl_distances[i] = float(params["stop_loss_pct"])
        return signals, sl_distances
