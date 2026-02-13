"""Donchian Channel Breakout Strategy."""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class DonchianChannel(BaseStrategy):
    name = "Donchian Channel"
    strategy_type = "breakout"
    default_params = {
        "channel_period": 20,
        "exit_period": 10,
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

        ch_period = int(params["channel_period"])
        ex_period = int(params["exit_period"])
        sl = params["stop_loss_pct"]
        tp = params["take_profit_pct"]
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        atr_period = int(params.get("atr_period", 14))
        atr = self.compute_atr(high, low, close, atr_period)

        # Compute channels
        upper_ch = np.full(n, np.nan)
        lower_ch = np.full(n, np.nan)
        exit_upper = np.full(n, np.nan)
        exit_lower = np.full(n, np.nan)

        for i in range(ch_period - 1, n):
            upper_ch[i] = np.max(high[i - ch_period + 1 : i + 1])
            lower_ch[i] = np.min(low[i - ch_period + 1 : i + 1])

        for i in range(ex_period - 1, n):
            exit_upper[i] = np.max(high[i - ex_period + 1 : i + 1])
            exit_lower[i] = np.min(low[i - ex_period + 1 : i + 1])

        warmup = max(ch_period, ex_period) + 1
        signals = np.zeros(n)
        position = 0
        entry_price = 0.0

        for i in range(warmup, n):
            if np.isnan(upper_ch[i]) or np.isnan(exit_lower[i]):
                signals[i] = position
                continue

            if position == 0:
                if close[i] > upper_ch[i]:
                    position = 1
                    entry_price = close[i]
                elif close[i] < lower_ch[i]:
                    position = -1
                    entry_price = close[i]
            elif position == 1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (close[i] - entry_price) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or close[i] < exit_lower[i]:
                    position = 0
            elif position == -1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (entry_price - close[i]) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or close[i] > exit_upper[i]:
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
