"""
Keltner Channel Strategy — Volatility-based channel breakout.

Uses EMA ± ATR multiplier as dynamic channels.
Long when price closes above upper channel, short below lower.
Complementary to Bollinger (ATR-based vs std-based).
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class KeltnerChannel(BaseStrategy):
    name = "Keltner Channel"
    strategy_type = "volatility_channel"
    default_params = {
        "ema_period": 20,
        "atr_period": 14,
        "atr_multiplier": 2.0,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
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

        ema_period = int(params["ema_period"])
        atr_period = int(params["atr_period"])
        atr_mult = float(params["atr_multiplier"])
        sl_pct = float(params["stop_loss_pct"])
        tp_pct = float(params["take_profit_pct"])
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))

        # EMA of close
        ema = np.full(n, np.nan)
        ema[0] = close[0]
        alpha = 2.0 / (ema_period + 1)
        for i in range(1, n):
            ema[i] = alpha * close[i] + (1 - alpha) * ema[i - 1]

        # ATR
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))
        atr = np.full(n, np.nan)
        atr[atr_period] = np.mean(tr[1:atr_period + 1])
        for i in range(atr_period + 1, n):
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

        upper = ema + atr_mult * atr
        lower = ema - atr_mult * atr

        signals = np.zeros(n, dtype=np.float64)
        position = 0.0
        entry_price = 0.0

        warmup = max(ema_period, atr_period) + 1
        for i in range(warmup, n):
            if np.isnan(upper[i]) or np.isnan(lower[i]):
                signals[i] = position
                continue

            # SL/TP
            if position != 0 and entry_price > 0:
                if atr_sl > 0 and not np.isnan(atr[i]) and atr[i] > 0 and close[i] > 0:
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
                if close[i] > upper[i]:
                    position = 1.0
                    entry_price = close[i]
                elif close[i] < lower[i]:
                    position = -1.0
                    entry_price = close[i]
            elif position == 1.0 and close[i] < ema[i]:
                position = 0.0
                entry_price = 0.0
            elif position == -1.0 and close[i] > ema[i]:
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
