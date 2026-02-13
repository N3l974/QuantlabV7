"""RSI Mean-Reversion Strategy."""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class RSIMeanReversion(BaseStrategy):
    name = "RSI Mean-Reversion"
    strategy_type = "mean_reversion"
    default_params = {
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
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

        rsi_period = int(params["rsi_period"])
        oversold = params["oversold"]
        overbought = params["overbought"]
        sl = params["stop_loss_pct"]
        tp = params["take_profit_pct"]
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        atr_period_val = int(params.get("atr_period", 14))
        high = data["high"].values.astype(np.float64) if "high" in data.columns else close
        low_arr = data["low"].values.astype(np.float64) if "low" in data.columns else close
        atr = self.compute_atr(high, low_arr, close, atr_period_val)

        # Compute RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period + 1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period + 1])

        for i in range(rsi_period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + loss[i]) / rsi_period

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        # Generate signals with SL/TP
        signals = np.zeros(n)
        position = 0
        entry_price = 0.0

        for i in range(rsi_period + 1, n):
            if position == 0:
                if rsi[i] < oversold:
                    position = 1
                    entry_price = close[i]
                elif rsi[i] > overbought:
                    position = -1
                    entry_price = close[i]
            elif position == 1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (close[i] - entry_price) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or rsi[i] > overbought:
                    position = 0
            elif position == -1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (entry_price - close[i]) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or rsi[i] < oversold:
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
        atr_period_val = int(params.get("atr_period", 14))
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr = self.compute_atr(high, low_arr, close, atr_period_val)
        signals = self.generate_signals(data, params)
        sl_distances = np.full(n, np.nan)
        for i in range(atr_period_val + 1, n):
            if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                sl_distances[i] = atr_sl * atr[i] / close[i]
            elif float(params["stop_loss_pct"]) > 0:
                sl_distances[i] = float(params["stop_loss_pct"])
        return signals, sl_distances
