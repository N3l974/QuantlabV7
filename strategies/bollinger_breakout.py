"""Bollinger Band Breakout Strategy."""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class BollingerBreakout(BaseStrategy):
    name = "Bollinger Band Breakout"
    strategy_type = "breakout"
    default_params = {
        "bb_period": 20,
        "bb_std": 2.0,
        "confirmation_candles": 1,
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

        period = int(params["bb_period"])
        std_mult = params["bb_std"]
        confirm = int(params["confirmation_candles"])
        sl = params["stop_loss_pct"]
        tp = params["take_profit_pct"]
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        atr_period = int(params.get("atr_period", 14))
        high = data["high"].values.astype(np.float64) if "high" in data.columns else close
        low = data["low"].values.astype(np.float64) if "low" in data.columns else close
        atr = self.compute_atr(high, low, close, atr_period)

        # Compute Bollinger Bands
        sma = np.full(n, np.nan)
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)

        for i in range(period - 1, n):
            window = close[i - period + 1 : i + 1]
            sma[i] = np.mean(window)
            std = np.std(window, ddof=1)
            upper[i] = sma[i] + std_mult * std
            lower[i] = sma[i] - std_mult * std

        signals = np.zeros(n)
        position = 0
        entry_price = 0.0
        confirm_count = 0
        pending_signal = 0

        for i in range(period, n):
            if np.isnan(upper[i]):
                continue

            if position == 0:
                # Check for breakout
                if close[i] > upper[i]:
                    if pending_signal == 1:
                        confirm_count += 1
                    else:
                        pending_signal = 1
                        confirm_count = 1
                elif close[i] < lower[i]:
                    if pending_signal == -1:
                        confirm_count += 1
                    else:
                        pending_signal = -1
                        confirm_count = 1
                else:
                    pending_signal = 0
                    confirm_count = 0

                if confirm_count >= confirm:
                    position = pending_signal
                    entry_price = close[i]
                    pending_signal = 0
                    confirm_count = 0
            elif position == 1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (close[i] - entry_price) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or close[i] < sma[i]:
                    position = 0
            elif position == -1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (entry_price - close[i]) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or close[i] > sma[i]:
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
