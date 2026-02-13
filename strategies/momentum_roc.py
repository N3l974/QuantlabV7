"""Rate of Change (ROC) Momentum Strategy.

Measures price momentum over multiple lookback periods.
Enters long when momentum is strongly positive, short when strongly negative.
Uses dual-timeframe confirmation: fast ROC for timing, slow ROC for trend.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class MomentumROC(BaseStrategy):
    name = "Momentum ROC"
    strategy_type = "momentum"
    default_params = {
        "fast_roc_period": 10,
        "slow_roc_period": 40,
        "entry_threshold": 2.0,    # % ROC threshold to enter
        "exit_threshold": 0.1,     # % ROC threshold to exit (must be > 0)
        "stop_loss_pct": 0.02,
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

        fast_p = int(params["fast_roc_period"])
        slow_p = int(params["slow_roc_period"])
        entry_thresh = params["entry_threshold"]
        exit_thresh = params["exit_threshold"]
        sl = params["stop_loss_pct"]
        tp = params["take_profit_pct"]
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        atr_period = int(params.get("atr_period", 14))
        high = data["high"].values.astype(np.float64) if "high" in data.columns else close
        low_arr = data["low"].values.astype(np.float64) if "low" in data.columns else close
        atr = self.compute_atr(high, low_arr, close, atr_period)

        # Compute ROC: (close - close[n-period]) / close[n-period] * 100
        fast_roc = np.full(n, 0.0)
        slow_roc = np.full(n, 0.0)

        for i in range(fast_p, n):
            if close[i - fast_p] != 0:
                fast_roc[i] = (close[i] - close[i - fast_p]) / close[i - fast_p] * 100

        for i in range(slow_p, n):
            if close[i - slow_p] != 0:
                slow_roc[i] = (close[i] - close[i - slow_p]) / close[i - slow_p] * 100

        warmup = slow_p + 1
        signals = np.zeros(n)
        position = 0
        entry_price = 0.0

        for i in range(warmup, n):
            if position == 0:
                # Long: fast momentum positive AND slow trend confirms
                if fast_roc[i] > entry_thresh and slow_roc[i] > 0:
                    position = 1
                    entry_price = close[i]
                # Short: fast momentum negative AND slow trend confirms
                elif fast_roc[i] < -entry_thresh and slow_roc[i] < 0:
                    position = -1
                    entry_price = close[i]
            elif position == 1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (close[i] - entry_price) / entry_price
                # Exit on SL, TP, or momentum reversal
                if pnl <= -sl_d or pnl >= tp_d or fast_roc[i] < exit_thresh:
                    position = 0
            elif position == -1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (entry_price - close[i]) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or fast_roc[i] > -exit_thresh:
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
