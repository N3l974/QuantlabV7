"""
Trend Multi-Factor Strategy — SuperTrend + Volume (OBV) + Momentum (ROC).

Combines three independent signal sources for high-conviction entries:
1. SuperTrend: primary trend direction and trailing stop
2. OBV slope: volume confirmation (smart money flow)
3. ROC momentum: price momentum confirmation

Entry requires ALL THREE factors to agree (confluence).
Exit on SuperTrend flip OR SL/TP hit.

This drastically reduces trade count but improves quality.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class TrendMultiFactor(BaseStrategy):
    name = "Trend Multi-Factor"
    strategy_type = "multi_factor"
    default_params = {
        # SuperTrend
        "atr_period": 10,
        "st_multiplier": 3.0,
        # OBV slope
        "obv_slope_period": 10,     # Lookback for OBV slope direction
        # ROC momentum
        "roc_period": 14,
        "roc_threshold": 1.0,       # Min % ROC to confirm momentum
        # Risk management
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.08,
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
        volume = data["volume"].values.astype(np.float64)
        n = len(close)

        atr_period = int(params["atr_period"])
        st_mult = float(params["st_multiplier"])
        obv_slope_period = int(params["obv_slope_period"])
        roc_period = int(params["roc_period"])
        roc_thresh = float(params["roc_threshold"])
        sl_pct = float(params["stop_loss_pct"])
        tp_pct = float(params["take_profit_pct"])
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))

        # ── Factor 1: SuperTrend ──
        atr = self.compute_atr(high, low, close, atr_period)

        hl2 = (high + low) / 2.0
        upper_band = np.zeros(n)
        lower_band = np.zeros(n)
        supertrend = np.zeros(n)

        st_warmup = atr_period + 1
        for i in range(st_warmup, n):
            basic_upper = hl2[i] + st_mult * atr[i]
            basic_lower = hl2[i] - st_mult * atr[i]

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

        # ── Factor 2: OBV slope ──
        obv = np.zeros(n)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        # OBV slope: linear regression slope over lookback
        obv_slope = np.zeros(n)
        for i in range(obv_slope_period, n):
            window = obv[i - obv_slope_period + 1:i + 1]
            x = np.arange(obv_slope_period, dtype=np.float64)
            x_mean = x.mean()
            y_mean = window.mean()
            denom = np.sum((x - x_mean) ** 2)
            if denom > 0:
                obv_slope[i] = np.sum((x - x_mean) * (window - y_mean)) / denom

        # ── Factor 3: ROC momentum ──
        roc = np.zeros(n)
        for i in range(roc_period, n):
            if close[i - roc_period] != 0:
                roc[i] = (close[i] - close[i - roc_period]) / close[i - roc_period] * 100

        # ── Signal generation: require confluence ──
        warmup = max(st_warmup, obv_slope_period, roc_period) + 1
        signals = np.zeros(n, dtype=np.float64)
        position = 0.0
        entry_price = 0.0

        for i in range(warmup, n):
            # SL/TP check
            if position != 0 and entry_price > 0:
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

            st_dir = float(supertrend[i])
            vol_bullish = obv_slope[i] > 0
            vol_bearish = obv_slope[i] < 0
            mom_bullish = roc[i] > roc_thresh
            mom_bearish = roc[i] < -roc_thresh

            if position == 0:
                # Long: all 3 factors agree bullish
                if st_dir == 1 and vol_bullish and mom_bullish:
                    position = 1.0
                    entry_price = close[i]
                # Short: all 3 factors agree bearish
                elif st_dir == -1 and vol_bearish and mom_bearish:
                    position = -1.0
                    entry_price = close[i]
            elif position == 1:
                # Exit on SuperTrend flip (primary exit)
                if st_dir == -1:
                    position = 0.0
                    entry_price = 0.0
            elif position == -1:
                if st_dir == 1:
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
