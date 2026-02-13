"""
SuperTrend + ADX Regime Filter Strategy.

Combines the SuperTrend (best holdout survivor) with an ADX regime filter:
- Only trades when ADX > threshold (trending market)
- SuperTrend provides direction and trailing stop
- ADX filters out whipsaw periods in ranging markets
- Reduces trade count and improves win rate

This is a multi-factor strategy: trend (SuperTrend) + regime (ADX).
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class SuperTrendADX(BaseStrategy):
    name = "SuperTrend + ADX"
    strategy_type = "multi_factor"
    default_params = {
        # SuperTrend params
        "atr_period": 10,
        "st_multiplier": 3.0,
        # ADX filter params
        "adx_period": 14,
        "adx_threshold": 20,       # Only trade when ADX > this
        # Risk management
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.08,
        "atr_sl_mult": 0.0,
        "atr_tp_mult": 0.0,
        "trailing_atr_mult": 0.0,
        "max_holding_bars": 0,
        "breakeven_trigger_pct": 0.0,
        # Trade filter
        "cooldown_bars": 3,         # Min bars between trades
    }

    def generate_signals(self, data: pd.DataFrame, params: dict) -> np.ndarray:
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(close)

        atr_period = int(params["atr_period"])
        st_mult = float(params["st_multiplier"])
        adx_period = int(params["adx_period"])
        adx_thresh = float(params["adx_threshold"])
        sl_pct = float(params["stop_loss_pct"])
        tp_pct = float(params["take_profit_pct"])
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        cooldown = int(params["cooldown_bars"])

        # ── ATR calculation (Wilder smoothing) ──
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))

        atr = np.zeros(n)
        if atr_period < n:
            atr[atr_period] = np.mean(tr[1:atr_period + 1])
            for i in range(atr_period + 1, n):
                atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

        # ── SuperTrend calculation ──
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

        # ── ADX calculation ──
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        def wilder_smooth(arr, period):
            result = np.zeros(len(arr))
            if period < len(arr):
                result[period] = np.sum(arr[1:period + 1])
                for i in range(period + 1, len(arr)):
                    result[i] = result[i - 1] - result[i - 1] / period + arr[i]
            return result

        atr_smooth = wilder_smooth(tr, adx_period)
        plus_dm_smooth = wilder_smooth(plus_dm, adx_period)
        minus_dm_smooth = wilder_smooth(minus_dm, adx_period)

        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        for i in range(adx_period, n):
            if atr_smooth[i] > 0:
                plus_di[i] = 100.0 * plus_dm_smooth[i] / atr_smooth[i]
                minus_di[i] = 100.0 * minus_dm_smooth[i] / atr_smooth[i]

        dx = np.zeros(n)
        for i in range(adx_period, n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

        adx = np.zeros(n)
        adx_start = adx_period * 2
        if adx_start < n:
            adx[adx_start] = np.mean(dx[adx_period:adx_start + 1]) if adx_start > adx_period else 0
            for i in range(adx_start + 1, n):
                adx[i] = (adx[i - 1] * (adx_period - 1) + dx[i]) / adx_period

        # ── Signal generation ──
        warmup = max(st_warmup, adx_start + 1)
        signals = np.zeros(n, dtype=np.float64)
        position = 0.0
        entry_price = 0.0
        bars_since_exit = cooldown  # Allow immediate first trade

        for i in range(warmup, n):
            # SL/TP check
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
                    bars_since_exit = 0
                    signals[i] = 0.0
                    continue

            # ADX regime filter + cooldown
            trending = adx[i] > adx_thresh
            can_trade = bars_since_exit >= cooldown

            if position == 0:
                bars_since_exit += 1
                if trending and can_trade:
                    new_pos = float(supertrend[i])
                    if new_pos != 0:
                        position = new_pos
                        entry_price = close[i]
            else:
                # Exit if regime changes to ranging OR supertrend flips
                st_pos = float(supertrend[i])
                if st_pos != position or not trending:
                    position = 0.0
                    entry_price = 0.0
                    bars_since_exit = 0

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
