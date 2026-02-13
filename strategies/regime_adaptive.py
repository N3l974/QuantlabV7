"""
Regime-Adaptive Strategy.

Fundamentally different from other strategies: it adapts its behavior
based on detected market regime.

In trending markets → trend-following (SuperTrend)
In ranging markets → mean-reversion (Bollinger band bounce)
In crisis → cash (no trading)

This is the first strategy with built-in regime awareness, providing
a structural edge over static strategies that use one approach regardless
of market conditions.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class RegimeAdaptive(BaseStrategy):
    name = "Regime Adaptive"
    strategy_type = "regime_adaptive"
    default_params = {
        # Regime detection
        "adx_period": 14,
        "adx_trend_threshold": 25,     # ADX > this = trending
        "vol_period": 20,              # Rolling vol lookback
        "vol_crisis_mult": 2.5,        # Vol > median * this = crisis

        # Trend-following params (used in trending regime)
        "st_atr_period": 10,
        "st_multiplier": 3.0,

        # Mean-reversion params (used in ranging regime)
        "bb_period": 20,
        "bb_std": 2.0,

        # Risk management
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
        "atr_sl_mult": 0.0,
        "atr_tp_mult": 0.0,
        "trailing_atr_mult": 0.0,
        "max_holding_bars": 0,
        "breakeven_trigger_pct": 0.0,
        "cooldown_bars": 3,
    }

    def generate_signals(self, data: pd.DataFrame, params: dict) -> np.ndarray:
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(close)

        adx_period = int(params["adx_period"])
        adx_thresh = float(params["adx_trend_threshold"])
        vol_period = int(params["vol_period"])
        vol_crisis = float(params["vol_crisis_mult"])
        st_atr_period = int(params["st_atr_period"])
        st_mult = float(params["st_multiplier"])
        bb_period = int(params["bb_period"])
        bb_std = float(params["bb_std"])
        sl_pct = float(params["stop_loss_pct"])
        tp_pct = float(params["take_profit_pct"])
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        cooldown = int(params["cooldown_bars"])

        # ── ADX calculation ──
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))

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

        atr_smooth_adx = wilder_smooth(tr, adx_period)
        plus_dm_smooth = wilder_smooth(plus_dm, adx_period)
        minus_dm_smooth = wilder_smooth(minus_dm, adx_period)

        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        for i in range(adx_period, n):
            if atr_smooth_adx[i] > 0:
                plus_di[i] = 100.0 * plus_dm_smooth[i] / atr_smooth_adx[i]
                minus_di[i] = 100.0 * minus_dm_smooth[i] / atr_smooth_adx[i]

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

        # ── Volatility for crisis detection ──
        returns = np.zeros(n)
        returns[1:] = np.diff(close) / np.where(close[:-1] != 0, close[:-1], 1.0)

        rolling_vol = np.zeros(n)
        for i in range(vol_period, n):
            rolling_vol[i] = np.std(returns[i - vol_period:i])

        # Rolling median of vol (expanding)
        vol_median = np.zeros(n)
        for i in range(vol_period * 5, n):
            valid_vol = rolling_vol[vol_period:i + 1]
            valid_vol = valid_vol[valid_vol > 0]
            if len(valid_vol) > 0:
                vol_median[i] = np.median(valid_vol)

        # ── SuperTrend (for trending regime) ──
        atr = np.zeros(n)
        if st_atr_period < n:
            atr[st_atr_period] = np.mean(tr[1:st_atr_period + 1])
            for i in range(st_atr_period + 1, n):
                atr[i] = (atr[i - 1] * (st_atr_period - 1) + tr[i]) / st_atr_period

        hl2 = (high + low) / 2.0
        upper_band = np.zeros(n)
        lower_band = np.zeros(n)
        supertrend = np.zeros(n)

        st_warmup = st_atr_period + 1
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
                supertrend[i] = -1 if close[i] < lower_band[i] else 1
            else:
                supertrend[i] = 1 if close[i] > upper_band[i] else -1

        # ── Bollinger Bands (for ranging regime) ──
        bb_mid = np.zeros(n)
        bb_upper = np.zeros(n)
        bb_lower = np.zeros(n)
        for i in range(bb_period - 1, n):
            window = close[i - bb_period + 1:i + 1]
            bb_mid[i] = np.mean(window)
            std = np.std(window)
            bb_upper[i] = bb_mid[i] + bb_std * std
            bb_lower[i] = bb_mid[i] - bb_std * std

        # ── Signal generation ──
        warmup = max(adx_start + 1, st_warmup + 10, bb_period + 5, vol_period * 5 + 1)
        signals = np.zeros(n, dtype=np.float64)
        position = 0.0
        entry_price = 0.0
        bars_since_exit = cooldown
        entry_regime = 0  # Track which regime we entered in

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
                    entry_regime = 0
                    signals[i] = 0.0
                    continue

            # Classify regime
            is_crisis = (vol_median[i] > 0 and
                         rolling_vol[i] > vol_crisis * vol_median[i])
            is_trending = adx[i] > adx_thresh and not is_crisis

            can_trade = bars_since_exit >= cooldown

            if position == 0:
                bars_since_exit += 1

                if is_crisis:
                    # No trading in crisis
                    signals[i] = 0.0
                    continue

                if is_trending and can_trade:
                    # TREND MODE: follow SuperTrend
                    st_dir = float(supertrend[i])
                    if st_dir != 0:
                        position = st_dir
                        entry_price = close[i]
                        entry_regime = 1  # trending

                elif not is_trending and not is_crisis and can_trade:
                    # RANGE MODE: mean-reversion on Bollinger bands
                    if close[i] < bb_lower[i] and bb_lower[i] > 0:
                        position = 1.0    # Buy at lower band
                        entry_price = close[i]
                        entry_regime = 2  # ranging
                    elif close[i] > bb_upper[i] and bb_upper[i] > 0:
                        position = -1.0   # Sell at upper band
                        entry_price = close[i]
                        entry_regime = 2  # ranging

            else:
                # Exit logic depends on entry regime
                if is_crisis:
                    # Always exit in crisis
                    position = 0.0
                    entry_price = 0.0
                    bars_since_exit = 0
                    entry_regime = 0
                elif entry_regime == 1:
                    # Trend exit: SuperTrend flip
                    st_dir = float(supertrend[i])
                    if st_dir != position:
                        position = 0.0
                        entry_price = 0.0
                        bars_since_exit = 0
                        entry_regime = 0
                elif entry_regime == 2:
                    # Range exit: return to mid band
                    if position == 1 and close[i] > bb_mid[i]:
                        position = 0.0
                        entry_price = 0.0
                        bars_since_exit = 0
                        entry_regime = 0
                    elif position == -1 and close[i] < bb_mid[i]:
                        position = 0.0
                        entry_price = 0.0
                        bars_since_exit = 0
                        entry_regime = 0

            signals[i] = position

        return self._apply_advanced_exits(signals, data, params)

    def generate_signals_v5(self, data: pd.DataFrame, params: dict):
        """V5 API: returns (signals, sl_distances) for risk-based sizing."""
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(close)
        atr_period = int(params["st_atr_period"])
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
