"""
Multi-Timeframe Momentum Breakout Strategy.

Combines higher-timeframe momentum confirmation with lower-timeframe breakout entry:

1. HTF Momentum Filter (longer lookback = proxy for HTF):
   - ROC (Rate of Change) over long period → confirms directional momentum
   - ADX filter → only trade in trending conditions

2. LTF Breakout Entry:
   - Donchian channel breakout on shorter lookback
   - Volume confirmation (above average)

3. Exit:
   - Opposite Donchian breakout OR ADX drops below threshold OR SL/TP

Edge: Breakouts confirmed by HTF momentum have higher follow-through rate.
Filtering by regime avoids false breakouts in choppy markets.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class MTFMomentumBreakout(BaseStrategy):
    name = "MTF Momentum Breakout"
    strategy_type = "multi_timeframe"
    default_params = {
        # HTF momentum (long lookback)
        "roc_period": 50,              # ~12 days on 4h
        "roc_threshold": 2.0,         # Min % ROC for momentum confirmation
        "adx_period": 14,
        "adx_threshold": 22,
        # LTF breakout
        "donchian_period": 15,         # Donchian channel for breakout
        # Volume filter
        "vol_ma_period": 20,
        "vol_spike_ratio": 1.1,
        # Risk management
        "stop_loss_pct": 0.04,
        "take_profit_pct": 0.10,
        "atr_sl_mult": 0.0,
        "atr_tp_mult": 0.0,
        "trailing_atr_mult": 0.0,
        "max_holding_bars": 0,
        "breakeven_trigger_pct": 0.0,
        "cooldown_bars": 5,
    }

    def generate_signals(self, data: pd.DataFrame, params: dict) -> np.ndarray:
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        volume = data["volume"].values.astype(np.float64)
        n = len(close)

        roc_period = int(params["roc_period"])
        roc_thresh = float(params["roc_threshold"])
        adx_period = int(params["adx_period"])
        adx_thresh = float(params["adx_threshold"])
        don_period = int(params["donchian_period"])
        vol_ma_period = int(params["vol_ma_period"])
        vol_spike = float(params["vol_spike_ratio"])
        sl_pct = float(params["stop_loss_pct"])
        tp_pct = float(params["take_profit_pct"])
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        cooldown = int(params["cooldown_bars"])

        # ── HTF: ROC momentum ──
        roc = np.zeros(n)
        for i in range(roc_period, n):
            if close[i - roc_period] != 0:
                roc[i] = (close[i] - close[i - roc_period]) / close[i - roc_period] * 100

        # ── HTF: ADX ──
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

        # ── LTF: Donchian channel ──
        don_high = np.zeros(n)
        don_low = np.zeros(n)
        for i in range(don_period, n):
            don_high[i] = np.max(high[i - don_period:i])
            don_low[i] = np.min(low[i - don_period:i])

        # ── Volume moving average ──
        vol_ma = np.zeros(n)
        for i in range(vol_ma_period, n):
            vol_ma[i] = np.mean(volume[i - vol_ma_period:i])

        # ── Signal generation ──
        warmup = max(roc_period, adx_start + 1, don_period + 1, vol_ma_period + 1)
        signals = np.zeros(n, dtype=np.float64)
        position = 0.0
        entry_price = 0.0
        bars_since_exit = cooldown

        for i in range(warmup, n):
            # SL/TP check
            if position != 0 and entry_price > 0:
                if atr_sl > 0 and atr_smooth[i] > 0 and close[i] > 0:
                    atr_val = atr_smooth[i] / adx_period  # approx ATR
                    sl_d = atr_sl * atr_val / close[i]
                    tp_d = atr_tp * atr_val / close[i] if atr_tp > 0 else tp_pct
                else:
                    sl_d, tp_d = sl_pct, tp_pct
                pnl = position * (close[i] - entry_price) / entry_price
                if pnl <= -sl_d or pnl >= tp_d:
                    position = 0.0
                    entry_price = 0.0
                    bars_since_exit = 0
                    signals[i] = 0.0
                    continue

            trending = adx[i] > adx_thresh
            mom_bullish = roc[i] > roc_thresh
            mom_bearish = roc[i] < -roc_thresh
            vol_ok = vol_ma[i] > 0 and volume[i] > vol_spike * vol_ma[i]
            can_trade = bars_since_exit >= cooldown

            if position == 0:
                bars_since_exit += 1
                if trending and can_trade and vol_ok:
                    # Long: HTF momentum bullish + Donchian breakout up
                    if mom_bullish and close[i] > don_high[i]:
                        position = 1.0
                        entry_price = close[i]
                    # Short: HTF momentum bearish + Donchian breakout down
                    elif mom_bearish and close[i] < don_low[i]:
                        position = -1.0
                        entry_price = close[i]
            elif position == 1:
                # Exit: opposite breakout OR regime change
                if close[i] < don_low[i] or not trending:
                    position = 0.0
                    entry_price = 0.0
                    bars_since_exit = 0
            elif position == -1:
                if close[i] > don_high[i] or not trending:
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
        atr_period = int(params.get("adx_period", 14))
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
