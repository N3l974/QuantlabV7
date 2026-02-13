"""
Multi-Timeframe Trend + Entry Strategy.

Uses a higher-timeframe (HTF) trend filter combined with a lower-timeframe (LTF)
entry signal for better timing:

1. HTF Trend (simulated via longer lookback):
   - SuperTrend with large ATR period → overall direction
   - Only trade in the direction of the HTF trend

2. LTF Entry:
   - RSI oversold/overbought for mean-reversion entry in trend direction
   - Enter long when HTF=bull AND LTF RSI is oversold (pullback buy)
   - Enter short when HTF=bear AND LTF RSI is overbought (rally sell)

3. Exit:
   - HTF trend flip OR SL/TP hit

This creates a structural edge: buying pullbacks in a strong trend
instead of chasing breakouts.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class MTFTrendEntry(BaseStrategy):
    name = "MTF Trend Entry"
    strategy_type = "multi_timeframe"
    default_params = {
        # HTF trend (simulated via longer periods)
        "htf_atr_period": 40,          # ~10 bars on 4h = 40 bars on 1h
        "htf_st_multiplier": 3.0,
        # LTF entry
        "rsi_period": 14,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        # Risk management
        "stop_loss_pct": 0.035,
        "take_profit_pct": 0.07,
        "atr_sl_mult": 0.0,
        "atr_tp_mult": 0.0,
        "trailing_atr_mult": 0.0,
        "max_holding_bars": 0,
        "breakeven_trigger_pct": 0.0,
        # Cooldown
        "cooldown_bars": 3,
    }

    def generate_signals(self, data: pd.DataFrame, params: dict) -> np.ndarray:
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(close)

        htf_atr_period = int(params["htf_atr_period"])
        htf_st_mult = float(params["htf_st_multiplier"])
        rsi_period = int(params["rsi_period"])
        rsi_os = float(params["rsi_oversold"])
        rsi_ob = float(params["rsi_overbought"])
        sl_pct = float(params["stop_loss_pct"])
        tp_pct = float(params["take_profit_pct"])
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        cooldown = int(params["cooldown_bars"])

        # ── HTF SuperTrend (longer period = proxy for higher timeframe) ──
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i - 1]),
                        abs(low[i] - close[i - 1]))

        atr = np.zeros(n)
        if htf_atr_period < n:
            atr[htf_atr_period] = np.mean(tr[1:htf_atr_period + 1])
            for i in range(htf_atr_period + 1, n):
                atr[i] = (atr[i - 1] * (htf_atr_period - 1) + tr[i]) / htf_atr_period

        hl2 = (high + low) / 2.0
        upper_band = np.zeros(n)
        lower_band = np.zeros(n)
        supertrend = np.zeros(n)

        st_warmup = htf_atr_period + 1
        for i in range(st_warmup, n):
            basic_upper = hl2[i] + htf_st_mult * atr[i]
            basic_lower = hl2[i] - htf_st_mult * atr[i]

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

        # ── LTF RSI ──
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        if rsi_period < n:
            avg_gain[rsi_period] = np.mean(gain[1:rsi_period + 1])
            avg_loss[rsi_period] = np.mean(loss[1:rsi_period + 1])
            for i in range(rsi_period + 1, n):
                avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gain[i]) / rsi_period
                avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + loss[i]) / rsi_period

        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
        rsi = 100.0 - 100.0 / (1.0 + rs)

        # ── Signal generation ──
        warmup = max(st_warmup + 10, rsi_period + 1)
        signals = np.zeros(n, dtype=np.float64)
        position = 0.0
        entry_price = 0.0
        bars_since_exit = cooldown

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

            htf_dir = float(supertrend[i])
            can_trade = bars_since_exit >= cooldown

            if position == 0:
                bars_since_exit += 1
                if can_trade:
                    # Long: HTF bullish + LTF RSI pullback (oversold)
                    if htf_dir == 1 and rsi[i] < rsi_os:
                        position = 1.0
                        entry_price = close[i]
                    # Short: HTF bearish + LTF RSI overbought (rally sell)
                    elif htf_dir == -1 and rsi[i] > rsi_ob:
                        position = -1.0
                        entry_price = close[i]
            else:
                # Exit on HTF trend flip
                if (position == 1 and htf_dir == -1) or \
                   (position == -1 and htf_dir == 1):
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
        atr_period = int(params["htf_atr_period"])
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
