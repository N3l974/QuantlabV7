"""ADX Regime Strategy.

Uses Average Directional Index (ADX) to detect market regime:
- High ADX (>25): trending market → trade breakouts with +DI/-DI
- Low ADX (<20): ranging market → stay flat (avoid whipsaws)

This filters out low-quality signals that occur in choppy markets.
"""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class ADXRegime(BaseStrategy):
    name = "ADX Regime"
    strategy_type = "regime_filter"
    default_params = {
        "adx_period": 14,
        "adx_trend_threshold": 25,   # ADX above this = trending
        "adx_range_threshold": 20,   # ADX below this = ranging (stay flat)
        "di_smoothing": 14,
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
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(close)

        adx_period = int(params["adx_period"])
        trend_thresh = params["adx_trend_threshold"]
        range_thresh = params["adx_range_threshold"]
        di_smooth = int(params["di_smoothing"])
        sl = params["stop_loss_pct"]
        tp = params["take_profit_pct"]
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        atr_period_val = int(params.get("atr_period", 14))
        atr = self.compute_atr(high, low, close, atr_period_val)

        # Compute True Range
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Compute +DM and -DM
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smooth with Wilder's method (EMA with alpha=1/period)
        def wilder_smooth(data_arr, period):
            result = np.zeros(len(data_arr))
            result[period] = np.sum(data_arr[1:period + 1])
            for i in range(period + 1, len(data_arr)):
                result[i] = result[i - 1] - result[i - 1] / period + data_arr[i]
            return result

        atr_smooth = wilder_smooth(tr, adx_period)
        plus_dm_smooth = wilder_smooth(plus_dm, adx_period)
        minus_dm_smooth = wilder_smooth(minus_dm, adx_period)

        # Compute +DI and -DI
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        for i in range(adx_period, n):
            if atr_smooth[i] > 0:
                plus_di[i] = 100.0 * plus_dm_smooth[i] / atr_smooth[i]
                minus_di[i] = 100.0 * minus_dm_smooth[i] / atr_smooth[i]

        # Compute DX and ADX
        dx = np.zeros(n)
        for i in range(adx_period, n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

        adx = np.zeros(n)
        start = adx_period * 2
        if start < n:
            adx[start] = np.mean(dx[adx_period:start + 1]) if start > adx_period else 0
            for i in range(start + 1, n):
                adx[i] = (adx[i - 1] * (adx_period - 1) + dx[i]) / adx_period

        warmup = start + 1
        signals = np.zeros(n)
        position = 0
        entry_price = 0.0

        for i in range(warmup, n):
            if position == 0:
                # Only trade in trending markets
                if adx[i] > trend_thresh:
                    # +DI crosses above -DI → bullish trend
                    if plus_di[i] > minus_di[i] and plus_di[i - 1] <= minus_di[i - 1]:
                        position = 1
                        entry_price = close[i]
                    # -DI crosses above +DI → bearish trend
                    elif minus_di[i] > plus_di[i] and minus_di[i - 1] <= plus_di[i - 1]:
                        position = -1
                        entry_price = close[i]
            elif position == 1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (close[i] - entry_price) / entry_price
                # Exit on SL, TP, trend weakening, or DI reversal
                if pnl <= -sl_d or pnl >= tp_d or adx[i] < range_thresh or minus_di[i] > plus_di[i]:
                    position = 0
            elif position == -1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (entry_price - close[i]) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or adx[i] < range_thresh or plus_di[i] > minus_di[i]:
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
        atr_period_val = int(params.get("atr_period", 14))
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr = self.compute_atr(high, low, close, atr_period_val)
        signals = self.generate_signals(data, params)
        sl_distances = np.full(n, np.nan)
        for i in range(atr_period_val + 1, n):
            if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                sl_distances[i] = atr_sl * atr[i] / close[i]
            elif float(params["stop_loss_pct"]) > 0:
                sl_distances[i] = float(params["stop_loss_pct"])
        return signals, sl_distances
