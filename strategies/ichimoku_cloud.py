"""Ichimoku Cloud Strategy."""

import numpy as np
import pandas as pd
from strategies.base import BaseStrategy


class IchimokuCloud(BaseStrategy):
    name = "Ichimoku Cloud"
    strategy_type = "trend_following"
    default_params = {
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_b_period": 52,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "atr_sl_mult": 0.0,
        "atr_tp_mult": 0.0,
        "atr_period": 14,
        "trailing_atr_mult": 0.0,
        "max_holding_bars": 0,
        "breakeven_trigger_pct": 0.0,
    }

    @staticmethod
    def _donchian_mid(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """Compute Donchian midline (Ichimoku-style)."""
        n = len(high)
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            highest = np.max(high[i - period + 1 : i + 1])
            lowest = np.min(low[i - period + 1 : i + 1])
            result[i] = (highest + lowest) / 2.0
        return result

    def generate_signals(self, data: pd.DataFrame, params: dict) -> np.ndarray:
        params = self.validate_params(params)
        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(close)

        tenkan_p = int(params["tenkan_period"])
        kijun_p = int(params["kijun_period"])
        senkou_b_p = int(params["senkou_b_period"])
        sl = params["stop_loss_pct"]
        tp = params["take_profit_pct"]
        atr_sl = float(params.get("atr_sl_mult", 0.0))
        atr_tp = float(params.get("atr_tp_mult", 0.0))
        atr_period = int(params.get("atr_period", 14))
        atr = self.compute_atr(high, low, close, atr_period)

        tenkan = self._donchian_mid(high, low, tenkan_p)
        kijun = self._donchian_mid(high, low, kijun_p)

        # Senkou Span A = (Tenkan + Kijun) / 2, shifted forward by kijun_p
        senkou_a = np.full(n, np.nan)
        for i in range(kijun_p - 1, n):
            if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
                senkou_a[i] = (tenkan[i] + kijun[i]) / 2.0

        # Senkou Span B = Donchian mid of senkou_b_period, shifted forward by kijun_p
        senkou_b = self._donchian_mid(high, low, senkou_b_p)

        warmup = senkou_b_p + kijun_p
        signals = np.zeros(n)
        position = 0
        entry_price = 0.0

        for i in range(warmup, n):
            if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]):
                signals[i] = position
                continue

            cloud_top = max(senkou_a[i], senkou_b[i])
            cloud_bottom = min(senkou_a[i], senkou_b[i])

            if position == 0:
                # Bullish: Tenkan crosses above Kijun AND price above cloud
                if (tenkan[i] > kijun[i] and tenkan[i - 1] <= kijun[i - 1]
                        and close[i] > cloud_top):
                    position = 1
                    entry_price = close[i]
                # Bearish: Tenkan crosses below Kijun AND price below cloud
                elif (tenkan[i] < kijun[i] and tenkan[i - 1] >= kijun[i - 1]
                      and close[i] < cloud_bottom):
                    position = -1
                    entry_price = close[i]
            elif position == 1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (close[i] - entry_price) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or tenkan[i] < kijun[i]:
                    position = 0
            elif position == -1:
                if atr_sl > 0 and atr[i] > 0 and close[i] > 0:
                    sl_d = atr_sl * atr[i] / close[i]
                    tp_d = atr_tp * atr[i] / close[i] if atr_tp > 0 else tp
                else:
                    sl_d, tp_d = sl, tp
                pnl = (entry_price - close[i]) / entry_price
                if pnl <= -sl_d or pnl >= tp_d or tenkan[i] > kijun[i]:
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
