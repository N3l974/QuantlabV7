"""
Abstract base class for all trading strategies.
Every strategy must implement generate_signals().
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base strategy.

    Subclasses must implement:
        - name: str
        - strategy_type: str
        - default_params: dict
        - generate_signals(data, params) -> np.ndarray

    V5 strategies can also implement:
        - generate_signals_v5(data, params) -> (signals, sl_distances)
          where sl_distances is the per-bar SL distance as fraction of price.
    """

    name: str = "BaseStrategy"
    strategy_type: str = "unknown"
    default_params: dict = {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, params: dict) -> np.ndarray:
        """
        Generate trading signals from OHLCV data and parameters.

        Args:
            data: DataFrame with columns [open, high, low, close, volume]
            params: Dictionary of strategy parameters

        Returns:
            numpy array of signals: +1 (long), -1 (short), 0 (flat)
        """
        raise NotImplementedError

    @staticmethod
    def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Compute Average True Range."""
        n = len(close)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros(n)
        if period < n:
            atr[period] = np.mean(tr[1:period + 1])
            for i in range(period + 1, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr

    def get_sl_tp_mode(self, params: dict):
        """Determine SL/TP mode: 'atr' if atr_sl_mult present, else 'pct'."""
        if "atr_sl_mult" in params and params["atr_sl_mult"] > 0:
            return "atr"
        return "pct"

    def _apply_advanced_exits(self, signals: np.ndarray, data: pd.DataFrame, params: dict) -> np.ndarray:
        """
        Post-process signals to apply advanced exit rules.

        - trailing_atr_mult: trailing stop at N × ATR from peak (0 = disabled)
        - breakeven_trigger_pct: move SL to entry after this % gain (0 = disabled)
        - max_holding_bars: force exit after N bars (0 = disabled)

        Backward compatible: all defaults = 0 → returns signals unchanged.
        """
        trailing_mult = float(params.get("trailing_atr_mult", 0.0))
        breakeven_pct = float(params.get("breakeven_trigger_pct", 0.0))
        max_hold = int(params.get("max_holding_bars", 0))

        if trailing_mult <= 0 and breakeven_pct <= 0 and max_hold <= 0:
            return signals

        close = data["close"].values.astype(np.float64)
        high = data["high"].values.astype(np.float64)
        low = data["low"].values.astype(np.float64)
        n = len(close)

        # Compute ATR for trailing stop
        atr_period = int(params.get("atr_period", 14))
        atr = self.compute_atr(high, low, close, atr_period) if trailing_mult > 0 else np.zeros(n)

        out = signals.copy()
        position = 0.0
        entry_price = 0.0
        entry_bar = 0
        highest_since_entry = 0.0
        lowest_since_entry = 1e18
        breakeven_active = False

        for i in range(1, n):
            prev_pos = out[i - 1]
            cur_pos = out[i]

            # Detect new entry (flat → position)
            if prev_pos == 0 and cur_pos != 0:
                position = cur_pos
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                breakeven_active = False
                continue

            # Detect strategy-initiated exit (position → flat)
            if prev_pos != 0 and cur_pos == 0:
                position = 0.0
                continue

            # If in position and strategy says hold
            if position != 0 and cur_pos == position:
                highest_since_entry = max(highest_since_entry, high[i])
                lowest_since_entry = min(lowest_since_entry, low[i])
                should_exit = False

                # Max holding period
                if max_hold > 0 and (i - entry_bar) >= max_hold:
                    should_exit = True

                # Trailing stop
                if not should_exit and trailing_mult > 0 and atr[i] > 0:
                    if position > 0:
                        trail_sl = highest_since_entry - trailing_mult * atr[i]
                        if close[i] < trail_sl:
                            should_exit = True
                    elif position < 0:
                        trail_sl = lowest_since_entry + trailing_mult * atr[i]
                        if close[i] > trail_sl:
                            should_exit = True

                # Breakeven stop
                if not should_exit and breakeven_pct > 0 and entry_price > 0:
                    pnl = position * (close[i] - entry_price) / entry_price
                    if not breakeven_active and pnl >= breakeven_pct:
                        breakeven_active = True
                    if breakeven_active:
                        if position > 0 and close[i] <= entry_price:
                            should_exit = True
                        elif position < 0 and close[i] >= entry_price:
                            should_exit = True

                if should_exit:
                    out[i] = 0.0
                    position = 0.0

        return out

    def get_param_with_default(self, params: dict, key: str) -> Any:
        """Get a parameter value, falling back to default."""
        return params.get(key, self.default_params.get(key))

    def validate_params(self, params: dict) -> dict:
        """Merge provided params with defaults."""
        merged = dict(self.default_params)
        merged.update(params)
        return merged

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.strategy_type}')"
