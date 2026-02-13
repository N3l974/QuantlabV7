"""
Market Regime Detection Module.

Classifies market conditions into regimes and provides exposure scalars.
Regimes:
    - STRONG_TREND: ADX > high_threshold, clear direction → full exposure
    - WEAK_TREND: ADX moderate, some direction → reduced exposure
    - RANGE: ADX low, no clear direction → minimal exposure
    - CRISIS: Extreme volatility spike + drawdown → near-zero exposure

This is the #1 edge improvement: not trading in unfavorable conditions
eliminates the biggest source of losses in the current system.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from loguru import logger


class MarketRegime(IntEnum):
    CRISIS = 0
    RANGE = 1
    WEAK_TREND = 2
    STRONG_TREND = 3


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # ADX thresholds
    adx_period: int = 14
    adx_strong: float = 30.0        # ADX > this → strong trend
    adx_weak: float = 20.0          # ADX between weak and strong → weak trend

    # Volatility thresholds (for crisis detection)
    vol_lookback: int = 30           # Rolling vol lookback (bars)
    vol_crisis_mult: float = 3.0     # Vol > median * this → crisis (high for crypto)
    vol_median_lookback: int = 252   # Lookback for vol median (1Y daily equiv)

    # Drawdown threshold for crisis
    dd_crisis_pct: float = 0.20      # DD > 20% from recent peak → crisis overlay
    # Crisis requires BOTH vol spike AND drawdown (True) or EITHER (False)
    crisis_require_both: bool = True

    # Exposure scalars per regime
    exposure_strong_trend: float = 1.0
    exposure_weak_trend: float = 0.6
    exposure_range: float = 0.2
    exposure_crisis: float = 0.0


def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    """Compute ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)

    if n < period * 3:
        return adx

    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))

    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Wilder smoothing
    def wilder_smooth(arr, p):
        result = np.zeros(len(arr))
        if p < len(arr):
            result[p] = np.sum(arr[1:p + 1])
            for i in range(p + 1, len(arr)):
                result[i] = result[i - 1] - result[i - 1] / p + arr[i]
        return result

    atr_smooth = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)

    # DI+ and DI-
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / atr_smooth[i]

    # DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX (smoothed DX)
    adx_start = period * 2
    if adx_start < n:
        adx[adx_start] = np.mean(dx[period:adx_start + 1]) if adx_start > period else 0
        for i in range(adx_start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def compute_rolling_volatility(returns: np.ndarray, lookback: int = 30) -> np.ndarray:
    """Compute rolling standard deviation of returns."""
    n = len(returns)
    vol = np.zeros(n)
    for i in range(lookback, n):
        vol[i] = np.std(returns[i - lookback:i])
    return vol


def detect_regimes(data: pd.DataFrame, config: Optional[RegimeConfig] = None) -> dict:
    """
    Detect market regimes from OHLCV data.

    Args:
        data: DataFrame with columns [open, high, low, close, volume]
        config: RegimeConfig parameters

    Returns:
        dict with:
            - 'regimes': np.ndarray of MarketRegime values
            - 'exposure': np.ndarray of exposure scalars [0.0 - 1.0]
            - 'adx': np.ndarray of ADX values
            - 'rolling_vol': np.ndarray of rolling volatility
            - 'regime_names': list of regime name strings
    """
    if config is None:
        config = RegimeConfig()

    close = data["close"].values.astype(np.float64)
    high = data["high"].values.astype(np.float64)
    low = data["low"].values.astype(np.float64)
    n = len(close)

    # Compute ADX
    adx = compute_adx(high, low, close, config.adx_period)

    # Compute returns and rolling volatility
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / np.where(close[:-1] != 0, close[:-1], 1.0)
    rolling_vol = compute_rolling_volatility(returns, config.vol_lookback)

    # Compute rolling vol median for crisis detection
    vol_median = np.zeros(n)
    for i in range(config.vol_median_lookback, n):
        nonzero_vol = rolling_vol[config.vol_lookback:i + 1]
        nonzero_vol = nonzero_vol[nonzero_vol > 0]
        if len(nonzero_vol) > 0:
            vol_median[i] = np.median(nonzero_vol)
    # Fill early values with first valid median
    first_valid = config.vol_median_lookback
    if first_valid < n and vol_median[first_valid] > 0:
        vol_median[:first_valid] = vol_median[first_valid]

    # Compute rolling drawdown from recent peak
    peak = np.zeros(n)
    dd = np.zeros(n)
    peak[0] = close[0]
    for i in range(1, n):
        peak[i] = max(peak[i - 1], close[i])
        dd[i] = (close[i] - peak[i]) / peak[i] if peak[i] > 0 else 0

    # Classify regimes
    regimes = np.full(n, MarketRegime.RANGE, dtype=np.int32)
    exposure = np.full(n, config.exposure_range, dtype=np.float64)

    warmup = max(config.adx_period * 3, config.vol_lookback, config.vol_median_lookback)

    for i in range(n):
        if i < warmup:
            regimes[i] = MarketRegime.RANGE
            exposure[i] = config.exposure_range
            continue

        # Crisis detection
        vol_spike = (vol_median[i] > 0 and
                     rolling_vol[i] > config.vol_crisis_mult * vol_median[i])
        severe_dd = dd[i] < -config.dd_crisis_pct

        if config.crisis_require_both:
            is_crisis = vol_spike and severe_dd
        else:
            is_crisis = vol_spike or severe_dd

        if is_crisis:
            regimes[i] = MarketRegime.CRISIS
            exposure[i] = config.exposure_crisis
        elif adx[i] >= config.adx_strong:
            regimes[i] = MarketRegime.STRONG_TREND
            exposure[i] = config.exposure_strong_trend
        elif adx[i] >= config.adx_weak:
            regimes[i] = MarketRegime.WEAK_TREND
            exposure[i] = config.exposure_weak_trend
        else:
            regimes[i] = MarketRegime.RANGE
            exposure[i] = config.exposure_range

    # Map regime names
    regime_map = {
        MarketRegime.CRISIS: "CRISIS",
        MarketRegime.RANGE: "RANGE",
        MarketRegime.WEAK_TREND: "WEAK_TREND",
        MarketRegime.STRONG_TREND: "STRONG_TREND",
    }
    regime_names = [regime_map[MarketRegime(r)] for r in regimes]

    logger.debug(
        f"Regime distribution: "
        f"STRONG={np.sum(regimes == MarketRegime.STRONG_TREND)}, "
        f"WEAK={np.sum(regimes == MarketRegime.WEAK_TREND)}, "
        f"RANGE={np.sum(regimes == MarketRegime.RANGE)}, "
        f"CRISIS={np.sum(regimes == MarketRegime.CRISIS)}"
    )

    return {
        "regimes": regimes,
        "exposure": exposure,
        "adx": adx,
        "rolling_vol": rolling_vol,
        "drawdown": dd,
        "regime_names": regime_names,
    }
