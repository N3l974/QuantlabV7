"""
Portfolio Overlays Module.

Post-signal overlays that modify trading signals or position sizing
to improve risk-adjusted returns. These are applied AFTER signal generation
but BEFORE backtesting.

Overlays:
    1. Regime Overlay: Scale signals by regime exposure scalar (cash in bad regimes)
    2. Volatility Targeting: Scale exposure to target constant annualized vol
    3. Combined: Chain multiple overlays together
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from engine.regime import RegimeConfig, detect_regimes


# ─────────────────────────────────────────────────────────
# Regime Overlay
# ─────────────────────────────────────────────────────────

@dataclass
class RegimeOverlayConfig:
    """Config for regime-based signal scaling."""
    regime_config: RegimeConfig = None
    # If True, regime=RANGE or CRISIS sets signal to 0 (hard cutoff)
    # If False, scale signal by exposure scalar (soft scaling)
    hard_cutoff: bool = False
    # Minimum exposure to keep signal alive (only for soft mode)
    min_exposure_threshold: float = 0.1

    def __post_init__(self):
        if self.regime_config is None:
            self.regime_config = RegimeConfig()


def apply_regime_overlay(
    signals: np.ndarray,
    data: pd.DataFrame,
    config: Optional[RegimeOverlayConfig] = None,
) -> tuple[np.ndarray, dict]:
    """
    Apply regime-based overlay to trading signals.

    In unfavorable regimes (RANGE, CRISIS), reduce or eliminate positions.

    Args:
        signals: Original signal array (+1, -1, 0)
        data: OHLCV DataFrame
        config: RegimeOverlayConfig

    Returns:
        (modified_signals, regime_info) where regime_info contains diagnostics
    """
    if config is None:
        config = RegimeOverlayConfig()

    regime_result = detect_regimes(data, config.regime_config)
    exposure = regime_result["exposure"]

    modified = signals.copy().astype(np.float64)
    n = len(signals)

    n_original_trades = np.sum(signals != 0)
    n_filtered = 0

    for i in range(n):
        if signals[i] == 0:
            continue

        if config.hard_cutoff:
            if exposure[i] < config.min_exposure_threshold:
                modified[i] = 0.0
                n_filtered += 1
        else:
            # Soft scaling: scale signal magnitude by exposure
            modified[i] = signals[i] * exposure[i]
            if abs(modified[i]) < config.min_exposure_threshold:
                modified[i] = 0.0
                n_filtered += 1

    filter_pct = n_filtered / max(n_original_trades, 1) * 100

    logger.debug(
        f"Regime overlay: {n_filtered}/{n_original_trades} signals filtered "
        f"({filter_pct:.1f}%)"
    )

    info = {
        "regime_result": regime_result,
        "n_original_signals": int(n_original_trades),
        "n_filtered": int(n_filtered),
        "filter_pct": float(filter_pct),
    }

    return modified, info


# ─────────────────────────────────────────────────────────
# Volatility Targeting Overlay
# ─────────────────────────────────────────────────────────

@dataclass
class VolTargetConfig:
    """Config for volatility targeting overlay."""
    target_vol_annual: float = 0.15    # 15% annualized target
    vol_lookback: int = 30              # Rolling vol window (bars)
    annualization_factor: float = 365.0 # For daily data; adjust for intraday
    max_leverage: float = 1.5           # Max exposure scalar
    min_exposure: float = 0.1           # Min exposure (don't go to zero)
    smoothing: int = 5                  # Smooth the exposure scalar to avoid whipsaws


# Annualization factors by timeframe
ANNUALIZATION_FACTORS = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}


def apply_vol_targeting(
    signals: np.ndarray,
    data: pd.DataFrame,
    config: Optional[VolTargetConfig] = None,
    timeframe: str = "1h",
) -> tuple[np.ndarray, dict]:
    """
    Apply volatility targeting overlay.

    Scales position sizes to target a constant annualized volatility.
    When vol is high → reduce exposure. When vol is low → increase exposure (up to cap).

    Args:
        signals: Original signal array
        data: OHLCV DataFrame
        config: VolTargetConfig
        timeframe: Data timeframe for annualization

    Returns:
        (modified_signals, vol_info)
    """
    if config is None:
        config = VolTargetConfig()

    close = data["close"].values.astype(np.float64)
    n = len(close)

    # Compute returns
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / np.where(close[:-1] != 0, close[:-1], 1.0)

    # Get annualization factor
    ann_factor = ANNUALIZATION_FACTORS.get(timeframe, config.annualization_factor)

    # Rolling realized volatility (annualized)
    realized_vol = np.zeros(n)
    for i in range(config.vol_lookback, n):
        window = returns[i - config.vol_lookback:i]
        realized_vol[i] = np.std(window) * np.sqrt(ann_factor)

    # Fill early values
    first_valid = config.vol_lookback
    if first_valid < n and realized_vol[first_valid] > 0:
        realized_vol[:first_valid] = realized_vol[first_valid]

    # Compute exposure scalar: target_vol / realized_vol
    exposure_scalar = np.ones(n)
    for i in range(n):
        if realized_vol[i] > 0:
            raw_scalar = config.target_vol_annual / realized_vol[i]
            exposure_scalar[i] = np.clip(raw_scalar, config.min_exposure, config.max_leverage)
        else:
            exposure_scalar[i] = 1.0

    # Smooth the exposure scalar to avoid whipsaws
    if config.smoothing > 1:
        smoothed = np.copy(exposure_scalar)
        for i in range(config.smoothing, n):
            smoothed[i] = np.mean(exposure_scalar[i - config.smoothing:i])
        exposure_scalar = smoothed

    # Apply to signals
    modified = signals.copy().astype(np.float64)
    for i in range(n):
        if signals[i] != 0:
            modified[i] = signals[i] * exposure_scalar[i]

    avg_scalar = np.mean(exposure_scalar[config.vol_lookback:])
    logger.debug(
        f"Vol targeting: target={config.target_vol_annual:.0%}, "
        f"avg realized={np.mean(realized_vol[config.vol_lookback:]):.2%}, "
        f"avg exposure scalar={avg_scalar:.2f}"
    )

    info = {
        "realized_vol": realized_vol,
        "exposure_scalar": exposure_scalar,
        "avg_realized_vol": float(np.mean(realized_vol[config.vol_lookback:])),
        "avg_exposure_scalar": float(avg_scalar),
    }

    return modified, info


# ─────────────────────────────────────────────────────────
# Combined Overlay Pipeline
# ─────────────────────────────────────────────────────────

@dataclass
class OverlayPipelineConfig:
    """Config for chaining multiple overlays."""
    use_regime: bool = True
    use_vol_targeting: bool = True
    regime_config: Optional[RegimeOverlayConfig] = None
    vol_config: Optional[VolTargetConfig] = None


def apply_overlay_pipeline(
    signals: np.ndarray,
    data: pd.DataFrame,
    config: Optional[OverlayPipelineConfig] = None,
    timeframe: str = "1h",
) -> tuple[np.ndarray, dict]:
    """
    Apply a chain of overlays to signals.

    Order: regime overlay → vol targeting (if both enabled).

    Args:
        signals: Original signal array
        data: OHLCV DataFrame
        config: OverlayPipelineConfig
        timeframe: Data timeframe

    Returns:
        (final_signals, combined_info)
    """
    if config is None:
        config = OverlayPipelineConfig()

    current_signals = signals.copy().astype(np.float64)
    combined_info = {}

    # Step 1: Regime overlay
    if config.use_regime:
        current_signals, regime_info = apply_regime_overlay(
            current_signals, data, config.regime_config
        )
        combined_info["regime"] = regime_info

    # Step 2: Volatility targeting
    if config.use_vol_targeting:
        current_signals, vol_info = apply_vol_targeting(
            current_signals, data, config.vol_config, timeframe
        )
        combined_info["vol_targeting"] = vol_info

    # Final stats
    n_original = np.sum(signals != 0)
    n_final = np.sum(current_signals != 0)

    combined_info["summary"] = {
        "n_original_active_bars": int(n_original),
        "n_final_active_bars": int(n_final),
        "bars_removed_pct": float((n_original - n_final) / max(n_original, 1) * 100),
    }

    logger.info(
        f"Overlay pipeline: {n_original} → {n_final} active bars "
        f"({combined_info['summary']['bars_removed_pct']:.1f}% removed)"
    )

    return current_signals, combined_info
