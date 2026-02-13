"""
Strategy registry — central catalog of all available strategies.
"""

from strategies.base import BaseStrategy
from strategies.rsi_mean_reversion import RSIMeanReversion
from strategies.macd_crossover import MACDCrossover
from strategies.bollinger_breakout import BollingerBreakout
from strategies.ema_ribbon import EMARibbon
from strategies.vwap_deviation import VWAPDeviation
from strategies.donchian_channel import DonchianChannel
from strategies.stochastic_oscillator import StochasticOscillator
from strategies.ichimoku_cloud import IchimokuCloud
from strategies.atr_volatility_breakout import ATRVolatilityBreakout
from strategies.volume_obv import VolumeOBV
from strategies.momentum_roc import MomentumROC
from strategies.adx_regime import ADXRegime
from strategies.keltner_channel import KeltnerChannel
from strategies.mean_reversion_zscore import MeanReversionZScore
from strategies.supertrend import SuperTrend
from strategies.williams_r import WilliamsR
from strategies.supertrend_adx import SuperTrendADX
from strategies.trend_multi_factor import TrendMultiFactor
from strategies.breakout_regime import BreakoutRegime
from strategies.mtf_trend_entry import MTFTrendEntry
from strategies.mtf_momentum_breakout import MTFMomentumBreakout
from strategies.regime_adaptive import RegimeAdaptive


STRATEGY_REGISTRY: dict[str, BaseStrategy] = {
    "rsi_mean_reversion": RSIMeanReversion(),
    "macd_crossover": MACDCrossover(),
    "bollinger_breakout": BollingerBreakout(),
    "ema_ribbon": EMARibbon(),
    "vwap_deviation": VWAPDeviation(),
    "donchian_channel": DonchianChannel(),
    "stochastic_oscillator": StochasticOscillator(),
    "ichimoku_cloud": IchimokuCloud(),
    "atr_volatility_breakout": ATRVolatilityBreakout(),
    "volume_obv": VolumeOBV(),
    "momentum_roc": MomentumROC(),
    "adx_regime": ADXRegime(),
    "keltner_channel": KeltnerChannel(),
    "mean_reversion_zscore": MeanReversionZScore(),
    "supertrend": SuperTrend(),
    "williams_r": WilliamsR(),
    "supertrend_adx": SuperTrendADX(),
    "trend_multi_factor": TrendMultiFactor(),
    "breakout_regime": BreakoutRegime(),
    "mtf_trend_entry": MTFTrendEntry(),
    "mtf_momentum_breakout": MTFMomentumBreakout(),
    "regime_adaptive": RegimeAdaptive(),
}


def get_strategy(name: str) -> BaseStrategy:
    """Get a strategy instance by name."""
    if name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """List all registered strategy names."""
    return list(STRATEGY_REGISTRY.keys())
