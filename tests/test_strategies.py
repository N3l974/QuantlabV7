"""Tests for all strategies — ensure they produce valid signals."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.registry import STRATEGY_REGISTRY, get_strategy, list_strategies


def make_sample_data(n: int = 500) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.normal(0, 100, n))
    high = close + np.abs(np.random.normal(50, 20, n))
    low = close - np.abs(np.random.normal(50, 20, n))
    open_ = close + np.random.normal(0, 30, n)
    volume = np.abs(np.random.normal(1000, 300, n))

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestStrategyRegistry:
    def test_all_strategies_registered(self):
        assert len(STRATEGY_REGISTRY) == 22

    def test_list_strategies(self):
        names = list_strategies()
        assert len(names) == 22
        assert "rsi_mean_reversion" in names

    def test_get_strategy(self):
        strat = get_strategy("rsi_mean_reversion")
        assert strat.name == "RSI Mean-Reversion"

    def test_get_unknown_strategy(self):
        with pytest.raises(ValueError):
            get_strategy("nonexistent")


class TestAllStrategiesSignals:
    """Test that every strategy produces valid signals on synthetic data."""

    @pytest.fixture
    def data(self):
        return make_sample_data(500)

    @pytest.mark.parametrize("strategy_name", list_strategies())
    def test_signal_shape(self, data, strategy_name):
        strategy = get_strategy(strategy_name)
        signals = strategy.generate_signals(data, strategy.default_params)
        assert len(signals) == len(data)

    @pytest.mark.parametrize("strategy_name", list_strategies())
    def test_signal_values(self, data, strategy_name):
        strategy = get_strategy(strategy_name)
        signals = strategy.generate_signals(data, strategy.default_params)
        unique_vals = set(np.unique(signals))
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    @pytest.mark.parametrize("strategy_name", list_strategies())
    def test_signal_dtype(self, data, strategy_name):
        strategy = get_strategy(strategy_name)
        signals = strategy.generate_signals(data, strategy.default_params)
        assert isinstance(signals, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
