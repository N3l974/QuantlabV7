"""Tests for the backtester module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.backtester import vectorized_backtest, BacktestResult, RiskConfig


def _no_risk():
    """RiskConfig with no dynamic slippage for simple tests."""
    return RiskConfig(
        max_position_pct=1.0,
        max_daily_loss_pct=1.0,
        max_drawdown_pct=1.0,
        dynamic_slippage=False,
        max_trades_per_day=1000,
    )


class TestVectorizedBacktest:
    def test_flat_signals(self):
        """No trades when signals are all zero."""
        close = np.array([100, 101, 102, 103, 104], dtype=np.float64)
        signals = np.array([0, 0, 0, 0, 0], dtype=np.float64)
        result = vectorized_backtest(close, signals, risk=_no_risk())
        assert result.n_trades == 0
        assert result.equity[-1] == 10000.0

    def test_long_trade(self):
        """Simple long trade should profit when price goes up."""
        close = np.array([100, 100, 110, 120, 120], dtype=np.float64)
        signals = np.array([0, 1, 1, 1, 0], dtype=np.float64)
        result = vectorized_backtest(close, signals, commission=0, slippage=0, risk=_no_risk())
        assert result.equity[-1] > 10000.0
        assert result.n_trades >= 1

    def test_short_trade(self):
        """Simple short trade should profit when price goes down."""
        close = np.array([100, 100, 90, 80, 80], dtype=np.float64)
        signals = np.array([0, -1, -1, -1, 0], dtype=np.float64)
        result = vectorized_backtest(close, signals, commission=0, slippage=0, risk=_no_risk())
        assert result.equity[-1] > 10000.0

    def test_commission_reduces_profit(self):
        """Commission should reduce final equity."""
        close = np.array([100, 100, 110, 120, 120], dtype=np.float64)
        signals = np.array([0, 1, 1, 1, 0], dtype=np.float64)
        result_no_comm = vectorized_backtest(close, signals, commission=0, slippage=0, risk=_no_risk())
        result_with_comm = vectorized_backtest(close, signals, commission=0.01, slippage=0, risk=_no_risk())
        assert result_with_comm.equity[-1] < result_no_comm.equity[-1]

    def test_equity_length(self):
        """Equity curve should have same length as input."""
        n = 100
        close = np.random.uniform(90, 110, n)
        signals = np.zeros(n)
        result = vectorized_backtest(close, signals, risk=_no_risk())
        assert len(result.equity) == n

    def test_result_structure(self):
        """BacktestResult should have all expected fields."""
        close = np.array([100, 105, 110], dtype=np.float64)
        signals = np.array([0, 1, 0], dtype=np.float64)
        result = vectorized_backtest(close, signals, risk=_no_risk())
        assert isinstance(result, BacktestResult)
        assert isinstance(result.equity, np.ndarray)
        assert isinstance(result.positions, np.ndarray)
        assert isinstance(result.trades_pnl, np.ndarray)
        assert isinstance(result.risk_events, dict)

    def test_position_sizing(self):
        """With 25% position sizing, profit should be ~25% of full allocation."""
        close = np.array([100, 100, 110, 120, 120], dtype=np.float64)
        signals = np.array([0, 1, 1, 1, 0], dtype=np.float64)
        risk_full = RiskConfig(max_position_pct=1.0, dynamic_slippage=False,
                               max_daily_loss_pct=1.0, max_drawdown_pct=1.0)
        risk_quarter = RiskConfig(max_position_pct=0.25, dynamic_slippage=False,
                                   max_daily_loss_pct=1.0, max_drawdown_pct=1.0)
        r_full = vectorized_backtest(close, signals, commission=0, slippage=0, risk=risk_full)
        r_quarter = vectorized_backtest(close, signals, commission=0, slippage=0, risk=risk_quarter)
        profit_full = r_full.equity[-1] - 10000
        profit_quarter = r_quarter.equity[-1] - 10000
        assert profit_quarter < profit_full
        assert profit_quarter > 0

    def test_circuit_breaker(self):
        """Circuit breaker should stop trading after max drawdown."""
        # Price crashes then recovers
        close = np.array([100, 100, 50, 30, 20, 50, 80, 100], dtype=np.float64)
        signals = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.float64)
        risk = RiskConfig(max_position_pct=1.0, max_drawdown_pct=0.10,
                          dynamic_slippage=False, max_daily_loss_pct=1.0)
        result = vectorized_backtest(close, signals, commission=0, slippage=0, risk=risk)
        assert result.risk_events["circuit_breaker_triggers"] >= 1

    def test_risk_events_tracked(self):
        """Risk events dict should always be present."""
        close = np.array([100, 105, 110], dtype=np.float64)
        signals = np.array([0, 1, 0], dtype=np.float64)
        result = vectorized_backtest(close, signals)
        assert "circuit_breaker_triggers" in result.risk_events
        assert "daily_loss_stops" in result.risk_events


    def test_funding_rate_reduces_equity(self):
        """Holding a position should incur funding costs over time."""
        # Long hold for many bars on 1h (funding every 8 bars)
        n = 100
        close = np.full(n, 100.0, dtype=np.float64)  # Flat price
        signals = np.zeros(n, dtype=np.float64)
        signals[1:] = 1.0  # Hold long entire time
        risk = RiskConfig(max_position_pct=1.0, dynamic_slippage=False,
                          max_daily_loss_pct=1.0, max_drawdown_pct=1.0)
        result = vectorized_backtest(close, signals, commission=0, slippage=0,
                                     risk=risk, timeframe="1h")
        # With flat price, only funding costs should reduce equity
        assert result.equity[-1] < 10000.0

    def test_timeframe_daily_reset(self):
        """Daily reset should vary by timeframe."""
        # On 1d, bars_per_day=1, so daily reset happens every bar
        close = np.array([100, 100, 90, 80, 70, 60, 50, 40], dtype=np.float64)
        signals = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.float64)
        risk = RiskConfig(max_position_pct=0.5, dynamic_slippage=False,
                          max_daily_loss_pct=0.05, max_drawdown_pct=1.0)
        # Should not crash and should produce valid equity
        result = vectorized_backtest(close, signals, commission=0, slippage=0,
                                     risk=risk, timeframe="1d")
        assert len(result.equity) == len(close)
        assert result.equity[0] == 10000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
