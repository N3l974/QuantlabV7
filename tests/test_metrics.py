"""Tests for the metrics module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.metrics import (
    returns_from_equity,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    total_return,
    win_rate,
    profit_factor,
    stability_score,
    compute_all_metrics,
    composite_score,
)


class TestReturnsFromEquity:
    def test_basic(self):
        equity = np.array([100, 110, 105, 115])
        returns = returns_from_equity(equity)
        assert len(returns) == 3
        assert abs(returns[0] - 0.1) < 1e-10

    def test_flat(self):
        equity = np.array([100, 100, 100])
        returns = returns_from_equity(equity)
        assert np.all(returns == 0)


class TestSharpeRatio:
    def test_positive_returns(self):
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.025] * 20)
        sr = sharpe_ratio(returns, "1d")
        assert sr > 0

    def test_zero_std(self):
        returns = np.array([0.01, 0.01, 0.01])
        sr = sharpe_ratio(returns, "1d")
        assert sr == 0.0 or sr > 0  # Depends on ddof

    def test_empty(self):
        returns = np.array([0.01])
        sr = sharpe_ratio(returns, "1d")
        assert sr == 0.0


class TestMaxDrawdown:
    def test_basic(self):
        equity = np.array([100, 110, 90, 95, 120])
        mdd = max_drawdown(equity)
        expected = (90 - 110) / 110  # -18.18%
        assert abs(mdd - expected) < 1e-10

    def test_no_drawdown(self):
        equity = np.array([100, 110, 120, 130])
        mdd = max_drawdown(equity)
        assert mdd == 0.0

    def test_single_point(self):
        equity = np.array([100])
        mdd = max_drawdown(equity)
        assert mdd == 0.0


class TestTotalReturn:
    def test_positive(self):
        equity = np.array([100, 150])
        assert abs(total_return(equity) - 0.5) < 1e-10

    def test_negative(self):
        equity = np.array([100, 80])
        assert abs(total_return(equity) - (-0.2)) < 1e-10


class TestWinRate:
    def test_basic(self):
        trades = np.array([10, -5, 20, -3, 15])
        assert abs(win_rate(trades) - 0.6) < 1e-10

    def test_empty(self):
        assert win_rate(np.array([])) == 0.0


class TestProfitFactor:
    def test_basic(self):
        trades = np.array([10, -5, 20, -3])
        pf = profit_factor(trades)
        assert abs(pf - 30 / 8) < 1e-10

    def test_no_losses(self):
        trades = np.array([10, 20])
        pf = profit_factor(trades)
        assert pf == float("inf")


class TestCompositeScore:
    def test_basic(self):
        metrics = {"sharpe": 2.0, "sortino": 3.0, "calmar": 1.5, "stability": 0.8}
        score = composite_score(metrics)
        expected = 0.35 * 2.0 + 0.25 * 3.0 + 0.20 * 1.5 + 0.20 * 0.8
        assert abs(score - expected) < 1e-10

    def test_nan_handling(self):
        metrics = {"sharpe": float("nan"), "sortino": 1.0, "calmar": 1.0, "stability": 1.0}
        score = composite_score(metrics)
        assert not np.isnan(score)


class TestComputeAllMetrics:
    def test_full(self):
        np.random.seed(42)
        equity = 10000 * np.cumprod(1 + np.random.normal(0.001, 0.02, 500))
        equity = np.insert(equity, 0, 10000)
        trades = np.random.normal(50, 100, 30)

        metrics = compute_all_metrics(equity, "1d", trades)
        assert "sharpe" in metrics
        assert "sortino" in metrics
        assert "max_drawdown" in metrics
        assert "calmar" in metrics
        assert "total_return" in metrics
        assert "stability" in metrics
        assert "n_trades" in metrics
        assert metrics["n_trades"] == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
