"""
Performance metrics for strategy evaluation.
All metrics operate on equity curves or return series (numpy arrays).
"""

import numpy as np
import pandas as pd


def returns_from_equity(equity: np.ndarray) -> np.ndarray:
    """Compute simple returns from an equity curve."""
    returns = np.diff(equity) / equity[:-1]
    return returns


def annualization_factor(timeframe: str) -> float:
    """
    Return the number of periods per year for annualization.
    """
    factors = {
        "1m": 365.25 * 24 * 60,
        "5m": 365.25 * 24 * 12,
        "15m": 365.25 * 24 * 4,
        "1h": 365.25 * 24,
        "4h": 365.25 * 6,
        "1d": 365.25,
    }
    return factors.get(timeframe, 365.25)


def sharpe_ratio(returns: np.ndarray, timeframe: str = "1d", risk_free: float = 0.0) -> float:
    """
    Annualized Sharpe Ratio.
    """
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    excess = returns - risk_free / annualization_factor(timeframe)
    factor = annualization_factor(timeframe)
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(factor))


def sortino_ratio(returns: np.ndarray, timeframe: str = "1d", risk_free: float = 0.0) -> float:
    """
    Annualized Sortino Ratio (downside deviation only).
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / annualization_factor(timeframe)
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return float(np.mean(excess)) * 100  # No downside = very good
    factor = annualization_factor(timeframe)
    return float(np.mean(excess) / np.std(downside, ddof=1) * np.sqrt(factor))


def max_drawdown(equity: np.ndarray) -> float:
    """
    Maximum drawdown as a negative fraction (e.g. -0.15 = -15%).
    """
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(np.min(drawdown))


def calmar_ratio(equity: np.ndarray, returns: np.ndarray, timeframe: str = "1d") -> float:
    """
    Calmar Ratio = annualized return / |max drawdown|.
    """
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    factor = annualization_factor(timeframe)
    ann_return = np.mean(returns) * factor
    return float(ann_return / abs(mdd))


def total_return(equity: np.ndarray) -> float:
    """Total return as a fraction."""
    if len(equity) < 2 or equity[0] == 0:
        return 0.0
    return float((equity[-1] - equity[0]) / equity[0])


def win_rate(trades_pnl: np.ndarray) -> float:
    """Fraction of winning trades."""
    if len(trades_pnl) == 0:
        return 0.0
    return float(np.sum(trades_pnl > 0) / len(trades_pnl))


def profit_factor(trades_pnl: np.ndarray) -> float:
    """Gross profit / gross loss."""
    gross_profit = np.sum(trades_pnl[trades_pnl > 0])
    gross_loss = abs(np.sum(trades_pnl[trades_pnl < 0]))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def stability_score(returns: np.ndarray, timeframe: str = "1d", n_splits: int = 4) -> float:
    """
    Stability = 1 - std(sub-period Sharpes) / mean(sub-period Sharpes).
    Higher is better (more consistent performance across sub-periods).
    Returns a value between 0 and 1 (clamped).
    """
    if len(returns) < n_splits * 10:
        return 0.0
    chunk_size = len(returns) // n_splits
    sharpes = []
    for i in range(n_splits):
        chunk = returns[i * chunk_size : (i + 1) * chunk_size]
        s = sharpe_ratio(chunk, timeframe)
        sharpes.append(s)
    sharpes = np.array(sharpes)
    mean_s = np.mean(sharpes)
    if mean_s == 0:
        return 0.0
    score = 1.0 - np.std(sharpes) / abs(mean_s)
    return float(np.clip(score, 0.0, 1.0))


def compute_all_metrics(
    equity: np.ndarray,
    timeframe: str = "1d",
    trades_pnl: np.ndarray | None = None,
) -> dict:
    """
    Compute all performance metrics from an equity curve.

    Returns:
        Dictionary with all metric values.
    """
    returns = returns_from_equity(equity)

    metrics = {
        "sharpe": sharpe_ratio(returns, timeframe),
        "sortino": sortino_ratio(returns, timeframe),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity, returns, timeframe),
        "total_return": total_return(equity),
        "stability": stability_score(returns, timeframe),
        "n_periods": len(equity),
    }

    if trades_pnl is not None and len(trades_pnl) > 0:
        metrics["n_trades"] = len(trades_pnl)
        metrics["win_rate"] = win_rate(trades_pnl)
        metrics["profit_factor"] = profit_factor(trades_pnl)
    else:
        metrics["n_trades"] = 0
        metrics["win_rate"] = 0.0
        metrics["profit_factor"] = 0.0

    return metrics


def composite_score(metrics: dict, weights: dict | None = None) -> float:
    """
    Compute a weighted composite score from metrics.

    Default weights from meta_search_space.yaml:
        sharpe: 0.35, sortino: 0.25, calmar: 0.20, stability: 0.20
    """
    if weights is None:
        weights = {
            "sharpe": 0.35,
            "sortino": 0.25,
            "calmar": 0.20,
            "stability": 0.20,
        }

    score = 0.0
    for key, w in weights.items():
        val = metrics.get(key, 0.0)
        if np.isnan(val) or np.isinf(val):
            val = 0.0
        score += w * val

    return float(score)
