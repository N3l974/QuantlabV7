"""
Inner Loop — Walk-Forward Optimizer.
Optimizes strategy parameters on rolling train/test windows using Optuna.
"""

import numpy as np
import pandas as pd
import optuna
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from tqdm import tqdm

from engine.backtester import backtest_strategy, BacktestResult, RiskConfig
from engine.metrics import compute_all_metrics, composite_score, returns_from_equity
from strategies.base import BaseStrategy


optuna.logging.set_verbosity(optuna.logging.WARNING)


FREQ_TO_PERIODS = {
    "1W": {"1m": 7 * 24 * 60, "5m": 7 * 24 * 12, "15m": 7 * 24 * 4,
            "1h": 7 * 24, "4h": 7 * 6, "1d": 7},
    "2W": {"1m": 14 * 24 * 60, "5m": 14 * 24 * 12, "15m": 14 * 24 * 4,
            "1h": 14 * 24, "4h": 14 * 6, "1d": 14},
    "1M": {"1m": 30 * 24 * 60, "5m": 30 * 24 * 12, "15m": 30 * 24 * 4,
            "1h": 30 * 24, "4h": 30 * 6, "1d": 30},
    "2M": {"1m": 60 * 24 * 60, "5m": 60 * 24 * 12, "15m": 60 * 24 * 4,
            "1h": 60 * 24, "4h": 60 * 6, "1d": 60},
    "3M": {"1m": 90 * 24 * 60, "5m": 90 * 24 * 12, "15m": 90 * 24 * 4,
            "1h": 90 * 24, "4h": 90 * 6, "1d": 90},
    "6M": {"1m": 180 * 24 * 60, "5m": 180 * 24 * 12, "15m": 180 * 24 * 4,
            "1h": 180 * 24, "4h": 180 * 6, "1d": 180},
    "1Y": {"1m": 365 * 24 * 60, "5m": 365 * 24 * 12, "15m": 365 * 24 * 4,
            "1h": 365 * 24, "4h": 365 * 6, "1d": 365},
    "2Y": {"1m": 730 * 24 * 60, "5m": 730 * 24 * 12, "15m": 730 * 24 * 4,
            "1h": 730 * 24, "4h": 730 * 6, "1d": 730},
}


def freq_to_n_periods(freq: str, timeframe: str) -> int:
    """Convert a frequency string to number of candle periods."""
    return FREQ_TO_PERIODS.get(freq, {}).get(timeframe, 30 * 24)


@dataclass
class WalkForwardConfig:
    """Configuration for a walk-forward optimization run."""
    strategy: BaseStrategy
    data: pd.DataFrame
    timeframe: str
    reoptim_frequency: str = "1M"
    training_window: str = "6M"
    param_bounds: dict = field(default_factory=dict)
    param_bounds_scale: float = 1.0
    optim_metric: str = "sharpe"
    n_optim_trials: int = 100
    commission: float = 0.001
    slippage: float = 0.0005
    risk: Optional[RiskConfig] = None
    seed: Optional[int] = 42
    use_pruning: bool = False


@dataclass
class WalkForwardResult:
    """Result of a walk-forward optimization."""
    oos_equity: np.ndarray
    oos_returns: np.ndarray
    metrics: dict
    composite: float
    n_oos_periods: int
    best_params_per_period: list[dict]
    period_metrics: list[dict]


def _scale_bounds(default_bounds: list, default_val, scale: float, param_type: str):
    """
    Scale parameter bounds around the default value.
    scale=1.0 -> full default bounds
    scale=0.3 -> tight bounds (30% of range around default)
    """
    low, high = default_bounds
    if scale >= 1.0:
        return low, high

    full_range = high - low
    half_range = full_range * scale / 2.0

    new_low = max(low, default_val - half_range)
    new_high = min(high, default_val + half_range)

    if param_type == "int":
        new_low = int(round(new_low))
        new_high = int(round(new_high))
        if new_low >= new_high:
            new_high = new_low + 1
    else:
        # Guarantee low < high for floats
        if new_low >= new_high:
            new_high = new_low + max(abs(new_low) * 0.1, 0.001)

    return new_low, new_high


def _build_optuna_params(trial: optuna.Trial, strategy: BaseStrategy,
                         bounds_scale: float, custom_bounds: dict) -> dict:
    """Build parameter dict from an Optuna trial."""
    params = {}
    for pname, pinfo in _get_param_info(strategy).items():
        ptype = pinfo["type"]
        default_val = pinfo["default"]
        default_bounds = pinfo["bounds"]

        # Apply custom bounds if provided, otherwise scale defaults
        if pname in custom_bounds:
            low, high = custom_bounds[pname]
        else:
            low, high = _scale_bounds(default_bounds, default_val, bounds_scale, ptype)

        if ptype == "int":
            params[pname] = trial.suggest_int(pname, int(low), int(high))
        elif ptype == "float":
            params[pname] = trial.suggest_float(pname, float(low), float(high))

    return params


V5_ATR_PARAMS = {
    "atr_sl_mult": {"low": 0.0, "high": 4.0},
    "atr_tp_mult": {"low": 0.0, "high": 8.0},
    "trailing_atr_mult": {"low": 0.0, "high": 5.0},
    "breakeven_trigger_pct": {"low": 0.0, "high": 0.05},
    "max_holding_bars": {"low": 0, "high": 200, "type": "int"},
}


def _get_param_info(strategy: BaseStrategy) -> dict:
    """
    Extract parameter info from strategy defaults.
    Returns dict of {name: {type, default, bounds}}.
    """
    info = {}
    for pname, default_val in strategy.default_params.items():
        # V5 ATR params: special handling (allow 0.0 = disabled)
        if pname in V5_ATR_PARAMS:
            spec = V5_ATR_PARAMS[pname]
            ptype = spec.get("type", "float")
            low = spec["low"]
            high = spec["high"]
            info[pname] = {"type": ptype, "default": default_val, "bounds": [low, high]}
            continue

        if isinstance(default_val, int):
            ptype = "int"
            # Default bounds: ±50% around default, minimum range of 2
            low = max(1, int(default_val * 0.3))
            high = int(default_val * 3.0) + 1
            if low >= high:
                high = low + 1
        elif isinstance(default_val, float):
            ptype = "float"
            if default_val <= 0:
                # Handle zero or negative defaults
                low = 0.001
                high = 1.0
            elif default_val < 0.01:
                # Very small values: widen range proportionally
                low = default_val * 0.1
                high = default_val * 10.0
            else:
                low = max(0.001, default_val * 0.2)
                high = default_val * 5.0
            # Final safety: guarantee low < high
            if low >= high:
                high = low + max(abs(low) * 0.1, 0.001)
        else:
            continue
        info[pname] = {"type": ptype, "default": default_val, "bounds": [low, high]}
    return info


def _optuna_tqdm_callback(n_trials: int):
    """Create a tqdm callback for Optuna optimization."""
    pbar = tqdm(
        total=n_trials,
        desc="      🔍 Optuna",
        unit="trial",
        bar_format="{desc} | {n}/{total} trials [{elapsed}<{remaining}, {rate_fmt}] best={postfix}",
        leave=False,
        dynamic_ncols=True,
    )
    pbar.set_postfix_str("?")

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        pbar.update(1)
        if study.best_value is not None and study.best_value > -50:
            pbar.set_postfix_str(f"{study.best_value:.3f}")
        if pbar.n >= n_trials:
            pbar.close()

    return callback


def _optimize_on_window(
    strategy: BaseStrategy,
    train_data: pd.DataFrame,
    config: WalkForwardConfig,
    window_seed: Optional[int] = None,
) -> dict:
    """
    Run Optuna optimization on a single training window.
    Returns the best parameters found.
    Uses a deterministic TPE sampler when a seed is provided.
    """
    metric_name = config.optim_metric

    def objective(trial: optuna.Trial) -> float:
        params = _build_optuna_params(
            trial, strategy, config.param_bounds_scale, config.param_bounds
        )
        try:
            result = backtest_strategy(
                strategy, train_data, params,
                commission=config.commission,
                slippage=config.slippage,
                risk=config.risk,
                timeframe=config.timeframe,
            )
            if result.n_trades < 5:
                return -100.0

            metrics = compute_all_metrics(
                result.equity, config.timeframe, result.trades_pnl
            )
            return metrics.get(metric_name, 0.0)
        except Exception:
            return -100.0

    sampler = (
        optuna.samplers.TPESampler(seed=window_seed)
        if window_seed is not None
        else optuna.samplers.TPESampler()
    )
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        if config.use_pruning
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=config.n_optim_trials, show_progress_bar=False,
                   callbacks=[_optuna_tqdm_callback(config.n_optim_trials)])

    return study.best_params


def run_walk_forward(config: WalkForwardConfig) -> WalkForwardResult:
    """
    Execute a full walk-forward optimization.

    Splits data into rolling train/test windows, optimizes on each training
    window, then tests on the subsequent OOS period. Concatenates all OOS
    equity curves into a composite result.
    """
    data = config.data
    n = len(data)

    train_size = freq_to_n_periods(config.training_window, config.timeframe)
    test_size = freq_to_n_periods(config.reoptim_frequency, config.timeframe)

    if train_size + test_size > n:
        logger.warning(f"Not enough data: need {train_size + test_size}, have {n}")
        return WalkForwardResult(
            oos_equity=np.array([10000.0]),
            oos_returns=np.array([0.0]),
            metrics={"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
                     "calmar": 0.0, "total_return": 0.0, "stability": 0.0,
                     "n_periods": 0, "n_trades": 0, "win_rate": 0.0,
                     "profit_factor": 0.0},
            composite=0.0,
            n_oos_periods=0,
            best_params_per_period=[],
            period_metrics=[],
        )

    all_oos_equity = []
    all_oos_trades = []
    best_params_list = []
    period_metrics_list = []
    current_capital = 10000.0

    start = 0
    period_count = 0

    # Estimate total periods for progress bar
    estimated_periods = max(1, (n - train_size) // test_size)
    strategy_name = getattr(config.strategy, 'name', 'strategy')
    pbar = tqdm(
        total=estimated_periods,
        desc=f"    📈 WF {strategy_name}/{config.timeframe}",
        unit="period",
        bar_format="{desc} | {n}/{total} periods [{elapsed}<{remaining}] capital=${postfix}",
        leave=False,
        dynamic_ncols=True,
    )
    pbar.set_postfix_str(f"{current_capital:,.0f}")

    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = min(train_end + test_size, n)

        train_data = data.iloc[start:train_end]
        test_data = data.iloc[train_end:test_end]

        if len(test_data) < 10:
            break

        # Compute per-window seed for deterministic optimization
        window_seed = (config.seed * 1000 + period_count) if config.seed is not None else None

        # Optimize on training window
        best_params = _optimize_on_window(config.strategy, train_data, config, window_seed=window_seed)
        best_params_list.append(best_params)

        # Test on OOS window
        oos_result = backtest_strategy(
            config.strategy, test_data, best_params,
            commission=config.commission,
            slippage=config.slippage,
            initial_capital=current_capital,
            risk=config.risk,
            timeframe=config.timeframe,
        )

        # Collect OOS results
        all_oos_equity.append(oos_result.equity)
        all_oos_trades.extend(oos_result.trades_pnl.tolist())

        # Compute period metrics
        p_metrics = compute_all_metrics(
            oos_result.equity, config.timeframe, oos_result.trades_pnl
        )
        period_metrics_list.append(p_metrics)

        # Update capital for next period
        current_capital = oos_result.equity[-1] if len(oos_result.equity) > 0 else current_capital

        # Slide forward
        start += test_size
        period_count += 1

        pbar.update(1)
        pbar.set_postfix_str(f"{current_capital:,.0f}")

    pbar.close()

    # Build composite equity curve
    if all_oos_equity:
        composite_equity = np.concatenate(all_oos_equity)
    else:
        composite_equity = np.array([10000.0])

    trades_pnl = np.array(all_oos_trades) if all_oos_trades else np.array([])

    # Compute overall metrics
    overall_metrics = compute_all_metrics(composite_equity, config.timeframe, trades_pnl)
    comp_score = composite_score(overall_metrics)

    logger.info(
        f"Walk-forward complete: {period_count} OOS periods, "
        f"Sharpe={overall_metrics['sharpe']:.2f}, "
        f"Return={overall_metrics['total_return']:.2%}, "
        f"MaxDD={overall_metrics['max_drawdown']:.2%}"
    )

    return WalkForwardResult(
        oos_equity=composite_equity,
        oos_returns=returns_from_equity(composite_equity) if len(composite_equity) > 1 else np.array([0.0]),
        metrics=overall_metrics,
        composite=comp_score,
        n_oos_periods=period_count,
        best_params_per_period=best_params_list,
        period_metrics=period_metrics_list,
    )


def run_walk_forward_robust(
    config: WalkForwardConfig,
    n_seeds: int = 5,
    aggregation: str = "median",
) -> WalkForwardResult:
    """
    Run walk-forward multiple times with different seeds and aggregate results.

    This reduces the variance caused by stochastic Optuna optimization.
    Each seed produces a different TPE sampling sequence, leading to different
    parameter selections per window. Aggregating across seeds gives a more
    reliable estimate of true OOS performance.

    Args:
        config: Walk-forward configuration (config.seed is used as base seed).
        n_seeds: Number of independent seeds to run (default 5).
        aggregation: "median" (robust to outliers) or "mean".

    Returns:
        WalkForwardResult with the seed closest to the aggregated Sharpe.
    """
    base_seed = config.seed if config.seed is not None else 42
    results: list[WalkForwardResult] = []
    seed_sharpes: list[float] = []

    logger.info(f"🔄 Robust WF: running {n_seeds} seeds for "
                f"{getattr(config.strategy, 'name', '?')}/{config.timeframe}")

    for i in range(n_seeds):
        seed_config = WalkForwardConfig(
            strategy=config.strategy,
            data=config.data,
            timeframe=config.timeframe,
            reoptim_frequency=config.reoptim_frequency,
            training_window=config.training_window,
            param_bounds=config.param_bounds,
            param_bounds_scale=config.param_bounds_scale,
            optim_metric=config.optim_metric,
            n_optim_trials=config.n_optim_trials,
            commission=config.commission,
            slippage=config.slippage,
            risk=config.risk,
            seed=base_seed + i,
        )
        try:
            result = run_walk_forward(seed_config)
            results.append(result)
            sharpe = result.metrics.get("sharpe", 0.0)
            seed_sharpes.append(sharpe)
            logger.info(f"  Seed {base_seed + i}: Sharpe={sharpe:.3f}, "
                        f"Return={result.metrics.get('total_return', 0):.2%}, "
                        f"DD={result.metrics.get('max_drawdown', 0):.2%}")
        except Exception as e:
            logger.warning(f"  Seed {base_seed + i} failed: {e}")
            continue

    if not results:
        logger.error("All seeds failed in robust walk-forward")
        return run_walk_forward(config)

    # Aggregate
    sharpes = np.array(seed_sharpes)
    if aggregation == "median":
        target = float(np.median(sharpes))
    else:
        target = float(np.mean(sharpes))

    # Pick the seed result closest to the aggregated value
    best_idx = int(np.argmin(np.abs(sharpes - target)))
    chosen = results[best_idx]

    # Log summary
    logger.info(
        f"🔄 Robust WF summary ({n_seeds} seeds): "
        f"Sharpe min={sharpes.min():.3f} / {aggregation}={target:.3f} / max={sharpes.max():.3f} | "
        f"std={sharpes.std():.3f} | chosen seed={base_seed + best_idx} (Sharpe={sharpes[best_idx]:.3f})"
    )

    # Enrich metrics with cross-seed stats
    chosen.metrics["robust_n_seeds"] = n_seeds
    chosen.metrics["robust_sharpe_median"] = float(np.median(sharpes))
    chosen.metrics["robust_sharpe_mean"] = float(np.mean(sharpes))
    chosen.metrics["robust_sharpe_std"] = float(np.std(sharpes))
    chosen.metrics["robust_sharpe_min"] = float(sharpes.min())
    chosen.metrics["robust_sharpe_max"] = float(sharpes.max())
    chosen.metrics["robust_consistency"] = float(
        1.0 - sharpes.std() / max(abs(np.mean(sharpes)), 0.001)
    ) if np.mean(sharpes) != 0 else 0.0

    return chosen
