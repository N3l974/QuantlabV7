"""
Outer Loop — Meta-Optimizer.
Searches for the optimal optimization framework per strategy using Optuna.
Each trial defines a full walk-forward configuration and evaluates it.
"""

import json
import os
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm

from engine.backtester import RiskConfig
from engine.metrics import composite_score
from engine.walk_forward import WalkForwardConfig, WalkForwardResult, run_walk_forward
from strategies.base import BaseStrategy
from strategies.registry import get_strategy, list_strategies


optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class MetaProfile:
    """A validated meta-optimal profile — result of the outer loop."""
    strategy_name: str
    timeframe: str
    reoptim_frequency: str
    training_window: str
    param_bounds_scale: float
    optim_metric: str
    n_optim_trials: int
    score: float
    metrics: dict
    n_oos_periods: int
    symbol: str = "BTCUSDT"
    best_params_per_period: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "symbol": self.symbol,
            "reoptim_frequency": self.reoptim_frequency,
            "training_window": self.training_window,
            "param_bounds_scale": self.param_bounds_scale,
            "optim_metric": self.optim_metric,
            "n_optim_trials": self.n_optim_trials,
            "score": self.score,
            "metrics": self.metrics,
            "n_oos_periods": self.n_oos_periods,
        }

    def summary(self) -> str:
        return (
            f"{self.symbol} | Strategy: {self.strategy_name} | TF: {self.timeframe} | "
            f"Reoptim: {self.reoptim_frequency} | Window: {self.training_window} | "
            f"Bounds scale: {self.param_bounds_scale:.1f} | "
            f"Sharpe: {self.metrics.get('sharpe', 0):.2f} | "
            f"MaxDD: {self.metrics.get('max_drawdown', 0):.2%} | "
            f"Return: {self.metrics.get('total_return', 0):.2%} | "
            f"Score: {self.score:.3f}"
        )


@dataclass
class MetaOptimizerConfig:
    """Configuration for the meta-optimizer."""
    data_by_timeframe: dict[str, pd.DataFrame]
    # Multi-asset: symbol -> timeframe -> DataFrame
    data_by_symbol: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    strategies: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=list)
    n_outer_trials: int = 500
    outer_metric: str = "sharpe"
    composite_weights: dict = field(default_factory=lambda: {
        "sharpe": 0.35, "sortino": 0.25, "calmar": 0.20, "stability": 0.20
    })
    pruning: bool = True
    pruner_n_warmup: int = 5
    commission: float = 0.001
    slippage: float = 0.0005
    risk: Optional[RiskConfig] = None
    min_trades: int = 30
    min_oos_periods: int = 4
    timeout_hours: float = 48.0
    results_dir: str = "results"

    # Search space
    reoptim_choices: list[str] = field(
        default_factory=lambda: ["1W", "2W", "1M", "2M", "3M", "6M"]
    )
    training_window_choices: list[str] = field(
        default_factory=lambda: ["1M", "2M", "3M", "6M", "1Y", "2Y"]
    )
    optim_metric_choices: list[str] = field(
        default_factory=lambda: ["sharpe", "sortino", "calmar", "pnl_net"]
    )
    n_trials_choices: list[int] = field(
        default_factory=lambda: [50, 100, 200, 500]
    )


def load_meta_config(
    data_by_timeframe: dict[str, pd.DataFrame],
    config_path: str = "config/meta_search_space.yaml",
    settings_path: str = "config/settings.yaml",
    data_by_symbol: Optional[dict[str, dict[str, pd.DataFrame]]] = None,
) -> MetaOptimizerConfig:
    """Load meta-optimizer config from YAML files."""
    with open(config_path, "r") as f:
        meta_cfg = yaml.safe_load(f)
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)

    search = meta_cfg["meta_optimization"]["search_space"]
    meta_settings = meta_cfg["meta_optimization"]["settings"]
    evaluation = meta_cfg["meta_optimization"]["evaluation"]

    # Use filtered strategies/timeframes if defined, else fallback to all
    if "strategies" in meta_cfg["meta_optimization"]:
        strategies = meta_cfg["meta_optimization"]["strategies"]
    else:
        strategies = list_strategies()

    if "meta_timeframes" in settings["data"]:
        timeframes = settings["data"]["meta_timeframes"]
    else:
        timeframes = settings["data"]["timeframes"]

    # Load risk config if present
    risk = None
    if "risk" in settings:
        rc = settings["risk"]
        risk = RiskConfig(
            max_position_pct=rc.get("max_position_pct", 0.25),
            max_daily_loss_pct=rc.get("max_daily_loss_pct", 0.03),
            max_drawdown_pct=rc.get("max_drawdown_pct", 0.15),
            dynamic_slippage=rc.get("dynamic_slippage", True),
            base_slippage=rc.get("base_slippage", 0.0005),
            max_slippage=rc.get("max_slippage", 0.005),
            volatility_lookback=rc.get("volatility_lookback", 20),
            max_trades_per_day=rc.get("max_trades_per_day", 10),
            cooldown_after_loss=rc.get("cooldown_after_loss", 0),
        )

    # Multi-asset support
    symbols = settings["data"].get("symbols", [settings["data"]["symbol"]])
    if data_by_symbol is None:
        # Wrap single-symbol data for backward compatibility
        default_symbol = settings["data"]["symbol"]
        data_by_symbol = {default_symbol: data_by_timeframe}

    return MetaOptimizerConfig(
        data_by_timeframe=data_by_timeframe,
        data_by_symbol=data_by_symbol,
        symbols=symbols,
        strategies=strategies,
        timeframes=timeframes,
        n_outer_trials=meta_settings["n_outer_trials"],
        outer_metric=meta_settings["outer_metric"],
        composite_weights=evaluation["composite_weights"],
        pruning=meta_settings["pruning"],
        pruner_n_warmup=meta_settings["pruner_n_warmup_steps"],
        commission=settings["engine"]["commission_rate"],
        slippage=settings["engine"]["slippage_rate"],
        risk=risk,
        min_trades=evaluation["min_trades"],
        min_oos_periods=evaluation["min_oos_periods"],
        timeout_hours=meta_settings["timeout_hours"],
        reoptim_choices=search["reoptim_frequency"]["choices"],
        training_window_choices=search["training_window"]["choices"],
        optim_metric_choices=search["optim_metric"]["choices"],
        n_trials_choices=search["n_optim_trials"]["choices"],
    )


def _meta_objective(
    trial: optuna.Trial,
    config: MetaOptimizerConfig,
) -> float:
    """
    Objective function for the outer loop.
    Each trial proposes a full walk-forward configuration and evaluates it.
    """
    # Sample meta-parameters
    strategy_name = trial.suggest_categorical("strategy", config.strategies)
    timeframe = trial.suggest_categorical("timeframe", config.timeframes)
    symbol = trial.suggest_categorical("symbol", config.symbols)
    reoptim_freq = trial.suggest_categorical("reoptim_frequency", config.reoptim_choices)
    train_window = trial.suggest_categorical("training_window", config.training_window_choices)
    bounds_scale = trial.suggest_float("param_bounds_scale", 0.3, 1.0, step=0.1)
    optim_metric = trial.suggest_categorical("optim_metric", config.optim_metric_choices)
    n_trials = trial.suggest_categorical("n_optim_trials", config.n_trials_choices)

    # Get data for this symbol/timeframe
    symbol_data = config.data_by_symbol.get(symbol, {})
    if timeframe not in symbol_data:
        logger.warning(f"No data for {symbol}/{timeframe}")
        return -100.0

    data = symbol_data[timeframe]
    if len(data) < 500:
        return -100.0

    # Get strategy
    try:
        strategy = get_strategy(strategy_name)
    except ValueError:
        return -100.0

    # Check strategy supports this timeframe (from strategies.yaml)
    # For now, allow all combinations — the walk-forward will filter bad ones

    # Build walk-forward config
    wf_config = WalkForwardConfig(
        strategy=strategy,
        data=data,
        timeframe=timeframe,
        reoptim_frequency=reoptim_freq,
        training_window=train_window,
        param_bounds_scale=bounds_scale,
        optim_metric=optim_metric if optim_metric != "pnl_net" else "total_return",
        n_optim_trials=n_trials,
        commission=config.commission,
        slippage=config.slippage,
        risk=config.risk,
    )

    # Run walk-forward
    try:
        wf_result = run_walk_forward(wf_config)
    except Exception as e:
        logger.error(f"Walk-forward failed: {e}")
        return -100.0

    # Validate minimum requirements
    if wf_result.n_oos_periods < config.min_oos_periods:
        return -50.0

    if wf_result.metrics.get("n_trades", 0) < config.min_trades:
        return -50.0

    # Compute composite score
    score = composite_score(wf_result.metrics, config.composite_weights)

    # Store extra info in trial
    trial.set_user_attr("metrics", wf_result.metrics)
    trial.set_user_attr("n_oos_periods", wf_result.n_oos_periods)
    trial.set_user_attr("n_trades", wf_result.metrics.get("n_trades", 0))

    trial.set_user_attr("symbol", symbol)

    logger.info(
        f"Trial {trial.number}: {symbol}/{strategy_name}/{timeframe} "
        f"reoptim={reoptim_freq} window={train_window} "
        f"score={score:.3f} sharpe={wf_result.metrics.get('sharpe', 0):.2f}"
    )

    return score


def run_meta_optimization(config: MetaOptimizerConfig) -> list[MetaProfile]:
    """
    Run the full meta-optimization (outer loop).

    Returns:
        List of MetaProfile sorted by score (best first).
    """
    logger.info("=" * 60)
    logger.info("STARTING META-OPTIMIZATION (OUTER LOOP)")
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Strategies: {config.strategies}")
    logger.info(f"Timeframes: {config.timeframes}")
    logger.info(f"Outer trials: {config.n_outer_trials}")
    logger.info(f"Timeout: {config.timeout_hours}h")
    logger.info("=" * 60)

    # Setup pruner
    if config.pruning:
        pruner = optuna.pruners.MedianPruner(
            n_warmup_steps=config.pruner_n_warmup
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="quantlab_v7_meta_optimization",
    )

    # Custom progress tracking
    start_time = _time.time()
    timeout_sec = config.timeout_hours * 3600
    pbar = tqdm(
        total=config.n_outer_trials,
        desc="🧠 Meta-Optimizer",
        unit="trial",
        bar_format=(
            "{desc} | {n}/{total} trials [{elapsed}<{remaining}] "
            "best={postfix}"
        ),
        dynamic_ncols=True,
    )
    pbar.set_postfix_str("searching...")

    completed = [0]
    best_score = [float('-inf')]
    best_info = [""]

    def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        completed[0] += 1
        pbar.update(1)

        elapsed = _time.time() - start_time
        elapsed_str = f"{elapsed/60:.0f}min"

        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            if trial.value > best_score[0]:
                best_score[0] = trial.value
                s = trial.params.get('strategy', '?')
                tf = trial.params.get('timeframe', '?')
                best_info[0] = f"{s}/{tf}"

        if best_score[0] > float('-inf'):
            pbar.set_postfix_str(
                f"score={best_score[0]:.3f} ({best_info[0]}) | {elapsed_str}/{config.timeout_hours:.0f}h"
            )
        else:
            pbar.set_postfix_str(f"searching... | {elapsed_str}/{config.timeout_hours:.0f}h")

    study.optimize(
        lambda trial: _meta_objective(trial, config),
        n_trials=config.n_outer_trials,
        timeout=timeout_sec,
        show_progress_bar=False,
        callbacks=[_progress_callback],
    )
    pbar.close()

    # Extract top profiles
    profiles = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if trial.value is None or trial.value < 0:
            continue

        profile = MetaProfile(
            strategy_name=trial.params.get("strategy", "unknown"),
            timeframe=trial.params.get("timeframe", "1d"),
            reoptim_frequency=trial.params.get("reoptim_frequency", "1M"),
            training_window=trial.params.get("training_window", "6M"),
            param_bounds_scale=trial.params.get("param_bounds_scale", 1.0),
            optim_metric=trial.params.get("optim_metric", "sharpe"),
            n_optim_trials=trial.params.get("n_optim_trials", 100),
            score=trial.value,
            metrics=trial.user_attrs.get("metrics", {}),
            n_oos_periods=trial.user_attrs.get("n_oos_periods", 0),
            symbol=trial.user_attrs.get("symbol", trial.params.get("symbol", "BTCUSDT")),
        )
        profiles.append(profile)

    profiles.sort(key=lambda p: p.score, reverse=True)

    logger.info("=" * 60)
    logger.info(f"META-OPTIMIZATION COMPLETE — {len(profiles)} valid profiles")
    for i, p in enumerate(profiles[:10]):
        logger.info(f"  #{i+1}: {p.summary()}")
    logger.info("=" * 60)

    return profiles


def run_single_strategy_meta(
    strategy_name: str,
    config: MetaOptimizerConfig,
    n_trials: Optional[int] = None,
) -> list[MetaProfile]:
    """
    Run meta-optimization for a single strategy only.
    Useful for focused exploration.
    """
    single_config = MetaOptimizerConfig(
        data_by_timeframe=config.data_by_timeframe,
        strategies=[strategy_name],
        timeframes=config.timeframes,
        n_outer_trials=n_trials or config.n_outer_trials,
        outer_metric=config.outer_metric,
        composite_weights=config.composite_weights,
        pruning=config.pruning,
        pruner_n_warmup=config.pruner_n_warmup,
        commission=config.commission,
        slippage=config.slippage,
        min_trades=config.min_trades,
        min_oos_periods=config.min_oos_periods,
        timeout_hours=config.timeout_hours,
        reoptim_choices=config.reoptim_choices,
        training_window_choices=config.training_window_choices,
        optim_metric_choices=config.optim_metric_choices,
        n_trials_choices=config.n_trials_choices,
    )
    return run_meta_optimization(single_config)


def save_profiles(profiles: list[MetaProfile], output_dir: str = "results") -> str:
    """Save meta-profiles to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"meta_profiles_{timestamp}.json")

    data = [p.to_dict() for p in profiles]
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Saved {len(profiles)} profiles to {filepath}")
    return filepath


def load_profiles(filepath: str) -> list[MetaProfile]:
    """Load meta-profiles from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    profiles = []
    for d in data:
        profiles.append(MetaProfile(**d))

    return profiles
