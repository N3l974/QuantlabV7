#!/usr/bin/env python3
"""
Meta-Optimization V3 — Targeted multi-asset.

Runs the outer loop ONLY on the 5 combos identified by Diagnostic V2:
  1. ETHUSDT / donchian_channel / 1d   (HIGH)
  2. XRPUSDT / stochastic_oscillator / 1d   (HIGH)
  3. BTCUSDT / donchian_channel / 1d   (MEDIUM)
  4. BTCUSDT / ema_ribbon / 1d   (MEDIUM)
  5. ETHUSDT / atr_volatility_breakout / 1d   (MEDIUM)

For each combo, we fix the strategy/symbol/timeframe and search only
the meta-parameters: reoptim_frequency, training_window, param_bounds_scale,
optim_metric, n_optim_trials.

Output: results/meta_profiles_v3_{timestamp}.json
"""

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.ingestion import load_all_symbols_data, load_settings
from engine.backtester import RiskConfig
from engine.metrics import compute_all_metrics, composite_score
from engine.walk_forward import WalkForwardConfig, WalkForwardResult, run_walk_forward
from strategies.registry import get_strategy

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Target combos from Diagnostic V2 ──
TARGET_COMBOS = [
    {"symbol": "ETHUSDT", "strategy": "donchian_channel",       "timeframe": "1d", "diag_confidence": "HIGH"},
    {"symbol": "XRPUSDT", "strategy": "stochastic_oscillator",  "timeframe": "1d", "diag_confidence": "HIGH"},
    {"symbol": "BTCUSDT", "strategy": "donchian_channel",       "timeframe": "1d", "diag_confidence": "MEDIUM"},
    {"symbol": "BTCUSDT", "strategy": "ema_ribbon",             "timeframe": "1d", "diag_confidence": "MEDIUM"},
    {"symbol": "ETHUSDT", "strategy": "atr_volatility_breakout","timeframe": "1d", "diag_confidence": "MEDIUM"},
]

# ── Meta search space ──
REOPTIM_CHOICES = ["1M", "2M", "3M", "6M"]
TRAIN_WINDOW_CHOICES = ["3M", "6M", "1Y", "2Y"]
OPTIM_METRIC_CHOICES = ["sharpe", "sortino", "calmar"]
N_TRIALS_CHOICES = [50, 100, 200]
BOUNDS_SCALE_RANGE = (0.3, 1.0)

# ── Settings ──
N_OUTER_TRIALS = 40       # Per combo (focused search, not random exploration)
COMPOSITE_WEIGHTS = {"sharpe": 0.35, "sortino": 0.25, "calmar": 0.20, "stability": 0.20}
MIN_TRADES = 20
MIN_OOS_PERIODS = 3


def _build_objective(strategy, data, timeframe, settings, risk):
    """Build an Optuna objective for a single combo."""

    def objective(trial: optuna.Trial) -> float:
        reoptim_freq = trial.suggest_categorical("reoptim_frequency", REOPTIM_CHOICES)
        train_window = trial.suggest_categorical("training_window", TRAIN_WINDOW_CHOICES)
        bounds_scale = trial.suggest_float("param_bounds_scale", *BOUNDS_SCALE_RANGE, step=0.1)
        optim_metric = trial.suggest_categorical("optim_metric", OPTIM_METRIC_CHOICES)
        n_trials = trial.suggest_categorical("n_optim_trials", N_TRIALS_CHOICES)

        wf_config = WalkForwardConfig(
            strategy=strategy,
            data=data,
            timeframe=timeframe,
            reoptim_frequency=reoptim_freq,
            training_window=train_window,
            param_bounds_scale=bounds_scale,
            optim_metric=optim_metric,
            n_optim_trials=n_trials,
            commission=settings["engine"]["commission_rate"],
            slippage=settings["engine"]["slippage_rate"],
            risk=risk,
        )

        try:
            wf_result = run_walk_forward(wf_config)
        except Exception as e:
            logger.warning(f"WF failed: {e}")
            return -100.0

        if wf_result.n_oos_periods < MIN_OOS_PERIODS:
            return -50.0
        if wf_result.metrics.get("n_trades", 0) < MIN_TRADES:
            return -50.0

        score = composite_score(wf_result.metrics, COMPOSITE_WEIGHTS)

        # Store for later extraction
        trial.set_user_attr("metrics", wf_result.metrics)
        trial.set_user_attr("n_oos_periods", wf_result.n_oos_periods)
        trial.set_user_attr("best_params_per_period", wf_result.best_params_per_period)

        return score

    return objective


def run_meta_v3():
    logger.info("=" * 70)
    logger.info("  QUANTLAB V7 — META-OPTIMIZATION V3 (TARGETED)")
    logger.info("=" * 70)

    settings = load_settings()
    data_by_symbol = load_all_symbols_data(settings)

    # Load risk config
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

    logger.info(f"Target combos: {len(TARGET_COMBOS)}")
    logger.info(f"Outer trials per combo: {N_OUTER_TRIALS}")
    logger.info(f"Total runs: ~{len(TARGET_COMBOS) * N_OUTER_TRIALS}")

    all_profiles = []
    total_start = time.time()

    for combo_idx, combo in enumerate(TARGET_COMBOS):
        symbol = combo["symbol"]
        strategy_name = combo["strategy"]
        timeframe = combo["timeframe"]
        confidence = combo["diag_confidence"]

        logger.info(f"\n{'─' * 70}")
        logger.info(f"[{combo_idx+1}/{len(TARGET_COMBOS)}] {symbol} / {strategy_name} / {timeframe} ({confidence})")
        logger.info(f"{'─' * 70}")

        # Get data
        symbol_data = data_by_symbol.get(symbol, {})
        if timeframe not in symbol_data:
            logger.error(f"No data for {symbol}/{timeframe}")
            continue

        data = symbol_data[timeframe]
        strategy = get_strategy(strategy_name)

        logger.info(f"Data: {len(data)} bars | Strategy: {strategy.name}")

        # Build objective
        objective = _build_objective(strategy, data, timeframe, settings, risk)

        # Run Optuna
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
            study_name=f"meta_v3_{symbol}_{strategy_name}_{timeframe}",
        )

        combo_start = time.time()
        pbar = tqdm(
            total=N_OUTER_TRIALS,
            desc=f"  🧠 Meta {strategy_name[:15]}",
            unit="trial",
            bar_format="{desc} | {n}/{total} [{elapsed}<{remaining}] best={postfix}",
            leave=True,
        )
        pbar.set_postfix_str("searching...")

        def _callback(study, trial, _pbar=pbar):
            _pbar.update(1)
            if study.best_value is not None and study.best_value > -50:
                _pbar.set_postfix_str(f"{study.best_value:.3f}")

        study.optimize(objective, n_trials=N_OUTER_TRIALS, callbacks=[_callback])
        pbar.close()

        combo_time = time.time() - combo_start
        logger.info(f"  Completed in {combo_time/60:.1f} min")

        # Extract best profile
        if study.best_trial and study.best_value is not None and study.best_value > -50:
            bt = study.best_trial
            metrics = bt.user_attrs.get("metrics", {})
            n_oos = bt.user_attrs.get("n_oos_periods", 0)
            best_params = bt.user_attrs.get("best_params_per_period", [])

            profile = {
                "symbol": symbol,
                "strategy": strategy_name,
                "timeframe": timeframe,
                "diag_confidence": confidence,
                "reoptim_frequency": bt.params.get("reoptim_frequency"),
                "training_window": bt.params.get("training_window"),
                "param_bounds_scale": bt.params.get("param_bounds_scale"),
                "optim_metric": bt.params.get("optim_metric"),
                "n_optim_trials": bt.params.get("n_optim_trials"),
                "score": round(study.best_value, 4),
                "sharpe": round(metrics.get("sharpe", 0), 4),
                "sortino": round(metrics.get("sortino", 0), 4),
                "calmar": round(metrics.get("calmar", 0), 4),
                "total_return": round(metrics.get("total_return", 0), 4),
                "max_drawdown": round(metrics.get("max_drawdown", 0), 4),
                "win_rate": round(metrics.get("win_rate", 0), 4),
                "profit_factor": round(metrics.get("profit_factor", 0), 4),
                "n_trades": metrics.get("n_trades", 0),
                "n_oos_periods": n_oos,
                "last_period_params": best_params[-1] if best_params else {},
            }
            all_profiles.append(profile)

            logger.info(f"  ✅ BEST: Score={profile['score']:.3f} | "
                       f"Sharpe={profile['sharpe']:.2f} | PF={profile['profit_factor']:.2f} | "
                       f"Return={profile['total_return']:.2%} | DD={profile['max_drawdown']:.2%}")
            logger.info(f"     Meta: reoptim={profile['reoptim_frequency']} | "
                       f"train={profile['training_window']} | "
                       f"scale={profile['param_bounds_scale']} | "
                       f"metric={profile['optim_metric']} | "
                       f"trials={profile['n_optim_trials']}")
        else:
            logger.warning(f"  ❌ No viable profile found for {symbol}/{strategy_name}/{timeframe}")

    # Sort by score
    all_profiles.sort(key=lambda x: x["score"], reverse=True)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"results/meta_profiles_v3_{timestamp}.json"
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(all_profiles, f, indent=2, default=str)

    # Final summary
    total_time = time.time() - total_start
    logger.info("\n" + "=" * 70)
    logger.info("  META-OPTIMIZATION V3 — FINAL RESULTS")
    logger.info("=" * 70)

    for i, p in enumerate(all_profiles):
        logger.info(
            f"  #{i+1} [{p['diag_confidence']:6s}] {p['symbol']:8s} | {p['strategy']:22s} | "
            f"Score: {p['score']:+.3f} | Sharpe: {p['sharpe']:.2f} | PF: {p['profit_factor']:.2f} | "
            f"Return: {p['total_return']:.1%} | DD: {p['max_drawdown']:.1%}"
        )
        logger.info(
            f"     Meta: reoptim={p['reoptim_frequency']} train={p['training_window']} "
            f"scale={p['param_bounds_scale']} metric={p['optim_metric']} trials={p['n_optim_trials']}"
        )

    # Viable profiles (PF > 1 and Sharpe > 0)
    viable = [p for p in all_profiles if p["profit_factor"] > 1.0 and p["sharpe"] > 0]
    logger.info(f"\n  Viable for live (PF>1 & Sharpe>0): {len(viable)}/{len(all_profiles)}")

    logger.info(f"\n  Saved to {filepath}")
    logger.info(f"  Total time: {total_time/60:.1f} min")
    logger.info("=" * 70)

    return all_profiles, filepath


if __name__ == "__main__":
    profiles, filepath = run_meta_v3()
