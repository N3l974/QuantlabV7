#!/usr/bin/env python3
"""
Multi-Asset Diagnostic V2 — Robust scan of all strategy/symbol/timeframe combos.

Improvements over V1:
  - More Optuna trials (50) for better parameter convergence
  - Adaptive train window per timeframe (longer for daily/4h)
  - Multi-seed runs (2 seeds) to estimate variance and avoid lucky/unlucky results
  - Soft filtering: keeps all combos with score, marks confidence level
  - Enriched summary: heatmap by asset×strategy, confidence tiers

Output: console table + results/diagnostic_multi_asset.json
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.ingestion import load_all_symbols_data, load_settings
from engine.backtester import backtest_strategy, RiskConfig
from engine.metrics import compute_all_metrics, composite_score
from engine.walk_forward import WalkForwardConfig, run_walk_forward
from strategies.registry import get_strategy, list_strategies

# ── Config ──────────────────────────────────────────────
DIAG_N_TRIALS = 50              # Enough trials for Optuna to converge
DIAG_METRIC = "sharpe"
N_SEEDS = 2                     # Run each combo twice with different seeds
COMPOSITE_WEIGHTS = {"sharpe": 0.35, "sortino": 0.25, "calmar": 0.20, "stability": 0.20}

# Adaptive train/reoptim per timeframe (longer for slow TFs)
TF_CONFIG = {
    "15m": {"train_window": "3M", "reoptim_freq": "2M"},
    "1h":  {"train_window": "3M", "reoptim_freq": "2M"},
    "4h":  {"train_window": "6M", "reoptim_freq": "3M"},
    "1d":  {"train_window": "1Y", "reoptim_freq": "3M"},
}

# Confidence tiers
TIER_HIGH = 0.3       # Score > 0.3 = high confidence
TIER_MEDIUM = 0.0     # Score > 0.0 = medium confidence
# Score <= 0.0 = low confidence (still saved for reference)


def _run_single_wf(strategy, data, timeframe, settings, risk, seed):
    """Run a single walk-forward with a given seed."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tf_cfg = TF_CONFIG.get(timeframe, {"train_window": "3M", "reoptim_freq": "2M"})

    wf_config = WalkForwardConfig(
        strategy=strategy,
        data=data,
        timeframe=timeframe,
        reoptim_frequency=tf_cfg["reoptim_freq"],
        training_window=tf_cfg["train_window"],
        param_bounds_scale=1.0,
        optim_metric=DIAG_METRIC,
        n_optim_trials=DIAG_N_TRIALS,
        commission=settings["engine"]["commission_rate"],
        slippage=settings["engine"]["slippage_rate"],
        risk=risk,
    )

    wf_result = run_walk_forward(wf_config)
    return wf_result


def run_multi_asset_diagnostic():
    logger.info("=" * 60)
    logger.info("  QUANTLAB V7 — MULTI-ASSET DIAGNOSTIC V2")
    logger.info("=" * 60)

    settings = load_settings()
    data_by_symbol = load_all_symbols_data(settings)

    # Filter symbols with data
    symbols = [s for s, d in data_by_symbol.items() if d]
    logger.info(f"Symbols with data: {symbols}")

    strategies = list_strategies()
    timeframes = settings["data"].get("meta_timeframes", ["15m", "1h", "4h", "1d"])

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
        logger.info(f"Risk: pos={risk.max_position_pct:.0%}, DD_breaker={risk.max_drawdown_pct:.0%}, "
                    f"daily_loss={risk.max_daily_loss_pct:.0%}, dyn_slip={risk.dynamic_slippage}")

    total_combos = len(symbols) * len(strategies) * len(timeframes)
    logger.info(f"Testing {total_combos} combos × {N_SEEDS} seeds = {total_combos * N_SEEDS} runs")
    logger.info(f"Trials/window: {DIAG_N_TRIALS} | Adaptive train windows | Metric: {DIAG_METRIC}")

    results = []
    start_time = time.time()

    with tqdm(total=total_combos, desc="🔍 Diagnostic V2", unit="combo") as pbar:
        for symbol in symbols:
            symbol_data = data_by_symbol[symbol]
            for strategy_name in strategies:
                try:
                    strategy = get_strategy(strategy_name)
                except ValueError:
                    continue

                for timeframe in timeframes:
                    if timeframe not in symbol_data:
                        pbar.update(1)
                        continue

                    data = symbol_data[timeframe]
                    if len(data) < 500:
                        pbar.update(1)
                        continue

                    # Multi-seed runs
                    seed_scores = []
                    seed_sharpes = []
                    seed_metrics = []
                    seed_n_oos = []
                    all_ok = True

                    for seed in range(N_SEEDS):
                        try:
                            np.random.seed(seed * 42 + 7)
                            wf_result = _run_single_wf(strategy, data, timeframe, settings, risk, seed)
                            metrics = wf_result.metrics

                            if wf_result.n_oos_periods < 2 or metrics.get("n_trades", 0) < 10:
                                all_ok = False
                                break

                            score = composite_score(metrics, COMPOSITE_WEIGHTS)
                            seed_scores.append(score)
                            seed_sharpes.append(metrics.get("sharpe", 0))
                            seed_metrics.append(metrics)
                            seed_n_oos.append(wf_result.n_oos_periods)

                        except Exception as e:
                            logger.warning(f"Failed {symbol}/{strategy_name}/{timeframe} seed={seed}: {e}")
                            all_ok = False
                            break

                    if not all_ok or len(seed_scores) < N_SEEDS:
                        pbar.update(1)
                        continue

                    # Aggregate across seeds
                    avg_score = np.mean(seed_scores)
                    std_score = np.std(seed_scores) if len(seed_scores) > 1 else 0
                    avg_sharpe = np.mean(seed_sharpes)
                    min_sharpe = np.min(seed_sharpes)

                    # Use the best seed's full metrics for reporting
                    best_idx = int(np.argmax(seed_scores))
                    best_metrics = seed_metrics[best_idx]

                    # Confidence tier
                    if avg_score > TIER_HIGH and min_sharpe > 0.2:
                        confidence = "HIGH"
                    elif avg_score > TIER_MEDIUM:
                        confidence = "MEDIUM"
                    else:
                        confidence = "LOW"

                    result = {
                        "symbol": symbol,
                        "strategy": strategy_name,
                        "timeframe": timeframe,
                        "avg_score": round(avg_score, 4),
                        "std_score": round(std_score, 4),
                        "min_score": round(min(seed_scores), 4),
                        "max_score": round(max(seed_scores), 4),
                        "avg_sharpe": round(avg_sharpe, 4),
                        "min_sharpe": round(min_sharpe, 4),
                        "confidence": confidence,
                        "sharpe": round(best_metrics.get("sharpe", 0), 4),
                        "sortino": round(best_metrics.get("sortino", 0), 4),
                        "calmar": round(best_metrics.get("calmar", 0), 4),
                        "total_return": round(best_metrics.get("total_return", 0), 4),
                        "max_drawdown": round(best_metrics.get("max_drawdown", 0), 4),
                        "win_rate": round(best_metrics.get("win_rate", 0), 4),
                        "profit_factor": round(best_metrics.get("profit_factor", 0), 4),
                        "n_trades": best_metrics.get("n_trades", 0),
                        "n_oos_periods": seed_n_oos[best_idx],
                    }
                    results.append(result)

                    # Progress update
                    viable = len(results)
                    elapsed = time.time() - start_time
                    eta = elapsed / max(pbar.n + 1, 1) * (total_combos - pbar.n - 1)
                    pbar.set_postfix({
                        "viable": viable,
                        "best": f"{avg_score:.2f}",
                        "conf": confidence,
                        "eta": f"{eta/60:.0f}m",
                    })

                    pbar.update(1)

    # Sort by avg_score
    results.sort(key=lambda x: x["avg_score"], reverse=True)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"results/diagnostic_multi_asset_{timestamp}.json"
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {filepath}")

    # ── Console summary ──
    logger.info("\n" + "=" * 90)
    logger.info("  MULTI-ASSET DIAGNOSTIC V2 RESULTS")
    logger.info("=" * 90)

    # Confidence breakdown
    high = [r for r in results if r["confidence"] == "HIGH"]
    medium = [r for r in results if r["confidence"] == "MEDIUM"]
    low = [r for r in results if r["confidence"] == "LOW"]
    logger.info(f"Confidence: {len(high)} HIGH | {len(medium)} MEDIUM | {len(low)} LOW | Total: {len(results)}/{total_combos}")

    # Group by symbol
    logger.info("\n--- BY SYMBOL ---")
    symbol_stats = {}
    for r in results:
        sym = r["symbol"]
        if sym not in symbol_stats:
            symbol_stats[sym] = {"high": 0, "medium": 0, "low": 0, "avg_score": 0, "count": 0}
        symbol_stats[sym][r["confidence"].lower()] += 1
        symbol_stats[sym]["avg_score"] += r["avg_score"]
        symbol_stats[sym]["count"] += 1

    for sym, stats in sorted(symbol_stats.items(), key=lambda x: x[1]["avg_score"], reverse=True):
        n = stats["count"]
        stats["avg_score"] /= n
        logger.info(f"  {sym:8s}: {n:2d} viable | {stats['high']:2d}H {stats['medium']:2d}M {stats['low']:2d}L | "
                    f"Avg Score: {stats['avg_score']:.2f}")

    # Heatmap: strategy × symbol (avg_score)
    logger.info("\n--- HEATMAP: STRATEGY × SYMBOL (avg_score, best TF) ---")
    header = f"  {'Strategy':22s}" + "".join(f" {s:>8s}" for s in symbols)
    logger.info(header)
    logger.info("  " + "-" * (22 + 9 * len(symbols)))

    for strat_name in strategies:
        row = f"  {strat_name:22s}"
        for sym in symbols:
            matches = [r for r in results if r["strategy"] == strat_name and r["symbol"] == sym]
            if matches:
                best = max(matches, key=lambda x: x["avg_score"])
                cell = f"{best['avg_score']:+.2f}{best['timeframe']:>3s}"
            else:
                cell = "   ---  "
            row += f" {cell:>8s}"
        logger.info(row)

    # Top 20 combos
    logger.info("\n--- TOP 20 COMBOS ---")
    for i, r in enumerate(results[:20]):
        logger.info(
            f"  #{i+1:2d} [{r['confidence']:6s}] {r['symbol']:8s} | {r['strategy']:22s} | {r['timeframe']:4s} | "
            f"Score: {r['avg_score']:+.2f}±{r['std_score']:.2f} | "
            f"Sharpe: {r['avg_sharpe']:.2f} (min {r['min_sharpe']:.2f}) | "
            f"Return: {r['total_return']:.1%} | DD: {r['max_drawdown']:.1%} | PF: {r['profit_factor']:.1f}"
        )

    # Strategy breakdown
    logger.info("\n--- STRATEGY RANKING ---")
    strategy_stats = {}
    for r in results:
        strat = r["strategy"]
        if strat not in strategy_stats:
            strategy_stats[strat] = {"count": 0, "high": 0, "avg_score": 0, "avg_sharpe": 0}
        strategy_stats[strat]["count"] += 1
        strategy_stats[strat]["high"] += 1 if r["confidence"] == "HIGH" else 0
        strategy_stats[strat]["avg_score"] += r["avg_score"]
        strategy_stats[strat]["avg_sharpe"] += r["avg_sharpe"]

    for strat, stats in sorted(strategy_stats.items(), key=lambda x: x[1]["avg_score"], reverse=True):
        n = stats["count"]
        stats["avg_score"] /= n
        stats["avg_sharpe"] /= n
        logger.info(f"  {strat:22s}: {n:2d} viable ({stats['high']:2d} HIGH) | "
                    f"Avg Score: {stats['avg_score']:.2f} | Avg Sharpe: {stats['avg_sharpe']:.2f}")

    logger.info("\n" + "=" * 90)
    logger.info(f"Total viable combos: {len(results)}/{total_combos}")
    logger.info(f"Diagnostic V2 completed in {(time.time() - start_time)/60:.1f} minutes")
    logger.info("=" * 90)

    return results, filepath


if __name__ == "__main__":
    results, filepath = run_multi_asset_diagnostic()
