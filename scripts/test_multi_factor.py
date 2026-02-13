#!/usr/bin/env python3
"""
Quick test of the 3 new multi-factor strategies on real data.
Runs walk-forward with 3 seeds on the best timeframes (4h, 1d) and top symbols.
Then runs holdout validation on survivors.

Combos tested: 3 strategies × 3 symbols × 2 TFs = 18 combos
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.ingestion import load_all_symbols_data, load_settings
from engine.backtester import RiskConfig, backtest_strategy
from engine.metrics import compute_all_metrics, composite_score
from engine.walk_forward import WalkForwardConfig, run_walk_forward
from strategies.registry import get_strategy

# ── Config ──
NEW_STRATEGIES = ["supertrend_adx", "trend_multi_factor", "breakout_regime"]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["4h", "1d"]
N_SEEDS = 3
BASE_SEED = 42
CUTOFF_DATE = "2025-02-01"

TF_CONFIGS = {
    "4h": {"training_window": "6M", "reoptim_frequency": "3M", "n_optim_trials": 80},
    "1d": {"training_window": "1Y",  "reoptim_frequency": "3M", "n_optim_trials": 80},
}


def _build_risk(settings):
    rc = settings.get("risk", {})
    return RiskConfig(
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


def test_combo(symbol, strategy_name, timeframe, data, settings, risk, cutoff_dt):
    """Test a single combo: in-sample WF + holdout."""
    tf_cfg = TF_CONFIGS[timeframe]
    strategy = get_strategy(strategy_name)
    commission = settings["engine"]["commission_rate"]
    slippage = settings["engine"]["slippage_rate"]

    in_sample = data[data.index < cutoff_dt]
    holdout = data[data.index >= cutoff_dt]

    if len(in_sample) < 200 or len(holdout) < 30:
        return None

    is_sharpes = []
    ho_sharpes = []
    ho_returns = []
    ho_trades_list = []

    for seed_i in range(N_SEEDS):
        seed = BASE_SEED + seed_i

        wf_config = WalkForwardConfig(
            strategy=strategy,
            data=in_sample,
            timeframe=timeframe,
            reoptim_frequency=tf_cfg["reoptim_frequency"],
            training_window=tf_cfg["training_window"],
            param_bounds_scale=1.0,
            optim_metric="sharpe",
            n_optim_trials=tf_cfg["n_optim_trials"],
            commission=commission,
            slippage=slippage,
            risk=risk,
            seed=seed,
        )

        try:
            is_result = run_walk_forward(wf_config)
        except Exception as e:
            logger.warning(f"    Seed {seed} IS failed: {e}")
            continue

        is_sharpes.append(is_result.metrics.get("sharpe", 0.0))

        if not is_result.best_params_per_period:
            continue

        last_params = is_result.best_params_per_period[-1]

        try:
            ho_result = backtest_strategy(
                strategy, holdout, last_params,
                commission=commission, slippage=slippage,
                initial_capital=10000.0, risk=risk, timeframe=timeframe,
            )
            ho_metrics = compute_all_metrics(ho_result.equity, timeframe, ho_result.trades_pnl)
        except Exception as e:
            logger.warning(f"    Seed {seed} HO failed: {e}")
            continue

        ho_sharpes.append(ho_metrics.get("sharpe", 0.0))
        ho_returns.append(ho_metrics.get("total_return", 0.0))
        ho_trades_list.append(ho_metrics.get("n_trades", 0))

    if not ho_sharpes:
        return None

    return {
        "symbol": symbol,
        "strategy": strategy_name,
        "timeframe": timeframe,
        "n_seeds": len(ho_sharpes),
        "is_sharpe_median": round(float(np.median(is_sharpes)), 4),
        "ho_sharpe_median": round(float(np.median(ho_sharpes)), 4),
        "ho_sharpe_min": round(float(min(ho_sharpes)), 4),
        "ho_sharpe_max": round(float(max(ho_sharpes)), 4),
        "ho_sharpe_std": round(float(np.std(ho_sharpes)), 4),
        "ho_return_median": round(float(np.median(ho_returns)), 4),
        "ho_trades_median": int(np.median(ho_trades_list)),
        "survives": bool(np.median(ho_sharpes) > -0.1),
        "strong": bool(np.median(ho_sharpes) > 0.0 and min(ho_sharpes) > -0.3),
    }


def run_test():
    logger.info("=" * 70)
    logger.info("  MULTI-FACTOR STRATEGY TEST")
    logger.info(f"  Strategies: {NEW_STRATEGIES}")
    logger.info(f"  Symbols: {SYMBOLS} | TFs: {TIMEFRAMES} | Seeds: {N_SEEDS}")
    logger.info(f"  Cutoff: {CUTOFF_DATE}")
    logger.info("=" * 70)

    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    settings = load_settings()
    data_by_symbol = load_all_symbols_data(settings)
    risk = _build_risk(settings)
    cutoff_dt = pd.Timestamp(CUTOFF_DATE)

    results = []
    total = len(NEW_STRATEGIES) * len(SYMBOLS) * len(TIMEFRAMES)
    step = 0

    for strat in NEW_STRATEGIES:
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                step += 1
                logger.info(f"\n[{step}/{total}] {symbol}/{strat}/{tf}")

                data = data_by_symbol.get(symbol, {}).get(tf)
                if data is None:
                    logger.warning(f"  No data, skipping")
                    continue

                result = test_combo(symbol, strat, tf, data, settings, risk, cutoff_dt)
                if result:
                    results.append(result)
                    verdict = "✅" if result["strong"] else ("⚠️" if result["survives"] else "❌")
                    logger.info(f"  {verdict} IS={result['is_sharpe_median']:.3f} → "
                                f"HO={result['ho_sharpe_median']:.3f} "
                                f"(trades={result['ho_trades_median']})")
                else:
                    logger.info(f"  ⛔ No valid results")

    elapsed = time.time() - t0

    # Sort by holdout Sharpe
    results.sort(key=lambda r: r["ho_sharpe_median"], reverse=True)

    # Summary
    strong = [r for r in results if r["strong"]]
    survivors = [r for r in results if r["survives"]]

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  MULTI-FACTOR TEST COMPLETE — {elapsed/60:.1f} min")
    logger.info(f"  {len(results)} tested | {len(strong)} STRONG | "
                f"{len(survivors) - len(strong)} WEAK | {len(results) - len(survivors)} FAIL")
    logger.info(f"{'=' * 70}")

    logger.info(f"\n  {'Combo':<45} {'IS':>7} {'HO':>7} {'HO Ret':>8} {'Trades':>7} {'Verdict':>8}")
    logger.info(f"  {'-'*85}")
    for r in results:
        verdict = "STRONG" if r["strong"] else ("WEAK" if r["survives"] else "FAIL")
        combo = f"{r['symbol']}/{r['strategy']}/{r['timeframe']}"
        logger.info(f"  {combo:<45} {r['is_sharpe_median']:>7.3f} {r['ho_sharpe_median']:>7.3f} "
                     f"{r['ho_return_median']:>7.1%} {r['ho_trades_median']:>7} {verdict:>8}")

    # Save
    output = {
        "timestamp": timestamp,
        "config": {
            "strategies": NEW_STRATEGIES,
            "symbols": SYMBOLS,
            "timeframes": TIMEFRAMES,
            "n_seeds": N_SEEDS,
            "cutoff": CUTOFF_DATE,
        },
        "summary": {
            "total": len(results),
            "strong": len(strong),
            "weak": len(survivors) - len(strong),
            "fail": len(results) - len(survivors),
            "elapsed_min": round(elapsed / 60, 1),
        },
        "results": results,
    }

    json_path = f"results/multi_factor_test_{timestamp}.json"
    Path("results").mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved: {json_path}")

    return results


if __name__ == "__main__":
    run_test()
