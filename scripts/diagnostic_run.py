#!/usr/bin/env python3
"""
Diagnostic Run — Quick scan of all strategy/timeframe combos.
Runs a lightweight walk-forward (few trials, short windows) to produce
a clean report showing which combos are viable for full meta-optimization.

Output: console table + results/diagnostic_report.json
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

from data.ingestion import load_all_data, load_settings
from engine.backtester import backtest_strategy, RiskConfig
from engine.metrics import compute_all_metrics, composite_score
from engine.walk_forward import WalkForwardConfig, run_walk_forward
from strategies.registry import get_strategy, list_strategies

# ── Config ──────────────────────────────────────────────
DIAG_N_TRIALS = 20          # Few Optuna trials per window (speed)
DIAG_TRAIN_WINDOW = "3M"    # Short training window
DIAG_REOPTIM_FREQ = "2M"    # Reoptim every 2 months
DIAG_METRIC = "sharpe"
COMPOSITE_WEIGHTS = {"sharpe": 0.35, "sortino": 0.25, "calmar": 0.20, "stability": 0.20}


def run_diagnostic():
    logger.info("=" * 60)
    logger.info("  QUANTLAB V7 — DIAGNOSTIC RUN")
    logger.info("=" * 60)

    settings = load_settings()
    data_by_tf = load_all_data(settings)

    strategies = list_strategies()
    # Use meta_timeframes if available (filtered from previous diagnostic)
    timeframes = settings["data"].get("meta_timeframes", list(data_by_tf.keys()))
    timeframes = [tf for tf in timeframes if tf in data_by_tf]

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

    logger.info(f"Strategies: {len(strategies)} | Timeframes: {len(timeframes)}")
    logger.info(f"Config: {DIAG_N_TRIALS} trials, train={DIAG_TRAIN_WINDOW}, reoptim={DIAG_REOPTIM_FREQ}")

    # Data summary
    logger.info("\n📊 DATA SUMMARY:")
    for tf, df in data_by_tf.items():
        logger.info(f"  {tf:>4s}: {len(df):>10,} candles | {df.index.min()} → {df.index.max()}")

    total_combos = len(strategies) * len(timeframes)
    results = []
    errors = []

    pbar = tqdm(total=total_combos, desc="🔬 Diagnostic", unit="combo",
                bar_format="{desc} | {n}/{total} [{elapsed}<{remaining}] {postfix}")
    pbar.set_postfix_str("starting...")

    for strat_name in strategies:
        strategy = get_strategy(strat_name)
        for tf in timeframes:
            combo_label = f"{strat_name}/{tf}"
            pbar.set_postfix_str(combo_label)

            data = data_by_tf[tf]
            if len(data) < 500:
                results.append({
                    "strategy": strat_name, "timeframe": tf,
                    "status": "SKIP", "reason": f"Not enough data ({len(data)})",
                    "composite": None, "sharpe": None, "sortino": None,
                    "calmar": None, "max_dd": None, "total_return": None,
                    "stability": None, "n_trades": None, "win_rate": None,
                    "n_oos_periods": None, "time_sec": 0,
                })
                pbar.update(1)
                continue

            t0 = time.time()
            try:
                wf_config = WalkForwardConfig(
                    strategy=strategy,
                    data=data,
                    timeframe=tf,
                    reoptim_frequency=DIAG_REOPTIM_FREQ,
                    training_window=DIAG_TRAIN_WINDOW,
                    optim_metric=DIAG_METRIC,
                    n_optim_trials=DIAG_N_TRIALS,
                    commission=settings["engine"]["commission_rate"],
                    slippage=settings["engine"]["slippage_rate"],
                    risk=risk,
                )
                wf_result = run_walk_forward(wf_config)
                elapsed = time.time() - t0

                m = wf_result.metrics
                comp = composite_score(m, COMPOSITE_WEIGHTS)

                status = "OK" if wf_result.n_oos_periods >= 2 and m.get("n_trades", 0) >= 5 else "WEAK"

                results.append({
                    "strategy": strat_name, "timeframe": tf,
                    "status": status,
                    "reason": "",
                    "composite": round(comp, 4),
                    "sharpe": round(m.get("sharpe", 0), 4),
                    "sortino": round(m.get("sortino", 0), 4),
                    "calmar": round(m.get("calmar", 0), 4),
                    "max_dd": round(m.get("max_drawdown", 0), 4),
                    "total_return": round(m.get("total_return", 0), 4),
                    "stability": round(m.get("stability", 0), 4),
                    "n_trades": m.get("n_trades", 0),
                    "win_rate": round(m.get("win_rate", 0), 4) if m.get("win_rate") else None,
                    "n_oos_periods": wf_result.n_oos_periods,
                    "time_sec": round(elapsed, 1),
                })

            except Exception as e:
                elapsed = time.time() - t0
                errors.append({"combo": combo_label, "error": str(e)})
                results.append({
                    "strategy": strat_name, "timeframe": tf,
                    "status": "ERROR", "reason": str(e)[:80],
                    "composite": None, "sharpe": None, "sortino": None,
                    "calmar": None, "max_dd": None, "total_return": None,
                    "stability": None, "n_trades": None, "win_rate": None,
                    "n_oos_periods": None, "time_sec": round(elapsed, 1),
                })

            pbar.update(1)

    pbar.close()

    # ── Print Report ──────────────────────────────────────
    print("\n")
    print("=" * 120)
    print("  QUANTLAB V7 — DIAGNOSTIC REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)

    # Sort by composite score descending
    ok_results = [r for r in results if r["status"] in ("OK", "WEAK")]
    ok_results.sort(key=lambda x: x["composite"] or -999, reverse=True)
    skip_results = [r for r in results if r["status"] in ("SKIP", "ERROR")]

    # Header
    header = f"{'Rank':>4} {'Strategy':<28} {'TF':>4} {'Status':>6} {'Composite':>10} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'MaxDD':>8} {'Return':>8} {'Stab':>6} {'Trades':>7} {'WinR':>6} {'OOS':>4} {'Time':>6}"
    print(header)
    print("-" * 120)

    for i, r in enumerate(ok_results, 1):
        status_icon = "✅" if r["status"] == "OK" else "⚠️"
        wr = f"{r['win_rate']:.3f}" if r['win_rate'] is not None else "  N/A"
        print(
            f"{i:>4} {r['strategy']:<28} {r['timeframe']:>4} {status_icon:>4} "
            f"{r['composite']:>10.4f} {r['sharpe']:>8.4f} {r['sortino']:>8.4f} "
            f"{r['calmar']:>8.4f} {r['max_dd']:>8.4f} {r['total_return']:>8.4f} "
            f"{r['stability']:>6.3f} {r['n_trades']:>7} "
            f"{wr:>6} {r['n_oos_periods']:>4} {r['time_sec']:>5.1f}s"
        )

    if skip_results:
        print(f"\n{'─'*60}")
        print(f"  SKIPPED / ERRORS: {len(skip_results)}")
        for r in skip_results:
            print(f"  ❌ {r['strategy']}/{r['timeframe']}: {r['reason']}")

    # ── Summary Stats ──────────────────────────────────────
    print(f"\n{'='*120}")
    print("  SUMMARY")
    print(f"{'='*120}")

    n_ok = len([r for r in results if r["status"] == "OK"])
    n_weak = len([r for r in results if r["status"] == "WEAK"])
    n_skip = len([r for r in results if r["status"] == "SKIP"])
    n_err = len([r for r in results if r["status"] == "ERROR"])
    total_time = sum(r["time_sec"] for r in results)

    print(f"  Total combos tested: {total_combos}")
    print(f"  ✅ OK:    {n_ok:>3}  (viable for meta-optimization)")
    print(f"  ⚠️  WEAK:  {n_weak:>3}  (few trades or OOS periods)")
    print(f"  ❌ SKIP:  {n_skip:>3}  (insufficient data)")
    print(f"  💥 ERROR: {n_err:>3}")
    print(f"  Total time: {total_time/60:.1f} min")

    if ok_results:
        top = ok_results[0]
        print(f"\n  🏆 BEST COMBO: {top['strategy']}/{top['timeframe']}")
        print(f"     Composite={top['composite']:.4f}  Sharpe={top['sharpe']:.4f}  Return={top['total_return']:.4f}")

        # Top 5 strategies by avg composite
        strat_scores = {}
        for r in ok_results:
            if r["composite"] is not None:
                strat_scores.setdefault(r["strategy"], []).append(r["composite"])
        strat_avg = {s: np.mean(v) for s, v in strat_scores.items()}
        strat_avg_sorted = sorted(strat_avg.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  📊 STRATEGY RANKING (avg composite across timeframes):")
        for i, (s, avg) in enumerate(strat_avg_sorted[:10], 1):
            n = len(strat_scores[s])
            print(f"     {i:>2}. {s:<28} avg={avg:.4f}  ({n} TFs)")

        # Best timeframes
        tf_scores = {}
        for r in ok_results:
            if r["composite"] is not None:
                tf_scores.setdefault(r["timeframe"], []).append(r["composite"])
        tf_avg = {t: np.mean(v) for t, v in tf_scores.items()}
        tf_avg_sorted = sorted(tf_avg.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  ⏱️  TIMEFRAME RANKING (avg composite across strategies):")
        for i, (t, avg) in enumerate(tf_avg_sorted, 1):
            n = len(tf_scores[t])
            print(f"     {i:>2}. {t:<6} avg={avg:.4f}  ({n} strategies)")

    # ── Save JSON ──────────────────────────────────────────
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_trials": DIAG_N_TRIALS,
            "train_window": DIAG_TRAIN_WINDOW,
            "reoptim_freq": DIAG_REOPTIM_FREQ,
            "metric": DIAG_METRIC,
        },
        "results": results,
        "errors": errors,
    }
    report_path = out_dir / "diagnostic_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  💾 Report saved to {report_path}")
    print("=" * 120)


if __name__ == "__main__":
    run_diagnostic()
