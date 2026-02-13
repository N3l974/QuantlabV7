#!/usr/bin/env python3
"""
Holdout Temporal Test — Validate Diagnostic V4 MEDIUM combos on unseen data.

Methodology:
  1. Split data at CUTOFF_DATE (default: 2025-02-01)
  2. Train walk-forward ONLY on data BEFORE cutoff (in-sample)
  3. Use the LAST optimized params from in-sample to trade the holdout period
  4. Multi-seed (5 seeds) for robustness
  5. Compare in-sample vs holdout performance

The 5 MEDIUM combos from Diagnostic V4:
  1. ETH / supertrend / 4h      (Score 0.130, Sharpe 0.166)
  2. SOL / bollinger_breakout / 4h (Score 0.128, Sharpe 0.194)
  3. BTC / momentum_roc / 1d     (Score 0.121, Sharpe 0.222)
  4. ETH / atr_volatility_breakout / 1d (Score 0.045, Sharpe 0.078)
  5. ETH / volume_obv / 4h       (Score 0.024, Sharpe 0.043)

Output:
  - results/holdout_test_{timestamp}.json
  - docs/results/09_holdout_test.md
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
from engine.metrics import compute_all_metrics, composite_score, returns_from_equity
from engine.walk_forward import WalkForwardConfig, run_walk_forward, run_walk_forward_robust
from strategies.registry import get_strategy

# ── Config ──
CUTOFF_DATE = "2025-02-01"
N_SEEDS = 5
BASE_SEED = 42

# The 5 MEDIUM combos from Diagnostic V4
COMBOS = [
    {"symbol": "ETHUSDT",  "strategy": "supertrend",              "timeframe": "4h"},
    {"symbol": "SOLUSDT",  "strategy": "bollinger_breakout",      "timeframe": "4h"},
    {"symbol": "BTCUSDT",  "strategy": "momentum_roc",            "timeframe": "1d"},
    {"symbol": "ETHUSDT",  "strategy": "atr_volatility_breakout", "timeframe": "1d"},
    {"symbol": "ETHUSDT",  "strategy": "volume_obv",              "timeframe": "4h"},
]

# Walk-forward defaults (same as Diagnostic V4)
TF_CONFIGS = {
    "4h": {"training_window": "6M", "reoptim_frequency": "3M", "n_optim_trials": 100},
    "1d": {"training_window": "1Y",  "reoptim_frequency": "3M", "n_optim_trials": 100},
}


def _build_risk(settings: dict) -> RiskConfig:
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


def run_holdout_for_combo(combo, data_by_symbol, settings, risk, cutoff_dt):
    """
    Run holdout test for a single combo:
    1. Split data at cutoff
    2. Walk-forward on in-sample (multi-seed) → get last params per seed
    3. Apply each seed's last params on holdout → backtest
    4. Aggregate holdout results
    """
    symbol = combo["symbol"]
    strategy_name = combo["strategy"]
    timeframe = combo["timeframe"]
    tf_cfg = TF_CONFIGS[timeframe]

    logger.info(f"\n{'─' * 60}")
    logger.info(f"  {symbol} / {strategy_name} / {timeframe}")
    logger.info(f"{'─' * 60}")

    data = data_by_symbol.get(symbol, {}).get(timeframe)
    if data is None or len(data) == 0:
        logger.error(f"  No data for {symbol}/{timeframe}")
        return None

    # Split at cutoff
    in_sample = data[data.index < cutoff_dt]
    holdout = data[data.index >= cutoff_dt]

    logger.info(f"  In-sample:  {in_sample.index.min()} → {in_sample.index.max()} ({len(in_sample)} bars)")
    logger.info(f"  Holdout:    {holdout.index.min()} → {holdout.index.max()} ({len(holdout)} bars)")

    if len(holdout) < 30:
        logger.warning(f"  Holdout too short ({len(holdout)} bars), skipping")
        return None

    strategy = get_strategy(strategy_name)
    commission = settings["engine"]["commission_rate"]
    slippage = settings["engine"]["slippage_rate"]

    # ── In-sample: walk-forward with multi-seed ──
    is_sharpes = []
    holdout_sharpes = []
    holdout_returns = []
    holdout_dds = []
    holdout_metrics_list = []
    seed_details = []

    for seed_i in range(N_SEEDS):
        seed = BASE_SEED + seed_i
        logger.info(f"  Seed {seed}: running in-sample walk-forward...")

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
            logger.warning(f"  Seed {seed} in-sample failed: {e}")
            continue

        is_sharpe = is_result.metrics.get("sharpe", 0.0)
        is_sharpes.append(is_sharpe)

        # Get last optimized params
        if not is_result.best_params_per_period:
            logger.warning(f"  Seed {seed}: no params found")
            continue

        last_params = is_result.best_params_per_period[-1]

        # ── Holdout: apply last params on unseen data ──
        try:
            ho_result = backtest_strategy(
                strategy, holdout, last_params,
                commission=commission,
                slippage=slippage,
                initial_capital=10000.0,
                risk=risk,
                timeframe=timeframe,
            )
            ho_metrics = compute_all_metrics(ho_result.equity, timeframe, ho_result.trades_pnl)
        except Exception as e:
            logger.warning(f"  Seed {seed} holdout backtest failed: {e}")
            continue

        ho_sharpe = ho_metrics.get("sharpe", 0.0)
        ho_return = ho_metrics.get("total_return", 0.0)
        ho_dd = ho_metrics.get("max_drawdown", 0.0)

        holdout_sharpes.append(ho_sharpe)
        holdout_returns.append(ho_return)
        holdout_dds.append(ho_dd)
        holdout_metrics_list.append(ho_metrics)

        seed_details.append({
            "seed": seed,
            "is_sharpe": round(is_sharpe, 4),
            "ho_sharpe": round(ho_sharpe, 4),
            "ho_return": round(ho_return, 4),
            "ho_dd": round(ho_dd, 4),
            "ho_pf": round(ho_metrics.get("profit_factor", 0.0), 4),
            "ho_trades": ho_metrics.get("n_trades", 0),
            "last_params": {k: round(v, 6) if isinstance(v, float) else v for k, v in last_params.items()},
        })

        logger.info(f"    IS Sharpe={is_sharpe:.3f} → HO Sharpe={ho_sharpe:.3f}, "
                     f"Return={ho_return:.1%}, DD={ho_dd:.1%}, Trades={ho_metrics.get('n_trades', 0)}")

    if not holdout_sharpes:
        logger.error(f"  All seeds failed for {symbol}/{strategy_name}/{timeframe}")
        return None

    # Aggregate
    ho_sharpes_arr = np.array(holdout_sharpes)
    is_sharpes_arr = np.array(is_sharpes[:len(holdout_sharpes)])

    result = {
        "symbol": symbol,
        "strategy": strategy_name,
        "timeframe": timeframe,
        "cutoff": CUTOFF_DATE,
        "in_sample_bars": len(in_sample),
        "holdout_bars": len(holdout),
        "n_seeds": len(holdout_sharpes),
        # In-sample stats
        "is_sharpe_median": round(float(np.median(is_sharpes_arr)), 4),
        "is_sharpe_mean": round(float(np.mean(is_sharpes_arr)), 4),
        "is_sharpe_std": round(float(np.std(is_sharpes_arr)), 4),
        # Holdout stats
        "ho_sharpe_median": round(float(np.median(ho_sharpes_arr)), 4),
        "ho_sharpe_mean": round(float(np.mean(ho_sharpes_arr)), 4),
        "ho_sharpe_std": round(float(np.std(ho_sharpes_arr)), 4),
        "ho_sharpe_min": round(float(ho_sharpes_arr.min()), 4),
        "ho_sharpe_max": round(float(ho_sharpes_arr.max()), 4),
        "ho_return_median": round(float(np.median(holdout_returns)), 4),
        "ho_dd_median": round(float(np.median(holdout_dds)), 4),
        # Degradation
        "sharpe_degradation": round(float(np.median(is_sharpes_arr) - np.median(ho_sharpes_arr)), 4),
        "degradation_pct": round(
            float((np.median(is_sharpes_arr) - np.median(ho_sharpes_arr)) / max(abs(np.median(is_sharpes_arr)), 0.001) * 100), 1
        ),
        # Verdict
        "survives": bool(np.median(ho_sharpes_arr) > -0.1),
        "strong_survive": bool(np.median(ho_sharpes_arr) > 0.0 and ho_sharpes_arr.min() > -0.3),
        # Seed details
        "seeds": seed_details,
    }

    verdict = "✅ STRONG" if result["strong_survive"] else ("⚠️ WEAK" if result["survives"] else "❌ FAIL")
    logger.info(f"  {verdict} | IS median={result['is_sharpe_median']:.3f} → "
                f"HO median={result['ho_sharpe_median']:.3f} "
                f"(degradation: {result['degradation_pct']:.0f}%)")

    return result


def generate_markdown_report(results, elapsed, timestamp):
    """Generate docs/results/09_holdout_test.md"""
    md = []
    md.append("# Holdout Temporal Test — Validation des 5 MEDIUM V4")
    md.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    md.append(f"**Durée** : {elapsed/60:.1f} min")
    md.append(f"**Cutoff** : {CUTOFF_DATE} (12 mois de holdout)")
    md.append(f"**Seeds** : {N_SEEDS} par combo")
    md.append(f"**Statut** : ✅ TERMINÉ")
    md.append("")
    md.append("---")
    md.append("")

    # Summary
    survivors = [r for r in results if r["survives"]]
    strong = [r for r in results if r["strong_survive"]]
    fails = [r for r in results if not r["survives"]]

    md.append("## Résumé")
    md.append("")
    md.append(f"- **Combos testés** : {len(results)}")
    md.append(f"- **Survivants (strong)** : {len(strong)}")
    md.append(f"- **Survivants (weak)** : {len(survivors) - len(strong)}")
    md.append(f"- **Échecs** : {len(fails)}")
    md.append("")

    # Results table
    md.append("## Résultats détaillés")
    md.append("")
    md.append("| # | Verdict | Symbol | Stratégie | TF | IS Sharpe | HO Sharpe | HO Return | HO DD | Dégradation |")
    md.append("|---|---------|--------|-----------|-----|-----------|-----------|-----------|-------|-------------|")

    for i, r in enumerate(results):
        verdict = "✅ STRONG" if r["strong_survive"] else ("⚠️ WEAK" if r["survives"] else "❌ FAIL")
        md.append(
            f"| {i+1} | {verdict} | {r['symbol']} | {r['strategy']} | {r['timeframe']} | "
            f"{r['is_sharpe_median']:.3f} | {r['ho_sharpe_median']:.3f} | "
            f"{r['ho_return_median']:.1%} | {r['ho_dd_median']:.1%} | "
            f"{r['degradation_pct']:.0f}% |"
        )

    md.append("")

    # Variance analysis
    md.append("## Variance inter-seeds (holdout)")
    md.append("")
    md.append("| Combo | HO Sharpe min | HO Sharpe med | HO Sharpe max | HO Sharpe std |")
    md.append("|-------|---------------|---------------|---------------|---------------|")

    for r in results:
        md.append(
            f"| {r['symbol']}/{r['strategy']}/{r['timeframe']} | "
            f"{r['ho_sharpe_min']:.3f} | {r['ho_sharpe_median']:.3f} | "
            f"{r['ho_sharpe_max']:.3f} | {r['ho_sharpe_std']:.3f} |"
        )

    md.append("")

    # Methodology
    md.append("## Méthodologie")
    md.append("")
    md.append("### Principe")
    md.append("1. **Split temporel** : données avant/après le cutoff")
    md.append("2. **In-sample** : walk-forward complet (train/test rolling) sur données pré-cutoff")
    md.append("3. **Holdout** : appliquer les DERNIERS params optimisés sur données post-cutoff")
    md.append("4. **Multi-seed** : 5 seeds indépendants pour robustesse")
    md.append("")
    md.append("### Critères de survie")
    md.append("- **STRONG** : HO Sharpe médian > 0 ET HO Sharpe min > -0.3")
    md.append("- **WEAK** : HO Sharpe médian > -0.1")
    md.append("- **FAIL** : HO Sharpe médian ≤ -0.1")
    md.append("")

    # Conclusion
    md.append("## Conclusion")
    md.append("")
    if strong:
        md.append(f"**{len(strong)} combo(s) survivent fortement** le holdout :")
        for r in strong:
            md.append(f"- {r['symbol']}/{r['strategy']}/{r['timeframe']} (HO Sharpe {r['ho_sharpe_median']:.3f})")
        md.append("")
        md.append("→ Ces combos sont candidats pour le Portfolio V3.")
    elif survivors:
        md.append(f"**{len(survivors)} combo(s) survivent faiblement** (Sharpe > -0.1 mais pas positif).")
        md.append("→ Prudence pour le Portfolio V3, diversification nécessaire.")
    else:
        md.append("**Aucun combo ne survit le holdout.** Les résultats V4 étaient du sur-fitting.")
        md.append("→ Repenser les stratégies avant de construire un portfolio.")

    md.append("")
    md.append("---")
    md.append(f"*Généré le {datetime.now().strftime('%d %B %Y')}*")

    md_path = "docs/results/09_holdout_test.md"
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    logger.info(f"Saved: {md_path}")


def run_holdout_test():
    logger.info("=" * 70)
    logger.info("  QUANTLAB V7 — HOLDOUT TEMPORAL TEST")
    logger.info(f"  Cutoff: {CUTOFF_DATE} | Seeds: {N_SEEDS} | Combos: {len(COMBOS)}")
    logger.info("=" * 70)

    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    settings = load_settings()
    data_by_symbol = load_all_symbols_data(settings)
    risk = _build_risk(settings)
    cutoff_dt = pd.Timestamp(CUTOFF_DATE)

    results = []
    for i, combo in enumerate(COMBOS):
        logger.info(f"\n[{i+1}/{len(COMBOS)}] Testing {combo['symbol']}/{combo['strategy']}/{combo['timeframe']}")
        result = run_holdout_for_combo(combo, data_by_symbol, settings, risk, cutoff_dt)
        if result:
            results.append(result)

    elapsed = time.time() - t0

    # Sort by holdout Sharpe
    results.sort(key=lambda r: r["ho_sharpe_median"], reverse=True)

    # Summary
    survivors = [r for r in results if r["survives"]]
    strong = [r for r in results if r["strong_survive"]]

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  HOLDOUT TEST COMPLETE")
    logger.info(f"  Duration: {elapsed/60:.1f} min")
    logger.info(f"  Results: {len(results)} tested | {len(strong)} STRONG | "
                f"{len(survivors) - len(strong)} WEAK | {len(results) - len(survivors)} FAIL")
    logger.info(f"{'=' * 70}")

    for r in results:
        verdict = "✅ STRONG" if r["strong_survive"] else ("⚠️ WEAK" if r["survives"] else "❌ FAIL")
        logger.info(f"  {verdict} {r['symbol']}/{r['strategy']}/{r['timeframe']} | "
                     f"IS={r['is_sharpe_median']:.3f} → HO={r['ho_sharpe_median']:.3f} "
                     f"({r['degradation_pct']:+.0f}%)")

    # Save JSON
    output = {
        "version": "holdout_v1",
        "timestamp": timestamp,
        "config": {
            "cutoff_date": CUTOFF_DATE,
            "n_seeds": N_SEEDS,
            "base_seed": BASE_SEED,
            "combos": COMBOS,
            "tf_configs": TF_CONFIGS,
        },
        "summary": {
            "total": len(results),
            "strong_survivors": len(strong),
            "weak_survivors": len(survivors) - len(strong),
            "fails": len(results) - len(survivors),
            "elapsed_min": round(elapsed / 60, 1),
        },
        "results": results,
    }

    json_path = f"results/holdout_test_{timestamp}.json"
    Path("results").mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved: {json_path}")

    # Generate markdown report
    generate_markdown_report(results, elapsed, timestamp)

    return results, json_path


if __name__ == "__main__":
    results, filepath = run_holdout_test()
