#!/usr/bin/env python3
"""
Diagnostic V4 — 2-pass scan with deterministic seeds.

Pass 1 (FAST SCAN): 1 seed, 50 trials, pruning → filter viable combos (~1-2h)
Pass 2 (ROBUST):    5 seeds, 100 trials → only on viable combos from pass 1 (~30-60min)

Key improvements over V2:
  - 2-pass approach: 5-10x faster than brute force
  - Deterministic seeds in Optuna (TPESampler seeded per window)
  - Defaults fixes (validated by A/B test: 3M/1Y/1.0/sharpe/100)
  - All 16 strategies tested (bugs fixed: williams_r, stochastic, donchian, zscore, keltner)
  - Pruning in pass 1 for speed
  - Variance stats per combo in pass 2
  - Markdown report auto-generated

Output:
  - results/diagnostic_v4_{timestamp}.json
  - docs/results/08_diagnostic_v4.md
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.ingestion import load_all_symbols_data, load_settings
from engine.backtester import RiskConfig
from engine.metrics import composite_score
from engine.walk_forward import (
    WalkForwardConfig,
    run_walk_forward,
    run_walk_forward_robust,
)
from strategies.registry import get_strategy, list_strategies

# ── Config ──
# Pass 1: fast scan
P1_TRIALS = 50
P1_SEEDS = 1
P1_PRUNING = True
P1_MIN_SCORE = -0.05  # Threshold to pass to phase 2

# Pass 2: robust validation
P2_TRIALS = 100
P2_SEEDS = 5
P2_PRUNING = False

DIAG_METRIC = "sharpe"
COMPOSITE_WEIGHTS = {"sharpe": 0.35, "sortino": 0.25, "calmar": 0.20, "stability": 0.20}

# Defaults fixes (validated by A/B test — beat meta-optimization)
DEFAULTS = {
    "reoptim_frequency": "3M",
    "training_window": "1Y",
    "param_bounds_scale": 1.0,
    "optim_metric": "sharpe",
    "n_optim_trials": 100,
}

# Adaptive per-timeframe overrides
TF_OVERRIDES = {
    "15m": {"training_window": "3M", "reoptim_frequency": "2M"},
    "1h":  {"training_window": "3M", "reoptim_frequency": "2M"},
    "4h":  {"training_window": "6M", "reoptim_frequency": "3M"},
    "1d":  {"training_window": "1Y",  "reoptim_frequency": "3M"},
}

# Confidence tiers
TIER_HIGH = 0.3
TIER_MEDIUM = 0.0


def get_tf_config(timeframe):
    """Get walk-forward config for a given timeframe."""
    cfg = dict(DEFAULTS)
    if timeframe in TF_OVERRIDES:
        cfg.update(TF_OVERRIDES[timeframe])
    return cfg


def run_diagnostic_v4():
    logger.info("=" * 70)
    logger.info("  QUANTLAB V7 — DIAGNOSTIC V4 (2-PASS: FAST SCAN + ROBUST)")
    logger.info(f"  Pass 1: {P1_SEEDS} seed, {P1_TRIALS} trials, pruning={P1_PRUNING}")
    logger.info(f"  Pass 2: {P2_SEEDS} seeds, {P2_TRIALS} trials (viable combos only)")
    logger.info("=" * 70)

    settings = load_settings()
    data_by_symbol = load_all_symbols_data(settings)

    symbols = [s for s, d in data_by_symbol.items() if d]
    logger.info(f"Symbols: {symbols}")

    strategies = list_strategies()
    timeframes = settings["data"].get("meta_timeframes", ["15m", "1h", "4h", "1d"])
    logger.info(f"Strategies: {len(strategies)} | Timeframes: {timeframes}")

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

    total_combos = len(symbols) * len(strategies) * len(timeframes)
    logger.info(f"Total combos: {total_combos}")

    start_time = time.time()

    # ================================================================
    # PASS 1: FAST SCAN (1 seed, 50 trials, pruning)
    # ================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  PASS 1: FAST SCAN ({P1_SEEDS} seed, {P1_TRIALS} trials, pruning={P1_PRUNING})")
    logger.info(f"{'=' * 70}")

    pass1_results = []
    pass1_failed = 0
    pass1_skipped = 0

    with tqdm(total=total_combos, desc="🔍 Pass 1 (fast)", unit="combo") as pbar:
        for symbol in symbols:
            symbol_data = data_by_symbol[symbol]

            for strategy_name in strategies:
                try:
                    strategy = get_strategy(strategy_name)
                except ValueError:
                    pbar.update(len(timeframes))
                    pass1_skipped += len(timeframes)
                    continue

                for timeframe in timeframes:
                    if timeframe not in symbol_data:
                        pbar.update(1)
                        pass1_skipped += 1
                        continue

                    data = symbol_data[timeframe]
                    if len(data) < 500:
                        pbar.update(1)
                        pass1_skipped += 1
                        continue

                    tf_cfg = get_tf_config(timeframe)

                    config = WalkForwardConfig(
                        strategy=strategy,
                        data=data,
                        timeframe=timeframe,
                        reoptim_frequency=tf_cfg["reoptim_frequency"],
                        training_window=tf_cfg["training_window"],
                        param_bounds_scale=tf_cfg["param_bounds_scale"],
                        optim_metric=tf_cfg["optim_metric"],
                        n_optim_trials=P1_TRIALS,
                        commission=settings["engine"]["commission_rate"],
                        slippage=settings["engine"]["slippage_rate"],
                        risk=risk,
                        seed=42,
                        use_pruning=P1_PRUNING,
                    )

                    try:
                        wf_result = run_walk_forward(config)
                    except Exception as e:
                        logger.warning(f"P1 Failed {symbol}/{strategy_name}/{timeframe}: {e}")
                        pbar.update(1)
                        pass1_failed += 1
                        continue

                    m = wf_result.metrics

                    if wf_result.n_oos_periods < 2 or m.get("n_trades", 0) < 10:
                        pbar.update(1)
                        pass1_skipped += 1
                        continue

                    score = composite_score(m, COMPOSITE_WEIGHTS)

                    pass1_results.append({
                        "symbol": symbol,
                        "strategy": strategy_name,
                        "timeframe": timeframe,
                        "p1_score": round(score, 4),
                        "p1_sharpe": round(m.get("sharpe", 0), 4),
                        "p1_return": round(m.get("total_return", 0), 4),
                        "p1_dd": round(m.get("max_drawdown", 0), 4),
                        "p1_trades": m.get("n_trades", 0),
                        "p1_pf": round(m.get("profit_factor", 0), 4),
                    })

                    pbar.set_postfix({
                        "viable": len(pass1_results),
                        "score": f"{score:.2f}",
                    })
                    pbar.update(1)

    pass1_elapsed = time.time() - start_time
    pass1_results.sort(key=lambda x: x["p1_score"], reverse=True)

    # Filter for pass 2
    viable = [r for r in pass1_results if r["p1_score"] >= P1_MIN_SCORE]

    logger.info(f"\nPass 1 done in {pass1_elapsed/60:.1f} min")
    logger.info(f"  Total scanned: {len(pass1_results)}/{total_combos}")
    logger.info(f"  Viable (score >= {P1_MIN_SCORE}): {len(viable)}")
    logger.info(f"  Eliminated: {len(pass1_results) - len(viable)}")
    logger.info(f"  Failed: {pass1_failed} | Skipped: {pass1_skipped}")

    # Show pass 1 top 10
    logger.info("\n--- PASS 1 TOP 10 ---")
    for i, r in enumerate(pass1_results[:10]):
        logger.info(f"  #{i+1:2d} {r['symbol']:8s} | {r['strategy']:25s} | {r['timeframe']:4s} | "
                    f"Score: {r['p1_score']:+.3f} | Sharpe: {r['p1_sharpe']:.3f} | "
                    f"Return: {r['p1_return']:.1%} | DD: {r['p1_dd']:.1%}")

    # ================================================================
    # PASS 2: ROBUST VALIDATION (5 seeds, 100 trials, viable only)
    # ================================================================
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  PASS 2: ROBUST VALIDATION ({P2_SEEDS} seeds, {P2_TRIALS} trials)")
    logger.info(f"  Testing {len(viable)} viable combos")
    logger.info(f"{'=' * 70}")

    pass2_start = time.time()
    results = []
    pass2_failed = 0

    with tqdm(total=len(viable), desc="💪 Pass 2 (robust)", unit="combo") as pbar:
        for p1 in viable:
            symbol = p1["symbol"]
            strategy_name = p1["strategy"]
            timeframe = p1["timeframe"]

            strategy = get_strategy(strategy_name)
            data = data_by_symbol[symbol][timeframe]
            tf_cfg = get_tf_config(timeframe)

            config = WalkForwardConfig(
                strategy=strategy,
                data=data,
                timeframe=timeframe,
                reoptim_frequency=tf_cfg["reoptim_frequency"],
                training_window=tf_cfg["training_window"],
                param_bounds_scale=tf_cfg["param_bounds_scale"],
                optim_metric=tf_cfg["optim_metric"],
                n_optim_trials=P2_TRIALS,
                commission=settings["engine"]["commission_rate"],
                slippage=settings["engine"]["slippage_rate"],
                risk=risk,
                seed=42,
                use_pruning=P2_PRUNING,
            )

            try:
                wf_result = run_walk_forward_robust(
                    config, n_seeds=P2_SEEDS, aggregation="median"
                )
            except Exception as e:
                logger.warning(f"P2 Failed {symbol}/{strategy_name}/{timeframe}: {e}")
                pbar.update(1)
                pass2_failed += 1
                continue

            m = wf_result.metrics
            score = composite_score(m, COMPOSITE_WEIGHTS)

            # Confidence tier based on robust stats
            robust_std = m.get("robust_sharpe_std", 999)
            robust_min = m.get("robust_sharpe_min", -999)
            avg_sharpe = m.get("robust_sharpe_median", m.get("sharpe", 0))

            if score > TIER_HIGH and robust_min > 0.0 and robust_std < 0.3:
                confidence = "HIGH"
            elif score > TIER_HIGH and avg_sharpe > 0.2:
                confidence = "HIGH"
            elif score > TIER_MEDIUM:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"

            result = {
                "symbol": symbol,
                "strategy": strategy_name,
                "timeframe": timeframe,
                "score": round(score, 4),
                "confidence": confidence,
                "sharpe": round(m.get("sharpe", 0), 4),
                "sortino": round(m.get("sortino", 0), 4),
                "calmar": round(m.get("calmar", 0), 4),
                "total_return": round(m.get("total_return", 0), 4),
                "max_drawdown": round(m.get("max_drawdown", 0), 4),
                "win_rate": round(m.get("win_rate", 0), 4),
                "profit_factor": round(m.get("profit_factor", 0), 4),
                "n_trades": m.get("n_trades", 0),
                "n_oos_periods": wf_result.n_oos_periods,
                "robust_sharpe_median": round(m.get("robust_sharpe_median", 0), 4),
                "robust_sharpe_mean": round(m.get("robust_sharpe_mean", 0), 4),
                "robust_sharpe_std": round(m.get("robust_sharpe_std", 0), 4),
                "robust_sharpe_min": round(m.get("robust_sharpe_min", 0), 4),
                "robust_sharpe_max": round(m.get("robust_sharpe_max", 0), 4),
                "robust_consistency": round(m.get("robust_consistency", 0), 4),
                "p1_score": p1["p1_score"],
                "wf_params": tf_cfg,
            }
            results.append(result)

            pbar.set_postfix({
                "viable": len(results),
                "score": f"{score:.2f}",
                "conf": confidence,
            })
            pbar.update(1)

    elapsed = time.time() - start_time
    pass2_elapsed = time.time() - pass2_start

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    # ── Save JSON ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"results/diagnostic_v4_{timestamp}.json"
    Path("results").mkdir(exist_ok=True)

    output = {
        "version": "v4",
        "timestamp": timestamp,
        "config": {
            "pass1": {"seeds": P1_SEEDS, "trials": P1_TRIALS, "pruning": P1_PRUNING, "min_score": P1_MIN_SCORE},
            "pass2": {"seeds": P2_SEEDS, "trials": P2_TRIALS, "pruning": P2_PRUNING},
            "defaults": DEFAULTS,
            "tf_overrides": TF_OVERRIDES,
            "composite_weights": COMPOSITE_WEIGHTS,
        },
        "summary": {
            "total_combos": total_combos,
            "pass1_viable": len(pass1_results),
            "pass2_candidates": len(viable),
            "pass2_validated": len(results),
            "pass2_failed": pass2_failed,
            "pass1_elapsed_min": round(pass1_elapsed / 60, 1),
            "pass2_elapsed_min": round(pass2_elapsed / 60, 1),
            "total_elapsed_min": round(elapsed / 60, 1),
            "high_confidence": len([r for r in results if r["confidence"] == "HIGH"]),
            "medium_confidence": len([r for r in results if r["confidence"] == "MEDIUM"]),
            "low_confidence": len([r for r in results if r["confidence"] == "LOW"]),
        },
        "pass1_all": pass1_results,
        "results": results,
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved: {json_path}")

    # ── Console summary ──
    logger.info("\n" + "=" * 90)
    logger.info("  DIAGNOSTIC V4 RESULTS (2-PASS)")
    logger.info("=" * 90)
    logger.info(f"Pass 1: {len(pass1_results)} viable / {total_combos} total ({pass1_elapsed/60:.1f} min)")
    logger.info(f"Pass 2: {len(results)} validated / {len(viable)} candidates ({pass2_elapsed/60:.1f} min)")

    high = [r for r in results if r["confidence"] == "HIGH"]
    medium = [r for r in results if r["confidence"] == "MEDIUM"]
    low = [r for r in results if r["confidence"] == "LOW"]
    logger.info(f"Confidence: {len(high)} HIGH | {len(medium)} MEDIUM | {len(low)} LOW")

    # Top 20
    logger.info("\n--- TOP 20 COMBOS ---")
    for i, r in enumerate(results[:20]):
        logger.info(
            f"  #{i+1:2d} [{r['confidence']:6s}] {r['symbol']:8s} | {r['strategy']:25s} | {r['timeframe']:4s} | "
            f"Score: {r['score']:+.3f} | "
            f"Sharpe: {r['sharpe']:.3f} (med={r['robust_sharpe_median']:.3f} std={r['robust_sharpe_std']:.3f}) | "
            f"Return: {r['total_return']:.1%} | DD: {r['max_drawdown']:.1%} | PF: {r['profit_factor']:.2f}"
        )

    # Strategy ranking
    logger.info("\n--- STRATEGY RANKING ---")
    strategy_stats = {}
    for r in results:
        s = r["strategy"]
        if s not in strategy_stats:
            strategy_stats[s] = {"count": 0, "high": 0, "scores": [], "sharpes": []}
        strategy_stats[s]["count"] += 1
        strategy_stats[s]["high"] += 1 if r["confidence"] == "HIGH" else 0
        strategy_stats[s]["scores"].append(r["score"])
        strategy_stats[s]["sharpes"].append(r["sharpe"])

    for strat, stats in sorted(strategy_stats.items(),
                                key=lambda x: np.mean(x[1]["scores"]), reverse=True):
        avg_score = np.mean(stats["scores"])
        avg_sharpe = np.mean(stats["sharpes"])
        logger.info(f"  {strat:25s}: {stats['count']:2d} viable ({stats['high']:2d} HIGH) | "
                    f"Avg Score: {avg_score:.3f} | Avg Sharpe: {avg_sharpe:.3f}")

    # Symbol ranking
    logger.info("\n--- SYMBOL RANKING ---")
    symbol_stats = {}
    for r in results:
        s = r["symbol"]
        if s not in symbol_stats:
            symbol_stats[s] = {"count": 0, "high": 0, "scores": []}
        symbol_stats[s]["count"] += 1
        symbol_stats[s]["high"] += 1 if r["confidence"] == "HIGH" else 0
        symbol_stats[s]["scores"].append(r["score"])

    for sym, stats in sorted(symbol_stats.items(),
                              key=lambda x: np.mean(x[1]["scores"]), reverse=True):
        avg = np.mean(stats["scores"])
        logger.info(f"  {sym:8s}: {stats['count']:2d} viable ({stats['high']:2d} HIGH) | Avg Score: {avg:.3f}")

    # Heatmap
    if strategy_stats and symbol_stats:
        logger.info("\n--- HEATMAP: STRATEGY × SYMBOL (best score, TF) ---")
        all_strats = sorted(strategy_stats.keys())
        all_syms = sorted(symbol_stats.keys())
        header = f"  {'Strategy':25s}" + "".join(f" {s:>10s}" for s in all_syms)
        logger.info(header)
        logger.info("  " + "-" * (25 + 11 * len(all_syms)))

        for strat in all_strats:
            row = f"  {strat:25s}"
            for sym in all_syms:
                matches = [r for r in results if r["strategy"] == strat and r["symbol"] == sym]
                if matches:
                    best = max(matches, key=lambda x: x["score"])
                    cell = f"{best['score']:+.2f}/{best['timeframe']}"
                else:
                    cell = "---"
                row += f" {cell:>10s}"
            logger.info(row)

    logger.info(f"\n⏱ Total: {elapsed/60:.1f} min (P1: {pass1_elapsed/60:.1f} + P2: {pass2_elapsed/60:.1f})")
    logger.info("=" * 90)

    # ── Generate Markdown report ──
    generate_markdown_report(output, elapsed, timestamp)

    return results, json_path


def generate_markdown_report(output, elapsed, timestamp):
    """Generate docs/results/08_diagnostic_v4.md"""
    results = output["results"]
    summary = output["summary"]
    cfg = output["config"]

    md = []
    md.append("# Diagnostic V4 — 2-Pass Scan (Fast + Robust)")
    md.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    md.append(f"**Durée totale** : {elapsed/60:.1f} min (P1: {summary['pass1_elapsed_min']} + P2: {summary['pass2_elapsed_min']})")
    md.append(f"**Pass 1** : {cfg['pass1']['seeds']} seed, {cfg['pass1']['trials']} trials, pruning={cfg['pass1']['pruning']}")
    md.append(f"**Pass 2** : {cfg['pass2']['seeds']} seeds, {cfg['pass2']['trials']} trials (médiane)")
    md.append(f"**Defaults** : reoptim=3M, window=1Y, bounds=1.0, sharpe (validés par A/B test)")
    md.append(f"**Bugs fixés** : williams_r, stochastic, donchian, zscore, keltner")
    md.append(f"**Statut** : ✅ TERMINÉ")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## Résumé")
    md.append("")
    md.append(f"- **Combos total** : {summary['total_combos']}")
    md.append(f"- **Pass 1 viable** : {summary['pass1_viable']}")
    md.append(f"- **Pass 2 candidats** : {summary['pass2_candidates']}")
    md.append(f"- **Pass 2 validés** : {summary['pass2_validated']}")
    md.append(f"- **HIGH confidence** : {summary['high_confidence']}")
    md.append(f"- **MEDIUM confidence** : {summary['medium_confidence']}")
    md.append(f"- **LOW confidence** : {summary['low_confidence']}")
    md.append("")

    md.append("## Top 20 combos")
    md.append("")
    md.append("| # | Conf | Symbol | Stratégie | TF | Score | Sharpe | Sharpe med | Sharpe std | Return | DD | PF |")
    md.append("|---|------|--------|-----------|-----|-------|--------|------------|------------|--------|-----|-----|")

    for i, r in enumerate(results[:20]):
        md.append(
            f"| {i+1} | {r['confidence']} | {r['symbol']} | {r['strategy']} | {r['timeframe']} | "
            f"{r['score']:.3f} | {r['sharpe']:.3f} | {r['robust_sharpe_median']:.3f} | "
            f"{r['robust_sharpe_std']:.3f} | {r['total_return']:.1%} | "
            f"{r['max_drawdown']:.1%} | {r['profit_factor']:.2f} |"
        )

    md.append("")
    md.append("## Ranking par stratégie")
    md.append("")
    md.append("| Stratégie | Viable | HIGH | Avg Score | Avg Sharpe |")
    md.append("|-----------|--------|------|-----------|------------|")

    strategy_stats = {}
    for r in results:
        s = r["strategy"]
        if s not in strategy_stats:
            strategy_stats[s] = {"count": 0, "high": 0, "scores": [], "sharpes": []}
        strategy_stats[s]["count"] += 1
        strategy_stats[s]["high"] += 1 if r["confidence"] == "HIGH" else 0
        strategy_stats[s]["scores"].append(r["score"])
        strategy_stats[s]["sharpes"].append(r["sharpe"])

    for strat, stats in sorted(strategy_stats.items(),
                                key=lambda x: np.mean(x[1]["scores"]), reverse=True):
        md.append(f"| {strat} | {stats['count']} | {stats['high']} | "
                  f"{np.mean(stats['scores']):.3f} | {np.mean(stats['sharpes']):.3f} |")

    md.append("")
    md.append("## Ranking par symbol")
    md.append("")
    md.append("| Symbol | Viable | HIGH | Avg Score |")
    md.append("|--------|--------|------|-----------|")

    symbol_stats = {}
    for r in results:
        s = r["symbol"]
        if s not in symbol_stats:
            symbol_stats[s] = {"count": 0, "high": 0, "scores": []}
        symbol_stats[s]["count"] += 1
        symbol_stats[s]["high"] += 1 if r["confidence"] == "HIGH" else 0
        symbol_stats[s]["scores"].append(r["score"])

    for sym, stats in sorted(symbol_stats.items(),
                              key=lambda x: np.mean(x[1]["scores"]), reverse=True):
        md.append(f"| {sym} | {stats['count']} | {stats['high']} | {np.mean(stats['scores']):.3f} |")

    md.append("")
    md.append("## Analyse de la variance")
    md.append("")
    md.append("Distribution de `robust_sharpe_std` (variance inter-seeds) :")
    md.append("")

    stds = [r["robust_sharpe_std"] for r in results]
    if stds:
        md.append(f"- **Min** : {min(stds):.4f}")
        md.append(f"- **Médiane** : {np.median(stds):.4f}")
        md.append(f"- **Max** : {max(stds):.4f}")
        md.append(f"- **Combos avec std < 0.1** : {sum(1 for s in stds if s < 0.1)}/{len(stds)}")
        md.append(f"- **Combos avec std > 0.3** : {sum(1 for s in stds if s > 0.3)}/{len(stds)}")

    md.append("")
    md.append("## Méthodologie")
    md.append("")
    md.append("### Pipeline 2-pass")
    md.append("1. **Pass 1 (fast scan)** : 1 seed, 50 trials, pruning → filtre les combos nuls")
    md.append("2. **Pass 2 (robust)** : 5 seeds, 100 trials → validation multi-seed sur combos viables")
    md.append("")
    md.append("### Paramètres")
    md.append("- **Walk-forward** : defaults fixes (validés par test A/B vs méta-optimisation)")
    md.append("- **Déterminisme** : TPESampler seeded par fenêtre")
    md.append("- **Pruning** : MedianPruner (pass 1 seulement)")
    md.append("- **Bugs fixés avant scan** : williams_r (signal mort), stochastic (crash), donchian/zscore (off-by-one), keltner (ATR)")
    md.append("")
    md.append("### Confidence tiers")
    md.append("- **HIGH** : score > 0.3 ET (min_sharpe > 0 ET std < 0.3) OU (score > 0.3 ET median > 0.2)")
    md.append("- **MEDIUM** : score > 0.0")
    md.append("- **LOW** : score ≤ 0.0")
    md.append("")
    md.append("---")
    md.append(f"*Généré le {datetime.now().strftime('%d %B %Y')}*")

    md_path = "docs/results/08_diagnostic_v4.md"
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    logger.info(f"Saved: {md_path}")


if __name__ == "__main__":
    results, filepath = run_diagnostic_v4()
