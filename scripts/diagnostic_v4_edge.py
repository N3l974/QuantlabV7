"""
Diagnostic V4 Edge — Walk-Forward + Overlays

Full scan of 22 strategies × 3 symbols × 2 TFs with:
- Walk-forward multi-seed (3 seeds, 50 trials for speed)
- Overlay variants: baseline vs regime+vol_targeting
- Holdout validation (cutoff 2025-02-01)
- Comprehensive report generation

This is the proper validation of the edge improvements.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.registry import get_strategy, list_strategies
from engine.walk_forward import (
    WalkForwardConfig, run_walk_forward, run_walk_forward_robust
)
from engine.backtester import backtest_strategy, RiskConfig, vectorized_backtest
from engine.metrics import compute_all_metrics, composite_score
from engine.overlays import (
    apply_overlay_pipeline, OverlayPipelineConfig,
    VolTargetConfig, RegimeOverlayConfig
)
from engine.regime import RegimeConfig


# ─── Configuration ───────────────────────────────────────

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["4h", "1d"]
CUTOFF = "2025-02-01"

# Walk-forward defaults (validated by A/B test)
WF_REOPTIM = "3M"
WF_WINDOW = "1Y"
WF_BOUNDS_SCALE = 1.0
WF_METRIC = "sharpe"
WF_TRIALS = 50          # Reduced for speed (was 100)
N_SEEDS = 3             # Reduced for speed (was 5)

# Overlay config
REGIME_CONFIG = RegimeConfig()
REGIME_OVERLAY = RegimeOverlayConfig(
    regime_config=REGIME_CONFIG,
    hard_cutoff=True,
    min_exposure_threshold=0.3,
)
VOL_CONFIG = VolTargetConfig(target_vol_annual=0.30)
OVERLAY_PIPELINE = OverlayPipelineConfig(
    regime_config=REGIME_OVERLAY,
    vol_config=VOL_CONFIG,
)

# Risk config
RISK = RiskConfig()

# Filtering thresholds
MIN_OOS_PERIODS = 3
MIN_TRADES = 5


# ─── Data Loading ────────────────────────────────────────

def load_data():
    """Load all symbol/timeframe data."""
    data = {}
    for sym in SYMBOLS:
        data[sym] = {}
        for tf in TIMEFRAMES:
            path = Path(f"data/raw/{sym}_{tf}.parquet")
            if path.exists():
                df = pd.read_parquet(path)
                data[sym][tf] = df
                logger.info(f"Loaded {sym}/{tf}: {len(df)} bars "
                           f"({df.index[0]} → {df.index[-1]})")
            else:
                logger.warning(f"Missing: {path}")
    return data


# ─── Walk-Forward with Overlay ───────────────────────────

def run_combo(strategy_name, symbol, timeframe, data_full, data_is,
              use_overlay=False):
    """
    Run walk-forward on IS data, then test on holdout.
    Returns dict with all metrics.
    """
    strategy = get_strategy(strategy_name)

    # Walk-forward on IS data
    wf_config = WalkForwardConfig(
        strategy=strategy,
        data=data_is,
        timeframe=timeframe,
        reoptim_frequency=WF_REOPTIM,
        training_window=WF_WINDOW,
        param_bounds_scale=WF_BOUNDS_SCALE,
        optim_metric=WF_METRIC,
        n_optim_trials=WF_TRIALS,
        risk=RISK,
        seed=42,
        use_pruning=True,
    )

    wf_result = run_walk_forward_robust(wf_config, n_seeds=N_SEEDS)

    if wf_result.n_oos_periods < MIN_OOS_PERIODS:
        return None

    # Get last optimized params for holdout
    if not wf_result.best_params_per_period:
        return None
    last_params = wf_result.best_params_per_period[-1]

    # Holdout backtest
    ho_data = data_full[data_full.index >= CUTOFF].copy()
    if len(ho_data) < 50:
        return None

    # Generate signals
    ho_signals = strategy.generate_signals(ho_data, last_params)

    # Apply overlay if requested
    if use_overlay:
        ho_signals, overlay_info = apply_overlay_pipeline(
            ho_signals, ho_data, OVERLAY_PIPELINE, timeframe=timeframe
        )

    close = ho_data["close"].values.astype(np.float64)
    high = ho_data["high"].values.astype(np.float64)
    low = ho_data["low"].values.astype(np.float64)

    ho_result = vectorized_backtest(
        close, ho_signals, risk=RISK, high=high, low=low, timeframe=timeframe
    )
    ho_metrics = compute_all_metrics(ho_result.equity, timeframe, ho_result.trades_pnl)

    return {
        "symbol": symbol,
        "strategy": strategy_name,
        "timeframe": timeframe,
        "overlay": use_overlay,
        # IS walk-forward metrics
        "is_sharpe": wf_result.metrics.get("sharpe", 0),
        "is_return": wf_result.metrics.get("total_return", 0),
        "is_dd": wf_result.metrics.get("max_drawdown", 0),
        "is_trades": wf_result.metrics.get("n_trades", 0),
        "is_composite": wf_result.composite,
        "n_oos_periods": wf_result.n_oos_periods,
        # Robust stats
        "robust_sharpe_med": wf_result.metrics.get("robust_sharpe_median", 0),
        "robust_sharpe_std": wf_result.metrics.get("robust_sharpe_std", 0),
        # Holdout metrics
        "ho_sharpe": ho_metrics.get("sharpe", 0),
        "ho_return": ho_metrics.get("total_return", 0),
        "ho_dd": ho_metrics.get("max_drawdown", 0),
        "ho_trades": ho_metrics.get("n_trades", 0),
        "ho_calmar": ho_metrics.get("calmar", 0),
        "ho_sortino": ho_metrics.get("sortino", 0),
        "ho_win_rate": ho_metrics.get("win_rate", 0),
        "ho_pf": ho_metrics.get("profit_factor", 0),
        # Params
        "last_params": last_params,
        # Holdout equity for portfolio construction
        "ho_equity": ho_result.equity.tolist(),
        "ho_returns": ho_result.returns.tolist(),
    }


# ─── Verdict Classification ─────────────────────────────

def classify_combo(r):
    """Classify combo into STRONG/WEAK/FAIL."""
    ho_sharpe = r["ho_sharpe"]
    ho_trades = r["ho_trades"]
    robust_std = r["robust_sharpe_std"]

    if ho_trades < MIN_TRADES:
        return "FAIL"
    if ho_sharpe > 0.3 and robust_std < 0.5:
        return "STRONG"
    if ho_sharpe > 0.0:
        return "WEAK"
    return "FAIL"


# ─── Main ────────────────────────────────────────────────

def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC V4 EDGE — Walk-Forward + Overlays")
    logger.info("=" * 60)

    all_data = load_data()
    strategies = list_strategies()

    logger.info(f"Strategies: {len(strategies)}")
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"Timeframes: {TIMEFRAMES}")
    logger.info(f"Cutoff: {CUTOFF}")
    logger.info(f"WF: {WF_REOPTIM} reoptim, {WF_WINDOW} window, {N_SEEDS} seeds, {WF_TRIALS} trials")

    total_combos = len(strategies) * len(SYMBOLS) * len(TIMEFRAMES) * 2  # ×2 for overlay
    logger.info(f"Total combos to test: {total_combos}")

    results = []
    combo_count = 0

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            if sym not in all_data or tf not in all_data[sym]:
                continue

            data_full = all_data[sym][tf]
            data_is = data_full[data_full.index < CUTOFF].copy()

            if len(data_is) < 500:
                logger.warning(f"Skipping {sym}/{tf}: only {len(data_is)} IS bars")
                continue

            for sname in strategies:
                for use_overlay in [False, True]:
                    combo_count += 1
                    overlay_tag = "+overlay" if use_overlay else "baseline"
                    logger.info(
                        f"[{combo_count}/{total_combos}] "
                        f"{sym}/{sname}/{tf} ({overlay_tag})"
                    )

                    try:
                        r = run_combo(
                            sname, sym, tf, data_full, data_is,
                            use_overlay=use_overlay
                        )
                        if r is not None:
                            r["verdict"] = classify_combo(r)
                            results.append(r)
                            logger.info(
                                f"  → {r['verdict']} | "
                                f"IS Sharpe={r['is_sharpe']:.3f} | "
                                f"HO Sharpe={r['ho_sharpe']:.3f} | "
                                f"HO Return={r['ho_return']*100:.1f}% | "
                                f"HO DD={r['ho_dd']*100:.1f}%"
                            )
                        else:
                            logger.info("  → SKIP (insufficient data/periods)")
                    except Exception as e:
                        logger.error(f"  → ERROR: {e}")

    elapsed = (time.time() - start_time) / 60
    logger.info(f"\nDiagnostic complete in {elapsed:.1f} min")
    logger.info(f"Results: {len(results)} combos")

    # ─── Save results ────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"results/diagnostic_v4_edge_{timestamp}.json")
    results_path.parent.mkdir(exist_ok=True)

    # Strip large arrays for JSON
    results_slim = []
    for r in results:
        r_slim = {k: v for k, v in r.items() if k not in ("ho_equity", "ho_returns", "last_params")}
        r_slim["last_params"] = r.get("last_params", {})
        results_slim.append(r_slim)

    with open(results_path, "w") as f:
        json.dump(results_slim, f, indent=2, default=str)
    logger.info(f"Saved: {results_path}")

    # ─── Generate report ─────────────────────────────────
    generate_report(results, elapsed, timestamp)

    return results


def generate_report(results, elapsed_min, timestamp):
    """Generate markdown report."""

    # Split baseline vs overlay
    baseline = [r for r in results if not r["overlay"]]
    overlay = [r for r in results if r["overlay"]]

    # Classify
    base_strong = [r for r in baseline if r["verdict"] == "STRONG"]
    base_weak = [r for r in baseline if r["verdict"] == "WEAK"]
    base_fail = [r for r in baseline if r["verdict"] == "FAIL"]
    ov_strong = [r for r in overlay if r["verdict"] == "STRONG"]
    ov_weak = [r for r in overlay if r["verdict"] == "WEAK"]
    ov_fail = [r for r in overlay if r["verdict"] == "FAIL"]

    lines = []
    lines.append("# Diagnostic V4 Edge — Walk-Forward + Overlays")
    lines.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    lines.append(f"**Durée** : {elapsed_min:.1f} min")
    lines.append(f"**Config** : {N_SEEDS} seeds, {WF_TRIALS} trials, reoptim={WF_REOPTIM}, window={WF_WINDOW}")
    lines.append(f"**Overlays** : regime (hard cutoff, min 0.3) + vol targeting (30% annual)")
    lines.append(f"**Cutoff holdout** : {CUTOFF}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    lines.append("## Résumé")
    lines.append("")
    lines.append(f"- **Combos testés** : {len(baseline)} baseline + {len(overlay)} overlay")
    lines.append(f"- **Baseline** : {len(base_strong)} STRONG, {len(base_weak)} WEAK, {len(base_fail)} FAIL")
    lines.append(f"- **+ Overlay** : {len(ov_strong)} STRONG, {len(ov_weak)} WEAK, {len(ov_fail)} FAIL")
    lines.append("")

    # Avg Sharpe comparison
    if baseline:
        avg_base = np.mean([r["ho_sharpe"] for r in baseline])
        avg_ov = np.mean([r["ho_sharpe"] for r in overlay]) if overlay else 0
        lines.append(f"- **Avg HO Sharpe baseline** : {avg_base:.3f}")
        lines.append(f"- **Avg HO Sharpe +overlay** : {avg_ov:.3f}")

        # Count improvements
        improved = 0
        for b in baseline:
            key = (b["symbol"], b["strategy"], b["timeframe"])
            ov_match = [o for o in overlay
                        if (o["symbol"], o["strategy"], o["timeframe"]) == key]
            if ov_match and ov_match[0]["ho_sharpe"] > b["ho_sharpe"]:
                improved += 1
        lines.append(f"- **Combos améliorés par overlay** : {improved}/{len(baseline)} ({improved/max(len(baseline),1)*100:.0f}%)")
    lines.append("")

    # Top combos table — baseline
    lines.append("## Top 20 combos — Baseline (sans overlay)")
    lines.append("")
    sorted_base = sorted(baseline, key=lambda x: x["ho_sharpe"], reverse=True)
    lines.append("| # | Verdict | Symbol | Stratégie | TF | IS Sharpe | HO Sharpe | HO Return | HO DD | HO Trades | Robust std |")
    lines.append("|---|---------|--------|-----------|-----|-----------|-----------|-----------|-------|-----------|------------|")
    for i, r in enumerate(sorted_base[:20]):
        lines.append(
            f"| {i+1} | {'✅' if r['verdict']=='STRONG' else '⚠️' if r['verdict']=='WEAK' else '❌'} {r['verdict']} "
            f"| {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {r['is_sharpe']:.3f} | {r['ho_sharpe']:.3f} "
            f"| {r['ho_return']*100:.1f}% | {r['ho_dd']*100:.1f}% "
            f"| {r['ho_trades']} | {r['robust_sharpe_std']:.3f} |"
        )
    lines.append("")

    # Top combos table — overlay
    lines.append("## Top 20 combos — Avec Overlays (regime + vol targeting)")
    lines.append("")
    sorted_ov = sorted(overlay, key=lambda x: x["ho_sharpe"], reverse=True)
    lines.append("| # | Verdict | Symbol | Stratégie | TF | IS Sharpe | HO Sharpe | HO Return | HO DD | HO Trades | Robust std |")
    lines.append("|---|---------|--------|-----------|-----|-----------|-----------|-----------|-------|-----------|------------|")
    for i, r in enumerate(sorted_ov[:20]):
        lines.append(
            f"| {i+1} | {'✅' if r['verdict']=='STRONG' else '⚠️' if r['verdict']=='WEAK' else '❌'} {r['verdict']} "
            f"| {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {r['is_sharpe']:.3f} | {r['ho_sharpe']:.3f} "
            f"| {r['ho_return']*100:.1f}% | {r['ho_dd']*100:.1f}% "
            f"| {r['ho_trades']} | {r['robust_sharpe_std']:.3f} |"
        )
    lines.append("")

    # Comparison: baseline vs overlay for each combo
    lines.append("## Comparaison directe : Baseline vs Overlay")
    lines.append("")
    lines.append("| Symbol | Stratégie | TF | Base Sharpe | +Ov Sharpe | Base DD | +Ov DD | Δ Sharpe |")
    lines.append("|--------|-----------|-----|-------------|------------|---------|--------|----------|")
    pairs = []
    for b in sorted_base:
        key = (b["symbol"], b["strategy"], b["timeframe"])
        ov_match = [o for o in overlay
                    if (o["symbol"], o["strategy"], o["timeframe"]) == key]
        if ov_match:
            o = ov_match[0]
            delta = o["ho_sharpe"] - b["ho_sharpe"]
            pairs.append((b, o, delta))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for b, o, delta in pairs[:25]:
        sign = "+" if delta > 0 else ""
        lines.append(
            f"| {b['symbol']} | {b['strategy']} | {b['timeframe']} "
            f"| {b['ho_sharpe']:.3f} | {o['ho_sharpe']:.3f} "
            f"| {b['ho_dd']*100:.1f}% | {o['ho_dd']*100:.1f}% "
            f"| {sign}{delta:.3f} |"
        )
    lines.append("")

    # Strategy ranking
    lines.append("## Ranking par stratégie (HO Sharpe moyen, baseline)")
    lines.append("")
    strat_scores = {}
    for r in baseline:
        s = r["strategy"]
        if s not in strat_scores:
            strat_scores[s] = []
        strat_scores[s].append(r["ho_sharpe"])
    strat_ranking = sorted(strat_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)
    lines.append("| Stratégie | Type | Avg HO Sharpe | N combos | Best |")
    lines.append("|-----------|------|---------------|----------|------|")
    for sname, sharpes in strat_ranking:
        stype = get_strategy(sname).strategy_type
        lines.append(
            f"| {sname} | {stype} | {np.mean(sharpes):.3f} | {len(sharpes)} | {max(sharpes):.3f} |"
        )
    lines.append("")

    # Symbol ranking
    lines.append("## Ranking par symbol (HO Sharpe moyen, baseline)")
    lines.append("")
    sym_scores = {}
    for r in baseline:
        s = r["symbol"]
        if s not in sym_scores:
            sym_scores[s] = []
        sym_scores[s].append(r["ho_sharpe"])
    lines.append("| Symbol | Avg HO Sharpe | N combos | STRONG | WEAK |")
    lines.append("|--------|---------------|----------|--------|------|")
    for sym in SYMBOLS:
        if sym in sym_scores:
            sharpes = sym_scores[sym]
            n_strong = sum(1 for r in baseline if r["symbol"] == sym and r["verdict"] == "STRONG")
            n_weak = sum(1 for r in baseline if r["symbol"] == sym and r["verdict"] == "WEAK")
            lines.append(
                f"| {sym} | {np.mean(sharpes):.3f} | {len(sharpes)} | {n_strong} | {n_weak} |"
            )
    lines.append("")

    # Pool of survivors for portfolio
    lines.append("## Pool de survivants pour Portfolio V4")
    lines.append("")
    # Combine best of baseline and overlay
    all_survivors = []
    for r in results:
        if r["verdict"] in ("STRONG", "WEAK") and r["ho_trades"] >= MIN_TRADES:
            all_survivors.append(r)
    all_survivors.sort(key=lambda x: x["ho_sharpe"], reverse=True)

    # Deduplicate: keep best version (baseline or overlay) per combo
    seen = set()
    unique_survivors = []
    for r in all_survivors:
        key = (r["symbol"], r["strategy"], r["timeframe"])
        if key not in seen:
            seen.add(key)
            unique_survivors.append(r)

    lines.append(f"**{len(unique_survivors)} combos uniques survivants** (meilleur de baseline/overlay)")
    lines.append("")
    lines.append("| # | Symbol | Stratégie | TF | Overlay? | HO Sharpe | HO Return | HO DD | Verdict |")
    lines.append("|---|--------|-----------|-----|---------|-----------|-----------|-------|---------|")
    for i, r in enumerate(unique_survivors[:20]):
        ov_tag = "✓" if r["overlay"] else "—"
        lines.append(
            f"| {i+1} | {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {ov_tag} | {r['ho_sharpe']:.3f} | {r['ho_return']*100:.1f}% "
            f"| {r['ho_dd']*100:.1f}% | {r['verdict']} |"
        )
    lines.append("")

    lines.append("---")
    lines.append(f"*Généré le {datetime.now().strftime('%d %B %Y')}*")

    # Write report
    report_path = Path("docs/results/15_diagnostic_v4_edge.md")
    report_path.write_text("\n".join(lines))
    logger.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
