"""
Diagnostic V4 Edge — FAST version (2 phases)

Phase 1 (rapide ~3 min): Scan defaults sur holdout → pré-filtrage
Phase 2 (ciblée ~20 min): Walk-forward 1 seed, 30 trials sur survivants phase 1
+ Overlay comparison sur les survivants

Optimisations vs version lente:
- Pré-filtrage élimine ~70% des combos avant walk-forward
- 1 seed au lieu de 3 (suffisant pour scan)
- 30 trials au lieu de 50
- Pruning Optuna activé
- joblib parallélisation phase 1
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.registry import get_strategy, list_strategies
from engine.walk_forward import WalkForwardConfig, run_walk_forward
from engine.backtester import vectorized_backtest, RiskConfig
from engine.metrics import compute_all_metrics, composite_score
from engine.overlays import (
    apply_overlay_pipeline, OverlayPipelineConfig,
    VolTargetConfig, RegimeOverlayConfig
)
from engine.regime import RegimeConfig

# ─── Config ──────────────────────────────────────────────

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["4h", "1d"]
CUTOFF = "2025-02-01"
RISK = RiskConfig()

# Phase 1 filter
PHASE1_MIN_SHARPE = -1.5   # Élimine les pires
PHASE1_MIN_TRADES = 3

# Phase 2 walk-forward
WF_TRIALS = 30
WF_REOPTIM = "3M"
WF_WINDOW = "1Y"

# Overlay
REGIME_CFG = RegimeOverlayConfig(
    regime_config=RegimeConfig(),
    hard_cutoff=True,
    min_exposure_threshold=0.3,
)
VOL_CFG = VolTargetConfig(target_vol_annual=0.30)
OVERLAY_CFG = OverlayPipelineConfig(regime_config=REGIME_CFG, vol_config=VOL_CFG)


# ─── Data ────────────────────────────────────────────────

def load_all_data():
    data = {}
    for sym in SYMBOLS:
        data[sym] = {}
        for tf in TIMEFRAMES:
            p = Path(f"data/raw/{sym}_{tf}.parquet")
            if p.exists():
                df = pd.read_parquet(p)
                data[sym][tf] = df
                logger.info(f"  {sym}/{tf}: {len(df)} bars")
    return data


# ─── Phase 1: Quick scan defaults on holdout ─────────────

def phase1_scan_combo(sname, sym, tf, ho_data):
    """Quick backtest with default params on holdout. No walk-forward."""
    try:
        strat = get_strategy(sname)
        signals = strat.generate_signals(ho_data, strat.default_params)

        close = ho_data["close"].values.astype(np.float64)
        high = ho_data["high"].values.astype(np.float64)
        low = ho_data["low"].values.astype(np.float64)

        # Baseline
        res = vectorized_backtest(close, signals, risk=RISK, high=high, low=low, timeframe=tf)
        m = compute_all_metrics(res.equity, tf, res.trades_pnl)

        # With overlay
        sig_ov, _ = apply_overlay_pipeline(signals, ho_data, OVERLAY_CFG, timeframe=tf)
        res_ov = vectorized_backtest(close, sig_ov, risk=RISK, high=high, low=low, timeframe=tf)
        m_ov = compute_all_metrics(res_ov.equity, tf, res_ov.trades_pnl)

        return {
            "symbol": sym, "strategy": sname, "timeframe": tf,
            "p1_sharpe": m["sharpe"], "p1_return": m["total_return"],
            "p1_dd": m["max_drawdown"], "p1_trades": res.n_trades,
            "p1_ov_sharpe": m_ov["sharpe"], "p1_ov_return": m_ov["total_return"],
            "p1_ov_dd": m_ov["max_drawdown"], "p1_ov_trades": res_ov.n_trades,
        }
    except Exception as e:
        return None


def run_phase1(all_data):
    """Parallel quick scan of all combos."""
    logger.info("=" * 60)
    logger.info("PHASE 1 — Quick scan (defaults on holdout)")
    logger.info("=" * 60)

    tasks = []
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            if sym not in all_data or tf not in all_data[sym]:
                continue
            ho_data = all_data[sym][tf][all_data[sym][tf].index >= CUTOFF].copy()
            if len(ho_data) < 100:
                continue
            for sname in list_strategies():
                tasks.append((sname, sym, tf, ho_data))

    logger.info(f"  {len(tasks)} combos to scan")
    t0 = time.time()

    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(phase1_scan_combo)(s, sym, tf, ho) for s, sym, tf, ho in tasks
    )

    results = [r for r in results if r is not None]
    elapsed = time.time() - t0
    logger.info(f"  Phase 1 done in {elapsed:.0f}s ({len(results)} results)")

    return results


def filter_phase1(results):
    """Select combos worth walk-forwarding."""
    survivors = []
    for r in results:
        # Keep if EITHER baseline or overlay has decent Sharpe
        best_sharpe = max(r["p1_sharpe"], r["p1_ov_sharpe"])
        best_trades = max(r["p1_trades"], r["p1_ov_trades"])
        if best_sharpe > PHASE1_MIN_SHARPE and best_trades >= PHASE1_MIN_TRADES:
            r["p1_best_sharpe"] = best_sharpe
            survivors.append(r)

    survivors.sort(key=lambda x: x["p1_best_sharpe"], reverse=True)
    logger.info(f"  Phase 1 filter: {len(survivors)}/{len(results)} pass (threshold > {PHASE1_MIN_SHARPE})")
    return survivors


# ─── Phase 2: Walk-forward on survivors ──────────────────

def phase2_wf_combo(sname, sym, tf, data_full, data_is, use_overlay):
    """Walk-forward + holdout for a single combo."""
    try:
        strategy = get_strategy(sname)

        wf_config = WalkForwardConfig(
            strategy=strategy,
            data=data_is,
            timeframe=tf,
            reoptim_frequency=WF_REOPTIM,
            training_window=WF_WINDOW,
            param_bounds_scale=1.0,
            optim_metric="sharpe",
            n_optim_trials=WF_TRIALS,
            risk=RISK,
            seed=42,
            use_pruning=True,
        )

        wf_result = run_walk_forward(wf_config)

        if wf_result.n_oos_periods < 3 or not wf_result.best_params_per_period:
            return None

        last_params = wf_result.best_params_per_period[-1]

        # Holdout
        ho_data = data_full[data_full.index >= CUTOFF].copy()
        if len(ho_data) < 50:
            return None

        signals = strategy.generate_signals(ho_data, last_params)

        if use_overlay:
            signals, _ = apply_overlay_pipeline(signals, ho_data, OVERLAY_CFG, timeframe=tf)

        close = ho_data["close"].values.astype(np.float64)
        high = ho_data["high"].values.astype(np.float64)
        low = ho_data["low"].values.astype(np.float64)

        ho_res = vectorized_backtest(close, signals, risk=RISK, high=high, low=low, timeframe=tf)
        ho_m = compute_all_metrics(ho_res.equity, tf, ho_res.trades_pnl)

        return {
            "symbol": sym, "strategy": sname, "timeframe": tf,
            "overlay": use_overlay,
            "is_sharpe": wf_result.metrics.get("sharpe", 0),
            "is_return": wf_result.metrics.get("total_return", 0),
            "is_dd": wf_result.metrics.get("max_drawdown", 0),
            "is_trades": wf_result.metrics.get("n_trades", 0),
            "n_oos": wf_result.n_oos_periods,
            "ho_sharpe": ho_m.get("sharpe", 0),
            "ho_return": ho_m.get("total_return", 0),
            "ho_dd": ho_m.get("max_drawdown", 0),
            "ho_trades": ho_m.get("n_trades", 0),
            "ho_calmar": ho_m.get("calmar", 0),
            "ho_sortino": ho_m.get("sortino", 0),
            "ho_wr": ho_m.get("win_rate", 0),
            "ho_pf": ho_m.get("profit_factor", 0),
            "last_params": last_params,
            "ho_equity": ho_res.equity.tolist(),
            "ho_returns": ho_res.returns.tolist(),
        }
    except Exception as e:
        logger.error(f"  WF error {sym}/{sname}/{tf}: {e}")
        return None


def run_phase2(survivors, all_data):
    """Walk-forward on filtered combos."""
    logger.info("=" * 60)
    logger.info(f"PHASE 2 — Walk-forward on {len(survivors)} survivors")
    logger.info("=" * 60)

    results = []
    total = len(survivors) * 2  # baseline + overlay
    count = 0

    for s in survivors:
        sym, sname, tf = s["symbol"], s["strategy"], s["timeframe"]
        data_full = all_data[sym][tf]
        data_is = data_full[data_full.index < CUTOFF].copy()

        if len(data_is) < 500:
            continue

        for use_overlay in [False, True]:
            count += 1
            tag = "+overlay" if use_overlay else "baseline"
            logger.info(f"  [{count}/{total}] {sym}/{sname}/{tf} ({tag})")

            r = phase2_wf_combo(sname, sym, tf, data_full, data_is, use_overlay)
            if r is not None:
                # Classify
                if r["ho_trades"] < 3:
                    r["verdict"] = "FAIL"
                elif r["ho_sharpe"] > 0.3:
                    r["verdict"] = "STRONG"
                elif r["ho_sharpe"] > 0.0:
                    r["verdict"] = "WEAK"
                else:
                    r["verdict"] = "FAIL"

                results.append(r)
                logger.info(
                    f"    → {r['verdict']} | IS={r['is_sharpe']:.3f} | "
                    f"HO={r['ho_sharpe']:.3f} | Ret={r['ho_return']*100:.1f}% | "
                    f"DD={r['ho_dd']*100:.1f}% | Tr={r['ho_trades']}"
                )

    return results


# ─── Report ──────────────────────────────────────────────

def generate_report(p1_results, p2_results, p1_survivors, elapsed):
    lines = []
    lines.append("# Diagnostic V4 Edge — Fast (2 phases)")
    lines.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    lines.append(f"**Durée** : {elapsed:.1f} min")
    lines.append(f"**Config** : WF 1 seed, {WF_TRIALS} trials, reoptim={WF_REOPTIM}, window={WF_WINDOW}")
    lines.append(f"**Cutoff** : {CUTOFF}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Phase 1 summary
    lines.append("## Phase 1 — Quick Scan (defaults sur holdout)")
    lines.append("")
    lines.append(f"- **Total combos scannés** : {len(p1_results)}")
    lines.append(f"- **Survivants (Sharpe > {PHASE1_MIN_SHARPE})** : {len(p1_survivors)}")
    lines.append("")

    # Phase 1 top
    p1_sorted = sorted(p1_results, key=lambda x: max(x["p1_sharpe"], x["p1_ov_sharpe"]), reverse=True)
    lines.append("### Top 15 Phase 1 (defaults)")
    lines.append("")
    lines.append("| Symbol | Stratégie | TF | Sharpe | +Ov Sharpe | Return | +Ov Return | DD | +Ov DD |")
    lines.append("|--------|-----------|-----|--------|------------|--------|------------|-----|--------|")
    for r in p1_sorted[:15]:
        lines.append(
            f"| {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {r['p1_sharpe']:.3f} | {r['p1_ov_sharpe']:.3f} "
            f"| {r['p1_return']*100:.1f}% | {r['p1_ov_return']*100:.1f}% "
            f"| {r['p1_dd']*100:.1f}% | {r['p1_ov_dd']*100:.1f}% |"
        )
    lines.append("")

    # Phase 2 summary
    baseline = [r for r in p2_results if not r["overlay"]]
    overlay = [r for r in p2_results if r["overlay"]]

    n_strong_b = sum(1 for r in baseline if r["verdict"] == "STRONG")
    n_weak_b = sum(1 for r in baseline if r["verdict"] == "WEAK")
    n_fail_b = sum(1 for r in baseline if r["verdict"] == "FAIL")
    n_strong_o = sum(1 for r in overlay if r["verdict"] == "STRONG")
    n_weak_o = sum(1 for r in overlay if r["verdict"] == "WEAK")
    n_fail_o = sum(1 for r in overlay if r["verdict"] == "FAIL")

    lines.append("## Phase 2 — Walk-Forward + Holdout")
    lines.append("")
    lines.append(f"- **Combos WF** : {len(baseline)} baseline + {len(overlay)} overlay")
    lines.append(f"- **Baseline** : {n_strong_b} STRONG, {n_weak_b} WEAK, {n_fail_b} FAIL")
    lines.append(f"- **+Overlay** : {n_strong_o} STRONG, {n_weak_o} WEAK, {n_fail_o} FAIL")
    if baseline:
        lines.append(f"- **Avg HO Sharpe baseline** : {np.mean([r['ho_sharpe'] for r in baseline]):.3f}")
    if overlay:
        lines.append(f"- **Avg HO Sharpe +overlay** : {np.mean([r['ho_sharpe'] for r in overlay]):.3f}")
    lines.append("")

    # Phase 2 top baseline
    sorted_base = sorted(baseline, key=lambda x: x["ho_sharpe"], reverse=True)
    lines.append("### Top combos — Baseline")
    lines.append("")
    lines.append("| # | V | Symbol | Stratégie | TF | IS Sharpe | HO Sharpe | HO Ret | HO DD | Trades | Calmar |")
    lines.append("|---|---|--------|-----------|-----|-----------|-----------|--------|-------|--------|--------|")
    for i, r in enumerate(sorted_base[:20]):
        v = "✅" if r["verdict"] == "STRONG" else ("⚠️" if r["verdict"] == "WEAK" else "❌")
        lines.append(
            f"| {i+1} | {v} | {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {r['is_sharpe']:.3f} | {r['ho_sharpe']:.3f} "
            f"| {r['ho_return']*100:.1f}% | {r['ho_dd']*100:.1f}% "
            f"| {r['ho_trades']} | {r['ho_calmar']:.2f} |"
        )
    lines.append("")

    # Phase 2 top overlay
    sorted_ov = sorted(overlay, key=lambda x: x["ho_sharpe"], reverse=True)
    lines.append("### Top combos — Avec Overlays")
    lines.append("")
    lines.append("| # | V | Symbol | Stratégie | TF | IS Sharpe | HO Sharpe | HO Ret | HO DD | Trades | Calmar |")
    lines.append("|---|---|--------|-----------|-----|-----------|-----------|--------|-------|--------|--------|")
    for i, r in enumerate(sorted_ov[:20]):
        v = "✅" if r["verdict"] == "STRONG" else ("⚠️" if r["verdict"] == "WEAK" else "❌")
        lines.append(
            f"| {i+1} | {v} | {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {r['is_sharpe']:.3f} | {r['ho_sharpe']:.3f} "
            f"| {r['ho_return']*100:.1f}% | {r['ho_dd']*100:.1f}% "
            f"| {r['ho_trades']} | {r['ho_calmar']:.2f} |"
        )
    lines.append("")

    # Delta comparison
    lines.append("### Δ Overlay (top améliorations)")
    lines.append("")
    lines.append("| Symbol | Stratégie | TF | Base Sharpe | +Ov Sharpe | Δ | Base DD | +Ov DD |")
    lines.append("|--------|-----------|-----|-------------|------------|---|---------|--------|")
    pairs = []
    for b in baseline:
        key = (b["symbol"], b["strategy"], b["timeframe"])
        ov = [o for o in overlay if (o["symbol"], o["strategy"], o["timeframe"]) == key]
        if ov:
            delta = ov[0]["ho_sharpe"] - b["ho_sharpe"]
            pairs.append((b, ov[0], delta))
    pairs.sort(key=lambda x: x[2], reverse=True)
    for b, o, d in pairs[:20]:
        sign = "+" if d > 0 else ""
        lines.append(
            f"| {b['symbol']} | {b['strategy']} | {b['timeframe']} "
            f"| {b['ho_sharpe']:.3f} | {o['ho_sharpe']:.3f} | {sign}{d:.3f} "
            f"| {b['ho_dd']*100:.1f}% | {o['ho_dd']*100:.1f}% |"
        )
    lines.append("")

    # Strategy ranking
    lines.append("### Ranking stratégies (avg HO Sharpe baseline)")
    lines.append("")
    strat_scores = {}
    for r in baseline:
        strat_scores.setdefault(r["strategy"], []).append(r["ho_sharpe"])
    ranking = sorted(strat_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)
    lines.append("| Stratégie | Avg HO Sharpe | Best | N |")
    lines.append("|-----------|---------------|------|---|")
    for sname, sharpes in ranking:
        lines.append(f"| {sname} | {np.mean(sharpes):.3f} | {max(sharpes):.3f} | {len(sharpes)} |")
    lines.append("")

    # Survivor pool
    all_surv = [r for r in p2_results if r["verdict"] in ("STRONG", "WEAK") and r["ho_trades"] >= 3]
    all_surv.sort(key=lambda x: x["ho_sharpe"], reverse=True)
    seen = set()
    unique = []
    for r in all_surv:
        key = (r["symbol"], r["strategy"], r["timeframe"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    lines.append("## Pool survivants pour Portfolio V4")
    lines.append("")
    lines.append(f"**{len(unique)} combos uniques**")
    lines.append("")
    lines.append("| # | Symbol | Stratégie | TF | Ov? | HO Sharpe | HO Ret | HO DD | Verdict |")
    lines.append("|---|--------|-----------|-----|-----|-----------|--------|-------|---------|")
    for i, r in enumerate(unique):
        ov = "✓" if r["overlay"] else "—"
        lines.append(
            f"| {i+1} | {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {ov} | {r['ho_sharpe']:.3f} | {r['ho_return']*100:.1f}% "
            f"| {r['ho_dd']*100:.1f}% | {r['verdict']} |"
        )
    lines.append("")
    lines.append("---")
    lines.append(f"*Généré le {datetime.now().strftime('%d %B %Y')}*")

    Path("docs/results/15_diagnostic_v4_edge.md").write_text("\n".join(lines))
    logger.info(f"Report: docs/results/15_diagnostic_v4_edge.md")


# ─── Main ────────────────────────────────────────────────

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC V4 EDGE — FAST (2 phases)")
    logger.info("=" * 60)

    all_data = load_all_data()

    # Phase 1
    p1_results = run_phase1(all_data)
    p1_survivors = filter_phase1(p1_results)

    # Log phase 1 top 10
    p1_sorted = sorted(p1_results, key=lambda x: max(x["p1_sharpe"], x["p1_ov_sharpe"]), reverse=True)
    logger.info("  Phase 1 Top 10:")
    for r in p1_sorted[:10]:
        best = max(r["p1_sharpe"], r["p1_ov_sharpe"])
        logger.info(f"    {r['symbol']}/{r['strategy']}/{r['timeframe']} → best Sharpe={best:.3f}")

    # Phase 2
    p2_results = run_phase2(p1_survivors, all_data)

    elapsed = (time.time() - t0) / 60
    logger.info(f"\nTotal: {elapsed:.1f} min")

    # Save JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)
    save_data = {
        "phase1": [{k: v for k, v in r.items()} for r in p1_results],
        "phase2": [{k: v for k, v in r.items() if k not in ("ho_equity", "ho_returns")} for r in p2_results],
    }
    with open(f"results/diagnostic_v4_fast_{ts}.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Report
    generate_report(p1_results, p2_results, p1_survivors, elapsed)

    # Summary
    baseline = [r for r in p2_results if not r["overlay"]]
    overlay = [r for r in p2_results if r["overlay"]]
    n_strong = sum(1 for r in p2_results if r["verdict"] == "STRONG")
    n_weak = sum(1 for r in p2_results if r["verdict"] == "WEAK")
    logger.info(f"\n{'='*60}")
    logger.info(f"RÉSUMÉ: {n_strong} STRONG + {n_weak} WEAK survivants")
    if baseline:
        logger.info(f"  Avg HO Sharpe baseline: {np.mean([r['ho_sharpe'] for r in baseline]):.3f}")
    if overlay:
        logger.info(f"  Avg HO Sharpe +overlay: {np.mean([r['ho_sharpe'] for r in overlay]):.3f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
