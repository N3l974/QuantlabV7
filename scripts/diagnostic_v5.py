"""
Diagnostic V5 — ATR SL/TP + Risk-based Sizing

Phase 1 (rapide ~3 min): Scan defaults sur holdout → pré-filtrage
Phase 2 (ciblée ~25 min): Walk-forward 1 seed, 30 trials sur survivants
  - Optuna optimise atr_sl_mult, atr_tp_mult en plus des params classiques
  - Compare V4 (pct SL/TP) vs V5 (ATR SL/TP) sur les mêmes combos

Objectif: identifier quels combos bénéficient de l'ATR SL/TP
et du risk-based sizing vs les SL/TP fixes de V4.
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
from engine.backtester import vectorized_backtest, backtest_strategy, RiskConfig
from engine.metrics import compute_all_metrics, composite_score
from engine.overlays import (
    apply_overlay_pipeline, OverlayPipelineConfig,
    VolTargetConfig, RegimeOverlayConfig
)
from engine.regime import RegimeConfig


RESULTS_DIR = Path("portfolio/v5b/results")

# ─── Config ──────────────────────────────────────────────

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["4h", "1d"]
CUTOFF = "2025-02-01"

# Risk configs
RISK_V4 = RiskConfig()  # default: no risk_per_trade_pct
RISK_V5 = RiskConfig(risk_per_trade_pct=0.01)  # 1% risk per trade

# Phase 1 filter
PHASE1_MIN_SHARPE = -1.5
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
    """Quick backtest with default params on holdout."""
    try:
        strat = get_strategy(sname)
        signals = strat.generate_signals(ho_data, strat.default_params)

        close = ho_data["close"].values.astype(np.float64)
        high = ho_data["high"].values.astype(np.float64)
        low = ho_data["low"].values.astype(np.float64)

        # Baseline (V4 mode: atr_sl_mult=0)
        res = vectorized_backtest(close, signals, risk=RISK_V4, high=high, low=low, timeframe=tf)
        m = compute_all_metrics(res.equity, tf, res.trades_pnl)

        # With overlay
        sig_ov, _ = apply_overlay_pipeline(signals, ho_data, OVERLAY_CFG, timeframe=tf)
        res_ov = vectorized_backtest(close, sig_ov, risk=RISK_V4, high=high, low=low, timeframe=tf)
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
        best_sharpe = max(r["p1_sharpe"], r["p1_ov_sharpe"])
        best_trades = max(r["p1_trades"], r["p1_ov_trades"])
        if best_sharpe > PHASE1_MIN_SHARPE and best_trades >= PHASE1_MIN_TRADES:
            r["p1_best_sharpe"] = best_sharpe
            survivors.append(r)

    survivors.sort(key=lambda x: x["p1_best_sharpe"], reverse=True)
    logger.info(f"  Phase 1 filter: {len(survivors)}/{len(results)} pass (threshold > {PHASE1_MIN_SHARPE})")
    return survivors


# ─── Phase 2: Walk-forward V4 vs V5 ─────────────────────

def phase2_wf_combo(sname, sym, tf, data_full, data_is, mode="v4", use_overlay=False):
    """
    Walk-forward + holdout for a single combo.
    mode="v4": standard pct SL/TP (atr_sl_mult will be optimized but can be 0)
    mode="v5": ATR SL/TP + risk-based sizing
    """
    try:
        strategy = get_strategy(sname)
        risk = RISK_V5 if mode == "v5" else RISK_V4

        wf_config = WalkForwardConfig(
            strategy=strategy,
            data=data_is,
            timeframe=tf,
            reoptim_frequency=WF_REOPTIM,
            training_window=WF_WINDOW,
            param_bounds_scale=1.0,
            optim_metric="sharpe",
            n_optim_trials=WF_TRIALS,
            risk=risk,
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

        # Use V5 API if available and mode is v5
        if mode == "v5" and hasattr(strategy, 'generate_signals_v5'):
            signals, sl_distances = strategy.generate_signals_v5(ho_data, last_params)
        else:
            signals = strategy.generate_signals(ho_data, last_params)
            sl_distances = None

        if use_overlay:
            signals, _ = apply_overlay_pipeline(signals, ho_data, OVERLAY_CFG, timeframe=tf)

        close = ho_data["close"].values.astype(np.float64)
        high = ho_data["high"].values.astype(np.float64)
        low = ho_data["low"].values.astype(np.float64)

        ho_res = vectorized_backtest(
            close, signals, risk=risk, high=high, low=low, timeframe=tf,
            sl_distances=sl_distances,
        )
        ho_m = compute_all_metrics(ho_res.equity, tf, ho_res.trades_pnl)

        # Extract V5 params from optimized params
        atr_sl = last_params.get("atr_sl_mult", 0.0)
        atr_tp = last_params.get("atr_tp_mult", 0.0)

        return {
            "symbol": sym, "strategy": sname, "timeframe": tf,
            "mode": mode, "overlay": use_overlay,
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
            "atr_sl_mult": atr_sl,
            "atr_tp_mult": atr_tp,
            "last_params": last_params,
            "ho_equity": ho_res.equity.tolist(),
            "ho_returns": ho_res.returns.tolist(),
        }
    except Exception as e:
        logger.error(f"  WF error {sym}/{sname}/{tf} ({mode}): {e}")
        return None


def run_phase2(survivors, all_data):
    """Walk-forward on filtered combos: V4 baseline + V5 ATR mode."""
    logger.info("=" * 60)
    logger.info(f"PHASE 2 — Walk-forward V4 vs V5 on {len(survivors)} survivors")
    logger.info("=" * 60)

    results = []
    # 4 variants per combo: v4 baseline, v4+overlay, v5 baseline, v5+overlay
    total = len(survivors) * 4
    count = 0

    for s in survivors:
        sym, sname, tf = s["symbol"], s["strategy"], s["timeframe"]
        data_full = all_data[sym][tf]
        data_is = data_full[data_full.index < CUTOFF].copy()

        if len(data_is) < 500:
            continue

        for mode in ["v4", "v5"]:
            for use_overlay in [False, True]:
                count += 1
                tag = f"{mode}" + ("+ov" if use_overlay else "")
                logger.info(f"  [{count}/{total}] {sym}/{sname}/{tf} ({tag})")

                r = phase2_wf_combo(sname, sym, tf, data_full, data_is, mode, use_overlay)
                if r is not None:
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
                        f"    -> {r['verdict']} | IS={r['is_sharpe']:.3f} | "
                        f"HO={r['ho_sharpe']:.3f} | Ret={r['ho_return']*100:.1f}% | "
                        f"DD={r['ho_dd']*100:.1f}% | Tr={r['ho_trades']} | "
                        f"ATR_SL={r['atr_sl_mult']:.2f} ATR_TP={r['atr_tp_mult']:.2f}"
                    )

    return results


# ─── Report ──────────────────────────────────────────────

def generate_report(p1_results, p2_results, p1_survivors, elapsed):
    lines = []
    lines.append("# Diagnostic V5 — ATR SL/TP + Risk-based Sizing")
    lines.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    lines.append(f"**Duree** : {elapsed:.1f} min")
    lines.append(f"**Config** : WF 1 seed, {WF_TRIALS} trials, reoptim={WF_REOPTIM}, window={WF_WINDOW}")
    lines.append(f"**Cutoff** : {CUTOFF}")
    lines.append(f"**V5 risk_per_trade_pct** : {RISK_V5.risk_per_trade_pct}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Phase 1 summary
    lines.append("## Phase 1 — Quick Scan (defaults sur holdout)")
    lines.append("")
    lines.append(f"- **Total combos scannes** : {len(p1_results)}")
    lines.append(f"- **Survivants (Sharpe > {PHASE1_MIN_SHARPE})** : {len(p1_survivors)}")
    lines.append("")

    # Split results by mode
    v4_base = [r for r in p2_results if r["mode"] == "v4" and not r["overlay"]]
    v4_ov = [r for r in p2_results if r["mode"] == "v4" and r["overlay"]]
    v5_base = [r for r in p2_results if r["mode"] == "v5" and not r["overlay"]]
    v5_ov = [r for r in p2_results if r["mode"] == "v5" and r["overlay"]]

    # Phase 2 summary
    lines.append("## Phase 2 — Walk-Forward V4 vs V5")
    lines.append("")
    for label, group in [("V4 baseline", v4_base), ("V4 +overlay", v4_ov),
                          ("V5 baseline", v5_base), ("V5 +overlay", v5_ov)]:
        n_s = sum(1 for r in group if r["verdict"] == "STRONG")
        n_w = sum(1 for r in group if r["verdict"] == "WEAK")
        n_f = sum(1 for r in group if r["verdict"] == "FAIL")
        avg_sh = np.mean([r["ho_sharpe"] for r in group]) if group else 0
        lines.append(f"- **{label}** : {n_s} STRONG, {n_w} WEAK, {n_f} FAIL | Avg Sharpe={avg_sh:.3f}")
    lines.append("")

    # V4 vs V5 comparison (key table)
    lines.append("## V4 vs V5 Comparison (baseline)")
    lines.append("")
    lines.append("| Symbol | Strategie | TF | V4 Sharpe | V5 Sharpe | Delta | V4 DD | V5 DD | V5 ATR_SL | V5 ATR_TP |")
    lines.append("|--------|-----------|-----|-----------|-----------|-------|-------|-------|-----------|-----------|")

    pairs = []
    for v4 in v4_base:
        key = (v4["symbol"], v4["strategy"], v4["timeframe"])
        v5 = [r for r in v5_base if (r["symbol"], r["strategy"], r["timeframe"]) == key]
        if v5:
            delta = v5[0]["ho_sharpe"] - v4["ho_sharpe"]
            pairs.append((v4, v5[0], delta))

    pairs.sort(key=lambda x: x[2], reverse=True)
    for v4, v5, d in pairs:
        sign = "+" if d > 0 else ""
        lines.append(
            f"| {v4['symbol']} | {v4['strategy']} | {v4['timeframe']} "
            f"| {v4['ho_sharpe']:.3f} | {v5['ho_sharpe']:.3f} | {sign}{d:.3f} "
            f"| {v4['ho_dd']*100:.1f}% | {v5['ho_dd']*100:.1f}% "
            f"| {v5['atr_sl_mult']:.2f} | {v5['atr_tp_mult']:.2f} |"
        )
    lines.append("")

    # V5 improvement stats
    if pairs:
        deltas = [d for _, _, d in pairs]
        n_improved = sum(1 for d in deltas if d > 0)
        n_degraded = sum(1 for d in deltas if d < 0)
        lines.append(f"**V5 vs V4** : {n_improved} improved, {n_degraded} degraded, avg delta={np.mean(deltas):.3f}")
        lines.append("")

    # Top V5 combos
    sorted_v5 = sorted(v5_base, key=lambda x: x["ho_sharpe"], reverse=True)
    lines.append("### Top V5 combos (baseline)")
    lines.append("")
    lines.append("| # | V | Symbol | Strategie | TF | HO Sharpe | HO Ret | HO DD | Trades | ATR_SL | ATR_TP |")
    lines.append("|---|---|--------|-----------|-----|-----------|--------|-------|--------|--------|--------|")
    for i, r in enumerate(sorted_v5[:20]):
        v = "+" if r["verdict"] == "STRONG" else ("~" if r["verdict"] == "WEAK" else "-")
        lines.append(
            f"| {i+1} | {v} | {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {r['ho_sharpe']:.3f} | {r['ho_return']*100:.1f}% | {r['ho_dd']*100:.1f}% "
            f"| {r['ho_trades']} | {r['atr_sl_mult']:.2f} | {r['atr_tp_mult']:.2f} |"
        )
    lines.append("")

    # Top V5 + overlay
    sorted_v5ov = sorted(v5_ov, key=lambda x: x["ho_sharpe"], reverse=True)
    lines.append("### Top V5 combos (+overlay)")
    lines.append("")
    lines.append("| # | V | Symbol | Strategie | TF | HO Sharpe | HO Ret | HO DD | Trades | ATR_SL | ATR_TP |")
    lines.append("|---|---|--------|-----------|-----|-----------|--------|-------|--------|--------|--------|")
    for i, r in enumerate(sorted_v5ov[:20]):
        v = "+" if r["verdict"] == "STRONG" else ("~" if r["verdict"] == "WEAK" else "-")
        lines.append(
            f"| {i+1} | {v} | {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {r['ho_sharpe']:.3f} | {r['ho_return']*100:.1f}% | {r['ho_dd']*100:.1f}% "
            f"| {r['ho_trades']} | {r['atr_sl_mult']:.2f} | {r['atr_tp_mult']:.2f} |"
        )
    lines.append("")

    # ATR SL/TP usage analysis
    lines.append("## ATR SL/TP Usage Analysis")
    lines.append("")
    v5_all = v5_base + v5_ov
    atr_used = [r for r in v5_all if r["atr_sl_mult"] > 0.01]
    atr_not_used = [r for r in v5_all if r["atr_sl_mult"] <= 0.01]
    lines.append(f"- **ATR SL/TP used** : {len(atr_used)}/{len(v5_all)} combos ({100*len(atr_used)/max(len(v5_all),1):.0f}%)")
    if atr_used:
        lines.append(f"  - Avg ATR_SL mult: {np.mean([r['atr_sl_mult'] for r in atr_used]):.2f}")
        lines.append(f"  - Avg ATR_TP mult: {np.mean([r['atr_tp_mult'] for r in atr_used]):.2f}")
        lines.append(f"  - Avg Sharpe: {np.mean([r['ho_sharpe'] for r in atr_used]):.3f}")
    if atr_not_used:
        lines.append(f"- **Pct SL/TP kept** : {len(atr_not_used)} combos")
        lines.append(f"  - Avg Sharpe: {np.mean([r['ho_sharpe'] for r in atr_not_used]):.3f}")
    lines.append("")

    # Survivor pool for Portfolio V5
    all_surv = [r for r in p2_results if r["verdict"] in ("STRONG", "WEAK") and r["ho_trades"] >= 3]
    all_surv.sort(key=lambda x: x["ho_sharpe"], reverse=True)

    lines.append("## Pool survivants pour Portfolio V5")
    lines.append("")
    lines.append(f"**{len(all_surv)} combos (toutes variantes)**")
    lines.append("")
    lines.append("| # | Symbol | Strategie | TF | Mode | Ov? | HO Sharpe | HO Ret | HO DD | Verdict |")
    lines.append("|---|--------|-----------|-----|------|-----|-----------|--------|-------|---------|")
    for i, r in enumerate(all_surv[:40]):
        ov = "Y" if r["overlay"] else "-"
        lines.append(
            f"| {i+1} | {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {r['mode']} | {ov} | {r['ho_sharpe']:.3f} | {r['ho_return']*100:.1f}% "
            f"| {r['ho_dd']*100:.1f}% | {r['verdict']} |"
        )
    lines.append("")
    lines.append("---")
    lines.append(f"*Genere le {datetime.now().strftime('%d %B %Y')}*")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "18_diagnostic_v5.md"
    report_path.write_text("\n".join(lines))
    logger.info(f"Report: {report_path}")


# ─── Main ────────────────────────────────────────────────

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC V5 — ATR SL/TP + Risk-based Sizing")
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
        logger.info(f"    {r['symbol']}/{r['strategy']}/{r['timeframe']} -> best Sharpe={best:.3f}")

    # Phase 2
    p2_results = run_phase2(p1_survivors, all_data)

    elapsed = (time.time() - t0) / 60
    logger.info(f"\nTotal: {elapsed:.1f} min")

    # Save JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_data = {
        "phase1": [{k: v for k, v in r.items()} for r in p1_results],
        "phase2": [{k: v for k, v in r.items() if k not in ("ho_equity", "ho_returns")} for r in p2_results],
    }
    with open(RESULTS_DIR / f"diagnostic_v5_{ts}.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Report
    generate_report(p1_results, p2_results, p1_survivors, elapsed)

    # Summary
    v4_base = [r for r in p2_results if r["mode"] == "v4" and not r["overlay"]]
    v5_base = [r for r in p2_results if r["mode"] == "v5" and not r["overlay"]]
    n_strong = sum(1 for r in p2_results if r["verdict"] == "STRONG")
    n_weak = sum(1 for r in p2_results if r["verdict"] == "WEAK")
    logger.info(f"\n{'='*60}")
    logger.info(f"RESUME: {n_strong} STRONG + {n_weak} WEAK survivants")
    if v4_base:
        logger.info(f"  Avg HO Sharpe V4 baseline: {np.mean([r['ho_sharpe'] for r in v4_base]):.3f}")
    if v5_base:
        logger.info(f"  Avg HO Sharpe V5 baseline: {np.mean([r['ho_sharpe'] for r in v5_base]):.3f}")

    # V5 vs V4 delta
    deltas = []
    for v4 in v4_base:
        key = (v4["symbol"], v4["strategy"], v4["timeframe"])
        v5 = [r for r in v5_base if (r["symbol"], r["strategy"], r["timeframe"]) == key]
        if v5:
            deltas.append(v5[0]["ho_sharpe"] - v4["ho_sharpe"])
    if deltas:
        n_up = sum(1 for d in deltas if d > 0)
        logger.info(f"  V5 vs V4: {n_up}/{len(deltas)} improved, avg delta={np.mean(deltas):.3f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
