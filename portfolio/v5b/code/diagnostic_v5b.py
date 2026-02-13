"""
Diagnostic V5b — Full enrichment diagnostic.

Enrichments over V5:
1. Trailing stop ATR (optimizable)
2. Max holding period (optimizable)
3. Breakeven stop (optimizable)
4. Multi-seed robustness (3 seeds)
5. Risk-pct grid (0%, 0.5%, 1%, 2%)
6. Correlation matrix between survivors

Phase 1: Quick scan defaults on holdout → pre-filtering
Phase 2: Walk-forward 3 seeds, 30 trials on survivors
  - V5b mode: ATR SL/TP + trailing + breakeven + max_hold
  - Risk grid: compare 4 risk_per_trade_pct values
  - Correlation matrix for portfolio construction
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

# ─── Config ──────────────────────────────────────────────

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["4h", "1d"]
CUTOFF = "2025-02-01"

# Risk configs grid
RISK_GRID = {
    "flat": RiskConfig(),
    "r0.5": RiskConfig(risk_per_trade_pct=0.005),
    "r1.0": RiskConfig(risk_per_trade_pct=0.01),
    "r2.0": RiskConfig(risk_per_trade_pct=0.02),
}

# Phase 1 filter
PHASE1_MIN_SHARPE = -1.5
PHASE1_MIN_TRADES = 3

# Phase 2 walk-forward
WF_TRIALS = 30
WF_REOPTIM = "3M"
WF_WINDOW = "1Y"
N_SEEDS = 3

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


# ─── Phase 1: Quick scan ─────────────────────────────────

def phase1_scan_combo(sname, sym, tf, ho_data):
    try:
        strat = get_strategy(sname)
        signals = strat.generate_signals(ho_data, strat.default_params)
        close = ho_data["close"].values.astype(np.float64)
        high = ho_data["high"].values.astype(np.float64)
        low = ho_data["low"].values.astype(np.float64)
        res = vectorized_backtest(close, signals, risk=RiskConfig(), high=high, low=low, timeframe=tf)
        m = compute_all_metrics(res.equity, tf, res.trades_pnl)
        sig_ov, _ = apply_overlay_pipeline(signals, ho_data, OVERLAY_CFG, timeframe=tf)
        res_ov = vectorized_backtest(close, sig_ov, risk=RiskConfig(), high=high, low=low, timeframe=tf)
        m_ov = compute_all_metrics(res_ov.equity, tf, res_ov.trades_pnl)
        return {
            "symbol": sym, "strategy": sname, "timeframe": tf,
            "p1_sharpe": m["sharpe"], "p1_return": m["total_return"],
            "p1_dd": m["max_drawdown"], "p1_trades": res.n_trades,
            "p1_ov_sharpe": m_ov["sharpe"], "p1_ov_return": m_ov["total_return"],
            "p1_ov_dd": m_ov["max_drawdown"], "p1_ov_trades": res_ov.n_trades,
        }
    except Exception:
        return None


def run_phase1(all_data):
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
    logger.info(f"  Phase 1 done in {time.time()-t0:.0f}s ({len(results)} results)")
    return results


def filter_phase1(results):
    survivors = []
    for r in results:
        best_sharpe = max(r["p1_sharpe"], r["p1_ov_sharpe"])
        best_trades = max(r["p1_trades"], r["p1_ov_trades"])
        if best_sharpe > PHASE1_MIN_SHARPE and best_trades >= PHASE1_MIN_TRADES:
            r["p1_best_sharpe"] = best_sharpe
            survivors.append(r)
    survivors.sort(key=lambda x: x["p1_best_sharpe"], reverse=True)
    logger.info(f"  Phase 1 filter: {len(survivors)}/{len(results)} pass")
    return survivors


# ─── Phase 2: Walk-forward multi-seed ────────────────────

def phase2_wf_single_seed(strategy, data_is, tf, risk, seed):
    """Single seed walk-forward, returns WF result or None."""
    try:
        wf_config = WalkForwardConfig(
            strategy=strategy, data=data_is, timeframe=tf,
            reoptim_frequency=WF_REOPTIM, training_window=WF_WINDOW,
            param_bounds_scale=1.0, optim_metric="sharpe",
            n_optim_trials=WF_TRIALS, risk=risk, seed=seed,
            use_pruning=True,
        )
        return run_walk_forward(wf_config)
    except Exception:
        return None


def phase2_wf_combo(sname, sym, tf, data_full, data_is, risk_key, risk, use_overlay):
    """Multi-seed walk-forward + holdout evaluation."""
    try:
        strategy = get_strategy(sname)

        # Multi-seed walk-forward
        wf_results = []
        for seed_offset in range(N_SEEDS):
            seed = 42 + seed_offset
            wf = phase2_wf_single_seed(strategy, data_is, tf, risk, seed)
            if wf and wf.n_oos_periods >= 3 and wf.best_params_per_period:
                wf_results.append(wf)

        if not wf_results:
            return None

        # Pick median-Sharpe seed
        sharpes = [wf.metrics.get("sharpe", -99) for wf in wf_results]
        median_sharpe = float(np.median(sharpes))
        best_idx = int(np.argmin(np.abs(np.array(sharpes) - median_sharpe)))
        wf_result = wf_results[best_idx]
        last_params = wf_result.best_params_per_period[-1]

        # Holdout
        ho_data = data_full[data_full.index >= CUTOFF].copy()
        if len(ho_data) < 50:
            return None

        if hasattr(strategy, 'generate_signals_v5'):
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

        return {
            "symbol": sym, "strategy": sname, "timeframe": tf,
            "risk_key": risk_key, "overlay": use_overlay,
            "n_seeds_ok": len(wf_results),
            "seed_sharpes": sharpes,
            "seed_sharpe_std": float(np.std(sharpes)) if len(sharpes) > 1 else 0,
            "is_sharpe": wf_result.metrics.get("sharpe", 0),
            "is_return": wf_result.metrics.get("total_return", 0),
            "is_dd": wf_result.metrics.get("max_drawdown", 0),
            "n_oos": wf_result.n_oos_periods,
            "ho_sharpe": ho_m.get("sharpe", 0),
            "ho_return": ho_m.get("total_return", 0),
            "ho_dd": ho_m.get("max_drawdown", 0),
            "ho_trades": ho_m.get("n_trades", 0),
            "ho_calmar": ho_m.get("calmar", 0),
            "ho_sortino": ho_m.get("sortino", 0),
            "ho_wr": ho_m.get("win_rate", 0),
            "ho_pf": ho_m.get("profit_factor", 0),
            "atr_sl_mult": last_params.get("atr_sl_mult", 0),
            "atr_tp_mult": last_params.get("atr_tp_mult", 0),
            "trailing_atr_mult": last_params.get("trailing_atr_mult", 0),
            "breakeven_trigger_pct": last_params.get("breakeven_trigger_pct", 0),
            "max_holding_bars": last_params.get("max_holding_bars", 0),
            "last_params": last_params,
            "ho_equity": ho_res.equity.tolist(),
            "ho_returns": ho_res.returns.tolist(),
        }
    except Exception as e:
        logger.error(f"  WF error {sym}/{sname}/{tf} ({risk_key}): {e}")
        return None


def run_phase2(survivors, all_data):
    """Walk-forward with risk grid + overlay variants."""
    logger.info("=" * 60)
    logger.info(f"PHASE 2 — Walk-forward V5b ({N_SEEDS} seeds) on {len(survivors)} survivors")
    logger.info(f"  Risk grid: {list(RISK_GRID.keys())}")
    logger.info("=" * 60)

    results = []
    # For each survivor: risk grid × overlay = 4 risk × 2 overlay = 8 variants
    # But to keep runtime reasonable, use only 2 risk keys: "flat" + "r1.0" and 2 overlay states
    risk_keys_to_test = ["flat", "r1.0"]
    total = len(survivors) * len(risk_keys_to_test) * 2
    count = 0

    for s in survivors:
        sym, sname, tf = s["symbol"], s["strategy"], s["timeframe"]
        data_full = all_data[sym][tf]
        data_is = data_full[data_full.index < CUTOFF].copy()
        if len(data_is) < 500:
            continue

        for risk_key in risk_keys_to_test:
            risk = RISK_GRID[risk_key]
            for use_overlay in [False, True]:
                count += 1
                tag = f"{risk_key}" + ("+ov" if use_overlay else "")
                logger.info(f"  [{count}/{total}] {sym}/{sname}/{tf} ({tag})")

                r = phase2_wf_combo(sname, sym, tf, data_full, data_is, risk_key, risk, use_overlay)
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
                        f"    -> {r['verdict']} | HO Sharpe={r['ho_sharpe']:.3f} | "
                        f"Ret={r['ho_return']*100:.1f}% | DD={r['ho_dd']*100:.1f}% | "
                        f"Tr={r['ho_trades']} | trail={r['trailing_atr_mult']:.2f} | "
                        f"BE={r['breakeven_trigger_pct']:.3f} | maxH={r['max_holding_bars']} | "
                        f"seeds_std={r['seed_sharpe_std']:.3f}"
                    )

    # Extended risk grid on top survivors only
    top_combos = [(s["symbol"], s["strategy"], s["timeframe"])
                  for s in sorted(results, key=lambda x: x["ho_sharpe"], reverse=True)
                  if s["verdict"] == "STRONG" and s["risk_key"] == "flat" and not s["overlay"]][:10]

    if top_combos:
        logger.info(f"\n  Risk grid extension on top {len(top_combos)} combos...")
        extra_keys = ["r0.5", "r2.0"]
        for sym, sname, tf in top_combos:
            data_full = all_data[sym][tf]
            data_is = data_full[data_full.index < CUTOFF].copy()
            for risk_key in extra_keys:
                risk = RISK_GRID[risk_key]
                r = phase2_wf_combo(sname, sym, tf, data_full, data_is, risk_key, risk, False)
                if r is not None:
                    r["verdict"] = "STRONG" if r["ho_sharpe"] > 0.3 else ("WEAK" if r["ho_sharpe"] > 0 else "FAIL")
                    results.append(r)
                    logger.info(f"    {sym}/{sname}/{tf} ({risk_key}): Sharpe={r['ho_sharpe']:.3f}")

    return results


# ─── Correlation matrix ──────────────────────────────────

def compute_correlation_matrix(results):
    """Compute pairwise correlation of HO returns for STRONG survivors."""
    strong = [r for r in results if r["verdict"] == "STRONG"
              and r["risk_key"] == "flat" and not r["overlay"]
              and len(r.get("ho_returns", [])) > 10]

    if len(strong) < 2:
        return None, []

    labels = [f"{r['symbol'][:3]}/{r['strategy'][:12]}/{r['timeframe']}" for r in strong]
    returns_matrix = []
    for r in strong:
        rets = np.array(r["ho_returns"])
        returns_matrix.append(rets)

    # Align lengths (pad shorter with 0)
    max_len = max(len(r) for r in returns_matrix)
    aligned = np.zeros((len(returns_matrix), max_len))
    for i, r in enumerate(returns_matrix):
        aligned[i, :len(r)] = r

    corr = np.corrcoef(aligned)
    return corr, labels


# ─── Report ──────────────────────────────────────────────

def generate_report(p1_results, p2_results, p1_survivors, elapsed):
    lines = []
    lines.append("# Diagnostic V5b — Full Enrichment")
    lines.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    lines.append(f"**Duree** : {elapsed:.1f} min")
    lines.append(f"**Config** : WF {N_SEEDS} seeds, {WF_TRIALS} trials, reoptim={WF_REOPTIM}, window={WF_WINDOW}")
    lines.append(f"**Cutoff** : {CUTOFF}")
    lines.append(f"**New features** : trailing_stop, breakeven, max_holding, multi-seed, risk_grid, correlation")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Phase 1
    lines.append("## Phase 1 — Quick Scan")
    lines.append(f"- **Combos scannes** : {len(p1_results)}")
    lines.append(f"- **Survivants** : {len(p1_survivors)}")
    lines.append("")

    # Phase 2 summary by risk key
    lines.append("## Phase 2 — Walk-Forward V5b")
    lines.append("")
    for risk_key in ["flat", "r1.0", "r0.5", "r2.0"]:
        group = [r for r in p2_results if r["risk_key"] == risk_key and not r["overlay"]]
        if not group:
            continue
        n_s = sum(1 for r in group if r["verdict"] == "STRONG")
        n_w = sum(1 for r in group if r["verdict"] == "WEAK")
        n_f = sum(1 for r in group if r["verdict"] == "FAIL")
        avg = np.mean([r["ho_sharpe"] for r in group])
        lines.append(f"- **{risk_key} baseline** : {n_s}S/{n_w}W/{n_f}F | Avg Sharpe={avg:.3f}")
    for risk_key in ["flat", "r1.0"]:
        group = [r for r in p2_results if r["risk_key"] == risk_key and r["overlay"]]
        if not group:
            continue
        n_s = sum(1 for r in group if r["verdict"] == "STRONG")
        avg = np.mean([r["ho_sharpe"] for r in group])
        lines.append(f"- **{risk_key} +overlay** : {n_s} STRONG | Avg Sharpe={avg:.3f}")
    lines.append("")

    # Top V5b combos (flat baseline)
    flat_base = sorted([r for r in p2_results if r["risk_key"] == "flat" and not r["overlay"]],
                       key=lambda x: x["ho_sharpe"], reverse=True)
    lines.append("### Top V5b combos (flat baseline)")
    lines.append("")
    lines.append("| # | V | Symbol | Strategy | TF | Sharpe | Ret | DD | Tr | Trail | BE | MaxH | Seed_std |")
    lines.append("|---|---|--------|----------|-----|--------|-----|-----|-----|-------|-----|------|----------|")
    for i, r in enumerate(flat_base[:25]):
        v = "+" if r["verdict"] == "STRONG" else ("~" if r["verdict"] == "WEAK" else "-")
        lines.append(
            f"| {i+1} | {v} | {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {r['ho_sharpe']:.3f} | {r['ho_return']*100:.1f}% | {r['ho_dd']*100:.1f}% "
            f"| {r['ho_trades']} | {r['trailing_atr_mult']:.2f} | {r['breakeven_trigger_pct']:.3f} "
            f"| {r['max_holding_bars']} | {r['seed_sharpe_std']:.3f} |"
        )
    lines.append("")

    # Top V5b combos (r1.0 + overlay — best mode)
    r1_ov = sorted([r for r in p2_results if r["risk_key"] == "r1.0" and r["overlay"]],
                   key=lambda x: x["ho_sharpe"], reverse=True)
    lines.append("### Top V5b combos (risk=1% + overlay)")
    lines.append("")
    lines.append("| # | V | Symbol | Strategy | TF | Sharpe | Ret | DD | Tr | Trail | ATR_SL | ATR_TP |")
    lines.append("|---|---|--------|----------|-----|--------|-----|-----|-----|-------|--------|--------|")
    for i, r in enumerate(r1_ov[:25]):
        v = "+" if r["verdict"] == "STRONG" else ("~" if r["verdict"] == "WEAK" else "-")
        lines.append(
            f"| {i+1} | {v} | {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {r['ho_sharpe']:.3f} | {r['ho_return']*100:.1f}% | {r['ho_dd']*100:.1f}% "
            f"| {r['ho_trades']} | {r['trailing_atr_mult']:.2f} | {r['atr_sl_mult']:.2f} "
            f"| {r['atr_tp_mult']:.2f} |"
        )
    lines.append("")

    # Feature usage analysis
    lines.append("## Feature Usage Analysis")
    lines.append("")
    all_v5b = [r for r in p2_results if not r["overlay"]]
    strong_v5b = [r for r in all_v5b if r["verdict"] == "STRONG"]
    for feat, key, fmt in [
        ("Trailing stop", "trailing_atr_mult", ".2f"),
        ("Breakeven", "breakeven_trigger_pct", ".3f"),
        ("Max holding", "max_holding_bars", "d"),
        ("ATR SL", "atr_sl_mult", ".2f"),
        ("ATR TP", "atr_tp_mult", ".2f"),
    ]:
        used = [r for r in strong_v5b if r.get(key, 0) > 0.01]
        pct = 100 * len(used) / max(len(strong_v5b), 1)
        if used:
            avg = np.mean([r[key] for r in used])
            if fmt == "d":
                avg = int(round(avg))
            lines.append(f"- **{feat}** : {len(used)}/{len(strong_v5b)} STRONG ({pct:.0f}%), avg={avg:{fmt}}")
        else:
            lines.append(f"- **{feat}** : 0/{len(strong_v5b)} STRONG (0%)")
    lines.append("")

    # Risk grid comparison
    lines.append("## Risk Grid Comparison")
    lines.append("")
    lines.append("| Symbol | Strategy | TF | flat | r0.5 | r1.0 | r2.0 | Best |")
    lines.append("|--------|----------|-----|------|------|------|------|------|")
    # Group by combo
    combos = {}
    for r in p2_results:
        if r["overlay"]:
            continue
        key = (r["symbol"], r["strategy"], r["timeframe"])
        combos.setdefault(key, {})[r["risk_key"]] = r["ho_sharpe"]
    for (sym, sname, tf), sharpes in sorted(combos.items(), key=lambda x: max(x[1].values()), reverse=True)[:20]:
        vals = {k: f"{v:.3f}" for k, v in sharpes.items()}
        best_key = max(sharpes, key=sharpes.get)
        lines.append(
            f"| {sym} | {sname} | {tf} "
            f"| {vals.get('flat', '-')} | {vals.get('r0.5', '-')} "
            f"| {vals.get('r1.0', '-')} | {vals.get('r2.0', '-')} | {best_key} |"
        )
    lines.append("")

    # Seed robustness
    lines.append("## Seed Robustness (multi-seed std)")
    lines.append("")
    robust = sorted([r for r in p2_results if r["verdict"] == "STRONG" and not r["overlay"]],
                    key=lambda x: x["seed_sharpe_std"])
    lines.append("| Symbol | Strategy | TF | Risk | Sharpe | Seed_std | Robust? |")
    lines.append("|--------|----------|-----|------|--------|----------|---------|")
    for r in robust[:20]:
        rob = "Y" if r["seed_sharpe_std"] < 0.3 else "N"
        lines.append(
            f"| {r['symbol']} | {r['strategy']} | {r['timeframe']} "
            f"| {r['risk_key']} | {r['ho_sharpe']:.3f} | {r['seed_sharpe_std']:.3f} | {rob} |"
        )
    lines.append("")

    # Correlation matrix
    corr, labels = compute_correlation_matrix(p2_results)
    if corr is not None and len(labels) > 1:
        lines.append("## Correlation Matrix (STRONG baseline returns)")
        lines.append("")
        # Find low-correlation pairs for portfolio
        pairs = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs.append((labels[i], labels[j], corr[i, j]))
        pairs.sort(key=lambda x: x[2])

        lines.append("### Lowest correlation pairs (best for portfolio)")
        lines.append("")
        lines.append("| Combo A | Combo B | Correlation |")
        lines.append("|---------|---------|-------------|")
        for a, b, c in pairs[:15]:
            lines.append(f"| {a} | {b} | {c:.3f} |")
        lines.append("")

        lines.append("### Highest correlation pairs (redundant)")
        lines.append("")
        lines.append("| Combo A | Combo B | Correlation |")
        lines.append("|---------|---------|-------------|")
        for a, b, c in pairs[-10:]:
            lines.append(f"| {a} | {b} | {c:.3f} |")
        lines.append("")

        # Avg correlation per combo
        lines.append("### Avg correlation per combo (lower = better diversifier)")
        lines.append("")
        lines.append("| Combo | Avg Corr | Max Corr |")
        lines.append("|-------|----------|----------|")
        for i, label in enumerate(labels):
            other_corrs = [corr[i, j] for j in range(len(labels)) if j != i]
            if other_corrs:
                lines.append(f"| {label} | {np.mean(other_corrs):.3f} | {np.max(other_corrs):.3f} |")
        lines.append("")

    # Survivor pool
    all_surv = sorted([r for r in p2_results if r["verdict"] in ("STRONG", "WEAK") and r["ho_trades"] >= 3],
                      key=lambda x: x["ho_sharpe"], reverse=True)
    lines.append(f"## Pool survivants ({len(all_surv)} combos)")
    lines.append("")
    lines.append("| # | Sym | Strategy | TF | Risk | Ov | Sharpe | Ret | DD | Trail | Seed_std | Verdict |")
    lines.append("|---|-----|----------|-----|------|-----|--------|-----|-----|-------|----------|---------|")
    for i, r in enumerate(all_surv[:50]):
        ov = "Y" if r["overlay"] else "-"
        lines.append(
            f"| {i+1} | {r['symbol'][:3]} | {r['strategy'][:16]} | {r['timeframe']} "
            f"| {r['risk_key']} | {ov} | {r['ho_sharpe']:.3f} | {r['ho_return']*100:.1f}% "
            f"| {r['ho_dd']*100:.1f}% | {r['trailing_atr_mult']:.2f} "
            f"| {r['seed_sharpe_std']:.3f} | {r['verdict']} |"
        )
    lines.append("")
    lines.append("---")
    lines.append(f"*Genere le {datetime.now().strftime('%d %B %Y %H:%M')}*")

    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path("docs/results/19_diagnostic_v5b.md").write_text("\n".join(lines))
    logger.info(f"Report: docs/results/19_diagnostic_v5b.md")


# ─── Main ────────────────────────────────────────────────

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC V5b — Full Enrichment (trailing, breakeven, max_hold, multi-seed, risk grid, correlation)")
    logger.info("=" * 60)

    all_data = load_all_data()

    # Phase 1
    p1_results = run_phase1(all_data)
    p1_survivors = filter_phase1(p1_results)

    p1_sorted = sorted(p1_results, key=lambda x: max(x["p1_sharpe"], x["p1_ov_sharpe"]), reverse=True)
    logger.info("  Phase 1 Top 10:")
    for r in p1_sorted[:10]:
        best = max(r["p1_sharpe"], r["p1_ov_sharpe"])
        logger.info(f"    {r['symbol']}/{r['strategy']}/{r['timeframe']} -> Sharpe={best:.3f}")

    # Phase 2
    p2_results = run_phase2(p1_survivors, all_data)

    elapsed = (time.time() - t0) / 60
    logger.info(f"\nTotal: {elapsed:.1f} min")

    # Save JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)
    save_data = {
        "phase1": p1_results,
        "phase2": [{k: v for k, v in r.items() if k not in ("ho_equity", "ho_returns")} for r in p2_results],
    }
    with open(f"results/diagnostic_v5b_{ts}.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Report
    generate_report(p1_results, p2_results, p1_survivors, elapsed)

    # Summary
    n_strong = sum(1 for r in p2_results if r["verdict"] == "STRONG")
    n_weak = sum(1 for r in p2_results if r["verdict"] == "WEAK")
    logger.info(f"\n{'='*60}")
    logger.info(f"RESUME: {n_strong} STRONG + {n_weak} WEAK survivants")
    for risk_key in ["flat", "r1.0"]:
        group = [r for r in p2_results if r["risk_key"] == risk_key and not r["overlay"]]
        if group:
            logger.info(f"  {risk_key} baseline: Avg Sharpe={np.mean([r['ho_sharpe'] for r in group]):.3f}")
    # Trailing usage
    strong_base = [r for r in p2_results if r["verdict"] == "STRONG" and not r["overlay"]]
    trail_used = sum(1 for r in strong_base if r.get("trailing_atr_mult", 0) > 0.01)
    logger.info(f"  Trailing stop used by {trail_used}/{len(strong_base)} STRONG combos")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
