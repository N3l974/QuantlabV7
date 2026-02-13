#!/usr/bin/env python3
"""
Diagnostic Temporel — Analyse du paradoxe Full Sharpe négatif / HO Sharpe positif.

Analyses:
  1. Performance par année (rolling annual Sharpe)
  2. Performance par semestre
  3. Regime detection: bull/bear/sideways periods
  4. Walk-forward stability: Sharpe par fenêtre OOS
  5. MC sur holdout returns seulement (vs full data)
  6. Concentration risk analysis (ETH dominance)

Output:
  results/diagnostic_temporal_{timestamp}.json
  docs/results/12_diagnostic_temporal.md
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
from engine.metrics import (
    compute_all_metrics,
    max_drawdown,
    returns_from_equity,
    sharpe_ratio,
    total_return,
)
from engine.walk_forward import WalkForwardConfig, run_walk_forward
from strategies.registry import get_strategy

# ── Config ──
CUTOFF_DATE = "2025-02-01"
N_SEEDS = 3
INITIAL_CAPITAL = 10_000.0
N_MONTE_CARLO = 1000

TF_CONFIGS = {
    "4h": {"training_window": "6M", "reoptim_frequency": "3M", "n_optim_trials": 80},
    "1d": {"training_window": "1Y",  "reoptim_frequency": "3M", "n_optim_trials": 80},
}

SURVIVORS = [
    {"symbol": "ETHUSDT",  "strategy": "breakout_regime",     "timeframe": "4h"},
    {"symbol": "ETHUSDT",  "strategy": "trend_multi_factor",  "timeframe": "1d"},
    {"symbol": "ETHUSDT",  "strategy": "supertrend_adx",      "timeframe": "4h"},
    {"symbol": "ETHUSDT",  "strategy": "trend_multi_factor",  "timeframe": "4h"},
    {"symbol": "SOLUSDT",  "strategy": "breakout_regime",     "timeframe": "1d"},
    {"symbol": "BTCUSDT",  "strategy": "trend_multi_factor",  "timeframe": "1d"},
    {"symbol": "BTCUSDT",  "strategy": "supertrend_adx",      "timeframe": "4h"},
    {"symbol": "ETHUSDT",  "strategy": "supertrend",          "timeframe": "4h"},
]

# Portfolio weights (ho_sharpe_weighted from V3)
PORTFOLIO_WEIGHTS = np.array([
    0.2450, 0.2042, 0.1820, 0.0473, 0.0038, 0.0840, 0.0508, 0.1828
])


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


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 1: Per-combo WF with OOS period tracking
# ═══════════════════════════════════════════════════════════════
def run_combo_with_periods(combo, data_by_symbol, settings, risk):
    """Run WF on full data and extract per-OOS-period metrics."""
    symbol = combo["symbol"]
    strat_name = combo["strategy"]
    tf = combo["timeframe"]
    tf_cfg = TF_CONFIGS[tf]

    data = data_by_symbol.get(symbol, {}).get(tf)
    if data is None:
        return None

    strategy = get_strategy(strat_name)
    commission = settings["engine"]["commission_rate"]
    slippage = settings["engine"]["slippage_rate"]

    seed = 42  # Single seed for diagnostic (speed)
    wf_config = WalkForwardConfig(
        strategy=strategy, data=data, timeframe=tf,
        reoptim_frequency=tf_cfg["reoptim_frequency"],
        training_window=tf_cfg["training_window"],
        param_bounds_scale=1.0, optim_metric="sharpe",
        n_optim_trials=tf_cfg["n_optim_trials"],
        commission=commission, slippage=slippage,
        risk=risk, seed=seed,
    )

    try:
        result = run_walk_forward(wf_config)
    except Exception as e:
        logger.error(f"  WF failed: {e}")
        return None

    # Extract full equity and returns
    full_eq = result.oos_equity
    full_ret = returns_from_equity(full_eq)

    # Get OOS period boundaries from the walk-forward result
    oos_periods = result.oos_period_indices if hasattr(result, 'oos_period_indices') else []

    return {
        "combo": combo,
        "full_equity": full_eq,
        "full_returns": full_ret,
        "full_metrics": result.metrics,
        "n_oos_periods": result.n_oos_periods if hasattr(result, 'n_oos_periods') else 0,
    }


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 2: Temporal breakdown (by year, semester)
# ═══════════════════════════════════════════════════════════════
def analyze_temporal_breakdown(returns, n_periods_per_year):
    """Break returns into yearly and semi-annual chunks, compute Sharpe for each."""
    n = len(returns)
    periods_per_semester = n_periods_per_year // 2

    # By year
    yearly = []
    for start in range(0, n, n_periods_per_year):
        end = min(start + n_periods_per_year, n)
        chunk = returns[start:end]
        if len(chunk) < 50:
            continue
        eq = INITIAL_CAPITAL * np.cumprod(1 + chunk)
        eq = np.insert(eq, 0, INITIAL_CAPITAL)
        yearly.append({
            "period_idx": len(yearly),
            "n_bars": len(chunk),
            "sharpe": float(sharpe_ratio(chunk, "4h")),
            "return": float(total_return(eq)),
            "max_dd": float(max_drawdown(eq)),
        })

    # By semester
    semesterly = []
    for start in range(0, n, periods_per_semester):
        end = min(start + periods_per_semester, n)
        chunk = returns[start:end]
        if len(chunk) < 30:
            continue
        eq = INITIAL_CAPITAL * np.cumprod(1 + chunk)
        eq = np.insert(eq, 0, INITIAL_CAPITAL)
        semesterly.append({
            "period_idx": len(semesterly),
            "n_bars": len(chunk),
            "sharpe": float(sharpe_ratio(chunk, "4h")),
            "return": float(total_return(eq)),
            "max_dd": float(max_drawdown(eq)),
        })

    return {"yearly": yearly, "semesterly": semesterly}


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 3: Rolling Sharpe
# ═══════════════════════════════════════════════════════════════
def rolling_sharpe(returns, window=500, step=100, tf="4h"):
    """Compute rolling Sharpe ratio."""
    n = len(returns)
    results = []
    for start in range(0, n - window, step):
        chunk = returns[start:start + window]
        s = sharpe_ratio(chunk, tf)
        results.append({
            "start_idx": start,
            "end_idx": start + window,
            "pct_through": round(start / n * 100, 1),
            "sharpe": round(float(s), 4),
        })
    return results


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 4: Monte Carlo on holdout vs full
# ═══════════════════════════════════════════════════════════════
def mc_comparison(full_returns, ho_returns, n_sims=1000, n_days=365):
    """Compare MC on full data vs holdout data only."""
    results = {}
    for label, rets in [("full", full_returns), ("holdout", ho_returns)]:
        n = len(rets)
        if n < 10:
            results[label] = {"median_return": 0, "p5_return": 0, "p95_return": 0}
            continue

        final_returns = np.zeros(n_sims)
        max_dds = np.zeros(n_sims)
        for sim in range(n_sims):
            idx = np.random.randint(0, n, size=min(n_days, n * 2))
            sim_rets = rets[idx]
            sim_eq = INITIAL_CAPITAL * np.cumprod(1 + sim_rets)
            sim_eq = np.insert(sim_eq, 0, INITIAL_CAPITAL)
            final_returns[sim] = total_return(sim_eq)
            max_dds[sim] = max_drawdown(sim_eq)

        results[label] = {
            "median_return": round(float(np.median(final_returns)), 4),
            "mean_return": round(float(np.mean(final_returns)), 4),
            "p5_return": round(float(np.percentile(final_returns, 5)), 4),
            "p25_return": round(float(np.percentile(final_returns, 25)), 4),
            "p75_return": round(float(np.percentile(final_returns, 75)), 4),
            "p95_return": round(float(np.percentile(final_returns, 95)), 4),
            "median_dd": round(float(np.median(max_dds)), 4),
            "p5_dd": round(float(np.percentile(max_dds, 5)), 4),
            "prob_positive": round(float(np.mean(final_returns > 0)), 4),
        }
    return results


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 5: Concentration risk
# ═══════════════════════════════════════════════════════════════
def analyze_concentration(combo_results, weights):
    """Analyze concentration by symbol and timeframe."""
    symbol_weights = {}
    tf_weights = {}
    strat_weights = {}

    for i, r in enumerate(combo_results):
        c = r["combo"]
        w = weights[i]
        symbol_weights[c["symbol"]] = symbol_weights.get(c["symbol"], 0) + w
        tf_weights[c["timeframe"]] = tf_weights.get(c["timeframe"], 0) + w
        strat_weights[c["strategy"]] = strat_weights.get(c["strategy"], 0) + w

    # HHI (Herfindahl-Hirschman Index) for concentration
    hhi_symbol = sum(v**2 for v in symbol_weights.values())
    hhi_tf = sum(v**2 for v in tf_weights.values())
    hhi_strat = sum(v**2 for v in strat_weights.values())

    return {
        "by_symbol": {k: round(v, 4) for k, v in sorted(symbol_weights.items(), key=lambda x: -x[1])},
        "by_timeframe": {k: round(v, 4) for k, v in sorted(tf_weights.items(), key=lambda x: -x[1])},
        "by_strategy": {k: round(v, 4) for k, v in sorted(strat_weights.items(), key=lambda x: -x[1])},
        "hhi_symbol": round(hhi_symbol, 4),
        "hhi_timeframe": round(hhi_tf, 4),
        "hhi_strategy": round(hhi_strat, 4),
        "effective_n_symbols": round(1 / max(hhi_symbol, 0.01), 2),
        "effective_n_strategies": round(1 / max(hhi_strat, 0.01), 2),
    }


# ═══════════════════════════════════════════════════════════════
# ANALYSIS 6: What-if without ETH concentration
# ═══════════════════════════════════════════════════════════════
def what_if_equal_symbol_weight(combo_results):
    """Rebalance to equal weight per symbol, then proportional within."""
    symbols = {}
    for i, r in enumerate(combo_results):
        s = r["combo"]["symbol"]
        if s not in symbols:
            symbols[s] = []
        symbols[s].append(i)

    n_symbols = len(symbols)
    new_weights = np.zeros(len(combo_results))
    for s, indices in symbols.items():
        per_symbol = 1.0 / n_symbols
        per_combo = per_symbol / len(indices)
        for idx in indices:
            new_weights[idx] = per_combo

    return new_weights


# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════
def generate_report(combo_analyses, portfolio_temporal, mc_comp, concentration,
                    rolling_sharpes, what_if_metrics):
    """Generate markdown diagnostic report."""
    md = []
    md.append("# Diagnostic Temporel — Portfolio V3")
    md.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    md.append(f"**Objectif** : Comprendre le paradoxe Full Sharpe negatif / HO Sharpe positif")
    md.append("")
    md.append("---")
    md.append("")

    # ── 1. Per-combo temporal breakdown ──
    md.append("## 1. Performance par combo — Breakdown temporel")
    md.append("")

    for ca in combo_analyses:
        c = ca["combo"]
        label = f"{c['symbol']}/{c['strategy']}/{c['timeframe']}"
        md.append(f"### {label}")
        md.append(f"- Full Sharpe: {ca['full_metrics'].get('sharpe', 0):.3f}")
        md.append("")

        if ca["temporal"]["yearly"]:
            md.append("**Par annee** :")
            md.append("")
            md.append("| Annee | Sharpe | Return | Max DD |")
            md.append("|-------|--------|--------|--------|")
            for y in ca["temporal"]["yearly"]:
                md.append(f"| Y{y['period_idx']+1} | {y['sharpe']:.2f} | {y['return']:+.1%} | {y['max_dd']:.1%} |")
            md.append("")

    # ── 2. Portfolio temporal ──
    md.append("## 2. Portfolio ho_sharpe_weighted — Breakdown temporel")
    md.append("")

    if portfolio_temporal["yearly"]:
        md.append("### Par annee")
        md.append("")
        md.append("| Annee | Sharpe | Return | Max DD |")
        md.append("|-------|--------|--------|--------|")
        for y in portfolio_temporal["yearly"]:
            md.append(f"| Y{y['period_idx']+1} | {y['sharpe']:.2f} | {y['return']:+.1%} | {y['max_dd']:.1%} |")
        md.append("")

    if portfolio_temporal["semesterly"]:
        md.append("### Par semestre")
        md.append("")
        md.append("| Semestre | Sharpe | Return | Max DD |")
        md.append("|----------|--------|--------|--------|")
        for s in portfolio_temporal["semesterly"]:
            md.append(f"| S{s['period_idx']+1} | {s['sharpe']:.2f} | {s['return']:+.1%} | {s['max_dd']:.1%} |")
        md.append("")

    # ── 3. Rolling Sharpe ──
    md.append("## 3. Rolling Sharpe (fenetre 500 bars)")
    md.append("")
    md.append("| Position (%) | Sharpe |")
    md.append("|-------------|--------|")
    for rs in rolling_sharpes:
        bar = "+" * max(0, int(rs["sharpe"] * 5)) + "-" * max(0, int(-rs["sharpe"] * 5))
        md.append(f"| {rs['pct_through']:.0f}% | {rs['sharpe']:+.2f} {bar} |")
    md.append("")

    # ── 4. MC comparison ──
    md.append("## 4. Monte Carlo : Full data vs Holdout only")
    md.append("")
    md.append("| Metrique | Full Data MC | Holdout MC |")
    md.append("|----------|-------------|------------|")
    for key in ["median_return", "mean_return", "p5_return", "p95_return", "median_dd", "prob_positive"]:
        full_val = mc_comp["full"].get(key, 0)
        ho_val = mc_comp["holdout"].get(key, 0)
        if "return" in key or "dd" in key:
            md.append(f"| {key} | {full_val:+.1%} | {ho_val:+.1%} |")
        else:
            md.append(f"| {key} | {full_val:.1%} | {ho_val:.1%} |")
    md.append("")

    # ── 5. Concentration ──
    md.append("## 5. Analyse de concentration")
    md.append("")
    md.append("### Par symbol")
    md.append("")
    md.append("| Symbol | Poids |")
    md.append("|--------|-------|")
    for s, w in concentration["by_symbol"].items():
        md.append(f"| {s} | {w:.1%} |")
    md.append("")
    md.append(f"- **HHI Symbol** : {concentration['hhi_symbol']:.4f} (N effectif = {concentration['effective_n_symbols']:.1f})")
    md.append(f"- **HHI Strategy** : {concentration['hhi_strategy']:.4f} (N effectif = {concentration['effective_n_strategies']:.1f})")
    md.append("")

    # ── 6. What-if ──
    md.append("## 6. What-if : Equal Symbol Weight")
    md.append("")
    md.append("Rebalancement pour donner 1/3 a chaque symbol (ETH, BTC, SOL).")
    md.append("")
    md.append("| Metrique | ho_sharpe_weighted | equal_symbol |")
    md.append("|----------|-------------------|--------------|")
    for key in ["sharpe", "total_return", "max_drawdown", "sortino"]:
        orig = what_if_metrics["original"].get(key, 0)
        alt = what_if_metrics["equal_symbol"].get(key, 0)
        if "return" in key or "drawdown" in key:
            md.append(f"| {key} | {orig:+.2f} | {alt:+.2f} |")
        else:
            md.append(f"| {key} | {orig:.2f} | {alt:.2f} |")
    md.append("")

    # ── 7. Conclusions ──
    md.append("## 7. Conclusions")
    md.append("")

    # Determine if recent performance is better
    yearly = portfolio_temporal.get("yearly", [])
    if len(yearly) >= 2:
        early_sharpes = [y["sharpe"] for y in yearly[:len(yearly)//2]]
        late_sharpes = [y["sharpe"] for y in yearly[len(yearly)//2:]]
        early_avg = np.mean(early_sharpes) if early_sharpes else 0
        late_avg = np.mean(late_sharpes) if late_sharpes else 0

        if late_avg > early_avg:
            md.append(f"- **Tendance positive** : Sharpe moyen passe de {early_avg:.2f} (1ere moitie) a {late_avg:.2f} (2eme moitie)")
        else:
            md.append(f"- **Tendance negative** : Sharpe moyen passe de {early_avg:.2f} (1ere moitie) a {late_avg:.2f} (2eme moitie)")

    ho_mc = mc_comp.get("holdout", {})
    full_mc = mc_comp.get("full", {})
    if ho_mc.get("median_return", 0) > full_mc.get("median_return", 0):
        md.append(f"- **MC Holdout superieur** : median {ho_mc['median_return']:+.1%} vs full {full_mc['median_return']:+.1%}")
        md.append("  → Les returns recents sont meilleurs que la moyenne historique")

    if concentration["hhi_symbol"] > 0.5:
        md.append(f"- **Concentration elevee** : HHI symbol = {concentration['hhi_symbol']:.2f}, N effectif = {concentration['effective_n_symbols']:.1f}")
        md.append("  → Risque de dependance a un seul actif")

    md.append("")
    md.append("---")
    md.append(f"*Genere le {datetime.now().strftime('%d %B %Y')}*")

    return "\n".join(md)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    logger.info("=" * 70)
    logger.info("  DIAGNOSTIC TEMPOREL — PORTFOLIO V3")
    logger.info("=" * 70)

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    settings = load_settings()
    data_by_symbol = load_all_symbols_data(settings)
    risk = _build_risk(settings)
    cutoff_dt = pd.Timestamp(CUTOFF_DATE)

    # ── Step 1: Run WF for each combo ──
    logger.info("\n--- STEP 1: Walk-forward per combo (1 seed) ---")
    combo_analyses = []
    all_full_returns = []
    all_ho_returns = []

    for idx, combo in enumerate(SURVIVORS):
        symbol = combo["symbol"]
        strat_name = combo["strategy"]
        tf = combo["timeframe"]
        logger.info(f"\n[{idx+1}/{len(SURVIVORS)}] {symbol}/{strat_name}/{tf}")

        result = run_combo_with_periods(combo, data_by_symbol, settings, risk)
        if result is None:
            continue

        # Temporal breakdown
        n_per_year = 365 * 6 if tf == "4h" else 365  # 4h = 6 bars/day
        temporal = analyze_temporal_breakdown(result["full_returns"], n_per_year)

        # Split returns at cutoff
        data = data_by_symbol[symbol][tf]
        cutoff_idx = data.index.searchsorted(cutoff_dt)
        # Map to return indices (returns are shorter by 1 from equity)
        n_full = len(result["full_returns"])
        # Approximate: cutoff position in returns
        total_bars = len(data)
        cutoff_pct = cutoff_idx / total_bars
        cutoff_ret_idx = int(cutoff_pct * n_full)

        ho_rets = result["full_returns"][cutoff_ret_idx:]
        full_rets = result["full_returns"]

        analysis = {
            "combo": combo,
            "full_metrics": result["full_metrics"],
            "full_returns": full_rets,
            "ho_returns": ho_rets,
            "temporal": temporal,
        }
        combo_analyses.append(analysis)
        all_full_returns.append(full_rets)
        all_ho_returns.append(ho_rets)

        logger.info(f"  Full Sharpe={result['full_metrics'].get('sharpe', 0):.3f} | "
                     f"Yearly breakdown: {len(temporal['yearly'])} years")

    if not combo_analyses:
        logger.error("No valid combos. Aborting.")
        return

    # ── Step 2: Portfolio-level temporal analysis ──
    logger.info("\n--- STEP 2: Portfolio temporal analysis ---")

    # Align returns and build portfolio
    max_len = max(len(r) for r in all_full_returns)
    aligned = np.zeros((len(combo_analyses), max_len))
    for i, rets in enumerate(all_full_returns):
        aligned[i, :len(rets)] = rets

    port_returns = aligned.T @ PORTFOLIO_WEIGHTS[:len(combo_analyses)]
    n_per_year = 365 * 6  # Use 4h as base
    portfolio_temporal = analyze_temporal_breakdown(port_returns, n_per_year)

    # Rolling Sharpe
    rolling = rolling_sharpe(port_returns, window=500, step=250)

    # ── Step 3: MC comparison ──
    logger.info("\n--- STEP 3: Monte Carlo comparison (full vs holdout) ---")

    # Build holdout portfolio returns
    max_ho_len = max(len(r) for r in all_ho_returns)
    aligned_ho = np.zeros((len(combo_analyses), max_ho_len))
    for i, rets in enumerate(all_ho_returns):
        aligned_ho[i, :len(rets)] = rets
    port_ho_returns = aligned_ho.T @ PORTFOLIO_WEIGHTS[:len(combo_analyses)]

    mc_comp = mc_comparison(port_returns, port_ho_returns, N_MONTE_CARLO)
    logger.info(f"  Full MC median: {mc_comp['full']['median_return']:+.1%}")
    logger.info(f"  HO MC median: {mc_comp['holdout']['median_return']:+.1%}")

    # ── Step 4: Concentration analysis ──
    logger.info("\n--- STEP 4: Concentration analysis ---")
    concentration = analyze_concentration(combo_analyses, PORTFOLIO_WEIGHTS[:len(combo_analyses)])
    logger.info(f"  HHI Symbol: {concentration['hhi_symbol']:.4f} (N eff = {concentration['effective_n_symbols']:.1f})")

    # ── Step 5: What-if equal symbol weight ──
    logger.info("\n--- STEP 5: What-if equal symbol weight ---")
    alt_weights = what_if_equal_symbol_weight(combo_analyses)

    # Original portfolio metrics on full data
    orig_port_eq = INITIAL_CAPITAL * np.cumprod(1 + port_returns)
    orig_port_eq = np.insert(orig_port_eq, 0, INITIAL_CAPITAL)
    orig_metrics = compute_all_metrics(orig_port_eq, "4h")

    # Alternative portfolio
    alt_port_returns = aligned.T @ alt_weights
    alt_port_eq = INITIAL_CAPITAL * np.cumprod(1 + alt_port_returns)
    alt_port_eq = np.insert(alt_port_eq, 0, INITIAL_CAPITAL)
    alt_metrics = compute_all_metrics(alt_port_eq, "4h")

    # Also compute holdout metrics for both
    orig_ho_eq = INITIAL_CAPITAL * np.cumprod(1 + port_ho_returns)
    orig_ho_eq = np.insert(orig_ho_eq, 0, INITIAL_CAPITAL)
    orig_ho_metrics = compute_all_metrics(orig_ho_eq, "4h")

    alt_ho_returns = aligned_ho.T @ alt_weights
    alt_ho_eq = INITIAL_CAPITAL * np.cumprod(1 + alt_ho_returns)
    alt_ho_eq = np.insert(alt_ho_eq, 0, INITIAL_CAPITAL)
    alt_ho_metrics = compute_all_metrics(alt_ho_eq, "4h")

    what_if_metrics = {
        "original": orig_metrics,
        "equal_symbol": alt_metrics,
        "original_ho": orig_ho_metrics,
        "equal_symbol_ho": alt_ho_metrics,
    }

    logger.info(f"  Original full Sharpe: {orig_metrics['sharpe']:.3f} | Alt: {alt_metrics['sharpe']:.3f}")
    logger.info(f"  Original HO Sharpe: {orig_ho_metrics['sharpe']:.3f} | Alt: {alt_ho_metrics['sharpe']:.3f}")

    # ── Step 6: Generate reports ──
    logger.info("\n--- STEP 6: Generating reports ---")

    md_report = generate_report(
        combo_analyses, portfolio_temporal, mc_comp, concentration,
        rolling, what_if_metrics)

    # JSON report
    json_report = {
        "timestamp": timestamp,
        "combos": [
            {
                "combo": ca["combo"],
                "full_sharpe": ca["full_metrics"].get("sharpe", 0),
                "temporal": ca["temporal"],
            }
            for ca in combo_analyses
        ],
        "portfolio_temporal": portfolio_temporal,
        "rolling_sharpe": rolling,
        "mc_comparison": mc_comp,
        "concentration": concentration,
        "what_if": {
            "original_weights": PORTFOLIO_WEIGHTS[:len(combo_analyses)].tolist(),
            "equal_symbol_weights": alt_weights.tolist(),
            "original_full": orig_metrics,
            "equal_symbol_full": alt_metrics,
            "original_ho": orig_ho_metrics,
            "equal_symbol_ho": alt_ho_metrics,
        },
    }

    Path("results").mkdir(exist_ok=True)
    Path("docs/results").mkdir(parents=True, exist_ok=True)

    json_path = f"results/diagnostic_temporal_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)

    md_path = "docs/results/12_diagnostic_temporal.md"
    with open(md_path, "w") as f:
        f.write(md_report)

    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  DIAGNOSTIC TEMPOREL COMPLETE — {elapsed/60:.1f} min")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Saved: {json_path}")
    logger.info(f"  Saved: {md_path}")


if __name__ == "__main__":
    main()
