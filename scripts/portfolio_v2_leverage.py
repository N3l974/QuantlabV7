#!/usr/bin/env python3
"""
Portfolio V2 — Sharpe-Weighted vs Positive-Only + Leverage Testing.

Steps:
  1. Load meta-optimisation V3 profiles
  2. Re-run walk-forward for each combo → get full equity curves
  3. Build portfolios:
     a) sharpe_weighted (all combos, weight ∝ max(sharpe, 0.01))
     b) positive_only  (filter Sharpe < 0, weight ∝ sharpe)
  4. Test leverage 1x, 2x, 3x on each portfolio
  5. Monte Carlo stress tests + projections for every variant
  6. Generate comprehensive report (JSON + TXT + docs/results MD)

Output: results/portfolio_v2_leverage_{timestamp}.json
        results/portfolio_v2_leverage_{timestamp}.txt
        docs/results/06_portfolio_v2_leverage_{date}.md
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.ingestion import load_all_symbols_data, load_settings
from engine.backtester import RiskConfig
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
META_V3_PATH = "results/meta_profiles_v3_20260211_195716.json"
INITIAL_CAPITAL = 10_000.0
N_MONTE_CARLO = 1000
PROJECTION_YEARS = [1, 2, 3, 5]
LEVERAGE_LEVELS = [1.0, 2.0, 3.0]
CONFIDENCE_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]


def load_meta_profiles(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


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


# ═══════════════════════════════════════════════════════════════
# STEP 1: Re-run walk-forward to get equity curves
# ═══════════════════════════════════════════════════════════════
def rerun_walk_forwards(profiles: list[dict], data_by_symbol: dict,
                        settings: dict, risk: RiskConfig) -> list[dict]:
    """Re-run WF for each profile with its optimal meta-params → equity curves."""
    results = []
    for i, p in enumerate(profiles):
        symbol = p["symbol"]
        strategy_name = p["strategy"]
        timeframe = p["timeframe"]

        logger.info(f"[{i+1}/{len(profiles)}] Re-running WF: {symbol}/{strategy_name}/{timeframe}")

        data = data_by_symbol.get(symbol, {}).get(timeframe)
        if data is None:
            logger.error(f"  No data for {symbol}/{timeframe}")
            continue

        strategy = get_strategy(strategy_name)
        wf_config = WalkForwardConfig(
            strategy=strategy,
            data=data,
            timeframe=timeframe,
            reoptim_frequency=p["reoptim_frequency"],
            training_window=p["training_window"],
            param_bounds_scale=p["param_bounds_scale"],
            optim_metric=p["optim_metric"],
            n_optim_trials=p["n_optim_trials"],
            commission=settings["engine"]["commission_rate"],
            slippage=settings["engine"]["slippage_rate"],
            risk=risk,
        )

        try:
            wf_result = run_walk_forward(wf_config)
        except Exception as e:
            logger.error(f"  WF failed: {e}")
            continue

        equity = wf_result.oos_equity
        rets = returns_from_equity(equity) if len(equity) > 1 else np.array([0.0])
        metrics = wf_result.metrics

        results.append({
            "profile": p,
            "equity": equity,
            "returns": rets,
            "metrics": metrics,
            "n_bars": len(equity),
            "timeframe": timeframe,
        })

        logger.info(f"  Sharpe={metrics['sharpe']:.2f} | Return={metrics['total_return']:.1%} | "
                    f"DD={metrics['max_drawdown']:.1%} | PF={metrics['profit_factor']:.2f} | "
                    f"Trades={metrics['n_trades']}")

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 2: Portfolio construction (sharpe_weighted vs positive_only)
# ═══════════════════════════════════════════════════════════════
def build_portfolio_from_returns(aligned_returns: np.ndarray, weights: np.ndarray,
                                 leverage: float = 1.0) -> dict:
    """Build a single portfolio from aligned returns matrix, weights, and leverage."""
    port_returns = (aligned_returns.T @ weights) * leverage
    port_equity = INITIAL_CAPITAL * np.cumprod(1 + port_returns)
    port_equity = np.insert(port_equity, 0, INITIAL_CAPITAL)
    metrics = compute_all_metrics(port_equity, "1d")
    return {
        "weights": weights.tolist(),
        "leverage": leverage,
        "returns": port_returns,
        "equity": port_equity,
        "metrics": metrics,
    }


def build_all_portfolios(combo_results: list[dict]) -> dict:
    """Build sharpe_weighted and positive_only portfolios at multiple leverage levels."""
    n = len(combo_results)
    if n == 0:
        return {}

    # Align returns to same length
    max_len = max(len(r["returns"]) for r in combo_results)
    aligned_all = np.zeros((n, max_len))
    for i, r in enumerate(combo_results):
        rets = r["returns"]
        aligned_all[i, :len(rets)] = rets

    # Identify positive-Sharpe combos
    positive_mask = [r["metrics"]["sharpe"] > 0 for r in combo_results]
    positive_indices = [i for i, m in enumerate(positive_mask) if m]
    n_positive = len(positive_indices)

    logger.info(f"  Total combos: {n} | Positive Sharpe: {n_positive} | "
                f"Negative Sharpe: {n - n_positive}")

    # ── Sharpe-weighted (all combos) ──
    sharpes_all = np.array([max(r["metrics"]["sharpe"], 0.01) for r in combo_results])
    sw_weights = sharpes_all / sharpes_all.sum()

    # ── Positive-only (filter Sharpe < 0) ──
    if n_positive > 0:
        sharpes_pos = np.array([combo_results[i]["metrics"]["sharpe"] for i in positive_indices])
        po_weights_raw = sharpes_pos / sharpes_pos.sum()
        aligned_pos = aligned_all[positive_indices]
    else:
        logger.warning("  No positive-Sharpe combos! Falling back to equal weight.")
        po_weights_raw = np.ones(n) / n
        aligned_pos = aligned_all
        positive_indices = list(range(n))

    # ── Equal weight (all combos, for reference) ──
    eq_weights = np.ones(n) / n

    # ── Risk parity (inverse DD, all combos) ──
    inv_dd = np.array([1.0 / max(abs(r["metrics"]["max_drawdown"]), 0.01) for r in combo_results])
    rp_weights = inv_dd / inv_dd.sum()

    portfolios = {}

    for leverage in LEVERAGE_LEVELS:
        lev_suffix = f"_{leverage:.0f}x" if leverage != 1.0 else ""

        # Sharpe-weighted (all combos)
        portfolios[f"sharpe_weighted{lev_suffix}"] = build_portfolio_from_returns(
            aligned_all, sw_weights, leverage)

        # Positive-only
        portfolios[f"positive_only{lev_suffix}"] = build_portfolio_from_returns(
            aligned_pos, po_weights_raw, leverage)

        # Equal weight (reference, only at 1x)
        if leverage == 1.0:
            portfolios["equal_weight"] = build_portfolio_from_returns(
                aligned_all, eq_weights, leverage)
            portfolios["risk_parity"] = build_portfolio_from_returns(
                aligned_all, rp_weights, leverage)

    return portfolios, positive_indices


# ═══════════════════════════════════════════════════════════════
# STEP 3: Monte Carlo Stress Tests
# ═══════════════════════════════════════════════════════════════
def monte_carlo_bootstrap(returns: np.ndarray, n_sims: int = 1000,
                          n_days: int = 365) -> dict:
    """Bootstrap resampling: draw random daily returns with replacement."""
    n = len(returns)
    final_returns = np.zeros(n_sims)
    max_dds = np.zeros(n_sims)
    sharpes = np.zeros(n_sims)

    for sim in range(n_sims):
        idx = np.random.randint(0, n, size=n_days)
        sim_returns = returns[idx]
        sim_equity = INITIAL_CAPITAL * np.cumprod(1 + sim_returns)
        sim_equity = np.insert(sim_equity, 0, INITIAL_CAPITAL)

        final_returns[sim] = total_return(sim_equity)
        max_dds[sim] = max_drawdown(sim_equity)
        sharpes[sim] = sharpe_ratio(sim_returns, "1d")

    return {
        "final_returns": final_returns,
        "max_drawdowns": max_dds,
        "sharpes": sharpes,
    }


def monte_carlo_block_bootstrap(returns: np.ndarray, n_sims: int = 1000,
                                n_days: int = 365, block_size: int = 20) -> dict:
    """Block bootstrap: preserves autocorrelation by sampling blocks."""
    n = len(returns)
    final_returns = np.zeros(n_sims)
    max_dds = np.zeros(n_sims)

    for sim in range(n_sims):
        sim_returns = []
        while len(sim_returns) < n_days:
            start = np.random.randint(0, max(1, n - block_size))
            block = returns[start:start + block_size]
            sim_returns.extend(block)
        sim_returns = np.array(sim_returns[:n_days])

        sim_equity = INITIAL_CAPITAL * np.cumprod(1 + sim_returns)
        sim_equity = np.insert(sim_equity, 0, INITIAL_CAPITAL)

        final_returns[sim] = total_return(sim_equity)
        max_dds[sim] = max_drawdown(sim_equity)

    return {"final_returns": final_returns, "max_drawdowns": max_dds}


def monte_carlo_ruin_probability(returns: np.ndarray, n_sims: int = 1000,
                                 n_days: int = 365, ruin_threshold: float = 0.5) -> float:
    """Probability of losing more than ruin_threshold of capital."""
    n = len(returns)
    ruin_count = 0

    for sim in range(n_sims):
        idx = np.random.randint(0, n, size=n_days)
        sim_returns = returns[idx]
        sim_equity = INITIAL_CAPITAL * np.cumprod(1 + sim_returns)
        if np.min(sim_equity) < INITIAL_CAPITAL * (1 - ruin_threshold):
            ruin_count += 1

    return ruin_count / n_sims


def run_all_stress_tests(returns: np.ndarray, name: str) -> dict:
    """Run all Monte Carlo stress tests on a return series."""
    logger.info(f"  Running stress tests for {name} ({N_MONTE_CARLO} sims)...")

    bootstrap = monte_carlo_bootstrap(returns, N_MONTE_CARLO, 365)
    block_boot = monte_carlo_block_bootstrap(returns, N_MONTE_CARLO, 365)
    ruin_prob = monte_carlo_ruin_probability(returns, N_MONTE_CARLO, 365, 0.5)

    # Multi-year projections
    projections = {}
    for years in PROJECTION_YEARS:
        n_days = int(years * 365)
        proj = monte_carlo_bootstrap(returns, N_MONTE_CARLO, n_days)
        percentiles = {}
        for pct in CONFIDENCE_LEVELS:
            pct_label = f"p{int(pct*100)}"
            percentiles[f"{pct_label}_return"] = float(np.percentile(proj["final_returns"], pct * 100))
            percentiles[f"{pct_label}_dd"] = float(np.percentile(proj["max_drawdowns"], pct * 100))
        projections[f"{years}Y"] = {
            "median_return": float(np.median(proj["final_returns"])),
            "mean_return": float(np.mean(proj["final_returns"])),
            "std_return": float(np.std(proj["final_returns"])),
            "median_dd": float(np.median(proj["max_drawdowns"])),
            "prob_positive": float(np.mean(proj["final_returns"] > 0)),
            "prob_gt_10pct": float(np.mean(proj["final_returns"] > 0.10)),
            "prob_gt_20pct": float(np.mean(proj["final_returns"] > 0.20)),
            **percentiles,
        }

    return {
        "bootstrap_1Y": {
            "median_return": float(np.median(bootstrap["final_returns"])),
            "mean_return": float(np.mean(bootstrap["final_returns"])),
            "std_return": float(np.std(bootstrap["final_returns"])),
            "p5_return": float(np.percentile(bootstrap["final_returns"], 5)),
            "p25_return": float(np.percentile(bootstrap["final_returns"], 25)),
            "p75_return": float(np.percentile(bootstrap["final_returns"], 75)),
            "p95_return": float(np.percentile(bootstrap["final_returns"], 95)),
            "median_dd": float(np.median(bootstrap["max_drawdowns"])),
            "p5_dd": float(np.percentile(bootstrap["max_drawdowns"], 5)),
            "median_sharpe": float(np.median(bootstrap["sharpes"])),
        },
        "block_bootstrap_1Y": {
            "median_return": float(np.median(block_boot["final_returns"])),
            "median_dd": float(np.median(block_boot["max_drawdowns"])),
            "p5_dd": float(np.percentile(block_boot["max_drawdowns"], 5)),
        },
        "ruin_probability_50pct": ruin_prob,
        "projections": projections,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 4: Generate Report
# ═══════════════════════════════════════════════════════════════
def generate_text_report(combo_results: list[dict], portfolios: dict,
                         stress_results: dict, positive_indices: list[int]) -> str:
    """Generate comprehensive text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  QUANTLAB V7 — PORTFOLIO V2: SHARPE-WEIGHTED vs POSITIVE-ONLY + LEVERAGE")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # ── 1. Individual Combos ──
    lines.append("\n" + "─" * 80)
    lines.append("  1. INDIVIDUAL STRATEGY PERFORMANCE (Walk-Forward OOS)")
    lines.append("─" * 80)
    lines.append(f"  {'Symbol':<10} {'Strategy':<25} {'Sharpe':>7} {'Return':>8} {'DD':>8} "
                 f"{'PF':>6} {'WR':>6} {'Trades':>7} {'Filter':>8}")
    lines.append("  " + "-" * 85)

    for i, r in enumerate(combo_results):
        p = r["profile"]
        m = r["metrics"]
        in_positive = "✓ IN" if i in positive_indices else "✗ OUT"
        lines.append(
            f"  {p['symbol']:<10} {p['strategy']:<25} {m['sharpe']:>7.2f} "
            f"{m['total_return']:>7.1%} {m['max_drawdown']:>7.1%} "
            f"{m['profit_factor']:>6.2f} {m['win_rate']:>5.1%} {m['n_trades']:>7} "
            f"{in_positive:>8}"
        )

    n_pos = len(positive_indices)
    n_neg = len(combo_results) - n_pos
    lines.append(f"\n  Positive Sharpe: {n_pos} combos | Negative Sharpe: {n_neg} combos (filtered out)")

    # ── 2. Portfolio Comparison (1x) ──
    lines.append("\n" + "─" * 80)
    lines.append("  2. PORTFOLIO COMPARISON (1x Leverage)")
    lines.append("─" * 80)
    lines.append(f"  {'Portfolio':<20} {'Sharpe':>7} {'Return':>8} {'DD':>8} {'Sortino':>8}")
    lines.append("  " + "-" * 55)

    base_portfolios = ["equal_weight", "risk_parity", "sharpe_weighted", "positive_only"]
    for name in base_portfolios:
        if name in portfolios:
            m = portfolios[name]["metrics"]
            lines.append(
                f"  {name:<20} {m['sharpe']:>7.2f} {m['total_return']:>7.1%} "
                f"{m['max_drawdown']:>7.1%} {m['sortino']:>8.2f}"
            )

    # Weights for sharpe_weighted
    lines.append(f"\n  Sharpe-Weighted Allocation (all {len(combo_results)} combos):")
    sw = portfolios.get("sharpe_weighted", {})
    if sw:
        for i, r in enumerate(combo_results):
            p = r["profile"]
            w = sw["weights"][i]
            lines.append(f"    {p['symbol']}/{p['strategy']:<35} {w:>7.1%}")

    # Weights for positive_only
    lines.append(f"\n  Positive-Only Allocation ({n_pos} combos, Sharpe > 0):")
    po = portfolios.get("positive_only", {})
    if po:
        for j, idx in enumerate(positive_indices):
            p = combo_results[idx]["profile"]
            w = po["weights"][j]
            lines.append(f"    {p['symbol']}/{p['strategy']:<35} {w:>7.1%}")

    # ── 3. Leverage Comparison ──
    lines.append("\n" + "─" * 80)
    lines.append("  3. LEVERAGE COMPARISON")
    lines.append("─" * 80)
    lines.append(f"  {'Portfolio':<25} {'Lev':>4} {'Sharpe':>7} {'Return':>8} {'Ret/yr':>8} "
                 f"{'DD':>8} {'Sortino':>8}")
    lines.append("  " + "-" * 70)

    for base_name in ["sharpe_weighted", "positive_only"]:
        for lev in LEVERAGE_LEVELS:
            suffix = f"_{lev:.0f}x" if lev != 1.0 else ""
            key = f"{base_name}{suffix}"
            if key in portfolios:
                m = portfolios[key]["metrics"]
                n_years = max(1, m.get("n_periods", 365) / 365.25)
                ann_ret = (1 + m["total_return"]) ** (1 / n_years) - 1
                lines.append(
                    f"  {base_name:<25} {lev:>3.0f}x {m['sharpe']:>7.2f} "
                    f"{m['total_return']:>7.1%} {ann_ret:>7.1%} "
                    f"{m['max_drawdown']:>7.1%} {m['sortino']:>8.2f}"
                )
        lines.append("")

    # ── 4. Monte Carlo Stress Tests ──
    lines.append("─" * 80)
    lines.append("  4. MONTE CARLO STRESS TESTS")
    lines.append("─" * 80)

    # Show stress tests for key portfolios
    key_portfolios = []
    for base_name in ["sharpe_weighted", "positive_only"]:
        for lev in LEVERAGE_LEVELS:
            suffix = f"_{lev:.0f}x" if lev != 1.0 else ""
            key = f"{base_name}{suffix}"
            if key in stress_results:
                key_portfolios.append(key)

    lines.append(f"\n  {'Portfolio':<25} {'Med Ret':>8} {'P5 Ret':>8} {'P95 Ret':>9} "
                 f"{'Med DD':>8} {'Worst DD':>9} {'P(ruin)':>8}")
    lines.append("  " + "-" * 80)

    for key in key_portfolios:
        st = stress_results[key]
        b = st["bootstrap_1Y"]
        lines.append(
            f"  {key:<25} {b['median_return']:>+7.1%} {b['p5_return']:>+7.1%} "
            f"{b['p95_return']:>+8.1%} {b['median_dd']:>7.1%} "
            f"{b['p5_dd']:>8.1%} {st['ruin_probability_50pct']:>7.1%}"
        )

    # ── 5. Multi-Year Projections (best portfolio) ──
    # Find best portfolio by Sharpe
    best_name = max(portfolios.keys(), key=lambda k: portfolios[k]["metrics"]["sharpe"])
    lines.append(f"\n" + "─" * 80)
    lines.append(f"  5. MULTI-YEAR PROJECTIONS — {best_name}")
    lines.append("─" * 80)

    if best_name in stress_results:
        st = stress_results[best_name]
        lines.append(f"\n  {'Horizon':<8} {'Med Ret':>8} {'Mean Ret':>9} {'P(>0)':>7} "
                     f"{'P(>10%)':>8} {'P(>20%)':>8} {'P5':>8} {'P95':>9}")
        lines.append("  " + "-" * 70)

        for horizon, proj in st["projections"].items():
            lines.append(
                f"  {horizon:<8} {proj['median_return']:>+7.1%} {proj['mean_return']:>+8.1%} "
                f"{proj['prob_positive']:>6.1%} {proj['prob_gt_10pct']:>7.1%} "
                f"{proj['prob_gt_20pct']:>7.1%} {proj['p5_return']:>+7.1%} "
                f"{proj['p95_return']:>+8.1%}"
            )

        # Capital projections
        lines.append(f"\n  Capital Projections (starting ${INITIAL_CAPITAL:,.0f})")
        lines.append(f"  {'Horizon':<8} {'Pessimist (P5)':>16} {'Median':>12} {'Optimist (P95)':>16}")
        lines.append("  " + "-" * 55)

        for horizon, proj in st["projections"].items():
            p5_cap = INITIAL_CAPITAL * (1 + proj["p5_return"])
            med_cap = INITIAL_CAPITAL * (1 + proj["median_return"])
            p95_cap = INITIAL_CAPITAL * (1 + proj["p95_return"])
            lines.append(f"  {horizon:<8} ${p5_cap:>14,.0f} ${med_cap:>10,.0f} ${p95_cap:>14,.0f}")

    # ── 6. Risk Assessment ──
    lines.append("\n" + "─" * 80)
    lines.append("  6. RISK ASSESSMENT (All Leverage Levels)")
    lines.append("─" * 80)

    lines.append(f"\n  {'Portfolio':<25} {'Risk':>6} {'Ruin%':>7} {'Med DD':>8} {'Worst DD':>9}")
    lines.append("  " + "-" * 60)

    for key in key_portfolios:
        st = stress_results[key]
        ruin = st["ruin_probability_50pct"]
        median_dd = abs(st["bootstrap_1Y"]["median_dd"])
        worst_dd = abs(st["bootstrap_1Y"]["p5_dd"])

        if ruin < 0.01 and worst_dd < 0.20:
            risk_level = "LOW"
        elif ruin < 0.05 and worst_dd < 0.30:
            risk_level = "MED"
        else:
            risk_level = "HIGH"

        lines.append(
            f"  {key:<25} {risk_level:>6} {ruin:>6.1%} {median_dd:>7.1%} {worst_dd:>8.1%}"
        )

    # ── 7. Verdict ──
    lines.append("\n" + "─" * 80)
    lines.append("  7. VERDICT & RECOMMENDATIONS")
    lines.append("─" * 80)

    # Compare sharpe_weighted vs positive_only at each leverage
    for lev in LEVERAGE_LEVELS:
        suffix = f"_{lev:.0f}x" if lev != 1.0 else ""
        sw_key = f"sharpe_weighted{suffix}"
        po_key = f"positive_only{suffix}"
        if sw_key in portfolios and po_key in portfolios:
            sw_sharpe = portfolios[sw_key]["metrics"]["sharpe"]
            po_sharpe = portfolios[po_key]["metrics"]["sharpe"]
            sw_dd = portfolios[sw_key]["metrics"]["max_drawdown"]
            po_dd = portfolios[po_key]["metrics"]["max_drawdown"]
            winner = "positive_only" if po_sharpe > sw_sharpe else "sharpe_weighted"
            lines.append(f"\n  At {lev:.0f}x leverage:")
            lines.append(f"    sharpe_weighted:  Sharpe={sw_sharpe:.2f}, DD={sw_dd:.1%}")
            lines.append(f"    positive_only:    Sharpe={po_sharpe:.2f}, DD={po_dd:.1%}")
            lines.append(f"    → Winner: {winner}")

    # Overall recommendation
    lines.append(f"\n  Overall best portfolio: {best_name}")
    best_m = portfolios[best_name]["metrics"]
    lines.append(f"  Sharpe: {best_m['sharpe']:.2f} | Return: {best_m['total_return']:.1%} | "
                 f"DD: {best_m['max_drawdown']:.1%}")

    if best_name in stress_results:
        ruin = stress_results[best_name]["ruin_probability_50pct"]
        if ruin < 0.01 and best_m["sharpe"] > 0.5:
            lines.append("  VERDICT: VIABLE for live deployment")
        elif best_m["sharpe"] > 0.3:
            lines.append("  VERDICT: MARGINAL — consider more strategies")
        else:
            lines.append("  VERDICT: NOT RECOMMENDED for live deployment")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def generate_md_report(combo_results: list[dict], portfolios: dict,
                       stress_results: dict, positive_indices: list[int]) -> str:
    """Generate Markdown report for docs/results/."""
    lines = []
    lines.append("# Rapport — Portfolio V2 : Sharpe-Weighted vs Positive-Only + Leverage")
    lines.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    lines.append(f"**Fichier source** : `results/meta_profiles_v3_20260211_195716.json`")
    lines.append(f"**Statut** : ✅ VALIDE")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Context ──
    lines.append("## Contexte")
    lines.append("")
    lines.append("Comparaison de deux approches de construction de portfolio :")
    lines.append("- **sharpe_weighted** : tous les combos, poids ∝ max(Sharpe, 0.01)")
    lines.append("- **positive_only** : filtre les combos avec Sharpe < 0, poids ∝ Sharpe")
    lines.append("")
    lines.append("Test de leverage 1x, 2x, 3x sur chaque variante avec Monte Carlo stress tests.")
    lines.append("")

    # ── 1. Individual Performance ──
    lines.append("## 1. Performance individuelle (Walk-Forward OOS)")
    lines.append("")
    lines.append("| Symbol | Stratégie | Sharpe | Return | DD | PF | WR | Trades | Filtre |")
    lines.append("|--------|-----------|--------|--------|-----|-----|-----|--------|--------|")

    for i, r in enumerate(combo_results):
        p = r["profile"]
        m = r["metrics"]
        filt = "✓ IN" if i in positive_indices else "✗ OUT"
        sharpe_fmt = f"**{m['sharpe']:.2f}**" if m['sharpe'] > 0 else f"{m['sharpe']:.2f}"
        lines.append(
            f"| {p['symbol']} | {p['strategy']} | {sharpe_fmt} | "
            f"{m['total_return']:.1%} | {m['max_drawdown']:.1%} | "
            f"{m['profit_factor']:.2f} | {m['win_rate']:.1%} | {m['n_trades']} | {filt} |"
        )

    n_pos = len(positive_indices)
    n_neg = len(combo_results) - n_pos
    lines.append(f"\n**{n_pos} combos retenus** (Sharpe > 0) | **{n_neg} combos filtrés** (Sharpe ≤ 0)")
    lines.append("")

    # ── 2. Portfolio Comparison 1x ──
    lines.append("## 2. Comparaison des portfolios (1x)")
    lines.append("")
    lines.append("| Portfolio | Sharpe | Return | DD | Sortino |")
    lines.append("|-----------|--------|--------|-----|---------|")

    for name in ["equal_weight", "risk_parity", "sharpe_weighted", "positive_only"]:
        if name in portfolios:
            m = portfolios[name]["metrics"]
            lines.append(
                f"| {name} | {m['sharpe']:.2f} | {m['total_return']:.1%} | "
                f"{m['max_drawdown']:.1%} | {m['sortino']:.2f} |"
            )

    # Allocations
    lines.append("")
    lines.append("### Allocation sharpe_weighted")
    lines.append("")
    sw = portfolios.get("sharpe_weighted", {})
    if sw:
        lines.append("| Combo | Poids |")
        lines.append("|-------|-------|")
        for i, r in enumerate(combo_results):
            p = r["profile"]
            w = sw["weights"][i]
            lines.append(f"| {p['symbol']}/{p['strategy']} | {w:.1%} |")

    lines.append("")
    lines.append("### Allocation positive_only")
    lines.append("")
    po = portfolios.get("positive_only", {})
    if po:
        lines.append("| Combo | Poids |")
        lines.append("|-------|-------|")
        for j, idx in enumerate(positive_indices):
            p = combo_results[idx]["profile"]
            w = po["weights"][j]
            lines.append(f"| {p['symbol']}/{p['strategy']} | {w:.1%} |")

    lines.append("")

    # ── 3. Leverage Comparison ──
    lines.append("## 3. Impact du Leverage")
    lines.append("")
    lines.append("| Portfolio | Lev | Sharpe | Return | Ret/an | DD | Sortino |")
    lines.append("|-----------|-----|--------|--------|--------|-----|---------|")

    for base_name in ["sharpe_weighted", "positive_only"]:
        for lev in LEVERAGE_LEVELS:
            suffix = f"_{lev:.0f}x" if lev != 1.0 else ""
            key = f"{base_name}{suffix}"
            if key in portfolios:
                m = portfolios[key]["metrics"]
                n_years = max(1, m.get("n_periods", 365) / 365.25)
                ann_ret = (1 + m["total_return"]) ** (1 / n_years) - 1
                lines.append(
                    f"| {base_name} | {lev:.0f}x | {m['sharpe']:.2f} | "
                    f"{m['total_return']:.1%} | {ann_ret:.1%} | "
                    f"{m['max_drawdown']:.1%} | {m['sortino']:.2f} |"
                )

    lines.append("")

    # ── 4. Monte Carlo Stress Tests ──
    lines.append("## 4. Monte Carlo Stress Tests (1Y, 1000 sims)")
    lines.append("")
    lines.append("| Portfolio | Med Ret | P5 Ret | P95 Ret | Med DD | Worst DD | P(ruin) |")
    lines.append("|-----------|---------|--------|---------|--------|----------|---------|")

    for base_name in ["sharpe_weighted", "positive_only"]:
        for lev in LEVERAGE_LEVELS:
            suffix = f"_{lev:.0f}x" if lev != 1.0 else ""
            key = f"{base_name}{suffix}"
            if key in stress_results:
                st = stress_results[key]
                b = st["bootstrap_1Y"]
                lines.append(
                    f"| {key} | {b['median_return']:+.1%} | {b['p5_return']:+.1%} | "
                    f"{b['p95_return']:+.1%} | {b['median_dd']:.1%} | "
                    f"{b['p5_dd']:.1%} | {st['ruin_probability_50pct']:.1%} |"
                )

    lines.append("")

    # ── 5. Multi-Year Projections ──
    best_name = max(portfolios.keys(), key=lambda k: portfolios[k]["metrics"]["sharpe"])
    lines.append(f"## 5. Projections multi-horizon — {best_name}")
    lines.append("")

    if best_name in stress_results:
        st = stress_results[best_name]
        lines.append("| Horizon | Med Ret | Mean Ret | P(>0) | P(>10%) | P(>20%) | P5 | P95 |")
        lines.append("|---------|---------|----------|-------|---------|---------|-----|-----|")

        for horizon, proj in st["projections"].items():
            lines.append(
                f"| {horizon} | {proj['median_return']:+.1%} | {proj['mean_return']:+.1%} | "
                f"{proj['prob_positive']:.1%} | {proj['prob_gt_10pct']:.1%} | "
                f"{proj['prob_gt_20pct']:.1%} | {proj['p5_return']:+.1%} | "
                f"{proj['p95_return']:+.1%} |"
            )

        lines.append("")
        lines.append(f"### Projections de capital (${INITIAL_CAPITAL:,.0f})")
        lines.append("")
        lines.append("| Horizon | Pessimiste (P5) | Médian | Optimiste (P95) |")
        lines.append("|---------|----------------|--------|-----------------|")

        for horizon, proj in st["projections"].items():
            p5_cap = INITIAL_CAPITAL * (1 + proj["p5_return"])
            med_cap = INITIAL_CAPITAL * (1 + proj["median_return"])
            p95_cap = INITIAL_CAPITAL * (1 + proj["p95_return"])
            lines.append(f"| {horizon} | ${p5_cap:,.0f} | ${med_cap:,.0f} | ${p95_cap:,.0f} |")

    lines.append("")

    # ── 6. Risk Assessment ──
    lines.append("## 6. Risk Assessment")
    lines.append("")
    lines.append("| Portfolio | Risk | Ruin% | Med DD | Worst DD |")
    lines.append("|-----------|------|-------|--------|----------|")

    for base_name in ["sharpe_weighted", "positive_only"]:
        for lev in LEVERAGE_LEVELS:
            suffix = f"_{lev:.0f}x" if lev != 1.0 else ""
            key = f"{base_name}{suffix}"
            if key in stress_results:
                st = stress_results[key]
                ruin = st["ruin_probability_50pct"]
                median_dd = abs(st["bootstrap_1Y"]["median_dd"])
                worst_dd = abs(st["bootstrap_1Y"]["p5_dd"])
                if ruin < 0.01 and worst_dd < 0.20:
                    risk_level = "🟢 LOW"
                elif ruin < 0.05 and worst_dd < 0.30:
                    risk_level = "🟡 MED"
                else:
                    risk_level = "🔴 HIGH"
                lines.append(
                    f"| {key} | {risk_level} | {ruin:.1%} | {median_dd:.1%} | {worst_dd:.1%} |"
                )

    lines.append("")

    # ── 7. Verdict ──
    lines.append("## 7. Verdict & Recommandations")
    lines.append("")

    for lev in LEVERAGE_LEVELS:
        suffix = f"_{lev:.0f}x" if lev != 1.0 else ""
        sw_key = f"sharpe_weighted{suffix}"
        po_key = f"positive_only{suffix}"
        if sw_key in portfolios and po_key in portfolios:
            sw_s = portfolios[sw_key]["metrics"]["sharpe"]
            po_s = portfolios[po_key]["metrics"]["sharpe"]
            sw_dd = portfolios[sw_key]["metrics"]["max_drawdown"]
            po_dd = portfolios[po_key]["metrics"]["max_drawdown"]
            winner = "**positive_only**" if po_s > sw_s else "**sharpe_weighted**"
            lines.append(f"### Leverage {lev:.0f}x")
            lines.append(f"- sharpe_weighted : Sharpe={sw_s:.2f}, DD={sw_dd:.1%}")
            lines.append(f"- positive_only : Sharpe={po_s:.2f}, DD={po_dd:.1%}")
            lines.append(f"- Gagnant : {winner}")
            lines.append("")

    lines.append(f"### Meilleur portfolio global : `{best_name}`")
    best_m = portfolios[best_name]["metrics"]
    lines.append(f"- Sharpe : {best_m['sharpe']:.2f}")
    lines.append(f"- Return : {best_m['total_return']:.1%}")
    lines.append(f"- Max DD : {best_m['max_drawdown']:.1%}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*Généré le {datetime.now().strftime('%d %B %Y')}*")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    logger.info("=" * 70)
    logger.info("  QUANTLAB V7 — PORTFOLIO V2: SHARPE-WEIGHTED vs POSITIVE-ONLY + LEVERAGE")
    logger.info("=" * 70)

    start_time = time.time()

    # Load data
    settings = load_settings()
    data_by_symbol = load_all_symbols_data(settings)
    risk = _build_risk(settings)

    # Load meta profiles
    profiles = load_meta_profiles(META_V3_PATH)
    logger.info(f"Loaded {len(profiles)} meta-profiles from {META_V3_PATH}")

    # Step 1: Re-run walk-forwards
    logger.info("\n--- STEP 1: Re-running walk-forwards ---")
    combo_results = rerun_walk_forwards(profiles, data_by_symbol, settings, risk)

    if not combo_results:
        logger.error("No valid combo results. Aborting.")
        return

    # Step 2: Build portfolios
    logger.info("\n--- STEP 2: Building portfolios (sharpe_weighted vs positive_only × leverage) ---")
    portfolios, positive_indices = build_all_portfolios(combo_results)

    logger.info(f"  Built {len(portfolios)} portfolio variants")
    for name, pf in portfolios.items():
        m = pf["metrics"]
        logger.info(f"  {name:<30} Sharpe={m['sharpe']:.2f} | Return={m['total_return']:.1%} | "
                    f"DD={m['max_drawdown']:.1%}")

    # Step 3: Monte Carlo stress tests (on key portfolios)
    logger.info("\n--- STEP 3: Monte Carlo stress tests ---")
    stress_results = {}
    for base_name in ["sharpe_weighted", "positive_only"]:
        for lev in LEVERAGE_LEVELS:
            suffix = f"_{lev:.0f}x" if lev != 1.0 else ""
            key = f"{base_name}{suffix}"
            if key in portfolios:
                stress_results[key] = run_all_stress_tests(portfolios[key]["returns"], key)

    # Step 4: Generate reports
    logger.info("\n--- STEP 4: Generating reports ---")
    report_text = generate_text_report(combo_results, portfolios, stress_results, positive_indices)
    report_md = generate_md_report(combo_results, portfolios, stress_results, positive_indices)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y%m%d")
    Path("results").mkdir(exist_ok=True)
    Path("docs/results").mkdir(parents=True, exist_ok=True)

    # Text report
    txt_path = f"results/portfolio_v2_leverage_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write(report_text)

    # JSON report
    json_report = {
        "timestamp": timestamp,
        "meta_profiles_source": META_V3_PATH,
        "initial_capital": INITIAL_CAPITAL,
        "n_monte_carlo": N_MONTE_CARLO,
        "leverage_levels": LEVERAGE_LEVELS,
        "combos": [
            {
                "profile": r["profile"],
                "metrics": r["metrics"],
                "n_bars": r["n_bars"],
            }
            for r in combo_results
        ],
        "positive_indices": positive_indices,
        "portfolios": {
            name: {
                "weights": pf["weights"],
                "leverage": pf["leverage"],
                "metrics": pf["metrics"],
            }
            for name, pf in portfolios.items()
        },
        "stress_tests": stress_results,
        "best_portfolio": max(portfolios.keys(), key=lambda k: portfolios[k]["metrics"]["sharpe"]),
    }
    json_path = f"results/portfolio_v2_leverage_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)

    # Markdown report
    md_path = f"docs/results/06_portfolio_v2_leverage_{date_str}.md"
    with open(md_path, "w") as f:
        f.write(report_md)

    # Print report
    print(report_text)

    elapsed = time.time() - start_time
    logger.info(f"\nSaved: {txt_path}")
    logger.info(f"Saved: {json_path}")
    logger.info(f"Saved: {md_path}")
    logger.info(f"Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
