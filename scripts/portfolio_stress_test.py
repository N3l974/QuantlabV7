#!/usr/bin/env python3
"""
Portfolio V1 — Construction, Monte Carlo Stress Test & Performance Report.

Steps:
  1. Load meta-optimisation V3 profiles
  2. Re-run walk-forward for each combo → get full equity curves
  3. Build portfolio (risk-parity, equal-weight, sharpe-weighted)
  4. Monte Carlo stress tests (bootstrap, shuffled returns, vol scaling)
  5. Generate comprehensive report with future performance projections

Output: results/portfolio_report_v1_{timestamp}.json
        results/portfolio_report_v1_{timestamp}.txt
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
    annualization_factor,
    calmar_ratio,
    compute_all_metrics,
    max_drawdown,
    profit_factor,
    returns_from_equity,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)
from engine.walk_forward import WalkForwardConfig, run_walk_forward
from strategies.registry import get_strategy

# ── Config ──
META_V3_PATH = "results/meta_profiles_v3_20260211_195716.json"
INITIAL_CAPITAL = 10_000.0
N_MONTE_CARLO = 1000
PROJECTION_YEARS = [1, 2, 3, 5]
CONFIDENCE_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]  # percentiles


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
# STEP 2: Portfolio construction
# ═══════════════════════════════════════════════════════════════
def build_portfolios(combo_results: list[dict]) -> dict:
    """Build multiple portfolio weighting schemes."""
    n = len(combo_results)
    if n == 0:
        return {}

    # Align returns to same length (pad shorter with 0)
    max_len = max(len(r["returns"]) for r in combo_results)
    aligned_returns = np.zeros((n, max_len))
    for i, r in enumerate(combo_results):
        rets = r["returns"]
        aligned_returns[i, :len(rets)] = rets

    # --- Equal Weight ---
    eq_weights = np.ones(n) / n

    # --- Sharpe Weighted ---
    sharpes = np.array([max(r["metrics"]["sharpe"], 0.01) for r in combo_results])
    sharpe_weights = sharpes / sharpes.sum()

    # --- Risk Parity (inverse DD) ---
    inv_dd = np.array([1.0 / max(abs(r["metrics"]["max_drawdown"]), 0.01) for r in combo_results])
    rp_weights = inv_dd / inv_dd.sum()

    portfolios = {}
    for name, weights in [("equal_weight", eq_weights),
                          ("sharpe_weighted", sharpe_weights),
                          ("risk_parity", rp_weights)]:
        port_returns = aligned_returns.T @ weights  # (T, n) @ (n,) = (T,)
        port_equity = INITIAL_CAPITAL * np.cumprod(1 + port_returns)
        port_equity = np.insert(port_equity, 0, INITIAL_CAPITAL)

        metrics = compute_all_metrics(port_equity, "1d")
        portfolios[name] = {
            "weights": weights.tolist(),
            "returns": port_returns,
            "equity": port_equity,
            "metrics": metrics,
        }

    return portfolios


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


def monte_carlo_vol_stress(returns: np.ndarray, n_sims: int = 1000,
                           n_days: int = 365, vol_multipliers: list = None) -> dict:
    """Stress test with increased volatility (1x, 1.5x, 2x, 3x)."""
    if vol_multipliers is None:
        vol_multipliers = [1.0, 1.5, 2.0, 3.0]

    results = {}
    for mult in vol_multipliers:
        stressed_returns = returns * mult
        sims = monte_carlo_bootstrap(stressed_returns, n_sims, n_days)
        results[f"vol_{mult:.1f}x"] = {
            "median_return": float(np.median(sims["final_returns"])),
            "p5_return": float(np.percentile(sims["final_returns"], 5)),
            "p95_return": float(np.percentile(sims["final_returns"], 95)),
            "median_dd": float(np.median(sims["max_drawdowns"])),
            "p5_dd": float(np.percentile(sims["max_drawdowns"], 5)),
        }

    return results


def monte_carlo_ruin_probability(returns: np.ndarray, n_sims: int = 1000,
                                 n_days: int = 365, ruin_threshold: float = 0.5) -> float:
    """Probability of losing more than ruin_threshold (e.g. 50%) of capital."""
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
    logger.info(f"  Running Monte Carlo stress tests for {name} ({N_MONTE_CARLO} sims)...")

    # 1-year projections
    bootstrap = monte_carlo_bootstrap(returns, N_MONTE_CARLO, 365)
    block_boot = monte_carlo_block_bootstrap(returns, N_MONTE_CARLO, 365)
    vol_stress = monte_carlo_vol_stress(returns, N_MONTE_CARLO, 365)
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
            percentiles[f"{pct_label}_sharpe"] = float(np.percentile(proj["sharpes"], pct * 100)) if "sharpes" in proj else 0
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
            "p5_sharpe": float(np.percentile(bootstrap["sharpes"], 5)),
        },
        "block_bootstrap_1Y": {
            "median_return": float(np.median(block_boot["final_returns"])),
            "median_dd": float(np.median(block_boot["max_drawdowns"])),
            "p5_dd": float(np.percentile(block_boot["max_drawdowns"], 5)),
        },
        "vol_stress": vol_stress,
        "ruin_probability_50pct": ruin_prob,
        "projections": projections,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 4: Generate Report
# ═══════════════════════════════════════════════════════════════
def generate_text_report(combo_results: list[dict], portfolios: dict,
                         stress_results: dict, best_portfolio_name: str) -> str:
    """Generate a comprehensive text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  QUANTLAB V7 — PORTFOLIO V1 REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # ── Individual Combos ──
    lines.append("\n" + "─" * 80)
    lines.append("  1. INDIVIDUAL STRATEGY PERFORMANCE (Walk-Forward OOS)")
    lines.append("─" * 80)
    lines.append(f"  {'Symbol':<10} {'Strategy':<25} {'Sharpe':>7} {'Return':>8} {'DD':>8} "
                 f"{'PF':>6} {'WR':>6} {'Trades':>7}")
    lines.append("  " + "-" * 75)

    for r in combo_results:
        p = r["profile"]
        m = r["metrics"]
        lines.append(
            f"  {p['symbol']:<10} {p['strategy']:<25} {m['sharpe']:>7.2f} "
            f"{m['total_return']:>7.1%} {m['max_drawdown']:>7.1%} "
            f"{m['profit_factor']:>6.2f} {m['win_rate']:>5.1%} {m['n_trades']:>7}"
        )

    # ── Portfolio Comparison ──
    lines.append("\n" + "─" * 80)
    lines.append("  2. PORTFOLIO COMPARISON")
    lines.append("─" * 80)
    lines.append(f"  {'Portfolio':<20} {'Sharpe':>7} {'Return':>8} {'DD':>8} {'Stability':>10}")
    lines.append("  " + "-" * 55)

    for name, pf in portfolios.items():
        m = pf["metrics"]
        lines.append(
            f"  {name:<20} {m['sharpe']:>7.2f} {m['total_return']:>7.1%} "
            f"{m['max_drawdown']:>7.1%} {m.get('stability', 0):>10.2f}"
        )

    # ── Best Portfolio Weights ──
    best = portfolios[best_portfolio_name]
    lines.append(f"\n  Best Portfolio: {best_portfolio_name}")
    lines.append(f"  {'Combo':<40} {'Weight':>8}")
    lines.append("  " + "-" * 50)
    for i, r in enumerate(combo_results):
        p = r["profile"]
        w = best["weights"][i]
        lines.append(f"  {p['symbol']}/{p['strategy']:<30} {w:>7.1%}")

    # ── Monte Carlo Stress Tests ──
    lines.append("\n" + "─" * 80)
    lines.append("  3. MONTE CARLO STRESS TESTS")
    lines.append("─" * 80)

    st = stress_results[best_portfolio_name]

    lines.append("\n  3a. Bootstrap (1Y, 1000 sims)")
    b = st["bootstrap_1Y"]
    lines.append(f"    Median Return:  {b['median_return']:>+7.1%}")
    lines.append(f"    Mean Return:    {b['mean_return']:>+7.1%}")
    lines.append(f"    Std Dev:        {b['std_return']:>7.1%}")
    lines.append(f"    5th percentile: {b['p5_return']:>+7.1%}  (worst case)")
    lines.append(f"    25th pctl:      {b['p25_return']:>+7.1%}")
    lines.append(f"    75th pctl:      {b['p75_return']:>+7.1%}")
    lines.append(f"    95th pctl:      {b['p95_return']:>+7.1%}  (best case)")
    lines.append(f"    Median DD:      {b['median_dd']:>7.1%}")
    lines.append(f"    Worst DD (p5):  {b['p5_dd']:>7.1%}")
    lines.append(f"    Median Sharpe:  {b['median_sharpe']:>7.2f}")

    lines.append(f"\n  3b. Block Bootstrap (1Y, preserves autocorrelation)")
    bb = st["block_bootstrap_1Y"]
    lines.append(f"    Median Return:  {bb['median_return']:>+7.1%}")
    lines.append(f"    Median DD:      {bb['median_dd']:>7.1%}")
    lines.append(f"    Worst DD (p5):  {bb['p5_dd']:>7.1%}")

    lines.append(f"\n  3c. Volatility Stress Test")
    lines.append(f"    {'Vol Mult':<10} {'Median Ret':>12} {'P5 Ret':>10} {'P95 Ret':>10} {'Median DD':>10}")
    lines.append("    " + "-" * 55)
    for key, vs in st["vol_stress"].items():
        lines.append(
            f"    {key:<10} {vs['median_return']:>+11.1%} {vs['p5_return']:>+9.1%} "
            f"{vs['p95_return']:>+9.1%} {vs['median_dd']:>9.1%}"
        )

    lines.append(f"\n  3d. Ruin Probability (losing >50% of capital in 1Y)")
    lines.append(f"    P(ruin): {st['ruin_probability_50pct']:.1%}")

    # ── Future Projections ──
    lines.append("\n" + "─" * 80)
    lines.append("  4. FUTURE PERFORMANCE PROJECTIONS")
    lines.append("─" * 80)

    for horizon, proj in st["projections"].items():
        lines.append(f"\n  {horizon} Projection ({N_MONTE_CARLO} simulations)")
        lines.append(f"    Median Return:    {proj['median_return']:>+8.1%}")
        lines.append(f"    Mean Return:      {proj['mean_return']:>+8.1%}")
        lines.append(f"    Std Dev:          {proj['std_return']:>8.1%}")
        lines.append(f"    Median DD:        {proj['median_dd']:>8.1%}")
        lines.append(f"    P(positive):      {proj['prob_positive']:>8.1%}")
        lines.append(f"    P(return > 10%):  {proj['prob_gt_10pct']:>8.1%}")
        lines.append(f"    P(return > 20%):  {proj['prob_gt_20pct']:>8.1%}")
        lines.append(f"    5th pctl return:  {proj['p5_return']:>+8.1%}")
        lines.append(f"    95th pctl return: {proj['p95_return']:>+8.1%}")

    # ── Capital Projections ──
    lines.append("\n" + "─" * 80)
    lines.append("  5. CAPITAL PROJECTIONS (starting $10,000)")
    lines.append("─" * 80)
    lines.append(f"  {'Horizon':<10} {'Pessimist (P5)':>16} {'Median':>12} {'Optimist (P95)':>16}")
    lines.append("  " + "-" * 55)

    for horizon, proj in st["projections"].items():
        p5_cap = INITIAL_CAPITAL * (1 + proj["p5_return"])
        med_cap = INITIAL_CAPITAL * (1 + proj["median_return"])
        p95_cap = INITIAL_CAPITAL * (1 + proj["p95_return"])
        lines.append(f"  {horizon:<10} ${p5_cap:>14,.0f} ${med_cap:>10,.0f} ${p95_cap:>14,.0f}")

    # ── Risk Assessment ──
    lines.append("\n" + "─" * 80)
    lines.append("  6. RISK ASSESSMENT")
    lines.append("─" * 80)

    ruin = st["ruin_probability_50pct"]
    median_dd = abs(st["bootstrap_1Y"]["median_dd"])
    worst_dd = abs(st["bootstrap_1Y"]["p5_dd"])

    if ruin < 0.01 and worst_dd < 0.20:
        risk_level = "LOW"
        risk_emoji = "GREEN"
    elif ruin < 0.05 and worst_dd < 0.30:
        risk_level = "MODERATE"
        risk_emoji = "YELLOW"
    else:
        risk_level = "HIGH"
        risk_emoji = "RED"

    lines.append(f"  Overall Risk Level: {risk_level} ({risk_emoji})")
    lines.append(f"  Ruin Probability:   {ruin:.1%}")
    lines.append(f"  Median Max DD:      {median_dd:.1%}")
    lines.append(f"  Worst-case DD (5%): {worst_dd:.1%}")

    # ── Recommendations ──
    lines.append("\n" + "─" * 80)
    lines.append("  7. RECOMMENDATIONS")
    lines.append("─" * 80)

    best_m = best["metrics"]
    lines.append(f"  Portfolio: {best_portfolio_name}")
    lines.append(f"  Expected Annual Return: {st['projections']['1Y']['median_return']:+.1%}")
    lines.append(f"  Expected Sharpe: {best_m['sharpe']:.2f}")
    lines.append(f"  Expected Max DD: {best_m['max_drawdown']:.1%}")

    if best_m["sharpe"] > 0.5:
        lines.append("  VERDICT: VIABLE for live deployment with proper position sizing")
    elif best_m["sharpe"] > 0.3:
        lines.append("  VERDICT: MARGINAL — consider adding more strategies or reducing allocation")
    else:
        lines.append("  VERDICT: NOT RECOMMENDED for live deployment")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    logger.info("=" * 70)
    logger.info("  QUANTLAB V7 — PORTFOLIO V1 + STRESS TEST")
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
    logger.info("\n--- STEP 2: Building portfolios ---")
    portfolios = build_portfolios(combo_results)

    # Find best portfolio
    best_name = max(portfolios.keys(), key=lambda k: portfolios[k]["metrics"]["sharpe"])
    logger.info(f"Best portfolio: {best_name} (Sharpe={portfolios[best_name]['metrics']['sharpe']:.2f})")

    # Step 3: Monte Carlo stress tests
    logger.info("\n--- STEP 3: Monte Carlo stress tests ---")
    stress_results = {}
    for name, pf in portfolios.items():
        stress_results[name] = run_all_stress_tests(pf["returns"], name)

    # Step 4: Generate report
    logger.info("\n--- STEP 4: Generating report ---")
    report_text = generate_text_report(combo_results, portfolios, stress_results, best_name)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)

    # Text report
    txt_path = f"results/portfolio_report_v1_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write(report_text)

    # JSON report (without numpy arrays)
    json_report = {
        "timestamp": timestamp,
        "meta_profiles_source": META_V3_PATH,
        "initial_capital": INITIAL_CAPITAL,
        "n_monte_carlo": N_MONTE_CARLO,
        "combos": [
            {
                "profile": r["profile"],
                "metrics": r["metrics"],
                "n_bars": r["n_bars"],
            }
            for r in combo_results
        ],
        "portfolios": {
            name: {
                "weights": pf["weights"],
                "metrics": pf["metrics"],
            }
            for name, pf in portfolios.items()
        },
        "stress_tests": stress_results,
        "best_portfolio": best_name,
    }
    json_path = f"results/portfolio_report_v1_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)

    # Print report
    print(report_text)

    elapsed = time.time() - start_time
    logger.info(f"\nSaved: {txt_path}")
    logger.info(f"Saved: {json_path}")
    logger.info(f"Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
