#!/usr/bin/env python3
"""
Portfolio V3 — Markowitz Mean-Variance + Holdout Validation + Monte Carlo.

Pipeline:
  1. Define survivor combos from holdout tests (simple + multi-factor)
  2. Re-run walk-forward on FULL data → equity curves & returns
  3. Re-run walk-forward on IN-SAMPLE only → holdout equity curves
  4. Build portfolios:
     a) Markowitz (max Sharpe, covariance-aware)
     b) Markowitz min-variance
     c) HO-Sharpe weighted (weight ∝ holdout Sharpe)
     d) Equal weight
     e) Risk parity (inverse vol)
  5. Evaluate each portfolio on holdout period
  6. Monte Carlo stress tests + multi-year projections
  7. Leverage testing (1x, 2x, 3x) on best portfolio
  8. Generate JSON + Markdown reports

Output:
  results/portfolio_v3_{timestamp}.json
  docs/results/11_portfolio_v3.md
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
INITIAL_CAPITAL = 10_000.0
N_MONTE_CARLO = 1000
PROJECTION_YEARS = [1, 2, 3, 5]
LEVERAGE_LEVELS = [1.0, 2.0, 3.0]
CONFIDENCE_LEVELS = [0.05, 0.25, 0.50, 0.75, 0.95]
CUTOFF_DATE = "2025-02-01"
N_SEEDS = 3

# Walk-forward defaults (same as holdout tests)
TF_CONFIGS = {
    "4h": {"training_window": "6M", "reoptim_frequency": "3M", "n_optim_trials": 80},
    "1d": {"training_window": "1Y",  "reoptim_frequency": "3M", "n_optim_trials": 80},
}

# ── Survivor combos from holdout tests ──
# Pool of 8 best survivors (HO Sharpe > 0) from sessions 7-8
SURVIVORS = [
    # Multi-factor survivors
    {"symbol": "ETHUSDT",  "strategy": "breakout_regime",     "timeframe": "4h", "ho_sharpe": 0.935, "verdict": "STRONG"},
    {"symbol": "ETHUSDT",  "strategy": "trend_multi_factor",  "timeframe": "1d", "ho_sharpe": 0.779, "verdict": "WEAK"},
    {"symbol": "ETHUSDT",  "strategy": "supertrend_adx",      "timeframe": "4h", "ho_sharpe": 0.694, "verdict": "WEAK"},
    {"symbol": "ETHUSDT",  "strategy": "trend_multi_factor",  "timeframe": "4h", "ho_sharpe": 0.180, "verdict": "STRONG"},
    {"symbol": "SOLUSDT",  "strategy": "breakout_regime",     "timeframe": "1d", "ho_sharpe": 0.015, "verdict": "STRONG"},
    {"symbol": "BTCUSDT",  "strategy": "trend_multi_factor",  "timeframe": "1d", "ho_sharpe": 0.321, "verdict": "WEAK"},
    {"symbol": "BTCUSDT",  "strategy": "supertrend_adx",      "timeframe": "4h", "ho_sharpe": 0.194, "verdict": "WEAK"},
    # Simple strategy survivors
    {"symbol": "ETHUSDT",  "strategy": "supertrend",          "timeframe": "4h", "ho_sharpe": 0.444, "verdict": "STRONG"},
]


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
# STEP 1: Re-run walk-forward → equity curves + holdout backtest
# ═══════════════════════════════════════════════════════════════
def run_all_combos(survivors, data_by_symbol, settings, risk, cutoff_dt):
    """
    For each survivor combo:
      - Run WF on full data → full equity curve (for portfolio construction)
      - Run WF on in-sample → last params → holdout backtest (for validation)
    Returns list of combo results with both full and holdout data.
    """
    results = []
    commission = settings["engine"]["commission_rate"]
    slippage = settings["engine"]["slippage_rate"]

    for idx, combo in enumerate(survivors):
        symbol = combo["symbol"]
        strat_name = combo["strategy"]
        tf = combo["timeframe"]
        tf_cfg = TF_CONFIGS[tf]

        logger.info(f"\n[{idx+1}/{len(survivors)}] {symbol}/{strat_name}/{tf}")

        data = data_by_symbol.get(symbol, {}).get(tf)
        if data is None:
            logger.error(f"  No data for {symbol}/{tf}")
            continue

        strategy = get_strategy(strat_name)
        in_sample = data[data.index < cutoff_dt]
        holdout = data[data.index >= cutoff_dt]

        if len(in_sample) < 200 or len(holdout) < 30:
            logger.warning(f"  Insufficient data (IS={len(in_sample)}, HO={len(holdout)})")
            continue

        # ── Multi-seed: full WF + holdout ──
        full_equities = []
        full_returns_list = []
        full_metrics_list = []
        ho_equities = []
        ho_returns_list = []
        ho_metrics_list = []

        for seed_i in range(N_SEEDS):
            seed = 42 + seed_i

            # Full data walk-forward
            wf_full = WalkForwardConfig(
                strategy=strategy, data=data, timeframe=tf,
                reoptim_frequency=tf_cfg["reoptim_frequency"],
                training_window=tf_cfg["training_window"],
                param_bounds_scale=1.0, optim_metric="sharpe",
                n_optim_trials=tf_cfg["n_optim_trials"],
                commission=commission, slippage=slippage,
                risk=risk, seed=seed,
            )
            try:
                full_result = run_walk_forward(wf_full)
                full_eq = full_result.oos_equity
                full_ret = returns_from_equity(full_eq)
                full_met = full_result.metrics
                full_equities.append(full_eq)
                full_returns_list.append(full_ret)
                full_metrics_list.append(full_met)
            except Exception as e:
                logger.warning(f"  Seed {seed} full WF failed: {e}")
                continue

            # In-sample WF → holdout backtest
            wf_is = WalkForwardConfig(
                strategy=strategy, data=in_sample, timeframe=tf,
                reoptim_frequency=tf_cfg["reoptim_frequency"],
                training_window=tf_cfg["training_window"],
                param_bounds_scale=1.0, optim_metric="sharpe",
                n_optim_trials=tf_cfg["n_optim_trials"],
                commission=commission, slippage=slippage,
                risk=risk, seed=seed,
            )
            try:
                is_result = run_walk_forward(wf_is)
                if not is_result.best_params_per_period:
                    continue
                last_params = is_result.best_params_per_period[-1]

                ho_result = backtest_strategy(
                    strategy, holdout, last_params,
                    commission=commission, slippage=slippage,
                    initial_capital=INITIAL_CAPITAL, risk=risk, timeframe=tf,
                )
                ho_met = compute_all_metrics(ho_result.equity, tf, ho_result.trades_pnl)
                ho_eq = ho_result.equity
                ho_ret = returns_from_equity(ho_eq)
                ho_equities.append(ho_eq)
                ho_returns_list.append(ho_ret)
                ho_metrics_list.append(ho_met)
            except Exception as e:
                logger.warning(f"  Seed {seed} holdout failed: {e}")
                continue

        if not full_returns_list or not ho_returns_list:
            logger.error(f"  No valid results for {symbol}/{strat_name}/{tf}")
            continue

        # Pick median-Sharpe seed for full and holdout
        full_sharpes = [m["sharpe"] for m in full_metrics_list]
        median_idx_full = int(np.argsort(full_sharpes)[len(full_sharpes) // 2])

        ho_sharpes = [m["sharpe"] for m in ho_metrics_list]
        median_idx_ho = int(np.argsort(ho_sharpes)[len(ho_sharpes) // 2])

        result = {
            "combo": combo,
            # Full data (for portfolio construction)
            "full_equity": full_equities[median_idx_full],
            "full_returns": full_returns_list[median_idx_full],
            "full_metrics": full_metrics_list[median_idx_full],
            # Holdout (for validation)
            "ho_equity": ho_equities[median_idx_ho],
            "ho_returns": ho_returns_list[median_idx_ho],
            "ho_metrics": ho_metrics_list[median_idx_ho],
            # Stats across seeds
            "full_sharpe_seeds": full_sharpes,
            "ho_sharpe_seeds": ho_sharpes,
        }

        logger.info(f"  Full: Sharpe={result['full_metrics']['sharpe']:.3f} | "
                     f"HO: Sharpe={result['ho_metrics']['sharpe']:.3f} | "
                     f"Return={result['ho_metrics']['total_return']:.1%}")

        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 2: Markowitz Mean-Variance Optimization
# ═══════════════════════════════════════════════════════════════
def markowitz_max_sharpe(returns_matrix, risk_free=0.0, n_portfolios=10000):
    """
    Monte Carlo Markowitz: sample random portfolios, find max Sharpe.
    returns_matrix: (n_combos, n_periods) aligned returns.
    """
    n_assets = returns_matrix.shape[0]
    mean_returns = np.mean(returns_matrix, axis=1)
    cov_matrix = np.cov(returns_matrix)

    best_sharpe = -np.inf
    best_weights = np.ones(n_assets) / n_assets
    best_ret = 0.0
    best_vol = 1.0

    all_sharpes = []
    all_weights = []

    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        port_ret = w @ mean_returns
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol == 0:
            continue
        port_sharpe = (port_ret - risk_free) / port_vol
        all_sharpes.append(port_sharpe)
        all_weights.append(w)

        if port_sharpe > best_sharpe:
            best_sharpe = port_sharpe
            best_weights = w
            best_ret = port_ret
            best_vol = port_vol

    return {
        "weights": best_weights,
        "expected_return": float(best_ret),
        "expected_vol": float(best_vol),
        "sharpe": float(best_sharpe),
    }


def markowitz_min_variance(returns_matrix):
    """Find minimum variance portfolio via Monte Carlo sampling."""
    n_assets = returns_matrix.shape[0]
    cov_matrix = np.cov(returns_matrix)
    mean_returns = np.mean(returns_matrix, axis=1)

    best_vol = np.inf
    best_weights = np.ones(n_assets) / n_assets

    for _ in range(10000):
        w = np.random.dirichlet(np.ones(n_assets))
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol < best_vol:
            best_vol = port_vol
            best_weights = w

    port_ret = best_weights @ mean_returns
    return {
        "weights": best_weights,
        "expected_return": float(port_ret),
        "expected_vol": float(best_vol),
    }


# ═══════════════════════════════════════════════════════════════
# STEP 3: Portfolio Construction
# ═══════════════════════════════════════════════════════════════
def align_returns(combo_results, key="full_returns"):
    """Align return series to same length (pad with 0)."""
    n = len(combo_results)
    max_len = max(len(r[key]) for r in combo_results)
    aligned = np.zeros((n, max_len))
    for i, r in enumerate(combo_results):
        rets = r[key]
        aligned[i, :len(rets)] = rets
    return aligned


def build_portfolio_equity(aligned_returns, weights, leverage=1.0):
    """Build portfolio equity curve from aligned returns and weights."""
    port_returns = (aligned_returns.T @ weights) * leverage
    port_equity = INITIAL_CAPITAL * np.cumprod(1 + port_returns)
    port_equity = np.insert(port_equity, 0, INITIAL_CAPITAL)
    return port_returns, port_equity


def build_all_portfolios(combo_results):
    """Build all portfolio variants from combo results."""
    n = len(combo_results)
    aligned_full = align_returns(combo_results, "full_returns")
    aligned_ho = align_returns(combo_results, "ho_returns")

    # Correlation matrix (for reporting)
    corr_matrix = np.corrcoef(aligned_full) if n > 1 else np.array([[1.0]])

    portfolios = {}

    # ── 1. Markowitz Max Sharpe ──
    logger.info("  Computing Markowitz Max Sharpe...")
    mkz = markowitz_max_sharpe(aligned_full)
    portfolios["markowitz_max_sharpe"] = {
        "weights": mkz["weights"],
        "method": "Markowitz Monte Carlo (10K samples, max Sharpe)",
    }

    # ── 2. Markowitz Min Variance ──
    logger.info("  Computing Markowitz Min Variance...")
    mkz_mv = markowitz_min_variance(aligned_full)
    portfolios["markowitz_min_var"] = {
        "weights": mkz_mv["weights"],
        "method": "Markowitz Monte Carlo (10K samples, min variance)",
    }

    # ── 3. HO-Sharpe Weighted ──
    ho_sharpes = np.array([max(r["ho_metrics"]["sharpe"], 0.01) for r in combo_results])
    ho_w = ho_sharpes / ho_sharpes.sum()
    portfolios["ho_sharpe_weighted"] = {
        "weights": ho_w,
        "method": "Weight proportional to holdout Sharpe",
    }

    # ── 4. Equal Weight ──
    eq_w = np.ones(n) / n
    portfolios["equal_weight"] = {
        "weights": eq_w,
        "method": "Equal weight (1/N)",
    }

    # ── 5. Risk Parity (inverse volatility) ──
    vols = np.array([np.std(r["full_returns"]) for r in combo_results])
    vols = np.maximum(vols, 1e-8)
    inv_vol = 1.0 / vols
    rp_w = inv_vol / inv_vol.sum()
    portfolios["risk_parity"] = {
        "weights": rp_w,
        "method": "Inverse volatility (risk parity)",
    }

    # ── Compute metrics for each portfolio on full + holdout ──
    for name, pf in portfolios.items():
        w = pf["weights"]

        # Full data metrics
        full_rets, full_eq = build_portfolio_equity(aligned_full, w)
        full_tf = "4h"  # Mixed TFs, use 4h as approximation
        full_metrics = compute_all_metrics(full_eq, full_tf)
        pf["full_returns"] = full_rets
        pf["full_equity"] = full_eq
        pf["full_metrics"] = full_metrics

        # Holdout metrics
        ho_rets, ho_eq = build_portfolio_equity(aligned_ho, w)
        ho_metrics = compute_all_metrics(ho_eq, full_tf)
        pf["ho_returns"] = ho_rets
        pf["ho_equity"] = ho_eq
        pf["ho_metrics"] = ho_metrics

        logger.info(f"  {name:<25} Full Sharpe={full_metrics['sharpe']:.2f} | "
                     f"HO Sharpe={ho_metrics['sharpe']:.2f} | "
                     f"HO Return={ho_metrics['total_return']:.1%}")

    return portfolios, corr_matrix


# ═══════════════════════════════════════════════════════════════
# STEP 4: Monte Carlo Stress Tests
# ═══════════════════════════════════════════════════════════════
def monte_carlo_bootstrap(returns, n_sims=1000, n_days=365):
    """Bootstrap resampling: draw random daily returns with replacement."""
    n = len(returns)
    if n < 10:
        return {"final_returns": np.zeros(n_sims), "max_drawdowns": np.zeros(n_sims),
                "sharpes": np.zeros(n_sims)}

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
        sharpes[sim] = sharpe_ratio(sim_returns, "4h")

    return {"final_returns": final_returns, "max_drawdowns": max_dds, "sharpes": sharpes}


def monte_carlo_block_bootstrap(returns, n_sims=1000, n_days=365, block_size=20):
    """Block bootstrap: preserves autocorrelation."""
    n = len(returns)
    if n < block_size:
        return {"final_returns": np.zeros(n_sims), "max_drawdowns": np.zeros(n_sims)}

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


def monte_carlo_ruin_probability(returns, n_sims=1000, n_days=365, ruin_threshold=0.5):
    """Probability of losing more than ruin_threshold of capital."""
    n = len(returns)
    if n < 10:
        return 0.0
    ruin_count = 0
    for sim in range(n_sims):
        idx = np.random.randint(0, n, size=n_days)
        sim_returns = returns[idx]
        sim_equity = INITIAL_CAPITAL * np.cumprod(1 + sim_returns)
        if np.min(sim_equity) < INITIAL_CAPITAL * (1 - ruin_threshold):
            ruin_count += 1
    return ruin_count / n_sims


def run_stress_tests(returns, name):
    """Run all Monte Carlo stress tests on a return series."""
    logger.info(f"  Stress tests: {name} ({N_MONTE_CARLO} sims)...")

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
# STEP 5: Leverage Testing
# ═══════════════════════════════════════════════════════════════
def test_leverage(aligned_returns, weights, name):
    """Test leverage levels on a portfolio."""
    leverage_results = {}
    for lev in LEVERAGE_LEVELS:
        rets, eq = build_portfolio_equity(aligned_returns, weights, leverage=lev)
        metrics = compute_all_metrics(eq, "4h")
        leverage_results[f"{name}_{lev:.0f}x"] = {
            "leverage": lev,
            "metrics": metrics,
            "returns": rets,
        }
        logger.info(f"  {name} {lev:.0f}x: Sharpe={metrics['sharpe']:.2f} | "
                     f"Return={metrics['total_return']:.1%} | DD={metrics['max_drawdown']:.1%}")
    return leverage_results


# ═══════════════════════════════════════════════════════════════
# STEP 6: Report Generation
# ═══════════════════════════════════════════════════════════════
def generate_markdown_report(combo_results, portfolios, corr_matrix,
                             stress_results, leverage_results, best_name):
    """Generate comprehensive Markdown report."""
    md = []
    md.append("# Portfolio V3 — Markowitz + Holdout Validation")
    md.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    md.append(f"**Cutoff holdout** : {CUTOFF_DATE}")
    md.append(f"**Seeds** : {N_SEEDS} par combo")
    md.append(f"**Monte Carlo** : {N_MONTE_CARLO} simulations")
    md.append(f"**Statut** : TERMINE")
    md.append("")
    md.append("---")
    md.append("")

    # ── 1. Combos ──
    md.append("## 1. Combos survivants")
    md.append("")
    md.append("| # | Symbol | Strategie | TF | Verdict HO | Full Sharpe | HO Sharpe | HO Return | HO DD |")
    md.append("|---|--------|-----------|-----|------------|-------------|-----------|-----------|-------|")

    for i, r in enumerate(combo_results):
        c = r["combo"]
        fm = r["full_metrics"]
        hm = r["ho_metrics"]
        md.append(
            f"| {i+1} | {c['symbol']} | {c['strategy']} | {c['timeframe']} | "
            f"{c['verdict']} | {fm['sharpe']:.2f} | {hm['sharpe']:.2f} | "
            f"{hm['total_return']:.1%} | {hm['max_drawdown']:.1%} |"
        )
    md.append("")

    # ── 2. Correlation Matrix ──
    md.append("## 2. Matrice de correlation")
    md.append("")
    labels = [f"{r['combo']['symbol'][:3]}/{r['combo']['strategy'][:8]}" for r in combo_results]
    header = "| | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    md.append(header)
    md.append(sep)
    for i, label in enumerate(labels):
        row = f"| **{label}** |"
        for j in range(len(labels)):
            val = corr_matrix[i, j] if i < corr_matrix.shape[0] and j < corr_matrix.shape[1] else 0
            row += f" {val:.2f} |"
        md.append(row)
    md.append("")

    # ── 3. Portfolio Comparison ──
    md.append("## 3. Comparaison des portfolios")
    md.append("")
    md.append("| Portfolio | Methode | Full Sharpe | HO Sharpe | HO Return | HO DD |")
    md.append("|-----------|---------|-------------|-----------|-----------|-------|")

    for name, pf in portfolios.items():
        fm = pf["full_metrics"]
        hm = pf["ho_metrics"]
        md.append(
            f"| {name} | {pf['method'][:40]} | {fm['sharpe']:.2f} | "
            f"{hm['sharpe']:.2f} | {hm['total_return']:.1%} | {hm['max_drawdown']:.1%} |"
        )
    md.append("")

    # ── 4. Allocations ──
    md.append("## 4. Allocations")
    md.append("")
    for pf_name, pf in portfolios.items():
        md.append(f"### {pf_name}")
        md.append("")
        md.append("| Combo | Poids |")
        md.append("|-------|-------|")
        w = pf["weights"]
        for i, r in enumerate(combo_results):
            c = r["combo"]
            combo_label = f"{c['symbol']}/{c['strategy']}/{c['timeframe']}"
            md.append(f"| {combo_label} | {w[i]:.1%} |")
        md.append("")

    # ── 5. Leverage ──
    md.append("## 5. Impact du Leverage")
    md.append("")
    md.append(f"### Portfolio : {best_name}")
    md.append("")
    md.append("| Leverage | Sharpe | Return | DD | Sortino |")
    md.append("|----------|--------|--------|-----|---------|")

    for lev in LEVERAGE_LEVELS:
        key = f"{best_name}_{lev:.0f}x"
        if key in leverage_results:
            m = leverage_results[key]["metrics"]
            md.append(
                f"| {lev:.0f}x | {m['sharpe']:.2f} | {m['total_return']:.1%} | "
                f"{m['max_drawdown']:.1%} | {m['sortino']:.2f} |"
            )
    md.append("")

    # ── 6. Monte Carlo ──
    md.append("## 6. Monte Carlo Stress Tests (1Y)")
    md.append("")
    md.append("| Portfolio | Med Ret | P5 Ret | P95 Ret | Med DD | Worst DD (P5) | P(ruin) |")
    md.append("|-----------|---------|--------|---------|--------|---------------|---------|")

    for key, st in stress_results.items():
        b = st["bootstrap_1Y"]
        md.append(
            f"| {key} | {b['median_return']:+.1%} | {b['p5_return']:+.1%} | "
            f"{b['p95_return']:+.1%} | {b['median_dd']:.1%} | "
            f"{b['p5_dd']:.1%} | {st['ruin_probability_50pct']:.1%} |"
        )
    md.append("")

    # ── 7. Multi-Year Projections ──
    best_stress_key = None
    for key in stress_results:
        if best_name in key and "_1x" in key:
            best_stress_key = key
            break
    if best_stress_key is None and stress_results:
        best_stress_key = list(stress_results.keys())[0]

    if best_stress_key and best_stress_key in stress_results:
        st = stress_results[best_stress_key]
        md.append(f"## 7. Projections multi-horizon ({best_stress_key})")
        md.append("")
        md.append("| Horizon | Med Ret | P(>0) | P(>10%) | P(>20%) | P5 | P95 |")
        md.append("|---------|---------|-------|---------|---------|-----|-----|")

        for horizon, proj in st["projections"].items():
            md.append(
                f"| {horizon} | {proj['median_return']:+.1%} | "
                f"{proj['prob_positive']:.1%} | {proj['prob_gt_10pct']:.1%} | "
                f"{proj['prob_gt_20pct']:.1%} | {proj['p5_return']:+.1%} | "
                f"{proj['p95_return']:+.1%} |"
            )

        md.append("")
        md.append(f"### Projections de capital (${INITIAL_CAPITAL:,.0f})")
        md.append("")
        md.append("| Horizon | Pessimiste (P5) | Median | Optimiste (P95) |")
        md.append("|---------|----------------|--------|-----------------|")

        for horizon, proj in st["projections"].items():
            p5 = INITIAL_CAPITAL * (1 + proj["p5_return"])
            med = INITIAL_CAPITAL * (1 + proj["median_return"])
            p95 = INITIAL_CAPITAL * (1 + proj["p95_return"])
            md.append(f"| {horizon} | ${p5:,.0f} | ${med:,.0f} | ${p95:,.0f} |")

    md.append("")

    # ── 8. Verdict ──
    md.append("## 8. Verdict")
    md.append("")

    best_pf = portfolios[best_name]
    ho_m = best_pf["ho_metrics"]
    full_m = best_pf["full_metrics"]

    md.append(f"**Meilleur portfolio** : `{best_name}`")
    md.append(f"- Full Sharpe : {full_m['sharpe']:.2f}")
    md.append(f"- Holdout Sharpe : {ho_m['sharpe']:.2f}")
    md.append(f"- Holdout Return : {ho_m['total_return']:.1%}")
    md.append(f"- Holdout Max DD : {ho_m['max_drawdown']:.1%}")
    md.append("")

    if best_stress_key and best_stress_key in stress_results:
        ruin = stress_results[best_stress_key]["ruin_probability_50pct"]
        if ruin < 0.01 and ho_m["sharpe"] > 0.3:
            md.append("**VERDICT : VIABLE pour deploiement live**")
        elif ho_m["sharpe"] > 0.1:
            md.append("**VERDICT : MARGINAL — ameliorations necessaires**")
        else:
            md.append("**VERDICT : NON RECOMMANDE pour deploiement live**")

    md.append("")
    md.append("---")
    md.append(f"*Genere le {datetime.now().strftime('%d %B %Y')}*")

    return "\n".join(md)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    logger.info("=" * 70)
    logger.info("  QUANTLAB V7 — PORTFOLIO V3: MARKOWITZ + HOLDOUT + MONTE CARLO")
    logger.info("=" * 70)

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    settings = load_settings()
    data_by_symbol = load_all_symbols_data(settings)
    risk = _build_risk(settings)
    cutoff_dt = pd.Timestamp(CUTOFF_DATE)

    # ── Step 1: Re-run walk-forwards ──
    logger.info("\n--- STEP 1: Re-running walk-forwards (full + holdout) ---")
    combo_results = run_all_combos(SURVIVORS, data_by_symbol, settings, risk, cutoff_dt)

    if len(combo_results) < 2:
        logger.error(f"Only {len(combo_results)} valid combos. Need at least 2. Aborting.")
        return

    logger.info(f"\n  {len(combo_results)} combos valid out of {len(SURVIVORS)}")

    # ── Step 2: Build portfolios ──
    logger.info("\n--- STEP 2: Building portfolios (Markowitz + baselines) ---")
    portfolios, corr_matrix = build_all_portfolios(combo_results)

    # Find best portfolio by holdout Sharpe
    best_name = max(portfolios.keys(), key=lambda k: portfolios[k]["ho_metrics"]["sharpe"])
    logger.info(f"\n  Best portfolio (by HO Sharpe): {best_name}")

    # ── Step 3: Leverage testing on best portfolio ──
    logger.info(f"\n--- STEP 3: Leverage testing on {best_name} ---")
    aligned_full = align_returns(combo_results, "full_returns")
    leverage_results = test_leverage(aligned_full, portfolios[best_name]["weights"], best_name)

    # ── Step 4: Monte Carlo stress tests ──
    logger.info("\n--- STEP 4: Monte Carlo stress tests ---")
    stress_results = {}

    # Stress test all portfolios at 1x
    for pf_name, pf in portfolios.items():
        key = f"{pf_name}_1x"
        stress_results[key] = run_stress_tests(pf["full_returns"], key)

    # Stress test best portfolio at all leverage levels
    for lev in LEVERAGE_LEVELS:
        lev_key = f"{best_name}_{lev:.0f}x"
        if lev_key in leverage_results and lev_key not in stress_results:
            stress_results[lev_key] = run_stress_tests(
                leverage_results[lev_key]["returns"], lev_key)

    # ── Step 5: Generate reports ──
    logger.info("\n--- STEP 5: Generating reports ---")

    md_report = generate_markdown_report(
        combo_results, portfolios, corr_matrix,
        stress_results, leverage_results, best_name)

    # JSON report
    json_report = {
        "timestamp": timestamp,
        "config": {
            "initial_capital": INITIAL_CAPITAL,
            "cutoff_date": CUTOFF_DATE,
            "n_seeds": N_SEEDS,
            "n_monte_carlo": N_MONTE_CARLO,
            "leverage_levels": LEVERAGE_LEVELS,
            "n_survivors": len(SURVIVORS),
        },
        "combos": [
            {
                "combo": r["combo"],
                "full_metrics": r["full_metrics"],
                "ho_metrics": r["ho_metrics"],
                "full_sharpe_seeds": [round(s, 4) for s in r["full_sharpe_seeds"]],
                "ho_sharpe_seeds": [round(s, 4) for s in r["ho_sharpe_seeds"]],
            }
            for r in combo_results
        ],
        "correlation_matrix": corr_matrix.tolist(),
        "portfolios": {
            name: {
                "weights": pf["weights"].tolist() if isinstance(pf["weights"], np.ndarray) else pf["weights"],
                "method": pf["method"],
                "full_metrics": pf["full_metrics"],
                "ho_metrics": pf["ho_metrics"],
            }
            for name, pf in portfolios.items()
        },
        "leverage_results": {
            key: {
                "leverage": lr["leverage"],
                "metrics": lr["metrics"],
            }
            for key, lr in leverage_results.items()
        },
        "stress_tests": stress_results,
        "best_portfolio": best_name,
    }

    # Save files
    Path("results").mkdir(exist_ok=True)
    Path("docs/results").mkdir(parents=True, exist_ok=True)

    json_path = f"results/portfolio_v3_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)

    md_path = "docs/results/11_portfolio_v3.md"
    with open(md_path, "w") as f:
        f.write(md_report)

    elapsed = time.time() - start_time

    # Print summary
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  PORTFOLIO V3 COMPLETE — {elapsed/60:.1f} min")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Combos: {len(combo_results)}")
    logger.info(f"  Best portfolio: {best_name}")

    best_pf = portfolios[best_name]
    logger.info(f"  Full Sharpe: {best_pf['full_metrics']['sharpe']:.2f}")
    logger.info(f"  HO Sharpe: {best_pf['ho_metrics']['sharpe']:.2f}")
    logger.info(f"  HO Return: {best_pf['ho_metrics']['total_return']:.1%}")
    logger.info(f"  HO Max DD: {best_pf['ho_metrics']['max_drawdown']:.1%}")

    best_stress_1x = stress_results.get(f"{best_name}_1x", {})
    if best_stress_1x:
        b = best_stress_1x["bootstrap_1Y"]
        logger.info(f"  MC Median Return (1Y): {b['median_return']:+.1%}")
        logger.info(f"  MC P5 Return (1Y): {b['p5_return']:+.1%}")
        logger.info(f"  MC Ruin Prob: {best_stress_1x['ruin_probability_50pct']:.1%}")

    logger.info(f"\n  Allocations ({best_name}):")
    w = best_pf["weights"]
    for i, r in enumerate(combo_results):
        c = r["combo"]
        wi = w[i] if isinstance(w, np.ndarray) else w[i]
        logger.info(f"    {c['symbol']}/{c['strategy']}/{c['timeframe']}: {wi:.1%}")

    logger.info(f"\n  Saved: {json_path}")
    logger.info(f"  Saved: {md_path}")
    logger.info(f"  Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
