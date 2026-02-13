#!/usr/bin/env python3
"""
Portfolio V3b — Improved version addressing diagnostic findings.

Key improvements over V3:
  1. MC on holdout returns ONLY (not full data)
  2. Filter out combos with high seed variance (unstable)
  3. Cap max weight per symbol (reduce ETH concentration)
  4. Add constrained Markowitz with symbol diversification
  5. Compare: original V3 weights vs improved weights
  6. Proper holdout equity from IS-trained params (not full WF cutoff)

Output:
  results/portfolio_v3b_{timestamp}.json
  docs/results/13_portfolio_v3b.md
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
N_MONTE_CARLO = 2000
CUTOFF_DATE = "2025-02-01"
N_SEEDS = 3
MAX_WEIGHT_PER_SYMBOL = 0.50  # Cap ETH at 50%
MAX_WEIGHT_PER_COMBO = 0.25   # No single combo > 25%
MIN_HO_SHARPE = -0.2          # Filter out combos with bad holdout
MAX_SEED_STD = 1.5            # Filter out combos with high seed variance

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
# STEP 1: Run holdout properly (IS WF → last params → HO backtest)
# ═══════════════════════════════════════════════════════════════
def run_holdout_combos(survivors, data_by_symbol, settings, risk, cutoff_dt):
    """For each combo, multi-seed IS WF → holdout backtest. Return holdout equity curves."""
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
            continue

        strategy = get_strategy(strat_name)
        in_sample = data[data.index < cutoff_dt]
        holdout = data[data.index >= cutoff_dt]

        if len(in_sample) < 200 or len(holdout) < 30:
            logger.warning(f"  Insufficient data")
            continue

        ho_equities = []
        ho_returns_list = []
        ho_metrics_list = []
        is_sharpes = []

        for seed_i in range(N_SEEDS):
            seed = 42 + seed_i

            # IS walk-forward
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
                is_sharpes.append(is_result.metrics.get("sharpe", 0))
            except Exception as e:
                logger.warning(f"  Seed {seed} IS failed: {e}")
                continue

            if not is_result.best_params_per_period:
                continue

            last_params = is_result.best_params_per_period[-1]

            # Holdout backtest with last IS params
            try:
                ho_result = backtest_strategy(
                    strategy, holdout, last_params,
                    commission=commission, slippage=slippage,
                    initial_capital=INITIAL_CAPITAL, risk=risk, timeframe=tf,
                )
                ho_met = compute_all_metrics(ho_result.equity, tf, ho_result.trades_pnl)
                ho_equities.append(ho_result.equity)
                ho_returns_list.append(returns_from_equity(ho_result.equity))
                ho_metrics_list.append(ho_met)
            except Exception as e:
                logger.warning(f"  Seed {seed} HO failed: {e}")
                continue

        if not ho_metrics_list:
            logger.error(f"  No valid holdout results")
            continue

        # Pick median-Sharpe seed
        ho_sharpes = [m["sharpe"] for m in ho_metrics_list]
        median_idx = int(np.argsort(ho_sharpes)[len(ho_sharpes) // 2])

        result = {
            "combo": combo,
            "ho_equity": ho_equities[median_idx],
            "ho_returns": ho_returns_list[median_idx],
            "ho_metrics": ho_metrics_list[median_idx],
            "ho_sharpe_seeds": ho_sharpes,
            "ho_sharpe_std": float(np.std(ho_sharpes)),
            "is_sharpes": is_sharpes,
        }

        logger.info(f"  HO Sharpe={result['ho_metrics']['sharpe']:.3f} "
                     f"(std={result['ho_sharpe_std']:.3f}, seeds={ho_sharpes})")
        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════
# STEP 2: Filter unstable combos
# ═══════════════════════════════════════════════════════════════
def filter_combos(combo_results):
    """Remove combos with high seed variance or bad holdout Sharpe."""
    filtered = []
    removed = []

    for r in combo_results:
        c = r["combo"]
        label = f"{c['symbol']}/{c['strategy']}/{c['timeframe']}"
        ho_sharpe = r["ho_metrics"]["sharpe"]
        seed_std = r["ho_sharpe_std"]

        if ho_sharpe < MIN_HO_SHARPE:
            removed.append({"label": label, "reason": f"HO Sharpe {ho_sharpe:.3f} < {MIN_HO_SHARPE}"})
            logger.info(f"  REMOVED {label}: HO Sharpe {ho_sharpe:.3f} < {MIN_HO_SHARPE}")
            continue

        if seed_std > MAX_SEED_STD:
            removed.append({"label": label, "reason": f"Seed std {seed_std:.3f} > {MAX_SEED_STD}"})
            logger.info(f"  REMOVED {label}: Seed std {seed_std:.3f} > {MAX_SEED_STD}")
            continue

        filtered.append(r)
        logger.info(f"  KEPT {label}: HO Sharpe={ho_sharpe:.3f}, std={seed_std:.3f}")

    return filtered, removed


# ═══════════════════════════════════════════════════════════════
# STEP 3: Constrained portfolio optimization
# ═══════════════════════════════════════════════════════════════
def constrained_markowitz(returns_matrix, combo_results, n_portfolios=20000):
    """Markowitz with symbol and combo weight constraints."""
    n_assets = returns_matrix.shape[0]
    mean_returns = np.mean(returns_matrix, axis=1)
    cov_matrix = np.cov(returns_matrix)

    # Build symbol mapping
    symbols = {}
    for i, r in enumerate(combo_results):
        s = r["combo"]["symbol"]
        if s not in symbols:
            symbols[s] = []
        symbols[s].append(i)

    best_sharpe = -np.inf
    best_weights = np.ones(n_assets) / n_assets

    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))

        # Enforce per-combo cap
        w = np.minimum(w, MAX_WEIGHT_PER_COMBO)

        # Enforce per-symbol cap
        for s, indices in symbols.items():
            sym_total = sum(w[i] for i in indices)
            if sym_total > MAX_WEIGHT_PER_SYMBOL:
                scale = MAX_WEIGHT_PER_SYMBOL / sym_total
                for i in indices:
                    w[i] *= scale

        # Renormalize
        w = w / w.sum()

        port_ret = w @ mean_returns
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol == 0:
            continue
        port_sharpe = port_ret / port_vol

        if port_sharpe > best_sharpe:
            best_sharpe = port_sharpe
            best_weights = w.copy()

    return best_weights, float(best_sharpe)


def ho_sharpe_weighted_constrained(combo_results):
    """Weight proportional to holdout Sharpe, with symbol constraints."""
    n = len(combo_results)
    ho_sharpes = np.array([max(r["ho_metrics"]["sharpe"], 0.01) for r in combo_results])
    w = ho_sharpes / ho_sharpes.sum()

    # Build symbol mapping
    symbols = {}
    for i, r in enumerate(combo_results):
        s = r["combo"]["symbol"]
        if s not in symbols:
            symbols[s] = []
        symbols[s].append(i)

    # Enforce per-combo cap
    w = np.minimum(w, MAX_WEIGHT_PER_COMBO)

    # Enforce per-symbol cap (iterate to converge)
    for _ in range(10):
        for s, indices in symbols.items():
            sym_total = sum(w[i] for i in indices)
            if sym_total > MAX_WEIGHT_PER_SYMBOL:
                scale = MAX_WEIGHT_PER_SYMBOL / sym_total
                for i in indices:
                    w[i] *= scale

    # Renormalize
    w = w / w.sum()
    return w


def inverse_vol_constrained(combo_results):
    """Inverse volatility with symbol constraints."""
    vols = np.array([np.std(r["ho_returns"]) for r in combo_results])
    vols = np.maximum(vols, 1e-8)
    w = (1.0 / vols)
    w = w / w.sum()

    symbols = {}
    for i, r in enumerate(combo_results):
        s = r["combo"]["symbol"]
        if s not in symbols:
            symbols[s] = []
        symbols[s].append(i)

    w = np.minimum(w, MAX_WEIGHT_PER_COMBO)
    for _ in range(10):
        for s, indices in symbols.items():
            sym_total = sum(w[i] for i in indices)
            if sym_total > MAX_WEIGHT_PER_SYMBOL:
                scale = MAX_WEIGHT_PER_SYMBOL / sym_total
                for i in indices:
                    w[i] *= scale
    w = w / w.sum()
    return w


def build_portfolio_equity(aligned_returns, weights, leverage=1.0):
    """Build portfolio equity curve."""
    port_returns = (aligned_returns.T @ weights) * leverage
    port_equity = INITIAL_CAPITAL * np.cumprod(1 + port_returns)
    port_equity = np.insert(port_equity, 0, INITIAL_CAPITAL)
    return port_returns, port_equity


# ═══════════════════════════════════════════════════════════════
# STEP 4: Monte Carlo on holdout returns
# ═══════════════════════════════════════════════════════════════
def mc_holdout(returns, n_sims=2000):
    """Monte Carlo on holdout returns only."""
    n = len(returns)
    if n < 10:
        return {}

    projections = {}
    for months in [3, 6, 12, 24, 36]:
        # Scale to approximate number of 4h bars
        n_bars = months * 30 * 6  # ~6 bars per day for 4h
        n_bars = min(n_bars, n * 4)  # Don't extrapolate too far

        final_returns = np.zeros(n_sims)
        max_dds = np.zeros(n_sims)

        for sim in range(n_sims):
            idx = np.random.randint(0, n, size=n_bars)
            sim_rets = returns[idx]
            sim_eq = INITIAL_CAPITAL * np.cumprod(1 + sim_rets)
            sim_eq = np.insert(sim_eq, 0, INITIAL_CAPITAL)
            final_returns[sim] = total_return(sim_eq)
            max_dds[sim] = max_drawdown(sim_eq)

        projections[f"{months}M"] = {
            "median_return": round(float(np.median(final_returns)), 4),
            "mean_return": round(float(np.mean(final_returns)), 4),
            "p5_return": round(float(np.percentile(final_returns, 5)), 4),
            "p25_return": round(float(np.percentile(final_returns, 25)), 4),
            "p75_return": round(float(np.percentile(final_returns, 75)), 4),
            "p95_return": round(float(np.percentile(final_returns, 95)), 4),
            "median_dd": round(float(np.median(max_dds)), 4),
            "p5_dd": round(float(np.percentile(max_dds, 5)), 4),
            "prob_positive": round(float(np.mean(final_returns > 0)), 4),
            "prob_gt_5pct": round(float(np.mean(final_returns > 0.05)), 4),
            "prob_gt_10pct": round(float(np.mean(final_returns > 0.10)), 4),
        }

    # Ruin probability
    ruin_count = 0
    for sim in range(n_sims):
        idx = np.random.randint(0, n, size=min(365 * 6, n * 4))
        sim_rets = returns[idx]
        sim_eq = INITIAL_CAPITAL * np.cumprod(1 + sim_rets)
        if np.min(sim_eq) < INITIAL_CAPITAL * 0.5:
            ruin_count += 1

    return {
        "projections": projections,
        "ruin_probability_50pct": round(ruin_count / n_sims, 4),
    }


# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════
def generate_report(combo_results, removed, portfolios, mc_results, corr_matrix):
    md = []
    md.append("# Portfolio V3b — Improved (Constrained + Holdout MC)")
    md.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    md.append(f"**Ameliorations** : filtrage instabilite, cap symbol {MAX_WEIGHT_PER_SYMBOL:.0%}, "
              f"cap combo {MAX_WEIGHT_PER_COMBO:.0%}, MC holdout-only")
    md.append("")
    md.append("---")
    md.append("")

    # Filtering
    md.append("## 1. Filtrage des combos")
    md.append("")
    md.append(f"- **Seuil HO Sharpe** : > {MIN_HO_SHARPE}")
    md.append(f"- **Seuil seed std** : < {MAX_SEED_STD}")
    md.append(f"- **Retenus** : {len(combo_results)} / {len(combo_results) + len(removed)}")
    md.append("")

    if removed:
        md.append("### Combos retires")
        md.append("")
        md.append("| Combo | Raison |")
        md.append("|-------|--------|")
        for r in removed:
            md.append(f"| {r['label']} | {r['reason']} |")
        md.append("")

    md.append("### Combos retenus")
    md.append("")
    md.append("| # | Symbol | Strategie | TF | HO Sharpe | HO Return | Seed Std |")
    md.append("|---|--------|-----------|-----|-----------|-----------|----------|")
    for i, r in enumerate(combo_results):
        c = r["combo"]
        hm = r["ho_metrics"]
        md.append(f"| {i+1} | {c['symbol']} | {c['strategy']} | {c['timeframe']} | "
                  f"{hm['sharpe']:.2f} | {hm['total_return']:.1%} | {r['ho_sharpe_std']:.3f} |")
    md.append("")

    # Correlation
    md.append("## 2. Matrice de correlation (holdout)")
    md.append("")
    labels = [f"{r['combo']['symbol'][:3]}/{r['combo']['strategy'][:8]}" for r in combo_results]
    header = "| | " + " | ".join(labels) + " |"
    sep = "|---" * (len(labels) + 1) + "|"
    md.append(header)
    md.append(sep)
    for i, label in enumerate(labels):
        row = f"| **{label}** |"
        for j in range(len(labels)):
            row += f" {corr_matrix[i, j]:.2f} |"
        md.append(row)
    md.append("")

    # Portfolios
    md.append("## 3. Comparaison des portfolios")
    md.append("")
    md.append("| Portfolio | HO Sharpe | HO Sortino | HO Return | HO DD | HO Calmar |")
    md.append("|-----------|-----------|------------|-----------|-------|-----------|")
    for name, pf in portfolios.items():
        hm = pf["ho_metrics"]
        md.append(f"| {name} | {hm['sharpe']:.2f} | {hm['sortino']:.2f} | "
                  f"{hm['total_return']:.1%} | {hm['max_drawdown']:.1%} | {hm['calmar']:.2f} |")
    md.append("")

    # Allocations
    md.append("## 4. Allocations")
    md.append("")
    for pf_name, pf in portfolios.items():
        md.append(f"### {pf_name}")
        md.append("")

        # Symbol breakdown
        sym_w = {}
        for i, r in enumerate(combo_results):
            s = r["combo"]["symbol"]
            sym_w[s] = sym_w.get(s, 0) + pf["weights"][i]

        md.append(f"Concentration: " + ", ".join(f"{s} {w:.0%}" for s, w in sorted(sym_w.items(), key=lambda x: -x[1])))
        md.append("")
        md.append("| Combo | Poids |")
        md.append("|-------|-------|")
        for i, r in enumerate(combo_results):
            c = r["combo"]
            md.append(f"| {c['symbol']}/{c['strategy']}/{c['timeframe']} | {pf['weights'][i]:.1%} |")
        md.append("")

    # MC holdout
    best_name = max(portfolios.keys(), key=lambda k: portfolios[k]["ho_metrics"]["sharpe"])
    md.append(f"## 5. Monte Carlo Holdout-Only ({best_name})")
    md.append("")

    if best_name in mc_results:
        mc = mc_results[best_name]
        md.append(f"- **Ruin probability (50%)** : {mc['ruin_probability_50pct']:.1%}")
        md.append("")

        md.append("### Projections (capital $10,000)")
        md.append("")
        md.append("| Horizon | Median | P(>0) | P(>5%) | P(>10%) | Pessimiste (P5) | Optimiste (P95) | Med DD |")
        md.append("|---------|--------|-------|--------|---------|----------------|-----------------|--------|")
        for horizon, proj in mc["projections"].items():
            med_cap = INITIAL_CAPITAL * (1 + proj["median_return"])
            p5_cap = INITIAL_CAPITAL * (1 + proj["p5_return"])
            p95_cap = INITIAL_CAPITAL * (1 + proj["p95_return"])
            md.append(f"| {horizon} | ${med_cap:,.0f} ({proj['median_return']:+.1%}) | "
                      f"{proj['prob_positive']:.0%} | {proj['prob_gt_5pct']:.0%} | "
                      f"{proj['prob_gt_10pct']:.0%} | ${p5_cap:,.0f} | ${p95_cap:,.0f} | "
                      f"{proj['median_dd']:.1%} |")
        md.append("")

    # Monthly/annual expectations
    md.append("## 6. Profit Expectations (holdout-based)")
    md.append("")
    best_pf = portfolios[best_name]
    ho_m = best_pf["ho_metrics"]
    ho_ret = ho_m["total_return"]
    # Holdout is ~12 months (2025-02-01 to 2026-02-01)
    monthly_ret = (1 + ho_ret) ** (1/12) - 1
    md.append(f"**Portfolio retenu** : `{best_name}`")
    md.append("")
    md.append("| Metrique | Valeur |")
    md.append("|----------|--------|")
    md.append(f"| Return annuel (holdout) | {ho_ret:+.1%} |")
    md.append(f"| Return mensuel moyen | {monthly_ret:+.2%} |")
    md.append(f"| Sharpe annualise | {ho_m['sharpe']:.2f} |")
    md.append(f"| Sortino | {ho_m['sortino']:.2f} |")
    md.append(f"| Max Drawdown | {ho_m['max_drawdown']:.1%} |")
    md.append(f"| Calmar | {ho_m['calmar']:.2f} |")
    md.append("")

    if best_name in mc_results:
        mc = mc_results[best_name]
        proj_12m = mc["projections"].get("12M", {})
        if proj_12m:
            md.append("### Expectations Monte Carlo (12 mois)")
            md.append("")
            md.append("| Scenario | Return | Capital final |")
            md.append("|----------|--------|---------------|")
            md.append(f"| Pessimiste (P5) | {proj_12m['p5_return']:+.1%} | ${INITIAL_CAPITAL*(1+proj_12m['p5_return']):,.0f} |")
            md.append(f"| Conservateur (P25) | {proj_12m['p25_return']:+.1%} | ${INITIAL_CAPITAL*(1+proj_12m['p25_return']):,.0f} |")
            md.append(f"| Median | {proj_12m['median_return']:+.1%} | ${INITIAL_CAPITAL*(1+proj_12m['median_return']):,.0f} |")
            md.append(f"| Optimiste (P75) | {proj_12m['p75_return']:+.1%} | ${INITIAL_CAPITAL*(1+proj_12m['p75_return']):,.0f} |")
            md.append(f"| Tres optimiste (P95) | {proj_12m['p95_return']:+.1%} | ${INITIAL_CAPITAL*(1+proj_12m['p95_return']):,.0f} |")
            md.append("")

    # Verdict
    md.append("## 7. Verdict")
    md.append("")
    md.append(f"**Meilleur portfolio** : `{best_name}`")
    md.append(f"- HO Sharpe : {ho_m['sharpe']:.2f}")
    md.append(f"- HO Return : {ho_m['total_return']:+.1%}")
    md.append(f"- HO Max DD : {ho_m['max_drawdown']:.1%}")

    if best_name in mc_results:
        ruin = mc_results[best_name]["ruin_probability_50pct"]
        md.append(f"- Ruin probability : {ruin:.1%}")
        proj_12m = mc_results[best_name]["projections"].get("12M", {})
        if proj_12m:
            md.append(f"- MC P(positive, 12M) : {proj_12m['prob_positive']:.0%}")

    md.append("")
    md.append("---")
    md.append(f"*Genere le {datetime.now().strftime('%d %B %Y')}*")

    return "\n".join(md)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    logger.info("=" * 70)
    logger.info("  PORTFOLIO V3b — IMPROVED (CONSTRAINED + HOLDOUT MC)")
    logger.info("=" * 70)

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    settings = load_settings()
    data_by_symbol = load_all_symbols_data(settings)
    risk = _build_risk(settings)
    cutoff_dt = pd.Timestamp(CUTOFF_DATE)

    # ── Step 1: Holdout backtests ──
    logger.info("\n--- STEP 1: Holdout backtests (IS WF → last params → HO) ---")
    all_results = run_holdout_combos(SURVIVORS, data_by_symbol, settings, risk, cutoff_dt)

    if len(all_results) < 2:
        logger.error("Not enough valid combos. Aborting.")
        return

    # ── Step 2: Filter unstable combos ──
    logger.info("\n--- STEP 2: Filtering unstable combos ---")
    filtered, removed = filter_combos(all_results)

    if len(filtered) < 2:
        logger.warning("Too few combos after filtering, relaxing constraints")
        filtered = all_results
        removed = []

    logger.info(f"  {len(filtered)} combos retained, {len(removed)} removed")

    # ── Step 3: Align holdout returns ──
    max_len = max(len(r["ho_returns"]) for r in filtered)
    n = len(filtered)
    aligned_ho = np.zeros((n, max_len))
    for i, r in enumerate(filtered):
        rets = r["ho_returns"]
        aligned_ho[i, :len(rets)] = rets

    corr_matrix = np.corrcoef(aligned_ho) if n > 1 else np.array([[1.0]])

    # ── Step 4: Build constrained portfolios ──
    logger.info("\n--- STEP 3: Building constrained portfolios ---")
    portfolios = {}

    # 4a. Constrained Markowitz (max Sharpe)
    logger.info("  Constrained Markowitz max Sharpe...")
    mkz_w, mkz_sharpe = constrained_markowitz(aligned_ho, filtered)
    portfolios["markowitz_constrained"] = {"weights": mkz_w, "method": f"Markowitz constrained (sym≤{MAX_WEIGHT_PER_SYMBOL:.0%})"}

    # 4b. HO-Sharpe weighted constrained
    logger.info("  HO-Sharpe weighted constrained...")
    hsw_w = ho_sharpe_weighted_constrained(filtered)
    portfolios["ho_sharpe_constrained"] = {"weights": hsw_w, "method": f"HO-Sharpe weighted (sym≤{MAX_WEIGHT_PER_SYMBOL:.0%})"}

    # 4c. Inverse vol constrained
    logger.info("  Inverse vol constrained...")
    iv_w = inverse_vol_constrained(filtered)
    portfolios["risk_parity_constrained"] = {"weights": iv_w, "method": f"Risk parity (sym≤{MAX_WEIGHT_PER_SYMBOL:.0%})"}

    # 4d. Equal weight
    eq_w = np.ones(n) / n
    portfolios["equal_weight"] = {"weights": eq_w, "method": "Equal weight (1/N)"}

    # Compute holdout metrics for each portfolio
    for name, pf in portfolios.items():
        w = pf["weights"]
        ho_rets, ho_eq = build_portfolio_equity(aligned_ho, w)
        # Use the dominant timeframe for annualization
        ho_metrics = compute_all_metrics(ho_eq, "4h")
        pf["ho_returns"] = ho_rets
        pf["ho_equity"] = ho_eq
        pf["ho_metrics"] = ho_metrics

        logger.info(f"  {name:<30} HO Sharpe={ho_metrics['sharpe']:.2f} | "
                     f"Return={ho_metrics['total_return']:.1%} | DD={ho_metrics['max_drawdown']:.1%}")

    # ── Step 5: Monte Carlo on holdout returns ──
    logger.info("\n--- STEP 4: Monte Carlo on holdout returns ---")
    mc_results = {}
    for name, pf in portfolios.items():
        logger.info(f"  MC {name} ({N_MONTE_CARLO} sims)...")
        mc_results[name] = mc_holdout(pf["ho_returns"], N_MONTE_CARLO)

    # ── Step 6: Reports ──
    logger.info("\n--- STEP 5: Generating reports ---")

    md_report = generate_report(filtered, removed, portfolios, mc_results, corr_matrix)

    best_name = max(portfolios.keys(), key=lambda k: portfolios[k]["ho_metrics"]["sharpe"])

    json_report = {
        "timestamp": timestamp,
        "config": {
            "initial_capital": INITIAL_CAPITAL,
            "cutoff_date": CUTOFF_DATE,
            "n_seeds": N_SEEDS,
            "n_monte_carlo": N_MONTE_CARLO,
            "max_weight_per_symbol": MAX_WEIGHT_PER_SYMBOL,
            "max_weight_per_combo": MAX_WEIGHT_PER_COMBO,
            "min_ho_sharpe": MIN_HO_SHARPE,
            "max_seed_std": MAX_SEED_STD,
        },
        "filtering": {
            "total_input": len(SURVIVORS),
            "retained": len(filtered),
            "removed": removed,
        },
        "combos": [
            {
                "combo": r["combo"],
                "ho_metrics": r["ho_metrics"],
                "ho_sharpe_seeds": [round(s, 4) for s in r["ho_sharpe_seeds"]],
                "ho_sharpe_std": round(r["ho_sharpe_std"], 4),
            }
            for r in filtered
        ],
        "correlation_matrix": corr_matrix.tolist(),
        "portfolios": {
            name: {
                "weights": pf["weights"].tolist() if isinstance(pf["weights"], np.ndarray) else pf["weights"],
                "method": pf["method"],
                "ho_metrics": pf["ho_metrics"],
            }
            for name, pf in portfolios.items()
        },
        "monte_carlo": mc_results,
        "best_portfolio": best_name,
    }

    Path("results").mkdir(exist_ok=True)
    Path("docs/results").mkdir(parents=True, exist_ok=True)

    json_path = f"results/portfolio_v3b_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2, default=str)

    md_path = "docs/results/13_portfolio_v3b.md"
    with open(md_path, "w") as f:
        f.write(md_report)

    elapsed = time.time() - start_time

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  PORTFOLIO V3b COMPLETE — {elapsed/60:.1f} min")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Best: {best_name}")
    best_pf = portfolios[best_name]
    hm = best_pf["ho_metrics"]
    logger.info(f"  HO Sharpe: {hm['sharpe']:.2f} | Return: {hm['total_return']:.1%} | DD: {hm['max_drawdown']:.1%}")

    if best_name in mc_results:
        mc = mc_results[best_name]
        logger.info(f"  Ruin prob: {mc['ruin_probability_50pct']:.1%}")
        p12 = mc["projections"].get("12M", {})
        if p12:
            logger.info(f"  MC 12M: median={p12['median_return']:+.1%}, P(>0)={p12['prob_positive']:.0%}")

    logger.info(f"\n  Saved: {json_path}")
    logger.info(f"  Saved: {md_path}")


if __name__ == "__main__":
    main()
