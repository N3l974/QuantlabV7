#!/usr/bin/env python3
"""
Portfolio V4 — Edge-Enhanced Portfolio Construction

Uses 39 survivors from diagnostic V4 (walk-forward + overlays).
Key improvements over V3b:
  1. Overlays integrated (regime + vol targeting)
  2. Hard constraints: ETH ≤ 60%, per-symbol cap, per-combo cap
  3. Markowitz with covariance shrinkage (Ledoit-Wolf)
  4. Correlation-based deduplication
  5. Monte Carlo on holdout only
  6. Multiple portfolio methods compared

Output:
  results/portfolio_v4_{timestamp}.json
  docs/results/16_portfolio_v4.md
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.backtester import RiskConfig, vectorized_backtest
from engine.metrics import (
    compute_all_metrics, max_drawdown, returns_from_equity,
    sharpe_ratio, total_return,
)
from engine.walk_forward import WalkForwardConfig, run_walk_forward
from engine.overlays import (
    apply_overlay_pipeline, OverlayPipelineConfig,
    VolTargetConfig, RegimeOverlayConfig,
)
from engine.regime import RegimeConfig
from strategies.registry import get_strategy

# ── Config ──────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
N_MONTE_CARLO = 2000
CUTOFF_DATE = "2025-02-01"
RISK = RiskConfig()

# Hard constraints
MAX_WEIGHT_PER_SYMBOL = 0.60    # ETH cap
MAX_WEIGHT_PER_COMBO = 0.20     # No single combo > 20%
MIN_WEIGHT = 0.01               # Minimum allocation
MAX_CORRELATION = 0.85          # Deduplicate highly correlated combos
MIN_HO_SHARPE = 0.0             # Only positive holdout Sharpe
MIN_HO_TRADES = 3               # Minimum trades on holdout

# Overlay config
REGIME_CFG = RegimeOverlayConfig(
    regime_config=RegimeConfig(),
    hard_cutoff=True,
    min_exposure_threshold=0.3,
)
VOL_CFG = VolTargetConfig(target_vol_annual=0.30)
OVERLAY_CFG = OverlayPipelineConfig(regime_config=REGIME_CFG, vol_config=VOL_CFG)

# WF config for rebuilding
WF_TRIALS = 30
WF_REOPTIM = "3M"
WF_WINDOW = "1Y"

# Load diagnostic results
DIAG_PATH = sorted(Path("results").glob("diagnostic_v4_fast_*.json"))[-1]


# ── Data Loading ────────────────────────────────────────

def load_survivors():
    """Load survivors from diagnostic JSON."""
    with open(DIAG_PATH) as f:
        data = json.load(f)

    # Get unique survivors (best of baseline/overlay per combo)
    p2 = data["phase2"]
    survivors = [r for r in p2
                 if r["verdict"] in ("STRONG", "WEAK")
                 and r["ho_trades"] >= MIN_HO_TRADES
                 and r["ho_sharpe"] >= MIN_HO_SHARPE]

    # Deduplicate: keep best version per (symbol, strategy, timeframe)
    best = {}
    for r in survivors:
        key = (r["symbol"], r["strategy"], r["timeframe"])
        if key not in best or r["ho_sharpe"] > best[key]["ho_sharpe"]:
            best[key] = r

    survivors = sorted(best.values(), key=lambda x: x["ho_sharpe"], reverse=True)
    logger.info(f"Loaded {len(survivors)} unique survivors from {DIAG_PATH}")
    return survivors


def rebuild_holdout_equities(survivors):
    """
    Rebuild holdout equity curves for each survivor.
    Uses last WF params + overlay if flagged.
    """
    logger.info("Rebuilding holdout equity curves...")
    equities = {}
    returns_dict = {}
    valid_survivors = []

    for i, s in enumerate(survivors):
        sym, sname, tf = s["symbol"], s["strategy"], s["timeframe"]
        use_overlay = s["overlay"]
        last_params = s["last_params"]
        key = f"{sym}/{sname}/{tf}"

        try:
            data = pd.read_parquet(f"data/raw/{sym}_{tf}.parquet")
            ho_data = data[data.index >= CUTOFF_DATE].copy()

            if len(ho_data) < 50:
                continue

            strategy = get_strategy(sname)
            signals = strategy.generate_signals(ho_data, last_params)

            if use_overlay:
                signals, _ = apply_overlay_pipeline(
                    signals, ho_data, OVERLAY_CFG, timeframe=tf
                )

            close = ho_data["close"].values.astype(np.float64)
            high = ho_data["high"].values.astype(np.float64)
            low = ho_data["low"].values.astype(np.float64)

            res = vectorized_backtest(
                close, signals, risk=RISK, high=high, low=low, timeframe=tf
            )

            rets = returns_from_equity(res.equity)
            equities[key] = res.equity
            returns_dict[key] = rets
            s["_equity"] = res.equity
            s["_returns"] = rets
            s["_key"] = key
            valid_survivors.append(s)

            logger.info(f"  [{i+1}/{len(survivors)}] {key} {'(+ov)' if use_overlay else ''}: "
                       f"Sharpe={s['ho_sharpe']:.3f}, Ret={s['ho_return']*100:.1f}%")

        except Exception as e:
            logger.error(f"  [{i+1}] {key}: {e}")

    logger.info(f"Rebuilt {len(valid_survivors)}/{len(survivors)} equity curves")
    return valid_survivors, equities, returns_dict


# ── Correlation Deduplication ───────────────────────────

def deduplicate_by_correlation(survivors, returns_dict, max_corr=MAX_CORRELATION):
    """Remove highly correlated combos, keeping the one with higher Sharpe."""
    if len(survivors) <= 1:
        return survivors

    # Align returns to same length (min)
    min_len = min(len(r) for r in returns_dict.values())
    keys = [s["_key"] for s in survivors]
    rets_matrix = np.column_stack([returns_dict[k][:min_len] for k in keys])

    corr = np.corrcoef(rets_matrix.T)
    to_remove = set()

    for i in range(len(keys)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(keys)):
            if j in to_remove:
                continue
            if abs(corr[i, j]) > max_corr:
                # Remove the one with lower Sharpe
                if survivors[i]["ho_sharpe"] >= survivors[j]["ho_sharpe"]:
                    to_remove.add(j)
                    logger.info(f"  Dedup: removing {keys[j]} (corr={corr[i,j]:.2f} with {keys[i]})")
                else:
                    to_remove.add(i)
                    logger.info(f"  Dedup: removing {keys[i]} (corr={corr[i,j]:.2f} with {keys[j]})")

    filtered = [s for idx, s in enumerate(survivors) if idx not in to_remove]
    logger.info(f"Correlation dedup: {len(survivors)} → {len(filtered)} (removed {len(to_remove)})")
    return filtered


# ── Covariance Shrinkage (Ledoit-Wolf) ──────────────────

def ledoit_wolf_shrinkage(returns_matrix):
    """Ledoit-Wolf covariance shrinkage estimator."""
    T, N = returns_matrix.shape
    X = returns_matrix - returns_matrix.mean(axis=0)

    # Sample covariance
    S = X.T @ X / T

    # Shrinkage target: diagonal (constant correlation)
    mu = np.trace(S) / N
    F = mu * np.eye(N)

    # Compute optimal shrinkage intensity
    d2 = np.sum((S - F) ** 2) / N

    # Estimate b2
    b2_sum = 0
    for t in range(T):
        xt = X[t:t+1, :]
        Mt = xt.T @ xt - S
        b2_sum += np.sum(Mt ** 2) / N
    b2 = b2_sum / (T ** 2)

    b2 = min(b2, d2)
    alpha = b2 / max(d2, 1e-10)
    alpha = np.clip(alpha, 0, 1)

    shrunk = alpha * F + (1 - alpha) * S
    logger.info(f"Ledoit-Wolf shrinkage: alpha={alpha:.3f}")
    return shrunk


# ── Portfolio Optimization ──────────────────────────────

def markowitz_constrained(survivors, returns_dict):
    """
    Markowitz mean-variance optimization with hard constraints.
    Maximizes Sharpe ratio subject to:
    - Per-symbol cap (ETH ≤ 60%)
    - Per-combo cap (≤ 20%)
    - Min weight (≥ 1%)
    - Weights sum to 1
    """
    n = len(survivors)
    keys = [s["_key"] for s in survivors]
    min_len = min(len(returns_dict[k]) for k in keys)
    rets_matrix = np.column_stack([returns_dict[k][:min_len] for k in keys])

    # Shrunk covariance
    cov = ledoit_wolf_shrinkage(rets_matrix)
    mu = rets_matrix.mean(axis=0)

    # Objective: negative Sharpe
    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return 10.0
        return -port_ret / port_vol

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Per-symbol cap
    symbols = {}
    for i, s in enumerate(survivors):
        sym = s["symbol"]
        symbols.setdefault(sym, []).append(i)

    for sym, indices in symbols.items():
        cap = MAX_WEIGHT_PER_SYMBOL
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=indices, c=cap: c - sum(w[i] for i in idx)
        })

    # Bounds: [MIN_WEIGHT, MAX_WEIGHT_PER_COMBO]
    bounds = [(MIN_WEIGHT, MAX_WEIGHT_PER_COMBO)] * n

    # Initial: equal weight
    w0 = np.ones(n) / n

    result = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if result.success:
        weights = result.x
        weights = np.maximum(weights, 0)
        weights /= weights.sum()
        logger.info(f"Markowitz converged: Sharpe={-result.fun:.3f}")
    else:
        logger.warning(f"Markowitz failed: {result.message}, using equal weight")
        weights = np.ones(n) / n

    return weights


def equal_weight(survivors):
    n = len(survivors)
    return np.ones(n) / n


def sharpe_weight(survivors):
    sharpes = np.array([max(s["ho_sharpe"], 0.01) for s in survivors])
    return sharpes / sharpes.sum()


def risk_parity(survivors):
    inv_dd = np.array([1.0 / max(abs(s["ho_dd"]), 0.01) for s in survivors])
    return inv_dd / inv_dd.sum()


def apply_hard_constraints(weights, survivors):
    """Apply hard constraints post-optimization."""
    n = len(weights)

    # Per-combo cap
    weights = np.minimum(weights, MAX_WEIGHT_PER_COMBO)

    # Per-symbol cap
    symbols = {}
    for i, s in enumerate(survivors):
        symbols.setdefault(s["symbol"], []).append(i)

    for sym, indices in symbols.items():
        total = sum(weights[i] for i in indices)
        if total > MAX_WEIGHT_PER_SYMBOL:
            scale = MAX_WEIGHT_PER_SYMBOL / total
            for i in indices:
                weights[i] *= scale

    # Renormalize
    weights /= weights.sum()
    return weights


# ── Portfolio Equity Simulation ─────────────────────────

def simulate_portfolio_equity(survivors, weights, returns_dict):
    """Simulate combined portfolio equity curve."""
    keys = [s["_key"] for s in survivors]
    min_len = min(len(returns_dict[k]) for k in keys)

    # Weighted returns
    port_returns = np.zeros(min_len)
    for i, k in enumerate(keys):
        port_returns += weights[i] * returns_dict[k][:min_len]

    # Build equity
    equity = np.zeros(min_len + 1)
    equity[0] = INITIAL_CAPITAL
    for t in range(min_len):
        equity[t + 1] = equity[t] * (1 + port_returns[t])

    return equity, port_returns


# ── Monte Carlo ─────────────────────────────────────────

def monte_carlo_simulation(returns, n_sims=N_MONTE_CARLO, horizons_months=[3, 6, 12, 24]):
    """Block bootstrap Monte Carlo on holdout returns."""
    n = len(returns)
    block_size = min(20, n // 5)
    results = {}

    for months in horizons_months:
        # Approximate bars per month (daily ~30, 4h ~180)
        bars = months * 30  # Rough daily equivalent
        if bars > n:
            bars = n

        sims = np.zeros(n_sims)
        for s in range(n_sims):
            sim_returns = []
            while len(sim_returns) < bars:
                start = np.random.randint(0, max(1, n - block_size))
                block = returns[start:start + block_size]
                sim_returns.extend(block.tolist())
            sim_returns = np.array(sim_returns[:bars])
            equity = INITIAL_CAPITAL * np.cumprod(1 + sim_returns)
            sims[s] = equity[-1]

        results[months] = {
            "p5": float(np.percentile(sims, 5)),
            "p25": float(np.percentile(sims, 25)),
            "median": float(np.percentile(sims, 50)),
            "p75": float(np.percentile(sims, 75)),
            "p95": float(np.percentile(sims, 95)),
            "prob_positive": float(np.mean(sims > INITIAL_CAPITAL)),
            "prob_ruin": float(np.mean(sims < INITIAL_CAPITAL * 0.5)),
        }

    return results


# ── Report Generation ───────────────────────────────────

def generate_report(portfolios, survivors, mc_results, elapsed):
    """Generate markdown report."""
    lines = []
    lines.append("# Portfolio V4 — Edge-Enhanced")
    lines.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    lines.append(f"**Durée** : {elapsed:.1f} min")
    lines.append(f"**Source** : {DIAG_PATH.name}")
    lines.append(f"**Cutoff holdout** : {CUTOFF_DATE}")
    lines.append(f"**Contraintes** : ETH ≤ {MAX_WEIGHT_PER_SYMBOL:.0%}, combo ≤ {MAX_WEIGHT_PER_COMBO:.0%}, corr < {MAX_CORRELATION}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Portfolio comparison
    lines.append("## Comparaison des méthodes")
    lines.append("")
    lines.append("| Portfolio | HO Sharpe | HO Sortino | HO Return | HO DD | HO Calmar | N combos |")
    lines.append("|-----------|-----------|------------|-----------|-------|-----------|----------|")

    best_name = None
    best_sharpe = -999

    for name, pdata in portfolios.items():
        m = pdata["metrics"]
        lines.append(
            f"| **{name}** | {m['sharpe']:.2f} | {m['sortino']:.2f} "
            f"| {m['total_return']*100:.1f}% | {m['max_drawdown']*100:.1f}% "
            f"| {m['calmar']:.2f} | {pdata['n_combos']} |"
        )
        if m["sharpe"] > best_sharpe:
            best_sharpe = m["sharpe"]
            best_name = name

    lines.append("")

    # Best portfolio details
    best = portfolios[best_name]
    lines.append(f"## 🏆 Meilleur : {best_name}")
    lines.append("")

    # Allocations
    lines.append("### Allocations")
    lines.append("")
    lines.append("| # | Poids | Symbol | Stratégie | TF | Overlay | HO Sharpe | HO DD |")
    lines.append("|---|-------|--------|-----------|-----|---------|-----------|-------|")
    for i, (s, w) in enumerate(zip(best["survivors"], best["weights"])):
        if w >= 0.005:
            ov = "✓" if s["overlay"] else "—"
            lines.append(
                f"| {i+1} | {w*100:.1f}% | {s['symbol']} | {s['strategy']} "
                f"| {s['timeframe']} | {ov} | {s['ho_sharpe']:.3f} | {s['ho_dd']*100:.1f}% |"
            )
    lines.append("")

    # Symbol allocation
    sym_alloc = {}
    for s, w in zip(best["survivors"], best["weights"]):
        sym_alloc[s["symbol"]] = sym_alloc.get(s["symbol"], 0) + w
    lines.append("### Allocation par symbol")
    lines.append("")
    lines.append("| Symbol | Allocation |")
    lines.append("|--------|-----------|")
    for sym, alloc in sorted(sym_alloc.items(), key=lambda x: -x[1]):
        lines.append(f"| {sym} | {alloc*100:.1f}% |")
    lines.append("")

    # Performance metrics
    m = best["metrics"]
    lines.append("### Performance holdout")
    lines.append("")
    lines.append("| Métrique | Valeur |")
    lines.append("|----------|--------|")
    lines.append(f"| **Return annuel** | **{m['total_return']*100:.1f}%** |")
    lines.append(f"| **Sharpe** | **{m['sharpe']:.2f}** |")
    lines.append(f"| **Sortino** | **{m['sortino']:.2f}** |")
    lines.append(f"| **Max DD** | **{m['max_drawdown']*100:.1f}%** |")
    lines.append(f"| **Calmar** | **{m['calmar']:.2f}** |")
    lines.append("")

    # Monte Carlo
    if best_name in mc_results:
        mc = mc_results[best_name]
        lines.append("### Projections Monte Carlo ($10,000)")
        lines.append("")
        lines.append("| Horizon | P5 (pessimiste) | Médian | P95 (optimiste) | P(>0) | P(ruin) |")
        lines.append("|---------|----------------|--------|-----------------|-------|---------|")
        for months, r in mc.items():
            lines.append(
                f"| {months} mois | ${r['p5']:,.0f} | ${r['median']:,.0f} "
                f"| ${r['p95']:,.0f} | {r['prob_positive']*100:.0f}% | {r['prob_ruin']*100:.1f}% |"
            )
        lines.append("")

    # Comparison with V3b
    lines.append("### Comparaison V3b vs V4")
    lines.append("")
    lines.append("| Métrique | V3b (markowitz) | V4 (best) | Δ |")
    lines.append("|----------|----------------|-----------|---|")
    v3b = {"sharpe": 1.19, "return": 9.8, "dd": -4.9, "calmar": 1.91}
    v4 = {"sharpe": m["sharpe"], "return": m["total_return"]*100,
           "dd": m["max_drawdown"]*100, "calmar": m["calmar"]}
    for metric, v3v, v4v in [
        ("Sharpe", v3b["sharpe"], v4["sharpe"]),
        ("Return", v3b["return"], v4["return"]),
        ("Max DD", v3b["dd"], v4["dd"]),
        ("Calmar", v3b["calmar"], v4["calmar"]),
    ]:
        delta = v4v - v3v
        sign = "+" if delta > 0 else ""
        unit = "%" if metric in ("Return", "Max DD") else ""
        lines.append(f"| {metric} | {v3v:.2f}{unit} | {v4v:.2f}{unit} | {sign}{delta:.2f} |")
    lines.append("")

    lines.append("---")
    lines.append(f"*Généré le {datetime.now().strftime('%d %B %Y')}*")

    Path("docs/results/16_portfolio_v4.md").write_text("\n".join(lines))
    logger.info("Report: docs/results/16_portfolio_v4.md")


# ── Main ────────────────────────────────────────────────

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PORTFOLIO V4 — Edge-Enhanced Construction")
    logger.info("=" * 60)

    # 1. Load survivors
    survivors = load_survivors()

    # 2. Rebuild holdout equity curves
    survivors, equities, returns_dict = rebuild_holdout_equities(survivors)

    if len(survivors) < 3:
        logger.error("Not enough survivors to build portfolio")
        return

    # 3. Correlation deduplication
    survivors = deduplicate_by_correlation(survivors, returns_dict)

    # 4. Build portfolios with different methods
    logger.info("=" * 60)
    logger.info("Building portfolios...")
    logger.info("=" * 60)

    methods = {
        "markowitz_constrained": markowitz_constrained(survivors, returns_dict),
        "equal_weight": apply_hard_constraints(equal_weight(survivors), survivors),
        "sharpe_weighted": apply_hard_constraints(sharpe_weight(survivors), survivors),
        "risk_parity": apply_hard_constraints(risk_parity(survivors), survivors),
    }

    portfolios = {}
    for name, weights in methods.items():
        equity, port_returns = simulate_portfolio_equity(survivors, weights, returns_dict)
        metrics = compute_all_metrics(equity, "1d")  # Approximate as daily
        portfolios[name] = {
            "weights": weights.tolist(),
            "survivors": survivors,
            "metrics": metrics,
            "equity": equity.tolist(),
            "returns": port_returns.tolist(),
            "n_combos": int(np.sum(weights > 0.005)),
        }

        # Symbol allocation
        sym_alloc = {}
        for s, w in zip(survivors, weights):
            sym_alloc[s["symbol"]] = sym_alloc.get(s["symbol"], 0) + w

        logger.info(
            f"  {name}: Sharpe={metrics['sharpe']:.2f}, "
            f"Return={metrics['total_return']*100:.1f}%, "
            f"DD={metrics['max_drawdown']*100:.1f}%, "
            f"Calmar={metrics['calmar']:.2f}, "
            f"Alloc={', '.join(f'{k}:{v:.0%}' for k,v in sorted(sym_alloc.items(), key=lambda x:-x[1]))}"
        )

    # 5. Monte Carlo on best portfolios
    logger.info("Running Monte Carlo simulations...")
    mc_results = {}
    for name, pdata in portfolios.items():
        mc = monte_carlo_simulation(np.array(pdata["returns"]))
        mc_results[name] = mc
        logger.info(f"  {name}: 12M median=${mc[12]['median']:,.0f}, "
                   f"P(>0)={mc[12]['prob_positive']*100:.0f}%")

    # 6. Save results
    elapsed = (time.time() - t0) / 60
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_data = {}
    for name, pdata in portfolios.items():
        save_data[name] = {
            "weights": pdata["weights"],
            "metrics": pdata["metrics"],
            "n_combos": pdata["n_combos"],
            "allocations": [
                {"weight": w, "symbol": s["symbol"], "strategy": s["strategy"],
                 "timeframe": s["timeframe"], "overlay": s["overlay"],
                 "ho_sharpe": s["ho_sharpe"], "last_params": s["last_params"]}
                for s, w in zip(pdata["survivors"], pdata["weights"])
                if w >= 0.005
            ],
            "monte_carlo": mc_results.get(name, {}),
        }

    results_path = f"results/portfolio_v4_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    logger.info(f"Saved: {results_path}")

    # 7. Generate report
    generate_report(portfolios, survivors, mc_results, elapsed)

    # 8. Summary
    best_name = max(portfolios, key=lambda k: portfolios[k]["metrics"]["sharpe"])
    best = portfolios[best_name]
    m = best["metrics"]

    logger.info(f"\n{'='*60}")
    logger.info(f"🏆 BEST PORTFOLIO: {best_name}")
    logger.info(f"  Sharpe: {m['sharpe']:.2f}")
    logger.info(f"  Return: {m['total_return']*100:.1f}%")
    logger.info(f"  Max DD: {m['max_drawdown']*100:.1f}%")
    logger.info(f"  Calmar: {m['calmar']:.2f}")
    logger.info(f"  Combos: {best['n_combos']}")
    logger.info(f"  Time: {elapsed:.1f} min")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
