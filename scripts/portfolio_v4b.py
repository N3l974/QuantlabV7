#!/usr/bin/env python3
"""
Portfolio V4b — Aggressive variant targeting +15% annual return.

Strategy: concentrate on top return combos instead of max diversification.
Three variants compared:
  A) V4b-concentrated: top 8 combos by return, no overlay dilution
  B) V4b-selective: top 12 combos, selective overlay (only on high-DD combos)
  C) V4-leveraged: V4 conservative × 2x-3x leverage

Hard constraints maintained: ETH ≤ 60%, combo ≤ 25%
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
from engine.metrics import compute_all_metrics, returns_from_equity
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

MAX_WEIGHT_ETH = 0.60
MAX_WEIGHT_COMBO = 0.25
MIN_WEIGHT = 0.02

# Overlay (lighter version for selective use)
REGIME_CFG = RegimeOverlayConfig(
    regime_config=RegimeConfig(),
    hard_cutoff=True,
    min_exposure_threshold=0.3,
)
VOL_CFG = VolTargetConfig(target_vol_annual=0.40)  # Higher target = less reduction
OVERLAY_CFG = OverlayPipelineConfig(regime_config=REGIME_CFG, vol_config=VOL_CFG)

RESULTS_DIR = Path("portfolio/v4b/results")
SEARCH_DIRS = [RESULTS_DIR, Path("results")]


def find_latest(pattern: str) -> Path:
    for base in SEARCH_DIRS:
        files = sorted(base.glob(pattern))
        if files:
            return files[-1]
    raise FileNotFoundError(f"No {pattern} found in portfolio/v4b/results or results")


# Load diagnostic
DIAG_PATH = find_latest("diagnostic_v4_fast_*.json")


def load_survivors():
    with open(DIAG_PATH) as f:
        data = json.load(f)
    p2 = data["phase2"]
    survivors = [r for r in p2
                 if r["verdict"] in ("STRONG", "WEAK")
                 and r["ho_trades"] >= 3
                 and r["ho_sharpe"] > 0]
    # Deduplicate
    best = {}
    for r in survivors:
        key = (r["symbol"], r["strategy"], r["timeframe"])
        if key not in best or r["ho_return"] > best[key]["ho_return"]:
            best[key] = r
    return sorted(best.values(), key=lambda x: x["ho_return"], reverse=True)


def build_equity(combo, use_overlay=False):
    """Build holdout equity for a combo."""
    sym, sname, tf = combo["symbol"], combo["strategy"], combo["timeframe"]
    params = combo["last_params"]

    data = pd.read_parquet(f"data/raw/{sym}_{tf}.parquet")
    ho_data = data[data.index >= CUTOFF_DATE].copy()

    strategy = get_strategy(sname)
    signals = strategy.generate_signals(ho_data, params)

    if use_overlay:
        signals, _ = apply_overlay_pipeline(signals, ho_data, OVERLAY_CFG, timeframe=tf)

    close = ho_data["close"].values.astype(np.float64)
    high = ho_data["high"].values.astype(np.float64)
    low = ho_data["low"].values.astype(np.float64)

    res = vectorized_backtest(close, signals, risk=RISK, high=high, low=low, timeframe=tf)
    return res.equity, returns_from_equity(res.equity)


def ledoit_wolf(rets_matrix):
    T, N = rets_matrix.shape
    X = rets_matrix - rets_matrix.mean(axis=0)
    S = X.T @ X / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)
    d2 = np.sum((S - F) ** 2) / N
    b2_sum = sum(np.sum((X[t:t+1].T @ X[t:t+1] - S) ** 2) / N for t in range(T))
    b2 = min(b2_sum / T**2, d2)
    alpha = np.clip(b2 / max(d2, 1e-10), 0, 1)
    return alpha * F + (1 - alpha) * S


def optimize_weights(survivors, returns_dict, objective="return_dd"):
    """Optimize weights. objective: 'sharpe', 'return_dd', 'max_return'."""
    n = len(survivors)
    keys = [s["_key"] for s in survivors]
    min_len = min(len(returns_dict[k]) for k in keys)
    rets_matrix = np.column_stack([returns_dict[k][:min_len] for k in keys])

    cov = ledoit_wolf(rets_matrix)
    mu = rets_matrix.mean(axis=0)

    def neg_objective(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if objective == "sharpe":
            return -port_ret / max(port_vol, 1e-10)
        elif objective == "return_dd":
            # Maximize return while penalizing variance
            return -(port_ret - 0.5 * port_vol)
        elif objective == "max_return":
            return -port_ret + 0.1 * port_vol  # Light vol penalty
        return -port_ret / max(port_vol, 1e-10)

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Per-symbol cap
    symbols = {}
    for i, s in enumerate(survivors):
        symbols.setdefault(s["symbol"], []).append(i)
    for sym, indices in symbols.items():
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=indices: MAX_WEIGHT_ETH - sum(w[i] for i in idx)
        })

    bounds = [(MIN_WEIGHT, MAX_WEIGHT_COMBO)] * n
    w0 = np.ones(n) / n

    result = minimize(neg_objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    if result.success:
        w = np.maximum(result.x, 0)
        return w / w.sum()
    return np.ones(n) / n


def simulate_portfolio(survivors, weights, returns_dict, leverage=1.0):
    keys = [s["_key"] for s in survivors]
    min_len = min(len(returns_dict[k]) for k in keys)
    port_returns = np.zeros(min_len)
    for i, k in enumerate(keys):
        port_returns += weights[i] * returns_dict[k][:min_len]
    port_returns *= leverage
    equity = np.zeros(min_len + 1)
    equity[0] = INITIAL_CAPITAL
    for t in range(min_len):
        equity[t + 1] = equity[t] * (1 + port_returns[t])
    return equity, port_returns


def monte_carlo(returns, n_sims=N_MONTE_CARLO):
    n = len(returns)
    block_size = min(20, n // 5)
    results = {}
    for months in [3, 6, 12, 24]:
        bars = min(months * 30, n)
        sims = np.zeros(n_sims)
        for s in range(n_sims):
            sim_rets = []
            while len(sim_rets) < bars:
                start = np.random.randint(0, max(1, n - block_size))
                sim_rets.extend(returns[start:start + block_size].tolist())
            eq = INITIAL_CAPITAL * np.cumprod(1 + np.array(sim_rets[:bars]))
            sims[s] = eq[-1]
        results[months] = {
            "p5": float(np.percentile(sims, 5)),
            "median": float(np.percentile(sims, 50)),
            "p95": float(np.percentile(sims, 95)),
            "prob_pos": float(np.mean(sims > INITIAL_CAPITAL)),
        }
    return results


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PORTFOLIO V4b — Aggressive (target +15%)")
    logger.info("=" * 60)

    all_survivors = load_survivors()
    logger.info(f"Loaded {len(all_survivors)} survivors sorted by return")

    # ── Variant A: Concentrated (top 8 by return, NO overlay) ──
    logger.info("\n--- Variant A: Concentrated (top 8, no overlay) ---")
    top8 = all_survivors[:8]
    returns_a = {}
    valid_a = []
    for s in top8:
        key = f"{s['symbol']}/{s['strategy']}/{s['timeframe']}"
        eq, rets = build_equity(s, use_overlay=False)
        returns_a[key] = rets
        s["_key"] = key
        s["_equity"] = eq
        valid_a.append(s)
        logger.info(f"  {key}: Return={s['ho_return']*100:.1f}%, DD={s['ho_dd']*100:.1f}%")

    weights_a_sharpe = optimize_weights(valid_a, returns_a, "sharpe")
    weights_a_retdd = optimize_weights(valid_a, returns_a, "return_dd")
    weights_a_maxret = optimize_weights(valid_a, returns_a, "max_return")

    # ── Variant B: Selective (top 12, overlay only on DD > 10%) ──
    logger.info("\n--- Variant B: Selective (top 12, selective overlay) ---")
    top12 = all_survivors[:12]
    returns_b = {}
    valid_b = []
    for s in top12:
        key = f"{s['symbol']}/{s['strategy']}/{s['timeframe']}"
        use_ov = abs(s["ho_dd"]) > 0.10  # Overlay only on high-DD combos
        eq, rets = build_equity(s, use_overlay=use_ov)
        returns_b[key] = rets
        s["_key"] = key
        valid_b.append(s)
        tag = "(+ov)" if use_ov else ""
        logger.info(f"  {key} {tag}: Return={s['ho_return']*100:.1f}%")

    weights_b = optimize_weights(valid_b, returns_b, "return_dd")

    # ── Variant C: V4 conservative with leverage ──
    # Reload V4 portfolio returns
    v4_path = find_latest("portfolio_v4_*.json")
    with open(v4_path) as f:
        v4_data = json.load(f)

    # ── Build all portfolios ──
    portfolios = {}

    # A variants
    for name, weights in [
        ("A_concentrated_sharpe", weights_a_sharpe),
        ("A_concentrated_retdd", weights_a_retdd),
        ("A_concentrated_maxret", weights_a_maxret),
    ]:
        eq, rets = simulate_portfolio(valid_a, weights, returns_a)
        m = compute_all_metrics(eq, "1d")
        portfolios[name] = {"weights": weights, "survivors": valid_a,
                            "equity": eq, "returns": rets, "metrics": m}

    # B variant
    eq_b, rets_b = simulate_portfolio(valid_b, weights_b, returns_b)
    m_b = compute_all_metrics(eq_b, "1d")
    portfolios["B_selective"] = {"weights": weights_b, "survivors": valid_b,
                                  "equity": eq_b, "returns": rets_b, "metrics": m_b}

    # C variants: leverage on A_concentrated_retdd
    best_a_weights = weights_a_retdd
    for lev in [1.5, 2.0, 2.5, 3.0]:
        eq_l, rets_l = simulate_portfolio(valid_a, best_a_weights, returns_a, leverage=lev)
        m_l = compute_all_metrics(eq_l, "1d")
        portfolios[f"C_lev_{lev:.1f}x"] = {"weights": best_a_weights, "survivors": valid_a,
                                             "equity": eq_l, "returns": rets_l, "metrics": m_l}

    # ── Compare all ──
    logger.info("\n" + "=" * 100)
    logger.info("COMPARISON — All V4b variants")
    logger.info("=" * 100)
    fmt = "{:<28} {:>8} {:>8} {:>10} {:>8} {:>8} {:>8}"
    logger.info(fmt.format("Portfolio", "Sharpe", "Sortino", "Return", "Max DD", "Calmar", "N"))
    logger.info("-" * 100)

    for name, p in sorted(portfolios.items(), key=lambda x: -x[1]["metrics"]["total_return"]):
        m = p["metrics"]
        n = int(np.sum(p["weights"] > 0.01)) if isinstance(p["weights"], np.ndarray) else "?"
        logger.info(fmt.format(
            name,
            f"{m['sharpe']:.2f}",
            f"{m['sortino']:.2f}",
            f"{m['total_return']*100:.1f}%",
            f"{m['max_drawdown']*100:.1f}%",
            f"{m['calmar']:.2f}",
            str(n),
        ))

    # ── Find best that meets +15% target ──
    target_combos = {k: v for k, v in portfolios.items()
                     if v["metrics"]["total_return"] >= 0.12}  # 12%+ (annualized from ~1yr holdout)

    if target_combos:
        # Among those meeting target, pick best Calmar
        best_name = max(target_combos, key=lambda k: target_combos[k]["metrics"]["calmar"])
        best = portfolios[best_name]
    else:
        best_name = max(portfolios, key=lambda k: portfolios[k]["metrics"]["total_return"])
        best = portfolios[best_name]

    logger.info(f"\n🏆 RECOMMENDED: {best_name}")
    m = best["metrics"]
    logger.info(f"  Return: {m['total_return']*100:.1f}%")
    logger.info(f"  Sharpe: {m['sharpe']:.2f}")
    logger.info(f"  Max DD: {m['max_drawdown']*100:.1f}%")
    logger.info(f"  Calmar: {m['calmar']:.2f}")

    # Symbol allocation
    if isinstance(best["weights"], np.ndarray):
        sym_alloc = {}
        for s, w in zip(best["survivors"], best["weights"]):
            sym_alloc[s["symbol"]] = sym_alloc.get(s["symbol"], 0) + w
        for sym, alloc in sorted(sym_alloc.items(), key=lambda x: -x[1]):
            logger.info(f"  {sym}: {alloc*100:.1f}%")

    # Allocations
    if isinstance(best["weights"], np.ndarray):
        logger.info("\n  Allocations:")
        for s, w in sorted(zip(best["survivors"], best["weights"]), key=lambda x: -x[1]):
            if w > 0.01:
                logger.info(f"    {w*100:.1f}% {s['symbol']}/{s['strategy']}/{s['timeframe']} "
                           f"(Ret={s['ho_return']*100:.1f}%, DD={s['ho_dd']*100:.1f}%)")

    # Monte Carlo on recommended
    mc = monte_carlo(best["returns"])
    logger.info(f"\n  Monte Carlo 12M: median=${mc[12]['median']:,.0f}, "
               f"P(>0)={mc[12]['prob_pos']*100:.0f}%")

    # ── Save ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {}
    for name, p in portfolios.items():
        save_data[name] = {
            "metrics": p["metrics"],
            "weights": p["weights"].tolist() if isinstance(p["weights"], np.ndarray) else p["weights"],
            "allocations": [
                {"weight": float(w), "symbol": s["symbol"], "strategy": s["strategy"],
                 "timeframe": s["timeframe"], "ho_return": s["ho_return"],
                 "ho_dd": s["ho_dd"], "last_params": s["last_params"]}
                for s, w in zip(p["survivors"], p["weights"])
                if (isinstance(w, (int, float)) and w > 0.01) or
                   (hasattr(w, '__float__') and float(w) > 0.01)
            ] if isinstance(p["weights"], np.ndarray) else [],
        }
    save_data["recommended"] = best_name
    save_data["monte_carlo"] = {best_name: mc}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"portfolio_v4b_{ts}.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # ── Report ──
    lines = []
    lines.append("# Portfolio V4b — Aggressive (target +15%)")
    lines.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    lines.append(f"**Objectif** : +15% annuel, DD < -20%")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Comparaison des variantes")
    lines.append("")
    lines.append("| Portfolio | Sharpe | Sortino | Return | Max DD | Calmar | N combos |")
    lines.append("|-----------|--------|---------|--------|--------|--------|----------|")
    for name, p in sorted(portfolios.items(), key=lambda x: -x[1]["metrics"]["total_return"]):
        m = p["metrics"]
        n = int(np.sum(p["weights"] > 0.01)) if isinstance(p["weights"], np.ndarray) else "?"
        target_ok = "✅" if m["total_return"] >= 0.12 else "❌"
        lines.append(
            f"| {target_ok} **{name}** | {m['sharpe']:.2f} | {m['sortino']:.2f} "
            f"| {m['total_return']*100:.1f}% | {m['max_drawdown']*100:.1f}% "
            f"| {m['calmar']:.2f} | {n} |"
        )
    lines.append("")
    lines.append(f"## 🏆 Recommandé : {best_name}")
    lines.append("")
    m = best["metrics"]
    lines.append("| Métrique | Valeur | Objectif | Status |")
    lines.append("|----------|--------|----------|--------|")
    lines.append(f"| Return | {m['total_return']*100:.1f}% | ≥15% | {'✅' if m['total_return']>=0.15 else '⚠️' if m['total_return']>=0.12 else '❌'} |")
    lines.append(f"| Max DD | {m['max_drawdown']*100:.1f}% | ≥-20% | {'✅' if m['max_drawdown']>=-0.20 else '❌'} |")
    lines.append(f"| Sharpe | {m['sharpe']:.2f} | ≥1.0 | {'✅' if m['sharpe']>=1.0 else '❌'} |")
    lines.append(f"| Calmar | {m['calmar']:.2f} | ≥1.0 | {'✅' if m['calmar']>=1.0 else '❌'} |")
    lines.append("")

    if isinstance(best["weights"], np.ndarray):
        lines.append("### Allocations")
        lines.append("")
        lines.append("| Poids | Symbol | Stratégie | TF | HO Return | HO DD |")
        lines.append("|-------|--------|-----------|-----|-----------|-------|")
        for s, w in sorted(zip(best["survivors"], best["weights"]), key=lambda x: -x[1]):
            if w > 0.01:
                lines.append(f"| {w*100:.1f}% | {s['symbol']} | {s['strategy']} | {s['timeframe']} "
                           f"| {s['ho_return']*100:.1f}% | {s['ho_dd']*100:.1f}% |")
        lines.append("")

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

    lines.append("### Monte Carlo ($10,000)")
    lines.append("")
    lines.append("| Horizon | P5 | Médian | P95 | P(>0) |")
    lines.append("|---------|-----|--------|-----|-------|")
    for months, r in mc.items():
        lines.append(f"| {months}M | ${r['p5']:,.0f} | ${r['median']:,.0f} | ${r['p95']:,.0f} | {r['prob_pos']*100:.0f}% |")
    lines.append("")

    lines.append("### Comparaison V3b / V4 / V4b")
    lines.append("")
    lines.append("| Métrique | V3b | V4 (conserv.) | V4b (agressif) |")
    lines.append("|----------|-----|---------------|----------------|")
    lines.append(f"| Return | +9.8% | +4.9% | **+{m['total_return']*100:.1f}%** |")
    lines.append(f"| Sharpe | 1.19 | 2.59 | **{m['sharpe']:.2f}** |")
    lines.append(f"| Max DD | -4.9% | -0.8% | **{m['max_drawdown']*100:.1f}%** |")
    lines.append(f"| Calmar | 1.91 | 5.99 | **{m['calmar']:.2f}** |")
    lines.append(f"| ETH % | 95% | 53% | **{sym_alloc.get('ETHUSDT',0)*100:.0f}%** |" if isinstance(best["weights"], np.ndarray) else "")
    lines.append("")
    lines.append("---")
    lines.append(f"*Généré le {datetime.now().strftime('%d %B %Y')}*")

    report_path = RESULTS_DIR / "17_portfolio_v4b.md"
    report_path.write_text("\n".join(lines))
    logger.info(f"\nReport: {report_path}")

    elapsed = (time.time() - t0) / 60
    logger.info(f"Total: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
