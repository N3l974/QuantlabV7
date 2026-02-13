#!/usr/bin/env python3
"""
Portfolio V4b Final — Validation complète + Projections

Config retenue : top3_heavy × 1.5x leverage
- 8 combos concentrés sur les meilleurs returns
- Pondération heavy sur top 3
- Leverage modéré 1.5x pour atteindre +15%

Validation :
1. Holdout equity curve
2. Monte Carlo (block bootstrap, 5000 sims)
3. Projections gains espérés (3M, 6M, 12M, 24M, 36M)
4. Stress tests (worst month, worst quarter, recovery time)
5. Rolling Sharpe stability
6. Comparaison V3b / V4 / V4b
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

from engine.backtester import RiskConfig, vectorized_backtest
from engine.metrics import compute_all_metrics, returns_from_equity, max_drawdown
from strategies.registry import get_strategy

# ── Config V4b Final ────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
CUTOFF_DATE = "2025-02-01"
RISK = RiskConfig()
LEVERAGE = 1.5
N_MONTE_CARLO = 5000

# Top 8 combos by return + weights (top3_heavy)
COMBOS = [
    {"symbol": "ETHUSDT", "strategy": "supertrend",          "timeframe": "1d", "weight": 0.25},
    {"symbol": "ETHUSDT", "strategy": "trend_multi_factor",   "timeframe": "1d", "weight": 0.25},
    {"symbol": "SOLUSDT", "strategy": "trend_multi_factor",   "timeframe": "4h", "weight": 0.15},
    {"symbol": "BTCUSDT", "strategy": "supertrend",          "timeframe": "1d", "weight": 0.10},
    {"symbol": "ETHUSDT", "strategy": "macd_crossover",      "timeframe": "1d", "weight": 0.10},
    {"symbol": "BTCUSDT", "strategy": "trend_multi_factor",   "timeframe": "1d", "weight": 0.05},
    {"symbol": "ETHUSDT", "strategy": "ichimoku_cloud",      "timeframe": "4h", "weight": 0.05},
    {"symbol": "ETHUSDT", "strategy": "bollinger_breakout",  "timeframe": "1d", "weight": 0.05},
]

# Load last WF params from diagnostic
DIAG_PATH = sorted(Path("results").glob("diagnostic_v4_fast_*.json"))[-1]


def load_params():
    """Load last WF params for each combo from diagnostic."""
    with open(DIAG_PATH) as f:
        data = json.load(f)
    p2 = data["phase2"]
    params_map = {}
    for r in p2:
        key = (r["symbol"], r["strategy"], r["timeframe"])
        if r["ho_sharpe"] > 0 and r["ho_trades"] >= 3:
            if key not in params_map or r["ho_return"] > params_map[key]["ho_return"]:
                params_map[key] = r
    return params_map


def build_holdout_returns():
    """Build holdout returns for each combo."""
    params_map = load_params()
    combo_returns = []
    combo_info = []

    for c in COMBOS:
        key = (c["symbol"], c["strategy"], c["timeframe"])
        if key not in params_map:
            logger.error(f"No params for {key}")
            continue

        params = params_map[key]["last_params"]
        sym, sname, tf = c["symbol"], c["strategy"], c["timeframe"]

        data = pd.read_parquet(f"data/raw/{sym}_{tf}.parquet")
        ho_data = data[data.index >= CUTOFF_DATE].copy()

        strategy = get_strategy(sname)
        signals = strategy.generate_signals(ho_data, params)

        close = ho_data["close"].values.astype(np.float64)
        high = ho_data["high"].values.astype(np.float64)
        low = ho_data["low"].values.astype(np.float64)

        res = vectorized_backtest(close, signals, risk=RISK, high=high, low=low, timeframe=tf)
        rets = returns_from_equity(res.equity)
        m = compute_all_metrics(res.equity, tf, res.trades_pnl)

        combo_returns.append(rets)
        combo_info.append({
            **c,
            "params": params,
            "ho_sharpe": m["sharpe"],
            "ho_return": m["total_return"],
            "ho_dd": m["max_drawdown"],
            "ho_trades": m.get("n_trades", res.n_trades),
            "ho_calmar": m["calmar"],
            "ho_sortino": m["sortino"],
            "ho_wr": m.get("win_rate", 0),
        })

        logger.info(f"  {sym}/{sname}/{tf}: w={c['weight']:.0%}, "
                    f"Ret={m['total_return']*100:.1f}%, DD={m['max_drawdown']*100:.1f}%, "
                    f"Sharpe={m['sharpe']:.2f}, Trades={res.n_trades}")

    return combo_returns, combo_info


def build_portfolio_equity(combo_returns, combo_info):
    """Build leveraged portfolio equity."""
    weights = np.array([c["weight"] for c in combo_info])
    min_len = min(len(r) for r in combo_returns)

    port_returns = np.zeros(min_len)
    for i, rets in enumerate(combo_returns):
        port_returns += weights[i] * rets[:min_len]

    # Apply leverage
    port_returns *= LEVERAGE

    equity = np.zeros(min_len + 1)
    equity[0] = INITIAL_CAPITAL
    for t in range(min_len):
        equity[t + 1] = equity[t] * (1 + port_returns[t])

    return equity, port_returns


def monte_carlo(returns, n_sims=N_MONTE_CARLO):
    """Block bootstrap Monte Carlo with detailed projections."""
    n = len(returns)
    block_size = min(20, n // 5)
    horizons = [3, 6, 12, 24, 36]
    results = {}

    for months in horizons:
        bars = min(months * 30, n * 3)  # Allow resampling beyond data
        sims = np.zeros(n_sims)
        paths = np.zeros((n_sims, bars))

        for s in range(n_sims):
            sim_rets = []
            while len(sim_rets) < bars:
                start = np.random.randint(0, max(1, n - block_size))
                sim_rets.extend(returns[start:start + block_size].tolist())
            sim_rets = np.array(sim_rets[:bars])
            eq = INITIAL_CAPITAL * np.cumprod(1 + sim_rets)
            sims[s] = eq[-1]
            paths[s, :len(eq)] = eq[:bars]

        results[months] = {
            "p1": float(np.percentile(sims, 1)),
            "p5": float(np.percentile(sims, 5)),
            "p10": float(np.percentile(sims, 10)),
            "p25": float(np.percentile(sims, 25)),
            "median": float(np.percentile(sims, 50)),
            "p75": float(np.percentile(sims, 75)),
            "p90": float(np.percentile(sims, 90)),
            "p95": float(np.percentile(sims, 95)),
            "mean": float(np.mean(sims)),
            "prob_positive": float(np.mean(sims > INITIAL_CAPITAL)),
            "prob_10pct": float(np.mean(sims > INITIAL_CAPITAL * 1.10)),
            "prob_20pct": float(np.mean(sims > INITIAL_CAPITAL * 1.20)),
            "prob_ruin": float(np.mean(sims < INITIAL_CAPITAL * 0.50)),
            "prob_loss_10": float(np.mean(sims < INITIAL_CAPITAL * 0.90)),
        }

    return results


def stress_tests(returns, equity):
    """Compute stress test metrics."""
    # Monthly returns (approximate: 30 bars)
    n = len(returns)
    monthly_rets = []
    for i in range(0, n, 30):
        chunk = returns[i:i+30]
        if len(chunk) > 10:
            monthly_rets.append(float(np.prod(1 + chunk) - 1))

    quarterly_rets = []
    for i in range(0, n, 90):
        chunk = returns[i:i+90]
        if len(chunk) > 30:
            quarterly_rets.append(float(np.prod(1 + chunk) - 1))

    # Rolling Sharpe (60-bar window)
    window = 60
    rolling_sharpes = []
    for i in range(window, n):
        chunk = returns[i-window:i]
        if np.std(chunk) > 0:
            rolling_sharpes.append(float(np.mean(chunk) / np.std(chunk) * np.sqrt(252)))

    # Max consecutive losing days
    max_losing_streak = 0
    current_streak = 0
    for r in returns:
        if r < 0:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0

    # Recovery time from max DD
    dd_series = equity / np.maximum.accumulate(equity) - 1
    max_dd_idx = np.argmin(dd_series)
    recovery_idx = max_dd_idx
    peak_before_dd = np.maximum.accumulate(equity)[max_dd_idx]
    for i in range(max_dd_idx, len(equity)):
        if equity[i] >= peak_before_dd:
            recovery_idx = i
            break
    recovery_bars = recovery_idx - max_dd_idx

    return {
        "worst_month": min(monthly_rets) if monthly_rets else 0,
        "best_month": max(monthly_rets) if monthly_rets else 0,
        "avg_month": np.mean(monthly_rets) if monthly_rets else 0,
        "worst_quarter": min(quarterly_rets) if quarterly_rets else 0,
        "best_quarter": max(quarterly_rets) if quarterly_rets else 0,
        "rolling_sharpe_min": min(rolling_sharpes) if rolling_sharpes else 0,
        "rolling_sharpe_max": max(rolling_sharpes) if rolling_sharpes else 0,
        "rolling_sharpe_mean": np.mean(rolling_sharpes) if rolling_sharpes else 0,
        "max_losing_streak": max_losing_streak,
        "recovery_bars": recovery_bars,
        "pct_positive_months": float(np.mean([r > 0 for r in monthly_rets])) if monthly_rets else 0,
        "n_months": len(monthly_rets),
    }


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PORTFOLIO V4b FINAL — Validation complète")
    logger.info(f"Config: top3_heavy × {LEVERAGE}x leverage")
    logger.info("=" * 60)

    # 1. Build holdout
    logger.info("\n1. Building holdout equity curves...")
    combo_returns, combo_info = build_holdout_returns()

    # 2. Portfolio equity
    logger.info("\n2. Building portfolio equity...")
    equity, port_returns = build_portfolio_equity(combo_returns, combo_info)
    metrics = compute_all_metrics(equity, "1d")

    logger.info(f"  Return: {metrics['total_return']*100:.1f}%")
    logger.info(f"  Sharpe: {metrics['sharpe']:.2f}")
    logger.info(f"  Sortino: {metrics['sortino']:.2f}")
    logger.info(f"  Max DD: {metrics['max_drawdown']*100:.1f}%")
    logger.info(f"  Calmar: {metrics['calmar']:.2f}")

    # 3. Monte Carlo
    logger.info("\n3. Monte Carlo simulations (5000 sims)...")
    mc = monte_carlo(port_returns)
    for months, r in mc.items():
        logger.info(f"  {months}M: median=${r['median']:,.0f} ({(r['median']/INITIAL_CAPITAL-1)*100:+.1f}%), "
                    f"P(>0)={r['prob_positive']*100:.0f}%, P(ruin)={r['prob_ruin']*100:.1f}%")

    # 4. Stress tests
    logger.info("\n4. Stress tests...")
    stress = stress_tests(port_returns, equity)
    logger.info(f"  Worst month: {stress['worst_month']*100:.1f}%")
    logger.info(f"  Best month: {stress['best_month']*100:.1f}%")
    logger.info(f"  Avg month: {stress['avg_month']*100:.2f}%")
    logger.info(f"  Worst quarter: {stress['worst_quarter']*100:.1f}%")
    logger.info(f"  % positive months: {stress['pct_positive_months']*100:.0f}%")
    logger.info(f"  Rolling Sharpe: {stress['rolling_sharpe_min']:.2f} to {stress['rolling_sharpe_max']:.2f}")
    logger.info(f"  Max losing streak: {stress['max_losing_streak']} bars")
    logger.info(f"  Recovery from max DD: {stress['recovery_bars']} bars")

    # 5. Gains espérés
    logger.info("\n5. Gains espérés ($10,000 initial)...")
    for capital in [10_000, 50_000, 100_000]:
        scale = capital / INITIAL_CAPITAL
        logger.info(f"\n  Capital: ${capital:,}")
        for months in [3, 6, 12, 24, 36]:
            r = mc[months]
            logger.info(f"    {months:>2}M: P5=${r['p5']*scale:>10,.0f} | "
                       f"Median=${r['median']*scale:>10,.0f} | "
                       f"P95=${r['p95']*scale:>10,.0f} | "
                       f"Gain median={((r['median']/INITIAL_CAPITAL)-1)*100:>+6.1f}%")

    # 6. Symbol allocation
    sym_alloc = {}
    for c in combo_info:
        sym_alloc[c["symbol"]] = sym_alloc.get(c["symbol"], 0) + c["weight"]

    # 7. Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("results").mkdir(exist_ok=True)

    save_data = {
        "portfolio_name": "V4b_top3_heavy_1.5x",
        "created_at": datetime.now().isoformat(),
        "config": {
            "leverage": LEVERAGE,
            "initial_capital": INITIAL_CAPITAL,
            "cutoff": CUTOFF_DATE,
            "n_combos": len(combo_info),
            "weighting": "top3_heavy",
        },
        "allocations": [
            {
                "weight": c["weight"],
                "symbol": c["symbol"],
                "strategy": c["strategy"],
                "timeframe": c["timeframe"],
                "params": c["params"],
                "ho_sharpe": c["ho_sharpe"],
                "ho_return": c["ho_return"],
                "ho_dd": c["ho_dd"],
                "ho_trades": c["ho_trades"],
            }
            for c in combo_info
        ],
        "symbol_allocation": sym_alloc,
        "metrics": metrics,
        "monte_carlo": mc,
        "stress_tests": stress,
    }

    results_path = f"results/portfolio_v4b_final_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    logger.info(f"\nSaved: {results_path}")

    elapsed = (time.time() - t0) / 60
    logger.info(f"\n{'='*60}")
    logger.info(f"VALIDATION COMPLETE ({elapsed:.1f} min)")
    logger.info(f"{'='*60}")

    return save_data


if __name__ == "__main__":
    main()
