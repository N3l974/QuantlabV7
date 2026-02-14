#!/usr/bin/env python3
"""
Portfolio V5b — Construction from V5b diagnostic survivors.

Lessons learned from previous portfolios:
  V3:  Concentration ETH 95% → cap symbol strict
  V3:  MC on full data → MC holdout-only
  V3b: Return +9.8% insuffisant → concentrate on top combos
  V4:  Sur-diversification (37 combos) dilue return → limit N combos
  V4:  Overlays trop agressifs coupent return → overlay sélectif
  V4:  Markowitz optimise Sharpe pas return → multi-objectif
  V4b: Pas de corrélation dedup → dedup corr > 0.85
  V4b: Pas d'overlay du tout → overlay sélectif sur high-DD

V5b innovations:
  - Uses generate_signals_v5() for risk-based sizing
  - Trailing stop, breakeven, max holding optimized per combo
  - Multi-risk-key: picks best risk_key per combo from diagnostic
  - Correlation deduplication on rebuilt holdout equity curves
  - Multiple portfolio variants compared (concentrated, diversified, leveraged)
  - 5000 MC sims (block bootstrap) on holdout returns
  - Overlay: selective (only on combos where overlay improved Sharpe)
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

from engine.backtester import RiskConfig, backtest_strategy, vectorized_backtest
from engine.metrics import compute_all_metrics, returns_from_equity
from engine.overlays import (
    apply_overlay_pipeline, OverlayPipelineConfig,
    VolTargetConfig, RegimeOverlayConfig,
)
from engine.regime import RegimeConfig
from strategies.registry import get_strategy

# ── Config ──────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
N_MONTE_CARLO = 5000
CUTOFF_DATE = "2025-02-01"

# Hard constraints (lessons from V3/V4)
MAX_WEIGHT_PER_SYMBOL = 0.55    # Avoid V3's 95% ETH concentration
MAX_WEIGHT_PER_COMBO = 0.25     # No single combo dominates
MIN_WEIGHT = 0.02               # Minimum allocation
MAX_CORRELATION = 0.85          # Dedup highly correlated combos
MIN_HO_SHARPE = 0.0             # Only positive holdout Sharpe
MIN_HO_TRADES = 3               # Minimum trades on holdout
MAX_SEED_STD = 1.0              # Seed robustness filter

# Overlay config (lighter than V4 — lesson: don't kill return)
REGIME_CFG = RegimeOverlayConfig(
    regime_config=RegimeConfig(),
    hard_cutoff=True,
    min_exposure_threshold=0.3,
)
VOL_CFG = VolTargetConfig(target_vol_annual=0.40)  # Higher = less aggressive
OVERLAY_CFG = OverlayPipelineConfig(regime_config=REGIME_CFG, vol_config=VOL_CFG)

RESULTS_DIR = Path("portfolio/v5b/results")
DIAG_SEARCH_DIRS = [RESULTS_DIR, Path("results")]


def find_latest_diagnostic() -> Path:
    for base in DIAG_SEARCH_DIRS:
        files = sorted(base.glob("diagnostic_v5b_*.json"))
        if files:
            return files[-1]
    raise FileNotFoundError(
        "No diagnostic_v5b_*.json found in portfolio/v5b/results or results"
    )


# Load diagnostic V5b
DIAG_PATH = find_latest_diagnostic()
logger.info(f"Using diagnostic: {DIAG_PATH}")


# ── Data Loading ────────────────────────────────────────

def load_survivors():
    """
    Load STRONG survivors from V5b diagnostic.
    For each (symbol, strategy, timeframe), pick the best risk_key variant.
    Also track if overlay version was better (for selective overlay).
    """
    with open(DIAG_PATH) as f:
        data = json.load(f)

    p2 = data["phase2"]

    # Filter: STRONG only, min trades, positive Sharpe, seed robust
    candidates = [r for r in p2
                  if r["verdict"] == "STRONG"
                  and r["ho_trades"] >= MIN_HO_TRADES
                  and r["ho_sharpe"] >= MIN_HO_SHARPE
                  and r.get("seed_sharpe_std", 0) <= MAX_SEED_STD]

    logger.info(f"Candidates after filtering: {len(candidates)}")

    # For each (symbol, strategy, timeframe), find best risk_key × overlay combo
    best = {}
    for r in candidates:
        key = (r["symbol"], r["strategy"], r["timeframe"])
        if key not in best or r["ho_sharpe"] > best[key]["ho_sharpe"]:
            best[key] = r

    survivors = sorted(best.values(), key=lambda x: x["ho_sharpe"], reverse=True)
    logger.info(f"Unique survivors (best risk_key per combo): {len(survivors)}")

    for i, s in enumerate(survivors[:15]):
        tag = f"r={s['risk_key']}" + (" +ov" if s["overlay"] else "")
        logger.info(f"  [{i+1}] {s['symbol']}/{s['strategy']}/{s['timeframe']} "
                    f"({tag}) Sharpe={s['ho_sharpe']:.3f} Ret={s['ho_return']*100:.1f}% "
                    f"DD={s['ho_dd']*100:.1f}% seed_std={s['seed_sharpe_std']:.3f}")

    return survivors


def build_equity(combo):
    """Build holdout equity for a combo using V5b features."""
    sym, sname, tf = combo["symbol"], combo["strategy"], combo["timeframe"]
    params = combo["last_params"]
    use_overlay = combo["overlay"]
    risk_key = combo["risk_key"]

    data = pd.read_parquet(f"data/raw/{sym}_{tf}.parquet")
    ho_data = data[data.index >= CUTOFF_DATE].copy()

    if len(ho_data) < 50:
        return None, None

    strategy = get_strategy(sname)

    # Determine risk config based on risk_key
    risk_pct = 0.0
    if risk_key.startswith("r"):
        try:
            risk_pct = float(risk_key[1:]) / 100.0  # "r1.0" -> 0.01
        except ValueError:
            risk_pct = 0.0

    risk = RiskConfig(risk_per_trade_pct=risk_pct)

    # Use V5 API for risk-based sizing
    sl_distances = None
    if hasattr(strategy, 'generate_signals_v5'):
        signals, sl_distances = strategy.generate_signals_v5(ho_data, params)
    else:
        signals = strategy.generate_signals(ho_data, params)

    if use_overlay:
        signals, _ = apply_overlay_pipeline(signals, ho_data, OVERLAY_CFG, timeframe=tf)

    close = ho_data["close"].values.astype(np.float64)
    high = ho_data["high"].values.astype(np.float64)
    low = ho_data["low"].values.astype(np.float64)

    min_len = min(len(close), len(signals))
    close = close[:min_len]
    signals = signals[:min_len]
    high = high[:min_len]
    low = low[:min_len]
    if sl_distances is not None:
        sl_distances = sl_distances[:min_len]

    res = vectorized_backtest(
        close, signals, risk=risk, high=high, low=low,
        timeframe=tf, sl_distances=sl_distances,
    )
    return res.equity, returns_from_equity(res.equity)


def rebuild_holdout_equities(survivors):
    """Rebuild holdout equity curves for all survivors."""
    logger.info("Rebuilding holdout equity curves (V5b mode)...")
    returns_dict = {}
    valid = []

    for i, s in enumerate(survivors):
        key = f"{s['symbol']}/{s['strategy']}/{s['timeframe']}"
        try:
            eq, rets = build_equity(s)
            if eq is None:
                logger.warning(f"  [{i+1}] {key}: skipped (too short)")
                continue

            returns_dict[key] = rets
            s["_key"] = key
            s["_equity"] = eq
            s["_returns"] = rets
            valid.append(s)

            m = compute_all_metrics(eq, s["timeframe"])
            tag = f"r={s['risk_key']}" + (" +ov" if s["overlay"] else "")
            logger.info(f"  [{i+1}/{len(survivors)}] {key} ({tag}): "
                       f"Sharpe={m['sharpe']:.3f} Ret={m['total_return']*100:.1f}% "
                       f"DD={m['max_drawdown']*100:.1f}%")
        except Exception as e:
            logger.error(f"  [{i+1}] {key}: {e}")

    logger.info(f"Rebuilt {len(valid)}/{len(survivors)} equity curves")
    return valid, returns_dict


# ── Correlation Deduplication ───────────────────────────

def deduplicate_by_correlation(survivors, returns_dict, max_corr=MAX_CORRELATION):
    """Remove highly correlated combos, keeping higher Sharpe."""
    if len(survivors) <= 1:
        return survivors

    keys = [s["_key"] for s in survivors]
    min_len = min(len(returns_dict[k]) for k in keys)
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
                if survivors[i]["ho_sharpe"] >= survivors[j]["ho_sharpe"]:
                    to_remove.add(j)
                    logger.info(f"  Dedup: {keys[j]} (corr={corr[i,j]:.2f} with {keys[i]})")
                else:
                    to_remove.add(i)
                    logger.info(f"  Dedup: {keys[i]} (corr={corr[i,j]:.2f} with {keys[j]})")

    filtered = [s for idx, s in enumerate(survivors) if idx not in to_remove]
    logger.info(f"Correlation dedup: {len(survivors)} → {len(filtered)} (removed {len(to_remove)})")
    return filtered


# ── Covariance Shrinkage ────────────────────────────────

def ledoit_wolf(rets_matrix):
    """Ledoit-Wolf covariance shrinkage estimator."""
    T, N = rets_matrix.shape
    X = rets_matrix - rets_matrix.mean(axis=0)
    S = X.T @ X / T
    mu = np.trace(S) / N
    F = mu * np.eye(N)
    d2 = np.sum((S - F) ** 2) / N
    b2_sum = sum(np.sum((X[t:t+1].T @ X[t:t+1] - S) ** 2) / N for t in range(T))
    b2 = min(b2_sum / T**2, d2)
    alpha = np.clip(b2 / max(d2, 1e-10), 0, 1)
    logger.info(f"Ledoit-Wolf shrinkage: alpha={alpha:.3f}")
    return alpha * F + (1 - alpha) * S


# ── Portfolio Optimization ──────────────────────────────

def optimize_weights(survivors, returns_dict, objective="sharpe"):
    """
    Optimize weights with Markowitz + hard constraints.
    Objectives: 'sharpe', 'return_dd', 'max_return'
    """
    n = len(survivors)
    keys = [s["_key"] for s in survivors]
    min_len = min(len(returns_dict[k]) for k in keys)
    rets_matrix = np.column_stack([returns_dict[k][:min_len] for k in keys])

    cov = ledoit_wolf(rets_matrix)
    mu = rets_matrix.mean(axis=0)

    def neg_objective(w):
        port_ret = w @ mu
        port_vol = np.sqrt(max(w @ cov @ w, 1e-20))
        if objective == "sharpe":
            return -port_ret / port_vol
        elif objective == "return_dd":
            return -(port_ret - 0.5 * port_vol)
        elif objective == "max_return":
            return -port_ret + 0.1 * port_vol
        return -port_ret / port_vol

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Per-symbol cap
    symbols = {}
    for i, s in enumerate(survivors):
        symbols.setdefault(s["symbol"], []).append(i)
    for sym, indices in symbols.items():
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=indices: MAX_WEIGHT_PER_SYMBOL - sum(w[i] for i in idx)
        })

    bounds = [(MIN_WEIGHT, MAX_WEIGHT_PER_COMBO)] * n
    w0 = np.ones(n) / n

    result = minimize(neg_objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    if result.success:
        w = np.maximum(result.x, 0)
        return w / w.sum()
    logger.warning(f"Optimization failed ({objective}): {result.message}, using equal weight")
    return np.ones(n) / n


def top3_heavy_weights(n):
    """V4b-style top3_heavy weighting."""
    if n <= 3:
        return np.ones(n) / n
    base = [0.25, 0.25, 0.15]
    remaining = 1.0 - sum(base)
    n_rest = n - 3
    rest_each = remaining / n_rest
    weights = base + [rest_each] * n_rest
    return np.array(weights[:n])


# ── Portfolio Simulation ────────────────────────────────

def simulate_portfolio(survivors, weights, returns_dict, leverage=1.0):
    """Simulate combined portfolio equity curve."""
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


# ── Monte Carlo ─────────────────────────────────────────

def monte_carlo(returns, n_sims=N_MONTE_CARLO):
    """Block bootstrap Monte Carlo on holdout returns."""
    n = len(returns)
    block_size = min(20, n // 5)
    results = {}

    for months in [3, 6, 12, 24, 36]:
        bars = min(months * 30, n * 3)  # Allow extrapolation up to 3x
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
            "p25": float(np.percentile(sims, 25)),
            "median": float(np.percentile(sims, 50)),
            "p75": float(np.percentile(sims, 75)),
            "p95": float(np.percentile(sims, 95)),
            "prob_pos": float(np.mean(sims > INITIAL_CAPITAL)),
            "prob_10pct": float(np.mean(sims > INITIAL_CAPITAL * 1.10)),
            "prob_20pct": float(np.mean(sims > INITIAL_CAPITAL * 1.20)),
            "prob_ruin": float(np.mean(sims < INITIAL_CAPITAL * 0.50)),
        }
    return results


# ── Report Generation ───────────────────────────────────

def generate_report(portfolios, best_name, best, mc, sym_alloc, elapsed):
    """Generate markdown report."""
    m = best["metrics"]
    lines = []
    lines.append("# Portfolio V5b — Construction & Validation")
    lines.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    lines.append(f"**Diagnostic** : {DIAG_PATH.name}")
    lines.append(f"**Durée** : {elapsed:.1f} min")
    lines.append(f"**Objectif** : +15% annuel, DD < -20%, Sharpe > 1.0")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Lessons applied
    lines.append("## Leçons appliquées (erreurs passées corrigées)")
    lines.append("")
    lines.append("| Erreur passée | Version | Correction V5b |")
    lines.append("|---------------|---------|----------------|")
    lines.append("| Concentration ETH 95% | V3 | Cap symbol 55% |")
    lines.append("| MC sur full data | V3 | MC holdout-only (5000 sims) |")
    lines.append("| Sur-diversification (37 combos) | V4 | Concentration top combos |")
    lines.append("| Overlays trop agressifs | V4 | Overlay sélectif (seulement si diagnostic meilleur) |")
    lines.append("| Markowitz optimise Sharpe pas return | V4 | Multi-objectif (sharpe + return_dd + max_return) |")
    lines.append("| Pas de corrélation dedup | V4b | Dedup corr > 0.85 |")
    lines.append("| Pas d'overlay du tout | V4b | Overlay sélectif |")
    lines.append("| 1 seed seulement | V4 | Multi-seed 3 + seed_std filter |")
    lines.append("")

    # Comparison table
    lines.append("## Comparaison des variantes")
    lines.append("")
    lines.append("| Portfolio | Sharpe | Sortino | Return | Max DD | Calmar | N |")
    lines.append("|-----------|--------|---------|--------|--------|--------|---|")
    for name, p in sorted(portfolios.items(), key=lambda x: -x[1]["metrics"]["total_return"]):
        pm = p["metrics"]
        n = int(np.sum(p["weights"] > 0.01)) if isinstance(p["weights"], np.ndarray) else "?"
        ok = "✅" if pm["total_return"] >= 0.12 else "❌"
        star = " **⭐**" if name == best_name else ""
        lines.append(
            f"| {ok} {name}{star} | {pm['sharpe']:.2f} | {pm['sortino']:.2f} "
            f"| {pm['total_return']*100:.1f}% | {pm['max_drawdown']*100:.1f}% "
            f"| {pm['calmar']:.2f} | {n} |"
        )
    lines.append("")

    # Recommended
    lines.append(f"## 🏆 Recommandé : {best_name}")
    lines.append("")
    lines.append("| Métrique | Valeur | Objectif | Status |")
    lines.append("|----------|--------|----------|--------|")
    lines.append(f"| Return | {m['total_return']*100:.1f}% | ≥15% | {'✅' if m['total_return']>=0.15 else '⚠️' if m['total_return']>=0.12 else '❌'} |")
    lines.append(f"| Max DD | {m['max_drawdown']*100:.1f}% | ≥-20% | {'✅' if m['max_drawdown']>=-0.20 else '❌'} |")
    lines.append(f"| Sharpe | {m['sharpe']:.2f} | ≥1.0 | {'✅' if m['sharpe']>=1.0 else '⚠️' if m['sharpe']>=0.5 else '❌'} |")
    lines.append(f"| Sortino | {m['sortino']:.2f} | ≥1.0 | {'✅' if m['sortino']>=1.0 else '❌'} |")
    lines.append(f"| Calmar | {m['calmar']:.2f} | ≥1.0 | {'✅' if m['calmar']>=1.0 else '❌'} |")
    lines.append("")

    # Allocations
    if isinstance(best["weights"], np.ndarray):
        lines.append("### Allocations")
        lines.append("")
        lines.append("| Poids | Symbol | Stratégie | TF | Risk | Ov | HO Sharpe | HO Return | HO DD | Trail | Seed_std |")
        lines.append("|-------|--------|-----------|-----|------|-----|-----------|-----------|-------|-------|----------|")
        for s, w in sorted(zip(best["survivors"], best["weights"]), key=lambda x: -x[1]):
            if w > 0.01:
                ov_tag = "Y" if s["overlay"] else "-"
                lines.append(
                    f"| {w*100:.1f}% | {s['symbol']} | {s['strategy']} | {s['timeframe']} "
                    f"| {s['risk_key']} | {ov_tag} | {s['ho_sharpe']:.3f} | {s['ho_return']*100:.1f}% "
                    f"| {s['ho_dd']*100:.1f}% | {s.get('trailing_atr_mult',0):.2f} "
                    f"| {s.get('seed_sharpe_std',0):.3f} |"
                )
        lines.append("")

        lines.append("### Allocation par symbol")
        lines.append("")
        lines.append("| Symbol | Allocation |")
        lines.append("|--------|-----------|")
        for sym, alloc in sorted(sym_alloc.items(), key=lambda x: -x[1]):
            lines.append(f"| {sym} | {alloc*100:.1f}% |")
        lines.append("")

    # Monte Carlo
    lines.append("### Monte Carlo ($10,000 — 5000 sims)")
    lines.append("")
    lines.append("| Horizon | P5 | P25 | Médian | P75 | P95 | P(>0) | P(>10%) | P(>20%) | P(ruine) |")
    lines.append("|---------|-----|-----|--------|-----|-----|-------|---------|---------|----------|")
    for months, r in mc.items():
        lines.append(
            f"| {months}M | ${r['p5']:,.0f} | ${r['p25']:,.0f} | ${r['median']:,.0f} "
            f"| ${r['p75']:,.0f} | ${r['p95']:,.0f} | {r['prob_pos']*100:.0f}% "
            f"| {r['prob_10pct']*100:.0f}% | {r['prob_20pct']*100:.0f}% | {r['prob_ruin']*100:.1f}% |"
        )
    lines.append("")

    # Comparison with previous versions
    lines.append("### Comparaison historique")
    lines.append("")
    lines.append("| Métrique | V3b | V4 | V4b | **V5b** |")
    lines.append("|----------|-----|-----|-----|---------|")
    lines.append(f"| Return | +9.8% | +4.9% | +19.8% | **+{m['total_return']*100:.1f}%** |")
    lines.append(f"| Sharpe | 1.19 | 2.59 | 1.35 | **{m['sharpe']:.2f}** |")
    lines.append(f"| Max DD | -4.9% | -0.8% | -8.5% | **{m['max_drawdown']*100:.1f}%** |")
    lines.append(f"| Calmar | 1.91 | 5.99 | 2.17 | **{m['calmar']:.2f}** |")
    if sym_alloc:
        lines.append(f"| ETH % | 95% | 53% | 70% | **{sym_alloc.get('ETHUSDT',0)*100:.0f}%** |")
    lines.append(f"| Objectif +15% | ❌ | ❌ | ✅ | {'✅' if m['total_return']>=0.15 else '⚠️' if m['total_return']>=0.12 else '❌'} |")
    lines.append("")

    # V5b features usage
    lines.append("### Features V5b utilisées")
    lines.append("")
    active = best["survivors"] if not isinstance(best["weights"], np.ndarray) else \
        [s for s, w in zip(best["survivors"], best["weights"]) if w > 0.01]
    n_trail = sum(1 for s in active if s.get("trailing_atr_mult", 0) > 0.01)
    n_be = sum(1 for s in active if s.get("breakeven_trigger_pct", 0) > 0.001)
    n_maxh = sum(1 for s in active if s.get("max_holding_bars", 0) > 0)
    n_risk = sum(1 for s in active if s.get("risk_key", "flat") != "flat")
    n_ov = sum(1 for s in active if s.get("overlay", False))
    lines.append(f"- **Trailing stop** : {n_trail}/{len(active)} combos")
    lines.append(f"- **Breakeven** : {n_be}/{len(active)} combos")
    lines.append(f"- **Max holding** : {n_maxh}/{len(active)} combos")
    lines.append(f"- **Risk-based sizing** : {n_risk}/{len(active)} combos")
    lines.append(f"- **Overlay** : {n_ov}/{len(active)} combos")
    lines.append("")

    lines.append("---")
    lines.append(f"*Généré le {datetime.now().strftime('%d %B %Y %H:%M')}*")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "20_portfolio_v5b.md"
    report_path.write_text("\n".join(lines))
    logger.info(f"Report: {report_path}")
    return report_path


# ── Main ────────────────────────────────────────────────

def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PORTFOLIO V5b — Construction")
    logger.info("=" * 60)

    # Step 1: Load survivors
    survivors = load_survivors()

    # Step 2: Rebuild holdout equity curves (V5b mode)
    valid, returns_dict = rebuild_holdout_equities(survivors)

    # Step 3: Correlation deduplication
    logger.info("\n--- Correlation Deduplication ---")
    deduped = deduplicate_by_correlation(valid, returns_dict)

    # Step 4: Build multiple portfolio variants
    logger.info("\n--- Building Portfolio Variants ---")
    portfolios = {}

    # ── Variant A: Concentrated top 8 (lesson from V4b: concentration works) ──
    top8 = deduped[:8]
    if len(top8) >= 3:
        logger.info(f"\n  Variant A: Concentrated top {len(top8)}")
        r_a = {s["_key"]: returns_dict[s["_key"]] for s in top8}

        # A1: top3_heavy weights (V4b style)
        w_t3h = top3_heavy_weights(len(top8))
        eq, rets = simulate_portfolio(top8, w_t3h, r_a)
        portfolios["A1_top3_heavy"] = {
            "weights": w_t3h, "survivors": top8,
            "equity": eq, "returns": rets,
            "metrics": compute_all_metrics(eq, "1d"),
        }

        # A2: Markowitz sharpe
        w_sharpe = optimize_weights(top8, r_a, "sharpe")
        eq, rets = simulate_portfolio(top8, w_sharpe, r_a)
        portfolios["A2_markowitz_sharpe"] = {
            "weights": w_sharpe, "survivors": top8,
            "equity": eq, "returns": rets,
            "metrics": compute_all_metrics(eq, "1d"),
        }

        # A3: Markowitz return_dd
        w_retdd = optimize_weights(top8, r_a, "return_dd")
        eq, rets = simulate_portfolio(top8, w_retdd, r_a)
        portfolios["A3_markowitz_retdd"] = {
            "weights": w_retdd, "survivors": top8,
            "equity": eq, "returns": rets,
            "metrics": compute_all_metrics(eq, "1d"),
        }

    # ── Variant B: Diversified top 12 ──
    top12 = deduped[:12]
    if len(top12) >= 5:
        logger.info(f"\n  Variant B: Diversified top {len(top12)}")
        r_b = {s["_key"]: returns_dict[s["_key"]] for s in top12}

        w_b = optimize_weights(top12, r_b, "sharpe")
        eq, rets = simulate_portfolio(top12, w_b, r_b)
        portfolios["B_diversified_12"] = {
            "weights": w_b, "survivors": top12,
            "equity": eq, "returns": rets,
            "metrics": compute_all_metrics(eq, "1d"),
        }

    # ── Variant C: Leverage on best A variant ──
    # Find best A variant first
    a_variants = {k: v for k, v in portfolios.items() if k.startswith("A")}
    if a_variants:
        best_a_name = max(a_variants, key=lambda k: a_variants[k]["metrics"]["calmar"])
        best_a = a_variants[best_a_name]
        r_c = {s["_key"]: returns_dict[s["_key"]] for s in best_a["survivors"]}

        for lev in [1.25, 1.5, 2.0]:
            eq, rets = simulate_portfolio(best_a["survivors"], best_a["weights"], r_c, leverage=lev)
            m = compute_all_metrics(eq, "1d")
            portfolios[f"C_{best_a_name}_{lev:.2f}x"] = {
                "weights": best_a["weights"], "survivors": best_a["survivors"],
                "equity": eq, "returns": rets, "metrics": m,
                "leverage": lev,
            }

    # ── Variant D: All deduped (max diversification) ──
    if len(deduped) >= 5:
        logger.info(f"\n  Variant D: All deduped ({len(deduped)} combos)")
        r_d = {s["_key"]: returns_dict[s["_key"]] for s in deduped}
        w_d = optimize_weights(deduped, r_d, "sharpe")
        eq, rets = simulate_portfolio(deduped, w_d, r_d)
        portfolios["D_all_deduped"] = {
            "weights": w_d, "survivors": deduped,
            "equity": eq, "returns": rets,
            "metrics": compute_all_metrics(eq, "1d"),
        }

    # ── Compare all ──
    logger.info("\n" + "=" * 110)
    logger.info("COMPARISON — All V5b variants")
    logger.info("=" * 110)
    fmt = "{:<35} {:>8} {:>8} {:>10} {:>10} {:>8} {:>4} {:>6}"
    logger.info(fmt.format("Portfolio", "Sharpe", "Sortino", "Return", "Max DD", "Calmar", "N", "Lev"))
    logger.info("-" * 110)

    for name, p in sorted(portfolios.items(), key=lambda x: -x[1]["metrics"]["total_return"]):
        pm = p["metrics"]
        n = int(np.sum(p["weights"] > 0.01)) if isinstance(p["weights"], np.ndarray) else "?"
        lev = p.get("leverage", 1.0)
        logger.info(fmt.format(
            name,
            f"{pm['sharpe']:.2f}",
            f"{pm['sortino']:.2f}",
            f"{pm['total_return']*100:.1f}%",
            f"{pm['max_drawdown']*100:.1f}%",
            f"{pm['calmar']:.2f}",
            str(n),
            f"{lev:.1f}x",
        ))

    # ── Select best ──
    # Among those meeting +12% return target, pick best Calmar (lesson: balance return + DD)
    target_ok = {k: v for k, v in portfolios.items()
                 if v["metrics"]["total_return"] >= 0.12
                 and v["metrics"]["max_drawdown"] >= -0.20}

    if target_ok:
        best_name = max(target_ok, key=lambda k: target_ok[k]["metrics"]["calmar"])
    else:
        # Fallback: best Calmar among all
        best_name = max(portfolios, key=lambda k: portfolios[k]["metrics"]["calmar"])

    best = portfolios[best_name]
    m = best["metrics"]

    logger.info(f"\n🏆 RECOMMENDED: {best_name}")
    logger.info(f"  Return: {m['total_return']*100:.1f}%")
    logger.info(f"  Sharpe: {m['sharpe']:.2f}")
    logger.info(f"  Sortino: {m['sortino']:.2f}")
    logger.info(f"  Max DD: {m['max_drawdown']*100:.1f}%")
    logger.info(f"  Calmar: {m['calmar']:.2f}")

    # Symbol allocation
    sym_alloc = {}
    if isinstance(best["weights"], np.ndarray):
        for s, w in zip(best["survivors"], best["weights"]):
            sym_alloc[s["symbol"]] = sym_alloc.get(s["symbol"], 0) + w
        logger.info("\n  Symbol allocation:")
        for sym, alloc in sorted(sym_alloc.items(), key=lambda x: -x[1]):
            logger.info(f"    {sym}: {alloc*100:.1f}%")

        logger.info("\n  Combo allocations:")
        for s, w in sorted(zip(best["survivors"], best["weights"]), key=lambda x: -x[1]):
            if w > 0.01:
                tag = f"r={s['risk_key']}" + (" +ov" if s["overlay"] else "")
                logger.info(f"    {w*100:.1f}% {s['_key']} ({tag}) "
                           f"Sharpe={s['ho_sharpe']:.3f} Ret={s['ho_return']*100:.1f}%")

    # Monte Carlo on recommended
    logger.info("\n--- Monte Carlo (5000 sims, block bootstrap) ---")
    mc = monte_carlo(best["returns"])
    for months, r in mc.items():
        logger.info(f"  {months}M: median=${r['median']:,.0f} "
                   f"P(>0)={r['prob_pos']*100:.0f}% "
                   f"P(>10%)={r['prob_10pct']*100:.0f}% "
                   f"P(ruine)={r['prob_ruin']*100:.1f}%")

    # ── Save JSON ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for name, p in portfolios.items():
        save_data[name] = {
            "metrics": p["metrics"],
            "leverage": p.get("leverage", 1.0),
            "weights": p["weights"].tolist() if isinstance(p["weights"], np.ndarray) else p["weights"],
            "allocations": [
                {
                    "weight": float(w),
                    "symbol": s["symbol"], "strategy": s["strategy"],
                    "timeframe": s["timeframe"],
                    "risk_key": s["risk_key"], "overlay": s["overlay"],
                    "ho_sharpe": s["ho_sharpe"], "ho_return": s["ho_return"],
                    "ho_dd": s["ho_dd"], "ho_trades": s["ho_trades"],
                    "trailing_atr_mult": s.get("trailing_atr_mult", 0),
                    "breakeven_trigger_pct": s.get("breakeven_trigger_pct", 0),
                    "max_holding_bars": s.get("max_holding_bars", 0),
                    "seed_sharpe_std": s.get("seed_sharpe_std", 0),
                    "last_params": s["last_params"],
                }
                for s, w in zip(p["survivors"], p["weights"])
                if (isinstance(w, (int, float)) and w > 0.01) or
                   (hasattr(w, '__float__') and float(w) > 0.01)
            ] if isinstance(p["weights"], np.ndarray) else [],
        }
    save_data["recommended"] = best_name
    save_data["monte_carlo"] = {best_name: mc}
    save_data["config"] = {
        "diagnostic": str(DIAG_PATH),
        "max_weight_per_symbol": MAX_WEIGHT_PER_SYMBOL,
        "max_weight_per_combo": MAX_WEIGHT_PER_COMBO,
        "max_correlation": MAX_CORRELATION,
        "min_ho_sharpe": MIN_HO_SHARPE,
        "min_ho_trades": MIN_HO_TRADES,
        "max_seed_std": MAX_SEED_STD,
        "n_monte_carlo": N_MONTE_CARLO,
        "cutoff_date": CUTOFF_DATE,
    }

    json_path = RESULTS_DIR / f"portfolio_v5b_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    logger.info(f"\nJSON: {json_path}")

    # ── Report ──
    elapsed = (time.time() - t0) / 60
    generate_report(portfolios, best_name, best, mc, sym_alloc, elapsed)

    # ── Final summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"PORTFOLIO V5b COMPLETE ({elapsed:.1f} min)")
    logger.info(f"  Recommended: {best_name}")
    logger.info(f"  Return: {m['total_return']*100:.1f}%")
    logger.info(f"  Sharpe: {m['sharpe']:.2f} | Sortino: {m['sortino']:.2f}")
    logger.info(f"  Max DD: {m['max_drawdown']*100:.1f}% | Calmar: {m['calmar']:.2f}")
    logger.info(f"  MC 12M: median=${mc[12]['median']:,.0f}, P(>0)={mc[12]['prob_pos']*100:.0f}%")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
