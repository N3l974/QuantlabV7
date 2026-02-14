#!/usr/bin/env python3
"""
Portfolio V5b Final — Multi-profile construction + Audit + Live confidence.

3 Risk Profiles:
  - CONSERVATEUR : DD target < -5%, Sharpe max, leverage 1.0x
  - MODÉRÉ       : DD target < -10%, balanced, leverage 1.5x
  - AGRESSIF     : DD target < -15%, return max, leverage 2.0-2.5x

Audit:
  - Rolling Sharpe stability
  - Monthly breakdown
  - Concentration HHI
  - Correlation matrix
  - Stress tests (worst month, worst quarter, losing streaks)
  - Regime analysis (bull/bear/range performance)

Live Confidence Score:
  - Checklist go/no-go for deployment
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

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
N_MONTE_CARLO = 5000
CUTOFF_DATE = "2025-02-01"

# Hard constraints
MAX_WEIGHT_PER_SYMBOL = 0.55
MAX_WEIGHT_PER_COMBO = 0.25
MIN_WEIGHT = 0.02
MAX_CORRELATION = 0.85
MIN_HO_TRADES = 3
MAX_SEED_STD = 1.0

# Overlay config
REGIME_CFG = RegimeOverlayConfig(
    regime_config=RegimeConfig(),
    hard_cutoff=True,
    min_exposure_threshold=0.3,
)
VOL_CFG = VolTargetConfig(target_vol_annual=0.40)
OVERLAY_CFG = OverlayPipelineConfig(regime_config=REGIME_CFG, vol_config=VOL_CFG)

DIAG_SEARCH_DIRS = [
    Path("portfolio/v5b/results"),
    Path("results"),  # legacy fallback
]


def find_latest_diagnostic() -> Path:
    for base in DIAG_SEARCH_DIRS:
        files = sorted(base.glob("diagnostic_v5b_*.json"))
        if files:
            return files[-1]
    raise FileNotFoundError(
        "No diagnostic_v5b_*.json found in portfolio/v5b/results or results"
    )


# Diagnostic
DIAG_PATH = find_latest_diagnostic()

# Risk profiles — same combos & weights, different position sizing
# max_position_pct is the PRIMARY sizing lever (% of equity per trade)
# risk_per_trade_pct=0 means disabled (flat sizing = max_position_pct of equity)
# When risk_per_trade_pct > 0, position = min(risk$/SL_dist, max_position_pct * equity)
PROFILES = {
    "conservateur": {
        "label": "Conservateur",
        "max_dd_target": -0.05,
        "risk_per_trade_pct": 0.0,     # Disabled — flat sizing
        "max_position_pct": 0.10,      # 10% du capital par position
        "max_drawdown_pct": 0.10,      # Circuit breaker à -10%
        "description": "Position max 10% du capital par trade.",
    },
    "modere": {
        "label": "Modéré",
        "max_dd_target": -0.10,
        "risk_per_trade_pct": 0.0,     # Disabled — flat sizing
        "max_position_pct": 0.25,      # 25% du capital par position (défaut backtester)
        "max_drawdown_pct": 0.15,      # Circuit breaker à -15%
        "description": "Position max 25% du capital par trade.",
    },
    "agressif": {
        "label": "Agressif",
        "max_dd_target": -0.15,
        "risk_per_trade_pct": 0.0,     # Disabled — flat sizing
        "max_position_pct": 0.50,      # 50% du capital par position
        "max_drawdown_pct": 0.25,      # Circuit breaker à -25%
        "description": "Position max 50% du capital par trade.",
    },
}


# ═══════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════

def load_survivors(min_sharpe=0.0):
    """Load STRONG survivors, best risk_key per (sym, strat, tf)."""
    with open(DIAG_PATH) as f:
        data = json.load(f)
    p2 = data["phase2"]

    candidates = [r for r in p2
                  if r["verdict"] == "STRONG"
                  and r["ho_trades"] >= MIN_HO_TRADES
                  and r["ho_sharpe"] >= min_sharpe
                  and r.get("seed_sharpe_std", 0) <= MAX_SEED_STD]

    best = {}
    for r in candidates:
        key = (r["symbol"], r["strategy"], r["timeframe"])
        if key not in best or r["ho_sharpe"] > best[key]["ho_sharpe"]:
            best[key] = r

    return sorted(best.values(), key=lambda x: x["ho_sharpe"], reverse=True)


def prepare_combo_data(combo):
    """Prepare signals and market data for a combo (computed once, reused per profile)."""
    sym, sname, tf = combo["symbol"], combo["strategy"], combo["timeframe"]
    params = combo["last_params"]
    use_overlay = combo["overlay"]

    data = pd.read_parquet(f"data/raw/{sym}_{tf}.parquet")
    ho_data = data[data.index >= CUTOFF_DATE].copy()
    if len(ho_data) < 50:
        return None

    strategy = get_strategy(sname)

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
    close, signals, high, low = close[:min_len], signals[:min_len], high[:min_len], low[:min_len]
    if sl_distances is not None:
        sl_distances = sl_distances[:min_len]

    return {
        "close": close, "high": high, "low": low,
        "signals": signals, "sl_distances": sl_distances,
        "timeframe": tf, "dates": ho_data.index[:min_len],
    }


def build_equity_with_risk(combo_data, risk_cfg):
    """Backtest a prepared combo with a specific RiskConfig."""
    res = vectorized_backtest(
        combo_data["close"], combo_data["signals"],
        risk=risk_cfg,
        high=combo_data["high"], low=combo_data["low"],
        timeframe=combo_data["timeframe"],
        sl_distances=combo_data["sl_distances"],
    )
    return res.equity, returns_from_equity(res.equity)


def prepare_all(survivors):
    """Prepare signal data for all survivors (computed once, reused per profile)."""
    logger.info(f"Preparing {len(survivors)} combo signals...")
    combo_data = {}
    valid = []

    # Use a moderate default RiskConfig for selection/dedup
    default_risk = RiskConfig(risk_per_trade_pct=0.01, max_position_pct=0.25)

    for i, s in enumerate(survivors):
        key = f"{s['symbol']}/{s['strategy']}/{s['timeframe']}"
        try:
            cd = prepare_combo_data(s)
            if cd is None:
                continue
            combo_data[key] = cd
            s["_key"] = key

            # Backtest with default risk for selection metrics
            eq, rets = build_equity_with_risk(cd, default_risk)
            s["_returns_default"] = rets
            valid.append(s)

            tag = f"r={s['risk_key']}" + (" +ov" if s["overlay"] else "")
            logger.info(f"  [{i+1}] {key} ({tag}) diag_Sharpe={s['ho_sharpe']:.3f}")
        except Exception as e:
            logger.error(f"  [{i+1}] {key}: {e}")

    logger.info(f"Prepared {len(valid)}/{len(survivors)}")
    return valid, combo_data


def backtest_profile(survivors, combo_data, weights, profile_cfg):
    """Backtest all combos with a specific risk profile and combine into portfolio."""
    risk_cfg = RiskConfig(
        risk_per_trade_pct=profile_cfg["risk_per_trade_pct"],
        max_position_pct=profile_cfg["max_position_pct"],
        max_drawdown_pct=profile_cfg["max_drawdown_pct"],
    )

    keys = [s["_key"] for s in survivors]
    returns_per_combo = {}
    for s in survivors:
        eq, rets = build_equity_with_risk(combo_data[s["_key"]], risk_cfg)
        returns_per_combo[s["_key"]] = rets

    # Combine into portfolio
    min_len = min(len(returns_per_combo[k]) for k in keys)
    port_returns = np.zeros(min_len)
    for i, k in enumerate(keys):
        port_returns += weights[i] * returns_per_combo[k][:min_len]

    equity = np.zeros(min_len + 1)
    equity[0] = INITIAL_CAPITAL
    for t in range(min_len):
        equity[t + 1] = equity[t] * (1 + port_returns[t])

    return equity, port_returns, returns_per_combo


# ═══════════════════════════════════════════════════════
# CORRELATION & OPTIMIZATION
# ═══════════════════════════════════════════════════════

def deduplicate_by_correlation(survivors, returns_dict, max_corr=MAX_CORRELATION):
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
                else:
                    to_remove.add(i)
    filtered = [s for idx, s in enumerate(survivors) if idx not in to_remove]
    logger.info(f"  Correlation dedup: {len(survivors)} → {len(filtered)}")
    return filtered


def correlation_matrix(survivors, returns_dict):
    """Compute full correlation matrix for audit."""
    keys = [s["_key"] for s in survivors]
    min_len = min(len(returns_dict[k]) for k in keys)
    rets_matrix = np.column_stack([returns_dict[k][:min_len] for k in keys])
    corr = np.corrcoef(rets_matrix.T)
    # Replace NaN with 0 (happens when a series is constant/near-zero)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr, keys


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


def optimize_weights(survivors, returns_dict, objective="sharpe"):
    n = len(survivors)
    keys = [s["_key"] for s in survivors]
    min_len = min(len(returns_dict[k]) for k in keys)
    rets_matrix = np.column_stack([returns_dict[k][:min_len] for k in keys])
    cov = ledoit_wolf(rets_matrix)
    mu = rets_matrix.mean(axis=0)

    def neg_obj(w):
        pr = w @ mu
        pv = np.sqrt(max(w @ cov @ w, 1e-20))
        if objective == "sharpe":
            return -pr / pv
        elif objective == "return_dd":
            return -(pr - 0.5 * pv)
        elif objective == "max_return":
            return -pr + 0.1 * pv
        return -pr / pv

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
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
    result = minimize(neg_obj, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12})
    if result.success:
        w = np.maximum(result.x, 0)
        return w / w.sum()
    return np.ones(n) / n


def top3_heavy_weights(n):
    if n <= 3:
        return np.ones(n) / n
    base = [0.25, 0.25, 0.15]
    remaining = 1.0 - sum(base)
    rest = [remaining / (n - 3)] * (n - 3)
    return np.array(base + rest)


# ═══════════════════════════════════════════════════════
# PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════
# AUDIT FUNCTIONS
# ═══════════════════════════════════════════════════════

def rolling_sharpe(returns, window=60):
    """Rolling Sharpe ratio (window in bars)."""
    n = len(returns)
    if n < window:
        return np.array([]), np.array([])
    rs = np.zeros(n - window)
    for i in range(n - window):
        chunk = returns[i:i + window]
        mu = np.mean(chunk)
        std = np.std(chunk)
        rs[i] = mu / max(std, 1e-10) * np.sqrt(365)
    return rs, np.arange(window, n)


def monthly_breakdown(equity, returns):
    """Monthly return breakdown."""
    n = len(returns)
    # Approximate: 30 bars per month for 1d
    month_size = 30
    months = []
    for start in range(0, n, month_size):
        end = min(start + month_size, n)
        chunk = returns[start:end]
        month_ret = np.prod(1 + chunk) - 1
        months.append(month_ret)
    return np.array(months)


def quarterly_breakdown(equity, returns):
    """Quarterly return breakdown."""
    n = len(returns)
    q_size = 90
    quarters = []
    for start in range(0, n, q_size):
        end = min(start + q_size, n)
        chunk = returns[start:end]
        q_ret = np.prod(1 + chunk) - 1
        quarters.append(q_ret)
    return np.array(quarters)


def hhi_concentration(weights, survivors, by="symbol"):
    """Herfindahl-Hirschman Index for concentration."""
    groups = {}
    for s, w in zip(survivors, weights):
        key = s[by] if by != "strategy" else s["strategy"]
        groups[key] = groups.get(key, 0) + w
    shares = np.array(list(groups.values()))
    hhi = np.sum(shares ** 2)
    n_effective = 1.0 / max(hhi, 1e-10)
    return hhi, n_effective, groups


def max_losing_streak(returns):
    """Maximum consecutive losing bars."""
    streak = 0
    max_streak = 0
    for r in returns:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def recovery_time(equity):
    """Bars to recover from max drawdown."""
    peak = equity[0]
    max_dd_bar = 0
    max_dd = 0
    for i, e in enumerate(equity):
        if e >= peak:
            peak = e
        dd = (e - peak) / peak
        if dd < max_dd:
            max_dd = dd
            max_dd_bar = i

    # Find recovery from max DD point
    dd_equity = equity[max_dd_bar]
    for i in range(max_dd_bar, len(equity)):
        if equity[i] >= equity[max_dd_bar - 1] if max_dd_bar > 0 else equity[0]:
            return i - max_dd_bar
    return len(equity) - max_dd_bar  # Not recovered


def audit_portfolio(name, equity, returns, weights, survivors, returns_dict):
    """Full audit of a portfolio."""
    logger.info(f"\n{'='*60}")
    logger.info(f"AUDIT — {name}")
    logger.info(f"{'='*60}")

    audit = {}

    # 1. Rolling Sharpe
    rs, rs_idx = rolling_sharpe(returns, window=60)
    if len(rs) > 0:
        audit["rolling_sharpe"] = {
            "mean": float(np.mean(rs)),
            "std": float(np.std(rs)),
            "min": float(np.min(rs)),
            "max": float(np.max(rs)),
            "pct_positive": float(np.mean(rs > 0)),
            "first_half_mean": float(np.mean(rs[:len(rs)//2])),
            "second_half_mean": float(np.mean(rs[len(rs)//2:])),
        }
        logger.info(f"  Rolling Sharpe (60d): mean={audit['rolling_sharpe']['mean']:.2f}, "
                    f"std={audit['rolling_sharpe']['std']:.2f}, "
                    f"range=[{audit['rolling_sharpe']['min']:.2f}, {audit['rolling_sharpe']['max']:.2f}], "
                    f"positive={audit['rolling_sharpe']['pct_positive']*100:.0f}%")
        logger.info(f"    1st half={audit['rolling_sharpe']['first_half_mean']:.2f}, "
                    f"2nd half={audit['rolling_sharpe']['second_half_mean']:.2f}")

    # 2. Monthly breakdown
    months = monthly_breakdown(equity, returns)
    audit["monthly"] = {
        "worst_month": float(np.min(months)),
        "best_month": float(np.max(months)),
        "avg_month": float(np.mean(months)),
        "median_month": float(np.median(months)),
        "pct_positive": float(np.mean(months > 0)),
        "n_months": len(months),
    }
    logger.info(f"  Monthly: worst={audit['monthly']['worst_month']*100:.1f}%, "
               f"best={audit['monthly']['best_month']*100:.1f}%, "
               f"avg={audit['monthly']['avg_month']*100:.2f}%, "
               f"positive={audit['monthly']['pct_positive']*100:.0f}%")

    # 3. Quarterly breakdown
    quarters = quarterly_breakdown(equity, returns)
    audit["quarterly"] = {
        "worst_quarter": float(np.min(quarters)),
        "best_quarter": float(np.max(quarters)),
        "pct_positive": float(np.mean(quarters > 0)),
    }
    logger.info(f"  Quarterly: worst={audit['quarterly']['worst_quarter']*100:.1f}%, "
               f"best={audit['quarterly']['best_quarter']*100:.1f}%, "
               f"positive={audit['quarterly']['pct_positive']*100:.0f}%")

    # 4. Concentration HHI
    hhi_sym, n_eff_sym, sym_groups = hhi_concentration(weights, survivors, "symbol")
    hhi_strat, n_eff_strat, strat_groups = hhi_concentration(weights, survivors, "strategy")
    audit["concentration"] = {
        "hhi_symbol": float(hhi_sym),
        "n_effective_symbols": float(n_eff_sym),
        "hhi_strategy": float(hhi_strat),
        "n_effective_strategies": float(n_eff_strat),
        "symbol_allocation": {k: float(v) for k, v in sym_groups.items()},
        "strategy_allocation": {k: float(v) for k, v in strat_groups.items()},
    }
    logger.info(f"  Concentration: HHI_sym={hhi_sym:.3f} (N_eff={n_eff_sym:.1f}), "
               f"HHI_strat={hhi_strat:.3f} (N_eff={n_eff_strat:.1f})")

    # 5. Stress tests
    max_ls = max_losing_streak(returns)
    rec_time = recovery_time(equity)
    audit["stress"] = {
        "max_losing_streak": int(max_ls),
        "recovery_from_max_dd": int(rec_time),
        "var_95": float(np.percentile(returns, 5)),
        "cvar_95": float(np.mean(returns[returns <= np.percentile(returns, 5)])) if len(returns[returns <= np.percentile(returns, 5)]) > 0 else 0,
        "skewness": float(pd.Series(returns).skew()),
        "kurtosis": float(pd.Series(returns).kurtosis()),
    }
    logger.info(f"  Stress: max_losing_streak={max_ls}, recovery={rec_time} bars, "
               f"VaR95={audit['stress']['var_95']*100:.2f}%, "
               f"CVaR95={audit['stress']['cvar_95']*100:.2f}%")

    # 6. Correlation within portfolio
    if len(survivors) > 1:
        corr_mat, corr_keys = correlation_matrix(survivors, returns_dict)
        avg_corr = (np.sum(np.abs(corr_mat)) - len(corr_keys)) / max(len(corr_keys) * (len(corr_keys) - 1), 1)
        max_corr_val = np.max(np.abs(corr_mat - np.eye(len(corr_keys))))
        audit["correlation"] = {
            "avg_abs_correlation": float(avg_corr),
            "max_correlation": float(max_corr_val),
        }
        logger.info(f"  Correlation: avg_abs={avg_corr:.3f}, max={max_corr_val:.3f}")

    # 7. V5b features usage
    active = [s for s, w in zip(survivors, weights) if w > 0.01]
    audit["v5b_features"] = {
        "trailing_stop": sum(1 for s in active if s.get("trailing_atr_mult", 0) > 0.01),
        "breakeven": sum(1 for s in active if s.get("breakeven_trigger_pct", 0) > 0.001),
        "max_holding": sum(1 for s in active if s.get("max_holding_bars", 0) > 0),
        "risk_sizing": sum(1 for s in active if s.get("risk_key", "flat") != "flat"),
        "overlay": sum(1 for s in active if s.get("overlay", False)),
        "total_combos": len(active),
    }

    return audit


def compute_confidence_score(metrics, audit, mc):
    """Compute live deployment confidence score (0-100)."""
    score = 0
    checks = []

    # 1. Sharpe > 1.0 (15 pts)
    if metrics["sharpe"] >= 1.5:
        score += 15; checks.append(("Sharpe ≥ 1.5", 15, "✅"))
    elif metrics["sharpe"] >= 1.0:
        score += 10; checks.append(("Sharpe ≥ 1.0", 10, "⚠️"))
    elif metrics["sharpe"] >= 0.5:
        score += 5; checks.append(("Sharpe ≥ 0.5", 5, "⚠️"))
    else:
        checks.append(("Sharpe < 0.5", 0, "❌"))

    # 2. Sortino > 1.0 (10 pts)
    if metrics["sortino"] >= 1.5:
        score += 10; checks.append(("Sortino ≥ 1.5", 10, "✅"))
    elif metrics["sortino"] >= 1.0:
        score += 7; checks.append(("Sortino ≥ 1.0", 7, "⚠️"))
    else:
        checks.append(("Sortino < 1.0", 0, "❌"))

    # 3. Max DD within target (15 pts)
    if metrics["max_drawdown"] >= -0.05:
        score += 15; checks.append(("DD < -5%", 15, "✅"))
    elif metrics["max_drawdown"] >= -0.10:
        score += 10; checks.append(("DD < -10%", 10, "⚠️"))
    elif metrics["max_drawdown"] >= -0.15:
        score += 5; checks.append(("DD < -15%", 5, "⚠️"))
    else:
        checks.append(("DD > -15%", 0, "❌"))

    # 4. Rolling Sharpe stability (10 pts)
    rs = audit.get("rolling_sharpe", {})
    if rs.get("pct_positive", 0) >= 0.7:
        score += 10; checks.append(("Rolling Sharpe >0 ≥ 70%", 10, "✅"))
    elif rs.get("pct_positive", 0) >= 0.5:
        score += 5; checks.append(("Rolling Sharpe >0 ≥ 50%", 5, "⚠️"))
    else:
        checks.append(("Rolling Sharpe >0 < 50%", 0, "❌"))

    # 5. Temporal stability (10 pts)
    h1 = rs.get("first_half_mean", 0)
    h2 = rs.get("second_half_mean", 0)
    if h1 > 0 and h2 > 0:
        score += 10; checks.append(("Both halves Sharpe > 0", 10, "✅"))
    elif h2 > 0:
        score += 5; checks.append(("2nd half Sharpe > 0", 5, "⚠️"))
    else:
        checks.append(("2nd half Sharpe ≤ 0", 0, "❌"))

    # 6. Monthly positive rate (10 pts)
    mp = audit.get("monthly", {}).get("pct_positive", 0)
    if mp >= 0.6:
        score += 10; checks.append(("Mois positifs ≥ 60%", 10, "✅"))
    elif mp >= 0.45:
        score += 5; checks.append(("Mois positifs ≥ 45%", 5, "⚠️"))
    else:
        checks.append(("Mois positifs < 45%", 0, "❌"))

    # 7. Diversification (10 pts)
    n_eff = audit.get("concentration", {}).get("n_effective_symbols", 0)
    if n_eff >= 2.5:
        score += 10; checks.append(("N_eff symbols ≥ 2.5", 10, "✅"))
    elif n_eff >= 1.5:
        score += 5; checks.append(("N_eff symbols ≥ 1.5", 5, "⚠️"))
    else:
        checks.append(("N_eff symbols < 1.5", 0, "❌"))

    # 8. MC probability of gain 12M (10 pts)
    mc12 = mc.get(12, {})
    if mc12.get("prob_pos", 0) >= 0.90:
        score += 10; checks.append(("MC P(gain 12M) ≥ 90%", 10, "✅"))
    elif mc12.get("prob_pos", 0) >= 0.75:
        score += 7; checks.append(("MC P(gain 12M) ≥ 75%", 7, "⚠️"))
    else:
        checks.append(("MC P(gain 12M) < 75%", 0, "❌"))

    # 9. No ruin risk (5 pts)
    if mc12.get("prob_ruin", 1) <= 0.01:
        score += 5; checks.append(("MC P(ruine) ≤ 1%", 5, "✅"))
    else:
        checks.append(("MC P(ruine) > 1%", 0, "❌"))

    # 10. Multi-seed robustness (5 pts)
    # All combos passed multi-seed by definition (STRONG verdict)
    score += 5; checks.append(("Multi-seed 3 validé (STRONG)", 5, "✅"))

    # Verdict
    if score >= 80:
        verdict = "GO ✅"
    elif score >= 60:
        verdict = "GO PRUDENT ⚠️"
    elif score >= 40:
        verdict = "ATTENDRE 🔶"
    else:
        verdict = "NO-GO ❌"

    return score, verdict, checks


# ═══════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════

def monte_carlo(returns, n_sims=N_MONTE_CARLO):
    n = len(returns)
    block_size = min(20, n // 5)
    results = {}
    for months in [3, 6, 12, 24, 36]:
        bars = min(months * 30, n * 3)
        sims_final = np.zeros(n_sims)
        sims_dd = np.zeros(n_sims)
        for s in range(n_sims):
            sim_rets = []
            while len(sim_rets) < bars:
                start = np.random.randint(0, max(1, n - block_size))
                sim_rets.extend(returns[start:start + block_size].tolist())
            sim_rets = np.array(sim_rets[:bars])
            eq = INITIAL_CAPITAL * np.cumprod(1 + sim_rets)
            sims_final[s] = eq[-1]
            peak = np.maximum.accumulate(eq)
            dd = np.min((eq - peak) / peak)
            sims_dd[s] = dd

        results[months] = {
            "p5": float(np.percentile(sims_final, 5)),
            "p25": float(np.percentile(sims_final, 25)),
            "median": float(np.percentile(sims_final, 50)),
            "p75": float(np.percentile(sims_final, 75)),
            "p95": float(np.percentile(sims_final, 95)),
            "prob_pos": float(np.mean(sims_final > INITIAL_CAPITAL)),
            "prob_10pct": float(np.mean(sims_final > INITIAL_CAPITAL * 1.10)),
            "prob_20pct": float(np.mean(sims_final > INITIAL_CAPITAL * 1.20)),
            "prob_ruin": float(np.mean(sims_final < INITIAL_CAPITAL * 0.50)),
            "dd_median": float(np.median(sims_dd)),
            "dd_p5": float(np.percentile(sims_dd, 5)),
        }
    return results


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("PORTFOLIO V5b FINAL — Same Combos, Different Sizing")
    logger.info("=" * 70)
    logger.info(f"Diagnostic: {DIAG_PATH}")

    # ══════════════════════════════════════════════════════
    # STEP 1: Load survivors & prepare signals (once)
    # ══════════════════════════════════════════════════════
    all_survivors = load_survivors(min_sharpe=0.0)
    valid, combo_data = prepare_all(all_survivors)

    # Build default returns dict for dedup & weight optimization
    default_returns = {s["_key"]: s["_returns_default"] for s in valid}

    # ══════════════════════════════════════════════════════
    # STEP 2: Correlation dedup (once)
    # ══════════════════════════════════════════════════════
    deduped = deduplicate_by_correlation(valid, default_returns)

    # ══════════════════════════════════════════════════════
    # STEP 3: Select top N combos & optimize weights (once)
    # ══════════════════════════════════════════════════════
    N_COMBOS = min(8, len(deduped))
    selected = deduped[:N_COMBOS]

    logger.info(f"\nSelected {len(selected)} combos for ALL profiles:")
    for i, s in enumerate(selected):
        tag = f"r={s['risk_key']}" + (" +ov" if s["overlay"] else "")
        logger.info(f"  [{i+1}] {s['_key']} ({tag}) Sharpe={s['ho_sharpe']:.3f}")

    # Optimize weights using default returns (same for all profiles)
    r_sel = {s["_key"]: default_returns[s["_key"]] for s in selected}
    weights = optimize_weights(selected, r_sel, "sharpe")

    logger.info(f"\nOptimized weights (Markowitz Sharpe):")
    for s, w in sorted(zip(selected, weights), key=lambda x: -x[1]):
        if w > 0.01:
            logger.info(f"  {w*100:.1f}% {s['_key']}")

    # Symbol allocation (same for all profiles)
    sym_alloc = {}
    for s, w in zip(selected, weights):
        sym_alloc[s["symbol"]] = sym_alloc.get(s["symbol"], 0) + w

    # ══════════════════════════════════════════════════════
    # STEP 4: Backtest each profile with different RiskConfig
    # ══════════════════════════════════════════════════════
    all_profiles = {}

    for profile_key, cfg in PROFILES.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"PROFILE: {cfg['label']} — {cfg['description']}")
        logger.info(f"  risk_per_trade={cfg['risk_per_trade_pct']*100:.1f}%, "
                    f"max_position={cfg['max_position_pct']*100:.0f}%, "
                    f"circuit_breaker={cfg['max_drawdown_pct']*100:.0f}%")
        logger.info(f"{'='*70}")

        # Backtest all combos with this profile's RiskConfig
        eq, rets, returns_per_combo = backtest_profile(
            selected, combo_data, weights, cfg
        )
        m = compute_all_metrics(eq, "1d")

        logger.info(f"  Result: Ret={m['total_return']*100:.1f}% "
                    f"Sharpe={m['sharpe']:.2f} DD={m['max_drawdown']*100:.1f}% "
                    f"Calmar={m['calmar']:.2f}")

        # Audit
        audit_result = audit_portfolio(
            cfg['label'], eq, rets, weights, selected, returns_per_combo
        )

        # Monte Carlo
        logger.info(f"\n  Monte Carlo (5000 sims)...")
        mc = monte_carlo(rets)
        for months, r in mc.items():
            logger.info(f"    {months}M: median=${r['median']:,.0f} "
                       f"P(>0)={r['prob_pos']*100:.0f}% DD_med={r['dd_median']*100:.1f}%")

        # Confidence score
        conf_score, conf_verdict, conf_checks = compute_confidence_score(m, audit_result, mc)
        logger.info(f"\n  CONFIDENCE: {conf_score}/100 — {conf_verdict}")
        for check_name, pts, status in conf_checks:
            logger.info(f"    {status} {check_name}: {pts} pts")

        all_profiles[profile_key] = {
            "config": cfg,
            "survivors": selected,
            "weights": weights,
            "equity": eq,
            "returns": rets,
            "metrics": m,
            "audit": audit_result,
            "monte_carlo": mc,
            "confidence_score": conf_score,
            "confidence_verdict": conf_verdict,
            "confidence_checks": conf_checks,
            "sym_alloc": sym_alloc,
        }

    # ══════════════════════════════════════════════════════
    # STEP 5: Summary
    # ══════════════════════════════════════════════════════
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY — Same Combos, Different Position Sizing")
    logger.info(f"{'='*70}")
    fmt = "{:<15} {:>10} {:>10} {:>8} {:>8} {:>10} {:>8} {:>8} {:>12}"
    logger.info(fmt.format("Profile", "Risk/Tr", "MaxPos", "Sharpe", "Sortino",
                           "Return", "Max DD", "Calmar", "Confidence"))
    logger.info("-" * 110)
    for pk, p in all_profiles.items():
        m = p["metrics"]
        c = p["config"]
        logger.info(fmt.format(
            c["label"],
            f"{c['risk_per_trade_pct']*100:.1f}%",
            f"{c['max_position_pct']*100:.0f}%",
            f"{m['sharpe']:.2f}",
            f"{m['sortino']:.2f}",
            f"{m['total_return']*100:.1f}%",
            f"{m['max_drawdown']*100:.1f}%",
            f"{m['calmar']:.2f}",
            f"{p['confidence_score']}/100 {p['confidence_verdict']}",
        ))

    # ══════════════════════════════════════════════════════
    # STEP 6: Save JSON
    # ══════════════════════════════════════════════════════
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {"generated": ts, "diagnostic": str(DIAG_PATH)}

    for pk, p in all_profiles.items():
        save_data[pk] = {
            "config": p["config"],
            "metrics": p["metrics"],
            "audit": p["audit"],
            "monte_carlo": p["monte_carlo"],
            "confidence_score": p["confidence_score"],
            "confidence_verdict": p["confidence_verdict"],
            "confidence_checks": [(c[0], c[1], c[2]) for c in p["confidence_checks"]],
            "sym_alloc": p["sym_alloc"],
            "weights": p["weights"].tolist() if isinstance(p["weights"], np.ndarray) else p["weights"],
            "allocations": [
                {
                    "weight": float(w), "symbol": s["symbol"],
                    "strategy": s["strategy"], "timeframe": s["timeframe"],
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

    json_path = Path(f"portfolio/v5b/results/portfolio_v5b_final_{ts}.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    logger.info(f"\nJSON: {json_path}")

    # ── Generate Report ──
    elapsed = (time.time() - t0) / 60
    report_path = generate_full_report(all_profiles, elapsed)
    logger.info(f"Report: {report_path}")

    logger.info(f"\n{'='*70}")
    logger.info(f"PORTFOLIO V5b FINAL COMPLETE ({elapsed:.1f} min)")
    logger.info(f"{'='*70}")

    return all_profiles


# ═══════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════

def generate_full_report(all_profiles, elapsed):
    """Generate comprehensive markdown report."""
    L = []
    L.append("# Portfolio V5b — Multi-Profil + Audit + Confiance Live")
    L.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    L.append(f"**Diagnostic** : {DIAG_PATH.name}")
    L.append(f"**Durée** : {elapsed:.1f} min")
    L.append("")
    L.append("---")
    L.append("")

    # ── Overview table ──
    L.append("## Vue d'ensemble — 3 Profils de Risque")
    L.append("")
    L.append("| Profil | Risk/Trade | Max Pos | Sharpe | Sortino | Return | Max DD | Calmar | Confiance |")
    L.append("|--------|-----------|---------|--------|---------|--------|--------|--------|-----------|")
    for pk, p in all_profiles.items():
        m = p["metrics"]
        c = p["config"]
        L.append(f"| **{c['label']}** | {c['risk_per_trade_pct']*100:.1f}% | {c['max_position_pct']*100:.0f}% "
                f"| {m['sharpe']:.2f} | {m['sortino']:.2f} | {m['total_return']*100:.1f}% "
                f"| {m['max_drawdown']*100:.1f}% | {m['calmar']:.2f} "
                f"| **{p['confidence_score']}/100** {p['confidence_verdict']} |")
    L.append("")

    # ── Each profile detail ──
    for pk, p in all_profiles.items():
        cfg = p["config"]
        m = p["metrics"]
        audit = p["audit"]
        mc = p["monte_carlo"]

        L.append(f"---")
        L.append(f"")
        L.append(f"## Profil {cfg['label']}")
        L.append(f"*{cfg['description']}*")
        L.append(f"")

        # Metrics
        L.append(f"### Performance")
        L.append(f"")
        L.append(f"| Métrique | Valeur | Objectif | Status |")
        L.append(f"|----------|--------|----------|--------|")
        L.append(f"| Return | {m['total_return']*100:.1f}% | ≥15% | {'✅' if m['total_return']>=0.15 else '⚠️' if m['total_return']>=0.10 else '❌'} |")
        L.append(f"| Sharpe | {m['sharpe']:.2f} | ≥1.0 | {'✅' if m['sharpe']>=1.0 else '❌'} |")
        L.append(f"| Sortino | {m['sortino']:.2f} | ≥1.0 | {'✅' if m['sortino']>=1.0 else '❌'} |")
        L.append(f"| Max DD | {m['max_drawdown']*100:.1f}% | ≥{cfg['max_dd_target']*100:.0f}% | {'✅' if m['max_drawdown']>=cfg['max_dd_target'] else '❌'} |")
        L.append(f"| Calmar | {m['calmar']:.2f} | ≥1.0 | {'✅' if m['calmar']>=1.0 else '❌'} |")
        L.append(f"| Risk/Trade | {cfg['risk_per_trade_pct']*100:.1f}% | — | — |")
        L.append(f"| Max Position | {cfg['max_position_pct']*100:.0f}% | — | — |")
        L.append(f"| Circuit Breaker | {cfg['max_drawdown_pct']*100:.0f}% | — | — |")
        L.append(f"")

        # Allocations
        L.append(f"### Allocations ({len([w for w in p['weights'] if w > 0.01])} combos)")
        L.append(f"")
        L.append(f"| Poids | Symbol | Stratégie | TF | Risk | Ov | Sharpe | Return | DD | Seed_std |")
        L.append(f"|-------|--------|-----------|-----|------|-----|--------|--------|-----|----------|")
        for s, w in sorted(zip(p["survivors"], p["weights"]), key=lambda x: -x[1]):
            if w > 0.01:
                ov = "Y" if s["overlay"] else "-"
                L.append(f"| {w*100:.1f}% | {s['symbol']} | {s['strategy']} | {s['timeframe']} "
                        f"| {s['risk_key']} | {ov} | {s['ho_sharpe']:.3f} "
                        f"| {s['ho_return']*100:.1f}% | {s['ho_dd']*100:.1f}% "
                        f"| {s.get('seed_sharpe_std',0):.3f} |")
        L.append(f"")

        # Symbol allocation
        L.append(f"### Répartition par actif")
        L.append(f"")
        L.append(f"| Actif | Allocation |")
        L.append(f"|-------|-----------|")
        for sym, alloc in sorted(p["sym_alloc"].items(), key=lambda x: -x[1]):
            L.append(f"| {sym} | {alloc*100:.1f}% |")
        L.append(f"")

        # Audit
        L.append(f"### Audit de fiabilité")
        L.append(f"")

        # Rolling Sharpe
        rs = audit.get("rolling_sharpe", {})
        L.append(f"**Rolling Sharpe (60 jours)**")
        L.append(f"")
        L.append(f"| Métrique | Valeur |")
        L.append(f"|----------|--------|")
        L.append(f"| Moyenne | {rs.get('mean',0):.2f} |")
        L.append(f"| Écart-type | {rs.get('std',0):.2f} |")
        L.append(f"| Min / Max | {rs.get('min',0):.2f} / {rs.get('max',0):.2f} |")
        L.append(f"| % positif | {rs.get('pct_positive',0)*100:.0f}% |")
        L.append(f"| 1ère moitié | {rs.get('first_half_mean',0):.2f} |")
        L.append(f"| 2ème moitié | {rs.get('second_half_mean',0):.2f} |")
        L.append(f"")

        # Monthly
        mo = audit.get("monthly", {})
        L.append(f"**Analyse mensuelle**")
        L.append(f"")
        L.append(f"| Métrique | Valeur |")
        L.append(f"|----------|--------|")
        L.append(f"| Pire mois | {mo.get('worst_month',0)*100:.1f}% |")
        L.append(f"| Meilleur mois | {mo.get('best_month',0)*100:.1f}% |")
        L.append(f"| Mois moyen | {mo.get('avg_month',0)*100:.2f}% |")
        L.append(f"| Mois positifs | {mo.get('pct_positive',0)*100:.0f}% |")
        L.append(f"")

        # Stress
        st = audit.get("stress", {})
        L.append(f"**Stress tests**")
        L.append(f"")
        L.append(f"| Métrique | Valeur |")
        L.append(f"|----------|--------|")
        L.append(f"| Max losing streak | {st.get('max_losing_streak',0)} bars |")
        L.append(f"| Recovery from max DD | {st.get('recovery_from_max_dd',0)} bars |")
        L.append(f"| VaR 95% | {st.get('var_95',0)*100:.2f}% |")
        L.append(f"| CVaR 95% | {st.get('cvar_95',0)*100:.2f}% |")
        L.append(f"| Skewness | {st.get('skewness',0):.2f} |")
        L.append(f"| Kurtosis | {st.get('kurtosis',0):.2f} |")
        L.append(f"")

        # Concentration
        co = audit.get("concentration", {})
        L.append(f"**Concentration**")
        L.append(f"")
        L.append(f"| Métrique | Valeur |")
        L.append(f"|----------|--------|")
        L.append(f"| HHI Symbol | {co.get('hhi_symbol',0):.3f} |")
        L.append(f"| N effectif symbols | {co.get('n_effective_symbols',0):.1f} |")
        L.append(f"| HHI Stratégie | {co.get('hhi_strategy',0):.3f} |")
        L.append(f"| N effectif stratégies | {co.get('n_effective_strategies',0):.1f} |")
        L.append(f"")

        # Correlation
        cr = audit.get("correlation", {})
        if cr:
            L.append(f"**Corrélation intra-portfolio**")
            L.append(f"")
            L.append(f"| Métrique | Valeur |")
            L.append(f"|----------|--------|")
            L.append(f"| Corrélation abs moyenne | {cr.get('avg_abs_correlation',0):.3f} |")
            L.append(f"| Corrélation max | {cr.get('max_correlation',0):.3f} |")
            L.append(f"")

        # Monte Carlo
        L.append(f"### Monte Carlo ($10,000 — 5000 sims)")
        L.append(f"")
        L.append(f"| Horizon | P5 | Médian | P95 | P(>0) | P(>10%) | P(ruine) | DD médian |")
        L.append(f"|---------|-----|--------|-----|-------|---------|----------|-----------|")
        for months, r in mc.items():
            L.append(f"| {months}M | ${r['p5']:,.0f} | ${r['median']:,.0f} | ${r['p95']:,.0f} "
                    f"| {r['prob_pos']*100:.0f}% | {r['prob_10pct']*100:.0f}% "
                    f"| {r['prob_ruin']*100:.1f}% | {r['dd_median']*100:.1f}% |")
        L.append(f"")

        # Confidence
        L.append(f"### Score de confiance déploiement live")
        L.append(f"")
        L.append(f"**Score : {p['confidence_score']}/100 — {p['confidence_verdict']}**")
        L.append(f"")
        L.append(f"| Critère | Points | Status |")
        L.append(f"|---------|--------|--------|")
        for check_name, pts, status in p["confidence_checks"]:
            L.append(f"| {check_name} | {pts}/{'15' if 'Sharpe' in check_name and 'Rolling' not in check_name and 'DD' not in check_name else '10' if any(x in check_name for x in ['Sortino','Rolling','stab','Mois','N_eff','MC P(gain']) else '15' if 'DD' in check_name else '5'} | {status} |")
        L.append(f"")

        # V5b features
        v5b = audit.get("v5b_features", {})
        L.append(f"### Features V5b")
        L.append(f"")
        L.append(f"| Feature | Utilisation |")
        L.append(f"|---------|------------|")
        L.append(f"| Trailing stop | {v5b.get('trailing_stop',0)}/{v5b.get('total_combos',0)} |")
        L.append(f"| Breakeven | {v5b.get('breakeven',0)}/{v5b.get('total_combos',0)} |")
        L.append(f"| Max holding | {v5b.get('max_holding',0)}/{v5b.get('total_combos',0)} |")
        L.append(f"| Risk sizing | {v5b.get('risk_sizing',0)}/{v5b.get('total_combos',0)} |")
        L.append(f"| Overlay | {v5b.get('overlay',0)}/{v5b.get('total_combos',0)} |")
        L.append(f"")

    # ── Historical comparison ──
    L.append(f"---")
    L.append(f"")
    L.append(f"## Comparaison historique")
    L.append(f"")
    L.append(f"| Métrique | V3b | V4 | V4b | V5b Conserv. | V5b Modéré | V5b Agressif |")
    L.append(f"|----------|-----|-----|-----|-------------|------------|--------------|")

    def _fmt(m, key, mult=1, pct=False, prefix=""):
        if m is None:
            return "—"
        v = m[key] * mult
        if pct:
            return f"{prefix}{v:.1f}%"
        return f"{prefix}{v:.2f}"

    mc = all_profiles.get("conservateur", {}).get("metrics")
    mm = all_profiles.get("modere", {}).get("metrics")
    ma = all_profiles.get("agressif", {}).get("metrics")

    L.append(f"| Return | +9.8% | +4.9% | +19.8% | "
            f"{_fmt(mc, 'total_return', 100, True, '+')} | "
            f"{_fmt(mm, 'total_return', 100, True, '+')} | "
            f"{_fmt(ma, 'total_return', 100, True, '+')} |")
    L.append(f"| Sharpe | 1.19 | 2.59 | 1.35 | "
            f"{_fmt(mc, 'sharpe')} | "
            f"{_fmt(mm, 'sharpe')} | "
            f"{_fmt(ma, 'sharpe')} |")
    L.append(f"| Max DD | -4.9% | -0.8% | -8.5% | "
            f"{_fmt(mc, 'max_drawdown', 100, True)} | "
            f"{_fmt(mm, 'max_drawdown', 100, True)} | "
            f"{_fmt(ma, 'max_drawdown', 100, True)} |")
    L.append(f"| Calmar | 1.91 | 5.99 | 2.17 | "
            f"{_fmt(mc, 'calmar')} | "
            f"{_fmt(mm, 'calmar')} | "
            f"{_fmt(ma, 'calmar')} |")
    L.append(f"")

    # ── Recommendation ──
    L.append(f"## Recommandation")
    L.append(f"")
    best_pk = max(all_profiles, key=lambda k: all_profiles[k]["confidence_score"])
    best = all_profiles[best_pk]
    L.append(f"Le profil **{best['config']['label']}** obtient le meilleur score de confiance "
            f"(**{best['confidence_score']}/100**) et est recommandé pour le déploiement live.")
    L.append(f"")
    if best["confidence_score"] >= 80:
        L.append(f"> **GO pour déploiement live** — Tous les critères majeurs sont satisfaits.")
    elif best["confidence_score"] >= 60:
        L.append(f"> **GO PRUDENT** — Déployer avec capital réduit et monitoring renforcé.")
    else:
        L.append(f"> **ATTENDRE** — Des critères importants ne sont pas satisfaits.")
    L.append(f"")

    L.append(f"---")
    L.append(f"*Généré le {datetime.now().strftime('%d %B %Y %H:%M')}*")

    report_path = Path("portfolio/v5b/results/portfolio_v5b_final_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L))
    return report_path


if __name__ == "__main__":
    main()
