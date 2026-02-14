#!/usr/bin/env python3
"""
Portfolio FTMO V1 — Optimisé pour passer le FTMO 2-Step Challenge (Swing mode).

Objectifs :
  1. Passer Phase 1 (10% profit target, DD < 10%, daily loss < 5%)
  2. Passer Phase 2 (5% profit target, mêmes limites)
  3. Tourner en funded de manière durable

Profils :
  - CHALLENGE : Phase 1+2, sizing modéré, DD strict (8% CB)
  - FUNDED    : FTMO Account, sizing conservateur, DD très strict (7% CB)

Différences clés vs V5b :
  - DD limits beaucoup plus serrés (8% vs 15%)
  - Daily loss limit enforced (4% vs 3% soft)
  - Timeframes 4h/1d uniquement (swing compatible)
  - Overlays obligatoires (regime + vol targeting 25%)
  - Emergency shields multi-couche
  - Monte Carlo calibré sur probabilité de passer le challenge
"""

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from engine.backtester import RiskConfig, vectorized_backtest
from engine.metrics import compute_all_metrics, returns_from_equity
from engine.overlays import (
    apply_overlay_pipeline, OverlayPipelineConfig,
    VolTargetConfig, RegimeOverlayConfig,
)
from engine.regime import RegimeConfig
from strategies.registry import get_strategy

# ═══════════════════════════════════════════════════════
# FTMO-SPECIFIC CONFIGURATION
# ═══════════════════════════════════════════════════════

INITIAL_CAPITAL = 10_000.0
N_MONTE_CARLO = 5000
CUTOFF_DATE = "2025-02-01"

# FTMO hard limits (with safety margins)
FTMO_MAX_DAILY_LOSS = 0.05       # 5% FTMO rule
FTMO_MAX_TOTAL_LOSS = 0.10       # 10% FTMO rule
FTMO_PHASE1_TARGET = 0.10        # 10% profit target
FTMO_PHASE2_TARGET = 0.05        # 5% profit target

# Portfolio constraints (stricter than V5b)
MAX_WEIGHT_PER_SYMBOL = 0.50
MAX_WEIGHT_PER_COMBO = 0.25
MIN_WEIGHT = 0.03
MAX_CORRELATION = 0.70
MIN_HO_TRADES = 5
MAX_SEED_STD = 0.5
MIN_HO_SHARPE = 0.5
MAX_HO_DD = -0.06               # Stricter: max -6% DD per combo
ALLOWED_TIMEFRAMES = {"4h", "1d"}
DEFAULT_WEIGHT_OBJECTIVE = "max_return"

# Overlay config — more conservative than V5b
REGIME_CFG = RegimeOverlayConfig(
    regime_config=RegimeConfig(),
    hard_cutoff=True,
    min_exposure_threshold=0.3,
)
VOL_CFG = VolTargetConfig(target_vol_annual=0.25)  # 25% vs 40% in V5b
OVERLAY_CFG = OverlayPipelineConfig(regime_config=REGIME_CFG, vol_config=VOL_CFG)

# Risk profiles
PROFILES = {
    "challenge_aggressive": {
        "label": "Challenge Aggressive",
        "description": "Phase 1 orientée passage rapide, garde-fous FTMO conservés",
        "risk_per_trade_pct": 0.0125,    # 1.25% risk per trade
        "max_position_pct": 0.20,        # 20% max per position
        "max_drawdown_pct": 0.085,       # 8.5% CB (marge vs 10%)
        "max_daily_loss_pct": 0.045,     # 4.5% daily loss (marge vs 5%)
        "max_trades_per_day": 6,
        "cooldown_after_loss": 1,
        # Emergency shields
        "dd_warning_pct": 0.06,
        "dd_reduce_pct": 0.06,
        "dd_emergency_pct": 0.08,
        "profit_lock_pct": 0.10,
    },
    "challenge": {
        "label": "Challenge",
        "description": "Phase 1+2 — sizing modéré, DD strict (8% CB)",
        "risk_per_trade_pct": 0.01,      # 1% risk per trade
        "max_position_pct": 0.15,        # 15% max per position
        "max_drawdown_pct": 0.08,        # 8% circuit breaker (2% marge vs 10%)
        "max_daily_loss_pct": 0.04,      # 4% daily loss (1% marge vs 5%)
        "max_trades_per_day": 5,
        "cooldown_after_loss": 2,
        # Emergency shields
        "dd_warning_pct": 0.06,
        "dd_reduce_pct": 0.06,
        "dd_emergency_pct": 0.075,
        "profit_lock_pct": 0.08,
    },
    "funded": {
        "label": "Funded",
        "description": "FTMO Account — sizing conservateur, DD très strict (7% CB)",
        "risk_per_trade_pct": 0.0075,    # 0.75% risk per trade
        "max_position_pct": 0.10,        # 10% max per position
        "max_drawdown_pct": 0.07,        # 7% circuit breaker (3% marge)
        "max_daily_loss_pct": 0.035,     # 3.5% daily loss (1.5% marge)
        "max_trades_per_day": 4,
        "cooldown_after_loss": 3,
        # Emergency shields
        "dd_warning_pct": 0.05,
        "dd_reduce_pct": 0.05,
        "dd_emergency_pct": 0.065,
        "profit_lock_pct": None,
    },
}

# Diagnostic path — find latest
DIAG_PATTERNS = ["diagnostic_v5b_*.json", "diagnostic_v5_*.json"]
DIAG_SEARCH_DIRS = [
    Path("portfolio/v5b/results"),
    Path("portfolio/ftmo-v1/results"),
    Path("results"),  # legacy fallback
]


# ═══════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════

def find_diagnostic():
    """Find the latest diagnostic file."""
    for pattern in DIAG_PATTERNS:
        for base in DIAG_SEARCH_DIRS:
            files = sorted(base.glob(pattern))
            if files:
                return files[-1]

    logger.error(
        "No diagnostic file found in portfolio/v5b/results, portfolio/ftmo-v1/results, or results. "
        "Run diagnostic_v5b.py first."
    )
    sys.exit(1)


def load_survivors(diag_path):
    """Load STRONG survivors filtered for FTMO constraints."""
    with open(diag_path) as f:
        data = json.load(f)
    p2 = data["phase2"]

    # FTMO-specific filtering
    candidates = []
    rejected = {"sharpe": 0, "trades": 0, "seed_std": 0, "dd": 0, "timeframe": 0}

    for r in p2:
        if r["verdict"] != "STRONG":
            continue

        # Timeframe filter (4h/1d only for swing)
        if r["timeframe"] not in ALLOWED_TIMEFRAMES:
            rejected["timeframe"] += 1
            continue

        if r["ho_sharpe"] < MIN_HO_SHARPE:
            rejected["sharpe"] += 1
            continue

        if r["ho_trades"] < MIN_HO_TRADES:
            rejected["trades"] += 1
            continue

        if r.get("seed_sharpe_std", 0) > MAX_SEED_STD:
            rejected["seed_std"] += 1
            continue

        if r["ho_dd"] < MAX_HO_DD:
            rejected["dd"] += 1
            continue

        candidates.append(r)

    logger.info(f"FTMO filtering: {len(p2)} phase2 → {len(candidates)} candidates")
    logger.info(f"  Rejected: {rejected}")

    # Best risk_key per (sym, strat, tf)
    best = {}
    for r in candidates:
        key = (r["symbol"], r["strategy"], r["timeframe"])
        if key not in best or r["ho_sharpe"] > best[key]["ho_sharpe"]:
            best[key] = r

    return sorted(best.values(), key=lambda x: x["ho_sharpe"], reverse=True)


def prepare_combo_data(combo):
    """Prepare signals and market data for a combo."""
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

    # Always apply overlays for FTMO (mandatory)
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
    return res


def prepare_all(survivors):
    """Prepare signal data for all survivors."""
    logger.info(f"Preparing {len(survivors)} combo signals...")
    combo_data = {}
    valid = []

    default_risk = RiskConfig(
        risk_per_trade_pct=0.01,
        max_position_pct=0.15,
        max_drawdown_pct=0.08,
        max_daily_loss_pct=0.04,
        max_trades_per_day=5,
        cooldown_after_loss=2,
    )

    for i, s in enumerate(survivors):
        key = f"{s['symbol']}/{s['strategy']}/{s['timeframe']}"
        try:
            cd = prepare_combo_data(s)
            if cd is None:
                continue
            combo_data[key] = cd
            s["_key"] = key

            res = build_equity_with_risk(cd, default_risk)
            s["_returns_default"] = returns_from_equity(res.equity)
            s["_returns_series_default"] = pd.Series(
                s["_returns_default"],
                index=pd.to_datetime(cd["dates"][1:len(s["_returns_default"]) + 1]),
            )
            s["_equity_default"] = res.equity
            s["_trades_pnl"] = res.trades_pnl
            s["_n_trades"] = res.n_trades

            # Compute FTMO-specific metrics
            metrics = compute_all_metrics(res.equity, s["timeframe"], res.trades_pnl)
            s["_ftmo_sharpe"] = metrics["sharpe"]
            s["_ftmo_dd"] = metrics["max_drawdown"]
            s["_ftmo_return"] = metrics["total_return"]
            s["_ftmo_win_rate"] = metrics["win_rate"]
            s["_ftmo_pf"] = metrics["profit_factor"]

            # Compute max daily loss on calendar days (safer for mixed TF handling)
            daily_ret_series = s["_returns_series_default"].groupby(s["_returns_series_default"].index.floor("D")).sum()
            s["_max_daily_loss"] = float(daily_ret_series.min()) if len(daily_ret_series) else 0.0

            # Max losing streak
            streak = 0
            max_streak = 0
            for r in s["_returns_default"]:
                if r < 0:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            s["_max_losing_streak"] = max_streak

            valid.append(s)

            tag = f"r={s['risk_key']}" + (" +ov" if s["overlay"] else "")
            logger.info(f"  [{i+1}] {key} ({tag}) "
                       f"Sharpe={s['_ftmo_sharpe']:.3f} DD={s['_ftmo_dd']*100:.1f}% "
                       f"WR={s['_ftmo_win_rate']*100:.0f}% PF={s['_ftmo_pf']:.2f} "
                       f"DailyMax={s['_max_daily_loss']*100:.2f}%")
        except Exception as e:
            logger.error(f"  [{i+1}] {key}: {e}")

    logger.info(f"Prepared {len(valid)}/{len(survivors)}")
    return valid, combo_data


# ═══════════════════════════════════════════════════════
# FTMO-SPECIFIC FILTERING
# ═══════════════════════════════════════════════════════

def ftmo_quality_filter(valid):
    """Additional FTMO quality filter on backtested results."""
    filtered = []
    for s in valid:
        # Win rate filter
        if s["_ftmo_win_rate"] < 0.35:
            logger.info(f"  SKIP {s['_key']}: win_rate={s['_ftmo_win_rate']*100:.0f}% < 35%")
            continue
        # Profit factor filter
        if s["_ftmo_pf"] < 1.2:
            logger.info(f"  SKIP {s['_key']}: PF={s['_ftmo_pf']:.2f} < 1.2")
            continue
        # Max losing streak filter
        if s["_max_losing_streak"] > 8:
            logger.info(f"  SKIP {s['_key']}: losing_streak={s['_max_losing_streak']} > 8")
            continue
        # Daily loss filter (must survive FTMO daily limit with margin)
        if s["_max_daily_loss"] < -0.035:
            logger.info(f"  SKIP {s['_key']}: daily_loss={s['_max_daily_loss']*100:.1f}% < -3.5%")
            continue
        filtered.append(s)

    logger.info(f"FTMO quality filter: {len(valid)} → {len(filtered)}")
    return filtered


# ═══════════════════════════════════════════════════════
# CORRELATION & OPTIMIZATION
# ═══════════════════════════════════════════════════════

def deduplicate_by_correlation(survivors, max_corr=MAX_CORRELATION):
    """Remove highly correlated combos, keeping the one with higher Sharpe."""
    if len(survivors) <= 1:
        return survivors
    keys = [s["_key"] for s in survivors]
    aligned = pd.concat(
        [s["_returns_series_default"].rename(s["_key"]) for s in survivors],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        logger.warning("Correlation dedup skipped: no common timestamps after alignment")
        return survivors
    rets_matrix = aligned.to_numpy()
    corr = np.corrcoef(rets_matrix.T)
    corr = np.nan_to_num(corr, nan=0.0)

    to_remove = set()
    for i in range(len(keys)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(keys)):
            if j in to_remove:
                continue
            if abs(corr[i, j]) > max_corr:
                if survivors[i]["_ftmo_sharpe"] >= survivors[j]["_ftmo_sharpe"]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)

    filtered = [s for idx, s in enumerate(survivors) if idx not in to_remove]
    logger.info(f"Correlation dedup: {len(survivors)} → {len(filtered)} (max_corr={max_corr})")
    return filtered


def ledoit_wolf(rets_matrix):
    """Ledoit-Wolf shrinkage covariance estimator."""
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


def optimize_weights_ftmo(survivors, objective="sharpe"):
    """Optimize portfolio weights with FTMO constraints."""
    n = len(survivors)
    aligned = pd.concat(
        [s["_returns_series_default"].rename(s["_key"]) for s in survivors],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        logger.warning("Weight optimization fallback: no aligned returns, using equal weights")
        return np.ones(n) / n

    rets_matrix = aligned.to_numpy()
    cov = ledoit_wolf(rets_matrix)
    mu = rets_matrix.mean(axis=0)

    def neg_obj(w):
        pr = w @ mu
        pv = np.sqrt(max(w @ cov @ w, 1e-20))
        if objective == "max_return":
            return -pr
        if objective == "sharpe":
            return -pr / pv
        elif objective == "min_dd":
            return -(pr - 1.0 * pv)  # Penalize variance more
        return -pr / pv

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Symbol concentration constraints
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
    logger.warning("Optimization failed, using equal weights")
    return np.ones(n) / n


# ═══════════════════════════════════════════════════════
# PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════

def backtest_profile(survivors, combo_data, weights, profile_cfg):
    """Backtest all combos with a specific FTMO risk profile."""
    risk_cfg = RiskConfig(
        risk_per_trade_pct=profile_cfg["risk_per_trade_pct"],
        max_position_pct=profile_cfg["max_position_pct"],
        max_drawdown_pct=profile_cfg["max_drawdown_pct"],
        max_daily_loss_pct=profile_cfg["max_daily_loss_pct"],
        max_trades_per_day=profile_cfg["max_trades_per_day"],
        cooldown_after_loss=profile_cfg["cooldown_after_loss"],
    )

    keys = [s["_key"] for s in survivors]
    returns_per_combo = {}
    returns_series_per_combo = {}
    equity_per_combo = {}
    trades_per_combo = {}

    for s in survivors:
        res = build_equity_with_risk(combo_data[s["_key"]], risk_cfg)
        combo_rets = returns_from_equity(res.equity)
        combo_idx = pd.to_datetime(combo_data[s["_key"]]["dates"][1:len(combo_rets) + 1])
        returns_per_combo[s["_key"]] = combo_rets
        returns_series_per_combo[s["_key"]] = pd.Series(combo_rets, index=combo_idx)
        equity_per_combo[s["_key"]] = res.equity
        trades_per_combo[s["_key"]] = res.trades_pnl

    # Combine into portfolio with timestamp alignment (critical for mixed TF portfolios)
    aligned = pd.concat(
        [returns_series_per_combo[k].rename(k) for k in keys],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("No aligned returns across combos for portfolio aggregation")

    weighted = np.zeros(len(aligned), dtype=np.float64)
    for i, k in enumerate(keys):
        weighted += weights[i] * aligned[k].to_numpy(dtype=np.float64)
    port_returns = weighted
    port_index = aligned.index

    aligned_returns_per_combo = {
        k: aligned[k].to_numpy(dtype=np.float64)
        for k in keys
    }

    # Proxy portfolio trade PnL for win_rate / profit_factor metrics.
    # We weight each combo's trade outcomes by its allocation then concatenate.
    # This avoids zeros at portfolio level while staying consistent with allocation logic.
    weighted_trade_pnls = []
    for i, k in enumerate(keys):
        tp = np.asarray(trades_per_combo.get(k, []), dtype=np.float64)
        if tp.size:
            weighted_trade_pnls.append(tp * float(weights[i]))
    portfolio_trades_pnl = (
        np.concatenate(weighted_trade_pnls)
        if weighted_trade_pnls
        else np.array([], dtype=np.float64)
    )

    min_len = len(port_returns)
    equity = np.zeros(min_len + 1)
    equity[0] = INITIAL_CAPITAL
    for t in range(min_len):
        equity[t + 1] = equity[t] * (1 + port_returns[t])

    return equity, port_returns, aligned_returns_per_combo, trades_per_combo, port_index, portfolio_trades_pnl


# ═══════════════════════════════════════════════════════
# FTMO-SPECIFIC ANALYSIS
# ═══════════════════════════════════════════════════════

def compute_ftmo_metrics(equity, returns, timeframe_mix="4h", returns_index=None, trades_pnl=None):
    """Compute FTMO-specific metrics beyond standard ones."""
    metrics = compute_all_metrics(equity, timeframe_mix, trades_pnl)

    # Daily P&L analysis on calendar days if index is available
    if returns_index is not None:
        rs = pd.Series(returns, index=pd.to_datetime(returns_index))
        daily_returns = rs.groupby(rs.index.floor("D")).sum().to_numpy(dtype=np.float64)
    else:
        bars_per_day = 6 if timeframe_mix == "4h" else 1
        daily_returns = []
        for d_start in range(0, len(returns), bars_per_day):
            d_end = min(d_start + bars_per_day, len(returns))
            daily_ret = np.sum(returns[d_start:d_end])
            daily_returns.append(daily_ret)
        daily_returns = np.array(daily_returns, dtype=np.float64)

    metrics["ftmo_max_daily_loss"] = float(np.min(daily_returns)) if len(daily_returns) > 0 else 0
    metrics["ftmo_daily_loss_safe"] = metrics["ftmo_max_daily_loss"] > -FTMO_MAX_DAILY_LOSS
    metrics["ftmo_total_loss_safe"] = metrics["max_drawdown"] > -FTMO_MAX_TOTAL_LOSS
    metrics["ftmo_avg_daily_return"] = float(np.mean(daily_returns)) if len(daily_returns) > 0 else 0
    metrics["ftmo_daily_std"] = float(np.std(daily_returns)) if len(daily_returns) > 0 else 0
    metrics["ftmo_pct_positive_days"] = float(np.mean(daily_returns > 0)) if len(daily_returns) > 0 else 0

    # Estimate days to reach targets
    if metrics["ftmo_avg_daily_return"] > 0:
        metrics["ftmo_est_days_phase1"] = int(np.ceil(FTMO_PHASE1_TARGET / metrics["ftmo_avg_daily_return"]))
        metrics["ftmo_est_days_phase2"] = int(np.ceil(FTMO_PHASE2_TARGET / metrics["ftmo_avg_daily_return"]))
    else:
        metrics["ftmo_est_days_phase1"] = 999
        metrics["ftmo_est_days_phase2"] = 999

    # Max losing streak
    streak = 0
    max_streak = 0
    for r in returns:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    metrics["ftmo_max_losing_streak"] = max_streak

    # Max consecutive losing days
    d_streak = 0
    max_d_streak = 0
    for d in daily_returns:
        if d < 0:
            d_streak += 1
            max_d_streak = max(max_d_streak, d_streak)
        else:
            d_streak = 0
    metrics["ftmo_max_losing_days"] = max_d_streak

    return metrics, daily_returns


def ftmo_challenge_monte_carlo(returns, n_sims=N_MONTE_CARLO):
    """Monte Carlo specifically for FTMO challenge pass probability."""
    n = len(returns)
    block_size = min(20, n // 5)
    results = {}

    for label, target, max_dd in [
        ("phase1", FTMO_PHASE1_TARGET, FTMO_MAX_TOTAL_LOSS),
        ("phase2", FTMO_PHASE2_TARGET, FTMO_MAX_TOTAL_LOSS),
        ("funded_monthly", 0.03, FTMO_MAX_TOTAL_LOSS),
    ]:
        # Simulate up to 6 months (180 days)
        max_bars = min(180 * 6, n * 5)  # 6 months at 4h = ~1080 bars

        passed = 0
        failed_dd = 0
        failed_daily = 0
        days_to_pass = []
        final_equity = []
        max_dds = []

        for s in range(n_sims):
            sim_rets = []
            while len(sim_rets) < max_bars:
                start = np.random.randint(0, max(1, n - block_size))
                sim_rets.extend(returns[start:start + block_size].tolist())
            sim_rets = np.array(sim_rets[:max_bars])

            eq = INITIAL_CAPITAL * np.cumprod(1 + sim_rets)
            peak = np.maximum.accumulate(eq)
            dd = (eq - peak) / peak

            # Approximate daily loss check (4h bars => 6 bars/day)
            daily_rets = [np.sum(sim_rets[i:i + 6]) for i in range(0, len(sim_rets), 6)]
            if any(dr < -FTMO_MAX_DAILY_LOSS for dr in daily_rets):
                failed_daily += 1
                final_equity.append(eq[-1])
                max_dds.append(float(np.min(dd)))
                continue

            # Check if passed target before hitting DD limit
            target_eq = INITIAL_CAPITAL * (1 + target)
            dd_limit = -max_dd

            sim_passed = False
            sim_failed = False
            for bar in range(len(eq)):
                if dd[bar] < dd_limit:
                    sim_failed = True
                    break
                if eq[bar] >= target_eq:
                    sim_passed = True
                    days_to_pass.append(bar / 6)  # Approximate days (4h bars)
                    break

            if sim_passed:
                passed += 1
            elif sim_failed:
                failed_dd += 1

            final_equity.append(eq[-1])
            max_dds.append(float(np.min(dd)))

        final_equity = np.array(final_equity)
        max_dds = np.array(max_dds)

        results[label] = {
            "pass_rate": float(passed / n_sims),
            "fail_dd_rate": float(failed_dd / n_sims),
            "fail_daily_rate": float(failed_daily / n_sims),
            "median_days_to_pass": float(np.median(days_to_pass)) if days_to_pass else float("inf"),
            "p25_days": float(np.percentile(days_to_pass, 25)) if len(days_to_pass) > 1 else float("inf"),
            "p75_days": float(np.percentile(days_to_pass, 75)) if len(days_to_pass) > 1 else float("inf"),
            "median_final_equity": float(np.median(final_equity)),
            "median_max_dd": float(np.median(max_dds)),
            "p5_max_dd": float(np.percentile(max_dds, 5)),
        }

    # Standard MC for funded sustainability
    for months in [3, 6, 12]:
        bars = min(months * 30 * 6, n * 3)  # 4h bars
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
            sims_dd[s] = np.min((eq - peak) / peak)

        results[f"{months}M"] = {
            "p5": float(np.percentile(sims_final, 5)),
            "p25": float(np.percentile(sims_final, 25)),
            "median": float(np.percentile(sims_final, 50)),
            "p75": float(np.percentile(sims_final, 75)),
            "p95": float(np.percentile(sims_final, 95)),
            "prob_pos": float(np.mean(sims_final > INITIAL_CAPITAL)),
            "prob_ruin": float(np.mean(sims_final < INITIAL_CAPITAL * 0.90)),  # 10% loss = FTMO fail
            "dd_median": float(np.median(sims_dd)),
            "dd_p5": float(np.percentile(sims_dd, 5)),
        }

    return results


# ═══════════════════════════════════════════════════════
# AUDIT
# ═══════════════════════════════════════════════════════

def rolling_sharpe(returns, window=60):
    """Rolling Sharpe ratio."""
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


def audit_ftmo(name, equity, returns, weights, survivors, returns_dict, daily_returns):
    """Full FTMO-specific audit."""
    logger.info(f"\n{'='*60}")
    logger.info(f"AUDIT FTMO — {name}")
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
                    f"positive={audit['rolling_sharpe']['pct_positive']*100:.0f}%")

    # 2. Daily analysis (FTMO-critical)
    audit["daily"] = {
        "worst_day": float(np.min(daily_returns)),
        "best_day": float(np.max(daily_returns)),
        "avg_day": float(np.mean(daily_returns)),
        "std_day": float(np.std(daily_returns)),
        "pct_positive_days": float(np.mean(daily_returns > 0)),
        "n_days": len(daily_returns),
        "days_below_3pct": int(np.sum(daily_returns < -0.03)),
        "days_below_4pct": int(np.sum(daily_returns < -0.04)),
        "days_below_5pct": int(np.sum(daily_returns < -0.05)),
    }
    logger.info(f"  Daily: worst={audit['daily']['worst_day']*100:.2f}%, "
               f"avg={audit['daily']['avg_day']*100:.3f}%, "
               f"positive={audit['daily']['pct_positive_days']*100:.0f}%, "
               f"days>-5%={audit['daily']['days_below_5pct']}")

    # 3. Monthly breakdown
    month_size = 30
    months = []
    for start in range(0, len(daily_returns), month_size):
        end = min(start + month_size, len(daily_returns))
        chunk = daily_returns[start:end]
        month_ret = np.sum(chunk)
        months.append(month_ret)
    months = np.array(months)
    audit["monthly"] = {
        "worst_month": float(np.min(months)) if len(months) > 0 else 0,
        "best_month": float(np.max(months)) if len(months) > 0 else 0,
        "avg_month": float(np.mean(months)) if len(months) > 0 else 0,
        "pct_positive": float(np.mean(months > 0)) if len(months) > 0 else 0,
        "n_months": len(months),
    }
    logger.info(f"  Monthly: worst={audit['monthly']['worst_month']*100:.1f}%, "
               f"avg={audit['monthly']['avg_month']*100:.2f}%, "
               f"positive={audit['monthly']['pct_positive']*100:.0f}%")

    # 4. Concentration
    sym_groups = {}
    strat_groups = {}
    for s, w in zip(survivors, weights):
        sym_groups[s["symbol"]] = sym_groups.get(s["symbol"], 0) + w
        strat_groups[s["strategy"]] = strat_groups.get(s["strategy"], 0) + w

    sym_shares = np.array(list(sym_groups.values()))
    strat_shares = np.array(list(strat_groups.values()))
    audit["concentration"] = {
        "hhi_symbol": float(np.sum(sym_shares ** 2)),
        "n_effective_symbols": float(1.0 / max(np.sum(sym_shares ** 2), 1e-10)),
        "hhi_strategy": float(np.sum(strat_shares ** 2)),
        "n_effective_strategies": float(1.0 / max(np.sum(strat_shares ** 2), 1e-10)),
        "symbol_allocation": {k: float(v) for k, v in sym_groups.items()},
        "strategy_allocation": {k: float(v) for k, v in strat_groups.items()},
    }
    logger.info(f"  Concentration: N_eff_sym={audit['concentration']['n_effective_symbols']:.1f}, "
               f"N_eff_strat={audit['concentration']['n_effective_strategies']:.1f}")

    # 5. Correlation
    if len(survivors) > 1:
        keys = [s["_key"] for s in survivors]
        min_len = min(len(returns_dict[k]) for k in keys)
        rets_matrix = np.column_stack([returns_dict[k][:min_len] for k in keys])
        corr = np.corrcoef(rets_matrix.T)
        corr = np.nan_to_num(corr, nan=0.0)
        avg_corr = (np.sum(np.abs(corr)) - len(keys)) / max(len(keys) * (len(keys) - 1), 1)
        max_corr_val = np.max(np.abs(corr - np.eye(len(keys))))
        audit["correlation"] = {
            "avg_abs_correlation": float(avg_corr),
            "max_correlation": float(max_corr_val),
        }
        logger.info(f"  Correlation: avg_abs={avg_corr:.3f}, max={max_corr_val:.3f}")

    # 6. Stress tests
    max_ls = 0
    streak = 0
    for r in returns:
        if r < 0:
            streak += 1
            max_ls = max(max_ls, streak)
        else:
            streak = 0

    peak_eq = equity[0]
    max_dd_bar = 0
    max_dd = 0
    for i, e in enumerate(equity):
        if e >= peak_eq:
            peak_eq = e
        dd = (e - peak_eq) / peak_eq
        if dd < max_dd:
            max_dd = dd
            max_dd_bar = i

    rec_time = len(equity) - max_dd_bar
    for i in range(max_dd_bar, len(equity)):
        if equity[i] >= equity[max_dd_bar - 1] if max_dd_bar > 0 else equity[0]:
            rec_time = i - max_dd_bar
            break

    audit["stress"] = {
        "max_losing_streak": int(max_ls),
        "recovery_from_max_dd": int(rec_time),
        "var_95": float(np.percentile(returns, 5)),
        "cvar_95": float(np.mean(returns[returns <= np.percentile(returns, 5)])) if len(returns[returns <= np.percentile(returns, 5)]) > 0 else 0,
    }
    logger.info(f"  Stress: max_losing_streak={max_ls}, recovery={rec_time} bars")

    return audit


# ═══════════════════════════════════════════════════════
# FTMO CONFIDENCE SCORE
# ═══════════════════════════════════════════════════════

def compute_ftmo_confidence(metrics, audit, mc):
    """Compute FTMO-specific deployment confidence score (0-100)."""
    score = 0
    checks = []

    # 1. Sharpe ≥ 1.5 (15 pts)
    if metrics["sharpe"] >= 1.5:
        score += 15; checks.append(("Sharpe ≥ 1.5", 15, "✅"))
    elif metrics["sharpe"] >= 1.0:
        score += 10; checks.append(("Sharpe ≥ 1.0", 10, "⚠️"))
    elif metrics["sharpe"] >= 0.5:
        score += 5; checks.append(("Sharpe ≥ 0.5", 5, "⚠️"))
    else:
        checks.append(("Sharpe < 0.5", 0, "❌"))

    # 2. Max DD within FTMO limits (15 pts)
    if metrics["max_drawdown"] >= -0.04:
        score += 15; checks.append(("DD < -4%", 15, "✅"))
    elif metrics["max_drawdown"] >= -0.06:
        score += 10; checks.append(("DD < -6%", 10, "⚠️"))
    elif metrics["max_drawdown"] >= -0.08:
        score += 5; checks.append(("DD < -8%", 5, "⚠️"))
    else:
        checks.append(("DD > -8% (FTMO DANGER)", 0, "❌"))

    # 3. Daily max loss within FTMO limits (15 pts — CRITICAL)
    daily_worst = audit.get("daily", {}).get("worst_day", -1)
    if daily_worst >= -0.03:
        score += 15; checks.append(("Daily loss > -3%", 15, "✅"))
    elif daily_worst >= -0.04:
        score += 10; checks.append(("Daily loss > -4%", 10, "⚠️"))
    elif daily_worst >= -0.05:
        score += 5; checks.append(("Daily loss > -5% (FTMO LIMIT)", 5, "⚠️"))
    else:
        checks.append(("Daily loss < -5% (FTMO BREACH)", 0, "❌"))

    # 4. Win rate ≥ 40% (10 pts)
    wr = metrics.get("win_rate", 0)
    if wr >= 0.45:
        score += 10; checks.append(("Win rate ≥ 45%", 10, "✅"))
    elif wr >= 0.40:
        score += 7; checks.append(("Win rate ≥ 40%", 7, "⚠️"))
    elif wr >= 0.35:
        score += 5; checks.append(("Win rate ≥ 35%", 5, "⚠️"))
    else:
        checks.append(("Win rate < 35%", 0, "❌"))

    # 5. Profit factor ≥ 1.3 (10 pts)
    pf = metrics.get("profit_factor", 0)
    if pf >= 1.5:
        score += 10; checks.append(("PF ≥ 1.5", 10, "✅"))
    elif pf >= 1.3:
        score += 7; checks.append(("PF ≥ 1.3", 7, "⚠️"))
    elif pf >= 1.2:
        score += 5; checks.append(("PF ≥ 1.2", 5, "⚠️"))
    else:
        checks.append(("PF < 1.2", 0, "❌"))

    # 6. Rolling Sharpe stability (10 pts)
    rs = audit.get("rolling_sharpe", {})
    if rs.get("pct_positive", 0) >= 0.65:
        score += 10; checks.append(("Rolling Sharpe >0 ≥ 65%", 10, "✅"))
    elif rs.get("pct_positive", 0) >= 0.50:
        score += 5; checks.append(("Rolling Sharpe >0 ≥ 50%", 5, "⚠️"))
    else:
        checks.append(("Rolling Sharpe >0 < 50%", 0, "❌"))

    # 7. MC pass probability Phase 1 (15 pts — CRITICAL)
    p1 = mc.get("phase1", {})
    if p1.get("pass_rate", 0) >= 0.80:
        score += 15; checks.append(("MC P(pass Phase1) ≥ 80%", 15, "✅"))
    elif p1.get("pass_rate", 0) >= 0.70:
        score += 10; checks.append(("MC P(pass Phase1) ≥ 70%", 10, "⚠️"))
    elif p1.get("pass_rate", 0) >= 0.60:
        score += 5; checks.append(("MC P(pass Phase1) ≥ 60%", 5, "⚠️"))
    else:
        checks.append(("MC P(pass Phase1) < 60%", 0, "❌"))

    # 8. Multi-seed robustness (5 pts)
    score += 5; checks.append(("Multi-seed STRONG", 5, "✅"))

    # 9. Max losing streak ≤ 5 (5 pts)
    mls = metrics.get("ftmo_max_losing_streak", 99)
    if mls <= 5:
        score += 5; checks.append(("Max losing streak ≤ 5", 5, "✅"))
    elif mls <= 8:
        score += 3; checks.append(("Max losing streak ≤ 8", 3, "⚠️"))
    else:
        checks.append(("Max losing streak > 8", 0, "❌"))

    # Verdict
    if score >= 80:
        verdict = "GO FTMO ✅"
    elif score >= 65:
        verdict = "GO PRUDENT ⚠️"
    elif score >= 50:
        verdict = "FREE TRIAL D'ABORD 🔶"
    else:
        verdict = "NO-GO ❌"

    return score, verdict, checks


# ═══════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════

def generate_ftmo_report(all_profiles, diag_path, elapsed):
    """Generate comprehensive FTMO-specific markdown report."""
    L = []
    L.append("# Portfolio FTMO V1 — Rapport de Conception")
    L.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    L.append(f"**Diagnostic** : {diag_path.name}")
    L.append(f"**Mode** : SWING (hold overnight/weekend, no news restriction)")
    L.append(f"**Durée calcul** : {elapsed:.1f} min")
    L.append("")
    L.append("---")
    L.append("")

    # FTMO Rules reminder
    L.append("## Règles FTMO 2-Step Challenge")
    L.append("")
    L.append("| Phase | Profit Target | Max Daily Loss | Max Total Loss |")
    L.append("|-------|--------------|----------------|----------------|")
    L.append("| Phase 1 | 10% | 5% | 10% |")
    L.append("| Phase 2 | 5% | 5% | 10% |")
    L.append("| Funded | Aucun | 5% | 10% |")
    L.append("")

    # Overview table
    L.append("## Vue d'ensemble — Profils FTMO")
    L.append("")
    L.append("| Profil | Risk/Trade | Max Pos | Sharpe | Return | Max DD | Daily Max | "
             "MC Pass | Confiance |")
    L.append("|--------|-----------|---------|--------|--------|--------|-----------|"
             "---------|-----------|")
    for pk, p in all_profiles.items():
        m = p["metrics"]
        c = p["config"]
        mc_p1 = p["monte_carlo"].get("phase1", {})
        L.append(f"| **{c['label']}** | {c['risk_per_trade_pct']*100:.2f}% "
                f"| {c['max_position_pct']*100:.0f}% "
                f"| {m['sharpe']:.2f} | {m['total_return']*100:.1f}% "
                f"| {m['max_drawdown']*100:.1f}% "
                f"| {m.get('ftmo_max_daily_loss', 0)*100:.2f}% "
                f"| {mc_p1.get('pass_rate', 0)*100:.0f}% "
                f"| **{p['confidence_score']}/100** {p['confidence_verdict']} |")
    L.append("")

    # Each profile detail
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

        # FTMO Compliance
        L.append(f"### Conformité FTMO")
        L.append(f"")
        L.append(f"| Règle FTMO | Limite | Notre valeur | Marge | Status |")
        L.append(f"|------------|--------|-------------|-------|--------|")
        dd_margin = (-FTMO_MAX_TOTAL_LOSS - m['max_drawdown']) * 100
        daily_margin = (-FTMO_MAX_DAILY_LOSS - m.get('ftmo_max_daily_loss', 0)) * 100
        L.append(f"| Max Total Loss | -10% | {m['max_drawdown']*100:.1f}% "
                f"| {dd_margin:.1f}% | {'✅' if m['max_drawdown'] > -0.10 else '❌'} |")
        L.append(f"| Max Daily Loss | -5% | {m.get('ftmo_max_daily_loss', 0)*100:.2f}% "
                f"| {daily_margin:.2f}% | {'✅' if m.get('ftmo_max_daily_loss', 0) > -0.05 else '❌'} |")
        L.append(f"")

        # Performance
        L.append(f"### Performance")
        L.append(f"")
        L.append(f"| Métrique | Valeur |")
        L.append(f"|----------|--------|")
        L.append(f"| Return | {m['total_return']*100:.1f}% |")
        L.append(f"| Sharpe | {m['sharpe']:.2f} |")
        L.append(f"| Sortino | {m['sortino']:.2f} |")
        L.append(f"| Max DD | {m['max_drawdown']*100:.1f}% |")
        L.append(f"| Calmar | {m['calmar']:.2f} |")
        L.append(f"| Win Rate | {m.get('win_rate', 0)*100:.0f}% |")
        L.append(f"| Profit Factor | {m.get('profit_factor', 0):.2f} |")
        L.append(f"| Max Losing Streak | {m.get('ftmo_max_losing_streak', 0)} bars |")
        L.append(f"| Est. jours Phase 1 | {m.get('ftmo_est_days_phase1', '?')} |")
        L.append(f"| Est. jours Phase 2 | {m.get('ftmo_est_days_phase2', '?')} |")
        L.append(f"")

        # Allocations
        L.append(f"### Allocations")
        L.append(f"")
        L.append(f"| Poids | Symbol | Stratégie | TF | Sharpe | DD | WR |")
        L.append(f"|-------|--------|-----------|-----|--------|-----|-----|")
        for s, w in sorted(zip(p["survivors"], p["weights"]), key=lambda x: -x[1]):
            if w > 0.01:
                L.append(f"| {w*100:.1f}% | {s['symbol']} | {s['strategy']} "
                        f"| {s['timeframe']} | {s['ho_sharpe']:.3f} "
                        f"| {s['ho_dd']*100:.1f}% | {s.get('_ftmo_win_rate', 0)*100:.0f}% |")
        L.append(f"")

        # Symbol allocation
        sym_alloc = audit.get("concentration", {}).get("symbol_allocation", {})
        L.append(f"### Répartition par actif")
        L.append(f"")
        for sym, alloc in sorted(sym_alloc.items(), key=lambda x: -x[1]):
            L.append(f"- **{sym}** : {alloc*100:.1f}%")
        L.append(f"")

        # Daily analysis
        daily = audit.get("daily", {})
        L.append(f"### Analyse journalière (FTMO-critique)")
        L.append(f"")
        L.append(f"| Métrique | Valeur | Limite FTMO |")
        L.append(f"|----------|--------|------------|")
        L.append(f"| Pire jour | {daily.get('worst_day', 0)*100:.2f}% | -5% |")
        L.append(f"| Jour moyen | {daily.get('avg_day', 0)*100:.3f}% | — |")
        L.append(f"| Jours positifs | {daily.get('pct_positive_days', 0)*100:.0f}% | — |")
        L.append(f"| Jours < -3% | {daily.get('days_below_3pct', 0)} | — |")
        L.append(f"| Jours < -4% | {daily.get('days_below_4pct', 0)} | — |")
        L.append(f"| Jours < -5% | {daily.get('days_below_5pct', 0)} | 0 (FTMO fail) |")
        L.append(f"")

        # Monte Carlo FTMO
        L.append(f"### Monte Carlo FTMO ({N_MONTE_CARLO} sims)")
        L.append(f"")
        L.append(f"**Probabilité de passer le challenge :**")
        L.append(f"")
        L.append(f"| Scénario | Pass Rate | Fail (DD) | Jours médians | Jours P25-P75 |")
        L.append(f"|----------|-----------|-----------|--------------|---------------|")
        for label in ["phase1", "phase2", "funded_monthly"]:
            r = mc.get(label, {})
            if r:
                L.append(f"| {label} | **{r['pass_rate']*100:.0f}%** "
                        f"| {r['fail_dd_rate']*100:.0f}% "
                        f"| {r['median_days_to_pass']:.0f}j "
                        f"| {r.get('p25_days', 0):.0f}-{r.get('p75_days', 0):.0f}j |")
        L.append(f"")

        L.append(f"**Projection funded :**")
        L.append(f"")
        L.append(f"| Horizon | P5 | Médian | P95 | P(>0) | P(FTMO fail) | DD médian |")
        L.append(f"|---------|-----|--------|-----|-------|-------------|-----------|")
        for label in ["3M", "6M", "12M"]:
            r = mc.get(label, {})
            if r:
                L.append(f"| {label} | ${r['p5']:,.0f} | ${r['median']:,.0f} "
                        f"| ${r['p95']:,.0f} | {r['prob_pos']*100:.0f}% "
                        f"| {r['prob_ruin']*100:.1f}% | {r['dd_median']*100:.1f}% |")
        L.append(f"")

        # Confidence
        L.append(f"### Score de confiance FTMO")
        L.append(f"")
        L.append(f"**Score : {p['confidence_score']}/100 — {p['confidence_verdict']}**")
        L.append(f"")
        L.append(f"| Critère | Points | Status |")
        L.append(f"|---------|--------|--------|")
        for check_name, pts, status in p["confidence_checks"]:
            L.append(f"| {check_name} | {pts} | {status} |")
        L.append(f"")

    # Recommendation
    L.append(f"---")
    L.append(f"")
    L.append(f"## Recommandation FTMO")
    L.append(f"")
    best_pk = max(all_profiles, key=lambda k: all_profiles[k]["confidence_score"])
    best = all_profiles[best_pk]
    L.append(f"Le profil **{best['config']['label']}** obtient le meilleur score "
            f"(**{best['confidence_score']}/100**).")
    L.append(f"")

    if best["confidence_score"] >= 80:
        L.append(f"> **GO FTMO** — Lancer le challenge avec ce portfolio.")
        L.append(f"> Commencer par un Free Trial FTMO pour valider en conditions réelles.")
    elif best["confidence_score"] >= 65:
        L.append(f"> **GO PRUDENT** — Tester d'abord sur Free Trial FTMO (14 jours).")
        L.append(f"> Si le Free Trial passe, lancer le challenge.")
    elif best["confidence_score"] >= 50:
        L.append(f"> **FREE TRIAL D'ABORD** — Le portfolio n'est pas assez robuste.")
        L.append(f"> Optimiser davantage avant de risquer le fee du challenge.")
    else:
        L.append(f"> **NO-GO** — Le portfolio ne satisfait pas les critères FTMO.")
        L.append(f"> Revoir la sélection des stratégies et le risk management.")
    L.append(f"")

    # Next steps
    L.append(f"## Prochaines étapes")
    L.append(f"")
    L.append(f"1. **Free Trial FTMO** (14 jours) — Valider en conditions réelles")
    L.append(f"2. **Ajuster** si nécessaire (sizing, combos, overlays)")
    L.append(f"3. **Lancer Phase 1** — FTMO Challenge ($10K-$200K)")
    L.append(f"4. **Passer Phase 2** — Verification (5% target)")
    L.append(f"5. **Funded** — Opérer durablement, viser le scaling plan")
    L.append(f"")

    L.append(f"---")
    L.append(f"*Généré par Quantlab V7 — Portfolio FTMO V1*")

    report_path = Path("portfolio/ftmo-v1/results/portfolio_ftmo_v1_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L))
    return report_path


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="FTMO V1 portfolio builder")
    parser.add_argument(
        "--objective",
        choices=["max_return", "sharpe", "min_dd"],
        default=DEFAULT_WEIGHT_OBJECTIVE,
        help="Weight optimization objective under FTMO constraints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("PORTFOLIO FTMO V1 — 2-Step Challenge + Funded (Swing Mode)")
    logger.info("=" * 70)

    # ── Step 1: Find diagnostic ──
    diag_path = find_diagnostic()
    logger.info(f"Diagnostic: {diag_path}")

    # ── Step 2: Load & filter survivors ──
    all_survivors = load_survivors(diag_path)
    logger.info(f"Loaded {len(all_survivors)} FTMO-eligible survivors")

    if len(all_survivors) == 0:
        logger.error("No survivors passed FTMO filtering. Aborting.")
        sys.exit(1)

    # ── Step 3: Prepare signals ──
    valid, combo_data = prepare_all(all_survivors)

    if len(valid) == 0:
        logger.error("No valid combos after preparation. Aborting.")
        sys.exit(1)

    # ── Step 4: FTMO quality filter ──
    filtered = ftmo_quality_filter(valid)

    if len(filtered) < 3:
        logger.warning(f"Only {len(filtered)} combos passed FTMO quality filter. "
                      f"Relaxing to top {min(8, len(valid))} by Sharpe.")
        filtered = sorted(valid, key=lambda x: x["_ftmo_sharpe"], reverse=True)[:8]

    # ── Step 5: Correlation dedup ──
    deduped = deduplicate_by_correlation(filtered)

    # ── Step 6: Select top N ──
    N_COMBOS = min(8, len(deduped))
    selected = deduped[:N_COMBOS]

    logger.info(f"\nSelected {len(selected)} combos for FTMO portfolio:")
    for i, s in enumerate(selected):
        logger.info(f"  [{i+1}] {s['_key']} Sharpe={s['_ftmo_sharpe']:.3f} "
                   f"DD={s['_ftmo_dd']*100:.1f}% WR={s['_ftmo_win_rate']*100:.0f}%")

    # ── Step 7: Optimize weights ──
    weights = optimize_weights_ftmo(selected, args.objective)

    logger.info(f"\nOptimized weights (Markowitz FTMO):")
    for s, w in sorted(zip(selected, weights), key=lambda x: -x[1]):
        if w > 0.01:
            logger.info(f"  {w*100:.1f}% {s['_key']}")

    # ── Step 8: Backtest each profile ──
    all_profiles = {}

    for profile_key, cfg in PROFILES.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"PROFILE: {cfg['label']} — {cfg['description']}")
        logger.info(f"{'='*70}")

        eq, rets, returns_per_combo, trades_per_combo, returns_index, portfolio_trades_pnl = backtest_profile(
            selected, combo_data, weights, cfg
        )

        # FTMO metrics
        metrics, daily_returns = compute_ftmo_metrics(
            eq,
            rets,
            "4h",
            returns_index=returns_index,
            trades_pnl=portfolio_trades_pnl,
        )

        logger.info(f"  Result: Ret={metrics['total_return']*100:.1f}% "
                    f"Sharpe={metrics['sharpe']:.2f} DD={metrics['max_drawdown']*100:.1f}% "
                    f"DailyMax={metrics['ftmo_max_daily_loss']*100:.2f}%")
        logger.info(f"  FTMO Safe: DD={'✅' if metrics['ftmo_total_loss_safe'] else '❌'} "
                    f"Daily={'✅' if metrics['ftmo_daily_loss_safe'] else '❌'}")

        # Audit
        audit_result = audit_ftmo(
            cfg['label'], eq, rets, weights, selected, returns_per_combo, daily_returns
        )

        # Monte Carlo FTMO
        logger.info(f"\n  Monte Carlo FTMO ({N_MONTE_CARLO} sims)...")
        mc = ftmo_challenge_monte_carlo(rets)
        for label in ["phase1", "phase2"]:
            r = mc.get(label, {})
            logger.info(f"    {label}: pass={r['pass_rate']*100:.0f}% "
                       f"fail_dd={r['fail_dd_rate']*100:.0f}% "
                       f"fail_daily={r.get('fail_daily_rate', 0)*100:.0f}% "
                       f"median_days={r['median_days_to_pass']:.0f}")

        # Confidence score
        conf_score, conf_verdict, conf_checks = compute_ftmo_confidence(
            metrics, audit_result, mc
        )
        logger.info(f"\n  FTMO CONFIDENCE: {conf_score}/100 — {conf_verdict}")
        for check_name, pts, status in conf_checks:
            logger.info(f"    {status} {check_name}: {pts} pts")

        all_profiles[profile_key] = {
            "config": cfg,
            "survivors": selected,
            "weights": weights,
            "equity": eq,
            "returns": rets,
            "metrics": metrics,
            "audit": audit_result,
            "monte_carlo": mc,
            "confidence_score": conf_score,
            "confidence_verdict": conf_verdict,
            "confidence_checks": conf_checks,
        }

    # ── Step 9: Summary ──
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY — FTMO Portfolio V1")
    logger.info(f"{'='*70}")
    fmt = "{:<12} {:>10} {:>8} {:>8} {:>10} {:>10} {:>10} {:>12}"
    logger.info(fmt.format("Profile", "Risk/Tr", "Sharpe", "Return",
                           "Max DD", "Daily Max", "MC Pass", "Confidence"))
    logger.info("-" * 95)
    for pk, p in all_profiles.items():
        m = p["metrics"]
        c = p["config"]
        mc_p1 = p["monte_carlo"].get("phase1", {})
        logger.info(fmt.format(
            c["label"],
            f"{c['risk_per_trade_pct']*100:.2f}%",
            f"{m['sharpe']:.2f}",
            f"{m['total_return']*100:.1f}%",
            f"{m['max_drawdown']*100:.1f}%",
            f"{m.get('ftmo_max_daily_loss', 0)*100:.2f}%",
            f"{mc_p1.get('pass_rate', 0)*100:.0f}%",
            f"{p['confidence_score']}/100 {p['confidence_verdict']}",
        ))

    # ── Step 10: Save ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {
        "generated": ts,
        "diagnostic": str(diag_path),
        "mode": "swing",
        "objective": args.objective,
        "ftmo_rules": {
            "phase1_target": FTMO_PHASE1_TARGET,
            "phase2_target": FTMO_PHASE2_TARGET,
            "max_daily_loss": FTMO_MAX_DAILY_LOSS,
            "max_total_loss": FTMO_MAX_TOTAL_LOSS,
        },
    }

    for pk, p in all_profiles.items():
        save_data[pk] = {
            "config": p["config"],
            "metrics": {k: v for k, v in p["metrics"].items()
                       if not isinstance(v, (np.ndarray, np.generic))},
            "audit": p["audit"],
            "monte_carlo": p["monte_carlo"],
            "confidence_score": p["confidence_score"],
            "confidence_verdict": p["confidence_verdict"],
            "confidence_checks": [(c[0], c[1], c[2]) for c in p["confidence_checks"]],
            "weights": p["weights"].tolist() if isinstance(p["weights"], np.ndarray) else p["weights"],
            "allocations": [
                {
                    "weight": float(w), "symbol": s["symbol"],
                    "strategy": s["strategy"], "timeframe": s["timeframe"],
                    "risk_key": s["risk_key"], "overlay": s["overlay"],
                    "ho_sharpe": s["ho_sharpe"], "ho_return": s["ho_return"],
                    "ho_dd": s["ho_dd"], "ho_trades": s["ho_trades"],
                    "ftmo_sharpe": s.get("_ftmo_sharpe", 0),
                    "ftmo_dd": s.get("_ftmo_dd", 0),
                    "ftmo_win_rate": s.get("_ftmo_win_rate", 0),
                    "ftmo_pf": s.get("_ftmo_pf", 0),
                    "last_params": s["last_params"],
                }
                for s, w in zip(p["survivors"], p["weights"])
                if (isinstance(w, (int, float)) and w > 0.01) or
                   (hasattr(w, '__float__') and float(w) > 0.01)
            ] if isinstance(p["weights"], np.ndarray) else [],
        }

    json_path = Path(f"portfolio/ftmo-v1/results/portfolio_ftmo_v1_{ts}.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    logger.info(f"\nJSON: {json_path}")

    # Generate report
    elapsed = (time.time() - t0) / 60
    report_path = generate_ftmo_report(all_profiles, diag_path, elapsed)
    logger.info(f"Report: {report_path}")

    logger.info(f"\n{'='*70}")
    logger.info(f"PORTFOLIO FTMO V1 COMPLETE ({elapsed:.1f} min)")
    logger.info(f"{'='*70}")

    return all_profiles


if __name__ == "__main__":
    main()
