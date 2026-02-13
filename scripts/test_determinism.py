#!/usr/bin/env python3
"""
Quick test: verify walk-forward determinism with seeds.
Runs the same combo twice with same seed → must get identical results.
Then runs with different seed → results should differ.
Also tests run_walk_forward_robust (multi-seed).
"""

import sys
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.ingestion import load_all_symbols_data, load_settings
from engine.backtester import RiskConfig
from engine.walk_forward import WalkForwardConfig, run_walk_forward, run_walk_forward_robust
from strategies.registry import get_strategy

logger.remove()
logger.add(sys.stderr, level="INFO")


def main():
    settings = load_settings()
    data_by_symbol = load_all_symbols_data(settings)

    # Use a fast combo: XRPUSDT/stochastic/1d with few trials
    symbol = "XRPUSDT"
    strategy_name = "stochastic_oscillator"
    timeframe = "1d"

    strategy = get_strategy(strategy_name)
    data = data_by_symbol[symbol][timeframe]

    risk = RiskConfig(
        max_position_pct=0.25,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.15,
        dynamic_slippage=True,
        base_slippage=0.0005,
        max_slippage=0.005,
        volatility_lookback=20,
        max_trades_per_day=10,
        cooldown_after_loss=0,
    )

    base_config = dict(
        strategy=strategy,
        data=data,
        timeframe=timeframe,
        reoptim_frequency="3M",
        training_window="1Y",
        param_bounds_scale=1.0,
        optim_metric="sharpe",
        n_optim_trials=20,  # Low for speed
        commission=settings["engine"]["commission_rate"],
        slippage=settings["engine"]["slippage_rate"],
        risk=risk,
    )

    # ── TEST 1: Same seed → identical results ──
    print("\n" + "=" * 60)
    print("TEST 1: Same seed (42) → must be IDENTICAL")
    print("=" * 60)

    config_a = WalkForwardConfig(**base_config, seed=42)
    result_a = run_walk_forward(config_a)

    config_b = WalkForwardConfig(**base_config, seed=42)
    result_b = run_walk_forward(config_b)

    sharpe_a = result_a.metrics["sharpe"]
    sharpe_b = result_b.metrics["sharpe"]
    ret_a = result_a.metrics["total_return"]
    ret_b = result_b.metrics["total_return"]

    print(f"\n  Run A: Sharpe={sharpe_a:.6f}, Return={ret_a:.6f}")
    print(f"  Run B: Sharpe={sharpe_b:.6f}, Return={ret_b:.6f}")

    identical = np.isclose(sharpe_a, sharpe_b) and np.isclose(ret_a, ret_b)
    print(f"  Identical: {'✅ YES' if identical else '❌ NO'}")

    # ── TEST 2: Different seed → different results ──
    print("\n" + "=" * 60)
    print("TEST 2: Different seeds (42 vs 99) → should DIFFER")
    print("=" * 60)

    config_c = WalkForwardConfig(**base_config, seed=99)
    result_c = run_walk_forward(config_c)

    sharpe_c = result_c.metrics["sharpe"]
    ret_c = result_c.metrics["total_return"]

    print(f"\n  Seed 42: Sharpe={sharpe_a:.6f}, Return={ret_a:.6f}")
    print(f"  Seed 99: Sharpe={sharpe_c:.6f}, Return={ret_c:.6f}")

    different = not (np.isclose(sharpe_a, sharpe_c) and np.isclose(ret_a, ret_c))
    print(f"  Different: {'✅ YES' if different else '⚠️ NO (could happen by chance)'}")

    # ── TEST 3: No seed → stochastic (backward compat) ──
    print("\n" + "=" * 60)
    print("TEST 3: seed=None → stochastic (backward compat)")
    print("=" * 60)

    config_d = WalkForwardConfig(**base_config, seed=None)
    result_d = run_walk_forward(config_d)

    config_e = WalkForwardConfig(**base_config, seed=None)
    result_e = run_walk_forward(config_e)

    sharpe_d = result_d.metrics["sharpe"]
    sharpe_e = result_e.metrics["sharpe"]

    print(f"\n  Run D: Sharpe={sharpe_d:.6f}")
    print(f"  Run E: Sharpe={sharpe_e:.6f}")
    print(f"  (May or may not differ — stochastic mode)")

    # ── TEST 4: Robust walk-forward (multi-seed) ──
    print("\n" + "=" * 60)
    print("TEST 4: run_walk_forward_robust (3 seeds)")
    print("=" * 60)

    config_robust = WalkForwardConfig(**base_config, seed=42)
    result_robust = run_walk_forward_robust(config_robust, n_seeds=3, aggregation="median")

    m = result_robust.metrics
    print(f"\n  Chosen Sharpe: {m['sharpe']:.4f}")
    print(f"  Robust stats:")
    print(f"    Seeds: {m.get('robust_n_seeds', '?')}")
    print(f"    Sharpe min/median/max: {m.get('robust_sharpe_min', 0):.4f} / "
          f"{m.get('robust_sharpe_median', 0):.4f} / {m.get('robust_sharpe_max', 0):.4f}")
    print(f"    Sharpe std: {m.get('robust_sharpe_std', 0):.4f}")
    print(f"    Consistency: {m.get('robust_consistency', 0):.4f}")

    # ── SUMMARY ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  TEST 1 (determinism):    {'✅ PASS' if identical else '❌ FAIL'}")
    print(f"  TEST 2 (seed variation): {'✅ PASS' if different else '⚠️ INCONCLUSIVE'}")
    print(f"  TEST 3 (backward compat): ✅ PASS (no crash)")
    print(f"  TEST 4 (robust multi-seed): ✅ PASS (no crash)")
    print(f"    → Variance across seeds: std={m.get('robust_sharpe_std', 0):.4f}")
    print("=" * 60)

    if identical:
        print("\n🎯 Walk-forward is now DETERMINISTIC with seeds!")
    else:
        print("\n❌ Determinism test FAILED — investigate!")
        sys.exit(1)


if __name__ == "__main__":
    main()
