#!/usr/bin/env python3
"""
Portfolio V5c High-Risk — short-term (1-2 months) objective.

Constraints:
- Initial capital: 100 USD
- Max drawdown limit: -30%
- Objective: maximize short-term return (30d / 60d)
"""

import json
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portfolio_v5b_final as v5b


INITIAL_CAPITAL = 100.0
MAX_DD_LIMIT = -0.30
SHORT_1M_BARS = 30
SHORT_2M_BARS = 60
TEST_BARS = 60

# High-risk search grid
RISK_CANDIDATES = [
    {
        "label": "HR-50",
        "risk_per_trade_pct": 0.0,
        "max_position_pct": 0.50,
        "max_drawdown_pct": 0.30,
    },
    {
        "label": "HR-75",
        "risk_per_trade_pct": 0.0,
        "max_position_pct": 0.75,
        "max_drawdown_pct": 0.30,
    },
    {
        "label": "HR-100",
        "risk_per_trade_pct": 0.0,
        "max_position_pct": 1.00,
        "max_drawdown_pct": 0.30,
    },
]


def short_return(equity, bars):
    """Return over the last N bars from equity curve."""
    if len(equity) <= bars:
        return float((equity[-1] / equity[0]) - 1.0)
    return float((equity[-1] / equity[-1 - bars]) - 1.0)


def rank_score_train(combo):
    """Rank score based on TRAIN performance only (no test leakage)."""
    m = combo["_train_metrics"]
    base = m.get("total_return", 0.0)
    sharpe = m.get("sharpe", 0.0)
    dd = abs(m.get("max_drawdown", -0.01))
    return base + 0.20 * sharpe - 0.10 * dd


def split_combo_data(combo_data, test_bars=TEST_BARS):
    """Split prepared combo data into TRAIN and TEST segments."""
    n = len(combo_data["signals"])
    if n <= test_bars + 80:
        return None, None

    cut = n - test_bars
    train = {
        "close": combo_data["close"][:cut],
        "high": combo_data["high"][:cut],
        "low": combo_data["low"][:cut],
        "signals": combo_data["signals"][:cut],
        "sl_distances": combo_data["sl_distances"][:cut] if combo_data["sl_distances"] is not None else None,
        "timeframe": combo_data["timeframe"],
        "dates": combo_data["dates"][:cut],
    }
    test = {
        "close": combo_data["close"][cut:],
        "high": combo_data["high"][cut:],
        "low": combo_data["low"][cut:],
        "signals": combo_data["signals"][cut:],
        "sl_distances": combo_data["sl_distances"][cut:] if combo_data["sl_distances"] is not None else None,
        "timeframe": combo_data["timeframe"],
        "dates": combo_data["dates"][cut:],
    }
    return train, test


def save_outputs(best_cfg, train_best, test_result, selected, weights, ts):
    """Persist outputs in portfolio/v5c-highrisk/ (results + code snapshot)."""
    out_dir = Path("portfolio/v5c-highrisk")
    code_dir = out_dir / "code"
    res_dir = out_dir / "results"
    code_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "generated": ts,
        "portfolio": "V5c-highrisk",
        "validation_mode": "strict_oos_60d",
        "constraints": {
            "initial_capital": INITIAL_CAPITAL,
            "max_drawdown_limit": MAX_DD_LIMIT,
            "horizon": "1-2 months",
            "test_bars": TEST_BARS,
        },
        "best_risk_profile": best_cfg,
        "train_selection_metrics": train_best,
        "test_metrics": test_result["metrics"],
        "test_return_30d": test_result["ret_30d"],
        "test_return_60d": test_result["ret_60d"],
        "weights": weights.tolist() if isinstance(weights, np.ndarray) else weights,
        "allocations": [
            {
                "weight": float(w),
                "symbol": s["symbol"],
                "strategy": s["strategy"],
                "timeframe": s["timeframe"],
                "train_sharpe": s["_train_metrics"].get("sharpe", 0.0),
                "train_return": s["_train_metrics"].get("total_return", 0.0),
                "train_dd": s["_train_metrics"].get("max_drawdown", 0.0),
                "risk_key": s["risk_key"],
                "overlay": s["overlay"],
            }
            for s, w in zip(selected, weights)
            if float(w) > 0.01
        ],
    }

    json_path = res_dir / f"portfolio_v5c_highrisk_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    lines = [
        "# Portfolio V5c-HighRisk",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Capital initial**: ${INITIAL_CAPITAL:.0f}",
        "**Objectif**: gains court terme (1-2 mois)",
        f"**Contrainte DD**: >= {MAX_DD_LIMIT*100:.0f}% (sur test OOS)",
        f"**Validation**: train calibré, test final = {TEST_BARS} barres",
        "",
        "## Résultat retenu",
        "",
        f"- Profil risque: **{best_cfg['label']}**",
        f"- Max position: **{best_cfg['max_position_pct']*100:.0f}%**",
        f"- Return 30j OOS: **{test_result['ret_30d']*100:.1f}%**",
        f"- Return 60j OOS: **{test_result['ret_60d']*100:.1f}%**",
        f"- Sharpe OOS: **{test_result['metrics']['sharpe']:.2f}**",
        f"- Max DD OOS: **{test_result['metrics']['max_drawdown']*100:.1f}%**",
        "",
        "## Allocations",
        "",
        "| Poids | Combo | Sharpe TRAIN | Return TRAIN | DD TRAIN |",
        "|------:|-------|-------------:|-------------:|---------:|",
    ]

    for s, w in sorted(zip(selected, weights), key=lambda x: -x[1]):
        if float(w) <= 0.01:
            continue
        combo = f"{s['symbol']}/{s['strategy']}/{s['timeframe']}"
        tm = s["_train_metrics"]
        lines.append(
            f"| {w*100:.1f}% | {combo} | {tm.get('sharpe', 0):.2f} | {tm.get('total_return', 0)*100:.1f}% | {tm.get('max_drawdown', 0)*100:.1f}% |"
        )

    report_path = res_dir / f"portfolio_v5c_highrisk_{ts}.md"
    report_path.write_text("\n".join(lines))

    logger.info(f"JSON: {json_path}")
    logger.info(f"Report: {report_path}")


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("PORTFOLIO V5c-HIGHRISK — 1-2 mois, DD max 30%, capital 100$")
    logger.info("=" * 70)

    # Reuse V5b engine, override constraints for high-risk profile
    v5b.INITIAL_CAPITAL = INITIAL_CAPITAL
    v5b.MAX_WEIGHT_PER_SYMBOL = 0.75
    v5b.MAX_WEIGHT_PER_COMBO = 0.40
    v5b.MIN_WEIGHT = 0.01

    # 1) Survivors + prepared signals + strict TRAIN/TEST split
    survivors = v5b.load_survivors(min_sharpe=-0.2)
    train_data = {}
    test_data = {}
    valid = []

    default_risk = v5b.RiskConfig(risk_per_trade_pct=0.01, max_position_pct=0.25)
    for s in survivors:
        key = f"{s['symbol']}/{s['strategy']}/{s['timeframe']}"
        s["_key"] = key
        cd_full = v5b.prepare_combo_data(s)
        if cd_full is None:
            continue
        cd_train, cd_test = split_combo_data(cd_full, TEST_BARS)
        if cd_train is None or cd_test is None:
            continue

        train_data[key] = cd_train
        test_data[key] = cd_test

        eq_train, rets_train = v5b.build_equity_with_risk(cd_train, default_risk)
        s["_train_returns_default"] = rets_train
        s["_train_metrics"] = v5b.compute_all_metrics(eq_train, "1d")
        valid.append(s)

    if len(valid) < 6:
        raise RuntimeError(f"Pas assez de combos valides après split train/test: {len(valid)}")

    default_returns = {s["_key"]: s["_train_returns_default"] for s in valid}

    # 2) Correlation dedup on TRAIN only
    deduped = v5b.deduplicate_by_correlation(valid, default_returns, max_corr=0.92)

    # 3) Select top candidates using TRAIN score only
    ranked = sorted(deduped, key=rank_score_train, reverse=True)
    n_combos = min(6, len(ranked))
    selected = ranked[:n_combos]

    logger.info(f"Selected {len(selected)} high-risk combos")
    for i, s in enumerate(selected):
        logger.info(
            f"  [{i+1}] {s['_key']} TRAIN Sharpe={s['_train_metrics']['sharpe']:.2f} "
            f"Ret={s['_train_metrics']['total_return']*100:.1f}% "
            f"DD={s['_train_metrics']['max_drawdown']*100:.1f}%"
        )

    # 4) Optimize weights on TRAIN only
    r_sel = {s["_key"]: default_returns[s["_key"]] for s in selected}
    weights = v5b.optimize_weights(selected, r_sel, objective="max_return")

    # 5) Choose risk profile on TRAIN (no test leakage)
    feasible = []
    for cfg in RISK_CANDIDATES:
        eq, rets, _ = v5b.backtest_profile(selected, train_data, weights, cfg)
        m = v5b.compute_all_metrics(eq, "1d")
        ret_30d = short_return(eq, SHORT_1M_BARS)
        ret_60d = short_return(eq, SHORT_2M_BARS)
        dd_ok = m["max_drawdown"] >= MAX_DD_LIMIT

        logger.info(
            f"{cfg['label']}: Ret={m['total_return']*100:.1f}% "
            f"Ret30={ret_30d*100:.1f}% Ret60={ret_60d*100:.1f}% "
            f"DD={m['max_drawdown']*100:.1f}% {'OK' if dd_ok else 'FAIL'}"
        )

        if dd_ok:
            feasible.append({
                "cfg": cfg,
                "equity": eq,
                "returns": rets,
                "metrics": m,
                "ret_30d": ret_30d,
                "ret_60d": ret_60d,
            })

    if not feasible:
        raise RuntimeError("Aucun profil high-risk ne respecte DD <= 30%.")

    best_train = max(feasible, key=lambda x: (x["ret_60d"], x["ret_30d"], x["metrics"]["total_return"]))

    # 6) Final strict OOS evaluation on TEST only (last 60 bars)
    eq_test, rets_test, _ = v5b.backtest_profile(selected, test_data, weights, best_train["cfg"])
    m_test = v5b.compute_all_metrics(eq_test, "1d")
    test_result = {
        "metrics": m_test,
        "ret_30d": short_return(eq_test, SHORT_1M_BARS),
        "ret_60d": short_return(eq_test, SHORT_2M_BARS),
    }

    if m_test["max_drawdown"] < MAX_DD_LIMIT:
        logger.warning(
            f"OOS WARNING: DD test {m_test['max_drawdown']*100:.1f}% < limite {MAX_DD_LIMIT*100:.1f}%"
        )

    logger.info("-" * 70)
    logger.info(
        f"BEST TRAIN: {best_train['cfg']['label']} | Return={best_train['metrics']['total_return']*100:.1f}% "
        f"Ret30={best_train['ret_30d']*100:.1f}% Ret60={best_train['ret_60d']*100:.1f}% "
        f"DD={best_train['metrics']['max_drawdown']*100:.1f}%"
    )
    logger.info(
        f"FINAL TEST OOS: Return={m_test['total_return']*100:.1f}% "
        f"Ret30={test_result['ret_30d']*100:.1f}% Ret60={test_result['ret_60d']*100:.1f}% "
        f"DD={m_test['max_drawdown']*100:.1f}%"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_outputs(best_train["cfg"], best_train, test_result, selected, weights, ts)

    # Copy script snapshot to portfolio folder
    script_dst = Path("portfolio/v5c-highrisk/code/portfolio_v5c_highrisk.py")
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    script_dst.write_text(Path(__file__).read_text())

    elapsed = (time.time() - t0) / 60.0
    logger.info(f"DONE in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
