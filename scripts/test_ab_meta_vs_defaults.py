#!/usr/bin/env python3
"""
Test A/B : Méta-optimisation vs Defaults fixes.

Pour chaque combo des 5 profils méta-optimisés V3, on compare :
  A) Walk-forward avec les meta-params trouvés par la méta-optimisation
  B) Walk-forward avec des defaults sensibles fixes

Chaque variante est testée avec run_walk_forward_robust (5 seeds, médiane)
pour éliminer la variance stochastique.

Output:
  - results/test_ab_meta_vs_defaults_{timestamp}.json
  - docs/results/07_test_ab_meta_vs_defaults.md
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.ingestion import load_all_symbols_data, load_settings
from engine.backtester import RiskConfig
from engine.metrics import composite_score
from engine.walk_forward import (
    WalkForwardConfig,
    run_walk_forward_robust,
)
from strategies.registry import get_strategy

# ── Config ──
META_V3_PATH = "results/meta_profiles_v3_20260211_195716.json"
N_SEEDS = 5
COMPOSITE_WEIGHTS = {"sharpe": 0.35, "sortino": 0.25, "calmar": 0.20, "stability": 0.20}

# Defaults sensibles pour daily crypto
DEFAULTS = {
    "reoptim_frequency": "3M",
    "training_window": "1Y",
    "param_bounds_scale": 1.0,
    "optim_metric": "sharpe",
    "n_optim_trials": 100,
}

# Alternative defaults (conservative)
DEFAULTS_CONSERVATIVE = {
    "reoptim_frequency": "6M",
    "training_window": "2Y",
    "param_bounds_scale": 0.8,
    "optim_metric": "sharpe",
    "n_optim_trials": 100,
}


def load_meta_profiles():
    with open(META_V3_PATH) as f:
        return json.load(f)


def run_variant(strategy, data, timeframe, meta_params, settings, risk, label):
    """Run a single variant with robust multi-seed."""
    config = WalkForwardConfig(
        strategy=strategy,
        data=data,
        timeframe=timeframe,
        reoptim_frequency=meta_params["reoptim_frequency"],
        training_window=meta_params["training_window"],
        param_bounds_scale=meta_params["param_bounds_scale"],
        optim_metric=meta_params["optim_metric"],
        n_optim_trials=meta_params["n_optim_trials"],
        commission=settings["engine"]["commission_rate"],
        slippage=settings["engine"]["slippage_rate"],
        risk=risk,
        seed=42,
    )

    logger.info(f"  ▶ {label}: reoptim={meta_params['reoptim_frequency']}, "
                f"window={meta_params['training_window']}, "
                f"bounds={meta_params['param_bounds_scale']}, "
                f"metric={meta_params['optim_metric']}, "
                f"trials={meta_params['n_optim_trials']}")

    result = run_walk_forward_robust(config, n_seeds=N_SEEDS, aggregation="median")
    return result


def main():
    logger.info("=" * 70)
    logger.info("  TEST A/B : META-OPTIMISATION vs DEFAULTS FIXES")
    logger.info(f"  Seeds par variante: {N_SEEDS}")
    logger.info("=" * 70)

    start_time = time.time()

    settings = load_settings()
    data_by_symbol = load_all_symbols_data(settings)
    profiles = load_meta_profiles()

    risk = RiskConfig(
        max_position_pct=settings["risk"]["max_position_pct"],
        max_daily_loss_pct=settings["risk"]["max_daily_loss_pct"],
        max_drawdown_pct=settings["risk"]["max_drawdown_pct"],
        dynamic_slippage=settings["risk"]["dynamic_slippage"],
        base_slippage=settings["risk"]["base_slippage"],
        max_slippage=settings["risk"]["max_slippage"],
        volatility_lookback=settings["risk"]["volatility_lookback"],
        max_trades_per_day=settings["risk"]["max_trades_per_day"],
        cooldown_after_loss=settings["risk"]["cooldown_after_loss"],
    )

    results = []

    for i, profile in enumerate(profiles):
        symbol = profile["symbol"]
        strategy_name = profile["strategy"] if "strategy" in profile else profile["strategy_name"]
        timeframe = profile["timeframe"]

        logger.info(f"\n{'─' * 60}")
        logger.info(f"COMBO {i+1}/{len(profiles)}: {symbol}/{strategy_name}/{timeframe}")
        logger.info(f"{'─' * 60}")

        strategy = get_strategy(strategy_name)
        data = data_by_symbol[symbol][timeframe]

        # ── Variant A: Meta-optimised params ──
        meta_params = {
            "reoptim_frequency": profile["reoptim_frequency"],
            "training_window": profile["training_window"],
            "param_bounds_scale": profile["param_bounds_scale"],
            "optim_metric": profile["optim_metric"],
            "n_optim_trials": profile["n_optim_trials"],
        }
        result_meta = run_variant(strategy, data, timeframe, meta_params, settings, risk, "META-OPT")

        # ── Variant B: Defaults fixes ──
        result_defaults = run_variant(strategy, data, timeframe, DEFAULTS, settings, risk, "DEFAULTS")

        # ── Variant C: Conservative defaults ──
        result_conservative = run_variant(strategy, data, timeframe, DEFAULTS_CONSERVATIVE, settings, risk, "CONSERVATIVE")

        # ── Collect ──
        combo_result = {
            "symbol": symbol,
            "strategy": strategy_name,
            "timeframe": timeframe,
            "meta_params": meta_params,
            "defaults_params": DEFAULTS,
            "conservative_params": DEFAULTS_CONSERVATIVE,
            "meta": {
                "sharpe": result_meta.metrics.get("sharpe", 0),
                "sortino": result_meta.metrics.get("sortino", 0),
                "total_return": result_meta.metrics.get("total_return", 0),
                "max_drawdown": result_meta.metrics.get("max_drawdown", 0),
                "win_rate": result_meta.metrics.get("win_rate", 0),
                "profit_factor": result_meta.metrics.get("profit_factor", 0),
                "n_trades": result_meta.metrics.get("n_trades", 0),
                "composite": composite_score(result_meta.metrics, COMPOSITE_WEIGHTS),
                "robust_sharpe_std": result_meta.metrics.get("robust_sharpe_std", 0),
                "robust_sharpe_min": result_meta.metrics.get("robust_sharpe_min", 0),
                "robust_sharpe_max": result_meta.metrics.get("robust_sharpe_max", 0),
                "robust_consistency": result_meta.metrics.get("robust_consistency", 0),
            },
            "defaults": {
                "sharpe": result_defaults.metrics.get("sharpe", 0),
                "sortino": result_defaults.metrics.get("sortino", 0),
                "total_return": result_defaults.metrics.get("total_return", 0),
                "max_drawdown": result_defaults.metrics.get("max_drawdown", 0),
                "win_rate": result_defaults.metrics.get("win_rate", 0),
                "profit_factor": result_defaults.metrics.get("profit_factor", 0),
                "n_trades": result_defaults.metrics.get("n_trades", 0),
                "composite": composite_score(result_defaults.metrics, COMPOSITE_WEIGHTS),
                "robust_sharpe_std": result_defaults.metrics.get("robust_sharpe_std", 0),
                "robust_sharpe_min": result_defaults.metrics.get("robust_sharpe_min", 0),
                "robust_sharpe_max": result_defaults.metrics.get("robust_sharpe_max", 0),
                "robust_consistency": result_defaults.metrics.get("robust_consistency", 0),
            },
            "conservative": {
                "sharpe": result_conservative.metrics.get("sharpe", 0),
                "sortino": result_conservative.metrics.get("sortino", 0),
                "total_return": result_conservative.metrics.get("total_return", 0),
                "max_drawdown": result_conservative.metrics.get("max_drawdown", 0),
                "win_rate": result_conservative.metrics.get("win_rate", 0),
                "profit_factor": result_conservative.metrics.get("profit_factor", 0),
                "n_trades": result_conservative.metrics.get("n_trades", 0),
                "composite": composite_score(result_conservative.metrics, COMPOSITE_WEIGHTS),
                "robust_sharpe_std": result_conservative.metrics.get("robust_sharpe_std", 0),
                "robust_sharpe_min": result_conservative.metrics.get("robust_sharpe_min", 0),
                "robust_sharpe_max": result_conservative.metrics.get("robust_sharpe_max", 0),
                "robust_consistency": result_conservative.metrics.get("robust_consistency", 0),
            },
        }

        # Winner
        sharpes = {
            "meta": combo_result["meta"]["sharpe"],
            "defaults": combo_result["defaults"]["sharpe"],
            "conservative": combo_result["conservative"]["sharpe"],
        }
        combo_result["winner_sharpe"] = max(sharpes, key=sharpes.get)

        composites = {
            "meta": combo_result["meta"]["composite"],
            "defaults": combo_result["defaults"]["composite"],
            "conservative": combo_result["conservative"]["composite"],
        }
        combo_result["winner_composite"] = max(composites, key=composites.get)

        results.append(combo_result)

        logger.info(f"\n  📊 {symbol}/{strategy_name}:")
        logger.info(f"    META:         Sharpe={combo_result['meta']['sharpe']:.3f}  "
                     f"Return={combo_result['meta']['total_return']:.2%}  "
                     f"DD={combo_result['meta']['max_drawdown']:.2%}  "
                     f"std={combo_result['meta']['robust_sharpe_std']:.3f}")
        logger.info(f"    DEFAULTS:     Sharpe={combo_result['defaults']['sharpe']:.3f}  "
                     f"Return={combo_result['defaults']['total_return']:.2%}  "
                     f"DD={combo_result['defaults']['max_drawdown']:.2%}  "
                     f"std={combo_result['defaults']['robust_sharpe_std']:.3f}")
        logger.info(f"    CONSERVATIVE: Sharpe={combo_result['conservative']['sharpe']:.3f}  "
                     f"Return={combo_result['conservative']['total_return']:.2%}  "
                     f"DD={combo_result['conservative']['max_drawdown']:.2%}  "
                     f"std={combo_result['conservative']['robust_sharpe_std']:.3f}")
        logger.info(f"    → Winner (Sharpe): {combo_result['winner_sharpe'].upper()}")

    # ── Aggregate ──
    elapsed = time.time() - start_time
    meta_sharpes = [r["meta"]["sharpe"] for r in results]
    default_sharpes = [r["defaults"]["sharpe"] for r in results]
    conservative_sharpes = [r["conservative"]["sharpe"] for r in results]

    meta_wins_sharpe = sum(1 for r in results if r["winner_sharpe"] == "meta")
    default_wins_sharpe = sum(1 for r in results if r["winner_sharpe"] == "defaults")
    conservative_wins_sharpe = sum(1 for r in results if r["winner_sharpe"] == "conservative")

    meta_wins_composite = sum(1 for r in results if r["winner_composite"] == "meta")
    default_wins_composite = sum(1 for r in results if r["winner_composite"] == "defaults")
    conservative_wins_composite = sum(1 for r in results if r["winner_composite"] == "conservative")

    summary = {
        "n_combos": len(results),
        "n_seeds": N_SEEDS,
        "elapsed_seconds": round(elapsed, 1),
        "meta_avg_sharpe": round(float(np.mean(meta_sharpes)), 4),
        "defaults_avg_sharpe": round(float(np.mean(default_sharpes)), 4),
        "conservative_avg_sharpe": round(float(np.mean(conservative_sharpes)), 4),
        "meta_wins_sharpe": meta_wins_sharpe,
        "defaults_wins_sharpe": default_wins_sharpe,
        "conservative_wins_sharpe": conservative_wins_sharpe,
        "meta_wins_composite": meta_wins_composite,
        "defaults_wins_composite": default_wins_composite,
        "conservative_wins_composite": conservative_wins_composite,
        "sharpe_diff_meta_vs_defaults": round(float(np.mean(meta_sharpes)) - float(np.mean(default_sharpes)), 4),
    }

    # Verdict
    if summary["sharpe_diff_meta_vs_defaults"] > 0.1:
        verdict = "META-OPT apporte un gain significatif"
    elif summary["sharpe_diff_meta_vs_defaults"] > 0.02:
        verdict = "META-OPT apporte un gain marginal"
    elif summary["sharpe_diff_meta_vs_defaults"] > -0.02:
        verdict = "PAS DE DIFFÉRENCE significative"
    else:
        verdict = "DEFAULTS font MIEUX que la méta-optimisation"
    summary["verdict"] = verdict

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  RÉSULTATS TEST A/B ({len(results)} combos × {N_SEEDS} seeds)")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Avg Sharpe META:         {summary['meta_avg_sharpe']:.4f}")
    logger.info(f"  Avg Sharpe DEFAULTS:     {summary['defaults_avg_sharpe']:.4f}")
    logger.info(f"  Avg Sharpe CONSERVATIVE: {summary['conservative_avg_sharpe']:.4f}")
    logger.info(f"  Diff (meta - defaults):  {summary['sharpe_diff_meta_vs_defaults']:.4f}")
    logger.info(f"  Wins (Sharpe):  META={meta_wins_sharpe}  DEFAULTS={default_wins_sharpe}  CONSERVATIVE={conservative_wins_sharpe}")
    logger.info(f"  Wins (Composite): META={meta_wins_composite}  DEFAULTS={default_wins_composite}  CONSERVATIVE={conservative_wins_composite}")
    logger.info(f"  ⏱ Durée: {elapsed/60:.1f} min")
    logger.info(f"\n  🎯 VERDICT: {verdict}")
    logger.info(f"{'=' * 70}")

    # ── Save JSON ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "n_seeds": N_SEEDS,
        "defaults_params": DEFAULTS,
        "conservative_params": DEFAULTS_CONSERVATIVE,
        "summary": summary,
        "combos": results,
    }

    json_path = f"results/test_ab_meta_vs_defaults_{timestamp}.json"
    Path("results").mkdir(exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved: {json_path}")

    # ── Generate Markdown report ──
    md_path = "docs/results/07_test_ab_meta_vs_defaults.md"
    Path("docs/results").mkdir(parents=True, exist_ok=True)

    md = []
    md.append("# Test A/B : Méta-Optimisation vs Defaults Fixes")
    md.append(f"**Date** : {datetime.now().strftime('%d %B %Y (%H:%M)')}")
    md.append(f"**Seeds par variante** : {N_SEEDS}")
    md.append(f"**Durée** : {elapsed/60:.1f} min")
    md.append(f"**Statut** : ✅ TERMINÉ")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Protocole")
    md.append("")
    md.append("Pour chaque combo (5 profils méta-optimisés V3) :")
    md.append("- **META** : walk-forward avec les meta-params trouvés par Optuna")
    md.append("- **DEFAULTS** : walk-forward avec defaults fixes (reoptim=3M, window=1Y, bounds=1.0, metric=sharpe, trials=100)")
    md.append("- **CONSERVATIVE** : walk-forward avec defaults conservateurs (reoptim=6M, window=2Y, bounds=0.8, metric=sharpe, trials=100)")
    md.append("")
    md.append("Chaque variante est testée avec `run_walk_forward_robust` (5 seeds, médiane) pour éliminer la variance.")
    md.append("")

    md.append("## Résultats par combo")
    md.append("")
    md.append("| Combo | Variante | Sharpe | Return | DD | PF | Trades | Sharpe std | Consistency |")
    md.append("|-------|----------|--------|--------|-----|-----|--------|------------|-------------|")

    for r in results:
        combo = f"{r['symbol']}/{r['strategy']}"
        for variant_name in ["meta", "defaults", "conservative"]:
            v = r[variant_name]
            winner_mark = " **←**" if r["winner_sharpe"] == variant_name else ""
            md.append(
                f"| {combo} | {variant_name.upper()}{winner_mark} | "
                f"{v['sharpe']:.3f} | {v['total_return']:.1%} | "
                f"{v['max_drawdown']:.1%} | {v['profit_factor']:.2f} | "
                f"{v['n_trades']} | {v['robust_sharpe_std']:.3f} | "
                f"{v['robust_consistency']:.2f} |"
            )
        combo = ""  # Don't repeat combo name

    md.append("")
    md.append("## Résumé agrégé")
    md.append("")
    md.append("| Métrique | META | DEFAULTS | CONSERVATIVE |")
    md.append("|----------|------|----------|--------------|")
    md.append(f"| Avg Sharpe | {summary['meta_avg_sharpe']:.4f} | {summary['defaults_avg_sharpe']:.4f} | {summary['conservative_avg_sharpe']:.4f} |")
    md.append(f"| Wins (Sharpe) | {meta_wins_sharpe}/5 | {default_wins_sharpe}/5 | {conservative_wins_sharpe}/5 |")
    md.append(f"| Wins (Composite) | {meta_wins_composite}/5 | {default_wins_composite}/5 | {conservative_wins_composite}/5 |")
    md.append("")

    md.append("## Analyse de la variance (robustesse)")
    md.append("")
    md.append("| Combo | META std | DEFAULTS std | CONSERVATIVE std |")
    md.append("|-------|---------|-------------|-----------------|")
    for r in results:
        combo = f"{r['symbol']}/{r['strategy']}"
        md.append(f"| {combo} | {r['meta']['robust_sharpe_std']:.3f} | "
                  f"{r['defaults']['robust_sharpe_std']:.3f} | "
                  f"{r['conservative']['robust_sharpe_std']:.3f} |")

    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(f"**{verdict}**")
    md.append("")

    if summary["sharpe_diff_meta_vs_defaults"] > 0.1:
        md.append("La méta-optimisation apporte un gain significatif par rapport aux defaults.")
        md.append("→ **Continuer** avec la méta-optimisation, mais stabiliser avec multi-seed.")
    elif summary["sharpe_diff_meta_vs_defaults"] > 0.02:
        md.append("La méta-optimisation apporte un gain marginal.")
        md.append("→ **Simplifier** : utiliser des defaults sensibles et investir le temps de compute ailleurs.")
    elif summary["sharpe_diff_meta_vs_defaults"] > -0.02:
        md.append("Pas de différence significative entre méta-opt et defaults.")
        md.append("→ **Abandonner** la méta-optimisation pour l'instant. Utiliser des defaults fixes.")
    else:
        md.append("Les defaults font mieux que la méta-optimisation !")
        md.append("→ **Abandonner** la méta-optimisation. Elle introduit du bruit sans valeur ajoutée.")

    md.append("")
    md.append("## Implications")
    md.append("")
    md.append("### Si méta-opt gagne")
    md.append("- Garder la boucle externe mais avec multi-seed obligatoire")
    md.append("- Considérer grid search exhaustif (espace petit)")
    md.append("- Augmenter n_seeds à 5-10 pour réduire la variance")
    md.append("")
    md.append("### Si defaults gagnent")
    md.append("- Simplifier le pipeline : Diagnostic → WF avec defaults → Portfolio")
    md.append("- Économiser ~80% du temps de compute")
    md.append("- Focus sur : plus de stratégies, portfolio optimization, features")
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"*Généré le {datetime.now().strftime('%d %B %Y')}*")

    with open(md_path, "w") as f:
        f.write("\n".join(md))
    logger.info(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
