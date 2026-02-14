# Test Multi-Factor Strategies — Holdout Validation
**Date** : 12 février 2026 (17:19)
**Durée** : 21.8 min
**Cutoff** : 2025-02-01 (12 mois de holdout)
**Seeds** : 3 par combo
**Statut** : ✅ TERMINÉ

---

## Contexte

Le holdout V4 (doc 09) a montré que les stratégies à indicateur unique sont fragiles :
- 1 seul STRONG survivant (ETH/supertrend/4h)
- 2 WEAK, 2 FAIL sur 5 MEDIUM testés

**Décision** : créer des stratégies multi-factor combinant trend + volume + régime.

## 3 nouvelles stratégies créées

| Stratégie | Type | Composants | Fichier |
|-----------|------|------------|---------|
| **SuperTrend + ADX** | Trend + Regime | SuperTrend direction + ADX filter (trending only) + cooldown | `strategies/supertrend_adx.py` |
| **Trend Multi-Factor** | Trend + Volume + Momentum | SuperTrend + OBV slope + ROC momentum (confluence 3/3) | `strategies/trend_multi_factor.py` |
| **Breakout + Regime** | Breakout + Regime + Volume | ATR breakout + ADX filter + volume spike confirmation + cooldown | `strategies/breakout_regime.py` |

## Résumé

- **Combos testés** : 18 (3 stratégies × 3 symbols × 2 TFs)
- **STRONG** : 3
- **WEAK** : 5
- **FAIL** : 10

## Résultats détaillés (triés par HO Sharpe)

| # | Verdict | Symbol | Stratégie | TF | IS Sharpe | HO Sharpe | HO Return | Trades | HO Sharpe std |
|---|---------|--------|-----------|-----|-----------|-----------|-----------|--------|---------------|
| 1 | ✅ STRONG | ETHUSDT | breakout_regime | 4h | -0.265 | **0.935** | +7.2% | 20 | 0.47 |
| 2 | ⚠️ WEAK | ETHUSDT | trend_multi_factor | 1d | -0.250 | 0.779 | +11.6% | 13 | 0.86 |
| 3 | ⚠️ WEAK | ETHUSDT | supertrend_adx | 4h | -0.368 | 0.694 | +8.2% | 26 | 1.10 |
| 4 | ⚠️ WEAK | BTCUSDT | trend_multi_factor | 1d | -0.249 | 0.321 | +2.8% | 7 | 0.54 |
| 5 | ⚠️ WEAK | BTCUSDT | supertrend_adx | 4h | -0.023 | 0.194 | +1.5% | 19 | 0.65 |
| 6 | ✅ STRONG | ETHUSDT | trend_multi_factor | 4h | -0.295 | **0.180** | +1.6% | 22 | **0.13** |
| 7 | ✅ STRONG | SOLUSDT | breakout_regime | 1d | -0.094 | **0.015** | -0.2% | 12 | **0.02** |
| 8 | ⚠️ WEAK | ETHUSDT | breakout_regime | 1d | 0.298 | 0.000 | 0.0% | 5 | 0.41 |
| 9 | ❌ FAIL | BTCUSDT | supertrend_adx | 1d | -0.416 | -0.324 | -2.6% | 10 | 0.01 |
| 10 | ❌ FAIL | SOLUSDT | trend_multi_factor | 4h | 0.294 | -0.370 | -6.0% | 26 | 0.04 |
| 11-18 | ❌ FAIL | ... | ... | ... | ... | < -0.4 | ... | ... | ... |

## Comparaison : Simples (V4) vs Multi-factor

| Métrique | Simples (5 MEDIUM V4) | Multi-factor (18 combos) |
|----------|----------------------|--------------------------|
| **STRONG** | 1 | **3** |
| **WEAK** | 2 | **5** |
| **Meilleur HO Sharpe** | 0.444 | **0.935** |
| **Survivants** | 3/5 (60%) | 8/18 (44%) |

Les multi-factor ont un taux de survie plus bas mais produisent des survivants **plus forts**.

## Pool combiné de survivants (11 combos)

| Source | Combo | HO Sharpe | Verdict | HO Sharpe std |
|--------|-------|-----------|---------|---------------|
| Multi-factor | ETH/breakout_regime/4h | **0.935** | STRONG | 0.47 |
| Multi-factor | ETH/trend_multi_factor/1d | 0.779 | WEAK | 0.86 |
| Multi-factor | ETH/supertrend_adx/4h | 0.694 | WEAK | 1.10 |
| V4 simple | ETH/supertrend/4h | 0.444 | STRONG | 0.35 |
| V4 simple | ETH/atr_breakout/1d | 0.397 | WEAK | 0.84 |
| Multi-factor | BTC/trend_multi_factor/1d | 0.321 | WEAK | 0.54 |
| Multi-factor | BTC/supertrend_adx/4h | 0.194 | WEAK | 0.65 |
| Multi-factor | ETH/trend_multi_factor/4h | 0.180 | STRONG | **0.13** |
| Multi-factor | SOL/breakout_regime/1d | 0.015 | STRONG | **0.02** |
| Multi-factor | ETH/breakout_regime/1d | 0.000 | WEAK | 0.41 |
| V4 simple | BTC/momentum_roc/1d | -0.028 | WEAK | 0.43 |

## Observations clés

1. **ETH domine** : 7/11 survivants sont sur ETH — marché le plus "tradable"
2. **4h est le sweet spot** : 6/11 survivants sur 4h
3. **Multi-factor > Simple** : le meilleur multi-factor (0.935) bat le meilleur simple (0.444) de 2x
4. **Les STRONG les plus stables** : ETH/trend_multi_factor/4h (std=0.13) et SOL/breakout_regime/1d (std=0.02)
5. **SOL et BTC difficiles** : la plupart des combos échouent sur ces actifs
6. **IS Sharpe négatif → HO positif** : les combos qui ne sur-fittent pas en IS performent mieux en holdout

## Fichiers générés

- `portfolio/archive/v4/multi_factor_test_20260212_165734.json`
- `strategies/supertrend_adx.py`
- `strategies/trend_multi_factor.py`
- `strategies/breakout_regime.py`
- `strategies/registry.py` (mis à jour, 19 stratégies)

---
*Généré le 12 février 2026*
