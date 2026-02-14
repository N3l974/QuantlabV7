# Holdout Temporal Test — Validation des 5 MEDIUM V4
**Date** : 12 February 2026 (16:35)
**Durée** : 9.3 min
**Cutoff** : 2025-02-01 (12 mois de holdout)
**Seeds** : 5 par combo
**Statut** : ✅ TERMINÉ

---

## Résumé

- **Combos testés** : 5
- **Survivants (strong)** : 1
- **Survivants (weak)** : 2
- **Échecs** : 2

## Résultats détaillés

| # | Verdict | Symbol | Stratégie | TF | IS Sharpe | HO Sharpe | HO Return | HO DD | Dégradation |
|---|---------|--------|-----------|-----|-----------|-----------|-----------|-------|-------------|
| 1 | ✅ STRONG | ETHUSDT | supertrend | 4h | 0.054 | 0.444 | 6.5% | -13.6% | -724% |
| 2 | ⚠️ WEAK | ETHUSDT | atr_volatility_breakout | 1d | 0.314 | 0.397 | 3.0% | -6.0% | -26% |
| 3 | ⚠️ WEAK | BTCUSDT | momentum_roc | 1d | 0.328 | -0.028 | -0.2% | -3.4% | 109% |
| 4 | ❌ FAIL | ETHUSDT | volume_obv | 4h | 0.037 | -0.902 | -3.9% | -5.7% | 2537% |
| 5 | ❌ FAIL | SOLUSDT | bollinger_breakout | 4h | 0.391 | -1.354 | -10.2% | -15.1% | 446% |

## Variance inter-seeds (holdout)

| Combo | HO Sharpe min | HO Sharpe med | HO Sharpe max | HO Sharpe std |
|-------|---------------|---------------|---------------|---------------|
| ETHUSDT/supertrend/4h | -0.247 | 0.444 | 0.704 | 0.348 |
| ETHUSDT/atr_volatility_breakout/1d | -1.188 | 0.397 | 0.952 | 0.837 |
| BTCUSDT/momentum_roc/1d | -0.053 | -0.028 | 1.054 | 0.434 |
| ETHUSDT/volume_obv/4h | -1.215 | -0.902 | 0.109 | 0.469 |
| SOLUSDT/bollinger_breakout/4h | -1.525 | -1.354 | -0.023 | 0.618 |

## Méthodologie

### Principe
1. **Split temporel** : données avant/après le cutoff
2. **In-sample** : walk-forward complet (train/test rolling) sur données pré-cutoff
3. **Holdout** : appliquer les DERNIERS params optimisés sur données post-cutoff
4. **Multi-seed** : 5 seeds indépendants pour robustesse

### Critères de survie
- **STRONG** : HO Sharpe médian > 0 ET HO Sharpe min > -0.3
- **WEAK** : HO Sharpe médian > -0.1
- **FAIL** : HO Sharpe médian ≤ -0.1

## Conclusion

**1 combo(s) survivent fortement** le holdout :
- ETHUSDT/supertrend/4h (HO Sharpe 0.444)

→ Ces combos sont candidats pour le Portfolio V3.

---
*Généré le 12 February 2026*