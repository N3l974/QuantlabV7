# Portfolio V5b — Construction & Validation
**Date** : 13 February 2026 (08:38)
**Diagnostic** : diagnostic_v5b_20260213_015039.json
**Durée** : 0.0 min
**Objectif** : +15% annuel, DD < -20%, Sharpe > 1.0

---

## Leçons appliquées (erreurs passées corrigées)

| Erreur passée | Version | Correction V5b |
|---------------|---------|----------------|
| Concentration ETH 95% | V3 | Cap symbol 55% |
| MC sur full data | V3 | MC holdout-only (5000 sims) |
| Sur-diversification (37 combos) | V4 | Concentration top combos |
| Overlays trop agressifs | V4 | Overlay sélectif (seulement si diagnostic meilleur) |
| Markowitz optimise Sharpe pas return | V4 | Multi-objectif (sharpe + return_dd + max_return) |
| Pas de corrélation dedup | V4b | Dedup corr > 0.85 |
| Pas d'overlay du tout | V4b | Overlay sélectif |
| 1 seed seulement | V4 | Multi-seed 3 + seed_std filter |

## Comparaison des variantes

| Portfolio | Sharpe | Sortino | Return | Max DD | Calmar | N |
|-----------|--------|---------|--------|--------|--------|---|
| ✅ C_A1_top3_heavy_2.00x **⭐** | 2.22 | 3.49 | 15.7% | -2.3% | 6.39 | 8 |
| ❌ C_A1_top3_heavy_1.50x | 2.22 | 3.49 | 11.6% | -1.7% | 6.39 | 8 |
| ❌ C_A1_top3_heavy_1.25x | 2.22 | 3.49 | 9.6% | -1.4% | 6.39 | 8 |
| ❌ A2_markowitz_sharpe | 2.36 | 4.09 | 8.1% | -1.7% | 4.61 | 8 |
| ❌ A1_top3_heavy | 2.22 | 3.49 | 7.6% | -1.1% | 6.40 | 8 |
| ❌ B_diversified_12 | 2.41 | 4.31 | 6.3% | -1.2% | 5.05 | 12 |
| ❌ A3_markowitz_retdd | 2.23 | 3.91 | 5.4% | -1.0% | 5.01 | 8 |
| ❌ D_all_deduped | 2.25 | 3.83 | 5.1% | -1.0% | 5.08 | 37 |

## 🏆 Recommandé : C_A1_top3_heavy_2.00x

| Métrique | Valeur | Objectif | Status |
|----------|--------|----------|--------|
| Return | 15.7% | ≥15% | ✅ |
| Max DD | -2.3% | ≥-20% | ✅ |
| Sharpe | 2.22 | ≥1.0 | ✅ |
| Sortino | 3.49 | ≥1.0 | ✅ |
| Calmar | 6.39 | ≥1.0 | ✅ |

### Allocations

| Poids | Symbol | Stratégie | TF | Risk | Ov | HO Sharpe | HO Return | HO DD | Trail | Seed_std |
|-------|--------|-----------|-----|------|-----|-----------|-----------|-------|-------|----------|
| 25.0% | ETHUSDT | breakout_regime | 1d | r2.0 | - | 1.803 | 2.4% | -0.1% | 1.04 | 0.173 |
| 25.0% | ETHUSDT | supertrend | 1d | flat | Y | 1.691 | 8.6% | -1.9% | 3.61 | 0.081 |
| 15.0% | SOLUSDT | mtf_momentum_breakout | 1d | flat | - | 1.657 | 9.7% | -3.5% | 1.66 | 0.224 |
| 7.0% | ETHUSDT | trend_multi_factor | 1d | r2.0 | - | 1.620 | 12.3% | -4.7% | 4.39 | 0.387 |
| 7.0% | SOLUSDT | atr_volatility_breakout | 1d | flat | Y | 1.592 | 6.4% | -1.9% | 3.37 | 0.307 |
| 7.0% | BTCUSDT | trend_multi_factor | 1d | flat | Y | 1.569 | 9.0% | -2.1% | 4.50 | 0.079 |
| 7.0% | ETHUSDT | keltner_channel | 4h | r2.0 | - | 1.515 | 5.5% | -2.0% | 4.57 | 0.221 |
| 7.0% | ETHUSDT | ema_ribbon | 1d | r1.0 | Y | 1.436 | 4.8% | -2.3% | 4.30 | 0.247 |

### Allocation par symbol

| Symbol | Allocation |
|--------|-----------|
| ETHUSDT | 71.0% |
| SOLUSDT | 22.0% |
| BTCUSDT | 7.0% |

### Monte Carlo ($10,000 — 5000 sims)

| Horizon | P5 | P25 | Médian | P75 | P95 | P(>0) | P(>10%) | P(>20%) | P(ruine) |
|---------|-----|-----|--------|-----|-----|-------|---------|---------|----------|
| 3M | $9,904 | $10,034 | $10,138 | $10,292 | $11,036 | 82% | 6% | 0% | 0.0% |
| 6M | $9,954 | $10,160 | $10,343 | $10,685 | $11,511 | 92% | 17% | 2% | 0.0% |
| 12M | $10,101 | $10,467 | $10,828 | $11,441 | $12,562 | 98% | 41% | 11% | 0.0% |
| 24M | $10,606 | $11,283 | $11,986 | $12,809 | $14,433 | 100% | 85% | 50% | 0.0% |
| 36M | $11,165 | $12,198 | $13,135 | $14,247 | $16,394 | 100% | 97% | 80% | 0.0% |

### Comparaison historique

| Métrique | V3b | V4 | V4b | **V5b** |
|----------|-----|-----|-----|---------|
| Return | +9.8% | +4.9% | +19.8% | **+15.7%** |
| Sharpe | 1.19 | 2.59 | 1.35 | **2.22** |
| Max DD | -4.9% | -0.8% | -8.5% | **-2.3%** |
| Calmar | 1.91 | 5.99 | 2.17 | **6.39** |
| ETH % | 95% | 53% | 70% | **71%** |
| Objectif +15% | ❌ | ❌ | ✅ | ✅ |

### Features V5b utilisées

- **Trailing stop** : 8/8 combos
- **Breakeven** : 8/8 combos
- **Max holding** : 8/8 combos
- **Risk-based sizing** : 4/8 combos
- **Overlay** : 4/8 combos

---
*Généré le 13 February 2026 08:38*