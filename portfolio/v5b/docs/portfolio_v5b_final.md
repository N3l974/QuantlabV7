# Portfolio V5b — Multi-Profil + Audit + Confiance Live
**Date** : 13 February 2026 (08:59)
**Diagnostic** : diagnostic_v5b_20260213_015039.json
**Durée** : 0.1 min

---

## Vue d'ensemble — 3 Profils de Risque

| Profil | Risk/Trade | Max Pos | Sharpe | Sortino | Return | Max DD | Calmar | Confiance |
|--------|-----------|---------|--------|---------|--------|--------|--------|-----------|
| **Conservateur** | 0.0% | 10% | 2.48 | 4.66 | 2.9% | -0.6% | 4.37 | **95/100** GO ✅ |
| **Modéré** | 0.0% | 25% | 2.48 | 4.64 | 7.4% | -1.6% | 4.37 | **95/100** GO ✅ |
| **Agressif** | 0.0% | 50% | 2.49 | 4.60 | 15.1% | -3.2% | 4.39 | **95/100** GO ✅ |

---

## Profil Conservateur
*Position max 10% du capital par trade.*

### Performance

| Métrique | Valeur | Objectif | Status |
|----------|--------|----------|--------|
| Return | 2.9% | ≥15% | ❌ |
| Sharpe | 2.48 | ≥1.0 | ✅ |
| Sortino | 4.66 | ≥1.0 | ✅ |
| Max DD | -0.6% | ≥-5% | ✅ |
| Calmar | 4.37 | ≥1.0 | ✅ |
| Risk/Trade | 0.0% | — | — |
| Max Position | 10% | — | — |
| Circuit Breaker | 10% | — | — |

### Allocations (8 combos)

| Poids | Symbol | Stratégie | TF | Risk | Ov | Sharpe | Return | DD | Seed_std |
|-------|--------|-----------|-----|------|-----|--------|--------|-----|----------|
| 25.0% | ETHUSDT | breakout_regime | 1d | r2.0 | - | 1.803 | 2.4% | -0.1% | 0.173 |
| 25.0% | SOLUSDT | mtf_momentum_breakout | 1d | flat | - | 1.657 | 9.7% | -3.5% | 0.224 |
| 20.0% | ETHUSDT | keltner_channel | 4h | r2.0 | - | 1.515 | 5.5% | -2.0% | 0.221 |
| 18.0% | SOLUSDT | atr_volatility_breakout | 1d | flat | Y | 1.592 | 6.4% | -1.9% | 0.307 |
| 4.7% | ETHUSDT | trend_multi_factor | 1d | r2.0 | - | 1.620 | 12.3% | -4.7% | 0.387 |
| 3.3% | ETHUSDT | supertrend | 1d | flat | Y | 1.691 | 8.6% | -1.9% | 0.081 |
| 2.0% | ETHUSDT | supertrend_adx | 1d | r1.0 | Y | 1.409 | 1.5% | -0.6% | 0.123 |
| 2.0% | BTCUSDT | bollinger_breakout | 1d | r1.0 | Y | 1.378 | 3.0% | -1.2% | 0.022 |

### Répartition par actif

| Actif | Allocation |
|-------|-----------|
| ETHUSDT | 55.0% |
| SOLUSDT | 43.0% |
| BTCUSDT | 2.0% |

### Audit de fiabilité

**Rolling Sharpe (60 jours)**

| Métrique | Valeur |
|----------|--------|
| Moyenne | 1.86 |
| Écart-type | 2.35 |
| Min / Max | -3.39 / 6.48 |
| % positif | 83% |
| 1ère moitié | 3.05 |
| 2ème moitié | 0.68 |

**Analyse mensuelle**

| Métrique | Valeur |
|----------|--------|
| Pire mois | -0.1% |
| Meilleur mois | 1.9% |
| Mois moyen | 0.23% |
| Mois positifs | 62% |

**Stress tests**

| Métrique | Valeur |
|----------|--------|
| Max losing streak | 8 bars |
| Recovery from max DD | 1 bars |
| VaR 95% | -0.05% |
| CVaR 95% | -0.09% |
| Skewness | 4.19 |
| Kurtosis | 34.37 |

**Concentration**

| Métrique | Valeur |
|----------|--------|
| HHI Symbol | 0.488 |
| N effectif symbols | 2.1 |
| HHI Stratégie | 0.202 |
| N effectif stratégies | 5.0 |

**Corrélation intra-portfolio**

| Métrique | Valeur |
|----------|--------|
| Corrélation abs moyenne | 0.243 |
| Corrélation max | 1.000 |

### Monte Carlo ($10,000 — 5000 sims)

| Horizon | P5 | Médian | P95 | P(>0) | P(>10%) | P(ruine) | DD médian |
|---------|-----|--------|-----|-------|---------|----------|-----------|
| 3M | $9,977 | $10,032 | $10,178 | 79% | 0% | 0.0% | -0.2% |
| 6M | $9,983 | $10,076 | $10,268 | 90% | 0% | 0.0% | -0.3% |
| 12M | $10,021 | $10,177 | $10,423 | 97% | 0% | 0.0% | -0.5% |
| 24M | $10,120 | $10,365 | $10,712 | 100% | 0% | 0.0% | -0.6% |
| 36M | $10,244 | $10,562 | $10,968 | 100% | 4% | 0.0% | -0.6% |

### Score de confiance déploiement live

**Score : 95/100 — GO ✅**

| Critère | Points | Status |
|---------|--------|--------|
| Sharpe ≥ 1.5 | 15/15 | ✅ |
| Sortino ≥ 1.5 | 10/10 | ✅ |
| DD < -5% | 15/15 | ✅ |
| Rolling Sharpe >0 ≥ 70% | 10/10 | ✅ |
| Both halves Sharpe > 0 | 10/15 | ✅ |
| Mois positifs ≥ 60% | 10/10 | ✅ |
| N_eff symbols ≥ 1.5 | 5/10 | ⚠️ |
| MC P(gain 12M) ≥ 90% | 10/10 | ✅ |
| MC P(ruine) ≤ 1% | 5/5 | ✅ |
| Multi-seed 3 validé (STRONG) | 5/5 | ✅ |

### Features V5b

| Feature | Utilisation |
|---------|------------|
| Trailing stop | 8/8 |
| Breakeven | 8/8 |
| Max holding | 8/8 |
| Risk sizing | 5/8 |
| Overlay | 4/8 |

---

## Profil Modéré
*Position max 25% du capital par trade.*

### Performance

| Métrique | Valeur | Objectif | Status |
|----------|--------|----------|--------|
| Return | 7.4% | ≥15% | ❌ |
| Sharpe | 2.48 | ≥1.0 | ✅ |
| Sortino | 4.64 | ≥1.0 | ✅ |
| Max DD | -1.6% | ≥-10% | ✅ |
| Calmar | 4.37 | ≥1.0 | ✅ |
| Risk/Trade | 0.0% | — | — |
| Max Position | 25% | — | — |
| Circuit Breaker | 15% | — | — |

### Allocations (8 combos)

| Poids | Symbol | Stratégie | TF | Risk | Ov | Sharpe | Return | DD | Seed_std |
|-------|--------|-----------|-----|------|-----|--------|--------|-----|----------|
| 25.0% | ETHUSDT | breakout_regime | 1d | r2.0 | - | 1.803 | 2.4% | -0.1% | 0.173 |
| 25.0% | SOLUSDT | mtf_momentum_breakout | 1d | flat | - | 1.657 | 9.7% | -3.5% | 0.224 |
| 20.0% | ETHUSDT | keltner_channel | 4h | r2.0 | - | 1.515 | 5.5% | -2.0% | 0.221 |
| 18.0% | SOLUSDT | atr_volatility_breakout | 1d | flat | Y | 1.592 | 6.4% | -1.9% | 0.307 |
| 4.7% | ETHUSDT | trend_multi_factor | 1d | r2.0 | - | 1.620 | 12.3% | -4.7% | 0.387 |
| 3.3% | ETHUSDT | supertrend | 1d | flat | Y | 1.691 | 8.6% | -1.9% | 0.081 |
| 2.0% | ETHUSDT | supertrend_adx | 1d | r1.0 | Y | 1.409 | 1.5% | -0.6% | 0.123 |
| 2.0% | BTCUSDT | bollinger_breakout | 1d | r1.0 | Y | 1.378 | 3.0% | -1.2% | 0.022 |

### Répartition par actif

| Actif | Allocation |
|-------|-----------|
| ETHUSDT | 55.0% |
| SOLUSDT | 43.0% |
| BTCUSDT | 2.0% |

### Audit de fiabilité

**Rolling Sharpe (60 jours)**

| Métrique | Valeur |
|----------|--------|
| Moyenne | 1.87 |
| Écart-type | 2.35 |
| Min / Max | -3.36 / 6.48 |
| % positif | 83% |
| 1ère moitié | 3.05 |
| 2ème moitié | 0.69 |

**Analyse mensuelle**

| Métrique | Valeur |
|----------|--------|
| Pire mois | -0.3% |
| Meilleur mois | 4.7% |
| Mois moyen | 0.56% |
| Mois positifs | 62% |

**Stress tests**

| Métrique | Valeur |
|----------|--------|
| Max losing streak | 8 bars |
| Recovery from max DD | 1 bars |
| VaR 95% | -0.12% |
| CVaR 95% | -0.23% |
| Skewness | 4.10 |
| Kurtosis | 33.09 |

**Concentration**

| Métrique | Valeur |
|----------|--------|
| HHI Symbol | 0.488 |
| N effectif symbols | 2.1 |
| HHI Stratégie | 0.202 |
| N effectif stratégies | 5.0 |

**Corrélation intra-portfolio**

| Métrique | Valeur |
|----------|--------|
| Corrélation abs moyenne | 0.240 |
| Corrélation max | 1.000 |

### Monte Carlo ($10,000 — 5000 sims)

| Horizon | P5 | Médian | P95 | P(>0) | P(>10%) | P(ruine) | DD médian |
|---------|-----|--------|-----|-------|---------|----------|-----------|
| 3M | $9,945 | $10,077 | $10,434 | 79% | 0% | 0.0% | -0.5% |
| 6M | $9,964 | $10,194 | $10,644 | 91% | 0% | 0.0% | -0.8% |
| 12M | $10,048 | $10,433 | $11,085 | 97% | 7% | 0.0% | -1.1% |
| 24M | $10,301 | $10,933 | $11,830 | 100% | 44% | 0.0% | -1.4% |
| 36M | $10,615 | $11,453 | $12,567 | 100% | 79% | 0.0% | -1.5% |

### Score de confiance déploiement live

**Score : 95/100 — GO ✅**

| Critère | Points | Status |
|---------|--------|--------|
| Sharpe ≥ 1.5 | 15/15 | ✅ |
| Sortino ≥ 1.5 | 10/10 | ✅ |
| DD < -5% | 15/15 | ✅ |
| Rolling Sharpe >0 ≥ 70% | 10/10 | ✅ |
| Both halves Sharpe > 0 | 10/15 | ✅ |
| Mois positifs ≥ 60% | 10/10 | ✅ |
| N_eff symbols ≥ 1.5 | 5/10 | ⚠️ |
| MC P(gain 12M) ≥ 90% | 10/10 | ✅ |
| MC P(ruine) ≤ 1% | 5/5 | ✅ |
| Multi-seed 3 validé (STRONG) | 5/5 | ✅ |

### Features V5b

| Feature | Utilisation |
|---------|------------|
| Trailing stop | 8/8 |
| Breakeven | 8/8 |
| Max holding | 8/8 |
| Risk sizing | 5/8 |
| Overlay | 4/8 |

---

## Profil Agressif
*Position max 50% du capital par trade.*

### Performance

| Métrique | Valeur | Objectif | Status |
|----------|--------|----------|--------|
| Return | 15.1% | ≥15% | ✅ |
| Sharpe | 2.49 | ≥1.0 | ✅ |
| Sortino | 4.60 | ≥1.0 | ✅ |
| Max DD | -3.2% | ≥-15% | ✅ |
| Calmar | 4.39 | ≥1.0 | ✅ |
| Risk/Trade | 0.0% | — | — |
| Max Position | 50% | — | — |
| Circuit Breaker | 25% | — | — |

### Allocations (8 combos)

| Poids | Symbol | Stratégie | TF | Risk | Ov | Sharpe | Return | DD | Seed_std |
|-------|--------|-----------|-----|------|-----|--------|--------|-----|----------|
| 25.0% | ETHUSDT | breakout_regime | 1d | r2.0 | - | 1.803 | 2.4% | -0.1% | 0.173 |
| 25.0% | SOLUSDT | mtf_momentum_breakout | 1d | flat | - | 1.657 | 9.7% | -3.5% | 0.224 |
| 20.0% | ETHUSDT | keltner_channel | 4h | r2.0 | - | 1.515 | 5.5% | -2.0% | 0.221 |
| 18.0% | SOLUSDT | atr_volatility_breakout | 1d | flat | Y | 1.592 | 6.4% | -1.9% | 0.307 |
| 4.7% | ETHUSDT | trend_multi_factor | 1d | r2.0 | - | 1.620 | 12.3% | -4.7% | 0.387 |
| 3.3% | ETHUSDT | supertrend | 1d | flat | Y | 1.691 | 8.6% | -1.9% | 0.081 |
| 2.0% | ETHUSDT | supertrend_adx | 1d | r1.0 | Y | 1.409 | 1.5% | -0.6% | 0.123 |
| 2.0% | BTCUSDT | bollinger_breakout | 1d | r1.0 | Y | 1.378 | 3.0% | -1.2% | 0.022 |

### Répartition par actif

| Actif | Allocation |
|-------|-----------|
| ETHUSDT | 55.0% |
| SOLUSDT | 43.0% |
| BTCUSDT | 2.0% |

### Audit de fiabilité

**Rolling Sharpe (60 jours)**

| Métrique | Valeur |
|----------|--------|
| Moyenne | 1.88 |
| Écart-type | 2.34 |
| Min / Max | -3.33 / 6.47 |
| % positif | 83% |
| 1ère moitié | 3.06 |
| 2ème moitié | 0.72 |

**Analyse mensuelle**

| Métrique | Valeur |
|----------|--------|
| Pire mois | -0.7% |
| Meilleur mois | 9.3% |
| Mois moyen | 1.12% |
| Mois positifs | 62% |

**Stress tests**

| Métrique | Valeur |
|----------|--------|
| Max losing streak | 8 bars |
| Recovery from max DD | 1 bars |
| VaR 95% | -0.23% |
| CVaR 95% | -0.45% |
| Skewness | 3.95 |
| Kurtosis | 31.16 |

**Concentration**

| Métrique | Valeur |
|----------|--------|
| HHI Symbol | 0.488 |
| N effectif symbols | 2.1 |
| HHI Stratégie | 0.202 |
| N effectif stratégies | 5.0 |

**Corrélation intra-portfolio**

| Métrique | Valeur |
|----------|--------|
| Corrélation abs moyenne | 0.235 |
| Corrélation max | 1.000 |

### Monte Carlo ($10,000 — 5000 sims)

| Horizon | P5 | Médian | P95 | P(>0) | P(>10%) | P(ruine) | DD médian |
|---------|-----|--------|-----|-------|---------|----------|-----------|
| 3M | $9,893 | $10,164 | $10,876 | 80% | 3% | 0.0% | -1.0% |
| 6M | $9,925 | $10,392 | $11,373 | 90% | 13% | 0.0% | -1.6% |
| 12M | $10,089 | $10,870 | $12,170 | 97% | 42% | 0.0% | -2.2% |
| 24M | $10,618 | $11,939 | $13,991 | 100% | 86% | 0.0% | -2.7% |
| 36M | $11,221 | $13,061 | $15,757 | 100% | 97% | 0.0% | -3.0% |

### Score de confiance déploiement live

**Score : 95/100 — GO ✅**

| Critère | Points | Status |
|---------|--------|--------|
| Sharpe ≥ 1.5 | 15/15 | ✅ |
| Sortino ≥ 1.5 | 10/10 | ✅ |
| DD < -5% | 15/15 | ✅ |
| Rolling Sharpe >0 ≥ 70% | 10/10 | ✅ |
| Both halves Sharpe > 0 | 10/15 | ✅ |
| Mois positifs ≥ 60% | 10/10 | ✅ |
| N_eff symbols ≥ 1.5 | 5/10 | ⚠️ |
| MC P(gain 12M) ≥ 90% | 10/10 | ✅ |
| MC P(ruine) ≤ 1% | 5/5 | ✅ |
| Multi-seed 3 validé (STRONG) | 5/5 | ✅ |

### Features V5b

| Feature | Utilisation |
|---------|------------|
| Trailing stop | 8/8 |
| Breakeven | 8/8 |
| Max holding | 8/8 |
| Risk sizing | 5/8 |
| Overlay | 4/8 |

---

## Comparaison historique

| Métrique | V3b | V4 | V4b | V5b Conserv. | V5b Modéré | V5b Agressif |
|----------|-----|-----|-----|-------------|------------|--------------|
| Return | +9.8% | +4.9% | +19.8% | +2.9% | +7.4% | +15.1% |
| Sharpe | 1.19 | 2.59 | 1.35 | 2.48 | 2.48 | 2.49 |
| Max DD | -4.9% | -0.8% | -8.5% | -0.6% | -1.6% | -3.2% |
| Calmar | 1.91 | 5.99 | 2.17 | 4.37 | 4.37 | 4.39 |

## Recommandation

Le profil **Conservateur** obtient le meilleur score de confiance (**95/100**) et est recommandé pour le déploiement live.

> **GO pour déploiement live** — Tous les critères majeurs sont satisfaits.

---
*Généré le 13 February 2026 08:59*