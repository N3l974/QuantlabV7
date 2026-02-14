# Portfolio FTMO V1 — Rapport de Conception
**Date** : 14 February 2026 (16:41)
**Diagnostic** : diagnostic_v5b_20260213_015039.json
**Mode** : SWING (hold overnight/weekend, no news restriction)
**Durée calcul** : 0.4 min

---

## Règles FTMO 2-Step Challenge

| Phase | Profit Target | Max Daily Loss | Max Total Loss |
|-------|--------------|----------------|----------------|
| Phase 1 | 10% | 5% | 10% |
| Phase 2 | 5% | 5% | 10% |
| Funded | Aucun | 5% | 10% |

## Vue d'ensemble — Profils FTMO

| Profil | Risk/Trade | Max Pos | Sharpe | Return | Max DD | Daily Max | MC Pass | Confiance |
|--------|-----------|---------|--------|--------|--------|-----------|---------|-----------|
| **Challenge Aggressive** | 1.25% | 20% | 3.06 | 1.7% | -0.6% | -0.58% | 0% | **83/100** GO FTMO ✅ |
| **Challenge** | 1.00% | 15% | 2.67 | 1.1% | -0.5% | -0.42% | 0% | **83/100** GO FTMO ✅ |
| **Funded** | 0.75% | 10% | 2.57 | 0.8% | -0.4% | -0.32% | 0% | **83/100** GO FTMO ✅ |

---

## Profil Challenge Aggressive
*Phase 1 orientée passage rapide, garde-fous FTMO conservés*

### Conformité FTMO

| Règle FTMO | Limite | Notre valeur | Marge | Status |
|------------|--------|-------------|-------|--------|
| Max Total Loss | -10% | -0.6% | -9.4% | ✅ |
| Max Daily Loss | -5% | -0.58% | -4.42% | ✅ |

### Performance

| Métrique | Valeur |
|----------|--------|
| Return | 1.7% |
| Sharpe | 3.06 |
| Sortino | 3.03 |
| Max DD | -0.6% |
| Calmar | 16.90 |
| Win Rate | 46% |
| Profit Factor | 2.83 |
| Max Losing Streak | 6 bars |
| Est. jours Phase 1 | 2188 |
| Est. jours Phase 2 | 1094 |

### Allocations

| Poids | Symbol | Stratégie | TF | Sharpe | DD | WR |
|-------|--------|-----------|-----|--------|-----|-----|
| 25.0% | ETHUSDT | supertrend | 1d | 1.691 | -1.9% | 67% |
| 25.0% | BTCUSDT | supertrend_adx | 1d | 0.860 | -2.6% | 50% |
| 25.0% | BTCUSDT | momentum_roc | 1d | 0.901 | -1.1% | 50% |
| 13.0% | ETHUSDT | keltner_channel | 4h | 0.907 | -3.7% | 37% |
| 3.0% | SOLUSDT | mtf_momentum_breakout | 1d | 1.657 | -3.5% | 67% |
| 3.0% | ETHUSDT | breakout_regime | 1d | 1.803 | -0.1% | 100% |
| 3.0% | ETHUSDT | ichimoku_cloud | 4h | 0.947 | -2.0% | 39% |
| 3.0% | SOLUSDT | ichimoku_cloud | 1d | 0.837 | -0.4% | 40% |

### Répartition par actif

- **BTCUSDT** : 50.0%
- **ETHUSDT** : 44.0%
- **SOLUSDT** : 6.0%

### Analyse journalière (FTMO-critique)

| Métrique | Valeur | Limite FTMO |
|----------|--------|------------|
| Pire jour | -0.58% | -5% |
| Jour moyen | 0.005% | — |
| Jours positifs | 27% | — |
| Jours < -3% | 0 | — |
| Jours < -4% | 0 | — |
| Jours < -5% | 0 | 0 (FTMO fail) |

### Monte Carlo FTMO (5000 sims)

**Probabilité de passer le challenge :**

| Scénario | Pass Rate | Fail (DD) | Jours médians | Jours P25-P75 |
|----------|-----------|-----------|--------------|---------------|
| phase1 | **0%** | 0% | 155j | 148-168j |
| phase2 | **15%** | 0% | 150j | 123-166j |
| funded_monthly | **54%** | 0% | 120j | 90-148j |

**Projection funded :**

| Horizon | P5 | Médian | P95 | P(>0) | P(FTMO fail) | DD médian |
|---------|-----|--------|-----|-------|-------------|-----------|
| 3M | $9,992 | $10,131 | $10,365 | 93% | 0.0% | -0.6% |
| 6M | $10,057 | $10,286 | $10,608 | 99% | 0.0% | -0.7% |
| 12M | $10,068 | $10,294 | $10,632 | 99% | 0.0% | -0.8% |

### Score de confiance FTMO

**Score : 83/100 — GO FTMO ✅**

| Critère | Points | Status |
|---------|--------|--------|
| Sharpe ≥ 1.5 | 15 | ✅ |
| DD < -4% | 15 | ✅ |
| Daily loss > -3% | 15 | ✅ |
| Win rate ≥ 45% | 10 | ✅ |
| PF ≥ 1.5 | 10 | ✅ |
| Rolling Sharpe >0 ≥ 65% | 10 | ✅ |
| MC P(pass Phase1) < 60% | 0 | ❌ |
| Multi-seed STRONG | 5 | ✅ |
| Max losing streak ≤ 8 | 3 | ⚠️ |

---

## Profil Challenge
*Phase 1+2 — sizing modéré, DD strict (8% CB)*

### Conformité FTMO

| Règle FTMO | Limite | Notre valeur | Marge | Status |
|------------|--------|-------------|-------|--------|
| Max Total Loss | -10% | -0.5% | -9.5% | ✅ |
| Max Daily Loss | -5% | -0.42% | -4.58% | ✅ |

### Performance

| Métrique | Valeur |
|----------|--------|
| Return | 1.1% |
| Sharpe | 2.67 |
| Sortino | 2.59 |
| Max DD | -0.5% |
| Calmar | 13.35 |
| Win Rate | 45% |
| Profit Factor | 2.58 |
| Max Losing Streak | 6 bars |
| Est. jours Phase 1 | 3430 |
| Est. jours Phase 2 | 1715 |

### Allocations

| Poids | Symbol | Stratégie | TF | Sharpe | DD | WR |
|-------|--------|-----------|-----|--------|-----|-----|
| 25.0% | ETHUSDT | supertrend | 1d | 1.691 | -1.9% | 67% |
| 25.0% | BTCUSDT | supertrend_adx | 1d | 0.860 | -2.6% | 50% |
| 25.0% | BTCUSDT | momentum_roc | 1d | 0.901 | -1.1% | 50% |
| 13.0% | ETHUSDT | keltner_channel | 4h | 0.907 | -3.7% | 37% |
| 3.0% | SOLUSDT | mtf_momentum_breakout | 1d | 1.657 | -3.5% | 67% |
| 3.0% | ETHUSDT | breakout_regime | 1d | 1.803 | -0.1% | 100% |
| 3.0% | ETHUSDT | ichimoku_cloud | 4h | 0.947 | -2.0% | 39% |
| 3.0% | SOLUSDT | ichimoku_cloud | 1d | 0.837 | -0.4% | 40% |

### Répartition par actif

- **BTCUSDT** : 50.0%
- **ETHUSDT** : 44.0%
- **SOLUSDT** : 6.0%

### Analyse journalière (FTMO-critique)

| Métrique | Valeur | Limite FTMO |
|----------|--------|------------|
| Pire jour | -0.42% | -5% |
| Jour moyen | 0.003% | — |
| Jours positifs | 26% | — |
| Jours < -3% | 0 | — |
| Jours < -4% | 0 | — |
| Jours < -5% | 0 | 0 (FTMO fail) |

### Monte Carlo FTMO (5000 sims)

**Probabilité de passer le challenge :**

| Scénario | Pass Rate | Fail (DD) | Jours médians | Jours P25-P75 |
|----------|-----------|-----------|--------------|---------------|
| phase1 | **0%** | 0% | infj | inf-infj |
| phase2 | **2%** | 0% | 163j | 146-174j |
| funded_monthly | **21%** | 0% | 140j | 113-162j |

**Projection funded :**

| Horizon | P5 | Médian | P95 | P(>0) | P(FTMO fail) | DD médian |
|---------|-----|--------|-----|-------|-------------|-----------|
| 3M | $9,981 | $10,084 | $10,238 | 91% | 0.0% | -0.5% |
| 6M | $10,025 | $10,179 | $10,390 | 97% | 0.0% | -0.6% |
| 12M | $10,025 | $10,184 | $10,398 | 97% | 0.0% | -0.6% |

### Score de confiance FTMO

**Score : 83/100 — GO FTMO ✅**

| Critère | Points | Status |
|---------|--------|--------|
| Sharpe ≥ 1.5 | 15 | ✅ |
| DD < -4% | 15 | ✅ |
| Daily loss > -3% | 15 | ✅ |
| Win rate ≥ 45% | 10 | ✅ |
| PF ≥ 1.5 | 10 | ✅ |
| Rolling Sharpe >0 ≥ 65% | 10 | ✅ |
| MC P(pass Phase1) < 60% | 0 | ❌ |
| Multi-seed STRONG | 5 | ✅ |
| Max losing streak ≤ 8 | 3 | ⚠️ |

---

## Profil Funded
*FTMO Account — sizing conservateur, DD très strict (7% CB)*

### Conformité FTMO

| Règle FTMO | Limite | Notre valeur | Marge | Status |
|------------|--------|-------------|-------|--------|
| Max Total Loss | -10% | -0.4% | -9.6% | ✅ |
| Max Daily Loss | -5% | -0.32% | -4.68% | ✅ |

### Performance

| Métrique | Valeur |
|----------|--------|
| Return | 0.8% |
| Sharpe | 2.57 |
| Sortino | 2.49 |
| Max DD | -0.4% |
| Calmar | 12.73 |
| Win Rate | 46% |
| Profit Factor | 2.57 |
| Max Losing Streak | 6 bars |
| Est. jours Phase 1 | 4754 |
| Est. jours Phase 2 | 2377 |

### Allocations

| Poids | Symbol | Stratégie | TF | Sharpe | DD | WR |
|-------|--------|-----------|-----|--------|-----|-----|
| 25.0% | ETHUSDT | supertrend | 1d | 1.691 | -1.9% | 67% |
| 25.0% | BTCUSDT | supertrend_adx | 1d | 0.860 | -2.6% | 50% |
| 25.0% | BTCUSDT | momentum_roc | 1d | 0.901 | -1.1% | 50% |
| 13.0% | ETHUSDT | keltner_channel | 4h | 0.907 | -3.7% | 37% |
| 3.0% | SOLUSDT | mtf_momentum_breakout | 1d | 1.657 | -3.5% | 67% |
| 3.0% | ETHUSDT | breakout_regime | 1d | 1.803 | -0.1% | 100% |
| 3.0% | ETHUSDT | ichimoku_cloud | 4h | 0.947 | -2.0% | 39% |
| 3.0% | SOLUSDT | ichimoku_cloud | 1d | 0.837 | -0.4% | 40% |

### Répartition par actif

- **BTCUSDT** : 50.0%
- **ETHUSDT** : 44.0%
- **SOLUSDT** : 6.0%

### Analyse journalière (FTMO-critique)

| Métrique | Valeur | Limite FTMO |
|----------|--------|------------|
| Pire jour | -0.32% | -5% |
| Jour moyen | 0.002% | — |
| Jours positifs | 26% | — |
| Jours < -3% | 0 | — |
| Jours < -4% | 0 | — |
| Jours < -5% | 0 | 0 (FTMO fail) |

### Monte Carlo FTMO (5000 sims)

**Probabilité de passer le challenge :**

| Scénario | Pass Rate | Fail (DD) | Jours médians | Jours P25-P75 |
|----------|-----------|-----------|--------------|---------------|
| phase1 | **0%** | 0% | infj | inf-infj |
| phase2 | **0%** | 0% | 176j | 146-177j |
| funded_monthly | **5%** | 0% | 151j | 130-166j |

**Projection funded :**

| Horizon | P5 | Médian | P95 | P(>0) | P(FTMO fail) | DD médian |
|---------|-----|--------|-----|-------|-------------|-----------|
| 3M | $9,987 | $10,059 | $10,171 | 90% | 0.0% | -0.4% |
| 6M | $10,011 | $10,127 | $10,282 | 96% | 0.0% | -0.5% |
| 12M | $10,016 | $10,133 | $10,286 | 97% | 0.0% | -0.5% |

### Score de confiance FTMO

**Score : 83/100 — GO FTMO ✅**

| Critère | Points | Status |
|---------|--------|--------|
| Sharpe ≥ 1.5 | 15 | ✅ |
| DD < -4% | 15 | ✅ |
| Daily loss > -3% | 15 | ✅ |
| Win rate ≥ 45% | 10 | ✅ |
| PF ≥ 1.5 | 10 | ✅ |
| Rolling Sharpe >0 ≥ 65% | 10 | ✅ |
| MC P(pass Phase1) < 60% | 0 | ❌ |
| Multi-seed STRONG | 5 | ✅ |
| Max losing streak ≤ 8 | 3 | ⚠️ |

---

## Recommandation FTMO

Le profil **Challenge Aggressive** obtient le meilleur score (**83/100**).

> **GO FTMO** — Lancer le challenge avec ce portfolio.
> Commencer par un Free Trial FTMO pour valider en conditions réelles.

## Prochaines étapes

1. **Free Trial FTMO** (14 jours) — Valider en conditions réelles
2. **Ajuster** si nécessaire (sizing, combos, overlays)
3. **Lancer Phase 1** — FTMO Challenge ($10K-$200K)
4. **Passer Phase 2** — Verification (5% target)
5. **Funded** — Opérer durablement, viser le scaling plan

---
*Généré par Quantlab V7 — Portfolio FTMO V1*