# Portfolio V4b — QuantLab V7

## Vue d'ensemble

Portfolio V4b est le portefeuille de trading quantitatif validé de QuantLab V7. Il combine 8 stratégies concentrées sur les meilleurs rendements, avec un leverage modéré de 1.5x.

### Performance clé (12 mois hors-échantillon)

- **Rendement annuel** : +19.8%
- **Sharpe** : 1.35
- **Sortino** : 1.79
- **Max Drawdown** : -8.5%
- **Calmar** : 2.17
- **Probabilité de gain (12M)** : 86%
- **Probabilité de ruine** : 0.0%

## Artefacts techniques

- Code : `portfolio/v4b/code/`
- Résultats : `portfolio/v4b/results/`

## Processus de construction et validation

### Logique de construction

1. Charger les métriques issues du diagnostic et les paramètres retenus.
2. Rejouer chaque combo sur la période holdout avec frais/slippage.
3. Conserver les 8 combos cibles (allocation top3_heavy) et appliquer les poids.
4. Appliquer le leverage 1.5x de manière uniforme au portefeuille agrégé.
5. Produire métriques finales + Monte Carlo + stress tests.

### Protocole train/validation

- **Train (calibration)**: optimisations historiques via pipeline V4 (walk-forward + sélection).
- **Validation finale**: holdout strict non vu pendant calibration.
- Aucun recalibrage sur la période holdout finale (pas de fuite de données).

### Périodes utilisées

- **Fenêtre holdout**: 12 mois, de **fév. 2025 à fév. 2026**.
- **Fenêtres de projection MC**: 3M, 6M, 12M, 24M, 36M.
- **Granularité d'analyse**: barres 1d/4h selon les combos; agrégation portfolio sur l'historique holdout.

## Composition

### Allocations (top3_heavy × 1.5x)

| # | Stratégie | Actif | Horizon | Poids | HO Return | HO DD |
|---|-----------|-------|---------|-------|-----------|-------|
| 1 | SuperTrend | ETH | 1j | 25.0% | +26.7% | -13.3% |
| 2 | Trend Multi-Factor | ETH | 1j | 25.0% | +20.0% | -11.2% |
| 3 | Trend Multi-Factor | SOL | 4h | 15.0% | +12.8% | -11.7% |
| 4 | SuperTrend | BTC | 1j | 10.0% | +1.3% | -12.0% |
| 5 | MACD Crossover | ETH | 1j | 10.0% | +7.4% | -7.8% |
| 6 | Trend Multi-Factor | BTC | 1j | 5.0% | +3.4% | -7.1% |
| 7 | Ichimoku Cloud | ETH | 4h | 5.0% | +1.7% | -10.9% |
| 8 | Bollinger Breakout | ETH | 1j | 5.0% | +6.0% | -7.7% |

### Répartition par actif

| Actif | Allocation |
|-------|-----------|
| **ETH** | **70%** |
| **BTC** | **15%** |
| **SOL** | **15%** |

## Projections Monte Carlo ($10,000)

| Horizon | P5 (pessimiste) | Médian | P95 (optimiste) | P(gain) |
|---------|-----------------|--------|-----------------|---------|
| 3 mois | $9,334 | $10,313 | $11,614 | 69% |
| 6 mois | $9,229 | $10,678 | $12,655 | 77% |
| 12 mois | $9,280 | $11,421 | $14,343 | 86% |
| 24 mois | $9,679 | $13,125 | $18,114 | 92% |
| 36 mois | $10,323 | $14,995 | $22,203 | 96% |

## Stress Tests

| Métrique | Valeur |
|----------|--------|
| Pire mois | -3.9% |
| Meilleur mois | +9.3% |
| Mois moyen | +1.47% |
| Pire trimestre | -2.8% |
| Mois positifs | 46% |
| Max losing streak | 12 bars |
| Recovery from max DD | 113 bars |

## Utilisation

```bash
python portfolio/v4b/code/portfolio_v4b_final.py
```

- Résultats: `portfolio/v4b/results/portfolio_v4b_final_*.json`

## Évolution des versions

| Métrique | V3b | V4 (conserv.) | **V4b** |
|----------|-----|---------------|---------|
| Return | +9.8% | +4.9% | **+19.8%** |
| Sharpe | 1.19 | 2.59 | **1.35** |
| Max DD | -4.9% | -0.8% | **-8.5%** |
| Calmar | 1.91 | 5.99 | **2.17** |
| Objectif +15% | ❌ | ❌ | **✅** |
