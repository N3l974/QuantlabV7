# Portfolio V3b — Improved (Constrained + Holdout MC)
**Date** : 12 February 2026 (18:39)
**Ameliorations** : filtrage instabilite, cap symbol 50%, cap combo 25%, MC holdout-only

---

## 1. Filtrage des combos

- **Seuil HO Sharpe** : > -0.2
- **Seuil seed std** : < 1.5
- **Retenus** : 8 / 8

### Combos retenus

| # | Symbol | Strategie | TF | HO Sharpe | HO Return | Seed Std |
|---|--------|-----------|-----|-----------|-----------|----------|
| 1 | ETHUSDT | breakout_regime | 4h | 0.93 | 7.2% | 0.468 |
| 2 | ETHUSDT | trend_multi_factor | 1d | 0.78 | 11.6% | 0.859 |
| 3 | ETHUSDT | supertrend_adx | 4h | 0.69 | 8.2% | 1.102 |
| 4 | ETHUSDT | trend_multi_factor | 4h | 0.18 | 1.7% | 0.130 |
| 5 | SOLUSDT | breakout_regime | 1d | 0.01 | -0.2% | 0.022 |
| 6 | BTCUSDT | trend_multi_factor | 1d | 0.32 | 2.8% | 0.539 |
| 7 | BTCUSDT | supertrend_adx | 4h | 0.19 | 1.5% | 0.652 |
| 8 | ETHUSDT | supertrend | 4h | 0.70 | 11.0% | 0.121 |

## 2. Matrice de correlation (holdout)

| | ETH/breakout | ETH/trend_mu | ETH/supertre | ETH/trend_mu | SOL/breakout | BTC/trend_mu | BTC/supertre | ETH/supertre |
|---|---|---|---|---|---|---|---|---|
| **ETH/breakout** | 1.00 | -0.00 | 0.55 | 0.41 | 0.01 | 0.00 | 0.33 | 0.41 |
| **ETH/trend_mu** | -0.00 | 1.00 | 0.03 | -0.01 | 0.21 | 0.55 | 0.02 | 0.00 |
| **ETH/supertre** | 0.55 | 0.03 | 1.00 | 0.65 | 0.05 | 0.04 | 0.53 | 0.64 |
| **ETH/trend_mu** | 0.41 | -0.01 | 0.65 | 1.00 | 0.01 | -0.00 | 0.62 | 0.89 |
| **SOL/breakout** | 0.01 | 0.21 | 0.05 | 0.01 | 1.00 | 0.17 | 0.05 | 0.01 |
| **BTC/trend_mu** | 0.00 | 0.55 | 0.04 | -0.00 | 0.17 | 1.00 | 0.05 | -0.00 |
| **BTC/supertre** | 0.33 | 0.02 | 0.53 | 0.62 | 0.05 | 0.05 | 1.00 | 0.61 |
| **ETH/supertre** | 0.41 | 0.00 | 0.64 | 0.89 | 0.01 | -0.00 | 0.61 | 1.00 |

## 3. Comparaison des portfolios

| Portfolio | HO Sharpe | HO Sortino | HO Return | HO DD | HO Calmar |
|-----------|-----------|------------|-----------|-------|-----------|
| markowitz_constrained | 1.19 | 1.57 | 9.8% | -4.9% | 1.91 |
| ho_sharpe_constrained | 1.03 | 1.42 | 8.0% | -6.3% | 1.24 |
| risk_parity_constrained | 0.73 | 1.02 | 5.3% | -6.5% | 0.81 |
| equal_weight | 0.80 | 1.11 | 5.9% | -6.1% | 0.96 |

## 4. Allocations

### markowitz_constrained

Concentration: ETHUSDT 95%, BTCUSDT 3%, SOLUSDT 2%

| Combo | Poids |
|-------|-------|
| ETHUSDT/breakout_regime/4h | 32.7% |
| ETHUSDT/trend_multi_factor/1d | 32.7% |
| ETHUSDT/supertrend_adx/4h | 4.2% |
| ETHUSDT/trend_multi_factor/4h | 1.2% |
| SOLUSDT/breakout_regime/1d | 2.2% |
| BTCUSDT/trend_multi_factor/1d | 3.1% |
| BTCUSDT/supertrend_adx/4h | 0.2% |
| ETHUSDT/supertrend/4h | 23.7% |

### ho_sharpe_constrained

Concentration: ETHUSDT 78%, BTCUSDT 21%, SOLUSDT 1%

| Combo | Poids |
|-------|-------|
| ETHUSDT/breakout_regime/4h | 22.3% |
| ETHUSDT/trend_multi_factor/1d | 18.6% |
| ETHUSDT/supertrend_adx/4h | 16.5% |
| ETHUSDT/trend_multi_factor/4h | 4.3% |
| SOLUSDT/breakout_regime/1d | 0.6% |
| BTCUSDT/trend_multi_factor/1d | 13.2% |
| BTCUSDT/supertrend_adx/4h | 8.0% |
| ETHUSDT/supertrend/4h | 16.6% |

### risk_parity_constrained

Concentration: ETHUSDT 59%, BTCUSDT 29%, SOLUSDT 12%

| Combo | Poids |
|-------|-------|
| ETHUSDT/breakout_regime/4h | 21.5% |
| ETHUSDT/trend_multi_factor/1d | 4.3% |
| ETHUSDT/supertrend_adx/4h | 13.3% |
| ETHUSDT/trend_multi_factor/4h | 9.7% |
| SOLUSDT/breakout_regime/1d | 11.9% |
| BTCUSDT/trend_multi_factor/1d | 8.5% |
| BTCUSDT/supertrend_adx/4h | 20.8% |
| ETHUSDT/supertrend/4h | 9.8% |

### equal_weight

Concentration: ETHUSDT 62%, BTCUSDT 25%, SOLUSDT 12%

| Combo | Poids |
|-------|-------|
| ETHUSDT/breakout_regime/4h | 12.5% |
| ETHUSDT/trend_multi_factor/1d | 12.5% |
| ETHUSDT/supertrend_adx/4h | 12.5% |
| ETHUSDT/trend_multi_factor/4h | 12.5% |
| SOLUSDT/breakout_regime/1d | 12.5% |
| BTCUSDT/trend_multi_factor/1d | 12.5% |
| BTCUSDT/supertrend_adx/4h | 12.5% |
| ETHUSDT/supertrend/4h | 12.5% |

## 5. Monte Carlo Holdout-Only (markowitz_constrained)

- **Ruin probability (50%)** : 0.0%

### Projections (capital $10,000)

| Horizon | Median | P(>0) | P(>5%) | P(>10%) | Pessimiste (P5) | Optimiste (P95) | Med DD |
|---------|--------|-------|--------|---------|----------------|-----------------|--------|
| 3M | $10,223 (+2.2%) | 71% | 24% | 3% | $9,578 | $10,917 | -3.4% |
| 6M | $10,471 (+4.7%) | 80% | 48% | 18% | $9,525 | $11,466 | -4.5% |
| 12M | $10,947 (+9.5%) | 86% | 69% | 47% | $9,605 | $12,489 | -5.9% |
| 24M | $11,928 (+19.3%) | 95% | 89% | 78% | $9,997 | $14,373 | -7.5% |
| 36M | $13,137 (+31.4%) | 97% | 94% | 89% | $10,413 | $16,329 | -8.4% |

## 6. Profit Expectations (holdout-based)

**Portfolio retenu** : `markowitz_constrained`

| Metrique | Valeur |
|----------|--------|
| Return annuel (holdout) | +9.8% |
| Return mensuel moyen | +0.78% |
| Sharpe annualise | 1.19 |
| Sortino | 1.57 |
| Max Drawdown | -4.9% |
| Calmar | 1.91 |

### Expectations Monte Carlo (12 mois)

| Scenario | Return | Capital final |
|----------|--------|---------------|
| Pessimiste (P5) | -4.0% | $9,605 |
| Conservateur (P25) | +3.5% | $10,345 |
| Median | +9.5% | $10,947 |
| Optimiste (P75) | +15.5% | $11,554 |
| Tres optimiste (P95) | +24.9% | $12,489 |

## 7. Verdict

**Meilleur portfolio** : `markowitz_constrained`
- HO Sharpe : 1.19
- HO Return : +9.8%
- HO Max DD : -4.9%
- Ruin probability : 0.0%
- MC P(positive, 12M) : 86%

---
*Genere le 12 February 2026*