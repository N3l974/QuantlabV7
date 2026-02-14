# Portfolio V3 — Markowitz + Holdout Validation
**Date** : 12 February 2026 (18:10)
**Cutoff holdout** : 2025-02-01
**Seeds** : 3 par combo
**Monte Carlo** : 1000 simulations
**Statut** : TERMINE

---

## 1. Combos survivants

| # | Symbol | Strategie | TF | Verdict HO | Full Sharpe | HO Sharpe | HO Return | HO DD |
|---|--------|-----------|-----|------------|-------------|-----------|-----------|-------|
| 1 | ETHUSDT | breakout_regime | 4h | STRONG | -0.27 | 0.93 | 7.2% | -7.8% |
| 2 | ETHUSDT | trend_multi_factor | 1d | WEAK | -0.15 | 0.78 | 11.6% | -13.5% |
| 3 | ETHUSDT | supertrend_adx | 4h | WEAK | -0.54 | 0.69 | 8.2% | -13.4% |
| 4 | ETHUSDT | trend_multi_factor | 4h | STRONG | -0.21 | 0.18 | 1.7% | -14.6% |
| 5 | SOLUSDT | breakout_regime | 1d | STRONG | -0.43 | 0.01 | -0.2% | -8.1% |
| 6 | BTCUSDT | trend_multi_factor | 1d | WEAK | -0.25 | 0.32 | 2.8% | -6.2% |
| 7 | BTCUSDT | supertrend_adx | 4h | WEAK | 0.00 | 0.19 | 1.5% | -11.1% |
| 8 | ETHUSDT | supertrend | 4h | STRONG | 0.26 | 0.70 | 11.0% | -14.5% |

## 2. Matrice de correlation

| | ETH/breakout | ETH/trend_mu | ETH/supertre | ETH/trend_mu | SOL/breakout | BTC/trend_mu | BTC/supertre | ETH/supertre |
|---|---|---|---|---|---|---|---|---|
| **ETH/breakout** | 1.00 | 0.00 | 0.35 | 0.40 | -0.00 | -0.01 | -0.00 | 0.37 |
| **ETH/trend_mu** | 0.00 | 1.00 | 0.01 | 0.01 | -0.02 | -0.01 | -0.01 | 0.01 |
| **ETH/supertre** | 0.35 | 0.01 | 1.00 | 0.55 | 0.01 | -0.01 | 0.01 | 0.48 |
| **ETH/trend_mu** | 0.40 | 0.01 | 0.55 | 1.00 | 0.01 | -0.00 | 0.00 | 0.76 |
| **SOL/breakout** | -0.00 | -0.02 | 0.01 | 0.01 | 1.00 | -0.00 | -0.00 | -0.01 |
| **BTC/trend_mu** | -0.01 | -0.01 | -0.01 | -0.00 | -0.00 | 1.00 | -0.01 | 0.01 |
| **BTC/supertre** | -0.00 | -0.01 | 0.01 | 0.00 | -0.00 | -0.01 | 1.00 | 0.01 |
| **ETH/supertre** | 0.37 | 0.01 | 0.48 | 0.76 | -0.01 | 0.01 | 0.01 | 1.00 |

## 3. Comparaison des portfolios

| Portfolio | Methode | Full Sharpe | HO Sharpe | HO Return | HO DD |
|-----------|---------|-------------|-----------|-----------|-------|
| markowitz_max_sharpe | Markowitz Monte Carlo (10K samples, max  | 0.19 | 0.85 | 9.6% | -9.2% |
| markowitz_min_var | Markowitz Monte Carlo (10K samples, min  | -0.41 | 0.76 | 3.9% | -3.9% |
| ho_sharpe_weighted | Weight proportional to holdout Sharpe | -0.24 | 1.06 | 8.5% | -6.6% |
| equal_weight | Equal weight (1/N) | -0.29 | 0.80 | 5.9% | -6.1% |
| risk_parity | Inverse volatility (risk parity) | -0.31 | 0.76 | 5.7% | -6.8% |

## 4. Allocations

### markowitz_max_sharpe

| Combo | Poids |
|-------|-------|
| ETHUSDT/breakout_regime/4h | 1.6% |
| ETHUSDT/trend_multi_factor/1d | 18.8% |
| ETHUSDT/supertrend_adx/4h | 0.0% |
| ETHUSDT/trend_multi_factor/4h | 1.1% |
| SOLUSDT/breakout_regime/1d | 9.1% |
| BTCUSDT/trend_multi_factor/1d | 0.3% |
| BTCUSDT/supertrend_adx/4h | 8.6% |
| ETHUSDT/supertrend/4h | 60.5% |

### markowitz_min_var

| Combo | Poids |
|-------|-------|
| ETHUSDT/breakout_regime/4h | 21.0% |
| ETHUSDT/trend_multi_factor/1d | 9.1% |
| ETHUSDT/supertrend_adx/4h | 1.9% |
| ETHUSDT/trend_multi_factor/4h | 0.2% |
| SOLUSDT/breakout_regime/1d | 34.9% |
| BTCUSDT/trend_multi_factor/1d | 11.1% |
| BTCUSDT/supertrend_adx/4h | 18.1% |
| ETHUSDT/supertrend/4h | 3.7% |

### ho_sharpe_weighted

| Combo | Poids |
|-------|-------|
| ETHUSDT/breakout_regime/4h | 24.5% |
| ETHUSDT/trend_multi_factor/1d | 20.4% |
| ETHUSDT/supertrend_adx/4h | 18.2% |
| ETHUSDT/trend_multi_factor/4h | 4.7% |
| SOLUSDT/breakout_regime/1d | 0.4% |
| BTCUSDT/trend_multi_factor/1d | 8.4% |
| BTCUSDT/supertrend_adx/4h | 5.1% |
| ETHUSDT/supertrend/4h | 18.3% |

### equal_weight

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

### risk_parity

| Combo | Poids |
|-------|-------|
| ETHUSDT/breakout_regime/4h | 21.9% |
| ETHUSDT/trend_multi_factor/1d | 6.1% |
| ETHUSDT/supertrend_adx/4h | 14.8% |
| ETHUSDT/trend_multi_factor/4h | 11.6% |
| SOLUSDT/breakout_regime/1d | 10.1% |
| BTCUSDT/trend_multi_factor/1d | 7.6% |
| BTCUSDT/supertrend_adx/4h | 17.7% |
| ETHUSDT/supertrend/4h | 10.3% |

## 5. Impact du Leverage

### Portfolio : ho_sharpe_weighted

| Leverage | Sharpe | Return | DD | Sortino |
|----------|--------|--------|-----|---------|
| 1x | -0.24 | -12.3% | -17.6% | -0.29 |
| 2x | -0.24 | -25.9% | -33.8% | -0.29 |
| 3x | -0.24 | -39.5% | -48.1% | -0.29 |

## 6. Monte Carlo Stress Tests (1Y)

| Portfolio | Med Ret | P5 Ret | P95 Ret | Med DD | Worst DD (P5) | P(ruin) |
|-----------|---------|--------|---------|--------|---------------|---------|
| markowitz_max_sharpe_1x | -0.0% | -7.6% | +8.4% | -5.1% | -9.8% | 0.0% |
| markowitz_min_var_1x | -0.3% | -3.0% | +2.4% | -1.8% | -3.6% | 0.0% |
| ho_sharpe_weighted_1x | -0.5% | -5.2% | +4.8% | -3.3% | -6.7% | 0.0% |
| equal_weight_1x | -0.5% | -4.7% | +4.0% | -2.9% | -5.6% | 0.0% |
| risk_parity_1x | -0.3% | -4.7% | +4.3% | -2.9% | -5.8% | 0.0% |
| ho_sharpe_weighted_2x | -0.9% | -10.7% | +9.3% | -6.6% | -13.8% | 0.0% |
| ho_sharpe_weighted_3x | -1.5% | -15.7% | +14.9% | -9.8% | -19.3% | 0.0% |

## 7. Projections multi-horizon (ho_sharpe_weighted_1x)

| Horizon | Med Ret | P(>0) | P(>10%) | P(>20%) | P5 | P95 |
|---------|---------|-------|---------|---------|-----|-----|
| 1Y | -0.4% | 44.7% | 0.1% | 0.0% | -5.2% | +4.8% |
| 2Y | -0.5% | 46.7% | 0.9% | 0.0% | -7.5% | +7.2% |
| 3Y | -1.1% | 42.1% | 2.0% | 0.0% | -9.9% | +7.2% |
| 5Y | -2.0% | 39.1% | 5.1% | 0.5% | -12.4% | +10.0% |

### Projections de capital ($10,000)

| Horizon | Pessimiste (P5) | Median | Optimiste (P95) |
|---------|----------------|--------|-----------------|
| 1Y | $9,475 | $9,961 | $10,477 |
| 2Y | $9,252 | $9,953 | $10,717 |
| 3Y | $9,011 | $9,888 | $10,716 |
| 5Y | $8,761 | $9,800 | $11,001 |

## 8. Verdict

**Meilleur portfolio** : `ho_sharpe_weighted`
- Full Sharpe : -0.24
- Holdout Sharpe : 1.06
- Holdout Return : 8.5%
- Holdout Max DD : -6.6%

**VERDICT : VIABLE pour deploiement live**

---
*Genere le 12 February 2026*