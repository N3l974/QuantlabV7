# Portfolio V4 — Edge-Enhanced
**Date** : 12 February 2026 (20:42)
**Durée** : 0.0 min
**Source** : diagnostic_v4_fast_20260212_203251.json
**Cutoff holdout** : 2025-02-01
**Contraintes** : ETH ≤ 60%, combo ≤ 20%, corr < 0.85

---

## Comparaison des méthodes

| Portfolio | HO Sharpe | HO Sortino | HO Return | HO DD | HO Calmar | N combos |
|-----------|-----------|------------|-----------|-------|-----------|----------|
| **markowitz_constrained** | 2.59 | 5.01 | 4.9% | -0.8% | 5.99 | 37 |
| **equal_weight** | 1.52 | 2.50 | 2.5% | -1.1% | 2.24 | 37 |
| **sharpe_weighted** | 1.88 | 3.23 | 4.6% | -1.1% | 4.07 | 31 |
| **risk_parity** | 1.48 | 2.42 | 1.8% | -0.7% | 2.54 | 37 |

## 🏆 Meilleur : markowitz_constrained

### Allocations

| # | Poids | Symbol | Stratégie | TF | Overlay | HO Sharpe | HO DD |
|---|-------|--------|-----------|-----|---------|-----------|-------|
| 1 | 6.0% | BTCUSDT | supertrend | 1d | ✓ | 1.440 | -4.3% |
| 2 | 3.2% | BTCUSDT | keltner_channel | 1d | ✓ | 1.386 | -1.7% |
| 3 | 5.1% | ETHUSDT | supertrend | 1d | — | 1.351 | -13.3% |
| 4 | 3.7% | ETHUSDT | ichimoku_cloud | 4h | ✓ | 1.253 | -4.6% |
| 5 | 1.0% | ETHUSDT | supertrend_adx | 4h | ✓ | 1.083 | -7.0% |
| 6 | 14.5% | ETHUSDT | ema_ribbon | 1d | ✓ | 1.037 | -1.6% |
| 7 | 9.0% | BTCUSDT | bollinger_breakout | 1d | ✓ | 1.003 | -2.5% |
| 8 | 9.7% | ETHUSDT | macd_crossover | 1d | — | 0.786 | -7.8% |
| 9 | 1.0% | SOLUSDT | atr_volatility_breakout | 1d | ✓ | 0.752 | -0.9% |
| 10 | 1.0% | BTCUSDT | mtf_momentum_breakout | 1d | — | 0.740 | -2.9% |
| 11 | 1.0% | ETHUSDT | atr_volatility_breakout | 1d | ✓ | 0.725 | -1.3% |
| 12 | 1.0% | SOLUSDT | trend_multi_factor | 4h | — | 0.717 | -11.7% |
| 13 | 1.0% | ETHUSDT | bollinger_breakout | 1d | — | 0.637 | -7.7% |
| 14 | 1.0% | BTCUSDT | breakout_regime | 1d | ✓ | 0.493 | -1.5% |
| 15 | 1.0% | SOLUSDT | ichimoku_cloud | 1d | — | 0.447 | -5.8% |
| 16 | 1.0% | SOLUSDT | adx_regime | 1d | — | 0.397 | -1.4% |
| 17 | 9.8% | ETHUSDT | trend_multi_factor | 4h | — | 0.386 | -15.4% |
| 18 | 1.0% | SOLUSDT | keltner_channel | 1d | ✓ | 0.375 | -3.1% |
| 19 | 1.0% | ETHUSDT | volume_obv | 4h | ✓ | 0.368 | -3.7% |
| 20 | 1.0% | ETHUSDT | keltner_channel | 4h | ✓ | 0.325 | -7.3% |
| 21 | 1.0% | SOLUSDT | macd_crossover | 1d | ✓ | 0.319 | -1.9% |
| 22 | 1.0% | ETHUSDT | keltner_channel | 1d | — | 0.310 | -6.8% |
| 23 | 1.0% | SOLUSDT | supertrend | 1d | ✓ | 0.310 | -6.6% |
| 24 | 1.0% | SOLUSDT | ichimoku_cloud | 4h | ✓ | 0.297 | -4.1% |
| 25 | 1.0% | BTCUSDT | atr_volatility_breakout | 1d | — | 0.277 | -4.0% |
| 26 | 1.0% | ETHUSDT | momentum_roc | 1d | — | 0.248 | -10.6% |
| 27 | 1.0% | SOLUSDT | mtf_momentum_breakout | 4h | ✓ | 0.246 | -5.7% |
| 28 | 10.8% | BTCUSDT | vwap_deviation | 1d | — | 0.207 | -4.0% |
| 29 | 1.0% | SOLUSDT | supertrend_adx | 1d | ✓ | 0.201 | -2.5% |
| 30 | 1.0% | ETHUSDT | breakout_regime | 4h | ✓ | 0.112 | -8.1% |
| 31 | 1.0% | SOLUSDT | breakout_regime | 4h | ✓ | 0.095 | -1.8% |
| 32 | 1.0% | BTCUSDT | ichimoku_cloud | 4h | ✓ | 0.087 | -4.5% |
| 33 | 1.0% | ETHUSDT | williams_r | 1d | ✓ | 0.065 | -1.6% |
| 34 | 1.0% | SOLUSDT | ema_ribbon | 1d | ✓ | 0.058 | -1.4% |
| 35 | 1.0% | BTCUSDT | supertrend | 4h | — | 0.049 | -12.3% |
| 36 | 1.0% | ETHUSDT | mtf_momentum_breakout | 4h | — | 0.035 | -6.5% |
| 37 | 1.0% | BTCUSDT | momentum_roc | 1d | ✓ | 0.026 | -2.8% |

### Allocation par symbol

| Symbol | Allocation |
|--------|-----------|
| ETHUSDT | 52.9% |
| BTCUSDT | 35.1% |
| SOLUSDT | 12.0% |

### Performance holdout

| Métrique | Valeur |
|----------|--------|
| **Return annuel** | **4.9%** |
| **Sharpe** | **2.59** |
| **Sortino** | **5.01** |
| **Max DD** | **-0.8%** |
| **Calmar** | **5.99** |

### Projections Monte Carlo ($10,000)

| Horizon | P5 (pessimiste) | Médian | P95 (optimiste) | P(>0) | P(ruin) |
|---------|----------------|--------|-----------------|-------|---------|
| 3 mois | $9,980 | $10,075 | $10,253 | 90% | 0.0% |
| 6 mois | $10,013 | $10,168 | $10,397 | 96% | 0.0% |
| 12 mois | $10,112 | $10,342 | $10,668 | 99% | 0.0% |
| 24 mois | $10,115 | $10,362 | $10,707 | 100% | 0.0% |

### Comparaison V3b vs V4

| Métrique | V3b (markowitz) | V4 (best) | Δ |
|----------|----------------|-----------|---|
| Sharpe | 1.19 | 2.59 | +1.40 |
| Return | 9.80% | 4.88% | -4.92 |
| Max DD | -4.90% | -0.78% | +4.12 |
| Calmar | 1.91 | 5.99 | +4.08 |

---
*Généré le 12 February 2026*