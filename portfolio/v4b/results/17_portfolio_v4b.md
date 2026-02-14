# Portfolio V4b — Aggressive (target +15%)
**Date** : 12 February 2026 (20:52)
**Objectif** : +15% annuel, DD < -20%

---

## Comparaison des variantes

| Portfolio | Sharpe | Sortino | Return | Max DD | Calmar | N combos |
|-----------|--------|---------|--------|--------|--------|----------|
| ✅ **C_lev_3.0x** | 0.97 | 1.46 | 12.4% | -11.0% | 1.10 | 8 |
| ❌ **C_lev_2.5x** | 0.97 | 1.46 | 10.4% | -9.3% | 1.10 | 8 |
| ❌ **A_concentrated_sharpe** | 1.45 | 2.09 | 10.3% | -4.3% | 2.28 | 8 |
| ❌ **A_concentrated_maxret** | 1.41 | 2.06 | 8.4% | -3.5% | 2.31 | 8 |
| ❌ **C_lev_2.0x** | 0.97 | 1.46 | 8.3% | -7.4% | 1.09 | 8 |
| ❌ **C_lev_1.5x** | 0.97 | 1.46 | 6.3% | -5.6% | 1.09 | 8 |
| ❌ **A_concentrated_retdd** | 0.97 | 1.46 | 4.2% | -3.8% | 1.08 | 8 |
| ❌ **B_selective** | 1.26 | 2.21 | 2.8% | -1.7% | 1.62 | 12 |

## 🏆 Recommandé : C_lev_3.0x

| Métrique | Valeur | Objectif | Status |
|----------|--------|----------|--------|
| Return | 12.4% | ≥15% | ⚠️ |
| Max DD | -11.0% | ≥-20% | ✅ |
| Sharpe | 0.97 | ≥1.0 | ❌ |
| Calmar | 1.10 | ≥1.0 | ✅ |

### Allocations

| Poids | Symbol | Stratégie | TF | HO Return | HO DD |
|-------|--------|-----------|-----|-----------|-------|
| 25.0% | ETHUSDT | ichimoku_cloud | 4h | 6.1% | -4.6% |
| 20.1% | ETHUSDT | macd_crossover | 1d | 7.4% | -7.8% |
| 19.8% | SOLUSDT | trend_multi_factor | 4h | 12.8% | -11.7% |
| 11.0% | BTCUSDT | supertrend | 1d | 8.2% | -4.3% |
| 10.9% | ETHUSDT | bollinger_breakout | 1d | 6.0% | -7.7% |
| 9.3% | BTCUSDT | trend_multi_factor | 1d | 6.3% | -3.9% |
| 2.0% | ETHUSDT | trend_multi_factor | 1d | 20.0% | -11.2% |
| 2.0% | ETHUSDT | supertrend | 1d | 26.7% | -13.3% |

### Allocation par symbol

| Symbol | Allocation |
|--------|-----------|
| ETHUSDT | 60.0% |
| BTCUSDT | 20.2% |
| SOLUSDT | 19.8% |

### Monte Carlo ($10,000)

| Horizon | P5 | Médian | P95 | P(>0) |
|---------|-----|--------|-----|-------|
| 3M | $9,262 | $10,183 | $11,257 | 61% |
| 6M | $9,117 | $10,384 | $11,842 | 67% |
| 12M | $8,829 | $10,771 | $13,291 | 72% |
| 24M | $8,981 | $10,829 | $13,360 | 74% |

### Comparaison V3b / V4 / V4b

| Métrique | V3b | V4 (conserv.) | V4b (agressif) |
|----------|-----|---------------|----------------|
| Return | +9.8% | +4.9% | **+12.4%** |
| Sharpe | 1.19 | 2.59 | **0.97** |
| Max DD | -4.9% | -0.8% | **-11.0%** |
| Calmar | 1.91 | 5.99 | **1.10** |
| ETH % | 95% | 53% | **60%** |

---
*Généré le 12 February 2026*