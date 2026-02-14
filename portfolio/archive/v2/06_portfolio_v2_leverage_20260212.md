# Rapport — Portfolio V2 : Sharpe-Weighted vs Positive-Only + Leverage
**Date** : 12 February 2026 (05:57)
**Fichier source** : `portfolio/archive/v3/meta_profiles_v3_20260211_195716.json`
**Statut** : ✅ VALIDE

---

## Contexte

Comparaison de deux approches de construction de portfolio :
- **sharpe_weighted** : tous les combos, poids ∝ max(Sharpe, 0.01)
- **positive_only** : filtre les combos avec Sharpe < 0, poids ∝ Sharpe

Test de leverage 1x, 2x, 3x sur chaque variante avec Monte Carlo stress tests.

## 1. Performance individuelle (Walk-Forward OOS)

| Symbol | Stratégie | Sharpe | Return | DD | PF | WR | Trades | Filtre |
|--------|-----------|--------|--------|-----|-----|-----|--------|--------|
| XRPUSDT | stochastic_oscillator | **0.43** | 22.7% | -16.8% | 1.73 | 50.0% | 34 | ✓ IN |
| ETHUSDT | atr_volatility_breakout | **0.32** | 8.3% | -9.6% | 1.29 | 54.1% | 37 | ✓ IN |
| ETHUSDT | donchian_channel | **0.49** | 29.2% | -10.1% | 1.28 | 52.5% | 120 | ✓ IN |
| BTCUSDT | ema_ribbon | -0.44 | -17.0% | -24.9% | 0.75 | 35.8% | 81 | ✗ OUT |
| BTCUSDT | donchian_channel | -0.50 | -19.2% | -23.7% | 0.77 | 40.4% | 99 | ✗ OUT |

**3 combos retenus** (Sharpe > 0) | **2 combos filtrés** (Sharpe ≤ 0)

## 2. Comparaison des portfolios (1x)

| Portfolio | Sharpe | Return | DD | Sortino |
|-----------|--------|--------|-----|---------|
| equal_weight | 0.21 | 4.3% | -5.1% | 0.24 |
| risk_parity | 0.45 | 10.4% | -4.1% | 0.52 |
| sharpe_weighted | 0.65 | 22.1% | -5.5% | 0.75 |
| positive_only | 0.66 | 22.9% | -5.6% | 0.64 |

### Allocation sharpe_weighted

| Combo | Poids |
|-------|-------|
| XRPUSDT/stochastic_oscillator | 34.3% |
| ETHUSDT/atr_volatility_breakout | 25.5% |
| ETHUSDT/donchian_channel | 38.7% |
| BTCUSDT/ema_ribbon | 0.8% |
| BTCUSDT/donchian_channel | 0.8% |

### Allocation positive_only

| Combo | Poids |
|-------|-------|
| XRPUSDT/stochastic_oscillator | 34.8% |
| ETHUSDT/atr_volatility_breakout | 25.9% |
| ETHUSDT/donchian_channel | 39.3% |

## 3. Impact du Leverage

| Portfolio | Lev | Sharpe | Return | Ret/an | DD | Sortino |
|-----------|-----|--------|--------|--------|-----|---------|
| sharpe_weighted | 1x | 0.65 | 22.1% | 3.2% | -5.5% | 0.75 |
| sharpe_weighted | 2x | 0.65 | 46.9% | 6.2% | -10.8% | 0.75 |
| sharpe_weighted | 3x | 0.65 | 74.2% | 9.0% | -15.9% | 0.75 |
| positive_only | 1x | 0.66 | 22.9% | 3.3% | -5.6% | 0.64 |
| positive_only | 2x | 0.66 | 48.7% | 6.4% | -10.9% | 0.64 |
| positive_only | 3x | 0.66 | 77.1% | 9.3% | -16.1% | 0.64 |

## 4. Monte Carlo Stress Tests (1Y, 1000 sims)

| Portfolio | Med Ret | P5 Ret | P95 Ret | Med DD | Worst DD | P(ruin) |
|-----------|---------|--------|---------|--------|----------|---------|
| sharpe_weighted | +3.0% | -4.2% | +12.0% | -3.1% | -6.8% | 0.0% |
| sharpe_weighted_2x | +6.5% | -9.2% | +25.8% | -6.2% | -13.6% | 0.0% |
| sharpe_weighted_3x | +8.4% | -13.0% | +39.3% | -9.3% | -19.4% | 0.0% |
| positive_only | +3.2% | -3.9% | +13.4% | -3.2% | -6.9% | 0.0% |
| positive_only_2x | +5.8% | -8.6% | +26.8% | -6.4% | -13.9% | 0.0% |
| positive_only_3x | +8.9% | -13.1% | +43.0% | -9.4% | -20.0% | 0.0% |

## 5. Projections multi-horizon — positive_only

| Horizon | Med Ret | Mean Ret | P(>0) | P(>10%) | P(>20%) | P5 | P95 |
|---------|---------|----------|-------|---------|---------|-----|-----|
| 1Y | +2.9% | +3.4% | 74.5% | 9.1% | 0.5% | -4.0% | +12.1% |
| 2Y | +6.1% | +6.9% | 81.1% | 31.5% | 6.1% | -4.0% | +21.0% |
| 3Y | +9.3% | +10.4% | 88.3% | 47.0% | 15.2% | -3.8% | +27.4% |
| 5Y | +17.5% | +18.3% | 93.4% | 73.0% | 43.1% | -1.4% | +39.4% |

### Projections de capital ($10,000)

| Horizon | Pessimiste (P5) | Médian | Optimiste (P95) |
|---------|----------------|--------|-----------------|
| 1Y | $9,599 | $10,289 | $11,213 |
| 2Y | $9,600 | $10,607 | $12,096 |
| 3Y | $9,617 | $10,933 | $12,735 |
| 5Y | $9,861 | $11,747 | $13,940 |

## 6. Risk Assessment

| Portfolio | Risk | Ruin% | Med DD | Worst DD |
|-----------|------|-------|--------|----------|
| sharpe_weighted | 🟢 LOW | 0.0% | 3.1% | 6.8% |
| sharpe_weighted_2x | 🟢 LOW | 0.0% | 6.2% | 13.6% |
| sharpe_weighted_3x | 🟢 LOW | 0.0% | 9.3% | 19.4% |
| positive_only | 🟢 LOW | 0.0% | 3.2% | 6.9% |
| positive_only_2x | 🟢 LOW | 0.0% | 6.4% | 13.9% |
| positive_only_3x | 🟡 MED | 0.0% | 9.4% | 20.0% |

## 7. Verdict & Recommandations

### Leverage 1x
- sharpe_weighted : Sharpe=0.65, DD=-5.5%
- positive_only : Sharpe=0.66, DD=-5.6%
- Gagnant : **positive_only**

### Leverage 2x
- sharpe_weighted : Sharpe=0.65, DD=-10.8%
- positive_only : Sharpe=0.66, DD=-10.9%
- Gagnant : **positive_only**

### Leverage 3x
- sharpe_weighted : Sharpe=0.65, DD=-15.9%
- positive_only : Sharpe=0.66, DD=-16.1%
- Gagnant : **positive_only**

### Meilleur portfolio global : `positive_only`
- Sharpe : 0.66
- Return : 22.9%
- Max DD : -5.6%

---

*Généré le 12 February 2026*