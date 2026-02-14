# Diagnostic V4 — 2-Pass Scan (Fast + Robust)
**Date** : 12 February 2026 (13:38)
**Durée totale** : 230.3 min (P1: 151.5 + P2: 78.8)
**Pass 1** : 1 seed, 50 trials, pruning=True
**Pass 2** : 5 seeds, 100 trials (médiane)
**Defaults** : reoptim=3M, window=1Y, bounds=1.0, sharpe (validés par A/B test)
**Bugs fixés** : williams_r, stochastic, donchian, zscore, keltner
**Statut** : ✅ TERMINÉ

---

## Résumé

- **Combos total** : 320
- **Pass 1 viable** : 295
- **Pass 2 candidats** : 32
- **Pass 2 validés** : 32
- **HIGH confidence** : 0
- **MEDIUM confidence** : 5
- **LOW confidence** : 27

## Top 20 combos

| # | Conf | Symbol | Stratégie | TF | Score | Sharpe | Sharpe med | Sharpe std | Return | DD | PF |
|---|------|--------|-----------|-----|-------|--------|------------|------------|--------|-----|-----|
| 1 | MEDIUM | ETHUSDT | supertrend | 4h | 0.130 | 0.166 | 0.166 | 0.164 | 9.1% | -32.5% | 1.11 |
| 2 | MEDIUM | SOLUSDT | bollinger_breakout | 4h | 0.128 | 0.194 | 0.194 | 0.199 | 8.6% | -18.5% | 1.10 |
| 3 | MEDIUM | BTCUSDT | momentum_roc | 1d | 0.121 | 0.222 | 0.222 | 0.163 | 6.8% | -16.3% | 1.20 |
| 4 | MEDIUM | ETHUSDT | atr_volatility_breakout | 1d | 0.045 | 0.078 | 0.078 | 0.195 | 1.8% | -24.8% | 1.03 |
| 5 | MEDIUM | ETHUSDT | volume_obv | 4h | 0.024 | 0.043 | 0.043 | 0.205 | 0.4% | -15.6% | 1.02 |
| 6 | LOW | BTCUSDT | stochastic_oscillator | 15m | -0.008 | -0.010 | -0.010 | 0.276 | -2.0% | -18.0% | 1.06 |
| 7 | LOW | BTCUSDT | keltner_channel | 1d | -0.016 | -0.029 | -0.029 | 0.316 | -1.9% | -20.7% | 0.97 |
| 8 | LOW | BTCUSDT | bollinger_breakout | 1d | -0.017 | -0.032 | -0.032 | 0.493 | -1.7% | -16.8% | 0.97 |
| 9 | LOW | BNBUSDT | momentum_roc | 1d | -0.037 | -0.061 | -0.061 | 0.155 | -6.3% | -24.7% | 0.94 |
| 10 | LOW | BTCUSDT | atr_volatility_breakout | 1d | -0.038 | -0.068 | -0.068 | 0.200 | -3.8% | -21.1% | 0.94 |
| 11 | LOW | ETHUSDT | keltner_channel | 4h | -0.065 | -0.108 | -0.108 | 0.145 | -11.1% | -36.0% | 0.97 |
| 12 | LOW | BTCUSDT | atr_volatility_breakout | 4h | -0.086 | -0.152 | -0.152 | 0.146 | -9.3% | -23.0% | 0.94 |
| 13 | LOW | XRPUSDT | atr_volatility_breakout | 1d | -0.090 | -0.157 | -0.157 | 0.253 | -12.8% | -36.4% | 0.86 |
| 14 | LOW | BTCUSDT | keltner_channel | 4h | -0.105 | -0.187 | -0.187 | 0.194 | -11.0% | -24.0% | 0.91 |
| 15 | LOW | ETHUSDT | keltner_channel | 1d | -0.128 | -0.225 | -0.225 | 0.275 | -13.2% | -37.3% | 0.85 |
| 16 | LOW | BNBUSDT | rsi_mean_reversion | 4h | -0.138 | -0.207 | -0.207 | 0.322 | -22.0% | -37.3% | 0.94 |
| 17 | LOW | BTCUSDT | supertrend | 1d | -0.142 | -0.192 | -0.192 | 0.192 | -18.4% | -43.3% | 0.89 |
| 18 | LOW | BNBUSDT | stochastic_oscillator | 1d | -0.148 | -0.291 | -0.291 | 0.190 | -14.2% | -22.3% | 0.82 |
| 19 | LOW | BTCUSDT | volume_obv | 1d | -0.151 | -0.304 | -0.304 | 0.430 | -8.0% | -19.8% | 0.78 |
| 20 | LOW | BNBUSDT | mean_reversion_zscore | 1d | -0.162 | -0.545 | -0.545 | 0.347 | -9.1% | -11.3% | 0.33 |

## Ranking par stratégie

| Stratégie | Viable | HIGH | Avg Score | Avg Sharpe |
|-----------|--------|------|-----------|------------|
| bollinger_breakout | 2 | 0 | 0.055 | 0.081 |
| momentum_roc | 2 | 0 | 0.042 | 0.080 |
| supertrend | 2 | 0 | -0.006 | -0.013 |
| keltner_channel | 4 | 0 | -0.078 | -0.137 |
| atr_volatility_breakout | 6 | 0 | -0.101 | -0.180 |
| stochastic_oscillator | 3 | 0 | -0.151 | -0.334 |
| rsi_mean_reversion | 3 | 0 | -0.159 | -0.241 |
| volume_obv | 4 | 0 | -0.159 | -0.318 |
| mean_reversion_zscore | 1 | 0 | -0.162 | -0.545 |
| ichimoku_cloud | 1 | 0 | -0.212 | -0.368 |
| williams_r | 1 | 0 | -0.212 | -0.381 |
| vwap_deviation | 1 | 0 | -0.241 | -0.445 |
| macd_crossover | 1 | 0 | -0.287 | -0.558 |
| adx_regime | 1 | 0 | -0.504 | -1.310 |

## Ranking par symbol

| Symbol | Viable | HIGH | Avg Score |
|--------|--------|------|-----------|
| ETHUSDT | 9 | 0 | -0.102 |
| BTCUSDT | 11 | 0 | -0.113 |
| BNBUSDT | 4 | 0 | -0.121 |
| SOLUSDT | 5 | 0 | -0.159 |
| XRPUSDT | 3 | 0 | -0.159 |

## Analyse de la variance

Distribution de `robust_sharpe_std` (variance inter-seeds) :

- **Min** : 0.1319
- **Médiane** : 0.1971
- **Max** : 0.4933
- **Combos avec std < 0.1** : 0/32
- **Combos avec std > 0.3** : 8/32

## Méthodologie

### Pipeline 2-pass
1. **Pass 1 (fast scan)** : 1 seed, 50 trials, pruning → filtre les combos nuls
2. **Pass 2 (robust)** : 5 seeds, 100 trials → validation multi-seed sur combos viables

### Paramètres
- **Walk-forward** : defaults fixes (validés par test A/B vs méta-optimisation)
- **Déterminisme** : TPESampler seeded par fenêtre
- **Pruning** : MedianPruner (pass 1 seulement)
- **Bugs fixés avant scan** : williams_r (signal mort), stochastic (crash), donchian/zscore (off-by-one), keltner (ATR)

### Confidence tiers
- **HIGH** : score > 0.3 ET (min_sharpe > 0 ET std < 0.3) OU (score > 0.3 ET median > 0.2)
- **MEDIUM** : score > 0.0
- **LOW** : score ≤ 0.0

---
*Généré le 12 February 2026*