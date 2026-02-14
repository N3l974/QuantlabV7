# Diagnostic V4 Edge — Fast (2 phases)
**Date** : 12 February 2026 (20:32)
**Durée** : 20.7 min
**Config** : WF 1 seed, 30 trials, reoptim=3M, window=1Y
**Cutoff** : 2025-02-01

---

## Phase 1 — Quick Scan (defaults sur holdout)

- **Total combos scannés** : 132
- **Survivants (Sharpe > -1.5)** : 81

### Top 15 Phase 1 (defaults)

| Symbol | Stratégie | TF | Sharpe | +Ov Sharpe | Return | +Ov Return | DD | +Ov DD |
|--------|-----------|-----|--------|------------|--------|------------|-----|--------|
| ETHUSDT | regime_adaptive | 1d | 0.812 | 1.756 | 7.4% | 6.0% | -5.6% | -1.2% |
| BTCUSDT | trend_multi_factor | 1d | -0.082 | 1.377 | -1.1% | 7.8% | -12.4% | -4.4% |
| SOLUSDT | ichimoku_cloud | 1d | 0.072 | 1.360 | 0.2% | 2.1% | -3.8% | -0.6% |
| BTCUSDT | supertrend | 1d | -1.926 | 1.310 | -14.1% | 7.4% | -15.9% | -3.9% |
| ETHUSDT | mtf_trend_entry | 4h | 0.687 | 1.043 | 1.2% | 0.6% | -1.0% | -0.2% |
| SOLUSDT | adx_regime | 1d | 1.026 | 0.883 | 4.3% | 0.8% | -2.0% | -0.1% |
| BTCUSDT | supertrend_adx | 1d | -0.495 | 0.939 | -4.4% | 4.8% | -10.9% | -4.4% |
| ETHUSDT | mean_reversion_zscore | 1d | -0.210 | 0.937 | -2.1% | 3.3% | -9.3% | -2.5% |
| ETHUSDT | mtf_trend_entry | 1d | 0.839 | 0.835 | 2.4% | 1.5% | -2.0% | -1.3% |
| ETHUSDT | mtf_momentum_breakout | 4h | -0.677 | 0.834 | -7.2% | 4.2% | -10.0% | -4.0% |
| SOLUSDT | regime_adaptive | 1d | -2.469 | 0.793 | -15.7% | 2.6% | -15.7% | -2.9% |
| BTCUSDT | breakout_regime | 1d | -0.582 | 0.647 | -2.6% | 1.4% | -5.9% | -1.3% |
| ETHUSDT | volume_obv | 4h | -0.445 | 0.599 | -2.5% | 1.4% | -6.7% | -2.3% |
| SOLUSDT | keltner_channel | 1d | 0.293 | 0.355 | 2.1% | 1.0% | -8.5% | -2.3% |
| SOLUSDT | trend_multi_factor | 1d | -0.794 | 0.316 | -10.3% | 1.3% | -15.7% | -2.9% |

## Phase 2 — Walk-Forward + Holdout

- **Combos WF** : 81 baseline + 81 overlay
- **Baseline** : 14 STRONG, 14 WEAK, 53 FAIL
- **+Overlay** : 16 STRONG, 11 WEAK, 54 FAIL
- **Avg HO Sharpe baseline** : -0.401
- **Avg HO Sharpe +overlay** : -0.237

### Top combos — Baseline

| # | V | Symbol | Stratégie | TF | IS Sharpe | HO Sharpe | HO Ret | HO DD | Trades | Calmar |
|---|---|--------|-----------|-----|-----------|-----------|--------|-------|--------|--------|
| 1 | ❌ | SOLUSDT | mtf_momentum_breakout | 1d | -0.195 | 1.522 | 6.2% | -0.9% | 1 | 6.68 |
| 2 | ✅ | ETHUSDT | supertrend | 1d | 0.097 | 1.351 | 26.7% | -13.3% | 7 | 1.85 |
| 3 | ✅ | ETHUSDT | trend_multi_factor | 1d | -0.079 | 1.263 | 20.0% | -11.2% | 8 | 1.68 |
| 4 | ✅ | ETHUSDT | macd_crossover | 1d | -0.115 | 0.786 | 7.4% | -7.8% | 10 | 0.95 |
| 5 | ✅ | BTCUSDT | mtf_momentum_breakout | 1d | 0.090 | 0.740 | 3.1% | -2.9% | 3 | 1.04 |
| 6 | ✅ | SOLUSDT | trend_multi_factor | 4h | 0.271 | 0.717 | 12.8% | -11.7% | 28 | 1.15 |
| 7 | ✅ | ETHUSDT | atr_volatility_breakout | 1d | 0.238 | 0.709 | 4.7% | -6.7% | 20 | 0.71 |
| 8 | ✅ | BTCUSDT | keltner_channel | 1d | -0.207 | 0.652 | 3.9% | -6.0% | 7 | 0.65 |
| 9 | ✅ | ETHUSDT | bollinger_breakout | 1d | 0.136 | 0.637 | 6.0% | -7.7% | 12 | 0.80 |
| 10 | ✅ | SOLUSDT | ichimoku_cloud | 1d | -0.013 | 0.447 | 3.1% | -5.8% | 13 | 0.56 |
| 11 | ✅ | BTCUSDT | bollinger_breakout | 1d | 0.236 | 0.411 | 2.2% | -3.3% | 7 | 0.68 |
| 12 | ✅ | SOLUSDT | adx_regime | 1d | -0.268 | 0.397 | 0.8% | -1.4% | 10 | 0.60 |
| 13 | ✅ | BTCUSDT | trend_multi_factor | 1d | -0.281 | 0.387 | 3.4% | -7.1% | 4 | 0.53 |
| 14 | ✅ | ETHUSDT | trend_multi_factor | 4h | -0.074 | 0.386 | 4.6% | -15.4% | 12 | 0.34 |
| 15 | ✅ | ETHUSDT | keltner_channel | 1d | 0.205 | 0.310 | 2.4% | -6.8% | 11 | 0.40 |
| 16 | ⚠️ | BTCUSDT | atr_volatility_breakout | 1d | -0.047 | 0.277 | 1.1% | -4.0% | 4 | 0.28 |
| 17 | ⚠️ | SOLUSDT | keltner_channel | 1d | -0.592 | 0.261 | 3.0% | -11.7% | 22 | 0.36 |
| 18 | ⚠️ | ETHUSDT | momentum_roc | 1d | 0.392 | 0.248 | 1.9% | -10.6% | 21 | 0.21 |
| 19 | ⚠️ | BTCUSDT | vwap_deviation | 1d | -0.907 | 0.207 | 1.1% | -4.0% | 10 | 0.30 |
| 20 | ⚠️ | ETHUSDT | ichimoku_cloud | 4h | 0.069 | 0.197 | 1.7% | -10.9% | 40 | 0.21 |

### Top combos — Avec Overlays

| # | V | Symbol | Stratégie | TF | IS Sharpe | HO Sharpe | HO Ret | HO DD | Trades | Calmar |
|---|---|--------|-----------|-----|-----------|-----------|--------|-------|--------|--------|
| 1 | ❌ | ETHUSDT | keltner_channel | 1d | 0.205 | 1.663 | 3.9% | -0.4% | 2 | 8.56 |
| 2 | ❌ | SOLUSDT | mtf_momentum_breakout | 1d | -0.195 | 1.520 | 3.5% | -0.5% | 1 | 6.70 |
| 3 | ✅ | BTCUSDT | supertrend | 1d | 0.111 | 1.440 | 8.2% | -4.3% | 4 | 1.80 |
| 4 | ❌ | SOLUSDT | trend_multi_factor | 1d | -1.161 | 1.433 | 5.2% | -1.5% | 2 | 3.36 |
| 5 | ✅ | BTCUSDT | keltner_channel | 1d | -0.207 | 1.386 | 5.6% | -1.7% | 3 | 3.08 |
| 6 | ❌ | ETHUSDT | trend_multi_factor | 1d | -0.079 | 1.320 | 5.1% | -1.9% | 2 | 2.64 |
| 7 | ✅ | ETHUSDT | ichimoku_cloud | 4h | 0.069 | 1.253 | 6.1% | -4.6% | 40 | 1.29 |
| 8 | ✅ | ETHUSDT | supertrend_adx | 4h | -0.235 | 1.083 | 5.5% | -7.0% | 23 | 0.76 |
| 9 | ❌ | BTCUSDT | mtf_momentum_breakout | 1d | 0.090 | 1.056 | 3.7% | -2.4% | 2 | 1.51 |
| 10 | ✅ | ETHUSDT | ema_ribbon | 1d | -0.731 | 1.037 | 3.1% | -1.6% | 4 | 1.86 |
| 11 | ✅ | BTCUSDT | trend_multi_factor | 1d | -0.281 | 1.029 | 6.3% | -3.9% | 4 | 1.56 |
| 12 | ✅ | BTCUSDT | bollinger_breakout | 1d | 0.236 | 1.003 | 3.9% | -2.5% | 3 | 1.49 |
| 13 | ✅ | ETHUSDT | supertrend | 1d | 0.097 | 0.781 | 3.4% | -2.6% | 3 | 1.28 |
| 14 | ✅ | SOLUSDT | atr_volatility_breakout | 1d | -0.688 | 0.752 | 1.5% | -0.9% | 3 | 1.59 |
| 15 | ✅ | ETHUSDT | atr_volatility_breakout | 1d | 0.238 | 0.725 | 1.6% | -1.3% | 7 | 1.28 |
| 16 | ❌ | BTCUSDT | atr_volatility_breakout | 1d | -0.047 | 0.620 | 1.6% | -0.9% | 2 | 1.82 |
| 17 | ✅ | BTCUSDT | breakout_regime | 1d | 0.018 | 0.493 | 1.0% | -1.5% | 3 | 0.66 |
| 18 | ✅ | SOLUSDT | keltner_channel | 1d | -0.592 | 0.375 | 1.7% | -3.1% | 8 | 0.55 |
| 19 | ✅ | ETHUSDT | volume_obv | 4h | -0.611 | 0.368 | 1.1% | -3.7% | 13 | 0.30 |
| 20 | ✅ | ETHUSDT | keltner_channel | 4h | 0.266 | 0.325 | 2.0% | -7.3% | 33 | 0.29 |

### Δ Overlay (top améliorations)

| Symbol | Stratégie | TF | Base Sharpe | +Ov Sharpe | Δ | Base DD | +Ov DD |
|--------|-----------|-----|-------------|------------|---|---------|--------|
| SOLUSDT | trend_multi_factor | 1d | -1.456 | 1.433 | +2.889 | -16.0% | -1.5% |
| SOLUSDT | supertrend | 1d | -1.069 | 0.310 | +1.379 | -15.3% | -6.6% |
| ETHUSDT | williams_r | 1d | -1.307 | 0.065 | +1.371 | -9.7% | -1.6% |
| ETHUSDT | keltner_channel | 1d | 0.310 | 1.663 | +1.353 | -6.8% | -0.4% |
| SOLUSDT | atr_volatility_breakout | 1d | -0.572 | 0.752 | +1.324 | -8.1% | -0.9% |
| BTCUSDT | macd_crossover | 1d | -1.512 | -0.230 | +1.281 | -11.1% | -3.9% |
| BTCUSDT | supertrend | 1d | 0.175 | 1.440 | +1.265 | -12.0% | -4.3% |
| BTCUSDT | ichimoku_cloud | 4h | -1.165 | 0.087 | +1.252 | -11.9% | -4.5% |
| ETHUSDT | breakout_regime | 1d | -1.477 | -0.252 | +1.225 | -15.6% | -4.0% |
| ETHUSDT | ichimoku_cloud | 4h | 0.197 | 1.253 | +1.056 | -10.9% | -4.6% |
| ETHUSDT | breakout_regime | 4h | -0.933 | 0.112 | +1.045 | -15.4% | -8.1% |
| ETHUSDT | supertrend_adx | 4h | 0.065 | 1.083 | +1.018 | -12.0% | -7.0% |
| ETHUSDT | volume_obv | 1d | -1.087 | -0.090 | +0.997 | -8.0% | -2.1% |
| ETHUSDT | volume_obv | 4h | -0.550 | 0.368 | +0.918 | -6.8% | -3.7% |
| ETHUSDT | ema_ribbon | 1d | 0.139 | 1.037 | +0.898 | -6.4% | -1.6% |
| BTCUSDT | mean_reversion_zscore | 1d | -1.686 | -0.805 | +0.881 | -10.6% | -4.4% |
| ETHUSDT | regime_adaptive | 1d | -1.695 | -0.821 | +0.873 | -15.7% | -7.3% |
| SOLUSDT | momentum_roc | 1d | -1.176 | -0.341 | +0.835 | -15.3% | -5.4% |
| BTCUSDT | keltner_channel | 1d | 0.652 | 1.386 | +0.733 | -6.0% | -1.7% |
| BTCUSDT | trend_multi_factor | 1d | 0.387 | 1.029 | +0.642 | -7.1% | -3.9% |

### Ranking stratégies (avg HO Sharpe baseline)

| Stratégie | Avg HO Sharpe | Best | N |
|-----------|---------------|------|---|
| mtf_momentum_breakout | 0.181 | 1.522 | 6 |
| bollinger_breakout | 0.134 | 0.637 | 3 |
| trend_multi_factor | 0.061 | 1.263 | 6 |
| keltner_channel | -0.036 | 0.652 | 5 |
| supertrend | -0.063 | 1.351 | 6 |
| adx_regime | -0.105 | 0.397 | 3 |
| ema_ribbon | -0.125 | 0.139 | 3 |
| atr_volatility_breakout | -0.405 | 0.709 | 4 |
| ichimoku_cloud | -0.416 | 0.447 | 6 |
| macd_crossover | -0.463 | 0.786 | 4 |
| vwap_deviation | -0.537 | 0.207 | 3 |
| breakout_regime | -0.564 | 0.104 | 4 |
| momentum_roc | -0.590 | 0.248 | 4 |
| supertrend_adx | -0.616 | 0.065 | 5 |
| stochastic_oscillator | -0.773 | -0.773 | 1 |
| regime_adaptive | -0.792 | -0.177 | 5 |
| williams_r | -0.825 | -0.344 | 2 |
| volume_obv | -0.920 | 0.000 | 5 |
| rsi_mean_reversion | -0.926 | -0.202 | 3 |
| mean_reversion_zscore | -1.344 | -0.975 | 3 |

## Pool survivants pour Portfolio V4

**39 combos uniques**

| # | Symbol | Stratégie | TF | Ov? | HO Sharpe | HO Ret | HO DD | Verdict |
|---|--------|-----------|-----|-----|-----------|--------|-------|---------|
| 1 | BTCUSDT | supertrend | 1d | ✓ | 1.440 | 8.2% | -4.3% | STRONG |
| 2 | BTCUSDT | keltner_channel | 1d | ✓ | 1.386 | 5.6% | -1.7% | STRONG |
| 3 | ETHUSDT | supertrend | 1d | — | 1.351 | 26.7% | -13.3% | STRONG |
| 4 | ETHUSDT | trend_multi_factor | 1d | — | 1.263 | 20.0% | -11.2% | STRONG |
| 5 | ETHUSDT | ichimoku_cloud | 4h | ✓ | 1.253 | 6.1% | -4.6% | STRONG |
| 6 | ETHUSDT | supertrend_adx | 4h | ✓ | 1.083 | 5.5% | -7.0% | STRONG |
| 7 | ETHUSDT | ema_ribbon | 1d | ✓ | 1.037 | 3.1% | -1.6% | STRONG |
| 8 | BTCUSDT | trend_multi_factor | 1d | ✓ | 1.029 | 6.3% | -3.9% | STRONG |
| 9 | BTCUSDT | bollinger_breakout | 1d | ✓ | 1.003 | 3.9% | -2.5% | STRONG |
| 10 | ETHUSDT | macd_crossover | 1d | — | 0.786 | 7.4% | -7.8% | STRONG |
| 11 | SOLUSDT | atr_volatility_breakout | 1d | ✓ | 0.752 | 1.5% | -0.9% | STRONG |
| 12 | BTCUSDT | mtf_momentum_breakout | 1d | — | 0.740 | 3.1% | -2.9% | STRONG |
| 13 | ETHUSDT | atr_volatility_breakout | 1d | ✓ | 0.725 | 1.6% | -1.3% | STRONG |
| 14 | SOLUSDT | trend_multi_factor | 4h | — | 0.717 | 12.8% | -11.7% | STRONG |
| 15 | ETHUSDT | bollinger_breakout | 1d | — | 0.637 | 6.0% | -7.7% | STRONG |
| 16 | BTCUSDT | breakout_regime | 1d | ✓ | 0.493 | 1.0% | -1.5% | STRONG |
| 17 | SOLUSDT | ichimoku_cloud | 1d | — | 0.447 | 3.1% | -5.8% | STRONG |
| 18 | SOLUSDT | adx_regime | 1d | — | 0.397 | 0.8% | -1.4% | STRONG |
| 19 | ETHUSDT | trend_multi_factor | 4h | — | 0.386 | 4.6% | -15.4% | STRONG |
| 20 | SOLUSDT | keltner_channel | 1d | ✓ | 0.375 | 1.7% | -3.1% | STRONG |
| 21 | ETHUSDT | volume_obv | 4h | ✓ | 0.368 | 1.1% | -3.7% | STRONG |
| 22 | ETHUSDT | keltner_channel | 4h | ✓ | 0.325 | 2.0% | -7.3% | STRONG |
| 23 | SOLUSDT | macd_crossover | 1d | ✓ | 0.319 | 0.5% | -1.9% | STRONG |
| 24 | ETHUSDT | keltner_channel | 1d | — | 0.310 | 2.4% | -6.8% | STRONG |
| 25 | SOLUSDT | supertrend | 1d | ✓ | 0.310 | 1.3% | -6.6% | STRONG |
| 26 | SOLUSDT | ichimoku_cloud | 4h | ✓ | 0.297 | 1.0% | -4.1% | WEAK |
| 27 | BTCUSDT | atr_volatility_breakout | 1d | — | 0.277 | 1.1% | -4.0% | WEAK |
| 28 | ETHUSDT | momentum_roc | 1d | — | 0.248 | 1.9% | -10.6% | WEAK |
| 29 | SOLUSDT | mtf_momentum_breakout | 4h | ✓ | 0.246 | 1.4% | -5.7% | WEAK |
| 30 | BTCUSDT | vwap_deviation | 1d | — | 0.207 | 1.1% | -4.0% | WEAK |
| 31 | SOLUSDT | supertrend_adx | 1d | ✓ | 0.201 | 0.5% | -2.5% | WEAK |
| 32 | ETHUSDT | breakout_regime | 4h | ✓ | 0.112 | 0.5% | -8.1% | WEAK |
| 33 | SOLUSDT | breakout_regime | 4h | ✓ | 0.095 | 0.1% | -1.8% | WEAK |
| 34 | BTCUSDT | ichimoku_cloud | 4h | ✓ | 0.087 | 0.3% | -4.5% | WEAK |
| 35 | ETHUSDT | williams_r | 1d | ✓ | 0.065 | 0.1% | -1.6% | WEAK |
| 36 | SOLUSDT | ema_ribbon | 1d | ✓ | 0.058 | 0.1% | -1.4% | WEAK |
| 37 | BTCUSDT | supertrend | 4h | — | 0.049 | -0.0% | -12.3% | WEAK |
| 38 | ETHUSDT | mtf_momentum_breakout | 4h | — | 0.035 | 0.0% | -6.5% | WEAK |
| 39 | BTCUSDT | momentum_roc | 1d | ✓ | 0.026 | 0.0% | -2.8% | WEAK |

---
*Généré le 12 February 2026*