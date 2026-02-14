# Diagnostic V5 — ATR SL/TP + Risk-based Sizing
**Date** : 12 February 2026 (22:32)
**Duree** : 49.2 min
**Config** : WF 1 seed, 30 trials, reoptim=3M, window=1Y
**Cutoff** : 2025-02-01
**V5 risk_per_trade_pct** : 0.01

---

## Phase 1 — Quick Scan (defaults sur holdout)

- **Total combos scannes** : 132
- **Survivants (Sharpe > -1.5)** : 81

## Phase 2 — Walk-Forward V4 vs V5

- **V4 baseline** : 14 STRONG, 6 WEAK, 61 FAIL | Avg Sharpe=-0.504
- **V4 +overlay** : 28 STRONG, 5 WEAK, 48 FAIL | Avg Sharpe=-0.135
- **V5 baseline** : 17 STRONG, 15 WEAK, 49 FAIL | Avg Sharpe=-0.250
- **V5 +overlay** : 29 STRONG, 7 WEAK, 45 FAIL | Avg Sharpe=0.056

## V4 vs V5 Comparison (baseline)

| Symbol | Strategie | TF | V4 Sharpe | V5 Sharpe | Delta | V4 DD | V5 DD | V5 ATR_SL | V5 ATR_TP |
|--------|-----------|-----|-----------|-----------|-------|-------|-------|-----------|-----------|
| ETHUSDT | macd_crossover | 1d | -0.480 | 2.012 | +2.492 | -13.6% | -3.2% | 1.33 | 6.37 |
| ETHUSDT | atr_volatility_breakout | 1d | -1.329 | 0.789 | +2.117 | -8.6% | -2.3% | 2.70 | 4.97 |
| SOLUSDT | atr_volatility_breakout | 4h | -2.006 | 0.023 | +2.029 | -15.3% | -3.9% | 1.62 | 1.75 |
| ETHUSDT | regime_adaptive | 1d | -1.900 | 0.010 | +1.910 | -15.6% | -7.5% | 1.32 | 3.28 |
| SOLUSDT | breakout_regime | 4h | -1.419 | 0.485 | +1.904 | -9.3% | -1.6% | 3.78 | 1.78 |
| BTCUSDT | williams_r | 1d | -2.118 | -0.390 | +1.728 | -8.2% | -0.7% | 3.99 | 3.14 |
| SOLUSDT | supertrend_adx | 1d | -1.807 | -0.172 | +1.635 | -15.2% | -5.9% | 1.92 | 6.86 |
| SOLUSDT | keltner_channel | 1d | -1.455 | 0.142 | +1.598 | -16.6% | -5.7% | 1.72 | 3.74 |
| ETHUSDT | trend_multi_factor | 1d | 0.138 | 1.585 | +1.447 | -15.8% | -3.0% | 3.00 | 5.44 |
| BTCUSDT | trend_multi_factor | 1d | -1.541 | -0.097 | +1.444 | -15.2% | -5.4% | 3.19 | 6.07 |
| ETHUSDT | adx_regime | 1d | -1.124 | 0.305 | +1.429 | -3.0% | -3.0% | 1.52 | 2.07 |
| SOLUSDT | atr_volatility_breakout | 1d | -2.111 | -0.749 | +1.362 | -15.7% | -3.6% | 3.60 | 6.94 |
| ETHUSDT | macd_crossover | 4h | -1.224 | 0.052 | +1.276 | -15.3% | -6.7% | 3.08 | 5.29 |
| SOLUSDT | trend_multi_factor | 4h | -0.560 | 0.661 | +1.221 | -15.1% | -8.4% | 2.83 | 7.92 |
| SOLUSDT | volume_obv | 4h | -0.607 | 0.467 | +1.074 | -11.6% | -8.0% | 1.35 | 1.29 |
| SOLUSDT | macd_crossover | 1d | -1.445 | -0.380 | +1.064 | -15.4% | -10.5% | 1.26 | 4.94 |
| ETHUSDT | supertrend | 4h | -1.036 | -0.045 | +0.991 | -15.2% | -15.3% | 2.68 | 5.76 |
| BTCUSDT | rsi_mean_reversion | 1d | -1.088 | -0.176 | +0.911 | -16.8% | -6.2% | 3.28 | 4.64 |
| ETHUSDT | supertrend | 1d | -0.786 | 0.123 | +0.909 | -15.7% | -10.6% | 1.56 | 7.02 |
| SOLUSDT | supertrend | 1d | -0.366 | 0.534 | +0.900 | -16.1% | -11.7% | 0.49 | 7.99 |
| SOLUSDT | ichimoku_cloud | 1d | -2.525 | -1.681 | +0.844 | -12.8% | -16.0% | 0.77 | 6.55 |
| ETHUSDT | vwap_deviation | 1d | -0.737 | 0.087 | +0.824 | -10.1% | -2.2% | 2.17 | 5.16 |
| ETHUSDT | momentum_roc | 1d | 0.089 | 0.764 | +0.675 | -11.8% | -6.8% | 1.23 | 3.04 |
| SOLUSDT | vwap_deviation | 1d | -0.344 | 0.310 | +0.654 | -15.5% | -4.8% | 1.44 | 5.60 |
| ETHUSDT | keltner_channel | 4h | -0.845 | -0.281 | +0.564 | -7.7% | -10.4% | 3.33 | 5.44 |
| ETHUSDT | ichimoku_cloud | 4h | 0.132 | 0.594 | +0.462 | -12.7% | -8.2% | 2.28 | 5.97 |
| BTCUSDT | adx_regime | 1d | -1.751 | -1.358 | +0.393 | -8.0% | -1.1% | 3.29 | 0.17 |
| BTCUSDT | keltner_channel | 1d | -1.025 | -0.656 | +0.369 | -3.4% | -7.7% | 0.14 | 7.97 |
| ETHUSDT | volume_obv | 4h | -0.208 | 0.147 | +0.355 | -7.0% | -6.0% | 2.58 | 6.78 |
| BTCUSDT | mtf_momentum_breakout | 1d | 0.190 | 0.535 | +0.345 | -6.5% | -3.3% | 2.72 | 3.80 |
| ETHUSDT | volume_obv | 1d | -0.444 | -0.112 | +0.332 | -9.7% | -3.4% | 2.13 | 2.94 |
| ETHUSDT | rsi_mean_reversion | 1d | -0.018 | 0.259 | +0.278 | -15.3% | -13.4% | 3.92 | 7.97 |
| ETHUSDT | ema_ribbon | 1d | -0.198 | 0.060 | +0.257 | -7.2% | -2.0% | 3.45 | 4.39 |
| ETHUSDT | supertrend_adx | 1d | -0.042 | 0.189 | +0.231 | -14.7% | -4.6% | 1.91 | 7.38 |
| BTCUSDT | vwap_deviation | 1d | -0.909 | -0.691 | +0.218 | -7.0% | -2.9% | 3.26 | 3.42 |
| ETHUSDT | supertrend_adx | 4h | -0.004 | 0.212 | +0.216 | -12.1% | -2.6% | 3.53 | 4.08 |
| ETHUSDT | breakout_regime | 4h | -0.017 | 0.193 | +0.209 | -8.9% | -4.4% | 2.79 | 3.33 |
| ETHUSDT | bollinger_breakout | 4h | -0.132 | 0.037 | +0.168 | -9.0% | -5.6% | 2.81 | 2.97 |
| ETHUSDT | mtf_momentum_breakout | 1d | 0.301 | 0.463 | +0.163 | -9.0% | -1.9% | 3.99 | 4.72 |
| SOLUSDT | mean_reversion_zscore | 1d | -0.573 | -0.428 | +0.146 | -14.1% | -7.4% | 0.95 | 4.42 |
| ETHUSDT | stochastic_oscillator | 1d | -0.385 | -0.274 | +0.111 | -7.7% | -2.9% | 1.27 | 0.38 |
| ETHUSDT | keltner_channel | 1d | 0.930 | 1.007 | +0.078 | -9.5% | -3.4% | 2.44 | 6.20 |
| BTCUSDT | mean_reversion_zscore | 1d | -1.846 | -1.772 | +0.074 | -11.0% | -10.3% | 0.67 | 1.27 |
| SOLUSDT | ichimoku_cloud | 4h | 0.403 | 0.471 | +0.068 | -8.9% | -3.5% | 3.97 | 7.46 |
| ETHUSDT | regime_adaptive | 4h | 0.837 | 0.858 | +0.021 | -9.8% | -6.6% | 3.38 | 6.87 |
| BTCUSDT | regime_adaptive | 1d | -1.019 | -0.999 | +0.019 | -13.1% | -6.9% | 2.66 | 4.03 |
| ETHUSDT | ichimoku_cloud | 1d | 1.058 | 1.075 | +0.017 | -4.8% | -4.6% | 0.71 | 7.99 |
| BTCUSDT | breakout_regime | 1d | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% | 0.18 | 5.36 |
| ETHUSDT | breakout_regime | 1d | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% | 0.18 | 5.36 |
| SOLUSDT | mtf_momentum_breakout | 1d | 1.181 | 1.163 | -0.018 | -2.1% | -0.6% | 1.95 | 4.73 |
| BTCUSDT | supertrend_adx | 1d | 0.072 | 0.034 | -0.038 | -2.9% | -2.9% | 0.79 | 5.61 |
| SOLUSDT | volume_obv | 1d | -0.146 | -0.239 | -0.093 | -12.2% | -6.7% | 0.65 | 4.07 |
| ETHUSDT | mean_reversion_zscore | 1d | -1.470 | -1.584 | -0.114 | -1.8% | -4.3% | 2.27 | 4.68 |
| BTCUSDT | mtf_momentum_breakout | 4h | -0.509 | -0.628 | -0.119 | -5.4% | -1.2% | 0.62 | 1.24 |
| SOLUSDT | adx_regime | 1d | -0.049 | -0.199 | -0.150 | -3.4% | -4.3% | 1.21 | 1.42 |
| BTCUSDT | supertrend | 4h | -1.269 | -1.462 | -0.192 | -15.0% | -15.1% | 2.83 | 7.19 |
| BTCUSDT | supertrend | 1d | -0.308 | -0.502 | -0.194 | -14.9% | -10.6% | 1.70 | 7.82 |
| SOLUSDT | regime_adaptive | 1d | -1.277 | -1.479 | -0.202 | -15.3% | -12.2% | 0.65 | 1.34 |
| SOLUSDT | supertrend | 4h | -0.082 | -0.315 | -0.233 | -15.2% | -15.0% | 1.06 | 5.82 |
| BTCUSDT | volume_obv | 1d | -0.044 | -0.325 | -0.281 | -9.3% | -4.1% | 3.27 | 4.75 |
| SOLUSDT | regime_adaptive | 4h | -0.941 | -1.241 | -0.300 | -15.2% | -15.2% | 0.60 | 4.74 |
| SOLUSDT | keltner_channel | 4h | -0.739 | -1.062 | -0.323 | -11.1% | -15.0% | 2.04 | 2.97 |
| ETHUSDT | bollinger_breakout | 1d | 0.402 | 0.055 | -0.348 | -8.4% | -6.2% | 1.12 | 4.06 |
| ETHUSDT | williams_r | 1d | -1.432 | -1.802 | -0.371 | -10.3% | -6.7% | 1.37 | 2.68 |
| SOLUSDT | ema_ribbon | 1d | -0.627 | -1.039 | -0.412 | -10.1% | -2.7% | 3.44 | 3.58 |
| BTCUSDT | macd_crossover | 1d | -1.274 | -1.692 | -0.419 | -4.3% | -1.9% | 3.94 | 0.12 |
| BTCUSDT | atr_volatility_breakout | 1d | 0.274 | -0.180 | -0.454 | -5.6% | -3.8% | 0.95 | 3.53 |
| SOLUSDT | rsi_mean_reversion | 1d | -0.332 | -0.842 | -0.510 | -15.3% | -7.6% | 2.04 | 7.81 |
| SOLUSDT | trend_multi_factor | 1d | 0.435 | -0.080 | -0.514 | -14.8% | -13.4% | 0.46 | 2.78 |
| BTCUSDT | momentum_roc | 1d | 0.412 | -0.115 | -0.527 | -7.2% | -4.1% | 1.66 | 1.70 |
| BTCUSDT | bollinger_breakout | 1d | -0.604 | -1.159 | -0.555 | -7.5% | -3.4% | 3.64 | 1.31 |
| BTCUSDT | supertrend_adx | 4h | -1.099 | -1.718 | -0.619 | -10.0% | -15.1% | 3.32 | 7.57 |
| BTCUSDT | ichimoku_cloud | 1d | 0.432 | -0.208 | -0.640 | -0.8% | -2.4% | 3.90 | 3.65 |
| SOLUSDT | mtf_momentum_breakout | 4h | 0.393 | -0.313 | -0.706 | -6.5% | -12.8% | 2.44 | 5.61 |
| BTCUSDT | ema_ribbon | 1d | 0.646 | -0.098 | -0.744 | -8.7% | -12.0% | 1.66 | 3.20 |
| SOLUSDT | momentum_roc | 1d | 0.336 | -0.541 | -0.877 | -14.0% | -10.6% | 1.68 | 3.92 |
| BTCUSDT | trend_multi_factor | 4h | -0.684 | -1.589 | -0.905 | -14.6% | -15.2% | 3.10 | 7.91 |
| BTCUSDT | ichimoku_cloud | 4h | -0.140 | -1.160 | -1.020 | -6.0% | -7.3% | 3.77 | 3.91 |
| ETHUSDT | trend_multi_factor | 4h | 0.783 | -0.249 | -1.032 | -12.8% | -9.0% | 3.49 | 4.81 |
| SOLUSDT | momentum_roc | 4h | -0.833 | -2.043 | -1.211 | -15.5% | -15.4% | 3.00 | 5.03 |
| ETHUSDT | mtf_momentum_breakout | 4h | 1.015 | -1.823 | -2.838 | -6.6% | -4.6% | 0.52 | 1.19 |

**V5 vs V4** : 47 improved, 32 degraded, avg delta=0.254

### Top V5 combos (baseline)

| # | V | Symbol | Strategie | TF | HO Sharpe | HO Ret | HO DD | Trades | ATR_SL | ATR_TP |
|---|---|--------|-----------|-----|-----------|--------|-------|--------|--------|--------|
| 1 | + | ETHUSDT | macd_crossover | 1d | 2.012 | 13.8% | -3.2% | 6 | 1.33 | 6.37 |
| 2 | + | ETHUSDT | trend_multi_factor | 1d | 1.585 | 6.4% | -3.0% | 10 | 3.00 | 5.44 |
| 3 | - | SOLUSDT | mtf_momentum_breakout | 1d | 1.163 | 1.6% | -0.6% | 1 | 1.95 | 4.73 |
| 4 | + | ETHUSDT | ichimoku_cloud | 1d | 1.075 | 6.0% | -4.6% | 5 | 0.71 | 7.99 |
| 5 | + | ETHUSDT | keltner_channel | 1d | 1.007 | 3.2% | -3.4% | 8 | 2.44 | 6.20 |
| 6 | + | ETHUSDT | regime_adaptive | 4h | 0.858 | 6.0% | -6.6% | 17 | 3.38 | 6.87 |
| 7 | + | ETHUSDT | atr_volatility_breakout | 1d | 0.789 | 2.2% | -2.3% | 5 | 2.70 | 4.97 |
| 8 | + | ETHUSDT | momentum_roc | 1d | 0.764 | 4.9% | -6.8% | 20 | 1.23 | 3.04 |
| 9 | + | SOLUSDT | trend_multi_factor | 4h | 0.661 | 6.1% | -8.4% | 21 | 2.83 | 7.92 |
| 10 | + | ETHUSDT | ichimoku_cloud | 4h | 0.594 | 4.4% | -8.2% | 22 | 2.28 | 5.97 |
| 11 | + | BTCUSDT | mtf_momentum_breakout | 1d | 0.535 | 2.1% | -3.3% | 10 | 2.72 | 3.80 |
| 12 | + | SOLUSDT | supertrend | 1d | 0.534 | 7.3% | -11.7% | 11 | 0.49 | 7.99 |
| 13 | + | SOLUSDT | breakout_regime | 4h | 0.485 | 1.3% | -1.6% | 12 | 3.78 | 1.78 |
| 14 | + | SOLUSDT | ichimoku_cloud | 4h | 0.471 | 2.2% | -3.5% | 18 | 3.97 | 7.46 |
| 15 | + | SOLUSDT | volume_obv | 4h | 0.467 | 3.8% | -8.0% | 56 | 1.35 | 1.29 |
| 16 | + | ETHUSDT | mtf_momentum_breakout | 1d | 0.463 | 1.0% | -1.9% | 10 | 3.99 | 4.72 |
| 17 | + | SOLUSDT | vwap_deviation | 1d | 0.310 | 2.0% | -4.8% | 19 | 1.44 | 5.60 |
| 18 | + | ETHUSDT | adx_regime | 1d | 0.305 | 0.9% | -3.0% | 17 | 1.52 | 2.07 |
| 19 | ~ | ETHUSDT | rsi_mean_reversion | 1d | 0.259 | 2.6% | -13.4% | 6 | 3.92 | 7.97 |
| 20 | ~ | ETHUSDT | supertrend_adx | 4h | 0.212 | 0.5% | -2.6% | 9 | 3.53 | 4.08 |

### Top V5 combos (+overlay)

| # | V | Symbol | Strategie | TF | HO Sharpe | HO Ret | HO DD | Trades | ATR_SL | ATR_TP |
|---|---|--------|-----------|-----|-----------|--------|-------|--------|--------|--------|
| 1 | + | BTCUSDT | ema_ribbon | 1d | 2.109 | 8.7% | -1.3% | 6 | 1.66 | 3.20 |
| 2 | - | SOLUSDT | supertrend_adx | 1d | 1.695 | 1.3% | -0.5% | 2 | 1.92 | 6.86 |
| 3 | + | SOLUSDT | supertrend | 1d | 1.587 | 4.5% | -1.5% | 3 | 0.49 | 7.99 |
| 4 | - | ETHUSDT | supertrend_adx | 1d | 1.560 | 1.7% | -0.6% | 2 | 1.91 | 7.38 |
| 5 | - | SOLUSDT | adx_regime | 1d | 1.528 | 1.6% | -0.3% | 2 | 1.21 | 1.42 |
| 6 | + | BTCUSDT | mtf_momentum_breakout | 1d | 1.423 | 5.2% | -1.6% | 4 | 2.72 | 3.80 |
| 7 | + | SOLUSDT | keltner_channel | 1d | 1.384 | 2.4% | -0.6% | 3 | 1.72 | 3.74 |
| 8 | + | ETHUSDT | mtf_momentum_breakout | 1d | 1.330 | 1.3% | -0.5% | 3 | 3.99 | 4.72 |
| 9 | + | BTCUSDT | trend_multi_factor | 1d | 1.277 | 3.2% | -1.1% | 3 | 3.19 | 6.07 |
| 10 | + | SOLUSDT | macd_crossover | 1d | 1.230 | 2.9% | -1.2% | 3 | 1.26 | 4.94 |
| 11 | + | ETHUSDT | trend_multi_factor | 1d | 1.174 | 1.6% | -0.5% | 3 | 3.00 | 5.44 |
| 12 | + | SOLUSDT | trend_multi_factor | 1d | 1.173 | 6.4% | -4.5% | 8 | 0.46 | 2.78 |
| 13 | - | SOLUSDT | mtf_momentum_breakout | 1d | 1.160 | 0.9% | -0.4% | 1 | 1.95 | 4.73 |
| 14 | + | ETHUSDT | supertrend_adx | 4h | 1.079 | 0.8% | -0.5% | 7 | 3.53 | 4.08 |
| 15 | + | SOLUSDT | ichimoku_cloud | 4h | 1.064 | 2.2% | -1.4% | 17 | 3.97 | 7.46 |
| 16 | + | SOLUSDT | atr_volatility_breakout | 1d | 1.040 | 0.8% | -0.5% | 3 | 3.60 | 6.94 |
| 17 | + | ETHUSDT | keltner_channel | 1d | 1.007 | 1.4% | -1.1% | 3 | 2.44 | 6.20 |
| 18 | - | ETHUSDT | macd_crossover | 1d | 1.003 | 1.6% | -0.9% | 2 | 1.33 | 6.37 |
| 19 | - | ETHUSDT | atr_volatility_breakout | 1d | 0.975 | 0.7% | -0.4% | 1 | 2.70 | 4.97 |
| 20 | + | SOLUSDT | breakout_regime | 4h | 0.942 | 1.1% | -0.6% | 11 | 3.78 | 1.78 |

## ATR SL/TP Usage Analysis

- **ATR SL/TP used** : 162/162 combos (100%)
  - Avg ATR_SL mult: 2.17
  - Avg ATR_TP mult: 4.56
  - Avg Sharpe: -0.097

## Pool survivants pour Portfolio V5

**121 combos (toutes variantes)**

| # | Symbol | Strategie | TF | Mode | Ov? | HO Sharpe | HO Ret | HO DD | Verdict |
|---|--------|-----------|-----|------|-----|-----------|--------|-------|---------|
| 1 | BTCUSDT | ema_ribbon | 1d | v5 | Y | 2.109 | 8.7% | -1.3% | STRONG |
| 2 | ETHUSDT | macd_crossover | 1d | v5 | - | 2.012 | 13.8% | -3.2% | STRONG |
| 3 | SOLUSDT | supertrend | 1d | v4 | Y | 1.743 | 8.0% | -2.5% | STRONG |
| 4 | SOLUSDT | supertrend | 1d | v5 | Y | 1.587 | 4.5% | -1.5% | STRONG |
| 5 | ETHUSDT | trend_multi_factor | 1d | v5 | - | 1.585 | 6.4% | -3.0% | STRONG |
| 6 | SOLUSDT | macd_crossover | 1d | v4 | Y | 1.551 | 6.9% | -2.0% | STRONG |
| 7 | BTCUSDT | mtf_momentum_breakout | 1d | v5 | Y | 1.423 | 5.2% | -1.6% | STRONG |
| 8 | SOLUSDT | keltner_channel | 1d | v5 | Y | 1.384 | 2.4% | -0.6% | STRONG |
| 9 | ETHUSDT | mtf_momentum_breakout | 4h | v4 | Y | 1.352 | 6.5% | -4.6% | STRONG |
| 10 | ETHUSDT | mtf_momentum_breakout | 1d | v5 | Y | 1.330 | 1.3% | -0.5% | STRONG |
| 11 | BTCUSDT | ema_ribbon | 1d | v4 | Y | 1.314 | 8.0% | -2.4% | STRONG |
| 12 | BTCUSDT | trend_multi_factor | 1d | v5 | Y | 1.277 | 3.2% | -1.1% | STRONG |
| 13 | BTCUSDT | momentum_roc | 1d | v4 | Y | 1.259 | 6.8% | -2.7% | STRONG |
| 14 | BTCUSDT | mtf_momentum_breakout | 1d | v4 | Y | 1.243 | 6.5% | -2.2% | STRONG |
| 15 | SOLUSDT | macd_crossover | 1d | v5 | Y | 1.230 | 2.9% | -1.2% | STRONG |
| 16 | ETHUSDT | mtf_momentum_breakout | 1d | v4 | Y | 1.194 | 5.3% | -2.8% | STRONG |
| 17 | ETHUSDT | keltner_channel | 1d | v4 | Y | 1.189 | 4.8% | -2.9% | STRONG |
| 18 | ETHUSDT | trend_multi_factor | 1d | v5 | Y | 1.174 | 1.6% | -0.5% | STRONG |
| 19 | SOLUSDT | trend_multi_factor | 1d | v5 | Y | 1.173 | 6.4% | -4.5% | STRONG |
| 20 | SOLUSDT | atr_volatility_breakout | 1d | v4 | Y | 1.137 | 4.9% | -2.0% | STRONG |
| 21 | ETHUSDT | supertrend_adx | 4h | v5 | Y | 1.079 | 0.8% | -0.5% | STRONG |
| 22 | ETHUSDT | ichimoku_cloud | 1d | v5 | - | 1.075 | 6.0% | -4.6% | STRONG |
| 23 | SOLUSDT | ichimoku_cloud | 4h | v5 | Y | 1.064 | 2.2% | -1.4% | STRONG |
| 24 | ETHUSDT | ichimoku_cloud | 1d | v4 | - | 1.058 | 6.6% | -4.8% | STRONG |
| 25 | SOLUSDT | atr_volatility_breakout | 1d | v5 | Y | 1.040 | 0.8% | -0.5% | STRONG |
| 26 | ETHUSDT | mtf_momentum_breakout | 4h | v4 | - | 1.015 | 8.0% | -6.6% | STRONG |
| 27 | ETHUSDT | keltner_channel | 1d | v5 | - | 1.007 | 3.2% | -3.4% | STRONG |
| 28 | ETHUSDT | keltner_channel | 1d | v5 | Y | 1.007 | 1.4% | -1.1% | STRONG |
| 29 | SOLUSDT | momentum_roc | 1d | v4 | Y | 0.972 | 4.6% | -3.1% | STRONG |
| 30 | SOLUSDT | breakout_regime | 4h | v5 | Y | 0.942 | 1.1% | -0.6% | STRONG |
| 31 | ETHUSDT | keltner_channel | 1d | v4 | - | 0.930 | 10.4% | -9.5% | STRONG |
| 32 | ETHUSDT | regime_adaptive | 4h | v5 | - | 0.858 | 6.0% | -6.6% | STRONG |
| 33 | BTCUSDT | atr_volatility_breakout | 1d | v4 | Y | 0.855 | 3.3% | -2.3% | STRONG |
| 34 | SOLUSDT | ichimoku_cloud | 4h | v4 | Y | 0.846 | 3.7% | -3.0% | STRONG |
| 35 | ETHUSDT | regime_adaptive | 4h | v4 | - | 0.837 | 10.7% | -9.8% | STRONG |
| 36 | BTCUSDT | trend_multi_factor | 1d | v4 | Y | 0.834 | 5.2% | -4.0% | STRONG |
| 37 | SOLUSDT | keltner_channel | 1d | v4 | Y | 0.806 | 3.9% | -2.8% | STRONG |
| 38 | ETHUSDT | stochastic_oscillator | 1d | v5 | Y | 0.796 | 0.4% | -0.3% | STRONG |
| 39 | BTCUSDT | volume_obv | 1d | v4 | Y | 0.789 | 4.4% | -2.9% | STRONG |
| 40 | ETHUSDT | atr_volatility_breakout | 1d | v5 | - | 0.789 | 2.2% | -2.3% | STRONG |

---
*Genere le 12 February 2026*