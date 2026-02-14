# Diagnostic V5b — Full Enrichment
**Date** : 13 February 2026 (08:32)
**Duree** : 183.3 min
**Config** : WF 3 seeds, 30 trials, reoptim=3M, window=1Y
**Cutoff** : 2025-02-01
**New features** : trailing_stop, breakeven, max_holding, multi-seed, risk_grid, correlation

---

## Phase 1 — Quick Scan
- **Combos scannes** : 132
- **Survivants** : 81

## Phase 2 — Walk-Forward V5b

- **flat baseline** : 9S/11W/61F | Avg Sharpe=-0.636
- **r1.0 baseline** : 13S/7W/61F | Avg Sharpe=-0.472
- **r0.5 baseline** : 5S/1W/3F | Avg Sharpe=0.378
- **r2.0 baseline** : 5S/2W/2F | Avg Sharpe=0.675
- **flat +overlay** : 23 STRONG | Avg Sharpe=-0.302
- **r1.0 +overlay** : 24 STRONG | Avg Sharpe=-0.100

### Top V5b combos (flat baseline)

| # | V | Symbol | Strategy | TF | Sharpe | Ret | DD | Tr | Trail | BE | MaxH | Seed_std |
|---|---|--------|----------|-----|--------|-----|-----|-----|-------|-----|------|----------|
| 1 | + | ETHUSDT | breakout_regime | 1d | 1.795 | 5.3% | -0.1% | 5 | 1.04 | 0.037 | 28 | 0.606 |
| 2 | + | SOLUSDT | mtf_momentum_breakout | 1d | 1.657 | 9.7% | -3.5% | 5 | 1.66 | 0.049 | 17 | 0.224 |
| 3 | + | ETHUSDT | trend_multi_factor | 1d | 1.169 | 18.3% | -9.9% | 25 | 3.40 | 0.041 | 56 | 0.230 |
| 4 | + | ETHUSDT | supertrend | 1d | 0.888 | 15.3% | -14.5% | 22 | 3.61 | 0.015 | 69 | 0.081 |
| 5 | + | SOLUSDT | ichimoku_cloud | 4h | 0.809 | 7.1% | -7.5% | 32 | 2.93 | 0.019 | 64 | 0.026 |
| 6 | - | SOLUSDT | vwap_deviation | 1d | 0.647 | 1.6% | -1.7% | 2 | 1.50 | 0.005 | 200 | 0.304 |
| 7 | + | ETHUSDT | trend_multi_factor | 4h | 0.464 | 6.2% | -15.2% | 50 | 4.56 | 0.050 | 153 | 0.270 |
| 8 | - | ETHUSDT | volume_obv | 1d | 0.457 | 2.1% | -3.2% | 2 | 2.02 | 0.009 | 170 | 0.055 |
| 9 | + | ETHUSDT | supertrend_adx | 1d | 0.438 | 3.9% | -9.4% | 11 | 4.37 | 0.013 | 148 | 0.114 |
| 10 | + | ETHUSDT | keltner_channel | 4h | 0.429 | 3.7% | -6.7% | 20 | 3.70 | 0.027 | 113 | 0.200 |
| 11 | + | ETHUSDT | momentum_roc | 1d | 0.301 | 2.3% | -9.3% | 15 | 3.80 | 0.032 | 179 | 0.374 |
| 12 | ~ | ETHUSDT | volume_obv | 4h | 0.287 | 1.3% | -6.7% | 9 | 3.07 | 0.015 | 89 | 0.299 |
| 13 | ~ | BTCUSDT | atr_volatility_breakout | 1d | 0.283 | 1.5% | -7.0% | 10 | 2.95 | 0.028 | 71 | 0.089 |
| 14 | ~ | SOLUSDT | ema_ribbon | 1d | 0.215 | 2.1% | -10.8% | 23 | 4.97 | 0.001 | 113 | 0.444 |
| 15 | ~ | SOLUSDT | volume_obv | 1d | 0.134 | 0.9% | -13.1% | 21 | 2.99 | 0.014 | 171 | 0.357 |
| 16 | ~ | ETHUSDT | rsi_mean_reversion | 1d | 0.133 | 0.9% | -15.4% | 27 | 3.69 | 0.021 | 71 | 0.297 |
| 17 | ~ | BTCUSDT | supertrend_adx | 1d | 0.126 | 0.7% | -8.1% | 19 | 2.75 | 0.008 | 151 | 0.278 |
| 18 | ~ | SOLUSDT | momentum_roc | 1d | 0.101 | 0.3% | -11.0% | 23 | 3.48 | 0.037 | 143 | 0.569 |
| 19 | ~ | ETHUSDT | bollinger_breakout | 1d | 0.079 | 0.3% | -11.5% | 29 | 0.98 | 0.011 | 180 | 0.080 |
| 20 | ~ | ETHUSDT | mtf_momentum_breakout | 4h | 0.076 | 0.3% | -11.9% | 23 | 3.90 | 0.045 | 34 | 0.469 |
| 21 | ~ | ETHUSDT | stochastic_oscillator | 1d | 0.049 | -0.2% | -10.2% | 12 | 3.92 | 0.034 | 23 | 0.357 |
| 22 | ~ | ETHUSDT | regime_adaptive | 4h | 0.049 | -0.1% | -8.7% | 39 | 4.08 | 0.025 | 106 | 0.396 |
| 23 | - | BTCUSDT | breakout_regime | 1d | 0.000 | 0.0% | 0.0% | 0 | 1.56 | 0.007 | 94 | 0.203 |
| 24 | - | BTCUSDT | mtf_momentum_breakout | 1d | 0.000 | 0.0% | 0.0% | 0 | 0.60 | 0.017 | 178 | 0.696 |
| 25 | - | ETHUSDT | keltner_channel | 1d | -0.067 | -0.6% | -7.7% | 10 | 3.26 | 0.014 | 152 | 0.136 |

### Top V5b combos (risk=1% + overlay)

| # | V | Symbol | Strategy | TF | Sharpe | Ret | DD | Tr | Trail | ATR_SL | ATR_TP |
|---|---|--------|----------|-----|--------|-----|-----|-----|-------|--------|--------|
| 1 | - | BTCUSDT | supertrend_adx | 4h | 1.451 | 1.0% | -0.3% | 2 | 3.26 | 1.95 | 2.75 |
| 2 | + | ETHUSDT | ema_ribbon | 1d | 1.436 | 4.8% | -2.3% | 6 | 4.30 | 1.12 | 3.52 |
| 3 | + | ETHUSDT | supertrend | 1d | 1.424 | 2.1% | -0.6% | 6 | 3.66 | 2.94 | 5.47 |
| 4 | + | SOLUSDT | trend_multi_factor | 1d | 1.414 | 1.9% | -0.8% | 7 | 4.22 | 2.83 | 3.39 |
| 5 | + | ETHUSDT | supertrend_adx | 1d | 1.409 | 1.5% | -0.6% | 7 | 4.04 | 3.21 | 6.00 |
| 6 | + | BTCUSDT | bollinger_breakout | 1d | 1.378 | 3.0% | -1.2% | 6 | 3.41 | 2.94 | 4.34 |
| 7 | + | SOLUSDT | regime_adaptive | 1d | 1.285 | 0.9% | -0.4% | 7 | 3.19 | 3.71 | 3.58 |
| 8 | + | BTCUSDT | macd_crossover | 1d | 1.245 | 2.5% | -0.8% | 6 | 3.28 | 3.79 | 4.46 |
| 9 | - | SOLUSDT | macd_crossover | 1d | 1.194 | 2.1% | -0.7% | 2 | 4.45 | 1.16 | 7.08 |
| 10 | + | BTCUSDT | supertrend | 1d | 1.188 | 2.7% | -1.6% | 9 | 4.26 | 2.85 | 1.35 |
| 11 | + | SOLUSDT | supertrend | 1d | 1.150 | 2.0% | -1.6% | 9 | 1.97 | 2.09 | 3.18 |
| 12 | + | SOLUSDT | supertrend_adx | 1d | 1.144 | 1.2% | -1.1% | 9 | 4.60 | 3.31 | 5.43 |
| 13 | + | BTCUSDT | trend_multi_factor | 1d | 1.095 | 3.1% | -1.5% | 5 | 4.49 | 3.04 | 5.71 |
| 14 | + | SOLUSDT | mtf_momentum_breakout | 1d | 1.053 | 0.9% | -0.5% | 7 | 1.55 | 3.28 | 3.85 |
| 15 | - | ETHUSDT | breakout_regime | 1d | 1.007 | 0.2% | -0.0% | 2 | 1.04 | 3.87 | 0.00 |
| 16 | + | BTCUSDT | keltner_channel | 1d | 0.996 | 1.5% | -0.7% | 3 | 4.99 | 3.43 | 4.44 |
| 17 | + | SOLUSDT | momentum_roc | 1d | 0.993 | 1.0% | -0.5% | 11 | 1.30 | 2.79 | 7.09 |
| 18 | - | ETHUSDT | keltner_channel | 1d | 0.977 | 0.9% | -0.6% | 1 | 2.61 | 1.86 | 4.64 |
| 19 | + | ETHUSDT | ichimoku_cloud | 4h | 0.947 | 3.6% | -2.0% | 32 | 4.37 | 2.35 | 4.17 |
| 20 | + | BTCUSDT | momentum_roc | 1d | 0.901 | 2.3% | -1.1% | 6 | 2.67 | 2.31 | 5.34 |
| 21 | + | SOLUSDT | ichimoku_cloud | 1d | 0.837 | 0.7% | -0.4% | 5 | 2.34 | 2.54 | 2.37 |
| 22 | + | SOLUSDT | atr_volatility_breakout | 1d | 0.819 | 0.8% | -0.4% | 10 | 1.47 | 3.70 | 5.29 |
| 23 | - | BTCUSDT | vwap_deviation | 1d | 0.807 | 1.0% | -1.0% | 2 | 4.04 | 2.05 | 1.92 |
| 24 | + | SOLUSDT | ema_ribbon | 1d | 0.725 | 4.0% | -2.9% | 10 | 3.56 | 0.33 | 3.07 |
| 25 | + | BTCUSDT | volume_obv | 1d | 0.628 | 0.6% | -1.1% | 11 | 0.61 | 2.59 | 4.18 |

## Feature Usage Analysis

- **Trailing stop** : 32/32 STRONG (100%), avg=3.29
- **Breakeven** : 28/32 STRONG (88%), avg=0.031
- **Max holding** : 32/32 STRONG (100%), avg=84
- **ATR SL** : 32/32 STRONG (100%), avg=2.57
- **ATR TP** : 29/32 STRONG (91%), avg=4.92

## Risk Grid Comparison

| Symbol | Strategy | TF | flat | r0.5 | r1.0 | r2.0 | Best |
|--------|----------|-----|------|------|------|------|------|
| ETHUSDT | breakout_regime | 1d | 1.795 | 0.394 | 1.803 | 1.803 | r2.0 |
| SOLUSDT | mtf_momentum_breakout | 1d | 1.657 | -0.084 | -0.085 | -0.089 | flat |
| ETHUSDT | trend_multi_factor | 1d | 1.169 | 1.318 | 0.285 | 1.620 | r2.0 |
| ETHUSDT | supertrend | 1d | 0.888 | 1.550 | 1.549 | -0.008 | r0.5 |
| ETHUSDT | keltner_channel | 4h | 0.429 | -0.791 | -0.728 | 1.515 | r2.0 |
| BTCUSDT | supertrend_adx | 4h | -0.544 | - | 1.323 | - | r1.0 |
| ETHUSDT | macd_crossover | 1d | -0.369 | - | 1.253 | - | r1.0 |
| ETHUSDT | supertrend_adx | 1d | 0.438 | 0.263 | 1.183 | 0.257 | r1.0 |
| ETHUSDT | bollinger_breakout | 1d | 0.079 | - | 1.098 | - | r1.0 |
| ETHUSDT | momentum_roc | 1d | 0.301 | 0.823 | 0.094 | 0.326 | r0.5 |
| SOLUSDT | ichimoku_cloud | 4h | 0.809 | 0.694 | -0.254 | 0.601 | flat |
| ETHUSDT | supertrend | 4h | -0.552 | - | 0.763 | - | r1.0 |
| SOLUSDT | vwap_deviation | 1d | 0.647 | - | -0.946 | - | flat |
| BTCUSDT | vwap_deviation | 1d | -0.165 | - | 0.568 | - | r1.0 |
| ETHUSDT | ichimoku_cloud | 4h | -0.948 | - | 0.555 | - | r1.0 |
| SOLUSDT | momentum_roc | 1d | 0.101 | - | 0.541 | - | r1.0 |
| ETHUSDT | ema_ribbon | 1d | -0.406 | - | 0.539 | - | r1.0 |
| ETHUSDT | bollinger_breakout | 4h | -0.307 | - | 0.533 | - | r1.0 |
| ETHUSDT | keltner_channel | 1d | -0.067 | - | 0.506 | - | r1.0 |
| ETHUSDT | trend_multi_factor | 4h | 0.464 | -0.766 | 0.219 | 0.048 | flat |

## Seed Robustness (multi-seed std)

| Symbol | Strategy | TF | Risk | Sharpe | Seed_std | Robust? |
|--------|----------|-----|------|--------|----------|---------|
| SOLUSDT | ichimoku_cloud | 4h | flat | 0.809 | 0.026 | Y |
| ETHUSDT | supertrend | 1d | flat | 0.888 | 0.081 | Y |
| SOLUSDT | ichimoku_cloud | 4h | r0.5 | 0.694 | 0.091 | Y |
| ETHUSDT | supertrend_adx | 1d | flat | 0.438 | 0.114 | Y |
| ETHUSDT | supertrend_adx | 1d | r1.0 | 1.183 | 0.123 | Y |
| ETHUSDT | ichimoku_cloud | 4h | r1.0 | 0.555 | 0.169 | Y |
| ETHUSDT | breakout_regime | 1d | r2.0 | 1.803 | 0.173 | Y |
| ETHUSDT | supertrend | 1d | r0.5 | 1.550 | 0.176 | Y |
| ETHUSDT | momentum_roc | 1d | r0.5 | 0.823 | 0.180 | Y |
| ETHUSDT | supertrend | 1d | r1.0 | 1.549 | 0.187 | Y |
| SOLUSDT | ichimoku_cloud | 4h | r2.0 | 0.601 | 0.194 | Y |
| ETHUSDT | keltner_channel | 4h | flat | 0.429 | 0.200 | Y |
| ETHUSDT | bollinger_breakout | 1d | r1.0 | 1.098 | 0.219 | Y |
| ETHUSDT | keltner_channel | 4h | r2.0 | 1.515 | 0.221 | Y |
| SOLUSDT | mtf_momentum_breakout | 1d | flat | 1.657 | 0.224 | Y |
| ETHUSDT | momentum_roc | 1d | r2.0 | 0.326 | 0.225 | Y |
| ETHUSDT | trend_multi_factor | 1d | flat | 1.169 | 0.230 | Y |
| ETHUSDT | breakout_regime | 1d | r1.0 | 1.803 | 0.232 | Y |
| ETHUSDT | bollinger_breakout | 4h | r1.0 | 0.533 | 0.232 | Y |
| BTCUSDT | vwap_deviation | 1d | r1.0 | 0.568 | 0.241 | Y |

## Pool survivants (114 combos)

| # | Sym | Strategy | TF | Risk | Ov | Sharpe | Ret | DD | Trail | Seed_std | Verdict |
|---|-----|----------|-----|------|-----|--------|-----|-----|-------|----------|---------|
| 1 | ETH | breakout_regime | 1d | r2.0 | - | 1.803 | 2.4% | -0.1% | 1.04 | 0.173 | STRONG |
| 2 | ETH | breakout_regime | 1d | r1.0 | - | 1.803 | 1.2% | -0.1% | 1.04 | 0.232 | STRONG |
| 3 | ETH | breakout_regime | 1d | flat | - | 1.795 | 5.3% | -0.1% | 1.04 | 0.606 | STRONG |
| 4 | ETH | supertrend | 1d | flat | Y | 1.691 | 8.6% | -1.9% | 3.61 | 0.081 | STRONG |
| 5 | SOL | mtf_momentum_bre | 1d | flat | - | 1.657 | 9.7% | -3.5% | 1.66 | 0.224 | STRONG |
| 6 | ETH | trend_multi_fact | 1d | r2.0 | - | 1.620 | 12.3% | -4.7% | 4.39 | 0.387 | STRONG |
| 7 | SOL | atr_volatility_b | 1d | flat | Y | 1.592 | 6.4% | -1.9% | 3.37 | 0.307 | STRONG |
| 8 | BTC | trend_multi_fact | 1d | flat | Y | 1.569 | 9.0% | -2.1% | 4.50 | 0.079 | STRONG |
| 9 | ETH | supertrend | 1d | r0.5 | - | 1.550 | 3.6% | -1.8% | 3.66 | 0.176 | STRONG |
| 10 | ETH | supertrend | 1d | r1.0 | - | 1.549 | 7.3% | -3.5% | 3.66 | 0.187 | STRONG |
| 11 | SOL | mtf_momentum_bre | 1d | flat | Y | 1.518 | 3.8% | -0.8% | 1.66 | 0.224 | STRONG |
| 12 | ETH | keltner_channel | 4h | r2.0 | - | 1.515 | 5.5% | -2.0% | 4.57 | 0.221 | STRONG |
| 13 | ETH | ema_ribbon | 1d | r1.0 | Y | 1.436 | 4.8% | -2.3% | 4.30 | 0.247 | STRONG |
| 14 | ETH | supertrend | 1d | r1.0 | Y | 1.424 | 2.1% | -0.6% | 3.66 | 0.187 | STRONG |
| 15 | SOL | trend_multi_fact | 1d | r1.0 | Y | 1.414 | 1.9% | -0.8% | 4.22 | 0.295 | STRONG |
| 16 | ETH | supertrend_adx | 1d | r1.0 | Y | 1.409 | 1.5% | -0.6% | 4.04 | 0.123 | STRONG |
| 17 | BTC | bollinger_breako | 1d | r1.0 | Y | 1.378 | 3.0% | -1.2% | 3.41 | 0.022 | STRONG |
| 18 | BTC | ema_ribbon | 1d | flat | Y | 1.343 | 7.9% | -2.5% | 2.21 | 0.522 | STRONG |
| 19 | ETH | trend_multi_fact | 1d | r0.5 | - | 1.318 | 3.3% | -1.3% | 3.06 | 0.255 | STRONG |
| 20 | SOL | regime_adaptive | 1d | r1.0 | Y | 1.285 | 0.9% | -0.4% | 3.19 | 0.432 | STRONG |
| 21 | ETH | macd_crossover | 1d | r1.0 | - | 1.253 | 4.5% | -2.0% | 3.43 | 0.293 | STRONG |
| 22 | BTC | macd_crossover | 1d | r1.0 | Y | 1.245 | 2.5% | -0.8% | 3.28 | 0.412 | STRONG |
| 23 | BTC | supertrend | 1d | r1.0 | Y | 1.188 | 2.7% | -1.6% | 4.26 | 0.196 | STRONG |
| 24 | ETH | supertrend_adx | 1d | r1.0 | - | 1.183 | 3.5% | -1.4% | 4.04 | 0.123 | STRONG |
| 25 | ETH | trend_multi_fact | 1d | flat | - | 1.169 | 18.3% | -9.9% | 3.40 | 0.230 | STRONG |
| 26 | BTC | atr_volatility_b | 1d | flat | Y | 1.151 | 4.7% | -2.5% | 2.95 | 0.089 | STRONG |
| 27 | SOL | supertrend | 1d | r1.0 | Y | 1.150 | 2.0% | -1.6% | 1.97 | 0.280 | STRONG |
| 28 | SOL | supertrend_adx | 1d | r1.0 | Y | 1.144 | 1.2% | -1.1% | 4.60 | 0.703 | STRONG |
| 29 | ETH | bollinger_breako | 1d | r1.0 | - | 1.098 | 9.8% | -6.8% | 3.54 | 0.219 | STRONG |
| 30 | BTC | trend_multi_fact | 1d | r1.0 | Y | 1.095 | 3.1% | -1.5% | 4.49 | 0.422 | STRONG |
| 31 | SOL | ema_ribbon | 1d | flat | Y | 1.082 | 4.7% | -2.7% | 4.97 | 0.444 | STRONG |
| 32 | SOL | macd_crossover | 1d | flat | Y | 1.053 | 4.5% | -2.0% | 1.93 | 0.161 | STRONG |
| 33 | SOL | mtf_momentum_bre | 1d | r1.0 | Y | 1.053 | 0.9% | -0.5% | 1.55 | 0.023 | STRONG |
| 34 | SOL | trend_multi_fact | 1d | flat | Y | 1.040 | 4.5% | -2.9% | 3.56 | 0.409 | STRONG |
| 35 | BTC | keltner_channel | 1d | r1.0 | Y | 0.996 | 1.5% | -0.7% | 4.99 | 0.243 | STRONG |
| 36 | SOL | momentum_roc | 1d | r1.0 | Y | 0.993 | 1.0% | -0.5% | 1.30 | 0.277 | STRONG |
| 37 | ETH | ichimoku_cloud | 4h | r1.0 | Y | 0.947 | 3.6% | -2.0% | 4.37 | 0.169 | STRONG |
| 38 | ETH | keltner_channel | 4h | flat | Y | 0.907 | 4.3% | -3.7% | 3.70 | 0.200 | STRONG |
| 39 | BTC | momentum_roc | 1d | r1.0 | Y | 0.901 | 2.3% | -1.1% | 2.67 | 0.294 | STRONG |
| 40 | ETH | supertrend | 1d | flat | - | 0.888 | 15.3% | -14.5% | 3.61 | 0.081 | STRONG |
| 41 | BTC | supertrend_adx | 1d | flat | Y | 0.860 | 4.7% | -2.6% | 2.75 | 0.278 | STRONG |
| 42 | SOL | ichimoku_cloud | 1d | r1.0 | Y | 0.837 | 0.7% | -0.4% | 2.34 | 0.144 | STRONG |
| 43 | ETH | momentum_roc | 1d | r0.5 | - | 0.823 | 1.7% | -2.4% | 2.38 | 0.180 | STRONG |
| 44 | SOL | atr_volatility_b | 1d | r1.0 | Y | 0.819 | 0.8% | -0.4% | 1.47 | 0.254 | STRONG |
| 45 | SOL | ichimoku_cloud | 4h | flat | - | 0.809 | 7.1% | -7.5% | 2.93 | 0.026 | STRONG |
| 46 | SOL | ichimoku_cloud | 4h | flat | Y | 0.802 | 2.7% | -2.1% | 2.93 | 0.026 | STRONG |
| 47 | SOL | supertrend_adx | 1d | flat | Y | 0.791 | 3.8% | -4.1% | 4.28 | 0.475 | STRONG |
| 48 | ETH | supertrend | 4h | r1.0 | - | 0.763 | 8.6% | -14.5% | 4.44 | 0.642 | STRONG |
| 49 | SOL | ema_ribbon | 1d | r1.0 | Y | 0.725 | 4.0% | -2.9% | 3.56 | 0.055 | STRONG |
| 50 | ETH | regime_adaptive | 4h | flat | Y | 0.707 | 3.8% | -4.1% | 4.08 | 0.396 | STRONG |

---
*Genere le 13 February 2026 08:32*