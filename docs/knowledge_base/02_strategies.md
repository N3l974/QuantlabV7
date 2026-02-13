# Catalogue des Stratégies — Quantlab V7

22 stratégies couvrant 10 familles différentes avec leurs paramètres, forces, faiblesses et résultats typiques.

Toutes les stratégies supportent :
- **V5** : `atr_sl_mult`, `atr_tp_mult`, `atr_period` (ATR-based SL/TP adaptatif)
- **V5** : `generate_signals_v5()` retournant `(signals, sl_distances)` pour risk-based sizing
- **V5b** : `trailing_atr_mult`, `breakeven_trigger_pct`, `max_holding_bars` (exits avancées)
- **Backward compatible** : tous les params V5/V5b à 0 = comportement V4 identique

---

## Répartition par famille

| Famille | Stratégies | Count | Description |
|---------|------------|-------|-------------|
| **Mean-Reversion** | rsi_mean_reversion, stochastic_oscillator, mean_reversion_zscore | 3 | Retour à la moyenne sur surachat/survente |
| **Trend-Following** | macd_crossover, ema_ribbon, ichimoku_cloud, supertrend | 4 | Suivi de tendance avec confirmation |
| **Breakout** | bollinger_breakout, donchian_channel, keltner_channel | 3 | Cassures de niveaux supports/résistances |
| **Volatility** | atr_volatility_breakout | 1 | Breakout basé sur volatilité |
| **Volume** | volume_obv | 1 | Signaux basés sur volume |
| **Momentum** | momentum_roc, williams_r | 2 | Oscillateurs de momentum |
| **Regime Filter** | adx_regime | 1 | Détection de régime trend/range |
| **Multi-Factor** | supertrend_adx, trend_multi_factor, breakout_regime | 3 | Combinaison multi-indicateurs |
| **Regime-Switching** | regime_adaptive | 1 | Adapte comportement au régime marché |
| **Multi-Timeframe** | mtf_trend_entry, mtf_momentum_breakout | 2 | HTF trend + LTF entry |

---

## Stratégies détaillées

### 1. RSI Mean Reversion (`rsi_mean_reversion.py`)

**Famille** : Mean-Reversion  
**Principe** : Surachat (>70) → sell, survente (<30) → buy

**Paramètres** :
- `rsi_period` (10-30) : période RSI
- `oversold` (20-40) : seuil survente
- `overbought` (60-80) : seuil surachat
- `stop_loss_pct` (0.02-0.10) : stop loss
- `take_profit_pct` (0.02-0.10) : take profit

**Forces** :
- Simple, robuste
- Fonctionne bien en range

**Faiblesses** :
- Perd en trend fort
- Signaux fréquents → overtrading

**Résultats typiques** : Sharpe 0.2-0.6, DD 15-25%

---

### 2. Stochastic Oscillator (`stochastic_oscillator.py`)

**Famille** : Mean-Reversion  
**Principe** : Stochastic %K/%D croisements avec surachat/survente

**Paramètres** :
- `k_period` (10-25) : période %K
- `d_period` (2-5) : période %D
- `oversold` (5-20) : seuil oversold
- `overbought` (80-95) : seuil overbought
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- **Meilleure stratégie actuelle** (XRP 1d)
- Momentum + mean-reversion

**Faiblesses** :
- Sensible aux paramètres
- Whipsaws en sideways

**Résultats typiques** : Sharpe 0.4-0.8, DD 10-20%

---

### 3. Mean Reversion Z-Score (`mean_reversion_zscore.py`)

**Famille** : Statistical/Mean-Reversion  
**Principe** : Z-score du prix → entrée quand |z| > threshold

**Paramètres** :
- `lookback_period` (10-50) : période rolling
- `z_threshold` (1.5-3.0) : seuil d'entrée
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Approche statistique rigoureuse
- Adaptatif à la volatilité

**Faiblesses** :
- Hypothèse normalité (fausse en crypto)
- Lent à réagir aux changements de régime

**Résultats typiques** : À évaluer (nouveau)

---

### 4. MACD Crossover (`macd_crossover.py`)

**Famille** : Trend-Following  
**Principe** : MACD line crossover signal line

**Paramètres** :
- `fast_ema` (8-15) : EMA rapide
- `slow_ema` (20-30) : EMA lente
- `signal_period` (8-12) : EMA signal
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Indicateur classique, bien documenté
- Bon pour trends modérés

**Faiblesses** :
- Lag important
- Signaux tardifs

**Résultats typiques** : Sharpe 0.1-0.4, DD 20-30%

---

### 5. EMA Ribbon (`ema_ribbon.py`)

**Famille** : Trend-Following  
**Principe** : Multiple EMAs → tendance basée sur alignement

**Paramètres** :
- `fast_ema` (8-15) : EMA rapide
- `medium_ema` (20-35) : EMA moyenne
- `slow_ema` (40-60) : EMA lente
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Visuel, intuitif
- Multiple timeframes dans une stratégie

**Faiblesses** :
- Trop de paramètres
- Performance moyenne sur crypto

**Résultats typiques** : Sharpe 0.1-0.3, DD 15-25%

---

### 6. Ichimoku Cloud (`ichimoku_cloud.py`)

**Famille** : Trend-Following  
**Principe** : Ichimoku complet (Tenkan, Kijun, Senkou, Chikou)

**Paramètres** :
- `tenkan_period` (8-12) : conversion
- `kijun_period` (20-30) : base
- `senkou_span_b_period` (40-60) : lagging span B
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Très complet (tendance + support/résistance)
- Système japonais éprouvé

**Faiblesses** :
- Complexe à debug
- Trop de signaux

**Résultats typiques** : Sharpe 0.0-0.3, DD 20-35%

---

### 7. SuperTrend (`supertrend.py`)

**Famille** : Trend-Following (trailing stop)  
**Principe** : ATR bands dynamiques avec trailing stop

**Paramètres** :
- `atr_period` (8-20) : période ATR
- `atr_multiplier` (2.0-4.0) : multiplicateur
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Trailing stop automatique
- Adaptatif à la volatilité

**Faiblesses** :
- Nouveau, peu testé
- Peut whipsaw en choppy

**Résultats typiques** : À évaluer (nouveau)

---

### 8. Bollinger Breakout (`bollinger_breakout.py`)

**Famille** : Breakout  
**Principe** : Cassure des bandes de Bollinger

**Paramètres** :
- `bb_period` (15-25) : période BB
- `bb_std` (1.5-2.5) : écart-type
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Volatilité intégrée
- Bon pour breakouts

**Faiblesses** :
- Faux breakouts fréquents
- Sensible aux paramètres

**Résultats typiques** : Sharpe 0.1-0.4, DD 20-30%

---

### 9. Donchian Channel (`donchian_channel.py`)

**Famille** : Breakout  
**Principe** : Highest high/Lowest low sur N périodes

**Paramètres** :
- `channel_period` (15-30) : période canal
- `exit_period` (5-15) : période sortie
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Simple, efficace
- **Bonne performance sur ETH 1d**

**Faiblesses** :
- Lag en trend rapide
- Ne capture pas les retournements

**Résultats typiques** : Sharpe 0.3-0.7, DD 15-25%

---

### 10. Keltner Channel (`keltner_channel.py`)

**Famille** : Volatility Channel/Breakout  
**Principe** : EMA ± ATR × multiplier

**Paramètres** :
- `ema_period` (15-25) : période EMA
- `atr_period` (10-20) : période ATR
- `atr_multiplier` (1.5-3.0) : multiplicateur
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Canal adaptatif à volatilité
- Mix de trend + volatilité

**Faiblesses** :
- Nouveau, peu testé
- Similaire à Bollinger

**Résultats typiques** : À évaluer (nouveau)

---

### 11. ATR Volatility Breakout (`atr_volatility_breakout.py`)

**Famille** : Volatility/Breakout  
**Principe** : Breakout quand price > ATR bands

**Paramètres** :
- `atr_period` (20-35) : période ATR
- `atr_multiplier` (2.0-4.0) : multiplicateur
- `lookback_period` (10-25) : lookback pour breakout
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- **Bonne performance sur ETH 1d**
- Volatilité-filtered entries

**Faiblesses** :
- Complexité modérée
- Timing critique

**Résultats typiques** : Sharpe 0.3-0.6, DD 15-20%

---

### 12. Volume OBV (`volume_obv.py`)

**Famille** : Volume  
**Principe** : On-Balance Volume divergences

**Paramètres** :
- `obv_ma_period` (10-30) : MA sur OBV
- `volume_ma_period` (10-30) : MA sur volume
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Confirme les mouvements par volume
- Indépendant du prix

**Faiblesses** :
- Volume peut être manipulé
- Signaux rares

**Résultats typiques** : Sharpe 0.0-0.3, DD 25-35%

---

### 13. VWAP Deviation (`vwap_deviation.py`)

**Famille** : Mean-Reversion  
**Principe** : Écart par rapport à VWAP intraday

**Paramètres** :
- `vwap_period` (20-40) : période VWAP
- `deviation_threshold` (0.5-2.0%) : seuil d'écart
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Mean-reversion sur prix moyen
- Adaptatif au volume

**Faiblesses** :
- Calcul VWAP complexe
- Moins efficace sur daily+

**Résultats typiques** : Sharpe 0.1-0.3, DD 20-30%

---

### 14. Momentum ROC (`momentum_roc.py`)

**Famille** : Momentum  
**Principe** : Dual Rate of Change (fast/slow)

**Paramètres** :
- `fast_roc_period` (5-15) : ROC rapide
- `slow_roc_period` (20-40) : ROC lent
- `exit_threshold` (0.05-0.20) : seuil sortie
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Momentum multi-timeframe
- Simple mais efficace

**Faiblesses** :
- Bounds bugs (fixés)
- Peu testé

**Résultats typiques** : À évaluer (nouveau)

---

### 15. Williams %R (`williams_r.py`)

**Famille** : Momentum  
**Principe** : Williams %R overbought/oversold

**Paramètres** :
- `williams_period` (10-20) : période Williams
- `oversold_threshold` (80-95) : seuil oversold
- `overbought_threshold` (5-20) : seuil overbought
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Similaire à Stochastic mais plus réactif
- Bon pour momentum extrême

**Faiblesses** :
- Whipsaws fréquent
- Bugs bounds (fixés)

**Résultats typiques** : À évaluer (nouveau)

---

### 16. ADX Regime (`adx_regime.py`)

**Famille** : Regime Filter  
**Principe** : ADX + trend slope pour détecter trend/range

**Paramètres** :
- `adx_period` (10-20) : période ADX
- `trend_period` (20-40) : période tendance
- `adx_threshold` (20-35) : seuil trend
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Filtre de régime
- Peut combiner avec autres stratégies

**Faiblesses** :
- Plus un filtre qu'une stratégie
- Signaux rares

**Résultats typiques** : Sharpe 0.0-0.2, DD 15-25%

---

## Performances observées (Diagnostic V2)

| Stratégie | Meilleur combo | Sharpe | Confiance |
|-----------|----------------|--------|-----------|
| stochastic_oscillator | XRPUSDT/1d | **0.58** | HIGH |
| donchian_channel | ETHUSDT/1d | **0.73** | HIGH |
| atr_volatility_breakout | ETHUSDT/1d | **0.51** | MEDIUM |
| ema_ribbon | BTCUSDT/1d | **0.31** | MEDIUM |
| donchian_channel | BTCUSDT/1d | **0.39** | MEDIUM |
| rsi_mean_reversion | SOLUSDT/15m | **0.47** | MEDIUM |
| bollinger_breakout | SOLUSDT/4h | **0.18** | MEDIUM |
| **Autres** | - | < 0.2 | LOW |

**Observations** :
- **Daily timeframe** domine (4/5 meilleurs combos)
- **ETH/XRP** plus prometteurs que BTC
- **Mean-reversion et breakout** sont les familles les plus performantes

---

## Stratégies Multi-Factor (nouvelles, 12 février 2026)

### 17. SuperTrend + ADX (`supertrend_adx.py`)

**Famille** : Multi-Factor (Trend + Regime)
**Principe** : SuperTrend pour la direction + ADX pour filtrer les marchés en range + cooldown

**Paramètres** :
- `atr_period` (8-20) : période ATR pour SuperTrend
- `st_multiplier` (2.0-4.0) : multiplicateur SuperTrend
- `adx_period` (10-20) : période ADX
- `adx_threshold` (15-30) : seuil trending
- `cooldown_bars` (2-8) : barres min entre trades
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- Élimine les whipsaws en range (ADX filter)
- Cooldown réduit l'overtrading
- Bon survivant holdout sur ETH/4h et BTC/4h

**Faiblesses** :
- Haute variance inter-seeds (std > 1.0 sur certains combos)
- Mauvais sur 1d (trop peu de signaux)

**Résultats holdout** : ETH/4h HO Sharpe 0.694, BTC/4h HO Sharpe 0.194

---

### 18. Trend Multi-Factor (`trend_multi_factor.py`)

**Famille** : Multi-Factor (Trend + Volume + Momentum)
**Principe** : Confluence 3/3 requise — SuperTrend + OBV slope + ROC momentum

**Paramètres** :
- `atr_period` (8-20) : ATR pour SuperTrend
- `st_multiplier` (2.0-4.0) : multiplicateur SuperTrend
- `obv_slope_period` (5-20) : lookback pente OBV
- `roc_period` (8-25) : période ROC
- `roc_threshold` (0.5-3.0) : seuil % ROC
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- **Très stable** : ETH/4h std=0.13 (le plus stable de tous)
- Confluence réduit les faux signaux
- Fonctionne sur ETH et BTC

**Faiblesses** :
- Signaux rares (confluence stricte)
- Mauvais sur SOL (volatilité trop élevée)

**Résultats holdout** : ETH/4h HO Sharpe 0.180 (STRONG), ETH/1d 0.779, BTC/1d 0.321

---

### 19. Breakout + Regime (`breakout_regime.py`)

**Famille** : Multi-Factor (Breakout + Regime + Volume)
**Principe** : ATR breakout filtré par ADX trending + volume spike confirmation + cooldown

**Paramètres** :
- `atr_period` (10-20) : période ATR
- `atr_multiplier` (1.5-3.0) : multiplicateur breakout
- `lookback_period` (15-30) : SMA reference
- `adx_period` (10-20) : période ADX
- `adx_threshold` (15-30) : seuil trending
- `vol_ma_period` (10-30) : MA volume
- `vol_spike_ratio` (1.0-2.0) : ratio spike volume
- `cooldown_bars` (3-10) : cooldown
- `stop_loss_pct`, `take_profit_pct`

**Forces** :
- **Meilleur survivant holdout** : ETH/4h HO Sharpe 0.935
- Triple filtre (ADX + volume + ATR) élimine les faux breakouts
- SOL/1d ultra-stable (std=0.02)

**Faiblesses** :
- 10 paramètres (risque de sur-fitting)
- Catastrophique sur BTC/4h et SOL/4h

**Résultats holdout** : ETH/4h HO Sharpe **0.935** (STRONG), SOL/1d 0.015 (STRONG)

---

### 20. Regime Adaptive (`regime_adaptive.py`)

**Famille** : Regime-Switching  
**Principe** : Adapte son comportement au régime marché détecté (ADX + vol + DD)

**Comportement par régime** :
- **STRONG_TREND** : SuperTrend trend-following
- **WEAK_TREND** : SuperTrend avec sizing réduit
- **RANGE** : Bollinger Bands mean-reversion
- **CRISIS** : Cash (pas de position)

**Paramètres** :
- `st_atr_period`, `st_multiplier` : SuperTrend pour trend
- `adx_period`, `adx_threshold` : Détection de régime
- `bb_period`, `bb_std` : Bollinger pour range
- `cooldown_bars` : Min barres entre trades
- `stop_loss_pct`, `take_profit_pct`, `atr_sl_mult`, `atr_tp_mult`

**Forces** :
- **Top combo V5** : ETH/1d Sharpe 1.569 avec overlays
- S'adapte automatiquement au marché
- Cash en crise = protection naturelle

**Faiblesses** :
- Beaucoup de paramètres (risque sur-fitting)
- Détection de régime peut être en retard

---

### 21. MTF Trend Entry (`mtf_trend_entry.py`)

**Famille** : Multi-Timeframe  
**Principe** : HTF SuperTrend pour direction + LTF RSI pullback pour timing d'entrée

**Paramètres** :
- `htf_atr_period`, `htf_st_multiplier` : SuperTrend HTF
- `rsi_period`, `rsi_oversold`, `rsi_overbought` : RSI LTF
- `cooldown_bars` : Min barres entre trades
- `stop_loss_pct`, `take_profit_pct`, `atr_sl_mult`, `atr_tp_mult`

**Forces** :
- Combine trend macro + timing micro
- RSI pullback = meilleur prix d'entrée

**Faiblesses** :
- Dépend de la qualité du signal HTF
- Signaux rares si trend et RSI ne s'alignent pas

---

### 22. MTF Momentum Breakout (`mtf_momentum_breakout.py`)

**Famille** : Multi-Timeframe  
**Principe** : HTF momentum (ROC + ADX) pour direction + LTF Donchian breakout pour entrée

**Paramètres** :
- `htf_roc_period`, `htf_roc_threshold` : ROC momentum HTF
- `adx_period`, `adx_threshold` : ADX filter HTF
- `donchian_period` : Donchian channel LTF
- `vol_ma_period`, `vol_spike_ratio` : Volume filter
- `cooldown_bars` : Min barres entre trades
- `stop_loss_pct`, `take_profit_pct`, `atr_sl_mult`, `atr_tp_mult`

**Forces** :
- Momentum HTF + breakout LTF = double confirmation
- Volume spike filter réduit les faux breakouts

**Faiblesses** :
- Beaucoup de paramètres
- Dépend de la qualité du signal momentum HTF

---

## Paramètres V5/V5b (communs à toutes les stratégies)

### V5 — ATR-based SL/TP + Risk Sizing

| Paramètre | Default | Range | Description |
|-----------|---------|-------|-------------|
| `atr_sl_mult` | 0.0 | [0, 4] | Multiplicateur ATR pour stop-loss (0 = utilise stop_loss_pct) |
| `atr_tp_mult` | 0.0 | [0, 8] | Multiplicateur ATR pour take-profit (0 = utilise take_profit_pct) |
| `atr_period` | 14 | [7, 28] | Période ATR pour SL/TP |

**API V5** : `generate_signals_v5(data, params)` retourne `(signals, sl_distances)` où `sl_distances[i] = atr_sl_mult × ATR[i] / close[i]`.

### V5b — Exits Avancées

| Paramètre | Default | Range | Description |
|-----------|---------|-------|-------------|
| `trailing_atr_mult` | 0.0 | [0, 5] | Trailing stop ATR (0 = désactivé) |
| `breakeven_trigger_pct` | 0.0 | [0, 0.05] | Seuil de gain pour breakeven (0 = désactivé) |
| `max_holding_bars` | 0 | [0, 200] | Max barres en position (0 = désactivé) |

**Logique** (dans `BaseStrategy._apply_advanced_exits()`) :
1. **Trailing** : suit le highest/lowest price depuis l'entrée, ferme si prix recule de `trailing_atr_mult × ATR`
2. **Breakeven** : si gain > `breakeven_trigger_pct`, SL ramené au prix d'entrée
3. **Max hold** : sortie forcée après `max_holding_bars` barres

---

## Performances observées — Holdout Validation (12 février 2026)

### Stratégies simples (Diagnostic V4 → Holdout)

| Stratégie | Meilleur combo | HO Sharpe | Verdict |
|-----------|----------------|-----------|---------|
| supertrend | ETHUSDT/4h | **0.444** | ✅ STRONG |
| atr_volatility_breakout | ETHUSDT/1d | 0.397 | ⚠️ WEAK |
| momentum_roc | BTCUSDT/1d | -0.028 | ⚠️ WEAK |
| volume_obv | ETHUSDT/4h | -0.902 | ❌ FAIL |
| bollinger_breakout | SOLUSDT/4h | -1.354 | ❌ FAIL |

### Stratégies multi-factor (Holdout)

| Stratégie | Meilleur combo | HO Sharpe | Verdict |
|-----------|----------------|-----------|---------|
| breakout_regime | ETHUSDT/4h | **0.935** | ✅ STRONG |
| trend_multi_factor | ETHUSDT/1d | 0.779 | ⚠️ WEAK |
| supertrend_adx | ETHUSDT/4h | 0.694 | ⚠️ WEAK |

### Leçon clé
**Multi-factor > Simple** : le meilleur multi-factor (0.935) bat le meilleur simple (0.444) de 2x.
Les filtres de régime (ADX) et de volume réduisent les faux signaux et améliorent la robustesse.

---

## Recommandations d'utilisation (mises à jour)

### Pour portfolio diversifié (validé holdout)
- **1 breakout multi-factor** : breakout_regime (ETH/4h) — **champion**
- **1 trend simple** : supertrend (ETH/4h) — robuste
- **1 trend multi-factor** : trend_multi_factor (ETH/4h) — très stable
- **1 breakout stable** : breakout_regime (SOL/1d) — ultra-stable

### Actifs par ordre de "tradabilité"
1. **ETHUSDT** — 7/11 survivants, le plus fiable
2. **BTCUSDT** — 3/11 survivants, modéré
3. **SOLUSDT** — 1/11 survivant, difficile

### Timeframes recommandés
- **4h** : sweet spot (6/11 survivants), bon compromis signal/bruit
- **1d** : viable mais moins de signaux
- **15m/1h** : non recommandé (trop de bruit, coûts élevés)

### Principes de design validés
1. **Filtres de régime** (ADX) améliorent toutes les stratégies
2. **Confirmation volume** réduit les faux breakouts
3. **Cooldown** entre trades réduit l'overtrading
4. **Confluence multi-indicateur** > indicateur unique
