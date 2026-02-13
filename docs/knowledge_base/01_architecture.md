# Architecture Technique — Quantlab V7

Architecture multi-niveaux : diagnostic 2-phases, walk-forward multi-seed, backtester vectorisé V5b (ATR SL/TP, trailing, breakeven, risk sizing), overlays adaptatifs, et portfolio Markowitz contraint.

---

## Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────┐
│              Diagnostic 2-phases                           │
│  Phase 1: Quick scan defaults → pré-filtre               │
│  Phase 2: Walk-forward multi-seed → survivants           │
└─────────────────────┴───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│          Walk-Forward (Optuna TPE, multi-seed)             │
│  Pour chaque période train/test :                           │
│  - Optuna TPE + MedianPruner optimise les params           │
│  - 3 seeds, médiane retenue                                │
│  - Backtest OOS + concaténation equity curves              │
└─────────────────────┴───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                Backtester V5b (Core)                       │
│  Vectorisé numpy, realistic costs, risk management         │
│  - Commission, slippage dynamique ATR, funding rate       │
│  - ATR SL/TP, trailing stop, breakeven, max holding       │
│  - Risk-based sizing: pos = equity×risk% / SL_distance    │
│  - Circuit breaker DD, daily loss limit                    │
└─────────────────────┴───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Overlays post-signal                          │
│  - Regime overlay (STRONG/WEAK/RANGE/CRISIS)              │
│  - Vol targeting (30% annualisée)                          │
└─────────────────────┴───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Portfolio (Markowitz + Monte Carlo)            │
│  - Covariance shrinkage Ledoit-Wolf                       │
│  - Hard constraints (symbol cap, combo cap)                │
│  - Monte Carlo stress tests (5000 sims)                   │
└─────────────────────────────────────────────────────────────┐
```

---

## 1. Backtester (`engine/backtester.py`)

### Fonctionnalités
- **Vectorisé numpy** : performance maximale sur arrays
- **Réalisme** : commission (0.1%), slippage dynamique (ATR-based), funding rate (0.01%/8h)
- **Risk management** : max position 25%, daily loss limit 3%, DD circuit breaker 15%
- **Multi-timeframe** : adaptation automatique du daily reset par TF
- **V5 : ATR SL/TP** : `atr_sl_mult × ATR / prix` adaptatif à la volatilité
- **V5 : Risk-based sizing** : `position = (equity × risk%) / SL_distance`
- **V5b : Exits avancées** : trailing stop, breakeven stop, max holding period
- **Signaux fractionnels** : support position sizing dynamique via overlays

### Fonctions clés
```python
def backtest_strategy(strategy, data, params, commission=0.001, 
                    slippage=0.0005, risk=None, timeframe="1d")
    → BacktestResult(equity, returns, trades_pnl, n_trades, ...)

def vectorized_backtest(close, signals, risk=RiskConfig(), high=None, low=None,
                       timeframe="1d", sl_distances=None)
    → BacktestResult  # V5: sl_distances pour risk-based sizing
```

### RiskConfig
```python
RiskConfig(
    max_position_pct=0.25,      # Max 25% du capital par position
    max_drawdown_pct=0.15,      # Circuit breaker DD
    max_daily_loss_pct=0.03,    # Perte journalière max
    max_trades_per_day=10,      # Limite trades/jour
    risk_per_trade_pct=0.0,     # V5: 0 = désactivé, >0 = risk-based sizing
)
```

### Risks modélisés
- **Dynamic slippage** : scale avec ATR volatilité (0.05% → 0.5% max)
- **Funding rate** : accumulation toutes les 8h sur positions ouvertes
- **Daily reset** : reset capital/position chaque jour (adapté au TF)
- **Circuit breakers** : arrêt trading si DD > 15% ou perte journalière > 3%

---

## 2. Walk-Forward (`engine/walk_forward.py`)

### Principe
1. **Split** les données en fenêtres train/test glissantes
2. **Optimize** les paramètres sur chaque fenêtre train (Optuna)
3. **Test** sur la période OOS suivante
4. **Concatène** toutes les equity curves OOS

### Configuration
```python
WalkForwardConfig(
    strategy=Strategy,
    data=DataFrame,
    timeframe="1d",
    reoptim_frequency="2M",    # Fréquence réoptimisation
    training_window="1Y",      # Taille fenêtre train
    param_bounds_scale=1.0,    # Scale des bounds autour defaults
    optim_metric="sharpe",     # Métrique d'optimisation
    n_optim_trials=100,        # Trials Optuna par fenêtre
    commission=0.001,
    slippage=0.0005,
    risk=RiskConfig
)
```

### Problème identifié et résolu
- ✅ **Seeds fixés** : `TPESampler(seed=window_seed)` par fenêtre (session 4)
- ✅ **Multi-seed** : `run_walk_forward_robust(config, n_seeds=5)` (session 4)
- ✅ **Pruning** : `MedianPruner(n_startup_trials=5, n_warmup_steps=3)` (session 6)

### Paramètres V5b optimisables
```python
V5_ATR_PARAMS = {
    "atr_sl_mult":           {"low": 0.0, "high": 4.0},          # ATR SL multiplier
    "atr_tp_mult":           {"low": 0.0, "high": 8.0},          # ATR TP multiplier
    "trailing_atr_mult":     {"low": 0.0, "high": 5.0},          # Trailing stop
    "breakeven_trigger_pct": {"low": 0.0, "high": 0.05},         # Breakeven trigger
    "max_holding_bars":      {"low": 0, "high": 200, "type": "int"}, # Max holding
}
```

---

## 3. Meta-Optimization (`engine/meta_optimizer.py`) — ABANDONNÉE

> **Test A/B (session 5)** : les defaults fixes (3M/1Y/sharpe/100 trials) font MIEUX que la méta-optimisation (+0.102 Sharpe en moyenne). La méta-opt sur-fitte les méta-paramètres.

**Décision** : defaults fixes utilisés partout. Le module existe encore mais n'est plus utilisé.

### Defaults retenus
```python
reoptim_frequency = "3M"
training_window = "1Y"
param_bounds_scale = 1.0
optim_metric = "sharpe"
n_optim_trials = 30  # avec pruning, équivalent à 100 sans
```

---

## 4. Portfolio Construction (`engine/portfolio.py`)

### Types de portefeuilles
- **Markowitz contraint** : max Sharpe avec covariance Ledoit-Wolf (V4+)
- **Markowitz min variance** : min variance avec contraintes
- **top3_heavy** : 25/25/15/10/10/5/5/5 concentré sur les meilleurs (V4b)
- **Equal weight** : poids égaux sur les combos viables
- **Sharpe-weighted** : poids ∝ max(Sharpe, 0.01)
- **Risk-parity** : poids ∝ 1/|max_drawdown|

### Contraintes (V4+)
- **Cap par symbol** : max 60% par symbol (ETH, BTC, SOL)
- **Cap par combo** : max 25% par combo
- **Déduplication** : corrélation > 0.85 → suppression du doublon
- **Covariance shrinkage** : Ledoit-Wolf (α auto-calibré)

### Monte Carlo
- 5000 simulations bootstrapées sur returns holdout
- Projections multi-horizon (3M, 6M, 12M, 24M, 36M)
- Ruin probability, P(gain), P(+10%), P(+20%)

---

## 5. Metrics (`engine/metrics.py`)

### Métriques principales
```python
sharpe_ratio(returns, timeframe="1d")         # Annualisé
sortino_ratio(returns, timeframe="1d")        # Downside deviation only
max_drawdown(equity)                          # Peak-to-trough
calmar_ratio(equity, returns, timeframe)      # Return / |DD|
total_return(equity)                          # (final - initial) / initial
win_rate(trades_pnl)                          # % trades positifs
profit_factor(trades_pnl)                     # Gross profit / gross loss
stability_score(returns, timeframe)          # Consistance des Sharpes
```

### Composite score
```python
composite_score = 0.35*sharpe + 0.25*sortino + 0.20*calmar + 0.20*stability
```

---

## 6. Data Pipeline (`data/ingestion.py`)

### Flux
1. **Binance API** → raw candles (1m) → `data/raw/`
2. **Resampling** → 5m, 15m, 1h, 4h, 1d → `data/processed/`
3. **Parquet** → compression, fast load
4. **Multi-actif** : BTC, ETH, SOL, BNB, XRP

### Gaps gérés
- **Missing candles** : forward fill OHLC, volume = 0
- **Market holidays** : pas de trading, data continue
- **Timezones** : UTC standard

---

## 7. Live Architecture (future)

### Modules prévus
- **`live/signal_runner.py`** : génère les signaux depuis les profils méta-opt
- **`live/executor.py`** : exécution sur Binance (orders, position management)
- **`live/scheduler.py`** : cron pour réoptimisations périodiques
- **`live/monitor.py`** : DD monitoring, alertes Telegram

### Déploiement
- **Mono-repo GitHub** avec `.dockerignore` (exclut scripts/, data/, results/)
- **VPS léger** (~5€/mois) : exécution live seulement
- **CI/CD** : GitHub Actions → SSH → restart service

---

## 8. Dépendances & Stack

- **Python 3.12** : typing, performance
- **NumPy/Pandas** : vectorisation, manipulation de données
- **Optuna 3.x** : optimisation bayésienne (TPE sampler)
- **Parquet** : stockage données efficient
- **Loguru** : logging structuré
- **Click** : CLI
- **Streamlit** : dashboard (optionnel)

---

## 9. Tests & Qualité

- **98 tests** dans `tests/` (tous passants)
- **Coverage** : backtester, walk-forward, metrics, 22 stratégies (shape, values, dtype)
- **Invariants** : tests de cohérence (PF vs return, equity monotonic)
- **Backward compat** : V5b params à 0 = comportement V4 identique
- **CI** : GitHub Actions sur chaque PR

---

## 10. Points de vigilance résolus

| Problème | Statut | Solution |
|----------|--------|----------|
| Stochasticité non contrôlée | ✅ Résolu | Seeds fixés + multi-seed (session 4) |
| Variance walk-forward | ✅ Résolu | Multi-seed averaging, médiane retenue |
| Portfolio sans covariance | ✅ Résolu | Markowitz + Ledoit-Wolf (V4) |
| Pas de holdout final | ✅ Résolu | Cutoff 2025-02-01, 12 mois holdout |
| Méta-opt non validée | ✅ Résolu | Test A/B → defaults gagnent, méta-opt abandonnée |
| SL/TP fixes non adaptatifs | ✅ Résolu | ATR-based SL/TP (V5) |
| Pas de trailing stop | ✅ Résolu | Trailing + breakeven + max hold (V5b) |

### Points restants
1. **Leverage linéaire** : modélisation simpliste (pas de margin calls)
2. **Concentration ETH** : 70% du portfolio V4b
3. **Module live/** : pas encore implémenté

---

## 11. Modules spécifiques

### `engine/regime.py` — Détection de régime
- **STRONG_TREND** : ADX > 25, trend fort
- **WEAK_TREND** : ADX 15-25, trend faible
- **RANGE** : ADX < 15, marché latéral
- **CRISIS** : vol spike 3× ET DD > 20% (les deux requis)
- Calibré crypto : pas de faux positifs en vol normale

### `engine/overlays.py` — Pipeline post-signal
- **Regime overlay** : coupe signaux en RANGE/CRISIS (hard cutoff)
- **Vol targeting** : normalise exposition pour 30% vol annualisée
- Pipeline chaînable : regime → vol targeting
- Résultat V4 : overlays améliorent 59% des combos, DD réduit massivement

### `strategies/base.py` — BaseStrategy
- `compute_atr(high, low, close, period)` : ATR Wilder partagé
- `get_sl_tp_mode(params)` : détecte ATR vs pct mode
- `_apply_advanced_exits(signals, data, params)` : trailing, breakeven, max hold
- `generate_signals_v5(data, params)` : API V5 retournant (signals, sl_distances)
