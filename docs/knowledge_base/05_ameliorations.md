# Audit d'Améliorations — Quantlab V7

Propositions d'améliorations classées par impact/effort, innovations spécifiques crypto, et roadmap priorisée.

---

## Classification des améliorations

| Impact | Effort | Priorité | Amélioration | Statut |
|--------|--------|----------|-------------|--------|
| HIGH | HIGH | 🚨 Critique | Stabiliser walk-forward (seeds, multi-seed) | ✅ FAIT |
| HIGH | MEDIUM | 🔥 Urgent | Test A/B méta-opt vs defaults | ✅ FAIT → DEFAULTS gagnent |
| HIGH | MEDIUM | 🔥 Urgent | Portfolio avec covariance (Markowitz + Ledoit-Wolf) | ✅ FAIT (V4) |
| HIGH | HIGH | 🚨 Critique | Holdout temporel (12 mois) | ✅ FAIT (V4, cutoff 2025-02-01) |
| HIGH | HIGH | 🚨 Critique | Diagnostic V4 (22 strats, walk-forward + overlays) | ✅ FAIT (39 survivants) |
| HIGH | HIGH | 🚨 Critique | Régime detection + Cash overlay | ✅ FAIT (engine/regime.py + overlays.py) |
| HIGH | MEDIUM | 🔥 Urgent | Volatility targeting overlay | ✅ FAIT (engine/overlays.py) |
| HIGH | MEDIUM | 🔥 Urgent | Stratégies multi-timeframe (3 nouvelles) | ✅ FAIT (mtf_trend_entry, mtf_momentum_breakout, regime_adaptive) |
| HIGH | MEDIUM | 🔥 Urgent | Backtester fractional signals (position sizing dynamique) | ✅ FAIT |
| HIGH | MEDIUM | 🔥 Urgent | Corrélation deduplication + hard constraints | ✅ FAIT (V4 portfolio) |
| HIGH | MEDIUM | 🔥 Urgent | ATR-based SL/TP adaptatif (V5) | ✅ FAIT (22 strats, +0.254 Sharpe avg) |
| HIGH | MEDIUM | 🔥 Urgent | Risk-based position sizing (V5) | ✅ FAIT (risk_per_trade_pct) |
| HIGH | MEDIUM | 🔥 Urgent | Trailing stop + breakeven + max holding (V5b) | ✅ FAIT (22 strats) |
| HIGH | MEDIUM | 🔥 Urgent | Diagnostic V5b multi-seed + risk grid + corrélation | 🔄 EN COURS |
| ~~MEDIUM~~ | ~~LOW~~ | ~~Quick win~~ | ~~Grid search méta (déterministe)~~ | ❌ ABANDONNÉ (méta-opt inutile) |
| ~~HIGH~~ | ~~HIGH~~ | ~~Critique~~ | ~~Multi-objectif meta-optimization~~ | ❌ ABANDONNÉ (méta-opt inutile) |
| MEDIUM | MEDIUM | 📈 Moyen | Funding rate contrarian signal | ⏳ Backlog |
| MEDIUM | MEDIUM | 📈 Moyen | Cross-asset lead-lag (BTC → alts) | ⏳ Backlog |
| MEDIUM | HIGH | 📈 Moyen | Features on-chain (flows, whale alerts) | ⏳ Backlog |
| LOW | LOW | ⚡ Quick win | Tests d'invariants automatiques | ⏳ Backlog |

---

## 🚨 Améliorations CRITIQUES (impact HIGH, effort HIGH)

### 1. Stabiliser le walk-forward (SEEDS + MULTI-SEED)

**Problème actuel** : Variance énorme entre runs (-0.50 à +0.72)

**Solution** :
```python
# Dans walk_forward.py
def _optimize_on_window(..., seed=42):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed)
    )
    # ...

def robust_walk_forward(config, n_seeds=5):
    results = []
    for seed in range(n_seeds):
        result = run_walk_forward(config, seed=seed)
        results.append(result)
    return median_metrics(results)
```

**Impact** : Reproductibilité, confiance dans les résultats

**Effort** : 2-3 jours (modifications walk-forward + meta-optimizer)

---

### 2. Portfolio avec covariance (Markowitz + contraintes)

**Problème actuel** : Pondération scalaire ignore les corrélations

**Solution** :
```python
def markowitz_portfolio(returns, target_sharpe=None):
    # Matrice de covariance avec shrinkage Ledoit-Wolf
    cov = ledoit_wolf_shrinkage(returns)
    
    # Optimisation quadratique
    # max w'μ - λ w'Σw
    # s.t. Σw = 1, w ≥ 0, w_i ≤ cap_i
    
    # Contraintes :
    # - Max 50% par symbol
    # - Max 35% par stratégie
    # - Min 3 stratégies différentes
```

**Impact** : Vraie diversification, risque maîtrisé

**Effort** : 3-4 jours (cvxpy + tests)

---

### 3. Holdout temporel (validation finale)

**Problème actuel** : Pas de test sur données jamais vues

**Solution** :
```python
# Séparation stricte
TRAIN_PERIOD = [2017-01, 2023-12]  # Diagnostic + Meta-opt
VALID_PERIOD = [2024-01, 2024-12]  # Portfolio construction
TEST_PERIOD = [2025-01, 2025-12]   # Validation finale (jamais touchée)

# Pipeline modifié
def pipeline_with_holdout():
    # 1. Diagnostic sur TRAIN
    # 2. Meta-opt sur TRAIN
    # 3. Portfolio sur VALID
    # 4. Test final sur TEST (une seule fois)
```

**Impact** : Preuve de généralisation, éviter overfitting

**Effort** : 2 jours (modifications scripts + data split)

---

### 4. Multi-objectif optimization (Pareto front)

**Problème actuel** : Score composite masque les trade-offs

**Solution** :
```python
# Multi-objectif Optuna
def multi_objective_meta_opt():
    study = optuna.create_study(
        directions=["maximize", "minimize"],  # Sharpe, DD
        study_name="multi_objective_meta"
    )
    
    def objective(trial):
        # ... run walk-forward
        return sharpe, max_drawdown
    
    # Résultat : Pareto front des solutions non-dominées
    # Choix selon tolérance au risque
```

**Impact** : Transparence des trade-offs, choix éclairé

**Effort** : 3 jours (modifications meta-optimizer + visualisation)

---

## 🔥 Améliorations URGENTES (impact HIGH, effort MEDIUM)

### 5. Test A/B méta-opt vs defaults

**Problème actuel** : Jamais validé que la méta-opt apporte quelque chose

**Solution** :
```python
def test_meta_vs_defaults():
    combos = load_top_combos()
    
    # A : Meta-optimisés
    results_a = []
    for combo in combos:
        result = run_walk_forward(combo.meta_params)
        results_a.append(result)
    
    # B : Defaults fixes
    defaults = WalkForwardConfig(
        reoptim_frequency="3M",
        training_window="1Y",
        param_bounds_scale=1.0,
        optim_metric="sharpe",
        n_optim_trials=100
    )
    results_b = []
    for combo in combos:
        result = run_walk_forward(defaults)
        results_b.append(result)
    
    # Test statistique
    t_stat, p_value = scipy.stats.ttest_rel(
        [r.sharpe for r in results_a],
        [r.sharpe for r in results_b]
    )
    
    return p_value > 0.05  # True si pas de différence significative
```

**Impact** : Décider si la méta-opt vaut le coût

**Effort** : 1 jour (script simple)

---

### 6. Corrélation-aware stress tests

**Problème actuel** : Stress tests ignorent les corrélations

**Solution** :
```python
def correlation_stress_test(portfolio_returns):
    # Scénarios de corrélation
    scenarios = {
        "normal": observed_correlation,
        "crisis": np.ones_like(observed_correlation) * 0.9,
        "moderate": observed_correlation * 1.5,
        "inverse": -observed_correlation * 0.5
    }
    
    results = {}
    for name, corr in scenarios.items():
        # Simuler avec nouvelle matrice de covariance
        stressed_returns = simulate_with_correlation(
            portfolio_returns, corr
        )
        results[name] = compute_metrics(stressed_returns)
    
    return results
```

**Impact** : Mesure de la vraie robustesse

**Effort** : 2 jours

---

## 📈 Améliorations MOYENNES (impact/effort modéré)

### 7. Régimes de marché (ADX filter)

**Innovation crypto** : Les crypto ont des régimes très marqués

**Solution** :
```python
def market_regime_classifier(data):
    adx = ADX(data, period=14)
    trend_slope = linear_regression_slope(data.close, period=30)
    
    if adx > 25 and trend_slope > 0:
        return "bull_trend"
    elif adx > 25 and trend_slope < 0:
        return "bear_trend"
    else:
        return "range"

def regime_aware_portfolio():
    regimes = classify_all_periods()
    
    # Stratégies par régime
    bull_strategies = ["supertrend", "ema_ribbon"]
    bear_strategies = ["stochastic", "williams_r"]
    range_strategies = ["donchian", "bollinger"]
    
    # Allocation dynamique selon régime
```

**Impact** : Performance adaptative, réduction DD

**Effort** : 3-4 jours

---

### 8. Features on-chain (flows, whale alerts)

**Innovation crypto** : Utiliser les données on-chain pour edge

**Solution** :
```python
# Exchange flows (net inflow/outflow)
def get_exchange_flows(symbol, period):
    # API : CryptoQuant, Glassnode
    # Net flow = inflow - outflow
    # Signal : flow positif = pression achat
    
# Whale activity
def detect_whale_moves(transactions):
    # Transactions > $1M
    # Accumulation vs distribution
    
# Intégration dans stratégies
def enhanced_signals(base_signals, on_chain_features):
    # Combine price signals + on-chain
    # Ex: RSI oversold + net inflow positif = strong buy
```

**Impact** : Alpha informationnel unique

**Effort** : 4-5 jours (API + intégration)

---

### 9. Grid search méta (déterministe)

**Problème actuel** : Optuna sur espace discret = inefficace

**Solution** :
```python
def exhaustive_meta_search():
    # Espace : ~1152 combos
    reoptim_freqs = ["1M", "2M", "3M", "6M"]
    train_windows = ["3M", "6M", "1Y", "2Y"]
    bounds_scales = [0.3, 0.5, 0.8, 1.0]
    metrics = ["sharpe", "sortino", "calmar"]
    trials = [50, 100, 200]
    
    best_score = -inf
    best_config = None
    
    for freq in reoptim_freqs:
        for window in train_windows:
            for scale in bounds_scales:
                for metric in metrics:
                    for trial in trials:
                        config = WalkForwardConfig(...)
                        result = robust_walk_forward(config)
                        if result.score > best_score:
                            best_score = result.score
                            best_config = config
    
    return best_config
```

**Impact** : Vrai optimum garanti (dans l'espace testé)

**Effort** : 1 jour (simple mais long en compute)

---

## ⚡ Quick Wins (impact/effort faible)

### 10. Tests d'invariants automatiques

**Solution** :
```python
def test_all_invariants():
    # PF vs Sharpe
    assert not (profit_factor < 1 and sharpe > 1.0)
    
    # Equity monotonic (sauf trades)
    assert np.all(np.diff(equity) >= -max_trade_loss)
    
    # Returns bounds
    assert np.all(returns >= -1)  # Pas < -100%
    
    # Capital consistency
    assert equity[0] == initial_capital
```

**Impact** : Détection automatique de bugs

**Effort** : 0.5 jour

---

### 11. Dashboard live robustesse

**Solution** :
```python
# Streamlit dashboard
def robustesse_dashboard():
    # Performance live vs backtest
    # Rolling Sharpe (30j, 90j)
    # DD tracking avec alertes
    # Corrélation actuelle vs historique
    # Régime de marché actuel
```

**Impact** : Monitoring continu, détection drift

**Effort** : 2 jours

---

### 12. Margin call simulation

**Solution** :
```python
def margin_call_simulation(portfolio, leverage):
    # Maintenance margin = 10% (Binance)
    # Si equity < maintenance_margin → liquidation forcée
    
    for period in portfolio:
        equity = compute_equity(period)
        maintenance = position_value * 0.1
        
        if equity < maintenance:
            # Liquidation forcée
            return False, period
    
    return True, None
```

**Impact** : Modélisation réaliste du leverage

**Effort** : 1 jour

---

## 🚀 Innovations spécifiques crypto (long terme)

### 13. Funding rate signal

**Idée** : Utiliser le funding rate comme signal contrarian

```python
def funding_rate_signal(funding_history):
    # Funding très positif = longs payent shorts
    # Signal : short bias (contrarian)
    
    if funding_rate > 0.02:  # > 2%
        return "short_bias"
    elif funding_rate < -0.02:
        return "long_bias"
    else:
        return "neutral"
```

### 14. Liquidation heatmaps

**Idée** : Zones de liquidation massives comme S/R dynamiques

```python
def liquidation_heatmap(symbol):
    # API : Coinalyze, Hyblock
    # Zones de liquidation long/short
    # Support/résistance dynamiques basées sur le pain
```

### 15. Beta-hedging altcoins vs BTC

**Idée** : Isoler l'alpha en hedging l'exposition BTC

```python
def beta_hedged_portfolio(alt_returns, btc_returns):
    # Régression linéaire : alt = α + β*btc
    # Portfolio hedged : alt - β*btc
    # Reste l'alpha pur
```

---

## Roadmap priorisée

### ✅ Phase 1 : Stabilisation (FAIT)
1. ✅ Fixer seeds walk-forward
2. ✅ Multi-seed averaging
3. ✅ Test A/B méta-opt vs defaults

### ✅ Phase 2 : Edge & Portfolio V4 (FAIT)
4. ✅ Régime detection + overlays (regime.py, overlays.py)
5. ✅ Volatility targeting
6. ✅ 3 stratégies multi-timeframe
7. ✅ Diagnostic V4 fast (2 phases, 20 min)
8. ✅ Portfolio V4 Markowitz contraint (Ledoit-Wolf, hard constraints)
9. ✅ Holdout validation (Sharpe 2.59, DD -0.8%)

### ✅ Phase 2b : Portfolio V4b Agressif (FAIT)
10. ✅ Portfolio V4b concentré (8 combos, top3_heavy, leverage 1.5x)
11. ✅ Return +19.8% (objectif +15% atteint), Sharpe 1.35, DD -8.5%
12. ✅ Monte Carlo 5000 sims, P(gain 12M) = 86%, P(ruine) = 0%

### ✅ Phase 3 : V5 ATR-based SL/TP + Risk Sizing (FAIT)
13. ✅ ATR SL/TP adaptatif (atr_sl_mult, atr_tp_mult) — 22 strats
14. ✅ Risk-based position sizing (risk_per_trade_pct)
15. ✅ generate_signals_v5() API (signals + sl_distances)
16. ✅ Diagnostic V5 : 121 survivants, V5 > V4 (+0.254 Sharpe, 47/81 combos)

### ✅ Phase 3b : V5b Exits Avancées (FAIT)
17. ✅ Trailing stop ATR (trailing_atr_mult)
18. ✅ Breakeven stop (breakeven_trigger_pct)
19. ✅ Max holding period (max_holding_bars)
20. ✅ _apply_advanced_exits() centralisé dans BaseStrategy
21. 🔄 Diagnostic V5b (multi-seed 3, risk grid, corrélation) — EN COURS

### Phase 4 : Portfolio V5 + Déploiement (À FAIRE)
22. Portfolio V5/V5b : construction + validation + comparaison vs V4b
23. Module live/ (signal_runner, executor, scheduler)
24. Dashboard robustesse
25. Monitoring Telegram

### Phase 5 : Innovation (Backlog)
26. Funding rate contrarian signal
27. Cross-asset lead-lag
28. On-chain features
29. Beta-hedging

---

## Métriques de succès

### Avant améliorations (V3b)
- Sharpe holdout : 1.19
- DD holdout : -4.9%
- Calmar : 1.91
- Concentration ETH : 95%
- Stratégies : 19 (TA textbook)

### Après améliorations (V4) — RÉSULTATS
- **Sharpe holdout : 2.59** (+118%)
- **DD holdout : -0.8%** (+84%)
- **Calmar : 5.99** (+214%)
- **Concentration ETH : 53%** (-42pp)
- **Stratégies : 22** (+ regime_adaptive, mtf_trend_entry, mtf_momentum_breakout)
- **Overlays : regime + vol targeting**
- **37 combos diversifiés** (3 symbols, 2 TFs)
- **P(gain 12M) : 99%**
- **P(ruine) : 0.0%**

### Après V4b (concentré + leverage)
- **Return : +19.8%** (objectif +15% atteint)
- **Sharpe : 1.35**, Sortino 1.79, Calmar 2.17
- **DD : -8.5%**
- **8 combos** concentrés, leverage 1.5x
- **P(gain 12M) : 86%**, P(ruine) : 0%

### Après V5 (ATR SL/TP + risk sizing)
- **121 survivants** (vs 81 en V4)
- **47/81 combos améliorés** par V5 (+0.254 Sharpe avg)
- **Top combo** : ETH/regime_adaptive/1d Sharpe 1.569
- **22 strats** avec generate_signals_v5() API

### Après V5b (exits avancées) — EN COURS
- **Trailing stop, breakeven, max holding** ajoutés aux 22 strats
- **Diagnostic V5b** en cours (multi-seed 3, risk grid, corrélation)
- **98 tests passent** (backward compat vérifiée)

---

## ROI estimé des améliorations

| Amélioration | Effort | ROI attendu | Pourquoi |
|-------------|--------|-------------|----------|
| Seeds + multi-seed | 3j | **Énorme** | Confiance dans les résultats |
| Markowitz portfolio | 4j | **Élevé** | Vraie diversification |
| Holdout temporel | 2j | **Élevé** | Preuve de généralisation |
| Test A/B méta-opt | 1j | **Moyen** | Décider de continuer ou non |
| Régimes de marché | 4j | **Moyen** | Performance adaptative |
| On-chain features | 5j | **Variable** | Alpha potentiel unique |
| Optuna pruning | 0.5j | **Élevé** | Réduit temps compute 20-30% sans perte |

---

## ⚡ Améliorations QUICK WIN (impact LOW, effort LOW)

### 1. Optuna Pruning (MedianPruner)

**Problème actuel** : 100 trials Optuna sans pruning → temps compute inutile

**Solution** :
```python
# Dans _optimize_on_window
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,      # Attendre 5 trials avant pruning
    n_warmup_steps=3,        # Attendre 3 steps dans chaque trial
    interval_steps=1
)
study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner
)
```

**Bénéfices attendus** :
- Réduction du temps compute de 20-30%
- Pas d'impact sur qualité (pruning conservatif)
- Standard Optuna, bien testé

**Implémentation** :
- Ajouter paramètre `use_pruning: bool = True` dans `WalkForwardConfig`
- Modifier `_optimize_on_window` pour utiliser pruner conditionnellement
- Tester sur quelques combos pour valider

---

## Checklist d'implémentation

Pour chaque amélioration :
- [ ] Spécification détaillée
- [ ] Tests unitaires
- [ ] Documentation
- [ ] Validation sur données historiques
- [ ] Benchmark vs baseline
- [ ] Review code
- [ ] Déploiement

---

## Philosophie d'amélioration

1. **Stabilité avant performance** : D'abord rendre les résultats fiables
2. **Simplicité avant complexité** : Ne pas ajouter de complexité non justifiée
3. **Mesure avant optimisation** : On ne peut pas améliorer ce qu'on ne mesure pas
4. **Robustesse avant innovation** : Assurer la base avant d'innover
5. **Validation avant déploiement** : Preuves > intuitions
