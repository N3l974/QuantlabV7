# Méthodologie de Recherche — Quantlab V7

Pipeline de recherche, biais identifiés, failles méthodologiques et bonnes pratiques pour maintenir la rigueur scientifique.

---

## Pipeline de recherche actuel (V5b)

```
1. Ingestion → 2. Diagnostic 2-phases → 3. Portfolio Markowitz → 4. Stress tests → 5. Déploiement
```

### Étape 1 : Ingestion (`data/ingestion.py`)
- **Input** : Binance API (1m candles)
- **Output** : Parquet multi-TF (5m, 15m, 1h, 4h, 1d)
- **Validation** : Continuité temporelle, gaps gérés

### Étape 2 : Diagnostic 2-phases (`scripts/diagnostic_v5b.py`)
- **Phase 1** : Quick scan defaults sur holdout (132 combos → ~80 survivants)
  - Backtest rapide avec params par défaut sur données post-cutoff
  - Filtre : Sharpe > -1.5, min 3 trades
- **Phase 2** : Walk-forward multi-seed sur survivants
  - Optuna TPE + MedianPruner, 30 trials, 3M reoptim, 1Y window
  - 3 seeds par combo, médiane retenue, std mesuré
  - Holdout validation (cutoff 2025-02-01)
  - Baseline + overlay variants
  - V5b : ATR SL/TP + trailing + breakeven + max hold optimisables
  - Risk grid : flat, r0.5%, r1.0%, r2.0%
  - Corrélation matrix des survivants STRONG
- **Output** : `portfolio/v5b/results/diagnostic_v5b_{ts}.json` + rapport markdown

### Étape 3 : Portfolio (`scripts/portfolio_v4b_final.py`)
- **Scope** : Survivants STRONG du diagnostic
- **Méthode** : Markowitz contraint (Ledoit-Wolf), top3_heavy, leverage testing
- **Contraintes** : cap symbol 60%, cap combo 25%, déduplication corr > 0.85
- **Output** : Allocation optimale, Monte Carlo stress tests

### Étape 4 : Stress tests
- **Monte Carlo** : 5000 sims bootstrapées sur returns holdout
- **Projections** : Multi-horizon (3M, 6M, 12M, 24M, 36M)
- **Ruin probability** : P(perte > 50%)
- **Stress months** : pire/meilleur mois, pire trimestre

### Étape 5 : Déploiement (future)
- Paper trading 2-4 semaines
- Module live/ (signal_runner, executor, scheduler)
- Monitoring Telegram

> **Note** : La méta-optimisation (ancienne étape 3) a été **abandonnée** après le test A/B (session 5). Les defaults fixes font mieux.

---

## Biais identifiés dans le pipeline

### Biais #1 : Sélection post-optimisation (Winner Selection Bias)

**Description** :
- On teste 216 combos en diagnostic
- On sélectionne les 5 meilleurs
- On les méta-optimise
- On les met en portfolio

**Problème** : Les 5 "meilleurs" sont probablement les plus chanceux, pas les meilleurs intrinsèquement.

**Preuve** : Variance énorme entre runs (même combo : Sharpe -0.50 à +0.72)

**Impact** : Sur-estimation de la performance future

---

### Biais #2 : Data Snooping cumulé

**Description** :
- Même données utilisées pour diagnostic, méta-opt, portfolio
- Pas de holdout final jamais touché
- Chaque optimisation "apprend" les spécificités des données

**Problème** : Performance qui va se dégrader en live (overfitting aux données historiques)

**Impact** : Sharpes observés 0.5-0.8 → Sharpes live probablement 0.2-0.4

---

### Biais #3 : Variance non contrôlée

**Description** :
- Walk-forward stochastique (pas de seed)
- 1 seul run par évaluation
- On optimise du bruit

**Problème** : Les "meilleurs" paramètres sont aléatoires

**Impact** : Instabilité des résultats, non-reproductibilité

---

### Biais #4 : Pondération scalaire vs optimisation portefeuille

**Description** :
- sharpe_weighted = poids ∝ Sharpe individuel
- Pas de matrice de covariance
- Corrélations ignorées

**Problème** : Diversification apparente mais risque caché

**Preuve** : Portfolio V2 : 2 stratégies sur ETH = 65% du poids

**Impact** : Sous-estimation du risque réel

---

### Biais #5 : Look-ahead implicite

**Description** :
- Diagnostic utilise toute l'histoire disponible
- Meta-opt utilise les mêmes données
- Pas de séparation temporelle stricte

**Problème** : Information du futur "fuite" dans le passé

**Impact** : Performance sur-optimiste

---

## Failles méthodologiques

### Faille #1 : Pas de baseline simple

**Ce qui manque** :
- Test A/B : meta-opt params vs defaults fixes
- Test random : params aléatoires vs optimaux
- Test simple : buy-and-hold, stratégies naïves

**Conséquence** : On ne sait pas si la complexité ajoute de la valeur

---

### Faille #2 : Pas de validation out-of-sample finale

**Ce qui manque** :
- Holdout period (dernière 12-24 mois) jamais utilisée
- Test final unique sur cette période
- Validation de la robustesse temporelle

**Conséquence** : Pas de preuve de généralisation

---

### Faille #3 : Single-run evaluation

**Ce qui manque** :
- Multi-seed averaging pour chaque évaluation
- Intervalles de confiance sur les métriques
- Tests de significativité statistique

**Conséquence** : Pas de notion d'incertitude

---

### Faille #4 : Espace de recherche mal défini

**Problème** :
- Meta-params discrets mais traités comme continus
- Espace petit (~1152 combos) mais exploré aléatoirement
- Pas de garantie de trouver l'optimum global

**Conséquence** : Optimisation inefficace

---

### Faille #5 : Risk modeling incomplet

**Manque** :
- Margin calls (leverage > 1x)
- Slippage extrême en crise
- Corrélation en crash (tous les actifs baissent ensemble)
- Liquidité limitée

**Conséquence** : Sous-estimation du risque extrême

---

## Bonnes pratiques à implémenter

### 1. Séparation temporelle stricte

```
Train : [2017-01, 2023-12]  ← Diagnostic + Meta-opt
Valid : [2024-01, 2024-12]  ← Portfolio construction
Test  : [2025-01, 2025-12]  ← Validation finale (jamais touchée)
```

### 2. Multi-seed systématique

```python
def robust_walk_forward(config, n_seeds=5):
    results = []
    for seed in range(n_seeds):
        np.random.seed(seed * 42 + 7)
        result = run_walk_forward(config)
        results.append(result)
    
    # Prendre la médiane, pas le max
    median_metrics = compute_median_metrics(results)
    return median_metrics
```

### 3. Baselines systématiques

- **Defaults fixes** : reoptim=3M, window=1Y, metric=sharpe, trials=100
- **Random search** : params aléatoires dans l'espace méta
- **Buy-and-hold** : performance passive de chaque actif
- **Equal weight portfolio** : benchmark simple

### 4. Tests de significativité

```python
def test_significance(strategy_a, strategy_b, n_bootstraps=1000):
    # Bootstrap des returns
    diff_samples = []
    for _ in range(n_bootstraps):
        sample_a = bootstrap_returns(strategy_a.returns)
        sample_b = bootstrap_returns(strategy_b.returns)
        diff = sample_a.sharpe - sample_b.sharpe
        diff_samples.append(diff)
    
    p_value = np.mean(np.abs(diff_samples) >= abs(observed_diff))
    return p_value
```

### 5. Portfolio-level optimization

```python
def mean_variance_portfolio(returns, target_sharpe=None):
    # Markowitz avec covariance
    # Contraintes : sum(w) = 1, w >= 0, max par actif/stratégie
    # Objectif : maximiser Sharpe ou minimiser variance pour target return
```

---

## Métriques de robustesse

### 1. Stability Score
```python
stability = 1 - std(sub_period_sharpes) / mean(sub_period_sharpes)
# > 0.7 = stable, < 0.3 = volatile
```

### 2. Cross-seed consistency
```python
consistency = 1 - std(seed_scores) / mean(seed_scores)
# Mesure la reproductibilité
```

### 3. Out-of-sample decay
```python
decay = oos_sharpe / is_sharpe
# < 0.7 = overfitting suspect
```

### 4. Correlation stress impact
```python
stress_impact = portfolio_sharpe(correlation=1.0) / portfolio_sharpe(correlation=observed)
# Test la sensibilité aux corrélations
```

---

## Processus de validation amélioré

### Phase 1 : Exploration robuste
- Multi-seed diagnostic
- Intervalles de confiance
- Filtrage par stabilité (pas seulement par performance)

### Phase 2 : Optimisation contrôlée
- Baselines obligatoires
- Grid search déterministe (si espace petit)
- Multi-objectif (Sharpe vs DD)

### Phase 3 : Validation temporelle
- Holdout final
- Test de dégradation temporelle
- Scénarios de crise (2008, 2020, 2022)

### Phase 4 : Portfolio optimisé
- Covariance-aware allocation
- Contraintes réalistes
- Stress tests multi-dimensionnels

---

## Checklist méthodologique

### Avant toute optimisation
- [x] Seeds fixés partout ✅ (session 4, 12 fév)
- [x] Baselines définis ✅ (test A/B, session 5)
- [x] Holdout period réservée ✅ (cutoff 2025-02-01, session 7)
- [ ] Tests d'invariants prêts

### Pendant l'optimisation
- [x] Multi-seed averaging ✅ (`run_walk_forward_robust`, 5 seeds)
- [ ] Intervalles de confiance calculés
- [x] Sur-apprentissage surveillé ✅ (IS vs HO comparison)
- [x] Logs complets ✅ (loguru + JSON reports)

### Après l'optimisation
- [x] Validation sur holdout ✅ (sessions 7-8, 23 combos testés)
- [ ] Tests de significativité
- [x] Analyse des erreurs ✅ (2 FAIL identifiés comme sur-fitting)
- [x] Documentation complète ✅ (carnet de bord + knowledge base)

---

## Exemples de pièges à éviter

### Piège #1 : "Ça marche sur BTC donc ça marchera sur ETH"
**Réalité** : Corrélations ≠ 1, régimes différents

### Piège #2 : "Sharpe 1.2 = stratégie géniale"
**Réalité** : Sharpe 1.2 sur 3 mois avec 10 trades = chance

### Piège #3 : "Plus de paramètres = plus puissant"
**Réalité** : Plus de paramètres = plus d'overfitting

### Piège #4 : "Le backtest est réaliste"
**Réalité** : Toujours plus optimiste que la réalité

### Piège #5 : "La méta-opt a trouvé le vrai optimum"
**Réalité** : Probablement un optimum local bruité

---

## Indicateurs d'alerte

### 🚩 Red flags immédiats
- Sharpe > 1.5 avec < 50 trades
- Profit Factor < 1 avec Sharpe > 0.5
- Variance inter-seeds > 50%
- Performance > 50%/an

### ⚠️ Yellow flags
- Sharpe 0.8-1.2
- DD > 25%
- Concentration > 40% sur un actif
- Corrélation portfolio > 0.8

### ✅ Green flags
- Sharpe 0.3-0.8
- DD < 20%
- Multi-seed consistency > 0.8
- Diversification réelle

---

## Philosophie

**La simplicité est la sophistication** :
- Préférer Sharpe 0.5 stable que Sharpe 2.0 volatile
- La reproductibilité > la performance apparente
- Les baselines > la complexité non justifiée
- L'incertitude mesurée > la fausse précision

**Rigueur avant tout** :
- Chaque optimisation doit être questionnée
- Chaque résultat doit être validé
- Chaque hypothèse doit être testée
- Chaque échec doit être appris

---

## Leçons apprises du holdout (12 février 2026)

### Leçon #1 : IS Sharpe négatif ≠ mauvaise stratégie
Les combos avec IS Sharpe faible ou négatif performent parfois MIEUX en holdout.
Exemple : ETH/supertrend/4h (IS 0.054 → HO 0.444). Cela indique une stratégie qui ne sur-fitte pas.

### Leçon #2 : Multi-factor > Single-indicator
Le meilleur multi-factor (HO Sharpe 0.935) bat le meilleur simple (0.444) de 2x.
Les filtres de régime (ADX) et de volume sont les améliorations les plus impactantes.

### Leçon #3 : La variance inter-seeds est le vrai signal
Un HO Sharpe de 0.180 avec std=0.13 est PLUS fiable qu'un HO Sharpe de 0.779 avec std=0.86.
Toujours regarder la stabilité, pas seulement la performance médiane.

### Leçon #4 : ETH est le marché le plus "tradable"
7/11 survivants sont sur ETH. BTC est modéré (3/11), SOL est difficile (1/11).

### Leçon #5 : 4h est le timeframe optimal
6/11 survivants sur 4h. Bon compromis entre fréquence de signaux et bruit.

---

## Pipeline de validation validé (13 février 2026)

```
1. Diagnostic V5b (2-pass, pruning, multi-seed 3, risk grid)
   → 132 combos → Phase 1 quick scan → Phase 2 WF multi-seed
2. Holdout temporel (cutoff 2025-02-01)
   → IS walk-forward + HO backtest → STRONG/WEAK/FAIL
3. V5b exits avancées
   → Trailing stop + breakeven + max holding (optimisables)
4. Risk grid comparison
   → flat vs r0.5% vs r1.0% vs r2.0%
5. Corrélation matrix
   → Déduplication des combos corrélés pour portfolio
6. Portfolio Markowitz contraint
   → Covariance Ledoit-Wolf, hard constraints, Monte Carlo
```

---

## Philosophie

**La simplicité est la sophistication** :
- Préférer Sharpe 0.5 stable que Sharpe 2.0 volatile
- La reproductibilité > la performance apparente
- Les baselines > la complexité non justifiée
- L'incertitude mesurée > la fausse précision

**Rigueur avant tout** :
- Chaque optimisation doit être questionnée
- Chaque résultat doit être validé
- Chaque hypothèse doit être testée
- Chaque échec doit être appris
