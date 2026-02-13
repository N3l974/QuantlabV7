# Bugs & Fixes — Quantlab V7

Historique des bugs rencontrés, root cause analysis, fixes appliqués et leçons apprises pour éviter les répéter.

---

## Bug #1 : Double-comptage dans le backtester

### Date
10 février 2026 (Meta-opt Run 1/2)

### Symptôme
- Sharpe élevé (2.34) mais Profit Factor < 1 (0.66)
- 94% des profils avec PF < 1
- Returns irréalistes

### Root Cause
Dans `engine/backtester.py`, l'equity était calculée comme :
```python
equity = cash + position_value  # DOUBLE COMPTAGE !
```

Quand on est 100% long :
- `cash` = 0
- `position_value` = 11000
- `equity` = 11000 (correct)
- Mais `position_value` inclut déjà le capital initial !

Le vrai calcul devrait être :
```python
equity = cash_non_allocated + capital_allocated * (1 + pnl_pct)
```

### Fix Appliqué
```python
# Nouveau modèle dans backtester.py
initial_capital = 10000.0
capital_allocated = position_size * entry_price
pnl_pct = (current_price - entry_price) / entry_price

equity = cash + capital_allocated * (1 + pnl_pct)
```

### Impact
- PF maintenant cohérent avec Sharpe
- Returns réalistes (30-40% au lieu de 200%+)
- Meta-opt Run 1/2 invalidées, nécessité de relancer

### Leçon
**Toujours vérifier les invariants** : PF < 1 avec Sharpe > 1 = impossible mathématique → bug immédiat.

---

## Bug #2 : Bounds Optuna invalides (momentum_roc)

### Date
11 février 2026 (Diagnostic V2)

### Symptôme
```
ValueError: low > high
```
Crash dans Optuna lors de l'optimisation momentum_roc.

### Root Cause
Dans `strategies/momentum_roc.py` :
```python
exit_threshold: float = 0.0  # DEFAULT = 0 !
```

Dans `engine/walk_forward.py`, `_get_param_info()` :
```python
if default_val <= 0:
    low = 0.001
    high = 1.0
# Mais avec scale < 1.0, les bounds deviennent [0.001, 0.001] → crash
```

### Fix Appliqué
1. **Strategy level** : Changé default de 0.0 à 0.1
```python
exit_threshold: float = 0.1
```

2. **Engine level** : Amélioré `_get_param_info()` pour gérer les floats proches de 0
```python
if low >= high:
    high = low + max(abs(low) * 0.1, 0.001)
```

### Impact
- momentum_roc maintenant fonctionnelle
- Plus de crashes bounds sur toutes les stratégies

### Leçon
**Toujours tester les edge cases** : defaults = 0 peuvent causer des bounds invalides avec scaling.

---

## Bug #3 : Williams %R paramètres négatifs

### Date
12 février 2026 (Élargissement stratégies)

### Symptôme
Bounds Optuna incorrectes pour Williams %R (oversold=-80, overbought=-20).

### Root Cause
Williams %R utilise des valeurs négatives par convention :
- Oversold = -80 (plus bas que -80 = survente)
- Overbought = -20 (plus haut que -20 = surachat)

Mais Optuna ne gère pas bien les bounds négatives avec scaling.

### Fix Appliqué
**Inverser la logique** : utiliser des seuils positifs et convertir en négatif dans la stratégie.
```python
# Paramètres (positifs)
oversold_threshold: float = 80    # Converti en -80
overbought_threshold: float = 20   # Converti en -20

# Dans generate_signals()
if williams_r < -oversold_threshold/100:  # < -0.80
    # Oversold signal
```

### Impact
- Williams %R bounds correctes
- Compatible avec scaling Optuna

### Leçon
**Adapter les conventions à l'outil** : si Optuna préfère le positif, adapter la stratégie plutôt que forcer l'outil.

---

## Bug #4 : Daily reset hardcodé

### Date
11 février 2026 (Correction backtester)

### Symptôme
Daily loss limit et max trades/jour ne fonctionnent pas correctement sur 15m/1h.

### Root Cause
Dans `engine/backtester.py` :
```python
BARS_PER_DAY = 24  # Hardcodé pour 1h !
```

Pour 15m : 96 bars/jour, pour 4h : 6 bars/jour.

### Fix Appliqué
```python
BARS_PER_DAY = {
    "15m": 96, "1h": 24, "4h": 6, "1d": 1
}
```

Et propagation du timeframe dans walk_forward → backtester.

### Impact
- Risk management fonctionne sur tous les TFs
- Daily loss limit correcte

### Leçon
**Pas de hardcoding** : toujours adapter aux paramètres du contexte.

---

## Bug #5 : Funding rate non appliqué

### Date
11 février 2026 (Correction backtester)

### Symptôme
Returns trop optimistes, pas de coûts de financement.

### Root Cause
Binance perpetual a un funding rate toutes les 8h (~0.01%), non modélisé.

### Fix Appliqué
```python
# Accumulateur de funding
funding_accumulator = 0.0
bars_since_funding = 0

# Toutes les 8h de bars
if bars_since_funding >= BARS_PER_8H[timeframe]:
    funding_rate = 0.0001  # 0.01%
    if position != 0:
        funding_cost = position_size * current_price * funding_rate
        cash -= funding_cost
    funding_accumulator = 0.0
```

### Impact
- Returns réduits de 1-3%/an sur positions longues
- Plus réaliste

### Leçon
**Modéliser tous les coûts réels** : commission, slippage, funding, etc.

---

## Bug #6 : Variance walk-forward non contrôlée

### Date
12 février 2026 (Portfolio V1/V2)

### Symptôme
Même profil méta-optimisé donne Sharpe entre -0.50 et +0.72 selon le run.

### Root Cause
Dans `engine/walk_forward.py` :
```python
study = optuna.create_study(direction="maximize")  # PAS DE SEED !
```

Chaque fenêtre de walk-forward utilise un sampler TPE avec initialisation aléatoire → variance cumulée.

### Fix Appliqué (proposé, non implémenté)
```python
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
```

### Impact
- Rendrait le walk-forward déterministe
- Prérequis pour toute méta-optimisation sensée

### Leçon
**La reproductibilité est primordiale** : sans seed fixe, on optimise du bruit.

---

## Bug #7 : Timeframe non propagé

### Date
11 février 2026 (Correction backtester)

### Symptôme
Le backtester ne sait pas quel timeframe il traite.

### Root Cause
`backtest_strategy()` n'avait pas de paramètre `timeframe`.

### Fix Appliqué
```python
def backtest_strategy(strategy, data, params, 
                    timeframe="1d", ...):
    # Utiliser timeframe pour daily reset, funding, etc.
```

Propagation depuis walk_forward → backtester.

### Impact
- Comportement cohérent sur tous les TFs
- Daily reset adapté

### Leçon
**Propager le contexte** : chaque fonction doit connaître son environnement.

---

## Bug #8 : Equity curve padding incorrect

### Date
12 février 2026 (Portfolio V2)

### Symptôme
Portfolio equity curve a des sauts/creux inexpliqués.

### Root Cause
Quand les stratégies ont des longueurs OOS différentes, le padding avec 0 crée des artefacts.

### Fix Appliqué (dans portfolio_v2_leverage.py)
```python
# Alignement correct avec padding de la dernière valeur
max_len = max(len(eq) for eq in equity_curves)
aligned_equities = []
for eq in equity_curves:
    if len(eq) < max_len:
        # Pad avec la dernière valeur, pas 0
        padding = np.full(max_len - len(eq), eq[-1])
        eq = np.concatenate([eq, padding])
    aligned_equities.append(eq)
```

### Impact
- Portfolio equity curves lisses
- Métriques correctes

### Leçon
**Le padding doit préserver la continuité** : padding avec 0 = artefacts.

---

## Bug #9 : Williams %R — Signal ne se déclenche JAMAIS

### Date
12 février 2026 (Audit stratégies pré-Diagnostic V4)

### Symptôme
La stratégie Williams %R ne génère aucun signal (toujours flat).

### Root Cause
Logique de confirmation contradictoire :
```python
# AVANT (bugué)
oversold_confirmed = all(wr[i - j] < oversold for j in range(confirm))
# Avec j=0, cela inclut wr[i] < oversold
if oversold_confirmed and wr[i] >= oversold:  # CONTRADICTION !
```
`range(confirm)` inclut `j=0` (barre courante). La condition exige que `wr[i] < oversold` ET `wr[i] >= oversold` simultanément → impossible.

De plus, off-by-one dans le calcul du %R : `high[i-period:i]` exclut la barre courante.

### Fix Appliqué
```python
# APRÈS (corrigé)
# 1. Off-by-one fix
highest = np.max(high[i - period + 1:i + 1])  # inclut barre courante

# 2. Confirmation sur barres PRÉCÉDENTES uniquement
oversold_confirmed = all(wr[i - j] < oversold for j in range(1, confirm + 1))
if oversold_confirmed and wr[i] >= oversold:  # OK : barres passées en zone, courante sort
```

### Impact
- Stratégie ressuscitée (générait 0 trades avant)
- Résultats Diagnostic V4 incluent maintenant Williams %R

### Leçon
**Tester les edge cases des conditions logiques** : une contradiction subtile peut rendre une stratégie entièrement morte sans erreur visible.

---

## Bug #10 : Stochastic Oscillator — Crash sur petites fenêtres

### Date
12 février 2026 (Audit stratégies pré-Diagnostic V4)

### Symptôme
`ValueError` / `KeyboardInterrupt` dans le Diagnostic V4 à 8% de progression.

### Root Cause
Quand Optuna propose un `k_period` grand (ex: 50) et que la fenêtre de train est petite, `np.min(low[i - k_period + 1 : i + 1])` reçoit un slice vide ou négatif.

De plus, zone d'entrée hardcodée `+ 20` non paramétrable.

### Fix Appliqué
```python
# Protection bounds
start = max(0, i - k_period + 1)
if start >= i + 1:
    continue

# Zone paramétrable
zone_buffer = params.get("zone_buffer", 20)
k_values[i] < oversold + zone_buffer
```

### Impact
- Plus de crash sur fenêtres courtes
- Zone d'entrée configurable par Optuna

### Leçon
**Protéger les indices de slice** : toujours vérifier que `start < end` avant `np.min/max`.

---

## Bug #11 : Donchian Channel — Off-by-one

### Date
12 février 2026 (Audit stratégies pré-Diagnostic V4)

### Symptôme
Channels légèrement décalés par rapport au standard.

### Root Cause
`high[i - ch_period : i]` exclut la barre courante. Le Donchian standard inclut les N dernières barres y compris la courante.

### Fix Appliqué
```python
# AVANT : high[i - ch_period : i]
# APRÈS : high[i - ch_period + 1 : i + 1]
```

### Impact
- Channels correctement alignés avec la définition standard

### Leçon
**Vérifier les conventions de fenêtrage** : `[i-N:i]` vs `[i-N+1:i+1]` est une source d'erreur récurrente.

---

## Bug #12 : Z-Score Mean Reversion — Off-by-one

### Date
12 février 2026 (Audit stratégies pré-Diagnostic V4)

### Symptôme
Z-score calculé sans inclure le prix courant dans la fenêtre.

### Root Cause
`close[i - lookback:i]` exclut `close[i]`.

### Fix Appliqué
```python
# AVANT : close[i - lookback:i]
# APRÈS : close[i - lookback + 1:i + 1]
```

### Impact
- Z-score plus réactif et cohérent

---

## Bug #13 : Keltner Channel — ATR avec EMA au lieu de Wilder

### Date
12 février 2026 (Audit stratégies pré-Diagnostic V4)

### Symptôme
Bandes Keltner plus larges/étroites que l'attendu, incohérent avec ATR Breakout et SuperTrend.

### Root Cause
ATR calculé avec `alpha = 2/(n+1)` (EMA classique) au lieu de `alpha = 1/n` (Wilder smoothing).

### Fix Appliqué
```python
# AVANT : atr[i] = atr_alpha * tr[i] + (1 - atr_alpha) * atr[i-1]  (EMA)
# APRÈS : atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period  (Wilder)
```

### Impact
- Cohérence ATR entre toutes les stratégies
- Bandes conformes à la définition standard

### Leçon
**Uniformiser les calculs d'indicateurs** : un même indicateur (ATR) doit utiliser la même formule partout.

---

## Patterns de bugs récurrents

### 1. Invariants mathématiques non vérifiés
- **Symptôme** : PF < 1 avec Sharpe > 1
- **Solution** : Tests d'invariants automatiques
- **Prévention** : `assert profit_factor >= 1 or sharpe < 1.0`

### 2. Paramètres par défaut problématiques
- **Symptôme** : Bounds invalides, crashes Optuna
- **Solution** : Revue systématique des defaults
- **Prévention** : Tests bounds pour toutes les stratégies

### 3. Hardcoding vs configuration
- **Symptôme** : Comportement incorrect sur certains TFs
- **Solution** : Configuration dynamique
- **Prévention** : Pas de constantes hardcodées

### 4. Stochasticité non contrôlée
- **Symptôme** : Résultats non reproductibles
- **Solution** : Seeds fixes partout
- **Prévention** : Tests de reproductibilité

### 5. Coûts réels omis
- **Symptôme** : Returns sur-optimistes
- **Solution** : Modéliser tous les coûts
- **Prévention** : Checklist des coûts par marché

---

## Tests préventifs ajoutés

### 1. Tests d'invariants (`tests/test_backtester.py`)
```python
def test_profit_factor_sharpe_consistency():
    # PF < 1 ⇒ Sharpe ne peut pas être > 1
    assert not (profit_factor < 1 and sharpe > 1.0)
```

### 2. Tests bounds (`tests/test_strategies.py`)
```python
def test_all_strategies_valid_bounds():
    for strategy in all_strategies:
        bounds = get_param_bounds(strategy, scale=0.3)
        for low, high in bounds.values():
            assert low < high
```

### 3. Tests reproductibilité
```python
def test_walk_forward_reproducible():
    result1 = run_walk_forward(config, seed=42)
    result2 = run_walk_forward(config, seed=42)
    assert np.allclose(result1.equity, result2.equity)
```

---

## Process de gestion des bugs

1. **Détection** : Tests automatiques + invariants
2. **Isolation** : Reproduire le bug sur un cas minimal
3. **Root cause** : Analyser le code, pas seulement les symptômes
4. **Fix** : Solution minimaliste et robuste
5. **Test** : Ajouter un test unitaire pour éviter régression
6. **Documentation** : Noter ici pour mémoire collective

---

## Prochaines zones à risque

1. **Portfolio construction** : Pas de covariance → corrélations ignorées
2. **Leverage modeling** : Linéaire vs réalité (margin calls, slippage)
3. **Meta-optimization** : Fonction objectif bruitée
4. **Holdout temporal** : Data snooping cumulé
5. **Multi-asset correlations** : Crises synchronisées

---

## Leçons fondamentales

1. **La rigueur mathématique est non négociable**
2. **Toute stochasticité doit être contrôlée**
3. **Les defaults sont aussi importants que la logique**
4. **La reproductibilité > performance apparente**
5. **Documenter chaque bug évite de le réinventer**
