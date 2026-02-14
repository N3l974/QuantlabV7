# Portfolio FTMO V1 — Passer le 2-Step Challenge & Tourner en Funded

**Date** : Juillet 2025
**Objectif** : Passer le FTMO 2-Step Challenge (Phase 1 + Verification) puis opérer en funded account de manière durable.

## 🔒 Statut figé (pause) — 14 février 2026

Le développement de `ftmo-v1` est temporairement mis en pause pour prioriser l'intégration des marchés classiques FTMO.

### État actuel validé

- Script principal: `portfolio/ftmo-v1/code/portfolio_ftmo_v1.py`
- Dernier report: `portfolio/ftmo-v1/results/portfolio_ftmo_v1_report.md`
- Dernier JSON: `portfolio/ftmo-v1/results/portfolio_ftmo_v1_20260214_164119.json`
- Contraintes FTMO bien respectées côté risque (daily loss / drawdown très en dessous des limites).

### Blocage principal avant challenge payant

- **Vitesse de gain insuffisante** en crypto-only: `MC pass phase1 = 0%` sur les profils actuels.
- Décision: ne pas lancer de challenge payant tant que l'univers n'intègre pas aussi FX/indices/or.

### Checklist de reprise

1. Intégrer les marchés classiques FTMO dans le pipeline (data + coûts + exécution).
2. Rejouer diagnostic + construction portfolio multi-actifs.
3. Recalculer Monte Carlo FTMO avec gate strict sur `phase1 pass_rate`.
4. Revalider `GO/NO-GO` uniquement si la probabilité de passage devient acceptable.

---

## 1. Règles FTMO 2-Step Challenge — Synthèse

### Phase 1 — FTMO Challenge

| Règle | Valeur | Type |
|-------|--------|------|
| **Profit Target** | **10%** du capital initial | Objectif |
| **Max Daily Loss** | **5%** du capital initial (fixe) | Hard limit |
| **Max Total Loss** | **10%** du capital initial | Hard limit (stop-out) |
| **Min Trading Days** | **4 jours** | Minimum |
| **Durée** | **Illimitée** | Pas de deadline |

### Phase 2 — Verification

| Règle | Valeur | Changement vs Phase 1 |
|-------|--------|----------------------|
| **Profit Target** | **5%** | Réduit de moitié |
| **Max Daily Loss** | **5%** | Identique |
| **Max Total Loss** | **10%** | Identique |
| **Min Trading Days** | **4 jours** | Identique |
| **Durée** | **Illimitée** | Identique |

### FTMO Account (Funded)

| Règle | Valeur |
|-------|--------|
| **Profit Target** | **Aucun** |
| **Max Daily Loss** | **5%** |
| **Max Total Loss** | **10%** |
| **Profit Split** | **80%** (jusqu'à 90% avec scaling) |
| **Scaling Plan** | +25% balance après 10% net profit sur 4 cycles |

---

## 2. Choix du Mode : SWING ✅ (Recommandé)

### Comparaison Standard vs Swing

| Critère | Standard | Swing | Impact pour nous |
|---------|----------|-------|------------------|
| **Hold overnight** | ❌ Interdit (funded) | ✅ Autorisé | **Critique** — nos strats 4h/1d tiennent des jours |
| **Hold weekend** | ❌ Interdit (funded) | ✅ Autorisé | **Important** — crypto trade 24/7 mais FTMO ferme le weekend |
| **News trading** | ❌ ±2min restriction | ✅ Aucune restriction | **Utile** — pas de filtre news à implémenter |
| **Leverage Forex** | 1:100 | 1:30 | Non applicable (crypto) |
| **Leverage Crypto** | 1:3.3 | 1:1 | **Impact** — leverage réduit en swing |
| **Leverage Indices** | 1:50 | 1:15 | Non applicable |

### Décision : **SWING**

**Raisons** :
1. **Nos stratégies sont 4h/1d** — les positions durent des jours/semaines, incompatible avec Standard en funded
2. **Crypto 24/7** — le marché crypto ne ferme pas, mais FTMO impose des fermetures weekend en Standard
3. **Pas de filtre news** — simplifie l'exécution, pas de logique de blackout à implémenter
4. **Leverage 1:1 crypto** — pas un problème car nos positions sont déjà dimensionnées en % du capital, pas en leverage

**Conséquence** : Le leverage 1:1 en crypto signifie que notre `max_position_pct` est effectivement le sizing réel. Pas de leverage implicite.

---

## 3. Contraintes de Risk Management FTMO → Mapping Backtester

### Mapping des règles FTMO vers notre RiskConfig

| Règle FTMO | Notre paramètre | Valeur | Marge de sécurité |
|------------|----------------|--------|-------------------|
| Max Daily Loss 5% | `max_daily_loss_pct` | **0.04** (4%) | 1% de marge |
| Max Total Loss 10% | `max_drawdown_pct` | **0.08** (8%) | 2% de marge |
| Profit Target Phase 1 (10%) | Objectif portfolio | **12-15%** | Marge pour frais |
| Profit Target Phase 2 (5%) | Objectif portfolio | **7-8%** | Marge pour frais |

### Pourquoi des marges de sécurité ?

- **Daily Loss** : Le calcul FTMO inclut les positions ouvertes (floating P&L). Notre circuit breaker doit se déclencher AVANT la limite réelle.
- **Total Loss** : Un drawdown de 8% déclenche notre circuit breaker, laissant 2% de marge pour le slippage et les frais non comptabilisés.
- **Profit Target** : Viser 12-15% au lieu de 10% car les frais réels (spread, commission, swap) réduisent le rendement net.

---

## 4. Architecture du Portfolio FTMO V1

### Philosophie

> **Objectif #1** : Ne PAS perdre le challenge (DD < 8%)
> **Objectif #2** : Atteindre le profit target (10% Phase 1, 5% Phase 2)
> **Objectif #3** : Être durable en funded (Sharpe > 1.0, DD contrôlé)

### Profils de risque FTMO

| Profil | Usage | Max Position | Daily Loss CB | Total DD CB | Objectif |
|--------|-------|-------------|---------------|-------------|----------|
| **Challenge** | Phase 1 (10% target) | 15% | 4% | 8% | Passer le challenge |
| **Verification** | Phase 2 (5% target) | 12% | 4% | 8% | Passer la vérification |
| **Funded** | FTMO Account | 10% | 3.5% | 7% | Durabilité long-terme |

### Sélection des stratégies — Critères FTMO-spécifiques

Les stratégies doivent satisfaire des critères plus stricts que le portfolio crypto :

| Critère | Seuil | Raison |
|---------|-------|--------|
| **HO Sharpe** | ≥ 0.5 | Rentabilité minimale |
| **HO Max DD** | ≥ -6% | Marge vs limite 8% |
| **HO Daily Max Loss** | ≥ -3% | Marge vs limite 4% |
| **Min trades HO** | ≥ 5 | Significativité statistique |
| **Seed std** | ≤ 0.5 | Robustesse accrue |
| **Win rate** | ≥ 35% | Éviter les séries perdantes longues |
| **Max losing streak** | ≤ 5 | Contrôle psychologique + daily loss |
| **Profit factor** | ≥ 1.2 | Edge réel |

### Stratégies candidates (par type)

Basé sur les résultats V5b du codebase, les meilleures candidates pour FTMO :

**Trend-following (core — 40-50% du portfolio)** :
- `supertrend_adx` — SuperTrend + ADX filter, très bon en trend, faible DD
- `ichimoku_cloud` — Trend robuste, bon en 4h
- `ema_ribbon` — Simple et efficace, faible variance

**Mean-reversion (diversification — 20-30%)** :
- `rsi_mean_reversion` — Classique, bon en range
- `mean_reversion_zscore` — Statistiquement solide

**Breakout (opportuniste — 15-25%)** :
- `breakout_regime` — ATR breakout + ADX filter + volume, multi-factor
- `donchian_channel` — Breakout classique, bon en 4h/1d

**Adaptive (stabilisateur — 10-20%)** :
- `regime_adaptive` — Switch trend/range/cash automatique

### Diversification multi-actif

| Actif | Allocation cible | Raison |
|-------|-----------------|--------|
| **BTCUSDT** | 35-45% | Liquidité max, spread min |
| **ETHUSDT** | 35-45% | Meilleur marché tradable (7/11 survivants HO) |
| **SOLUSDT** | 10-20% | Diversification, plus volatile |

### Timeframes

| TF | Allocation | Raison FTMO |
|----|-----------|-------------|
| **4h** | 60-70% | Optimal signal/bruit, compatible swing |
| **1d** | 30-40% | Positions longues, faible fréquence, faible DD |

> **Pas de 15m/1h** : Trop de trades/jour → risque de daily loss limit. Le swing mode favorise les TF longs.

---

## 5. Risk Management Multi-Couche

### Couche 1 — Position Level

| Paramètre | Valeur Challenge | Valeur Funded |
|-----------|-----------------|---------------|
| `max_position_pct` | 15% | 10% |
| `risk_per_trade_pct` | 1.0% | 0.75% |
| ATR SL mult | Optimisé par combo | Idem |
| ATR TP mult | Optimisé par combo | Idem |
| Trailing stop | Activé si optimisé | Idem |
| Breakeven stop | Activé si optimisé | Idem |
| Max holding bars | Activé si optimisé | Idem |

### Couche 2 — Daily Level (CRITIQUE pour FTMO)

| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| `max_daily_loss_pct` | 4% (challenge) / 3.5% (funded) | Marge vs 5% FTMO |
| `max_trades_per_day` | 5 | Anti-overtrading |
| `cooldown_after_loss` | 2 bars (4h) / 1 bar (1d) | Éviter revenge trading |

### Couche 3 — Portfolio Level

| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| `max_drawdown_pct` | 8% (challenge) / 7% (funded) | Marge vs 10% FTMO |
| Max correlation | 0.70 | Diversification |
| Max weight/combo | 25% | Pas de concentration |
| Max weight/symbol | 50% | Diversification actifs |
| Overlay regime | Hard cutoff en CRISIS | Protège le capital |
| Vol targeting | 25% annualisé | Normalise l'exposition |

### Couche 4 — Emergency (FTMO-specific)

| Mécanisme | Trigger | Action |
|-----------|---------|--------|
| **Daily Loss Shield** | Perte jour > 3% | Stop all trading pour la journée |
| **DD Shield** | DD > 6% | Réduire position sizing de 50% |
| **DD Emergency** | DD > 7.5% | Stop all trading, close all positions |
| **Profit Lock** | Profit > 8% (Phase 1) | Réduire sizing, protéger le gain |

---

## 6. Scénarios de Passage

### Phase 1 — Challenge (10% target)

**Scénario conservateur** (2-3 mois) :
- Return mensuel cible : 3-5%
- DD max toléré : -6%
- Probabilité estimée : 70-80% (basé sur MC V5b)

**Scénario modéré** (1-2 mois) :
- Return mensuel cible : 5-8%
- DD max toléré : -8%
- Probabilité estimée : 60-70%

### Phase 2 — Verification (5% target)

- Même portfolio, même sizing
- Target plus facile (5% vs 10%)
- Probabilité estimée : 80-90%

### Funded — Opération durable

- Sizing réduit (10% max position vs 15%)
- Circuit breaker plus serré (7% vs 8%)
- Objectif : Sharpe > 1.0, DD < 5%, profit mensuel 2-4%
- Scaling plan : +25% balance après 4 cycles à +10%

---

## 7. Métriques de Suivi FTMO

### Dashboard quotidien

| Métrique | Calcul | Alerte si |
|----------|--------|-----------|
| **Daily P&L** | Closed + floating | > -3% (warning), > -4% (stop) |
| **Total DD** | Equity vs initial | > -6% (warning), > -7.5% (emergency) |
| **Progress** | Equity vs target | Tracking vs plan |
| **Trades today** | Count | > 5 (stop) |
| **Exposure** | Sum positions / capital | > 40% |

### Critères de confiance FTMO (scoring /100)

| Critère | Points | Seuil GO |
|---------|--------|----------|
| Sharpe ≥ 1.5 | 15 | ≥ 10 |
| Max DD > -6% | 15 | ≥ 10 |
| Daily max loss > -3% | 15 | ≥ 15 |
| Win rate ≥ 40% | 10 | ≥ 5 |
| Profit factor ≥ 1.3 | 10 | ≥ 5 |
| Rolling Sharpe stable | 10 | ≥ 5 |
| MC P(pass challenge) ≥ 70% | 15 | ≥ 10 |
| Multi-seed robust | 5 | ≥ 5 |
| Max losing streak ≤ 5 | 5 | ≥ 5 |
| **Total** | **100** | **≥ 75 = GO** |

---

## 8. Fichiers du Portfolio

```
portfolio/ftmo-v1/
├── README.md                    # Ce document
├── code/
│   └── portfolio_ftmo_v1.py     # Script principal
├── config/
│   └── ftmo_config.yaml         # Configuration FTMO
└── results/                     # Résultats (gitignored)
    ├── portfolio_ftmo_v1_*.json
    └── portfolio_ftmo_v1_report.md
```

---

## 9. Différences clés vs Portfolio V5b

| Aspect | V5b (Crypto perso) | FTMO V1 |
|--------|-------------------|---------|
| **Objectif** | Maximiser Sharpe | Passer le challenge + durabilité |
| **DD limit** | 15% (circuit breaker) | 8% (challenge) / 7% (funded) |
| **Daily loss** | 3% (soft) | 4% (hard, FTMO rule) |
| **Position sizing** | 10-50% selon profil | 10-15% max |
| **Leverage** | Binance margin | 1:1 (Swing crypto) |
| **Timeframes** | 15m-1d | 4h-1d uniquement |
| **Overlays** | Optionnel | Obligatoire (regime + vol target) |
| **Emergency stops** | Circuit breaker simple | Multi-couche (daily + DD + profit lock) |
| **Profit target** | Aucun | 10% (P1) / 5% (P2) |

---

*Généré par Quantlab V7 — Portfolio FTMO V1*
