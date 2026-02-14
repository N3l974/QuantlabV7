# Quantlab V7 — Système de Trading Algorithmique Multi-Actif

*Dernière mise à jour : 14 février 2026*

---

## 1. Vision du Projet

Quantlab V7 est un framework de trading algorithmique crypto qui :

1. **Découvre** les meilleures combinaisons stratégie × actif × timeframe via un diagnostic exhaustif (2 phases)
2. **Optimise** les paramètres de chaque combinaison via walk-forward bayésien (Optuna TPE, multi-seed)
3. **Construit** un portfolio diversifié avec Markowitz contraint + Monte Carlo
4. **Exécute** en live sur Binance avec réoptimisation trimestrielle automatique
5. **Protège** via overlays adaptatifs (régime, vol targeting) et risk management multi-couche

### Principes fondamentaux

- **Tout ce qui tourne en live doit avoir été backtesté** — pas de logique ad-hoc non validée
- **Réalisme du backtest** — commissions, slippage dynamique, funding rate, circuit breakers, position sizing
- **Robustesse > performance** — on préfère un Sharpe 0.8 stable à un Sharpe 2.0 fragile
- **Simplicité opérationnelle** — le système cloud doit être minimal et fiable

### Architecture multi-niveaux

```
Diagnostic (2 phases)  → identifie les combos viables
    └── Walk-Forward (Optuna TPE, multi-seed) → optimise les paramètres
        └── Backtest (vectorisé numpy) → score chaque jeu de paramètres
            └── Overlays (regime + vol targeting) → filtre post-signal
                └── Portfolio (Markowitz + Monte Carlo) → allocation optimale
```

| Niveau | Nom | Rôle |
|--------|-----|------|
| **Diagnostic** | Quick scan + Walk-forward | Filtre 132 combos → ~40 survivants |
| **Walk-Forward** | Optuna TPE + MedianPruner | Optimise les params sur fenêtres glissantes (multi-seed) |
| **Overlays** | Regime + Vol Targeting | Coupe les signaux en régime défavorable, normalise l'exposition |
| **Portfolio** | Markowitz contraint | Allocation optimale avec covariance shrinkage Ledoit-Wolf |

---

## 2. Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Workflow low-context (recommandé)

- Hub docs: `docs/README.md`
- Contexte actif: `docs/context/ACTIVE_CONTEXT.md`
- Template session: `docs/context/SESSION_TEMPLATE.md`
- Guide utilisateur: `docs/GUIDE_UTILISATEUR.md`
- Initialisation Git: `docs/GIT_INIT_GUIDE.md`

### Clés API Binance (optionnel, pour ingestion)

```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

### Commandes principales

```bash
# Ingestion des données
python main.py ingest

# Méta-optimisation complète
python main.py optimize

# Test rapide (50 trials, 2h timeout)
python main.py optimize --trials 50 --timeout 2

# Stratégie unique
python main.py optimize --strategy rsi_mean_reversion --trials 100

# Portfolio
python main.py portfolio results/meta_profiles_XXXXXX.json --top-n 5 --method diversified

# Dashboard
streamlit run dashboard/app.py

# Live (dry run)
python -c "from live.executor import run_live; run_live('results/meta_profiles_XXXXXX.json', dry_run=True)"

# Service portfolio live/paper (config JSON)
python -m live.run_portfolio --config config/live/portfolios/v5c-highrisk-paper.json
```

### Exploitation paper (Makefile)

```bash
# Rapport complet depuis le début des logs (par défaut)
make paper-report

# Rapport fenêtre glissante
make paper-report-window HOURS=24
make paper-report-48h

# État VPS
make vps-status
make vps-logs
make vps-tail
```

Notes:
- `make paper-report` utilise `--since-start` (historique complet).
- Le "start equity" affiché correspond à la première ligne de `pnl.jsonl` encore présente.
- Pour repartir d'une baseline propre (ex: 1000), purger `trades.jsonl`, `pnl.jsonl` et `state.json`.

### CLI

| Commande | Description |
|----------|-------------|
| `python main.py ingest` | Télécharger/mettre à jour les données Binance |
| `python main.py optimize` | Lancer la méta-optimisation |
| `python main.py portfolio <file>` | Construire un portfolio |
| `python main.py strategies` | Lister les stratégies disponibles |
| `python main.py status` | Statut du projet et des données |

### Tests

```bash
pytest tests/ -v   # 98 tests
```

---

## 3. Univers de Trading

### Actifs (3 principaux + 2 secondaires)

| Symbole | Paire | Données depuis | Statut |
|---------|-------|----------------|--------|
| BTCUSDT | Bitcoin / USDT | 2017 | ✅ Principal |
| ETHUSDT | Ethereum / USDT | 2017 | ✅ Principal |
| SOLUSDT | Solana / USDT | 2020 | ✅ Principal |
| BNBUSDT | Binance Coin / USDT | 2017 | ⚠️ Secondaire |
| XRPUSDT | Ripple / USDT | 2019 | ⚠️ Secondaire |

### Timeframes (4)

| TF | Bars/jour | Usage |
|----|-----------|-------|
| 15m | 96 | Scalping / intraday |
| 1h | 24 | Intraday / swing |
| 4h | 6 | Swing |
| 1d | 1 | Position / trend |

> 1m et 5m exclus : trop lents à ingérer, spread/slippage trop impactant, pas adaptés aux stratégies actuelles.

### Stratégies (22)

#### Stratégies single-indicator (16)

| # | Nom | Type | Description |
|---|-----|------|-------------|
| 1 | `rsi_mean_reversion` | Mean-reversion | RSI oversold/overbought |
| 2 | `macd_crossover` | Trend | MACD signal cross |
| 3 | `bollinger_breakout` | Breakout | Bollinger Band breakout |
| 4 | `ema_ribbon` | Trend | Multi-EMA alignment |
| 5 | `vwap_deviation` | Mean-reversion | VWAP deviation |
| 6 | `donchian_channel` | Breakout | Donchian channel breakout |
| 7 | `stochastic_oscillator` | Mean-reversion | Stochastic %K/%D |
| 8 | `ichimoku_cloud` | Trend | Ichimoku cloud signals |
| 9 | `atr_volatility_breakout` | Volatility | ATR-based breakout |
| 10 | `volume_obv` | Volume | OBV + volume spikes |
| 11 | `momentum_roc` | Momentum | Dual ROC (fast + slow) |
| 12 | `adx_regime` | Regime filter | ADX trend/range detection |
| 13 | `keltner_channel` | Volatility channel | EMA ± ATR dynamic channels |
| 14 | `mean_reversion_zscore` | Statistical | Rolling z-score mean reversion |
| 15 | `supertrend` | Trend | ATR-based trailing stop system |
| 16 | `williams_r` | Momentum | Williams %R overbought/oversold |

#### Stratégies multi-factor (3)

| # | Nom | Type | Description |
|---|-----|------|-------------|
| 17 | `supertrend_adx` | Trend + Regime | SuperTrend filtré par ADX + cooldown |
| 18 | `trend_multi_factor` | Trend + Volume + Momentum | Confluence 3/3 : SuperTrend + OBV slope + ROC |
| 19 | `breakout_regime` | Breakout + Regime + Volume | ATR breakout + ADX filter + volume spike |

#### Stratégies adaptatives (3)

| # | Nom | Type | Description |
|---|-----|------|-------------|
| 20 | `regime_adaptive` | Regime-switching | Trend-following en trend, mean-reversion en range, cash en crise |
| 21 | `mtf_trend_entry` | Multi-timeframe | HTF SuperTrend trend + LTF RSI pullback entry |
| 22 | `mtf_momentum_breakout` | Multi-timeframe | HTF momentum (ROC+ADX) + LTF Donchian breakout |

**Espace total** : 22 stratégies × 3 actifs × 2 TFs = **132 combinaisons** (focus sur BTC/ETH/SOL, 4h/1d)

---

## 4. Pipeline de Recherche (Local)

### 4.1 Ingestion des données

```
Binance API → OHLCV parquet (data/raw/{SYMBOL}_{TF}.parquet)
```

- Source : Binance public API (klines)
- Format : parquet (compact, rapide)
- Fréquence : hebdomadaire ou avant chaque diagnostic
- Historique : maximum disponible (5-8 ans selon l'actif)

### 4.2 Diagnostic 2-phases

Scan robuste de l'univers pour identifier les combos viables.

**Phase 1 — Quick scan** (defaults sur holdout) :
- Backtest rapide avec paramètres par défaut sur données post-cutoff
- Filtre : Sharpe > -1.5, min 3 trades
- Durée : ~2 min

**Phase 2 — Walk-forward** (sur survivants) :
- Walk-forward complet (Optuna TPE, 30 trials, 3M reoptim, 1Y window)
- Multi-seed (3 seeds) pour robustesse
- Holdout validation sur données post-cutoff
- Baseline + overlay variants testées
- Durée : ~20-60 min selon nombre de survivants

| Paramètre | Valeur |
|-----------|--------|
| Trials Optuna / fenêtre | 30 |
| Train window | 1Y |
| Reoptim frequency | 3M |
| Seeds | 3 (médiane retenue) |
| Métrique d'optimisation | Sharpe |
| Pruning | MedianPruner (5 startup, 3 warmup) |
| Cutoff holdout | 2025-02-01 |

**Filtres de viabilité** :
- HO Sharpe > 0.3 → STRONG
- HO Sharpe > 0.0 → WEAK
- Min 3 trades en holdout
- Seed std < 0.3 → robuste

**Output** : `portfolio/v5b/results/diagnostic_v5b_{timestamp}.json` + rapport markdown

### 4.3 Méta-optimisation (abandonnée)

> **Résultat du test A/B** : les defaults fixes (3M/1Y/sharpe/100 trials) font MIEUX que la méta-optimisation (+0.102 Sharpe en moyenne). La méta-opt sur-fitte les méta-paramètres.

**Décision** : defaults fixes utilisés partout. La méta-optimisation n'est plus utilisée.

### 4.4 Validation OOS finale (holdout)

Avant déploiement, chaque profil est validé sur une période out-of-sample jamais vue :
- **Cutoff** : 2025-02-01 (12 mois de holdout)
- Walk-forward sur données pré-cutoff (in-sample)
- Derniers params optimisés appliqués sur données post-cutoff
- Vérification : Sharpe > 0, min trades, DD acceptable
- Si échec → le profil n'est pas déployé

---

## 5. Backtester — Caractéristiques Réalistes

### Coûts & réalisme

| Feature | Implémentation |
|---------|---------------|
| **Commission** | 0.1% par trade (configurable) |
| **Slippage** | Dynamique basé sur la volatilité (ATR), 0.05% → 0.5% |
| **Funding rate** | 0.01% / 8h sur positions ouvertes (standard Binance perp) |
| **Daily reset** | Adaptatif par timeframe (`BARS_PER_DAY = {"15m": 96, "1h": 24, "4h": 6, "1d": 1}`) |
| **Equity** | Modèle cash + capital alloué (pas de double-comptage) |

### Risk management

| Feature | Implémentation |
|---------|---------------|
| **Position sizing** | % max du capital par position (défaut 25%) |
| **Risk-based sizing (V5)** | `risk_per_trade_pct` : position = (equity × risk%) / SL_distance |
| **Circuit breaker** | Arrêt si drawdown > seuil (défaut 15%) |
| **Daily loss limit** | Arrêt si perte journalière > seuil (défaut 3%) |
| **Max trades/jour** | Limite configurable (défaut 10) |
| **Cooldown** | Pause après perte (configurable) |

### Stop-Loss / Take-Profit (V5)

| Mode | Description |
|------|-------------|
| **Pourcentage fixe** | `stop_loss_pct`, `take_profit_pct` (V1-V4) |
| **ATR-based (V5)** | `atr_sl_mult × ATR / prix`, `atr_tp_mult × ATR / prix` — adaptatif à la volatilité |
| **Trailing stop (V5b)** | `trailing_atr_mult × ATR` — suit le prix, verrouille les gains |
| **Breakeven stop (V5b)** | `breakeven_trigger_pct` — SL ramené à l'entrée après X% de gain |
| **Max holding (V5b)** | `max_holding_bars` — sortie forcée après N barres |

### Overlays post-signal

| Overlay | Description |
|---------|-------------|
| **Regime overlay** | Coupe les signaux en RANGE/CRISIS (ADX + vol + DD) |
| **Vol targeting** | Normalise l'exposition pour viser 30% vol annualisée |

---

## 6. Architecture Hybride — Local + Cloud

### Vue d'ensemble

```
LOCAL (ton PC)                          CLOUD (VPS)
┌──────────────────────┐                ┌──────────────────────┐
│ Ingestion            │                │ Signal Runner        │
│ Diagnostic           │   git push     │ Order Executor       │
│ Méta-optimisation    │ ──────────────>│ Position Tracker     │
│ Validation OOS       │   (profiles)   │ Scheduler (réoptim)  │
│ Réoptimisation       │                │ Monitor + Telegram   │
└──────────────────────┘                └──────────────────────┘
```

### Local — Recherche & Optimisation

- **Quand** : à la demande (pas 24/7)
- **Rôle** : tout le calcul lourd (diagnostic, méta-optim, réoptimisation)
- **Données** : stockées localement en parquet (gitignored)

### Cloud — Exécution Live

- **Quand** : 24/7
- **Rôle** : exécuter les signaux, placer les ordres, monitorer
- **VPS** : léger (~5€/mois, Hetzner/DigitalOcean/OVH)
- **Pas de calcul lourd** : juste appliquer les paramètres optimisés

---

## 7. Structure du Projet

```
Quantlab-V7/
├── config/                         # LOCAL + CLOUD
│   ├── settings.yaml               # Paramètres globaux
│   ├── strategies.yaml             # Catalogue stratégies
│   ├── meta_search_space.yaml      # Espace de recherche méta
│   └── live_config.yaml            # Config live
├── data/                           # LOCAL ONLY (gitignored)
│   ├── ingestion.py                # Pipeline Binance → Parquet
│   └── raw/                        # Fichiers parquet
├── strategies/                     # LOCAL + CLOUD
│   ├── base.py                     # BaseStrategy (compute_atr, _apply_advanced_exits)
│   ├── registry.py                 # Catalogue central
│   └── *.py                        # 22 implémentations (16 single + 3 multi-factor + 3 adaptive)
├── engine/                         # LOCAL + CLOUD
│   ├── backtester.py               # Backtest vectorisé (V5: risk sizing, sl_distances)
│   ├── metrics.py                  # Métriques de performance
│   ├── walk_forward.py             # Walk-forward optimizer (seeds, pruning, V5b bounds)
│   ├── meta_optimizer.py           # Méta-optimizer (abandonné — defaults gagnent)
│   ├── regime.py                   # Détection de régime (STRONG/WEAK/RANGE/CRISIS)
│   ├── overlays.py                 # Pipeline overlays (regime + vol targeting)
│   └── portfolio.py                # Construction de portfolio
├── scripts/                        # LOCAL ONLY
│   ├── diagnostic_v4_fast.py       # Diagnostic V4 (2 phases, overlays)
│   ├── diagnostic_v5.py            # Diagnostic V5 (ATR SL/TP, risk sizing)
│   ├── diagnostic_v5b.py           # Diagnostic V5b (trailing, breakeven, multi-seed, risk grid, correlation)
│   ├── portfolio_v4b_final.py      # Portfolio V4b (actif)
│   ├── portfolio_v3_markowitz.py   # Portfolio V3 (archivé)
│   └── push_profiles.py            # Push vers cloud
├── live/                           # CLOUD ONLY
│   ├── executor.py                 # Exécution legacy mono-combo
│   ├── portfolio_executor.py       # Exécution multi-combos (agrégation par symbole)
│   ├── run_portfolio.py            # Entrée service portfolio (auto multi/legacy)
│   ├── monitor.py                  # Logs trades/pnl + metadata
│   └── ...
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── cloud_requirements.txt
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── docs/
│   ├── carnet_de_bord.md           # Journal chronologique (sessions)
│   ├── presentation_investisseur.md # Présentation investisseur V4b
│   ├── knowledge_base/             # 5 fichiers techniques
│   ├── portfolios/                 # Index de compatibilité (legacy)
│   └── results/                    # Index de compatibilité des rapports migrés
├── portfolio/
│   ├── ftmo-v1/                    # Portfolio FTMO (README + code + config + results)
│   ├── v4b/                        # Portfolio V4b (README + code + results)
│   ├── v5b/                        # Portfolio V5b (README + code + results)
│   └── v5c-highrisk/               # Portfolio V5c high-risk (README + code + results)
├── tests/                          # 98 tests
├── Makefile                        # Commandes rapides (report, status, deploy)
├── .github/workflows/deploy-portfolio.yml # CI/CD portfolio
├── main.py                         # CLI entry point
├── requirements.txt
└── README.md                       # Ce document
```

### Déploiement sélectif

Le **Dockerfile** ne copie que ce qui est nécessaire au cloud :
- `engine/`, `strategies/`, `live/`, `config/`
- `results/active_profiles.json`
- **Exclut** : `scripts/`, `data/`, `results/` (sauf active_profiles), `tests/`

### CI/CD

```
git push main → GitHub Actions → Build Docker → SSH deploy VPS → Restart service
```

### Runtime paper portfolio (V5c)

- Service: `v5c-highrisk-paper`
- Exécution: `live.run_portfolio` + `PortfolioExecutor` multi-combos
- Contraintes Binance Cross Margin simulées: 1 position nette par symbole
- Persistance d'état: `runtime/logs/v5c-highrisk-paper/state.json`
  - restauré au boot (equity, positions nettes, derniers prix, signaux/params)
  - évite le reset d'equity à chaque patch/restart
- Logs:
  - `trades.jsonl` (inclut `metadata.combo_breakdown`)
  - `pnl.jsonl`

---

## 8. Réoptimisation — Workflow Semi-Automatique

### Principe fondamental

> La fréquence de réoptimisation est un **méta-paramètre backtesté**. On ne réoptimise pas sur un coup de tête — on suit le schedule validé par la méta-optimisation.

### Contrainte : le PC local ne tourne pas 24/7

La réoptimisation nécessite du calcul lourd (Optuna) qui ne peut pas tourner sur le VPS. Le PC local n'est pas allumé en permanence.

### Workflow

```
1. Le CLOUD détecte qu'une stratégie doit être réoptimisée
   (date actuelle > dernière réoptim + reoptim_frequency)
   │
   ▼
2. Le CLOUD STOPPE la stratégie concernée
   → Plus aucun ordre passé pour cette stratégie
   → Les positions ouvertes sont fermées proprement
   → Alerte Telegram : "⏸️ Stratégie X stoppée — réoptimisation requise"
   │
   ▼
3. TU reçois la notification Telegram
   → Tu ouvres ton PC quand tu peux
   → Tu lances la réoptimisation (un script one-click)
   → Le script :
     a. Télécharge les dernières données
     b. Relance Optuna sur la training_window
     c. Valide les nouveaux paramètres
     d. Met à jour active_profiles.json
     e. git push → CI/CD → déploiement automatique
   │
   ▼
4. Le CLOUD détecte les nouveaux paramètres
   → Reprend le trading avec les params frais
   → Alerte Telegram : "▶️ Stratégie X réoptimisée et active"
```

### Pourquoi stopper plutôt que continuer ?

- Des paramètres périmés peuvent **perdre de l'argent**
- Le `reoptim_frequency` a été optimisé : au-delà, les params ne sont plus fiables
- Mieux vaut **ne pas trader** que trader avec des params obsolètes
- Le temps d'arrêt est court (quelques heures max)

### Cas limites

| Situation | Comportement |
|-----------|-------------|
| Tu es en vacances 1 semaine | La stratégie reste stoppée. Pas de perte, pas de gain. |
| Plusieurs stratégies à réoptimiser | Le script les traite toutes en batch. |
| La réoptimisation donne de mauvais résultats | Le script t'alerte. Tu décides de ne pas redéployer. |
| Le VPS crash | Systemd restart automatique. Positions fermées au restart. |

---

## 9. Monitoring & Alertes (Telegram)

### Alertes automatiques

| Type | Message | Quand |
|------|---------|-------|
| 📊 Rapport quotidien | PnL du jour, positions ouvertes, equity | Tous les jours 20h |
| ⏸️ Réoptimisation requise | "Stratégie X stoppée, réoptimisation requise" | Quand reoptim_frequency atteint |
| ▶️ Stratégie active | "Stratégie X réoptimisée et active" | Après push des nouveaux params |
| � Erreur technique | API down, ordre rejeté, connexion perdue | Immédiat |
| ⚠️ Circuit breaker | "DD max atteint, stratégie X en pause" | Quand DD > seuil backtesté |

### Ce qu'on ne fait PAS

- Pas d'alerte "performance dégradée" → c'est du bruit, le reoptim_frequency gère ça
- Pas de décision automatique de réoptimisation → c'est le schedule backtesté qui décide
- Pas de modification de paramètres en live → tout passe par le pipeline local

---

## 10. Décisions Techniques

| Question | Options | Décision |
|----------|---------|----------|
| Optimiseur | Optuna (TPE + MedianPruner) | ✅ Converge vite, prune les mauvais trials |
| Backtester | Vectorisé numpy | ✅ ~100x plus rapide qu'event-driven |
| Données | Parquet | ✅ Lectures rapides, bonne compression |
| Évaluation | Walk-forward multi-seed | ✅ Tout est OOS, reproductible, anti-overfitting |
| Méta-optimisation | Optuna sur méta-params | ❌ Abandonné (defaults gagnent, test A/B) |
| SL/TP | ATR-based (V5) | ✅ Adaptatif à la volatilité, R:R optimisable |
| Position sizing | Risk-based (V5) | ✅ position = equity × risk% / SL_distance |
| Exits avancées | Trailing + breakeven + max hold (V5b) | ✅ Optimisables par Optuna |
| Overlays | Regime + Vol targeting | ✅ Coupe signaux en range/crise, normalise vol |
| Portfolio | Markowitz contraint (Ledoit-Wolf) | ✅ Covariance-aware, hard constraints |
| Cloud provider | Hetzner (3€), DigitalOcean (5€), OVH (3.50€) | À décider |
| Alertes | Telegram Bot | ✅ Décidé |
| Allocation portfolio | top3_heavy (V4b) | ✅ 25/25/15/10/10/5/5/5 |

---

## 11. Évolution des Versions

### Backtester

| Version | Feature | Impact |
|---------|---------|--------|
| V1 | Bug double-comptage equity | ❌ PF < 1 avec Sharpe > 1 |
| V2 | Cash + capital alloué, funding rate, daily reset | ✅ Métriques cohérentes |
| V3 | Slippage dynamique ATR, circuit breaker | ✅ Réalisme accru |
| V4 | Overlays (regime + vol targeting), signaux fractionnels | ✅ DD réduit massivement |
| V5 | ATR-based SL/TP, risk-based position sizing | ✅ R:R adaptatif, sizing intelligent |
| V5b | Trailing stop, breakeven stop, max holding period | ✅ Exits avancées optimisables |

### Stratégies

| Version | Count | Nouveautés |
|---------|-------|------------|
| V1 | 10 | RSI, MACD, Bollinger, EMA, VWAP, Donchian, Stochastic, Ichimoku, ATR, OBV |
| V2 | 12 | + momentum_roc, adx_regime |
| V3 | 16 | + keltner_channel, mean_reversion_zscore, supertrend, williams_r |
| V4 | 22 | + supertrend_adx, trend_multi_factor, breakout_regime, regime_adaptive, mtf_trend_entry, mtf_momentum_breakout |
| V5 | 22 | + generate_signals_v5() API (ATR SL/TP + sl_distances) |
| V5b | 22 | + trailing_atr_mult, breakeven_trigger_pct, max_holding_bars |

### Portfolios

| Version | Return | Sharpe | DD | Statut |
|---------|--------|--------|----|--------|
| V1 | +5.7% | 0.26 | -8.1% | Archivé |
| V2 | +22.9% | 0.66 | -5.6% | Archivé |
| V3 | +8.5% | 1.06 | -6.6% | Archivé |
| V3b | +9.8% | 1.19 | -4.9% | Archivé |
| V4 | +4.9% | 2.59 | -0.8% | Archivé (trop conservateur) |
| V4b | +19.8% | 1.35 | -8.5% | Archivé (remplacé par V5b) |
| **V5b Conserv.** | **+2.9%** | **2.48** | **-0.6%** | **✅ 95/100 GO** |
| **V5b Modéré** | **+7.4%** | **2.48** | **-1.6%** | **✅ 95/100 GO** |
| **V5b Agressif** | **+15.1%** | **2.49** | **-3.2%** | **✅ 95/100 GO** |
| **V5c HighRisk** | **+12.1% (OOS 60j)** | **3.93** | **-2.3%** | **⚠️ Spéculatif (1-2 mois, capital 100$)** |

---

## 12. Roadmap

| Phase | Tâche | Statut |
|-------|-------|--------|
| **1** | Corrections backtester (DD, funding, daily reset) | ✅ Terminé |
| **2** | Nouvelles stratégies (12 → 16 → 22) | ✅ Terminé |
| **3** | Ingestion multi-actif (5 actifs × 4 TFs) | ✅ Terminé |
| **4** | Diagnostic V2/V3/V4 (multi-seed, pruning, 2-pass) | ✅ Terminé |
| **5** | Test A/B méta-opt vs defaults → defaults gagnent | ✅ Terminé |
| **6** | Audit edge + modules (regime.py, overlays.py) | ✅ Terminé |
| **7** | Portfolio V1 → V3b → V4 → V4b (+19.8%, objectif atteint) | ✅ Terminé |
| **8** | V5 : ATR-based SL/TP + risk-based sizing (22 strats) | ✅ Terminé |
| **9** | Diagnostic V5 : 121 survivants, V5 > V4 (+0.254 Sharpe) | ✅ Terminé |
| **10** | V5b : trailing stop + breakeven + max holding (22 strats) | ✅ Terminé |
| **11** | Diagnostic V5b : 79 STRONG, multi-seed 3 + risk grid + correlation | ✅ Terminé |
| **12** | Portfolio V5b : 3 profils, audit complet, confiance 90-95/100 | ✅ Terminé |
| **13** | Module live/ (signal_runner, executor, scheduler) | ⏳ Pending |
| **14** | Monitoring + alertes Telegram | ⏳ Pending |
| **15** | Dockerisation + déploiement VPS | ⏳ Pending |
| **16** | Paper trading (2-4 semaines) | ⏳ Pending |
| **17** | Go live capital réel | ⏳ Pending |

---

## 13. Leçons Clés Apprises

1. **Multi-factor > Single-indicator** : le meilleur multi-factor (Sharpe 0.935) bat le meilleur simple (0.444) de 2×
2. **Defaults > Méta-optimisation** : les defaults fixes font mieux que la méta-opt (+0.102 Sharpe)
3. **IS Sharpe négatif ≠ mauvaise stratégie** : les combos qui ne sur-fittent pas performent mieux en holdout
4. **La variance inter-seeds est le vrai signal** : Sharpe 0.18 (std 0.13) > Sharpe 0.78 (std 0.86)
5. **ETH est le marché le plus tradable** : 7/11 survivants holdout
6. **4h est le timeframe optimal** : 6/11 survivants, bon compromis signal/bruit
7. **Les overlays réduisent le DD** mais aussi le return — à utiliser sélectivement
8. **ATR-based SL/TP (V5)** améliore 47/81 combos de +0.254 Sharpe en moyenne

---

*Voir `docs/carnet_de_bord.md` pour le journal chronologique détaillé (16 sessions).*
