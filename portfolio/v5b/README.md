# Portfolio V5b — QuantLab V7

## Vue d'ensemble

Portfolio V5b est le portefeuille de trading quantitatif de nouvelle génération de QuantLab V7. Il intègre les features V5b (trailing stop, breakeven, max holding) et propose **3 profils de risque** différenciés par la **taille des positions** (`max_position_pct`), validés par un audit complet et un score de confiance pour le déploiement live.

**Principe** : mêmes 8 combos, mêmes poids Markowitz, seul le sizing change.

### Résumé des 3 profils

| Profil | Max Position | Sharpe | Sortino | Return | Max DD | Calmar | Confiance |
|--------|-------------|--------|---------|--------|--------|--------|-----------|
| **Conservateur** | 10% | 2.48 | 4.66 | +2.9% | -0.6% | 4.37 | **95/100 GO ✅** |
| **Modéré** | 25% | 2.48 | 4.64 | +7.4% | -1.6% | 4.37 | **95/100 GO ✅** |
| **Agressif** | 50% | 2.49 | 4.60 | +15.1% | -3.2% | 4.39 | **95/100 GO ✅** |

> Les 3 profils passent le seuil GO (≥80/100). DD max = -3.2% (bien sous la limite de -15%).

## Structure

```
v5b/
├── README.md                              # Documentation de référence
├── code/
│   ├── portfolio_v5b_final.py             # Construction multi-profil + audit + confiance
│   └── diagnostic_v5b.py                  # Diagnostic V5b (multi-seed, risk grid)
└── results/
    └── portfolio_v5b_final_*.json         # Résultats (métriques, audit, MC, allocations)
```

## Processus détaillé de construction

1. **Diagnostic amont** (multi-seed, multi-paramètres) pour extraire les survivants robustes.
2. **Préparation des combos** (signaux + distances SL si dispo V5).
3. **Déduplication corrélation** pour éviter les redondances (seuil corrélation max).
4. **Sélection des 8 combos** les plus solides sur métriques de calibration.
5. **Optimisation des poids** (Markowitz orienté Sharpe) sur la période de calibration.
6. **Backtest multi-profils** avec mêmes combos/poids, seul le sizing change (`max_position_pct`).
7. **Audit complet** (rolling Sharpe, mensuel, stress tests, concentration, corrélation).
8. **Monte Carlo** block-bootstrap + score de confiance live.

## Protocole train/validation

- **Train / calibration**: pipeline walk-forward + sélection de combos + optimisation de poids.
- **Validation**: exécution sur période holdout non vue pour mesurer robustesse réelle.
- **Anti-fuite**: les décisions de sélection/poids/profil sont prises avant l'analyse finale holdout.

## Périodes et fenêtres utilisées

- **Fenêtre holdout finale**: 12 mois (fév. 2025 → fév. 2026).
- **Rolling Sharpe audit**: fenêtre 60 barres.
- **Analyse mensuelle**: agrégation par blocs ~30 barres.
- **Horizons Monte Carlo**: 3M, 6M, 12M, 24M, 36M.
- **Réoptimisation de référence** (méta-profils source): fréquence typique 1M à 3M selon profil.

## Comment ça marche : Position Sizing

Les 3 profils utilisent les **mêmes stratégies et poids**. La seule différence est `max_position_pct` — le % maximum du capital alloué par trade.

| Paramètre | Conservateur | Modéré | Agressif |
|-----------|-------------|--------|----------|
| `max_position_pct` | **10%** | **25%** | **50%** |
| `max_drawdown_pct` (circuit breaker) | 10% | 15% | 25% |
| Leverage Binance recommandé | 3x | 5x | 5x |

> Sur Binance Margin, le leverage est juste le plafond de marge disponible. On ne l'utilise pas à fond — c'est `max_position_pct` qui contrôle le risque réel.

## Profils de risque

### 🟢 Conservateur — Position max 10%

| Métrique | Valeur |
|----------|--------|
| Return | +2.9% |
| Sharpe | 2.48 |
| Max DD | -0.6% |
| MC P(gain 12M) | 95% |
| MC P(ruine) | 0.0% |

### 🟡 Modéré — Position max 25%

| Métrique | Valeur |
|----------|--------|
| Return | +7.4% |
| Sharpe | 2.48 |
| Max DD | -1.6% |
| MC P(gain 12M) | 97% |
| MC P(ruine) | 0.0% |

### 🔴 Agressif — Position max 50%

| Métrique | Valeur |
|----------|--------|
| Return | **+15.1%** |
| Sharpe | 2.49 |
| Max DD | -3.2% |
| MC P(gain 12M) | 97% |
| MC P(ruine) | 0.0% |

## Audit de fiabilité

Chaque profil est audité sur :

1. **Rolling Sharpe (60j)** — Stabilité temporelle, % positif, 1ère vs 2ème moitié
2. **Analyse mensuelle** — Pire/meilleur mois, % mois positifs
3. **Stress tests** — VaR 95%, CVaR 95%, max losing streak, recovery time
4. **Concentration HHI** — N effectif symbols et stratégies
5. **Corrélation intra-portfolio** — Corrélation moyenne et max entre combos
6. **Features V5b** — Utilisation trailing stop, breakeven, max holding

## Score de confiance live (10 critères, /100)

| Critère | Points max |
|---------|-----------|
| Sharpe ≥ 1.5 | 15 |
| Sortino ≥ 1.5 | 10 |
| DD dans target | 15 |
| Rolling Sharpe >0 ≥ 70% | 10 |
| Stabilité temporelle (2 moitiés >0) | 10 |
| Mois positifs ≥ 60% | 10 |
| Diversification (N_eff ≥ 2.5) | 10 |
| MC P(gain 12M) ≥ 90% | 10 |
| MC P(ruine) ≤ 1% | 5 |
| Multi-seed 3 validé | 5 |

**Verdict** : ≥80 = GO ✅ | 60-79 = GO PRUDENT ⚠️ | 40-59 = ATTENDRE 🔶 | <40 = NO-GO ❌

## Innovations V5b vs V4b

| Feature | V4b | V5b |
|---------|-----|-----|
| Trailing stop ATR | ❌ | ✅ (100% des combos) |
| Breakeven | ❌ | ✅ |
| Max holding period | ❌ | ✅ |
| Multi-seed validation | 1 seed | 3 seeds |
| Corrélation dedup | ❌ | ✅ (corr > 0.85) |
| Profils de risque (sizing) | 1 (leverage) | 3 (position sizing) |
| Audit complet | ❌ | ✅ (rolling Sharpe, HHI, stress) |
| Score confiance live | ❌ | ✅ (10 critères /100) |

## Comparaison historique

| Métrique | V3b | V4 | V4b | V5b Conserv. | V5b Modéré | V5b Agressif |
|----------|-----|-----|-----|-------------|------------|--------------|
| Return | +9.8% | +4.9% | +19.8% | +2.9% | +7.4% | **+15.1%** |
| Sharpe | 1.19 | 2.59 | 1.35 | **2.48** | **2.48** | **2.49** |
| Max DD | -4.9% | -0.8% | -8.5% | **-0.6%** | -1.6% | -3.2% |
| Calmar | 1.91 | 5.99 | 2.17 | 4.37 | 4.37 | 4.39 |
| Confiance | — | — | — | 95/100 | 95/100 | 95/100 |

## Utilisation

### Rejouer la construction

```bash
cd /path/to/Quantlab-V7
python scripts/portfolio_v5b_final.py
```

### Voir les résultats

- `results/portfolio_v5b_final_*.json` : Données brutes (métriques, audit, MC, allocations)
- `README.md` (ce fichier) : synthèse de référence (thèse, protocole, périodes, résultats)

---

*Portfolio V5b — QuantLab V7 — Février 2026*
