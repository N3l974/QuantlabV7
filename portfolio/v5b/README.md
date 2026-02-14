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

## Artefacts techniques

- Code : `portfolio/v5b/code/`
- Résultats : `portfolio/v5b/results/`

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

## Position sizing

| Paramètre | Conservateur | Modéré | Agressif |
|-----------|-------------|--------|----------|
| `max_position_pct` | **10%** | **25%** | **50%** |
| `max_drawdown_pct` (circuit breaker) | 10% | 15% | 25% |
| Leverage Binance recommandé | 3x | 5x | 5x |

> Sur Binance Margin, le leverage est un plafond de marge. Le risque réel est piloté par `max_position_pct`.

## Utilisation

```bash
python portfolio/v5b/code/portfolio_v5b_final.py
```

- Résultats: `portfolio/v5b/results/portfolio_v5b_final_*.json`
