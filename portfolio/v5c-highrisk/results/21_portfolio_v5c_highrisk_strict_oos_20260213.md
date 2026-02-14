# 21 — Portfolio V5c HighRisk (Strict OOS)

**Date**: 13 fév 2026  
**Script**: `scripts/portfolio_v5c_highrisk.py`  
**Mode**: `strict_oos_60d`

## Objectif

Construire un portefeuille high-risk court terme (1-2 mois) avec:
- capital initial: 100 USD
- contrainte max drawdown: -30%

## Protocole de validation (strict)

1. Calibration sur **TRAIN** (toutes les barres sauf les 60 dernières).
2. Choix du profil risque sur TRAIN seulement.
3. Évaluation finale sur **TEST OOS** (60 dernières barres, jamais utilisées pour calibrer).

## Résultat final (TEST OOS)

- Profil sélectionné (depuis TRAIN): **HR-75**
- Return 30j OOS: **+12.7%**
- Return 60j OOS: **+12.1%**
- Sharpe OOS: **3.93**
- Max DD OOS: **-2.3%**

✅ Contrainte DD respectée (`-2.3% > -30%`).

## Artefacts

- JSON final: `portfolio/v5c-highrisk/results/portfolio_v5c_highrisk_20260213_093913.json`
- Documentation canonique portfolio: `portfolio/v5c-highrisk/README.md`
- Pointeur technique portfolio: `portfolio/v5c-highrisk/README.md`

## Note méthodologique

Le précédent run V5c (+39.6%) était plus optimiste car la validation n'était pas strictement séparée train/test.
Ce run devient la référence canonique V5c car il applique un protocole OOS strict.
