# Portfolio V5c-HighRisk (court terme)

## Objectif

Construire un portefeuille **high risk** orienté gains court terme (1-2 mois), avec:
- **capital initial**: 100 USD
- **contrainte de drawdown max**: -30%

## Résultat actuel (run strict OOS, 2026-02-13 09:39)

| Profil retenu (calibré train) | Return test OOS | Return 30j OOS | Return 60j OOS | Sharpe OOS | Max DD OOS |
|---|---:|---:|---:|---:|---:|
| **HR-75** (max_position=75%) | **+12.1%** | **+12.7%** | **+12.1%** | 3.93 | -2.3% |

✅ La contrainte est respectée: **DD -2.3% > -30%**.

> Méthode de validation: sélection/optimisation sur TRAIN, évaluation finale sur les 60 dernières barres (test hors calibration).

## Structure

```
v5c-highrisk/
├── README.md
├── code/
│   └── portfolio_v5c_highrisk.py
├── docs/
│   └── portfolio_v5c_highrisk.md
└── results/
    └── portfolio_v5c_highrisk_*.json
```

## Logique de construction

1. Charger les survivants STRONG du diagnostic V5b.
2. Déduplication corrélation (seuil 0.92).
3. Split temporel strict: TRAIN + TEST (60 dernières barres).
4. Sélection top 6 combos orientés court terme sur TRAIN uniquement.
5. Optimisation Markowitz en mode `max_return` sur TRAIN.
6. Choix du profil risque sur TRAIN puis évaluation finale sur TEST uniquement.

## Exécution

```bash
python scripts/portfolio_v5c_highrisk.py
```

## Déploiement VPS (paper)

- **Service**: `v5c-highrisk-paper`
- **Mode**: paper (`dry_run=true`)
- **Capital paper de suivi**: **1000 USD**
- **Fréquence de réoptimisation**: **1M** (pause si échéance dépassée)
- **Fenêtre avant passage réel**: **8 à 12 semaines** de paper stable
- **Garde-fou GO live réel**: DD paper max **15%**

Configuration source:
- `config/live/portfolios/v5c-highrisk-paper.json`
