# Portfolio V5c-HighRisk (court terme)

## Objectif

Construire un portefeuille **high risk** orienté gains court terme (1-2 mois), avec:
- **capital initial**: 100 USD
- **contrainte de drawdown max**: -30%

## Thèse d'investissement

V5c-HighRisk est une version opportuniste de V5, conçue pour capter des impulsions de marché sur un horizon court.

- Univers de départ: survivants STRONG issus du diagnostic V5b.
- Construction concentrée: jusqu'à 6 combos, orientés rendement court terme.
- Allocation: optimisation Markowitz orientée `max_return`.
- Risque: profil sélectionné sur TRAIN dans une grille high-risk (`HR-50`, `HR-75`, `HR-100`).

## Résultat actuel (run strict OOS, 2026-02-13 09:39)

| Profil retenu (calibré train) | Return test OOS | Return 30j OOS | Return 60j OOS | Sharpe OOS | Max DD OOS |
|---|---:|---:|---:|---:|---:|
| **HR-75** (max_position=75%) | **+12.1%** | **+12.7%** | **+12.1%** | 3.93 | -2.3% |

✅ La contrainte est respectée: **DD -2.3% > -30%**.

> Méthode de validation: sélection/optimisation sur TRAIN, évaluation finale sur les 60 dernières barres (test hors calibration stricte).

## Processus train/validation (sans fuite)

1. Charger les survivants et préparer les signaux/SL.
2. **Split temporel strict**:
   - **TRAIN** = historique avant la fenêtre finale.
   - **TEST OOS** = **60 dernières barres** (jamais utilisées pour calibrer).
3. Déduplication corrélation sur TRAIN (`max_corr=0.92`).
4. Ranking et sélection top combos sur métriques TRAIN uniquement.
5. Optimisation des poids sur TRAIN (`objective=max_return`).
6. Sélection du profil de risque high-risk sur TRAIN uniquement.
7. Évaluation finale unique sur TEST OOS (60 barres).

## Périodes et fenêtres utilisées

- **Fenêtre TEST OOS**: `TEST_BARS = 60` barres.
- **Horizon court terme évalué**:
  - `SHORT_1M_BARS = 30` (proxy ~30 jours sur TF 1d).
  - `SHORT_2M_BARS = 60` (proxy ~60 jours sur TF 1d).
- **TRAIN**: tout l'historique disponible avant les 60 dernières barres.

## Allocations retenues (run strict OOS)

| Poids | Combo | Sharpe TRAIN | Return TRAIN | DD TRAIN |
|------:|-------|-------------:|-------------:|---------:|
| 40.0% | ETHUSDT/breakout_regime/1d | 1.70 | 0.9% | -0.1% |
| 25.0% | SOLUSDT/mtf_momentum_breakout/1d | 1.01 | 0.6% | -0.5% |
| 12.7% | ETHUSDT/ema_ribbon/1d | 0.86 | 1.4% | -1.1% |
| 11.9% | ETHUSDT/macd_crossover/1d | 1.15 | 2.8% | -1.6% |
| 9.0% | ETHUSDT/trend_multi_factor/1d | 1.19 | 3.5% | -2.3% |
| 1.4% | ETHUSDT/bollinger_breakout/1d | 1.10 | 7.5% | -5.0% |

## Structure

```
v5c-highrisk/
├── README.md
├── code/
│   └── portfolio_v5c_highrisk.py
└── results/
    └── portfolio_v5c_highrisk_*.json
```

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

### Reporting quotidien (paper)

```bash
python scripts/paper_daily_report.py \
  --log-dir runtime/logs/v5c-highrisk-paper \
  --hours 24
```

Ce rapport synthétise:
- nombre de trades (total + fenêtre),
- equity début/fin,
- return cumulé,
- max drawdown observé,
- pnl cumulé sur la fenêtre.
