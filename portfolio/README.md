# Portfolios — QuantLab V7

Ce dossier contient les portfolios validés et prêts pour déploiement live.

Seuls les portfolios ayant passé la validation complète (walk-forward + holdout + Monte Carlo) sont conservés ici.

## Portfolios actifs

| Portfolio | Return | Sharpe | Max DD | Calmar | Statut |
|-----------|--------|--------|--------|--------|--------|
| [V4b](v4b/) | +19.8% | 1.35 | -8.5% | 2.17 | Archivé (remplacé par V5b) |
| **[V5b Conservateur](v5b/)** | **+2.9%** | **2.48** | **-0.6%** | **4.37** | **✅ 95/100 GO** |
| **[V5b Modéré](v5b/)** | **+7.4%** | **2.48** | **-1.6%** | **4.37** | **✅ 95/100 GO** |
| **[V5b Agressif](v5b/)** | **+15.1%** | **2.49** | **-3.2%** | **4.39** | **✅ 95/100 GO** |
| **[V5c HighRisk](v5c-highrisk/)** | **+12.1% (OOS)** | **3.93** | **-2.3%** | **31.27** | **⚠️ Spéculatif (court terme 1-2 mois)** |

## Structure

```
portfolio/
├── README.md              # Ce fichier
├── v4b/                   # Portfolio V4b (archivé)
│   ├── code/              # Scripts de construction et diagnostic
│   ├── docs/              # Documentation et présentation investisseur
│   └── results/           # Résultats JSON de validation
├── v5b/                   # Portfolio V5b (actif — 3 profils de risque)
│   ├── code/              # portfolio_v5b_final.py, diagnostic_v5b.py
│   ├── docs/              # Rapport multi-profil + audit + confiance
│   └── results/           # Résultats JSON (métriques, audit, MC)
└── v5c-highrisk/          # Portfolio V5c (spéculatif court terme)
    ├── code/              # portfolio_v5c_highrisk.py
    ├── docs/              # Rapport high-risk
    └── results/           # Résultats JSON
```

## Historique

| Version | Date | Return | Sharpe | DD | Statut | Raison |
|---------|------|--------|--------|----|--------|--------|
| V3 | Jan 2026 | +8.5% | 1.06 | -6.6% | Archivé | Remplacé par V3b |
| V3b | Fév 2026 | +9.8% | 1.19 | -4.9% | Archivé | Concentration ETH 95%, return insuffisant |
| V4 | Fév 2026 | +4.9% | 2.59 | -0.8% | Archivé | Trop conservateur (+4.9% vs objectif +15%) |
| V4b | Fév 2026 | +19.8% | 1.35 | -8.5% | Archivé | Remplacé par V5b |
| **V5b** | **Fév 2026** | **+2.9% → +15.1%** | **2.48-2.49** | **-0.6% → -3.2%** | **Actif** | **3 profils (sizing), audit complet, confiance 95/100** |
| **V5c-highrisk** | **Fév 2026** | **+12.1% (OOS 60j)** | **3.93** | **-2.3%** | **Actif (spéculatif)** | **Validation stricte train/test, objectif 1-2 mois, capital 100$, DD max 30%** |

## Évolution technique

| Version | Innovation clé |
|---------|---------------|
| V3 | Holdout temporel, multi-seed |
| V3b | MC holdout-only, cap par symbol |
| V4 | Overlays (regime + vol targeting), Markowitz Ledoit-Wolf, 37 combos |
| V4b | Concentration top 8, top3_heavy, leverage 1.5x |
| V5 | ATR-based SL/TP, risk-based sizing, generate_signals_v5() |
| V5b | Trailing stop, breakeven stop, max holding, multi-seed 3, risk grid, corrélation |

---

*QuantLab V7 — Février 2026*
