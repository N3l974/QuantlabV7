# Portfolio artifacts — QuantLab V7

Chaque portfolio est désormais **auto-contenu**:
- `README.md` : documentation métier + état du portfolio
- `code/` : scripts de diagnostic/construction
- `results/` : diagnostics, JSON de runs, rapports markdown

## Règle de gouvernance (workflow courant)

- Pas de séparation `docs/portfolios` vs `portfolio/*`.
- Les artefacts et la doc de chaque portfolio vivent ensemble dans `portfolio/<version>/`.
- `docs/results/README.md` et `docs/portfolios/README.md` servent d'index de compatibilité.

## Structure

```
portfolio/
├── README.md
├── ftmo-v1/
│   ├── README.md
│   ├── code/
│   ├── config/
│   └── results/
├── v4b/
│   ├── README.md
│   ├── code/
│   └── results/
├── v5b/
│   ├── README.md
│   ├── code/
│   └── results/
└── v5c-highrisk/
    ├── README.md
    ├── code/
    └── results/
```
