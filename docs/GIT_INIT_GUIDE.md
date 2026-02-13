# Git Init Guide — Quantlab V7

Ce projet est prêt pour une initialisation Git propre.

## Pré-requis

- `.gitignore` est configuré pour exclure:
  - environnements virtuels
  - caches
  - data locale
  - résultats runtime
  - secrets (`.env`)

## Étapes

```bash
git init
git add .
git commit -m "chore: initialize Quantlab V7 project structure"
```

## Vérifications recommandées avant commit

```bash
git status
```

Vérifier que ces dossiers/fichiers **ne sont pas trackés**:
- `venv/`
- `data/raw/`
- `data/processed/`
- `results/`
- `logs/`
- `.env`

## Optionnel: branche principale

```bash
git branch -M main
```

## Optionnel: remote GitHub

```bash
git remote add origin <URL_REPO>
git push -u origin main
```
