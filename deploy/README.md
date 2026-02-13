# Deploy VPS par portfolio (GHCR + docker-compose)

Ce dossier contient le déploiement runtime minimal pour les services portfolio.

## Fichiers

- `Dockerfile` : image cloud minimale (runtime uniquement)
- `docker-compose.portfolio.yml` : service `v5c-highrisk-paper`

## Pipeline CI/CD

Workflow: `.github/workflows/deploy-portfolio.yml`

Déclenchement manuel (`workflow_dispatch`) avec:
- `portfolio_id` (ex: `v5c-highrisk-paper`)
- `image_tag` (ex: `latest`)

## Secrets GitHub requis

- `VPS_HOST`
- `VPS_USER`
- `VPS_SSH_KEY`
- `GHCR_USERNAME`
- `GHCR_TOKEN`
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`

## Runtime config (portfolio)

- `config/live/portfolios/v5c-highrisk-paper.json`
- `config/live/profiles/v5c-highrisk-paper.meta_profiles.json`

## Notes importantes

1. Mode actuel: **paper** (`dry_run=true`), capital paper de suivi = 1000 USD.
2. Réoptimisation: fréquence `1M`, pause si échéance dépassée.
3. Le runner live actuel exécute 1 profile (MVP). Le passage au vrai moteur multi-combos portfolio sera une phase suivante.
