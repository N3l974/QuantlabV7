# 22 — Préparation déploiement VPS par portfolio (V5c paper)

**Date**: 13 fév 2026  
**Scope**: infrastructure de déploiement cloud minimal + configuration portfolio-scopée

## Objectif

Préparer un déploiement VPS de `v5c-highrisk` en mode paper, avec:
- capital paper de suivi: **1000 USD**
- pipeline **local code -> GitHub -> VPS**
- runtime minimal côté VPS (pas le repo complet)
- base extensible pour déployer plusieurs portfolios en parallèle

## Décisions validées

1. Déploiement **par portfolio** (service dédié).
2. Registry image: **GHCR** (GitHub Container Registry).
3. Orchestration VPS: **docker-compose**.
4. Réoptimisation V5c paper: **1M**.
5. Passage paper -> réel: revue après **8-12 semaines**, avec DD paper max **15%**.

## Changements implémentés

- Runner portfolio-scopé: `live/run_portfolio.py`
- Config portfolio paper: `config/live/portfolios/v5c-highrisk-paper.json`
- Source profile live (MVP): `config/live/profiles/v5c-highrisk-paper.meta_profiles.json`
- Docker build minimal: `deploy/Dockerfile`
- Compose VPS portfolio: `deploy/docker-compose.portfolio.yml`
- CI/CD GitHub: `.github/workflows/deploy-portfolio.yml`
- Contexte build réduit: `.dockerignore`
- Documentation ops: `deploy/README.md`

## Limitation actuelle (connue)

Le runtime live actuel exécute un **profile unique** (MVP), pas encore l’agrégation multi-combos complète d’un portfolio V5c. L’infra est prête pour le déploiement par portfolio; le moteur d’exécution portfolio natif est la prochaine étape.

## Artefacts

- Compose VPS: `deploy/docker-compose.portfolio.yml`
- Workflow CI/CD: `.github/workflows/deploy-portfolio.yml`
- Config portfolio: `config/live/portfolios/v5c-highrisk-paper.json`
