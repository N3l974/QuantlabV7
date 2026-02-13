# Active Context — Quantlab V7

## Objectif actuel
Maintenir et faire évoluer Portfolio V5b avec profils de risque par **position sizing**.

## État validé
- V5b utilise les mêmes combos et poids entre profils.
- Différence de risque via `max_position_pct`.
- Résultats validés : Conservateur < Modéré < Agressif (return).

## Priorités
1. Stabiliser workflow de recherche / livraison.
2. Garder un contexte court et actionnable.
3. Documenter les décisions clés, archiver le bruit historique.

## Fichiers de référence rapide
- `README.md`
- `portfolio/v5b/README.md`
- `scripts/portfolio_v5b_final.py`
- `docs/knowledge_base/01_architecture.md`

## Règles de mise à jour
- Ajouter uniquement : objectif, décisions, next steps.
- Déplacer les détails d'implémentation longs dans `docs/archive/`.
- Supprimer les points obsolètes à chaque session.
