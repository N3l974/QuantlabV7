# Base de Connaissances — Quantlab V7

Documentation technique complète du projet, architecture, stratégies, bugs, méthodologie et pistes d'amélioration.

---

## Structure

| Fichier | Contenu | Pourquoi ? |
|---------|---------|------------|
| [`01_architecture.md`](01_architecture.md) | Architecture technique (backtester, walk-forward, meta-opt, portfolio) | Comprendre les briques fondamentales |
| [`02_strategies.md`](02_strategies.md) | Catalogue des 16 stratégies (famille, params, forces/faiblesses) | Savoir ce qu'on a et ce qui manque |
| [`03_bugs_fixes.md`](03_bugs_fixes.md) | Bugs rencontrés, root cause, fix appliqué, leçons | Éviter de répéter les mêmes erreurs |
| [`04_methodologie.md`](04_methodologie.md) | Méthodologie de recherche, biais, failles identifiées | Améliorer la rigueur scientifique |
| [`05_ameliorations.md`](05_ameliorations.md) | Audit complet : améliorations par impact/effort | Roadmap d'innovation |

---

## Usage

- **Nouveau dev** : lire `01_architecture.md` puis `02_strategies.md`
- **Debug** : consulter `03_bugs_fixes.md` avant de chercher
- **Recherche** : lire `04_methodologie.md` pour les biais à éviter
- **Innovation** : `05_ameliorations.md` pour les idées prioritaires

---

## Principe

Cette base de connaissances est **vivante** :
- Chaque bug/fix est documenté immédiatement
- Chaque amélioration est analysée avant implémentation
- Chaque décision est justifiée avec des preuves (code, résultats)

**Objectif** : éviter la perte de savoir, accélérer le développement, et maintenir la rigueur face à la complexité croissante du projet.
