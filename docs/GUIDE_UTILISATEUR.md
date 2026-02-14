# Guide Utilisateur — Workflow optimal (low-context)

Ce guide explique comment travailler vite sur Quantlab V7 **sans gonfler le contexte**.

---

## 1) Règle d'or

Toujours séparer:
1. **Contexte actif** (court, actionnable)
2. **Historique** (long, archivé)

Le contexte actif vit dans `docs/context/ACTIVE_CONTEXT.md`.
Tout le reste est référencé via `docs/context/ARCHIVE_INDEX.md`.

### Méthode unique (obligatoire)

Pour **chaque run**, appliquer toujours ce contrat:

1. **Raw outputs** (JSON/logs) dans `portfolio/<version>/results/`
2. **Rapport court** dans `portfolio/<version>/results/` (1 markdown par run significatif)
3. **Documentation portfolio** dans `portfolio/<version>/README.md`
4. **Carnet**: ajouter une entrée courte dans `docs/carnet_de_bord.md` (objectif, commande, résultat, lien)
5. **Index**: mettre à jour `docs/results/README.md` si le run est important

Objectif: traçabilité complète sans dupliquer les artefacts partout.

---

## 2) Démarrer une session (2 min)

1. Lire:
   - `docs/context/ACTIVE_CONTEXT.md`
   - `portfolio/v5b/README.md` (si tâche portfolio)
2. Copier le template:
   - `docs/context/SESSION_TEMPLATE.md`
3. Remplir seulement:
   - objectif
   - contraintes
   - fichiers touchés
   - succès attendu

> Limite recommandée: 5 lignes d'état max.

---

## 3) Travailler sans perdre de détails

### A. Pendant l'implémentation
- Documenter les décisions en 1 ligne (quoi/pourquoi)
- Éviter les copiers-collers de logs dans les prompts
- Référencer les chemins de fichiers plutôt que le contenu complet

### B. Après exécution
- Garder dans le compte rendu:
  - commande exécutée
  - résultat clé (succès/échec)
  - impact sur métriques
- Archiver les détails longs dans:
  - `docs/archive/`
  - ou `portfolio/<version>/results/`

Règle anti-bloat:
- Ne pas créer de rapport pour un run mineur/non significatif
- Éviter de versionner les JSON intermédiaires volumineux
- Garder 1 artefact final par version de portfolio (ex: V5b final, V5c strict OOS)

### C. Mise à jour du contexte actif
Mettre à jour `ACTIVE_CONTEXT.md` avec:
- ce qui est validé
- prochaine action
- blocages

Supprimer ce qui est obsolète.

---

## 4) Structure documentaire recommandée

- `README.md` : vision + quick start
- `docs/context/` : pilotage quotidien
- `portfolio/<version>/` : documentation + code + résultats d'un portfolio
- `docs/knowledge_base/` : savoir technique stable
- `docs/archive/` : historique long
- `docs/results/` : index de compatibilité des rapports migrés

---

## 5) Workflow quotidien conseillé

1. Préparer le mini-contexte (template session)
2. Implémenter en petites étapes
3. Valider par commande/test
4. Mettre à jour docs minimales
5. Archiver le bruit

Format de sortie recommandé:
- **Changements**
- **Validation**
- **Docs mises à jour**
- **Next step**

---

## 6) Anti-patterns à éviter

- Envoyer un historique complet en prompt à chaque session
- Mélanger objectifs, logs, et décisions dans un seul fichier
- Laisser `ACTIVE_CONTEXT.md` grossir sans nettoyage
- Oublier de lier les archives dans `ARCHIVE_INDEX.md`

---

## 7) Checklist fin de session

- [ ] Code validé
- [ ] Résultat clé consigné
- [ ] Rapport run créé dans `portfolio/<version>/results/` (si run significatif)
- [ ] Entrée ajoutée dans `docs/carnet_de_bord.md`
- [ ] `ACTIVE_CONTEXT.md` à jour (<120 lignes)
- [ ] Détails longs archivés
- [ ] README(s) impactés mis à jour
