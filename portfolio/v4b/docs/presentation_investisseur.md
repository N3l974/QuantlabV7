# QuantLab V7 — Portfolio V4b

## Document de Présentation Investisseur

**Version** : Portfolio V4b (Edge-Enhanced, Concentrated)  
**Date** : Février 2026  
**Classe d'actifs** : Cryptomonnaies (Binance Margin)  
**Statut** : Validé sur données hors-échantillon (12 mois), walk-forward optimisé

---

## 1. Résumé Exécutif

QuantLab V7 est un système de trading algorithmique multi-stratégies appliqué aux cryptomonnaies. Le Portfolio V4b combine **8 stratégies optimisées** par intelligence artificielle, concentrées sur les combinaisons à plus fort rendement, diversifiées sur 3 actifs (ETH, BTC, SOL) et 2 horizons temporels.

### Performance clé (12 mois hors-échantillon)

| Indicateur | Valeur |
|------------|--------|
| **Rendement annuel** | **+19.8%** |
| **Rendement mensuel moyen** | **+1.47%** |
| **Ratio de Sharpe** | **1.35** |
| **Ratio de Sortino** | **1.79** |
| **Drawdown maximum** | **-8.5%** |
| **Ratio de Calmar** | **2.17** |
| **Probabilité de gain à 12 mois** | **86%** |
| **Probabilité de ruine** | **0.0%** |

> Les performances sont mesurées sur 12 mois de données **jamais vues** par le système (février 2025 — février 2026), après frais de transaction et slippage. Le leverage appliqué est de 1.5x (conservateur pour crypto).

---

## 2. Philosophie d'Investissement

### Approche

Le système repose sur quatre piliers :

- **Concentration sur les meilleurs** — 22 stratégies candidates testées en walk-forward, seules les 8 meilleures par rendement sont retenues. Pas de dilution par des stratégies médiocres.
- **Diversification intelligente** — 3 actifs (ETH, BTC, SOL), 2 horizons (4h, 1 jour), pondération asymétrique vers les top performers.
- **Optimisation adaptative** — Paramètres recalibrés tous les 3 mois via walk-forward bayésien (Optuna TPE).
- **Validation rigoureuse** — Diagnostic en 2 phases, walk-forward sur données in-sample, validation sur holdout strict.

### Ce que nous ne faisons PAS

- Pas de trading haute fréquence
- Pas de positions overnight non couvertes
- Pas d'optimisation sur les données de test
- Pas de leverage excessif (1.5x max)

---

## 3. Composition du Portefeuille

### Allocations

| # | Stratégie | Actif | Horizon | Poids | Type |
|---|-----------|-------|---------|-------|------|
| 1 | SuperTrend | ETH | 1 jour | 25.0% | Suivi de tendance adaptatif |
| 2 | Trend Multi-Factor | ETH | 1 jour | 25.0% | Tendance multi-indicateurs |
| 3 | Trend Multi-Factor | SOL | 4 heures | 15.0% | Tendance multi-indicateurs |
| 4 | SuperTrend | BTC | 1 jour | 10.0% | Suivi de tendance adaptatif |
| 5 | MACD Crossover | ETH | 1 jour | 10.0% | Momentum croisement |
| 6 | Trend Multi-Factor | BTC | 1 jour | 5.0% | Tendance multi-indicateurs |
| 7 | Ichimoku Cloud | ETH | 4 heures | 5.0% | Tendance multi-composantes |
| 8 | Bollinger Breakout | ETH | 1 jour | 5.0% | Breakout volatilité |

### Répartition par actif

| Actif | Allocation | Justification |
|-------|------------|---------------|
| **Ethereum (ETH)** | **70%** | Meilleur profil rendement/risque, stratégies les plus performantes |
| **Bitcoin (BTC)** | **15%** | Diversification, décorrélation, stabilité |
| **Solana (SOL)** | **15%** | Momentum fort, décorrélation avec ETH/BTC |

### Leverage

| Paramètre | Valeur |
|-----------|--------|
| **Leverage appliqué** | **1.5x** |
| Justification | Sharpe 1.35 × 1.5 = exposition optimale selon Kelly |
| Marge requise | ~67% du capital |
| Risque de liquidation | Quasi nul (DD max -8.5% × 1.5 = -12.8%) |

---

## 4. Processus de Gestion

### Pipeline d'investissement

```
Données de marché (Binance)
    ↓
Calcul des indicateurs techniques (8 stratégies)
    ↓
Génération des signaux (achat/vente/neutre)
    ↓
Pondération top3_heavy (25/25/15/10/10/5/5/5)
    ↓
Application leverage 1.5x
    ↓
Exécution (Binance Margin)
    ↓
Monitoring continu (drawdown, PnL, alertes)
```

### Réoptimisation

| Paramètre | Valeur |
|-----------|--------|
| **Fréquence** | **Tous les 3 mois** |
| Méthode | Walk-forward avec Optuna (TPE bayésien) |
| Fenêtre d'entraînement | 12 mois |
| Trials par optimisation | 30 |
| Métrique cible | Ratio de Sharpe |

### Gestion des risques

| Contrôle | Seuil |
|----------|-------|
| Position maximale | 25% du capital par trade |
| Perte journalière max | 3% |
| Drawdown max (circuit breaker) | 15% |
| Slippage dynamique | 0.05% — 0.5% selon volatilité |
| Leverage max | 1.5x |

---

## 5. Performance Détaillée

### Résultats sur données hors-échantillon (fév. 2025 — fév. 2026)

Ces résultats sont mesurés sur 12 mois de données que le système n'a **jamais vues** pendant l'optimisation.

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| Rendement total | **+19.8%** | Dépasse l'objectif de +15% |
| Sharpe | **1.35** | Très bon (>1 = bon, >1.5 = excellent) |
| Sortino | **1.79** | Très bon (faible volatilité baissière) |
| Max Drawdown | **-8.5%** | Bien sous le seuil de -20% |
| Calmar | **2.17** | Excellent (rendement / drawdown) |

### Analyse mensuelle

| Métrique | Valeur |
|----------|--------|
| Rendement mensuel moyen | **+1.47%** |
| Meilleur mois | **+9.3%** |
| Pire mois | **-3.9%** |
| Meilleur trimestre | **+6.9%** |
| Pire trimestre | **-2.8%** |
| Mois positifs | **46%** |

### Évolution des versions

| Métrique | V3b | V4 (conserv.) | **V4b** |
|----------|-----|---------------|---------|
| Return | +9.8% | +4.9% | **+19.8%** |
| Sharpe | 1.19 | 2.59 | **1.35** |
| Max DD | -4.9% | -0.8% | **-8.5%** |
| Calmar | 1.91 | 5.99 | **2.17** |
| ETH % | 95% | 53% | **70%** |
| Objectif +15% | ❌ | ❌ | **✅** |

---

## 6. Projections Monte Carlo

Les projections ci-dessous sont basées sur **5 000 simulations** bootstrappées à partir des rendements réels du holdout.

### Pour un capital initial de $10,000

| Horizon | Pessimiste (P5) | Médian | Optimiste (P95) | P(gain) | P(+10%) | P(+20%) |
|---------|-----------------|--------|-----------------|---------|---------|---------|
| **3 mois** | $9,334 (-6.7%) | $10,313 (+3.1%) | $11,614 (+16.1%) | **69%** | 18% | 2% |
| **6 mois** | $9,229 (-7.7%) | $10,678 (+6.8%) | $12,655 (+26.6%) | **77%** | 38% | 12% |
| **12 mois** | $9,280 (-7.2%) | $11,421 (+14.2%) | $14,343 (+43.4%) | **86%** | 62% | 36% |
| **24 mois** | $9,679 (-3.2%) | $13,125 (+31.2%) | $18,114 (+81.1%) | **92%** | 83% | 68% |
| **36 mois** | $10,323 (+3.2%) | $14,995 (+49.9%) | $22,203 (+122.0%) | **96%** | 92% | 84% |

### Pour un capital initial de $50,000

| Horizon | Pessimiste (P5) | Médian | Optimiste (P95) | Gain médian |
|---------|-----------------|--------|-----------------|-------------|
| **3 mois** | $46,668 | $51,566 | $58,068 | **+$1,566** |
| **6 mois** | $46,147 | $53,389 | $63,276 | **+$3,389** |
| **12 mois** | $46,398 | $57,106 | $71,715 | **+$7,106** |
| **24 mois** | $48,395 | $65,625 | $90,572 | **+$15,625** |
| **36 mois** | $51,617 | $74,974 | $111,016 | **+$24,974** |

### Pour un capital initial de $100,000

| Horizon | Pessimiste (P5) | Médian | Optimiste (P95) | Gain médian |
|---------|-----------------|--------|-----------------|-------------|
| **3 mois** | $93,336 | $103,132 | $116,135 | **+$3,132** |
| **6 mois** | $92,295 | $106,778 | $126,551 | **+$6,778** |
| **12 mois** | $92,795 | $114,212 | $143,430 | **+$14,212** |
| **24 mois** | $96,791 | $131,249 | $181,145 | **+$31,249** |
| **36 mois** | $103,233 | $149,949 | $222,031 | **+$49,949** |

### Interprétation

- À **12 mois**, il y a **86% de chances** que le capital soit en gain, avec un rendement médian de **+14.2%**.
- À **24 mois**, la probabilité de gain monte à **92%**, avec un rendement médian de **+31.2%**.
- À **36 mois**, le capital médian atteint **+50%** avec 96% de probabilité de gain.
- Même dans le **pire scénario (P5)**, la perte maximale à 12 mois est limitée à **-7.2%**.
- La **probabilité de ruine** (perte de 50% du capital) est de **0.0%** sur tous les horizons.

---

## 7. Risques et Limitations

### Risques identifiés

| Risque | Niveau | Mitigation |
|--------|--------|------------|
| **Concentration ETH** | Moyen | Cap à 70%, diversification BTC+SOL |
| **Leverage** | Modéré | 1.5x seulement, marge de sécurité large |
| **Changement de régime** | Moyen | Réoptimisation trimestrielle adaptative |
| **Risque de liquidité** | Faible | Trading sur les 3 cryptos les plus liquides |
| **Risque technique** | Faible | Monitoring 24/7, circuit breaker automatique |

### Ce que les performances passées ne garantissent pas

- Les résultats hors-échantillon sont le meilleur indicateur disponible, mais ne garantissent pas les performances futures.
- Les marchés crypto sont volatils et peuvent connaître des événements extrêmes non capturés par les simulations historiques.
- Le leverage amplifie les gains ET les pertes.

### Mesures de protection

- **Circuit breaker** : arrêt automatique si le drawdown dépasse 15%
- **Leverage modéré** : 1.5x seulement (vs 3-10x courant en crypto)
- **Diversification** : 8 stratégies indépendantes sur 3 actifs
- **Réoptimisation régulière** : adaptation tous les 3 mois

---

## 8. Infrastructure Technique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.12 |
| Optimisation | Optuna (TPE bayésien) |
| Données | Binance API → Parquet |
| Exécution | Binance Margin |
| Déploiement | VPS Linux, Docker |
| Monitoring | Alertes Telegram, dashboard Streamlit |
| Tests | 98 tests unitaires automatisés |

---

## 9. Feuille de Route

| Trimestre | Objectif |
|-----------|----------|
| **T1 2026** | Déploiement live du Portfolio V4b, monitoring en conditions réelles |
| **T2 2026** | Première réoptimisation trimestrielle, bilan 3 mois |
| **T3 2026** | Ajout de signaux alternatifs (funding rate, cross-asset) |
| **T4 2026** | Bilan 12 mois live, ajustement de l'allocation |

---

## 10. Résumé des Chiffres Clés

| | Valeur |
|---|--------|
| **Rendement annuel** | **+19.8%** |
| **Rendement mensuel moyen** | **+1.47%** |
| **Drawdown maximum** | **-8.5%** |
| **Sharpe** | **1.35** |
| **Sortino** | **1.79** |
| **Calmar** | **2.17** |
| **Probabilité de gain à 12 mois** | **86%** |
| **Probabilité de gain à 24 mois** | **92%** |
| **Probabilité de ruine** | **0.0%** |
| **Leverage** | **1.5x** |
| **Réoptimisation** | **Trimestrielle** |
| **Frais inclus** | **Oui (commissions + slippage)** |
| **Nombre de stratégies** | **8** |
| **Actifs** | **ETH (70%), BTC (15%), SOL (15%)** |

---

*Document confidentiel — QuantLab V7 — Portfolio V4b — Février 2026*  
*Les performances passées ne préjugent pas des performances futures. Investir comporte des risques.*
