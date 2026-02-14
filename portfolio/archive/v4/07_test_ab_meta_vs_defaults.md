# Test A/B : Méta-Optimisation vs Defaults Fixes
**Date** : 12 February 2026 (07:07)
**Seeds par variante** : 5
**Durée** : 33.5 min
**Statut** : ✅ TERMINÉ

---

## Protocole

Pour chaque combo (5 profils méta-optimisés V3) :
- **META** : walk-forward avec les meta-params trouvés par Optuna
- **DEFAULTS** : walk-forward avec defaults fixes (reoptim=3M, window=1Y, bounds=1.0, metric=sharpe, trials=100)
- **CONSERVATIVE** : walk-forward avec defaults conservateurs (reoptim=6M, window=2Y, bounds=0.8, metric=sharpe, trials=100)

Chaque variante est testée avec `run_walk_forward_robust` (5 seeds, médiane) pour éliminer la variance.

## Résultats par combo

| Combo | Variante | Sharpe | Return | DD | PF | Trades | Sharpe std | Consistency |
|-------|----------|--------|--------|-----|-----|--------|------------|-------------|
| XRPUSDT/stochastic_oscillator | META **←** | 0.607 | 19.5% | -5.6% | 2.00 | 31 | 0.244 | 0.55 |
| XRPUSDT/stochastic_oscillator | DEFAULTS | 0.445 | 34.1% | -20.9% | 1.44 | 55 | 0.292 | -0.27 |
| XRPUSDT/stochastic_oscillator | CONSERVATIVE | 0.120 | 3.8% | -20.8% | 1.08 | 67 | 0.397 | -2.37 |
| ETHUSDT/atr_volatility_breakout | META | 0.094 | 2.0% | -8.7% | 1.06 | 39 | 0.201 | -6.07 |
| ETHUSDT/atr_volatility_breakout | DEFAULTS | 0.078 | 1.8% | -24.8% | 1.03 | 53 | 0.195 | -0.57 |
| ETHUSDT/atr_volatility_breakout | CONSERVATIVE **←** | 0.176 | 5.3% | -12.1% | 1.09 | 73 | 0.269 | -0.15 |
| ETHUSDT/donchian_channel | META | 0.310 | 16.7% | -16.0% | 1.16 | 113 | 0.035 | 0.89 |
| ETHUSDT/donchian_channel | DEFAULTS **←** | 0.708 | 38.9% | -9.0% | 1.78 | 49 | 0.184 | 0.76 |
| ETHUSDT/donchian_channel | CONSERVATIVE | -0.090 | -5.9% | -20.8% | 0.94 | 85 | 0.161 | -0.05 |
| BTCUSDT/ema_ribbon | META | -0.170 | -8.1% | -22.4% | 0.91 | 90 | 0.143 | 0.23 |
| BTCUSDT/ema_ribbon | DEFAULTS **←** | -0.084 | -2.4% | -7.8% | 0.92 | 27 | 0.226 | -1.41 |
| BTCUSDT/ema_ribbon | CONSERVATIVE | -0.434 | -19.0% | -23.4% | 0.82 | 108 | 0.145 | 0.62 |
| BTCUSDT/donchian_channel | META | -0.240 | -12.2% | -26.6% | 0.86 | 90 | 0.182 | 0.18 |
| BTCUSDT/donchian_channel | DEFAULTS **←** | -0.036 | -2.3% | -25.9% | 0.97 | 39 | 0.187 | -8.38 |
| BTCUSDT/donchian_channel | CONSERVATIVE | -0.668 | -23.4% | -31.9% | 0.69 | 75 | 0.191 | 0.71 |

## Résumé agrégé

| Métrique | META | DEFAULTS | CONSERVATIVE |
|----------|------|----------|--------------|
| Avg Sharpe | 0.1200 | 0.2222 | -0.1792 |
| Wins (Sharpe) | 1/5 | 3/5 | 1/5 |
| Wins (Composite) | 1/5 | 3/5 | 1/5 |

## Analyse de la variance (robustesse)

| Combo | META std | DEFAULTS std | CONSERVATIVE std |
|-------|---------|-------------|-----------------|
| XRPUSDT/stochastic_oscillator | 0.244 | 0.292 | 0.397 |
| ETHUSDT/atr_volatility_breakout | 0.201 | 0.195 | 0.269 |
| ETHUSDT/donchian_channel | 0.035 | 0.184 | 0.161 |
| BTCUSDT/ema_ribbon | 0.143 | 0.226 | 0.145 |
| BTCUSDT/donchian_channel | 0.182 | 0.187 | 0.191 |

## Verdict

**DEFAULTS font MIEUX que la méta-optimisation**

Les defaults font mieux que la méta-optimisation !
→ **Abandonner** la méta-optimisation. Elle introduit du bruit sans valeur ajoutée.

## Implications

### Si méta-opt gagne
- Garder la boucle externe mais avec multi-seed obligatoire
- Considérer grid search exhaustif (espace petit)
- Augmenter n_seeds à 5-10 pour réduire la variance

### Si defaults gagnent
- Simplifier le pipeline : Diagnostic → WF avec defaults → Portfolio
- Économiser ~80% du temps de compute
- Focus sur : plus de stratégies, portfolio optimization, features

---

*Généré le 12 February 2026*