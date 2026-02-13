# Portfolio V5c-HighRisk

**Date**: 2026-02-13 09:39
**Capital initial**: $100
**Objectif**: gains court terme (1-2 mois)
**Contrainte DD**: >= -30% (sur test OOS)
**Validation**: train calibré, test final = 60 barres

## Résultat retenu

- Profil risque: **HR-75**
- Max position: **75%**
- Return 30j OOS: **12.7%**
- Return 60j OOS: **12.1%**
- Sharpe OOS: **3.93**
- Max DD OOS: **-2.3%**

## Allocations

| Poids | Combo | Sharpe TRAIN | Return TRAIN | DD TRAIN |
|------:|-------|-------------:|-------------:|---------:|
| 40.0% | ETHUSDT/breakout_regime/1d | 1.70 | 0.9% | -0.1% |
| 25.0% | SOLUSDT/mtf_momentum_breakout/1d | 1.01 | 0.6% | -0.5% |
| 12.7% | ETHUSDT/ema_ribbon/1d | 0.86 | 1.4% | -1.1% |
| 11.9% | ETHUSDT/macd_crossover/1d | 1.15 | 2.8% | -1.6% |
| 9.0% | ETHUSDT/trend_multi_factor/1d | 1.19 | 3.5% | -2.3% |
| 1.4% | ETHUSDT/bollinger_breakout/1d | 1.10 | 7.5% | -5.0% |