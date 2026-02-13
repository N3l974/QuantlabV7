"""
Meta-Portfolio Builder.
Combines multiple meta-optimal profiles into a diversified portfolio.
Includes realistic allocation constraints and portfolio-level risk limits.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from engine.meta_optimizer import MetaProfile


@dataclass
class PortfolioConstraints:
    """Constraints for portfolio construction."""
    max_allocation_per_strategy: float = 0.30   # Max 30% in one strategy type
    max_allocation_per_timeframe: float = 0.40  # Max 40% in one timeframe
    min_unique_strategies: int = 3              # At least 3 different strategies
    min_unique_timeframes: int = 2              # At least 2 different timeframes
    max_correlation_threshold: float = 0.7      # Skip if too correlated (future)
    min_sharpe: float = 0.5                     # Minimum Sharpe to include
    min_oos_periods: int = 4                    # Minimum OOS periods
    max_drawdown_threshold: float = -0.50       # Reject profiles with DD > 50%


@dataclass
class MetaPortfolio:
    """A portfolio of meta-optimal profiles with allocation weights."""
    profiles: list[MetaProfile]
    weights: list[float]
    name: str = "MetaPortfolio"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    constraints_applied: Optional[PortfolioConstraints] = None

    @property
    def n_strategies(self) -> int:
        return len(self.profiles)

    @property
    def unique_strategies(self) -> set:
        return set(p.strategy_name for p in self.profiles)

    @property
    def unique_timeframes(self) -> set:
        return set(p.timeframe for p in self.profiles)

    @property
    def portfolio_sharpe(self) -> float:
        """Weighted average Sharpe of the portfolio."""
        return sum(
            w * p.metrics.get("sharpe", 0)
            for w, p in zip(self.weights, self.profiles)
        )

    @property
    def portfolio_max_dd(self) -> float:
        """Worst-case max drawdown (weighted)."""
        return sum(
            w * p.metrics.get("max_drawdown", 0)
            for w, p in zip(self.weights, self.profiles)
        )

    def summary(self) -> str:
        lines = [
            f"{'='*70}",
            f"  {self.name} ({self.n_strategies} slots)",
            f"  {len(self.unique_strategies)} unique strategies, "
            f"{len(self.unique_timeframes)} unique timeframes",
            f"  Portfolio Sharpe: {self.portfolio_sharpe:.2f}  |  "
            f"Weighted MaxDD: {self.portfolio_max_dd:.1%}",
            f"{'='*70}",
            "",
        ]
        for i, (p, w) in enumerate(zip(self.profiles, self.weights)):
            m = p.metrics
            lines.append(
                f"  [{i+1}] {w:>5.1%}  {p.strategy_name:<25} {p.timeframe:>3}  "
                f"Sharpe={m.get('sharpe', 0):>5.2f}  "
                f"DD={m.get('max_drawdown', 0):>6.1%}  "
                f"Ret={m.get('total_return', 0):>8.1%}  "
                f"WR={m.get('win_rate', 0):>5.1%}  "
                f"reoptim={p.reoptim_frequency} win={p.training_window}"
            )
        lines.append(f"{'='*70}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "n_strategies": self.n_strategies,
            "unique_strategies": list(self.unique_strategies),
            "unique_timeframes": list(self.unique_timeframes),
            "portfolio_sharpe": self.portfolio_sharpe,
            "portfolio_max_dd": self.portfolio_max_dd,
            "allocations": [
                {"weight": w, "profile": p.to_dict()}
                for p, w in zip(self.profiles, self.weights)
            ],
        }


def _filter_viable_profiles(
    profiles: list[MetaProfile],
    constraints: PortfolioConstraints,
) -> list[MetaProfile]:
    """Filter profiles that meet minimum quality thresholds."""
    viable = []
    for p in profiles:
        m = p.metrics
        if m.get("sharpe", 0) < constraints.min_sharpe:
            continue
        if p.n_oos_periods < constraints.min_oos_periods:
            continue
        if m.get("max_drawdown", 0) < constraints.max_drawdown_threshold:
            continue
        viable.append(p)
    logger.info(f"Filtered {len(profiles)} → {len(viable)} viable profiles "
                f"(Sharpe≥{constraints.min_sharpe}, OOS≥{constraints.min_oos_periods}, "
                f"DD>{constraints.max_drawdown_threshold:.0%})")
    return viable


def build_diversified_portfolio(
    profiles: list[MetaProfile],
    top_n: int = 8,
    constraints: Optional[PortfolioConstraints] = None,
    name: str = "Diversified",
) -> MetaPortfolio:
    """
    Build a diversified portfolio with realistic constraints.
    Ensures variety across strategies and timeframes, respects allocation limits.
    """
    if constraints is None:
        constraints = PortfolioConstraints()

    # Step 1: Filter viable profiles
    viable = _filter_viable_profiles(profiles, constraints)
    if not viable:
        logger.warning("No viable profiles after filtering, using top profiles as-is")
        viable = profiles[:top_n]

    # Step 2: Select with diversification — max 1 per strategy/timeframe combo
    selected = []
    combo_seen = set()
    strat_count = {}
    tf_count = {}

    for p in viable:
        combo = (p.strategy_name, p.timeframe)
        if combo in combo_seen:
            continue

        selected.append(p)
        combo_seen.add(combo)
        strat_count[p.strategy_name] = strat_count.get(p.strategy_name, 0) + 1
        tf_count[p.timeframe] = tf_count.get(p.timeframe, 0) + 1

        if len(selected) >= top_n:
            break

    if len(selected) < constraints.min_unique_strategies:
        logger.warning(f"Only {len(selected)} profiles selected, "
                       f"minimum {constraints.min_unique_strategies} required")

    # Step 3: Compute weights — risk-parity with allocation caps
    inv_dd = np.array([
        1.0 / max(abs(p.metrics.get("max_drawdown", -0.01)), 0.01)
        for p in selected
    ])
    raw_weights = inv_dd / inv_dd.sum()

    # Apply per-strategy cap
    unique_strats = set(p.strategy_name for p in selected)
    for s in unique_strats:
        strat_indices = [i for i, p in enumerate(selected) if p.strategy_name == s]
        strat_total = sum(raw_weights[i] for i in strat_indices)
        if strat_total > constraints.max_allocation_per_strategy:
            scale = constraints.max_allocation_per_strategy / strat_total
            for i in strat_indices:
                raw_weights[i] *= scale

    # Apply per-timeframe cap
    unique_tfs = set(p.timeframe for p in selected)
    for tf in unique_tfs:
        tf_indices = [i for i, p in enumerate(selected) if p.timeframe == tf]
        tf_total = sum(raw_weights[i] for i in tf_indices)
        if tf_total > constraints.max_allocation_per_timeframe:
            scale = constraints.max_allocation_per_timeframe / tf_total
            for i in tf_indices:
                raw_weights[i] *= scale

    # Renormalize
    weights = (raw_weights / raw_weights.sum()).tolist()

    portfolio = MetaPortfolio(
        profiles=selected,
        weights=weights,
        name=name,
        constraints_applied=constraints,
    )

    logger.info(
        f"Built {name} portfolio: {len(selected)} slots, "
        f"{len(set(p.strategy_name for p in selected))} strategies, "
        f"{len(set(p.timeframe for p in selected))} timeframes, "
        f"Sharpe={portfolio.portfolio_sharpe:.2f}"
    )
    return portfolio


def build_equal_weight_portfolio(
    profiles: list[MetaProfile],
    top_n: int = 5,
    constraints: Optional[PortfolioConstraints] = None,
    name: str = "EqualWeight",
) -> MetaPortfolio:
    """Build a portfolio with equal weights from viable profiles."""
    if constraints is None:
        constraints = PortfolioConstraints()

    viable = _filter_viable_profiles(profiles, constraints)
    if not viable:
        viable = profiles[:top_n]

    # Deduplicate by strategy/timeframe combo
    selected = []
    seen = set()
    for p in viable:
        combo = (p.strategy_name, p.timeframe)
        if combo not in seen:
            selected.append(p)
            seen.add(combo)
        if len(selected) >= top_n:
            break

    n = len(selected)
    weights = [1.0 / n] * n

    return MetaPortfolio(profiles=selected, weights=weights, name=name,
                         constraints_applied=constraints)


def build_sharpe_weighted_portfolio(
    profiles: list[MetaProfile],
    top_n: int = 5,
    constraints: Optional[PortfolioConstraints] = None,
    name: str = "SharpeWeighted",
) -> MetaPortfolio:
    """Build a portfolio weighted by Sharpe ratio."""
    if constraints is None:
        constraints = PortfolioConstraints()

    viable = _filter_viable_profiles(profiles, constraints)
    if not viable:
        viable = profiles[:top_n]

    selected = []
    seen = set()
    for p in viable:
        combo = (p.strategy_name, p.timeframe)
        if combo not in seen:
            selected.append(p)
            seen.add(combo)
        if len(selected) >= top_n:
            break

    sharpes = np.array([max(p.metrics.get("sharpe", 0), 0.01) for p in selected])
    weights = (sharpes / sharpes.sum()).tolist()

    return MetaPortfolio(profiles=selected, weights=weights, name=name,
                         constraints_applied=constraints)


def build_risk_parity_portfolio(
    profiles: list[MetaProfile],
    top_n: int = 5,
    constraints: Optional[PortfolioConstraints] = None,
    name: str = "RiskParity",
) -> MetaPortfolio:
    """Build a risk-parity portfolio (weight inversely proportional to DD)."""
    if constraints is None:
        constraints = PortfolioConstraints()

    viable = _filter_viable_profiles(profiles, constraints)
    if not viable:
        viable = profiles[:top_n]

    selected = []
    seen = set()
    for p in viable:
        combo = (p.strategy_name, p.timeframe)
        if combo not in seen:
            selected.append(p)
            seen.add(combo)
        if len(selected) >= top_n:
            break

    inv_dd = np.array([
        1.0 / max(abs(p.metrics.get("max_drawdown", -0.01)), 0.01)
        for p in selected
    ])
    weights = (inv_dd / inv_dd.sum()).tolist()

    return MetaPortfolio(profiles=selected, weights=weights, name=name,
                         constraints_applied=constraints)


def build_all_portfolios(
    profiles: list[MetaProfile],
    constraints: Optional[PortfolioConstraints] = None,
) -> dict[str, MetaPortfolio]:
    """Build all portfolio types and return as dict."""
    if constraints is None:
        constraints = PortfolioConstraints()

    portfolios = {
        "diversified": build_diversified_portfolio(profiles, constraints=constraints),
        "equal_weight": build_equal_weight_portfolio(profiles, constraints=constraints),
        "sharpe_weighted": build_sharpe_weighted_portfolio(profiles, constraints=constraints),
        "risk_parity": build_risk_parity_portfolio(profiles, constraints=constraints),
    }
    return portfolios


def save_portfolio(portfolio: MetaPortfolio, output_dir: str = "results") -> str:
    """Save portfolio to JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"portfolio_{portfolio.name}_{timestamp}.json")

    with open(filepath, "w") as f:
        json.dump(portfolio.to_dict(), f, indent=2, default=str)

    logger.info(f"Saved portfolio to {filepath}")
    return filepath


def save_all_portfolios(
    portfolios: dict[str, MetaPortfolio],
    output_dir: str = "results",
) -> list[str]:
    """Save all portfolios and print comparison."""
    paths = []

    logger.info("\n" + "=" * 70)
    logger.info("  PORTFOLIO COMPARISON")
    logger.info("=" * 70)
    logger.info(f"{'Name':<20} {'Slots':>5} {'Strats':>6} {'TFs':>4} "
                f"{'Sharpe':>8} {'MaxDD':>8}")
    logger.info("-" * 60)

    for name, pf in portfolios.items():
        logger.info(
            f"{name:<20} {pf.n_strategies:>5} "
            f"{len(pf.unique_strategies):>6} {len(pf.unique_timeframes):>4} "
            f"{pf.portfolio_sharpe:>8.2f} {pf.portfolio_max_dd:>8.1%}"
        )
        paths.append(save_portfolio(pf, output_dir))

    logger.info("=" * 70)

    # Print best portfolio details
    best = max(portfolios.values(), key=lambda p: p.portfolio_sharpe)
    logger.info(f"\n🏆 Best portfolio: {best.name}")
    logger.info(best.summary())

    return paths
