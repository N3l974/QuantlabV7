"""
Quantlab V7 — Meta-Optimizer for BTC Trading Strategies.
Main entry point with CLI interface.
"""

import sys
import os
from pathlib import Path

import click
import yaml
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.ingestion import (
    ingest_all, ingest_all_symbols,
    load_all_data, load_all_symbols_data,
    load_settings,
)
from engine.meta_optimizer import (
    MetaOptimizerConfig,
    load_meta_config,
    run_meta_optimization,
    run_single_strategy_meta,
    save_profiles,
    load_profiles,
)
from engine.portfolio import (
    build_equal_weight_portfolio,
    build_sharpe_weighted_portfolio,
    build_risk_parity_portfolio,
    build_diversified_portfolio,
    build_all_portfolios,
    save_portfolio,
    save_all_portfolios,
)
from strategies.registry import list_strategies


def setup_logging(level: str = "INFO"):
    """Configure loguru logging."""
    logger.remove()
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level:<7} | {message}")
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/quantlab.log", level="DEBUG", rotation="10 MB")


@click.group()
def cli():
    """Quantlab V7 — Meta-Optimizer for BTC Trading Strategies."""
    setup_logging()


@cli.command()
@click.option("--all-symbols", "-a", is_flag=True, help="Ingest all symbols (multi-asset)")
def ingest(all_symbols):
    """Download/update historical data from Binance."""
    settings = load_settings()
    if all_symbols:
        symbols = settings["data"].get("symbols", [settings["data"]["symbol"]])
        logger.info(f"Starting multi-asset ingestion for {symbols}...")
        results = ingest_all_symbols(settings)
        for symbol, tf_data in results.items():
            total = sum(len(df) for df in tf_data.values())
            logger.info(f"  {symbol}: {len(tf_data)} TFs, {total:,} candles")
    else:
        logger.info(f"Starting data ingestion for {settings['data']['symbol']}...")
        results = ingest_all(settings)
        for tf, df in results.items():
            if len(df) > 0:
                logger.info(f"  {tf}: {len(df)} candles ({df.index.min()} → {df.index.max()})")
            else:
                logger.warning(f"  {tf}: no data")
    logger.info("Ingestion complete.")


@cli.command()
@click.option("--trials", "-n", default=None, type=int, help="Override number of outer trials")
@click.option("--timeout", "-t", default=None, type=float, help="Override timeout in hours")
@click.option("--strategy", "-s", default=None, type=str, help="Run for a single strategy only")
def optimize(trials, timeout, strategy):
    """Run the full meta-optimization (outer loop)."""
    logger.info("Loading data...")
    settings = load_settings()

    # Try to load multi-symbol data
    data_by_symbol = load_all_symbols_data(settings)
    symbols_with_data = [s for s, d in data_by_symbol.items() if d]

    # Fallback: default symbol data for backward compat
    default_symbol = settings["data"]["symbol"]
    data = data_by_symbol.get(default_symbol, load_all_data(settings))

    if not data:
        logger.error("No data found. Run 'python main.py ingest' first.")
        return

    logger.info(f"Symbols with data: {symbols_with_data}")
    logger.info(f"Default data TFs: {list(data.keys())}")

    config = load_meta_config(data, data_by_symbol=data_by_symbol)

    if trials:
        config.n_outer_trials = trials
    if timeout:
        config.timeout_hours = timeout

    if strategy:
        if strategy not in list_strategies():
            logger.error(f"Unknown strategy '{strategy}'. Available: {list_strategies()}")
            return
        logger.info(f"Running meta-optimization for strategy: {strategy}")
        profiles = run_single_strategy_meta(strategy, config, n_trials=trials)
    else:
        profiles = run_meta_optimization(config)

    if profiles:
        filepath = save_profiles(profiles)
        logger.info(f"Results saved to {filepath}")

        # Print top 10
        logger.info("\n--- TOP 10 META-PROFILES ---")
        for i, p in enumerate(profiles[:10]):
            logger.info(f"  #{i+1}: {p.summary()}")
    else:
        logger.warning("No valid profiles found.")


@cli.command()
@click.argument("profiles_file")
@click.option("--top-n", "-n", default=5, type=int, help="Number of strategies in portfolio")
@click.option("--method", "-m", default="diversified",
              type=click.Choice(["equal", "sharpe", "risk_parity", "diversified"]),
              help="Portfolio construction method")
def portfolio(profiles_file, top_n, method):
    """Build a meta-portfolio from saved profiles."""
    logger.info(f"Loading profiles from {profiles_file}...")
    profiles = load_profiles(profiles_file)
    logger.info(f"Loaded {len(profiles)} profiles")

    builders = {
        "equal": build_equal_weight_portfolio,
        "sharpe": build_sharpe_weighted_portfolio,
        "risk_parity": build_risk_parity_portfolio,
        "diversified": build_diversified_portfolio,
    }

    builder = builders[method]
    ptf = builder(profiles, top_n=top_n, name=method.title())

    logger.info(f"\n{ptf.summary()}")
    filepath = save_portfolio(ptf)
    logger.info(f"Portfolio saved to {filepath}")


@cli.command()
def strategies():
    """List all available strategies."""
    from strategies.registry import STRATEGY_REGISTRY
    logger.info("Available strategies:")
    for name, strat in STRATEGY_REGISTRY.items():
        logger.info(f"  - {name}: {strat.name} ({strat.strategy_type})")


@cli.command()
def status():
    """Show project status and data availability."""
    settings = load_settings()
    logger.info(f"Project: {settings['project']['name']} v{settings['project']['version']}")
    logger.info(f"Symbol: {settings['data']['symbol']}")
    logger.info(f"Exchange: {settings['data']['exchange']} ({settings['data']['market_type']})")

    # Check data
    data = load_all_data(settings)
    if data:
        logger.info("Data status:")
        for tf, df in data.items():
            logger.info(f"  {tf}: {len(df)} candles ({df.index.min()} → {df.index.max()})")
    else:
        logger.warning("No data found. Run 'python main.py ingest' first.")

    # Check results
    results_dir = Path("results")
    if results_dir.exists():
        files = list(results_dir.glob("*.json"))
        logger.info(f"Results: {len(files)} files in results/")
        for f in files:
            logger.info(f"  - {f.name}")
    else:
        logger.info("No results yet.")


if __name__ == "__main__":
    cli()
