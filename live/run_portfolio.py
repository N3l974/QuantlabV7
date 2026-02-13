"""Run a portfolio-scoped live service from a deployment config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger


DEFAULT_CONFIG = "config/live/portfolios/v5c-highrisk-paper.json"


def load_portfolio_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Portfolio config not found: {config_path}")
    return json.loads(path.read_text())


def run_from_config(config_path: str):
    cfg = load_portfolio_config(config_path)

    portfolio_id = cfg.get("portfolio_id", "unknown")
    logger.info(f"Starting portfolio service: {portfolio_id}")
    logger.info(f"  Reoptimization policy: {cfg.get('reoptimization_policy', {})}")

    # Multi-combo mode (new PortfolioExecutor)
    if "combos" in cfg:
        from live.portfolio_executor import run_portfolio
        logger.info(f"  Engine: PortfolioExecutor (multi-combo, {len(cfg['combos'])} combos)")
        run_portfolio(config_path)
        return

    # Legacy mono-combo mode (old LiveExecutor)
    from live.executor import run_live

    dry_run = bool(cfg.get("execution", {}).get("dry_run", True))
    paper_capital = float(cfg.get("paper_capital_usd", 0))

    profile_cfg = cfg.get("profile_source", {})
    profiles_file = profile_cfg.get("profiles_file")
    profile_index = int(profile_cfg.get("profile_index", 0))

    runtime = cfg.get("runtime", {})
    interval_seconds = int(runtime.get("interval_seconds", 60))
    settings_path = runtime.get("settings_path", "config/settings.yaml")

    if not profiles_file:
        raise ValueError("Missing profile_source.profiles_file in portfolio config")

    logger.info(f"  Engine: LiveExecutor (legacy mono-combo)")
    logger.info(f"  Mode: {'PAPER' if dry_run else 'LIVE'}")
    logger.info(f"  Paper capital (tracking): ${paper_capital:.2f}")

    run_live(
        profiles_file=profiles_file,
        profile_index=profile_index,
        dry_run=dry_run,
        interval_seconds=interval_seconds,
        settings_path=settings_path,
        portfolio_id=portfolio_id,
        paper_capital=paper_capital,
    )


def main():
    parser = argparse.ArgumentParser(description="Run one portfolio-scoped live service")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Portfolio config JSON path")
    args = parser.parse_args()

    run_from_config(args.config)


if __name__ == "__main__":
    main()
