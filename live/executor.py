"""
Live Executor — Runs validated meta-profiles in real-time.
Connects to Binance Margin and executes signals.
"""

import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion import load_settings, get_binance_client, fetch_klines, TIMEFRAME_MAP
from engine.backtester import backtest_strategy
from engine.meta_optimizer import MetaProfile, load_profiles
from engine.walk_forward import WalkForwardConfig, _optimize_on_window
from strategies.registry import get_strategy


HEARTBEAT_PATH = Path("logs/heartbeat.txt")


def _write_heartbeat() -> None:
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_PATH.write_text(datetime.utcnow().isoformat())


class LiveExecutor:
    """
    Executes a meta-profile in live mode.
    Periodically re-optimizes parameters and generates trading signals.
    """

    def __init__(self, profile: MetaProfile, settings: dict, dry_run: bool = True):
        self.profile = profile
        self.settings = settings
        self.dry_run = dry_run
        self.strategy = get_strategy(profile.strategy_name)
        self.client = get_binance_client(settings)
        self.current_params: dict = {}
        self.current_position: int = 0
        self.last_optimization: Optional[datetime] = None

        logger.info(f"LiveExecutor initialized: {profile.strategy_name}/{profile.timeframe}")
        logger.info(f"  Reoptim: {profile.reoptim_frequency}, Window: {profile.training_window}")
        logger.info(f"  Dry run: {dry_run}")

    def fetch_recent_data(self, n_candles: int = 1000) -> pd.DataFrame:
        """Fetch recent candles from Binance."""
        interval = TIMEFRAME_MAP.get(self.profile.timeframe)
        if not interval:
            raise ValueError(f"Unknown timeframe: {self.profile.timeframe}")

        klines = self.client.get_klines(
            symbol=self.settings["data"]["symbol"],
            interval=interval,
            limit=n_candles,
        )

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)
        df = df.set_index("open_time")
        df.index.name = "timestamp"
        df = df.drop(columns=["ignore"])

        return df

    def should_reoptimize(self) -> bool:
        """Check if it's time to re-optimize parameters."""
        if self.last_optimization is None:
            return True

        freq = self.profile.reoptim_frequency
        freq_days = {
            "1W": 7, "2W": 14, "1M": 30, "2M": 60,
            "3M": 90, "6M": 180,
        }
        days = freq_days.get(freq, 30)
        elapsed = (datetime.utcnow() - self.last_optimization).days
        return elapsed >= days

    def reoptimize(self):
        """Re-optimize strategy parameters on recent data."""
        logger.info("Re-optimizing parameters...")
        data = self.fetch_recent_data(n_candles=1000)

        wf_config = WalkForwardConfig(
            strategy=self.strategy,
            data=data,
            timeframe=self.profile.timeframe,
            reoptim_frequency=self.profile.reoptim_frequency,
            training_window=self.profile.training_window,
            param_bounds_scale=self.profile.param_bounds_scale,
            optim_metric=self.profile.optim_metric,
            n_optim_trials=self.profile.n_optim_trials,
            commission=self.settings["engine"]["commission_rate"],
            slippage=self.settings["engine"]["slippage_rate"],
        )

        self.current_params = _optimize_on_window(self.strategy, data, wf_config)
        self.last_optimization = datetime.utcnow()
        logger.info(f"New params: {self.current_params}")

    def get_current_signal(self) -> int:
        """Get the latest trading signal."""
        data = self.fetch_recent_data(n_candles=200)
        signals = self.strategy.generate_signals(data, self.current_params)
        return int(signals[-1])

    def execute_signal(self, signal: int):
        """Execute a trading signal (or log it in dry run mode)."""
        if signal == self.current_position:
            return

        symbol = self.settings["data"]["symbol"]

        if self.dry_run:
            logger.info(f"[DRY RUN] Signal: {signal} (was {self.current_position}) on {symbol}")
        else:
            logger.info(f"[LIVE] Executing signal: {signal} on {symbol}")
            # TODO: Implement actual Binance margin order execution
            # self._place_order(signal)

        self.current_position = signal

    def run_loop(self, interval_seconds: int = 60):
        """
        Main execution loop.
        Checks for re-optimization, generates signals, and executes.
        """
        logger.info("Starting live execution loop...")

        while True:
            try:
                # Check if re-optimization needed
                if self.should_reoptimize():
                    self.reoptimize()

                # Get signal and execute
                signal = self.get_current_signal()
                self.execute_signal(signal)

                logger.debug(
                    f"Signal={signal}, Position={self.current_position}, "
                    f"Params={self.current_params}"
                )
                _write_heartbeat()

            except KeyboardInterrupt:
                logger.info("Execution stopped by user.")
                break
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                _write_heartbeat()

            time.sleep(interval_seconds)


def run_live(
    profiles_file: str,
    profile_index: int = 0,
    dry_run: bool = True,
    interval_seconds: int = 60,
    settings_path: str = "config/settings.yaml",
):
    """
    Start live execution for a specific profile.
    """
    profiles = load_profiles(profiles_file)
    if profile_index >= len(profiles):
        logger.error(f"Profile index {profile_index} out of range (max {len(profiles)-1})")
        return

    profile = profiles[profile_index]
    settings = load_settings(settings_path)

    executor = LiveExecutor(profile, settings, dry_run=dry_run)
    executor.run_loop(interval_seconds=interval_seconds)
