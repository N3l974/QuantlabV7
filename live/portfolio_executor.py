"""
Portfolio Executor — Multi-combo live execution with Binance Margin constraints.

Runs N combos in parallel, aggregates signals per symbol into net positions,
and tracks paper equity using mark-to-market on net positions.

Binance Cross Margin constraints:
- One net position per symbol (can't be long+short same symbol simultaneously)
- Multiple symbols can have independent positions
- Equity is shared across all positions
"""

import time
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingestion import load_settings, get_binance_client, TIMEFRAME_MAP
from engine.walk_forward import WalkForwardConfig, _optimize_on_window
from live.monitor import LiveMonitor
from strategies.registry import get_strategy


HEARTBEAT_PATH = Path("logs/heartbeat.txt")


def _write_heartbeat() -> None:
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_PATH.write_text(datetime.utcnow().isoformat())


@dataclass
class ComboConfig:
    """Configuration for a single combo within the portfolio."""
    strategy_name: str
    symbol: str
    timeframe: str
    weight: float
    reoptim_frequency: str = "1M"
    training_window: str = "1Y"
    param_bounds_scale: float = 1.0
    optim_metric: str = "sharpe"
    n_optim_trials: int = 30


@dataclass
class RiskProfile:
    """Portfolio-level risk constraints (mirrors backtest RiskConfig)."""
    max_position_pct: float = 0.75
    max_drawdown_pct: float = 0.30


@dataclass
class ComboState:
    """Runtime state for a single combo."""
    config: ComboConfig
    strategy: object  # BaseStrategy instance
    current_params: dict = field(default_factory=dict)
    last_signal: int = 0
    last_optimization: Optional[datetime] = None


class PortfolioExecutor:
    """
    Executes a multi-combo portfolio in live mode.

    Each combo generates signals independently. Signals are aggregated
    per symbol using Markowitz weights to produce net positions,
    matching Binance Cross Margin behavior (1 net position per symbol).
    """

    def __init__(
        self,
        combos: list[ComboConfig],
        risk_profile: RiskProfile,
        settings: dict,
        dry_run: bool = True,
        portfolio_id: str = "paper-service",
        paper_capital: float = 1000.0,
    ):
        self.combos = combos
        self.risk_profile = risk_profile
        self.settings = settings
        self.dry_run = dry_run
        self.portfolio_id = portfolio_id

        self.client = get_binance_client(settings)
        self.monitor = LiveMonitor(log_dir=f"logs/{portfolio_id}")

        # Paper tracking
        self.paper_equity: float = paper_capital
        self.paper_start_equity: float = paper_capital

        # Per-symbol net position tracking (for mark-to-market)
        # net_position is a float in [-1, 1] representing fraction of equity
        self.net_positions: dict[str, float] = {}
        self.last_prices: dict[str, float] = {}

        # Initialize combo states
        self.combo_states: list[ComboState] = []
        for combo in combos:
            strategy = get_strategy(combo.strategy_name)
            self.combo_states.append(ComboState(config=combo, strategy=strategy))

        # Identify unique symbols
        self.symbols = sorted(set(c.symbol for c in combos))

        # Validate weights sum
        total_weight = sum(c.weight for c in combos)
        if abs(total_weight - 1.0) > 0.05:
            logger.warning(f"Combo weights sum to {total_weight:.3f}, expected ~1.0")

        logger.info(f"PortfolioExecutor initialized: {len(combos)} combos, {len(self.symbols)} symbols")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Risk: max_position={risk_profile.max_position_pct*100:.0f}%, "
                     f"max_dd={risk_profile.max_drawdown_pct*100:.0f}%")
        logger.info(f"  Dry run: {dry_run}, Paper capital: ${paper_capital:.2f}")
        for cs in self.combo_states:
            c = cs.config
            logger.info(f"  [{c.weight*100:.1f}%] {c.strategy_name}/{c.symbol}/{c.timeframe}")

    # ── Data fetching ──

    def fetch_data(self, symbol: str, timeframe: str, n_candles: int = 1000) -> pd.DataFrame:
        """Fetch recent candles from Binance for a specific symbol/timeframe."""
        interval = TIMEFRAME_MAP.get(timeframe)
        if not interval:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        klines = self.client.get_klines(
            symbol=symbol,
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

    # ── Re-optimization ──

    def should_reoptimize(self, state: ComboState) -> bool:
        """Check if a combo needs re-optimization."""
        if state.last_optimization is None:
            return True
        freq_days = {
            "1W": 7, "2W": 14, "1M": 30, "2M": 60,
            "3M": 90, "6M": 180,
        }
        days = freq_days.get(state.config.reoptim_frequency, 30)
        elapsed = (datetime.utcnow() - state.last_optimization).days
        return elapsed >= days

    def reoptimize_combo(self, state: ComboState):
        """Re-optimize a single combo's parameters."""
        c = state.config
        logger.info(f"Re-optimizing {c.strategy_name}/{c.symbol}/{c.timeframe}...")

        data = self.fetch_data(c.symbol, c.timeframe, n_candles=1000)

        wf_config = WalkForwardConfig(
            strategy=state.strategy,
            data=data,
            timeframe=c.timeframe,
            reoptim_frequency=c.reoptim_frequency,
            training_window=c.training_window,
            param_bounds_scale=c.param_bounds_scale,
            optim_metric=c.optim_metric,
            n_optim_trials=c.n_optim_trials,
            commission=self.settings["engine"]["commission_rate"],
            slippage=self.settings["engine"]["slippage_rate"],
        )

        state.current_params = _optimize_on_window(state.strategy, data, wf_config)
        state.last_optimization = datetime.utcnow()
        logger.info(f"  New params for {c.strategy_name}: {state.current_params}")

    # ── Signal generation ──

    def get_combo_signal(self, state: ComboState) -> tuple[int, float]:
        """Get the latest signal and mark price for a combo."""
        c = state.config
        data = self.fetch_data(c.symbol, c.timeframe, n_candles=200)
        signals = state.strategy.generate_signals(data, state.current_params)
        mark_price = float(data["close"].iloc[-1])
        signal = int(signals[-1])
        return signal, mark_price

    # ── Portfolio aggregation (Binance Margin model) ──

    def compute_net_positions(self) -> dict[str, float]:
        """
        Aggregate combo signals into net positions per symbol.

        For each symbol, net_position = Σ(signal_i × weight_i) for all combos on that symbol.
        Then clamp to [-max_position_pct, +max_position_pct].

        Returns dict: symbol -> net_position_fraction (e.g. 0.35 means 35% of equity long)
        """
        raw_positions: dict[str, float] = defaultdict(float)

        for state in self.combo_states:
            c = state.config
            raw_positions[c.symbol] += state.last_signal * c.weight

        # Clamp to risk limits
        max_pos = self.risk_profile.max_position_pct
        net = {}
        for symbol in self.symbols:
            raw = raw_positions.get(symbol, 0.0)
            net[symbol] = np.clip(raw, -max_pos, max_pos)

        return net

    # ── Mark-to-market ──

    def mark_to_market(self, current_prices: dict[str, float]) -> float:
        """
        Update paper equity based on price changes and net positions.
        Returns total interval PnL.
        """
        total_pnl = 0.0

        for symbol in self.symbols:
            net_pos = self.net_positions.get(symbol, 0.0)
            current_price = current_prices.get(symbol)
            last_price = self.last_prices.get(symbol)

            if current_price is None or last_price is None or last_price <= 0:
                continue

            if abs(net_pos) > 1e-6:
                price_return = (current_price / last_price) - 1.0
                # PnL = equity_allocated × return
                # net_pos is fraction of equity, so:
                pnl = self.paper_equity * net_pos * price_return
                total_pnl += pnl

        self.paper_equity += total_pnl
        self.last_prices.update(current_prices)
        return total_pnl

    # ── Execution ──

    def execute_position_changes(self, new_positions: dict[str, float], prices: dict[str, float]):
        """Log position changes (paper) or execute orders (live)."""
        for symbol in self.symbols:
            old_pos = self.net_positions.get(symbol, 0.0)
            new_pos = new_positions.get(symbol, 0.0)

            if abs(new_pos - old_pos) < 1e-4:
                continue

            # Determine trade direction
            if new_pos > old_pos:
                side = "BUY"
            elif new_pos < old_pos:
                side = "SELL"
            else:
                side = "FLAT"

            price = prices.get(symbol, 0.0)
            trade_label = f"{self.portfolio_id}:{symbol}"

            if self.dry_run:
                logger.info(
                    f"[DRY RUN] {symbol}: position {old_pos:+.3f} → {new_pos:+.3f} "
                    f"(Δ={new_pos - old_pos:+.3f}) @ {price:.2f}"
                )
            else:
                logger.info(
                    f"[LIVE] {symbol}: position {old_pos:+.3f} → {new_pos:+.3f} @ {price:.2f}"
                )
                # TODO: Implement actual Binance margin order execution
                # self._place_margin_order(symbol, old_pos, new_pos, price)

            self.monitor.log_trade(
                strategy=trade_label,
                timeframe="portfolio",
                side=side,
                price=price,
                size=abs(new_pos - old_pos),
                pnl=0.0,  # PnL tracked via mark-to-market
            )

        self.net_positions = new_positions.copy()

    # ── Main loop ──

    def run_loop(self, interval_seconds: int = 60):
        """
        Main portfolio execution loop.

        Each iteration:
        1. Re-optimize combos if needed
        2. Generate signals for each combo
        3. Aggregate into net positions per symbol
        4. Mark-to-market on price changes
        5. Execute position changes
        6. Log portfolio PnL
        """
        logger.info("Starting portfolio execution loop...")
        logger.info(f"  Interval: {interval_seconds}s")

        while True:
            try:
                current_prices: dict[str, float] = {}

                # 1. Re-optimize + generate signals for each combo
                for state in self.combo_states:
                    if self.should_reoptimize(state):
                        self.reoptimize_combo(state)

                    signal, mark_price = self.get_combo_signal(state)
                    state.last_signal = signal
                    current_prices[state.config.symbol] = mark_price

                # 2. Initialize last_prices on first iteration
                if not self.last_prices:
                    self.last_prices = current_prices.copy()

                # 3. Mark-to-market before position change
                interval_pnl = self.mark_to_market(current_prices)

                # 4. Compute new net positions
                new_positions = self.compute_net_positions()

                # 5. Execute position changes
                self.execute_position_changes(new_positions, current_prices)

                # 6. Log portfolio-level PnL
                total_return = (self.paper_equity / self.paper_start_equity) - 1.0
                self.monitor.log_pnl(
                    strategy=f"{self.portfolio_id}:portfolio",
                    equity=self.paper_equity,
                    daily_pnl=interval_pnl,
                )
                self.monitor.check_alerts(
                    equity=self.paper_equity,
                    max_drawdown_pct=self.risk_profile.max_drawdown_pct,
                )

                # 7. Debug log
                pos_str = " | ".join(
                    f"{sym}={self.net_positions.get(sym, 0):+.3f}@{current_prices.get(sym, 0):.2f}"
                    for sym in self.symbols
                )
                combo_signals = " ".join(
                    f"{s.config.strategy_name[:8]}={s.last_signal:+d}"
                    for s in self.combo_states
                )
                logger.debug(
                    f"Equity=${self.paper_equity:.2f} ({total_return:+.2%}) | "
                    f"PnL={interval_pnl:+.4f} | Pos: {pos_str} | Sig: {combo_signals}"
                )
                _write_heartbeat()

            except KeyboardInterrupt:
                logger.info("Portfolio execution stopped by user.")
                break
            except Exception as e:
                logger.error(f"Error in portfolio loop: {e}")
                _write_heartbeat()

            time.sleep(interval_seconds)


def load_portfolio_combos(config: dict) -> tuple[list[ComboConfig], RiskProfile]:
    """
    Load combo configs and risk profile from portfolio config JSON.

    Expected config structure:
    {
        "combos": [
            {"strategy_name": "...", "symbol": "...", "timeframe": "...", "weight": 0.4, ...},
            ...
        ],
        "risk_profile": {"max_position_pct": 0.75, "max_drawdown_pct": 0.30}
    }
    """
    combos_data = config.get("combos", [])
    if not combos_data:
        raise ValueError("No combos defined in portfolio config")

    combos = []
    for cd in combos_data:
        combos.append(ComboConfig(
            strategy_name=cd["strategy_name"],
            symbol=cd["symbol"],
            timeframe=cd["timeframe"],
            weight=cd["weight"],
            reoptim_frequency=cd.get("reoptim_frequency", "1M"),
            training_window=cd.get("training_window", "1Y"),
            param_bounds_scale=cd.get("param_bounds_scale", 1.0),
            optim_metric=cd.get("optim_metric", "sharpe"),
            n_optim_trials=cd.get("n_optim_trials", 30),
        ))

    risk_data = config.get("risk_profile", {})
    risk_profile = RiskProfile(
        max_position_pct=risk_data.get("max_position_pct", 0.75),
        max_drawdown_pct=risk_data.get("max_drawdown_pct", 0.30),
    )

    return combos, risk_profile


def run_portfolio(
    config_path: str,
    settings_path: str = "config/settings.yaml",
):
    """
    Start portfolio execution from a config file.
    Entry point for run_portfolio.py.
    """
    import json

    with open(config_path) as f:
        config = json.load(f)

    portfolio_id = config.get("portfolio_id", "unknown")
    dry_run = bool(config.get("execution", {}).get("dry_run", True))
    paper_capital = float(config.get("paper_capital_usd", 1000.0))
    interval_seconds = int(config.get("runtime", {}).get("interval_seconds", 60))
    settings_path = config.get("runtime", {}).get("settings_path", settings_path)

    combos, risk_profile = load_portfolio_combos(config)
    settings = load_settings(settings_path)

    executor = PortfolioExecutor(
        combos=combos,
        risk_profile=risk_profile,
        settings=settings,
        dry_run=dry_run,
        portfolio_id=portfolio_id,
        paper_capital=paper_capital,
    )
    executor.run_loop(interval_seconds=interval_seconds)
