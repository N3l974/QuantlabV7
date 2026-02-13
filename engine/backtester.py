"""
Vectorized backtesting engine.
Operates on numpy arrays for maximum speed.
Supports long/short with commission, slippage, and risk management.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class RiskConfig:
    """Risk management configuration for realistic backtesting."""
    max_position_pct: float = 0.25       # Max 25% of capital per trade
    risk_per_trade_pct: float = 0.0      # Risk % per trade (0 = disabled, use max_position_pct)
    max_daily_loss_pct: float = 0.03     # Stop trading if daily loss > 3%
    max_drawdown_pct: float = 0.15       # Circuit breaker at 15% drawdown
    dynamic_slippage: bool = True        # Scale slippage with volatility
    base_slippage: float = 0.0005        # Base slippage (0.05%)
    max_slippage: float = 0.005          # Max slippage in volatile markets (0.5%)
    volatility_lookback: int = 20        # ATR lookback for dynamic slippage
    max_trades_per_day: int = 10         # Max trades per day (anti-overtrading)
    cooldown_after_loss: int = 0         # Bars to skip after a losing trade


@dataclass
class BacktestResult:
    """Container for backtest results."""
    equity: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    trades_pnl: np.ndarray
    n_trades: int
    entry_times: list
    exit_times: list
    risk_events: dict = field(default_factory=dict)


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Compute Average True Range for dynamic slippage."""
    n = len(close)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(period - 1, n):
        atr[i] = np.mean(tr[i - period + 1 : i + 1])
    return atr


def _dynamic_slippage(atr_val: float, close_val: float, risk: RiskConfig) -> float:
    """Compute slippage scaled by current volatility."""
    if np.isnan(atr_val) or close_val == 0:
        return risk.base_slippage
    vol_ratio = atr_val / close_val
    # Scale: low vol → base_slippage, high vol → up to max_slippage
    slippage = risk.base_slippage + vol_ratio * 2.0
    return min(slippage, risk.max_slippage)


# Bars per day for each timeframe (used for daily reset)
BARS_PER_DAY = {
    "1m": 1440, "5m": 288, "15m": 96,
    "1h": 24, "4h": 6, "1d": 1,
}

# Funding rate: charged every 8h on open positions (Binance perpetual/margin)
FUNDING_RATE_PER_8H = 0.0001  # 0.01% per 8h
BARS_PER_8H = {
    "1m": 480, "5m": 96, "15m": 32,
    "1h": 8, "4h": 2, "1d": 0.333,
}


def vectorized_backtest(
    close: np.ndarray,
    signals: np.ndarray,
    commission: float = 0.001,
    slippage: float = 0.0005,
    initial_capital: float = 10000.0,
    risk: Optional[RiskConfig] = None,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    timeframe: str = "1h",
    sl_distances: Optional[np.ndarray] = None,
) -> BacktestResult:
    """
    Run a vectorized backtest on signal array with risk management.

    Args:
        close: Array of close prices.
        signals: Array of signals: +1 long, -1 short, 0 flat.
        commission: Commission rate per trade (fraction).
        slippage: Slippage rate per trade (fraction).
        initial_capital: Starting capital.
        risk: RiskConfig for position sizing and risk limits.
        high: High prices (for dynamic slippage ATR).
        low: Low prices (for dynamic slippage ATR).
        sl_distances: Per-bar SL distance as fraction of price (for risk-based sizing).
                      When risk.risk_per_trade_pct > 0 and sl_distances is provided,
                      position size = (equity * risk_pct) / sl_distance.

    Returns:
        BacktestResult with equity curve, trades, etc.
    """
    n = len(close)
    assert len(signals) == n, "close and signals must have same length"

    if risk is None:
        risk = RiskConfig()

    # Pre-compute ATR for dynamic slippage
    atr = None
    if risk.dynamic_slippage and high is not None and low is not None:
        atr = _compute_atr(high, low, close, risk.volatility_lookback)

    positions = np.zeros(n, dtype=np.float64)
    equity = np.zeros(n, dtype=np.float64)
    equity[0] = initial_capital

    trades_pnl = []
    entry_times = []
    exit_times = []

    current_pos = 0.0       # +1 long, -1 short, 0 flat
    entry_price = 0.0       # Price at which position was opened (incl. slippage)
    allocated_capital = 0.0  # Dollar amount allocated to current trade
    cash = initial_capital   # Unallocated cash
    peak_equity = initial_capital

    # Risk tracking
    daily_pnl = 0.0
    last_day_bar = 0
    trades_today = 0
    cooldown_remaining = 0
    circuit_breaker_active = False
    bars_per_day = BARS_PER_DAY.get(timeframe, 24)
    bars_per_funding = BARS_PER_8H.get(timeframe, 8)
    funding_accumulator = 0.0

    risk_events = {
        "circuit_breaker_triggers": 0,
        "daily_loss_stops": 0,
        "max_trades_stops": 0,
        "cooldown_skips": 0,
    }

    def _get_slip(i_bar):
        if risk.dynamic_slippage and atr is not None:
            return _dynamic_slippage(atr[i_bar], close[i_bar], risk)
        return slippage

    def _close_position(i_bar):
        """Close current position, return realized PnL in dollars."""
        nonlocal current_pos, entry_price, allocated_capital, cash
        slip = _get_slip(i_bar)
        exit_price = close[i_bar] * (1 - slip * np.sign(current_pos))
        pnl_pct = current_pos * (exit_price - entry_price) / entry_price
        trade_cost = commission * 2  # Entry + exit commission
        net_pnl_pct = pnl_pct - trade_cost
        trade_pnl_dollar = net_pnl_pct * allocated_capital
        # Return allocated capital + PnL to cash
        cash += allocated_capital + trade_pnl_dollar
        trades_pnl.append(trade_pnl_dollar)
        exit_times.append(i_bar)
        current_pos = 0.0
        entry_price = 0.0
        allocated_capital = 0.0
        return trade_pnl_dollar

    def _open_position(i_bar, direction, magnitude=1.0):
        """Open a new position. magnitude scales position size (0.0-1.0+)."""
        nonlocal current_pos, entry_price, allocated_capital, cash
        slip = _get_slip(i_bar)
        entry_price = close[i_bar] * (1 + slip * np.sign(direction))
        current_pos = direction
        size_scalar = np.clip(abs(magnitude), 0.0, 2.0)  # Cap at 2x

        # Risk-based sizing: position = (equity * risk%) / SL_distance
        if (risk.risk_per_trade_pct > 0
                and sl_distances is not None
                and not np.isnan(sl_distances[i_bar])
                and sl_distances[i_bar] > 0):
            total_eq = _total_equity(i_bar)
            risk_dollar = total_eq * risk.risk_per_trade_pct * size_scalar
            allocated_capital = risk_dollar / sl_distances[i_bar]
            # Cap at max_position_pct of equity
            max_alloc = risk.max_position_pct * total_eq
            allocated_capital = min(allocated_capital, max_alloc)
        else:
            # Fallback: fraction of cash
            allocated_capital = risk.max_position_pct * size_scalar * cash

        allocated_capital = min(allocated_capital, cash)  # Never exceed cash
        cash -= allocated_capital
        entry_times.append(i_bar)

    def _total_equity(i_bar):
        """Compute total equity = cash + unrealized value of position."""
        if current_pos == 0 or entry_price == 0:
            return cash
        unrealized_pnl_pct = current_pos * (close[i_bar] - entry_price) / entry_price
        return cash + allocated_capital * (1 + unrealized_pnl_pct)

    for i in range(1, n):
        sig = signals[i]
        prev_pos = current_pos

        # ── Daily reset (timeframe-aware) ──
        if i - last_day_bar >= bars_per_day:
            daily_pnl = 0.0
            trades_today = 0
            last_day_bar = i

        # ── Circuit breaker check ──
        total_eq = _total_equity(i)
        drawdown = (total_eq - peak_equity) / peak_equity if peak_equity > 0 else 0
        if drawdown < -risk.max_drawdown_pct:
            if not circuit_breaker_active:
                circuit_breaker_active = True
                risk_events["circuit_breaker_triggers"] += 1
            # Force flat
            if current_pos != 0:
                pnl = _close_position(i)
                daily_pnl += pnl
            positions[i] = 0.0
            equity[i] = _total_equity(i)
            continue

        # Reset circuit breaker if recovered
        if circuit_breaker_active and drawdown > -risk.max_drawdown_pct * 0.5:
            circuit_breaker_active = False

        # ── Daily loss limit ──
        if daily_pnl / max(_total_equity(i), 1) < -risk.max_daily_loss_pct:
            risk_events["daily_loss_stops"] += 1
            sig = 0  # Force flat for rest of day

        # ── Cooldown after loss ──
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            risk_events["cooldown_skips"] += 1
            sig = 0  # Force flat during cooldown

        # ── Max trades per day ──
        if trades_today >= risk.max_trades_per_day and sig != prev_pos and sig != 0:
            risk_events["max_trades_stops"] += 1
            sig = prev_pos  # Keep current position

        # ── Position change ──
        # Extract direction and magnitude from signal
        sig_dir = np.sign(sig)      # -1, 0, +1
        sig_mag = abs(sig)           # 0.0 to N (from overlays)
        if sig_mag == 0:
            sig_dir = 0.0

        if sig_dir != prev_pos:
            # Close existing position
            if prev_pos != 0:
                pnl = _close_position(i)
                trades_today += 1
                daily_pnl += pnl
                # Cooldown after losing trade
                if pnl < 0 and risk.cooldown_after_loss > 0:
                    cooldown_remaining = risk.cooldown_after_loss

            # Open new position with position sizing
            if sig_dir != 0:
                _open_position(i, sig_dir, magnitude=sig_mag)

        # ── Funding rate for open positions ──
        if current_pos != 0 and bars_per_funding > 0:
            funding_accumulator += 1.0
            if funding_accumulator >= bars_per_funding:
                funding_cost = FUNDING_RATE_PER_8H * allocated_capital
                cash -= funding_cost
                daily_pnl -= funding_cost
                funding_accumulator -= bars_per_funding

        # Update equity and peak
        total_eq = _total_equity(i)
        if total_eq > peak_equity:
            peak_equity = total_eq

        positions[i] = current_pos
        equity[i] = total_eq

    # Close any remaining position at the end
    if current_pos != 0:
        _close_position(n - 1)
        equity[-1] = _total_equity(n - 1)

    # Fix first equity value
    equity[0] = initial_capital

    # Compute returns
    returns = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1.0)

    return BacktestResult(
        equity=equity,
        returns=returns,
        positions=positions,
        trades_pnl=np.array(trades_pnl),
        n_trades=len(trades_pnl),
        entry_times=entry_times,
        exit_times=exit_times,
        risk_events=risk_events,
    )


def backtest_strategy(
    strategy,
    data: pd.DataFrame,
    params: dict,
    commission: float = 0.001,
    slippage: float = 0.0005,
    initial_capital: float = 10000.0,
    risk: Optional[RiskConfig] = None,
    timeframe: str = "1h",
) -> BacktestResult:
    """
    High-level backtest: generate signals from strategy, then run vectorized backtest.

    Args:
        strategy: Strategy instance with generate_signals() method.
        data: OHLCV DataFrame.
        params: Strategy parameters dict.
        commission: Commission rate.
        slippage: Slippage rate.
        initial_capital: Starting capital.
        risk: RiskConfig for position sizing and risk limits.
        timeframe: Candle timeframe for daily reset and funding rate.

    Returns:
        BacktestResult
    """
    # Try V5 API first (returns signals + sl_distances), fallback to V4
    sl_distances = None
    if hasattr(strategy, 'generate_signals_v5'):
        signals, sl_distances = strategy.generate_signals_v5(data, params)
    else:
        signals = strategy.generate_signals(data, params)

    close = data["close"].values.astype(np.float64)
    signals_arr = signals.values.astype(np.float64) if isinstance(signals, pd.Series) else signals

    # Extract high/low for dynamic slippage
    high = data["high"].values.astype(np.float64) if "high" in data.columns else None
    low = data["low"].values.astype(np.float64) if "low" in data.columns else None

    # Ensure same length
    min_len = min(len(close), len(signals_arr))
    close = close[:min_len]
    signals_arr = signals_arr[:min_len]
    if high is not None:
        high = high[:min_len]
    if low is not None:
        low = low[:min_len]
    if sl_distances is not None:
        sl_distances = sl_distances[:min_len]

    return vectorized_backtest(
        close=close,
        signals=signals_arr,
        commission=commission,
        slippage=slippage,
        initial_capital=initial_capital,
        risk=risk,
        high=high,
        low=low,
        timeframe=timeframe,
        sl_distances=sl_distances,
    )
