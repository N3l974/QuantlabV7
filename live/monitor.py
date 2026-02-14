"""
Live Monitor — Real-time monitoring of running strategies.
Tracks PnL, positions, and alerts.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from loguru import logger


class LiveMonitor:
    """
    Monitors live strategy execution.
    Logs trades, PnL, and generates alerts.
    """

    def __init__(self, log_dir: str = "logs/live"):
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.trades: list[dict] = []
        self.pnl_history: list[dict] = []

    def log_trade(
        self,
        strategy: str,
        timeframe: str,
        side: str,
        price: float,
        size: float,
        pnl: Optional[float] = None,
        metadata: Optional[dict] = None,
    ):
        """Log a trade execution."""
        trade = {
            "timestamp": datetime.now(UTC).isoformat(),
            "strategy": strategy,
            "timeframe": timeframe,
            "side": side,
            "price": price,
            "size": size,
            "pnl": pnl,
        }
        if metadata is not None:
            trade["metadata"] = metadata
        self.trades.append(trade)
        logger.info(f"TRADE: {side} {size} @ {price} | {strategy}/{timeframe} | PnL: {pnl}")

        # Append to log file
        filepath = os.path.join(self.log_dir, "trades.jsonl")
        with open(filepath, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def log_pnl(
        self,
        strategy: str,
        equity: float,
        daily_pnl: float,
        execution_cost: Optional[float] = None,
        realized_equity: Optional[float] = None,
        floating_pnl: Optional[float] = None,
        gross_exposure: Optional[float] = None,
        net_exposure: Optional[float] = None,
    ):
        """Log current PnL state."""
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "strategy": strategy,
            "equity": equity,
            "daily_pnl": daily_pnl,
        }
        if execution_cost is not None:
            entry["execution_cost"] = execution_cost
        if realized_equity is not None:
            entry["realized_equity"] = realized_equity
        if floating_pnl is not None:
            entry["floating_pnl"] = floating_pnl
        if gross_exposure is not None:
            entry["gross_exposure"] = gross_exposure
        if net_exposure is not None:
            entry["net_exposure"] = net_exposure
        self.pnl_history.append(entry)

        filepath = os.path.join(self.log_dir, "pnl.jsonl")
        with open(filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def check_alerts(self, equity: float, max_drawdown_pct: float = 0.10) -> list[str]:
        """Check for alert conditions."""
        alerts = []

        if len(self.pnl_history) > 1:
            peak = max(e["equity"] for e in self.pnl_history)
            dd = (equity - peak) / peak
            if dd < -max_drawdown_pct:
                alerts.append(
                    f"ALERT: Drawdown {dd:.2%} exceeds threshold {-max_drawdown_pct:.2%}"
                )

        for alert in alerts:
            logger.warning(alert)

        return alerts

    def get_summary(self) -> dict:
        """Get a summary of live performance."""
        if not self.trades:
            return {"n_trades": 0, "total_pnl": 0}

        pnls = [t["pnl"] for t in self.trades if t["pnl"] is not None]
        return {
            "n_trades": len(self.trades),
            "total_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
            "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0,
            "last_trade": self.trades[-1] if self.trades else None,
        }
