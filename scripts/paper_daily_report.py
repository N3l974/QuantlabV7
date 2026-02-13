#!/usr/bin/env python3
"""Daily report for paper trading logs (trades.jsonl + pnl.jsonl)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def max_drawdown(equity_series: list[float]) -> float:
    if not equity_series:
        return 0.0

    peak = equity_series[0]
    worst = 0.0
    for eq in equity_series:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (eq - peak) / peak
            if dd < worst:
                worst = dd
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a short daily paper-trading report")
    parser.add_argument(
        "--log-dir",
        default="runtime/logs/v5c-highrisk-paper",
        help="Directory containing trades.jsonl and pnl.jsonl",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Rolling window in hours for short-term stats",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    trades = load_jsonl(log_dir / "trades.jsonl")
    pnl = load_jsonl(log_dir / "pnl.jsonl")

    if not trades and not pnl:
        print(f"No data found in: {log_dir}")
        return

    now = datetime.utcnow()
    cutoff = now - timedelta(hours=args.hours)

    recent_trades = [t for t in trades if "timestamp" in t and parse_ts(t["timestamp"]) >= cutoff]
    recent_pnl = [p for p in pnl if "timestamp" in p and parse_ts(p["timestamp"]) >= cutoff]

    buy_count = sum(1 for t in trades if t.get("side") == "BUY")
    sell_count = sum(1 for t in trades if t.get("side") == "SELL")
    flat_count = sum(1 for t in trades if t.get("side") == "FLAT")

    equity_series = [float(e.get("equity", 0.0)) for e in pnl if "equity" in e]
    start_eq = equity_series[0] if equity_series else 0.0
    last_eq = equity_series[-1] if equity_series else 0.0
    total_ret = ((last_eq / start_eq) - 1.0) if start_eq > 0 else 0.0
    dd = max_drawdown(equity_series)

    recent_interval_pnl = sum(float(e.get("daily_pnl", 0.0)) for e in recent_pnl)

    print("=" * 72)
    print("PAPER DAILY REPORT")
    print("=" * 72)
    print(f"log_dir: {log_dir}")
    print(f"window: last {args.hours}h")
    print()
    print("[Service activity]")
    print(f"total trades: {len(trades)}")
    print(f"  BUY={buy_count} SELL={sell_count} FLAT={flat_count}")
    print(f"trades in window: {len(recent_trades)}")
    print()
    print("[Equity]")
    print(f"start equity: {start_eq:.2f}")
    print(f"last equity:  {last_eq:.2f}")
    print(f"total return: {total_ret * 100:.2f}%")
    print(f"max drawdown: {dd * 100:.2f}%")
    print(f"window pnl:   {recent_interval_pnl:.4f}")

    if recent_pnl:
        first_ts = recent_pnl[0].get("timestamp", "-")
        last_ts = recent_pnl[-1].get("timestamp", "-")
        print()
        print("[Window coverage]")
        print(f"first pnl ts: {first_ts}")
        print(f"last  pnl ts: {last_ts}")

    print("=" * 72)


if __name__ == "__main__":
    main()
