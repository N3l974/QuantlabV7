#!/usr/bin/env python3
"""Daily report for paper trading logs (trades.jsonl + pnl.jsonl)."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def parse_ts(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def ts_in_window(raw: str, cutoff: datetime) -> bool:
    try:
        return parse_ts(raw) >= cutoff
    except (TypeError, ValueError):
        return False


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


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


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
    parser.add_argument(
        "--since-start",
        action="store_true",
        help="Ignore rolling window and report on full log history",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Optional state.json path (used to enrich exposure/realized snapshot)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    trades = load_jsonl(log_dir / "trades.jsonl")
    pnl = load_jsonl(log_dir / "pnl.jsonl")
    state_path = Path(args.state_file) if args.state_file else (log_dir / "state.json")
    state = load_state(state_path)

    if not trades and not pnl:
        print(f"No data found in: {log_dir}")
        return

    now = datetime.now(UTC)
    cutoff = None if args.since_start else now - timedelta(hours=args.hours)

    if cutoff is None:
        recent_trades = trades
        recent_pnl = pnl
    else:
        recent_trades = [t for t in trades if "timestamp" in t and ts_in_window(t["timestamp"], cutoff)]
        recent_pnl = [p for p in pnl if "timestamp" in p and ts_in_window(p["timestamp"], cutoff)]

    buy_count = sum(1 for t in trades if t.get("side") == "BUY")
    sell_count = sum(1 for t in trades if t.get("side") == "SELL")
    flat_count = sum(1 for t in trades if t.get("side") == "FLAT")

    equity_series = [float(e.get("equity", 0.0)) for e in pnl if "equity" in e]
    start_eq = equity_series[0] if equity_series else 0.0
    last_eq = equity_series[-1] if equity_series else 0.0
    total_ret = ((last_eq / start_eq) - 1.0) if start_eq > 0 else 0.0
    dd = max_drawdown(equity_series)

    last_pnl_entry = pnl[-1] if pnl else {}
    realized_eq = float(last_pnl_entry.get("realized_equity", state.get("last_flat_equity", last_eq)))
    floating_pnl = float(last_pnl_entry.get("floating_pnl", last_eq - realized_eq))
    net_positions = state.get("net_positions", {}) if isinstance(state, dict) else {}
    gross_exposure = last_pnl_entry.get("gross_exposure")
    net_exposure = last_pnl_entry.get("net_exposure")
    if gross_exposure is None and isinstance(net_positions, dict) and net_positions:
        gross_exposure = sum(abs(float(v)) for v in net_positions.values())
    if net_exposure is None and isinstance(net_positions, dict) and net_positions:
        net_exposure = sum(float(v) for v in net_positions.values())

    recent_interval_pnl = sum(float(e.get("daily_pnl", 0.0)) for e in recent_pnl)
    recent_execution_cost = sum(float(e.get("execution_cost", 0.0)) for e in recent_pnl)

    print("=" * 72)
    print("PAPER DAILY REPORT")
    print("=" * 72)
    print(f"log_dir: {log_dir}")
    window_label = "since start" if args.since_start else f"last {args.hours}h"
    print(f"window: {window_label}")
    print()
    print("[Service activity]")
    print(f"total trades: {len(trades)}")
    print(f"  BUY={buy_count} SELL={sell_count} FLAT={flat_count}")
    print(f"trades in window: {len(recent_trades)}")
    print()
    print("[Equity]")
    print(f"start equity: {start_eq:.2f}")
    print(f"last equity (mtm):      {last_eq:.2f}")
    print(f"realized base (flat):  {realized_eq:.2f}")
    print(f"floating pnl:          {floating_pnl:+.4f}")
    print(f"total return: {total_ret * 100:.2f}%")
    print(f"max drawdown: {dd * 100:.2f}%")
    print(f"window pnl net:        {recent_interval_pnl:.4f}")
    if recent_execution_cost > 0:
        print(f"window execution cost: {recent_execution_cost:.4f}")
    if gross_exposure is not None:
        print(f"gross exposure:        {float(gross_exposure):+.3f}")
    if net_exposure is not None:
        print(f"net exposure:          {float(net_exposure):+.3f}")

    if isinstance(net_positions, dict) and net_positions:
        open_symbols = sum(1 for v in net_positions.values() if abs(float(v)) > 1e-6)
        print(f"open symbols:          {open_symbols}/{len(net_positions)}")

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
