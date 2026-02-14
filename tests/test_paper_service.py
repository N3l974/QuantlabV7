"""Tests for paper-service reporting and execution accounting."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from live.monitor import LiveMonitor
import live.portfolio_executor as pe
from scripts import paper_daily_report


class DummyStrategy:
    """Minimal strategy stub for executor tests."""

    def generate_signals(self, data, params):  # pragma: no cover - not used in these tests
        return [0] * len(data)


class DummyMonitor:
    """In-memory monitor to avoid filesystem writes in executor unit tests."""

    def __init__(self, log_dir: str = "logs/live"):
        self.log_dir = log_dir
        self.trades = []
        self.pnl_history = []

    def log_trade(self, **kwargs):
        self.trades.append(kwargs)

    def log_pnl(self, **kwargs):
        self.pnl_history.append(kwargs)

    def check_alerts(self, equity: float, max_drawdown_pct: float = 0.10):
        return []


def test_monitor_logs_enriched_pnl_fields(tmp_path):
    monitor = LiveMonitor(log_dir=str(tmp_path))

    monitor.log_pnl(
        strategy="paper:test",
        equity=1012.5,
        daily_pnl=12.5,
        execution_cost=0.8,
        realized_equity=1008.0,
        floating_pnl=4.5,
        gross_exposure=0.70,
        net_exposure=0.42,
    )

    lines = (tmp_path / "pnl.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])

    assert row["execution_cost"] == pytest.approx(0.8)
    assert row["realized_equity"] == pytest.approx(1008.0)
    assert row["floating_pnl"] == pytest.approx(4.5)
    assert row["gross_exposure"] == pytest.approx(0.70)
    assert row["net_exposure"] == pytest.approx(0.42)


def test_executor_applies_execution_cost(monkeypatch):
    monkeypatch.setattr(pe, "get_binance_client", lambda settings: object())
    monkeypatch.setattr(pe, "get_strategy", lambda name: DummyStrategy())
    monkeypatch.setattr(pe, "LiveMonitor", DummyMonitor)

    combos = [
        pe.ComboConfig(
            strategy_name="dummy",
            symbol="ETHUSDT",
            timeframe="1d",
            weight=1.0,
        )
    ]
    risk_profile = pe.RiskProfile(max_position_pct=1.0, max_drawdown_pct=0.30)
    settings = {"engine": {"commission_rate": 0.001, "slippage_rate": 0.0005}}

    executor = pe.PortfolioExecutor(
        combos=combos,
        risk_profile=risk_profile,
        settings=settings,
        dry_run=True,
        portfolio_id="test-paper-exec",
        paper_capital=1000.0,
        pause_on_expiry=True,
    )

    cost = executor.execute_position_changes(
        new_positions={"ETHUSDT": 0.5},
        prices={"ETHUSDT": 100.0},
        combo_breakdown={"ETHUSDT": []},
    )

    # Cost = equity * delta_position * (commission + slippage)
    expected_cost = 1000.0 * 0.5 * (0.001 + 0.0005)
    assert cost == pytest.approx(expected_cost)
    assert executor.paper_equity == pytest.approx(1000.0 - expected_cost)

    trade = executor.monitor.trades[-1]
    assert trade["metadata"]["execution_cost"] == pytest.approx(expected_cost)


def test_pause_on_expiry_stops_loop_on_reopt_failure(monkeypatch):
    monkeypatch.setattr(pe, "get_binance_client", lambda settings: object())
    monkeypatch.setattr(pe, "get_strategy", lambda name: DummyStrategy())
    monkeypatch.setattr(pe, "LiveMonitor", DummyMonitor)
    monkeypatch.setattr(pe, "_write_heartbeat", lambda: None)

    combos = [
        pe.ComboConfig(
            strategy_name="dummy",
            symbol="ETHUSDT",
            timeframe="1d",
            weight=1.0,
        )
    ]
    risk_profile = pe.RiskProfile(max_position_pct=1.0, max_drawdown_pct=0.30)
    settings = {"engine": {"commission_rate": 0.001, "slippage_rate": 0.0005}}

    executor = pe.PortfolioExecutor(
        combos=combos,
        risk_profile=risk_profile,
        settings=settings,
        dry_run=True,
        portfolio_id="test-paper-pause",
        paper_capital=1000.0,
        pause_on_expiry=True,
    )

    monkeypatch.setattr(executor, "_save_state", lambda: None)
    monkeypatch.setattr(executor, "should_reoptimize", lambda state: True)

    def boom(_state):
        raise RuntimeError("reopt failure")

    monkeypatch.setattr(executor, "reoptimize_combo", boom)

    # Must return (stop loop) instead of spinning forever.
    executor.run_loop(interval_seconds=0)

    # No pnl logged because loop exits before signal / pnl steps.
    assert executor.monitor.pnl_history == []


def test_paper_daily_report_prints_mtm_realized_and_cost(tmp_path, monkeypatch, capsys):
    trades = [
        {
            "timestamp": "2026-02-14T10:00:00+00:00",
            "side": "BUY",
            "strategy": "paper:test",
            "timeframe": "portfolio",
            "price": 100.0,
            "size": 0.5,
            "pnl": 0.0,
        }
    ]
    pnl = [
        {
            "timestamp": "2026-02-14T10:00:00+00:00",
            "strategy": "paper:test",
            "equity": 1000.0,
            "daily_pnl": 0.0,
            "execution_cost": 0.0,
            "realized_equity": 1000.0,
            "floating_pnl": 0.0,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
        },
        {
            "timestamp": "2026-02-14T11:00:00+00:00",
            "strategy": "paper:test",
            "equity": 1010.0,
            "daily_pnl": 9.2,
            "execution_cost": 0.8,
            "realized_equity": 1006.0,
            "floating_pnl": 4.0,
            "gross_exposure": 0.6,
            "net_exposure": 0.4,
        },
    ]
    state = {"last_flat_equity": 1006.0, "net_positions": {"ETHUSDT": 0.4, "SOLUSDT": 0.0}}

    (tmp_path / "trades.jsonl").write_text("\n".join(json.dumps(t) for t in trades) + "\n", encoding="utf-8")
    (tmp_path / "pnl.jsonl").write_text("\n".join(json.dumps(p) for p in pnl) + "\n", encoding="utf-8")
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paper_daily_report.py",
            "--log-dir",
            str(tmp_path),
            "--state-file",
            str(state_path),
            "--since-start",
        ],
    )

    paper_daily_report.main()
    output = capsys.readouterr().out

    assert "last equity (mtm)" in output
    assert "realized base (flat)" in output
    assert "floating pnl" in output
    assert "window pnl net" in output
    assert "window execution cost" in output
    assert "open symbols" in output
