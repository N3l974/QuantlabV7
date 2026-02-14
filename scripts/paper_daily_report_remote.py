#!/usr/bin/env python3
"""Fetch paper logs from VPS then run local daily report."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def build_ssh_opts(ssh_key: str | None) -> list[str]:
    opts: list[str] = []
    if ssh_key:
        opts.extend(["-i", ssh_key])
    return opts


def fetch_file(vps_user: str, vps_host: str, remote_log_dir: str, filename: str, out_dir: Path, ssh_opts: list[str]) -> bool:
    remote = f"{vps_user}@{vps_host}:{remote_log_dir.rstrip('/')}/{filename}"
    dest = str(out_dir / filename)
    cmd = ["scp", *ssh_opts, remote, dest]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[warn] unable to fetch {filename}: {res.stderr.strip()}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch VPS paper logs and build local daily report")
    parser.add_argument("--vps-host", required=True, help="VPS host or IP")
    parser.add_argument("--vps-user", required=True, help="SSH user on VPS")
    parser.add_argument(
        "--remote-log-dir",
        default="~/quantlab-deploy/runtime/logs/v5c-highrisk-paper",
        help="Remote directory containing trades.jsonl and pnl.jsonl",
    )
    parser.add_argument("--ssh-key", default=None, help="Optional SSH private key path")
    parser.add_argument("--hours", type=int, default=24, help="Rolling window in hours")
    parser.add_argument(
        "--since-start",
        action="store_true",
        help="Report on full history (ignores --hours)",
    )
    args = parser.parse_args()

    ssh_opts = build_ssh_opts(args.ssh_key)

    with tempfile.TemporaryDirectory(prefix="paper_logs_") as tmp:
        tmp_dir = Path(tmp)
        got_trades = fetch_file(args.vps_user, args.vps_host, args.remote_log_dir, "trades.jsonl", tmp_dir, ssh_opts)
        got_pnl = fetch_file(args.vps_user, args.vps_host, args.remote_log_dir, "pnl.jsonl", tmp_dir, ssh_opts)
        got_state = fetch_file(args.vps_user, args.vps_host, args.remote_log_dir, "state.json", tmp_dir, ssh_opts)

        if not got_trades and not got_pnl:
            print("No log file fetched from VPS.")
            raise SystemExit(1)

        report_cmd = [
            sys.executable,
            "scripts/paper_daily_report.py",
            "--log-dir",
            str(tmp_dir),
            "--hours",
            str(args.hours),
        ]
        if got_state:
            report_cmd.extend(["--state-file", str(tmp_dir / "state.json")])
        if args.since_start:
            report_cmd.append("--since-start")
        subprocess.run(report_cmd, check=True)


if __name__ == "__main__":
    main()
