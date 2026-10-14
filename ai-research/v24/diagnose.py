#!/usr/bin/env python3
"""
Crypto Signal Radar AI - V24
Data Integrity & Backtest Diagnostics

Research only.
No trading, no API keys, no orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from datetime import datetime, timezone


VERSION = "V24"
STATUS = "RESEARCH_ONLY_NOT_APPROVED"


def load_csv(path: Path):
    """Load a CSV file safely."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        return list(reader)


def check_data(rows):
    """Basic OHLCV data-quality diagnostics."""

    report = {
        "total_rows": len(rows),
        "missing_values": 0,
        "invalid_prices": 0,
        "invalid_volume": 0,
        "duplicate_timestamps": 0,
        "out_of_order_timestamps": 0,
    }

    seen = set()
    previous_time = None

    for row in rows:

        if any(
            value is None or str(value).strip() == ""
            for value in row.values()
        ):
            report["missing_values"] += 1

        timestamp = (
            row.get("timestamp")
            or row.get("time")
            or row.get("date")
        )

        if timestamp is not None:

            if timestamp in seen:
                report["duplicate_timestamps"] += 1

            seen.add(timestamp)

            if (
                previous_time is not None
                and timestamp < previous_time
            ):
                report["out_of_order_timestamps"] += 1

            previous_time = timestamp

        try:
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])

            if (
                not all(math.isfinite(x) for x in (o, h, l, c))
                or min(o, h, l, c) <= 0
                or h < max(o, c, l)
                or l > min(o, c, h)
            ):
                report["invalid_prices"] += 1

        except (KeyError, ValueError, TypeError):
            report["invalid_prices"] += 1

        try:
            volume = float(row["volume"])

            if not math.isfinite(volume) or volume < 0:
                report["invalid_volume"] += 1

        except (KeyError, ValueError, TypeError):
            report["invalid_volume"] += 1

    return report


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to OHLCV CSV file"
    )

    parser.add_argument(
        "--output",
        default="v24_diagnostics.json"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_csv(input_path)

    diagnostics = check_data(rows)

    result = {
        "version": VERSION,
        "status": STATUS,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_file": str(input_path),
        "diagnostics": diagnostics,
        "approved_for_live_trading": False,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
