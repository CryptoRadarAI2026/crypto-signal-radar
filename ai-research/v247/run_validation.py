"""V24.7 chronological parameter comparison.

Validation selects parameters; holdout is evaluated once.
Research only. No real trading.
"""

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import backtest_engine as engine
from signal_generator import generate_signals


HERE = Path(__file__).resolve().parent

DEFAULT_OUTPUT = (
    Path("research-output")
    / "v247"
    / "validation_report.json"
)


def audit(path, candles):
    raw = path.read_bytes()

    return {
        "source_file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "candles": len(candles),
        "start_utc": datetime.fromtimestamp(
            candles[0].timestamp,
            timezone.utc,
        ).isoformat(),
        "end_utc": datetime.fromtimestamp(
            candles[-1].timestamp,
            timezone.utc,
        ).isoformat(),
        "timestamp_continuity_checked": True,
        "origin_verified": False,
        "origin_note": (
            "SHA256 proves file identity only, "
            "not Coinbase provenance."
        ),
    }


def validate_candles(candles):
    if len(candles) < 1000:
        raise ValueError("Need at least 1000 candles")

    if any(c.timestamp % 60 for c in candles):
        raise ValueError("Non-minute-aligned timestamp")

    for a, b in zip(candles, candles[1:]):
        if b.timestamp - a.timestamp != 60:
            raise ValueError(
                "Missing, duplicated, or unordered minute candles"
            )


def simulate(candles, config, settings, params):
    holding, stop, target = params

    signals = generate_signals(
        candles,
        engine,
        settings,
        stop,
        target,
        holding,
    )

    result = engine.BTCBacktestEngine(config).run(
        candles,
        signals,
    )

    return {
        "holding_minutes": holding,
        "stop_loss_pct": stop,
        "take_profit_pct": target,
        "signals": len(signals),
        "trades": result["closed_trades"],
        "net_pnl_usdt": result["net_pnl_usdt"],
        "return_pct": result["return_pct"],
        "profit_factor": result["profit_factor"],
        "max_drawdown_pct": result["max_drawdown_pct"],
        "fees_usdt": round(
            result["total_entry_fees_usdt"]
            + result["total_exit_fees_usdt"],
            6,
        ),
        "rejected_signals": result["rejected_signals"],
        "rejection_reasons": result["rejection_reasons"],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candles",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
    )

    args = parser.parse_args()

    config = json.loads(
        (HERE / "research_config.json").read_text(
            encoding="utf-8"
        )
    )

    engine.verify_safety(config)

    settings = config["research"]

    path = Path(args.candles)

    candles = engine.load_candles_csv(path)

    validate_candles(candles)

    n = len(candles)

    train_end = int(
        n * settings["train_fraction"]
    )

    validation_end = int(
        n * (
            settings["train_fraction"]
            + settings["validation_fraction"]
        )
    )

    embargo = settings["embargo_minutes"]
    warmup = settings["minimum_history"] + 2

    if embargo < max(settings["holding_minutes"]):
        raise ValueError(
            "Embargo must cover maximum holding period"
        )

    # Training is reserved for future model fitting.
    # This baseline has no fitted parameters.
    # Validation and holdout use local indicator warmup.

    val_start = train_end + embargo
    hold_start = validation_end + embargo

    if (
        validation_end - val_start < warmup + 100
        or n - hold_start < warmup + 100
    ):
        raise ValueError(
            "Not enough independent validation/holdout candles"
        )

    validation = candles[
        val_start:validation_end
    ]

    holdout = candles[
        hold_start:
    ]

    candidates = list(
        itertools.product(
            settings["holding_minutes"],
            settings["stop_loss_pct"],
            settings["take_profit_pct"],
        )
    )

    rows = [
        simulate(
            validation,
            config,
            settings,
            params,
        )
        for params in candidates
    ]

    eligible = [
        row
        for row in rows
        if row["trades"]
        >= settings["min_trade_count_validation"]
    ]

    chosen = (
        max(
            eligible,
            key=lambda row: (
                row["net_pnl_usdt"],
                -row["max_drawdown_pct"],
            ),
        )
        if eligible
        else None
    )

    final = None

    if chosen:
        params = (
            chosen["holding_minutes"],
            chosen["stop_loss_pct"],
            chosen["take_profit_pct"],
        )

        final = simulate(
            holdout,
            config,
            settings,
            params,
        )

    report = {
        "version": "V24.7",
        "status": "RESEARCH_ONLY_NOT_APPROVED",
        "data_audit": audit(path, candles),
        "split": {
            "train_end_index": train_end,
            "validation_start_index": val_start,
            "validation_end_index": validation_end,
            "holdout_start_index": hold_start,
            "embargo_minutes": embargo,
            "train_is_reserved": True,
        },
        "validation_candidates": rows,
        "selected_on_validation_only": chosen,
        "holdout_one_pass": final,
        "methodology": [
            "Baseline indicators are deterministic; no model is trained.",
            "Independent chronological validation and holdout windows with embargo.",
            "Indicator warmup is local to each evaluation window.",
            "Parameter selection uses validation only; holdout is run once.",
            "Not walk-forward: future V24.8 work.",
            "No ATR stops, trailing stops, partial exits, funding, or liquidation.",
            "BTC-USD spot candles are not futures fills.",
            "File origin is unverified; hash is for reproducibility.",
            "Small sample sizes invalidate strong profitability claims.",
        ],
    }

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "report": str(output),
                "selected": chosen,
                "holdout": final,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
