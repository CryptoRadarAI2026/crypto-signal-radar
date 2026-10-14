"""
V24.6 - BTC Historical Research Backtest Runner

RESEARCH ONLY.
NO EXCHANGE CONNECTION.
NO REAL ORDERS.
NO LIVE TRADING.
NO TRADING APPROVAL.

This script:
- Loads historical BTC-USD 1-minute candles from CSV.
- Creates deterministic EMA/RSI research signals.
- Uses only completed candles for signal generation.
- Executes signals through the V24.5 research engine.
- Writes a JSON report.

This is a baseline experiment, NOT a validated strategy.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path


VERSION = "V24.6"
STATUS = "RESEARCH_ONLY_NOT_APPROVED"

HERE = Path(__file__).resolve().parent
RESEARCH_DIR = HERE.parent

ENGINE_PATH = (
    RESEARCH_DIR
    / "v245"
    / "backtest_engine.py"
)

CONFIG_PATH = (
    RESEARCH_DIR
    / "v245"
    / "research_config.json"
)


def load_engine_module():
    if not ENGINE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing V24.5 engine: {ENGINE_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "v245_backtest_engine",
        ENGINE_PATH
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load V24.5 engine"
        )

    import sys

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


def ema_series(values, period):
    if period < 2:
        raise ValueError(
            "EMA period must be >= 2"
        )

    if not values:
        return []

    alpha = 2.0 / (period + 1.0)

    result = []
    previous = float(values[0])

    for value in values:
        previous = (
            alpha * float(value)
            + (1.0 - alpha) * previous
        )

        result.append(previous)

    return result


def rsi_series(values, period=14):
    if period < 2:
        raise ValueError(
            "RSI period must be >= 2"
        )

    result = [None] * len(values)

    if len(values) <= period:
        return result

    gains = []
    losses = []

    for index in range(1, period + 1):
        change = (
            values[index]
            - values[index - 1]
        )

        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    def calculate_rsi(gain, loss):
        if loss == 0:
            if gain == 0:
                return 50.0

            return 100.0

        relative_strength = gain / loss

        return (
            100.0
            - 100.0 / (1.0 + relative_strength)
        )

    result[period] = calculate_rsi(
        average_gain,
        average_loss
    )

    for index in range(
        period + 1,
        len(values)
    ):
        change = (
            values[index]
            - values[index - 1]
        )

        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        average_gain = (
            average_gain * (period - 1)
            + gain
        ) / period

        average_loss = (
            average_loss * (period - 1)
            + loss
        ) / period

        result[index] = calculate_rsi(
            average_gain,
            average_loss
        )

    return result


def validate_parameters(args):
    if args.fast_ema < 2:
        raise ValueError(
            "fast_ema must be >= 2"
        )

    if args.slow_ema <= args.fast_ema:
        raise ValueError(
            "slow_ema must exceed fast_ema"
        )

    if args.rsi_period < 2:
        raise ValueError(
            "rsi_period must be >= 2"
        )

    if not (
        0.0 < args.stop_loss_pct < 1.0
    ):
        raise ValueError(
            "stop_loss_pct must be between 0 and 1"
        )

    if not (
        0.0 < args.take_profit_pct < 1.0
    ):
        raise ValueError(
            "take_profit_pct must be between 0 and 1"
        )

    if not (
        0.0 < args.risk_fraction <= 1.0
    ):
        raise ValueError(
            "risk_fraction must be between 0 and 1"
        )

    if args.max_holding_minutes < 1:
        raise ValueError(
            "max_holding_minutes must be positive"
        )

    if not (
        0.0 <= args.long_rsi_min <= 100.0
    ):
        raise ValueError(
            "Invalid long_rsi_min"
        )

    if not (
        0.0 <= args.short_rsi_max <= 100.0
    ):
        raise ValueError(
            "Invalid short_rsi_max"
        )

    if args.minimum_history < (
        max(
            args.slow_ema,
            args.rsi_period
        ) + 1
    ):
        raise ValueError(
            "minimum_history is too short"
        )


def generate_signals(
    candles,
    engine_module,
    args
):
    closes = [
        candle.close
        for candle in candles
    ]

    fast = ema_series(
        closes,
        args.fast_ema
    )

    slow = ema_series(
        closes,
        args.slow_ema
    )

    rsi = rsi_series(
        closes,
        args.rsi_period
    )

    signals = []

    statistics = {
        "eligible_candles": 0,
        "long_signals": 0,
        "short_signals": 0,
        "wait_candles": 0,
    }

    # At index i, calculations use closes[0:i+1].
    # The signal is available only after
    # candle i has completed.
    #
    # V24.5 executes it at candle i+1 open.

    for index in range(
        args.minimum_history,
        len(candles) - 1
    ):
        current_rsi = rsi[index]

        if current_rsi is None:
            continue

        statistics["eligible_candles"] += 1

        direction = "WAIT"

        bullish_cross = (
            fast[index - 1]
            <= slow[index - 1]
            and fast[index] > slow[index]
        )

        bearish_cross = (
            fast[index - 1]
            >= slow[index - 1]
            and fast[index] < slow[index]
        )

        if (
            bullish_cross
            and current_rsi
            >= args.long_rsi_min
        ):
            direction = "LONG"

        elif (
            args.allow_short
            and bearish_cross
            and current_rsi
            <= args.short_rsi_max
        ):
            direction = "SHORT"

        if direction == "WAIT":
            statistics["wait_candles"] += 1
            continue

        signal = engine_module.ResearchSignal(
            timestamp=candles[index].timestamp,
            direction=direction,
            stop_loss_pct=args.stop_loss_pct,
            take_profit_pct=args.take_profit_pct,
            max_holding_minutes=(
                args.max_holding_minutes
            ),
            risk_fraction=args.risk_fraction,
        )

        signals.append(signal)

        if direction == "LONG":
            statistics["long_signals"] += 1
        else:
            statistics["short_signals"] += 1

    return signals, statistics


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "V24.6 BTC historical research backtest"
        )
    )

    parser.add_argument(
        "--candles",
        required=True,
        help=(
            "BTC-USD 1-minute CSV containing "
            "timestamp,open,high,low,close,volume"
        )
    )

    parser.add_argument(
        "--output",
        default=(
            "ai-research/v246/"
            "historical_backtest_report.json"
        )
    )

    parser.add_argument(
        "--fast-ema",
        type=int,
        default=12
    )

    parser.add_argument(
        "--slow-ema",
        type=int,
        default=26
    )

    parser.add_argument(
        "--rsi-period",
        type=int,
        default=14
    )

    parser.add_argument(
        "--long-rsi-min",
        type=float,
        default=50.0
    )

    parser.add_argument(
        "--short-rsi-max",
        type=float,
        default=50.0
    )

    parser.add_argument(
        "--allow-short",
        action="store_true",
        help=(
            "Enable hypothetical SHORT signals. "
            "Not a futures execution model."
        )
    )

    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=0.02
    )

    parser.add_argument(
        "--max-holding-minutes",
        type=int,
        default=60
    )

    parser.add_argument(
        "--risk-fraction",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--minimum-history",
        type=int,
        default=100
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    validate_parameters(args)

    engine_module = load_engine_module()

    config = engine_module.load_config(
        CONFIG_PATH
    )

    engine = engine_module.BTCBacktestEngine(
        config
    )

    candles_path = Path(
        args.candles
    )

    if not candles_path.is_file():
        raise FileNotFoundError(
            f"Candle CSV not found: {candles_path}"
        )

    candles = engine_module.load_candles_csv(
        candles_path
    )

    if len(candles) < (
        args.minimum_history + 2
    ):
        raise ValueError(
            "Not enough candles for the "
            "requested indicator history"
        )

    signals, signal_statistics = (
        generate_signals(
            candles,
            engine_module,
            args
        )
    )

    report = engine.run(
        candles,
        signals
    )

    report["runner_version"] = VERSION

    report["runner_status"] = STATUS

    report["research_only"] = True
    report["trading_enabled"] = False
    report["model_approved"] = False
    report["execution_ready"] = False

    report["data_summary"] = {
        "source_file": candles_path.name,
        "candles": len(candles),
        "first_timestamp":
            candles[0].timestamp,
        "last_timestamp":
            candles[-1].timestamp,
        "interval_minutes": 1,
        "data_origin_verified": False,
        "note": (
            "CSV provenance must be independently "
            "verified before research conclusions."
        ),
    }

    report["signal_generation"] = {
        "method": "EMA_CROSS_WITH_RSI_FILTER",
        "fast_ema": args.fast_ema,
        "slow_ema": args.slow_ema,
        "rsi_period": args.rsi_period,
        "long_rsi_min": args.long_rsi_min,
        "short_rsi_max": args.short_rsi_max,
        "allow_short": args.allow_short,
        "stop_loss_pct": args.stop_loss_pct,
        "take_profit_pct": args.take_profit_pct,
        "max_holding_minutes":
            args.max_holding_minutes,
        "risk_fraction": args.risk_fraction,
        "minimum_history": args.minimum_history,
        **signal_statistics,
    }

    report["research_limitations"] = [
        "Baseline strategy only.",
        "No model training or approval.",
        "No independent holdout evaluation.",
        "No walk-forward validation.",
        "No proof of signal profitability.",
        "No proof of realistic live fills.",
        "Spot candles are not futures prices.",
        "Intrabar price sequence is unknown.",
        "Fees and slippage are assumptions.",
        "No live orders or exchange connection.",
    ]

    output_path = Path(
        args.output
    )

    engine_module.save_report(
        report,
        output_path
    )

    summary = {
        "version": VERSION,
        "status": STATUS,
        "research_only": True,
        "trading_enabled": False,
        "input_candles": len(candles),
        "generated_signals": len(signals),
        "accepted_signals":
            report["accepted_signals"],
        "rejected_signals":
            report["rejected_signals"],
        "closed_trades":
            report["closed_trades"],
        "net_pnl_usdt":
            report["net_pnl_usdt"],
        "return_pct":
            report["return_pct"],
        "max_drawdown_pct":
            report["max_drawdown_pct"],
        "report_path": str(output_path),
        "warning": (
            "Research-only baseline. "
            "Not independently validated."
        ),
    }

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
