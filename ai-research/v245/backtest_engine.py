"""
V24.5 - BTC Research Backtest Engine

BTC-only historical research simulator.

STRICT SAFETY:
- NO exchange connection
- NO real orders
- NO live trading
- NO trading approval
- NO automatic signal generation

Features:
- One-minute OHLC candles
- Historical LONG / SHORT simulation
- Multiple hypothetical positions
- Stop loss and take profit
- Maximum holding time
- Fees and slippage
- Conservative intrabar handling
- Portfolio risk limits
- Daily loss lock
- Research-only reporting

IMPORTANT:
Signals must be generated without future-data leakage.
This module does not generate signals.

All entries and exits are hypothetical.
"""

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


VERSION = "V24.5"

STATUS = "RESEARCH_ONLY_NOT_APPROVED"

CONFIG_PATH = Path(__file__).with_name(
    "research_config.json"
)


def valid_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def positive(name, value):
    if not valid_number(value) or value <= 0:
        raise ValueError(
            f"{name} must be positive and finite"
        )


def nonnegative(name, value):
    if not valid_number(value) or value < 0:
        raise ValueError(
            f"{name} must be nonnegative and finite"
        )


def integer_at_least(name, value, minimum):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer >= {minimum}"
        )


def load_config(path=CONFIG_PATH):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    verify_safety(config)

    return config


def verify_safety(config):
    if not isinstance(config, dict):
        raise ValueError("Invalid configuration")

    if config.get("status") != STATUS:
        raise ValueError(
            "Research status lock failed"
        )

    safety = config.get("safety", {})

    required = {
        "research_only": True,
        "trading_enabled": False,
        "live_orders_enabled": False,
        "exchange_order_access": False,
        "model_approved": False,
    }

    for key, expected in required.items():
        if safety.get(key) is not expected:
            raise ValueError(
                f"Safety lock failed: {key}"
            )

    market = config.get("market", {})

    if market.get("symbol") != "BTC":
        raise ValueError(
            "V24.5 supports BTC only"
        )

    return True


@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self):
        integer_at_least(
            "timestamp",
            self.timestamp,
            0
        )

        for name in (
            "open",
            "high",
            "low",
            "close"
        ):
            positive(
                name,
                getattr(self, name)
            )

        nonnegative(
            "volume",
            self.volume
        )

        if self.high < max(
            self.open,
            self.close,
            self.low
        ):
            raise ValueError(
                "Invalid candle high"
            )

        if self.low > min(
            self.open,
            self.close,
            self.high
        ):
            raise ValueError(
                "Invalid candle low"
            )


@dataclass(frozen=True)
class ResearchSignal:
    timestamp: int
    direction: str
    stop_loss_pct: float
    take_profit_pct: float
    max_holding_minutes: int
    risk_fraction: float

    def __post_init__(self):
        integer_at_least(
            "signal timestamp",
            self.timestamp,
            0
        )

        if self.direction not in (
            "LONG",
            "SHORT"
        ):
            raise ValueError(
                "Direction must be LONG or SHORT"
            )

        positive(
            "stop_loss_pct",
            self.stop_loss_pct
        )

        positive(
            "take_profit_pct",
            self.take_profit_pct
        )

        integer_at_least(
            "max_holding_minutes",
            self.max_holding_minutes,
            1
        )

        positive(
            "risk_fraction",
            self.risk_fraction
        )

        if self.risk_fraction > 1:
            raise ValueError(
                "risk_fraction cannot exceed 1"
            )


@dataclass
class SimulatedPosition:
    position_id: int
    direction: str
    entry_timestamp: int
    entry_price: float
    quantity_btc: float
    stop_price: float
    target_price: float
    expiry_timestamp: int
    entry_fee: float
    entry_notional: float
    planned_risk: float


class BTCBacktestEngine:

    def __init__(self, config):
        verify_safety(config)

        self.config = config

        portfolio = config.get(
            "portfolio",
            {}
        )

        risk = config.get(
            "risk_management",
            {}
        )

        execution = config.get(
            "backtest",
            {}
        )

        self.initial_capital = float(
            portfolio.get(
                "initial_capital_usdt",
                1000.0
            )
        )

        positive(
            "initial_capital",
            self.initial_capital
        )

        self.max_positions = risk.get(
            "maximum_simultaneous_positions",
            3
        )

        integer_at_least(
            "maximum_simultaneous_positions",
            self.max_positions,
            1
        )

        self.max_risk_fraction = float(
            risk.get(
                "max_total_planned_risk_fraction",
                0.03
            )
        )

        self.max_gross_ratio = float(
            risk.get(
                "max_gross_notional_to_equity",
                1.0
            )
        )

        self.daily_loss_fraction = float(
            risk.get(
                "maximum_daily_loss_fraction",
                0.03
            )
        )

        self.fee_rate = float(
            execution.get(
                "fee_rate_per_side",
                0.001
            )
        )

        self.slippage_rate = float(
            execution.get(
                "slippage_rate_per_side",
                0.0005
            )
        )

        self.minimum_equity = float(
            execution.get(
                "minimum_equity_usdt",
                1.0
            )
        )

        for name, value in (
            (
                "max_risk_fraction",
                self.max_risk_fraction
            ),
            (
                "max_gross_ratio",
                self.max_gross_ratio
            ),
            (
                "daily_loss_fraction",
                self.daily_loss_fraction
            ),
            (
                "minimum_equity",
                self.minimum_equity
            )
        ):
            positive(name, value)

        if self.max_risk_fraction > 1:
            raise ValueError(
                "Portfolio risk fraction exceeds 1"
            )

        if self.daily_loss_fraction > 1:
            raise ValueError(
                "Daily loss fraction exceeds 1"
            )

        nonnegative(
            "fee_rate",
            self.fee_rate
        )

        nonnegative(
            "slippage_rate",
            self.slippage_rate
        )

        if self.fee_rate >= 1:
            raise ValueError(
                "Invalid fee rate"
            )

        if self.slippage_rate >= 1:
            raise ValueError(
                "Invalid slippage rate"
            )

        self.reset()

    def reset(self):
        self.cash_equity = (
            self.initial_capital
        )

        self.positions = []

        self.closed_trades = []

        self.equity_curve = []

        self.next_position_id = 1

        self.rejected_signals = 0

        self.accepted_signals = 0

        self.daily_start_equity = {}

        self.locked_days = set()

        self.last_timestamp = None

    @staticmethod
    def day_key(timestamp):
        return datetime.fromtimestamp(
            timestamp,
            tz=timezone.utc
        ).strftime("%Y-%m-%d")

    def current_equity(self, price):
        positive("price", price)

        unrealized = 0.0

        for position in self.positions:
            sign = (
                1
                if position.direction == "LONG"
                else -1
            )

            unrealized += (
                sign
                * (
                    price
                    - position.entry_price
                )
                * position.quantity_btc
            )

        return (
            self.cash_equity
            + unrealized
        )

    def gross_notional(self, price):
        return sum(
            price * p.quantity_btc
            for p in self.positions
        )

    def total_planned_risk(self):
        return sum(
            p.planned_risk
            for p in self.positions
        )

    def apply_slippage(
        self,
        price,
        direction,
        entering
    ):
        positive("price", price)

        if direction == "LONG":
            sign = (
                1 if entering else -1
            )
        else:
            sign = (
                -1 if entering else 1
            )

        return price * (
            1 + sign * self.slippage_rate
        )

    def exit_position(
        self,
        position,
        timestamp,
        raw_exit_price,
        reason
    ):
        exit_price = self.apply_slippage(
            raw_exit_price,
            position.direction,
            entering=False
        )

        sign = (
            1
            if position.direction == "LONG"
            else -1
        )

        gross_pnl = (
            sign
            * (
                exit_price
                - position.entry_price
            )
            * position.quantity_btc
        )

        exit_notional = (
            exit_price
            * position.quantity_btc
        )

        exit_fee = (
            exit_notional
            * self.fee_rate
        )

        self.cash_equity += (
            gross_pnl
            - exit_fee
        )

        net_pnl = (
            gross_pnl
            - position.entry_fee
            - exit_fee
        )

        trade = {
            "position_id":
                position.position_id,
            "direction":
                position.direction,
            "entry_timestamp":
                position.entry_timestamp,
            "exit_timestamp":
                timestamp,
            "entry_price":
                round(
                    position.entry_price,
                    8
                ),
            "exit_price":
                round(exit_price, 8),
            "quantity_btc":
                position.quantity_btc,
            "gross_pnl_usdt":
                round(gross_pnl, 6),
            "entry_fee_usdt":
                round(
                    position.entry_fee,
                    6
                ),
            "exit_fee_usdt":
                round(exit_fee, 6),
            "net_pnl_usdt":
                round(net_pnl, 6),
            "exit_reason":
                reason,
            "research_only": True,
        }

        self.closed_trades.append(
            trade
        )

        self.positions.remove(
            position
        )

    def check_exits(self, candle):
        for position in list(
            self.positions
        ):
            stop_hit = False
            target_hit = False

            if position.direction == "LONG":
                stop_hit = (
                    candle.low
                    <= position.stop_price
                )

                target_hit = (
                    candle.high
                    >= position.target_price
                )

            else:
                stop_hit = (
                    candle.high
                    >= position.stop_price
                )

                target_hit = (
                    candle.low
                    <= position.target_price
                )

            # Conservative handling:
            # If both stop and target are touched
            # in the same minute, assume STOP first.

            if stop_hit:
                self.exit_position(
                    position,
                    candle.timestamp,
                    position.stop_price,
                    "STOP_LOSS"
                )

            elif target_hit:
                self.exit_position(
                    position,
                    candle.timestamp,
                    position.target_price,
                    "TAKE_PROFIT"
                )

            elif (
                candle.timestamp
                >= position.expiry_timestamp
            ):
                self.exit_position(
                    position,
                    candle.timestamp,
                    candle.close,
                    "TIME_EXIT"
                )

    def daily_risk_check(
        self,
        candle
    ):
        day = self.day_key(
            candle.timestamp
        )

        equity = self.current_equity(
            candle.close
        )

        if day not in (
            self.daily_start_equity
        ):
            self.daily_start_equity[
                day
            ] = equity

        starting_equity = (
            self.daily_start_equity[
                day
            ]
        )

        limit = (
            starting_equity
            * self.daily_loss_fraction
        )

        loss = (
            starting_equity
            - equity
        )

        if loss >= limit:
            self.locked_days.add(
                day
            )

        return (
            day in self.locked_days
        )

    def try_open_position(
        self,
        signal,
        candle
    ):
        day = self.day_key(
            candle.timestamp
        )

        if day in self.locked_days:
            self.rejected_signals += 1
            return

        if len(
            self.positions
        ) >= self.max_positions:
            self.rejected_signals += 1
            return

        equity = self.current_equity(
            candle.close
        )

        if equity <= (
            self.minimum_equity
        ):
            self.rejected_signals += 1
            return

        # Signal is considered available
        # at the close of this candle.
        #
        # Entry happens at NEXT candle open.
        # The caller must provide that candle.

        raise RuntimeError(
            "Use enter_at_next_open()"
        )

    def enter_at_next_open(
        self,
        signal,
        candle
    ):
        day = self.day_key(
            candle.timestamp
        )

        if day in self.locked_days:
            self.rejected_signals += 1
            return

        if len(
            self.positions
        ) >= self.max_positions:
            self.rejected_signals += 1
            return

        equity = self.current_equity(
            candle.open
        )

        if equity <= (
            self.minimum_equity
        ):
            self.rejected_signals += 1
            return

        entry_price = (
            self.apply_slippage(
                candle.open,
                signal.direction,
                entering=True
            )
        )

        if signal.direction == "LONG":
            stop_price = (
                entry_price
                * (
                    1
                    - signal.stop_loss_pct
                )
            )

            target_price = (
                entry_price
                * (
                    1
                    + signal.take_profit_pct
                )
            )

        else:
            stop_price = (
                entry_price
                * (
                    1
                    + signal.stop_loss_pct
                )
            )

            target_price = (
                entry_price
                * (
                    1
                    - signal.take_profit_pct
                )
            )

        if target_price <= 0:
            self.rejected_signals += 1
            return

        risk_budget = (
            equity
            * signal.risk_fraction
        )

        stop_distance = abs(
            entry_price
            - stop_price
        )

        estimated_cost_per_btc = (
            entry_price
            * (
                2 * self.fee_rate
                + 2 * self.slippage_rate
            )
        )

        risk_per_btc = (
            stop_distance
            + estimated_cost_per_btc
        )

        if risk_per_btc <= 0:
            self.rejected_signals += 1
            return

        quantity = (
            risk_budget
            / risk_per_btc
        )

        planned_risk = (
            quantity
            * risk_per_btc
        )

        existing_risk = (
            self.total_planned_risk()
        )

        portfolio_risk_limit = (
            equity
            * self.max_risk_fraction
        )

        if (
            existing_risk
            + planned_risk
            > portfolio_risk_limit
            + 1e-9
        ):
            self.rejected_signals += 1
            return

        existing_gross = (
            self.gross_notional(
                candle.open
            )
        )

        new_gross = (
            existing_gross
            + entry_price
            * quantity
        )

        gross_limit = (
            equity
            * self.max_gross_ratio
        )

        if new_gross > (
            gross_limit + 1e-9
        ):
            self.rejected_signals += 1
            return

        entry_notional = (
            entry_price
            * quantity
        )

        entry_fee = (
            entry_notional
            * self.fee_rate
        )

        position = SimulatedPosition(
            position_id=(
                self.next_position_id
            ),
            direction=(
                signal.direction
            ),
            entry_timestamp=(
                candle.timestamp
            ),
            entry_price=(
                entry_price
            ),
            quantity_btc=quantity,
            stop_price=stop_price,
            target_price=target_price,
            expiry_timestamp=(
                candle.timestamp
                + (
                    signal.max_holding_minutes
                    * 60
                )
            ),
            entry_fee=entry_fee,
            entry_notional=(
                entry_notional
            ),
            planned_risk=(
                planned_risk
            ),
        )

        self.cash_equity -= (
            entry_fee
        )

        self.positions.append(
            position
        )

        self.next_position_id += 1

        self.accepted_signals += 1

    def run(
        self,
        candles,
        signals
    ):
        self.reset()

        if not isinstance(
            candles,
            list
        ):
            raise ValueError(
                "candles must be a list"
            )

        if not isinstance(
            signals,
            list
        ):
            raise ValueError(
                "signals must be a list"
            )

        if len(candles) < 2:
            raise ValueError(
                "At least two candles required"
            )

        for candle in candles:
            if not isinstance(
                candle,
                Candle
            ):
                raise ValueError(
                    "Invalid candle object"
                )

        for signal in signals:
            if not isinstance(
                signal,
                ResearchSignal
            ):
                raise ValueError(
                    "Invalid signal object"
                )

        for i in range(
            1,
            len(candles)
        ):
            if (
                candles[i].timestamp
                - candles[i - 1].timestamp
                != 60
            ):
                raise ValueError(
                    "Candles must be contiguous "
                    "one-minute data"
                )

        signal_map = {}

        for signal in signals:
            signal_map.setdefault(
                signal.timestamp,
                []
            ).append(signal)

        # A signal timestamp means:
        # signal becomes known after that
        # candle has closed.
        #
        # The next candle is used for entry.

        for index, candle in enumerate(
            candles
        ):
            self.last_timestamp = (
                candle.timestamp
            )

            # First process positions
            # already open before this candle.

            self.check_exits(
                candle
            )

            self.daily_risk_check(
                candle
            )

            if index > 0:
                previous = candles[
                    index - 1
                ]

                pending = signal_map.get(
                    previous.timestamp,
                    []
                )

                for signal in pending:
                    self.enter_at_next_open(
                        signal,
                        candle
                    )

            equity = (
                self.current_equity(
                    candle.close
                )
            )

            self.equity_curve.append({
                "timestamp":
                    candle.timestamp,
                "equity_usdt":
                    round(equity, 6),
                "open_positions":
                    len(self.positions),
            })

        # Close remaining hypothetical
        # positions at final candle close.

        last = candles[-1]

        for position in list(
            self.positions
        ):
            self.exit_position(
                position,
                last.timestamp,
                last.close,
                "END_OF_DATA"
            )

        return self.report()

    def report(self):
        total_trades = len(
            self.closed_trades
        )

        wins = [
            t
            for t in self.closed_trades
            if t["net_pnl_usdt"] > 0
        ]

        losses = [
            t
            for t in self.closed_trades
            if t["net_pnl_usdt"] < 0
        ]

        gross_profit = sum(
            t["net_pnl_usdt"]
            for t in wins
        )

        gross_loss = abs(
            sum(
                t["net_pnl_usdt"]
                for t in losses
            )
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else None
        )

        win_rate = (
            len(wins)
            / total_trades
            * 100
            if total_trades
            else 0.0
        )

        peak = (
            self.initial_capital
        )

        max_drawdown = 0.0

        for item in self.equity_curve:
            equity = item[
                "equity_usdt"
            ]

            peak = max(
                peak,
                equity
            )

            if peak > 0:
                drawdown = (
                    peak - equity
                ) / peak

                max_drawdown = max(
                    max_drawdown,
                    drawdown
                )

        final_equity = (
            self.cash_equity
        )

        net_pnl = (
            final_equity
            - self.initial_capital
        )

        return {
            "version": VERSION,
            "status": STATUS,
            "symbol": "BTC",
            "research_only": True,
            "trading_enabled": False,
            "model_approved": False,
            "execution_ready": False,
            "initial_capital_usdt":
                round(
                    self.initial_capital,
                    2
                ),
            "final_equity_usdt":
                round(
                    final_equity,
                    2
                ),
            "net_pnl_usdt":
                round(net_pnl, 2),
            "return_pct":
                round(
                    net_pnl
                    / self.initial_capital
                    * 100,
                    4
                ),
            "closed_trades":
                total_trades,
            "winning_trades":
                len(wins),
            "losing_trades":
                len(losses),
            "win_rate_pct":
                round(win_rate, 2),
            "profit_factor":
                (
                    round(
                        profit_factor,
                        4
                    )
                    if profit_factor is not None
                    else None
                ),
            "max_drawdown_pct":
                round(
                    max_drawdown * 100,
                    4
                ),
            "accepted_signals":
                self.accepted_signals,
            "rejected_signals":
                self.rejected_signals,
            "locked_days":
                sorted(
                    self.locked_days
                ),
            "trades":
                self.closed_trades,
            "equity_curve":
                self.equity_curve,
            "limitations": [
                "Historical simulation only",
                "No real exchange execution",
                "No liquidation modeling",
                "No funding modeling",
                "Intrabar stop assumed first",
                "Signal quality not validated",
                "No guarantee of profitability"
            ],
        }


def load_candles_csv(path):
    candles = []

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        if not required.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "Missing candle CSV columns"
            )

        for row in reader:
            candles.append(
                Candle(
                    timestamp=int(
                        row["timestamp"]
                    ),
                    open=float(
                        row["open"]
                    ),
                    high=float(
                        row["high"]
                    ),
                    low=float(
                        row["low"]
                    ),
                    close=float(
                        row["close"]
                    ),
                    volume=float(
                        row["volume"]
                    ),
                )
            )

    return candles


def load_signals_csv(path):
    signals = []

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "timestamp",
            "direction",
            "stop_loss_pct",
            "take_profit_pct",
            "max_holding_minutes",
            "risk_fraction",
        }

        if not required.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "Missing signal CSV columns"
            )

        for row in reader:
            signals.append(
                ResearchSignal(
                    timestamp=int(
                        row["timestamp"]
                    ),
                    direction=(
                        row["direction"]
                    ),
                    stop_loss_pct=float(
                        row["stop_loss_pct"]
                    ),
                    take_profit_pct=float(
                        row["take_profit_pct"]
                    ),
                    max_holding_minutes=int(
                        row[
                            "max_holding_minutes"
                        ]
                    ),
                    risk_fraction=float(
                        row["risk_fraction"]
                    ),
                )
            )

    return signals


def save_report(report, path):
    output = Path(path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )


def main():
    print(
        "V24.5 BTC Research Backtest Engine"
    )

    print(
        "Status:",
        STATUS
    )

    print(
        "No live trading."
    )

    print(
        "This module requires historical "
        "candles and precomputed signals."
    )


if __name__ == "__main__":
    main()
