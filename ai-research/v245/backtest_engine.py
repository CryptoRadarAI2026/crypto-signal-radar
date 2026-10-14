"""
V24.5 - BTC Research Backtest Engine
Corrected historical simulation engine.

RESEARCH ONLY.
NO EXCHANGE CONNECTION.
NO REAL ORDERS.
NO LIVE TRADING.
NO TRADING APPROVAL.

Important limitations:
- Historical BTC-USD spot candles are used as a proxy.
- Futures liquidation is NOT modeled.
- Funding is NOT modeled.
- Intrabar price sequence is unknown.
- Signals must be computed without future leakage.
- No model profitability or execution approval is implied.
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


def is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def require_number(name, value, minimum=None):
    if not is_number(value):
        raise ValueError(
            f"{name} must be a finite number"
        )

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be >= {minimum}"
        )


def require_positive(name, value):
    require_number(name, value)

    if value <= 0:
        raise ValueError(
            f"{name} must be positive"
        )


def require_fraction(name, value, allow_zero=False):
    require_number(name, value)

    lower = 0 if allow_zero else 0

    if allow_zero:
        valid = lower <= value <= 1
    else:
        valid = lower < value <= 1

    if not valid:
        raise ValueError(
            f"{name} must be between 0 and 1"
        )


def require_integer(name, value, minimum=0):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer >= {minimum}"
        )


def required_section(config, name):
    section = config.get(name)

    if not isinstance(section, dict):
        raise ValueError(
            f"Missing or invalid section: {name}"
        )

    return section


def required_value(section, key):
    if key not in section:
        raise ValueError(
            f"Missing required setting: {key}"
        )

    return section[key]


def verify_safety(config):
    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must be an object"
        )

    if config.get("status") != STATUS:
        raise ValueError(
            "Research status lock failed"
        )

    safety = required_section(
        config,
        "safety"
    )

    required_safety = {
        "research_only": True,
        "trading_enabled": False,
        "live_orders_enabled": False,
        "exchange_order_access": False,
        "model_approved": False,
        "allow_real_capital": False,
    }

    for key, expected in required_safety.items():
        if safety.get(key) is not expected:
            raise ValueError(
                f"Research safety lock failed: {key}"
            )

    approval = required_section(
        config,
        "approval"
    )

    required_approval = {
        "automatic_approval": False,
        "trading_approval": False,
        "execution_ready": False,
    }

    for key, expected in required_approval.items():
        if approval.get(key) is not expected:
            raise ValueError(
                f"Approval lock failed: {key}"
            )

    market = required_section(
        config,
        "market"
    )

    if market.get("symbol") != "BTC":
        raise ValueError(
            "Only BTC is supported"
        )

    if market.get("reference_pair") != "BTC-USD":
        raise ValueError(
            "Expected BTC-USD reference data"
        )

    if market.get("candle_interval_minutes") != 1:
        raise ValueError(
            "Only one-minute candles are supported"
        )

    execution = required_section(
        config,
        "execution_assumptions"
    )

    if execution.get("mode") != (
        "historical_simulation_only"
    ):
        raise ValueError(
            "Historical simulation lock failed"
        )

    if execution.get(
        "signal_at_completed_candle"
    ) is not True:
        raise ValueError(
            "Signals must use completed candles"
        )

    if execution.get("earliest_entry") != (
        "next_candle_open"
    ):
        raise ValueError(
            "Only next-candle-open entry is supported"
        )

    if execution.get(
        "same_candle_stop_and_target_policy"
    ) != "stop_first":
        raise ValueError(
            "Only stop-first ambiguity policy supported"
        )

    if execution.get(
        "gap_through_stop_policy"
    ) != "adverse_open_with_slippage":
        raise ValueError(
            "Unsupported stop gap policy"
        )

    if execution.get(
        "allow_execution_ready_claim"
    ) is not False:
        raise ValueError(
            "Execution-ready claims must be disabled"
        )

    risk = required_section(
        config,
        "risk_management"
    )

    if risk.get(
        "allow_widening_protective_stop"
    ) is not False:
        raise ValueError(
            "Stop widening must remain disabled"
        )

    if risk.get(
        "emergency_stop_enabled"
    ) is not True:
        raise ValueError(
            "Emergency risk lock must remain enabled"
        )

    return True


def load_config(path=CONFIG_PATH):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    verify_safety(config)

    return config


@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self):
        require_integer(
            "timestamp",
            self.timestamp
        )

        for name in (
            "open",
            "high",
            "low",
            "close"
        ):
            require_positive(
                name,
                getattr(self, name)
            )

        require_number(
            "volume",
            self.volume,
            minimum=0
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
        require_integer(
            "signal timestamp",
            self.timestamp
        )

        if self.direction not in (
            "LONG",
            "SHORT",
            "WAIT"
        ):
            raise ValueError(
                "Invalid signal direction"
            )

        require_fraction(
            "stop_loss_pct",
            self.stop_loss_pct
        )

        require_fraction(
            "take_profit_pct",
            self.take_profit_pct
        )

        require_integer(
            "max_holding_minutes",
            self.max_holding_minutes,
            minimum=1
        )

        require_fraction(
            "risk_fraction",
            self.risk_fraction
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
    entry_fee_usdt: float
    planned_risk_usdt: float


class BTCBacktestEngine:

    def __init__(self, config):
        verify_safety(config)

        self.config = config

        portfolio = required_section(
            config,
            "portfolio"
        )

        risk = required_section(
            config,
            "risk_management"
        )

        execution = required_section(
            config,
            "execution_assumptions"
        )

        management = required_section(
            config,
            "position_management"
        )

        self.initial_capital = required_value(
            portfolio,
            "initial_capital_usdt"
        )

        self.reserve_fraction = required_value(
            portfolio,
            "reserve_capital_fraction"
        )

        self.risk_per_position = required_value(
            risk,
            "risk_per_position_fraction"
        )

        self.max_total_risk = required_value(
            risk,
            "max_total_planned_risk_fraction"
        )

        self.max_daily_loss = required_value(
            risk,
            "max_daily_loss_fraction"
        )

        self.max_gross_ratio = required_value(
            risk,
            "max_gross_notional_to_equity"
        )

        self.max_leverage = required_value(
            risk,
            "maximum_simulated_leverage"
        )

        self.fee_rate = required_value(
            execution,
            "market_order_fee_fraction_per_side"
        )

        self.slippage_rate = required_value(
            execution,
            "base_slippage_fraction_per_side"
        )

        self.stress_slippage_rate = required_value(
            execution,
            "stress_slippage_fraction_per_side"
        )

        self.max_holding_minutes = required_value(
            management,
            "maximum_holding_minutes"
        )

        self.position_count_limit = portfolio.get(
            "fixed_position_count_limit"
        )

        require_positive(
            "initial_capital",
            self.initial_capital
        )

        require_fraction(
            "reserve_capital_fraction",
            self.reserve_fraction,
            allow_zero=True
        )

        if self.reserve_fraction >= 1:
            raise ValueError(
                "Reserve must leave usable capital"
            )

        for name, value in (
            (
                "risk_per_position_fraction",
                self.risk_per_position
            ),
            (
                "max_total_planned_risk_fraction",
                self.max_total_risk
            ),
            (
                "max_daily_loss_fraction",
                self.max_daily_loss
            )
        ):
            require_fraction(name, value)

        require_positive(
            "max_gross_notional_to_equity",
            self.max_gross_ratio
        )

        require_positive(
            "maximum_simulated_leverage",
            self.max_leverage
        )

        for name, value in (
            ("fee_rate", self.fee_rate),
            ("slippage_rate", self.slippage_rate),
            (
                "stress_slippage_rate",
                self.stress_slippage_rate
            )
        ):
            require_fraction(
                name,
                value,
                allow_zero=True
            )

            if value >= 1:
                raise ValueError(
                    f"{name} must be below 1"
                )

        if (
            self.stress_slippage_rate
            < self.slippage_rate
        ):
            raise ValueError(
                "Stress slippage cannot be lower "
                "than base slippage"
            )

        require_integer(
            "maximum_holding_minutes",
            self.max_holding_minutes,
            minimum=1
        )

        if self.position_count_limit is not None:
            require_integer(
                "fixed_position_count_limit",
                self.position_count_limit,
                minimum=1
            )

        if (
            self.risk_per_position
            > self.max_total_risk
        ):
            raise ValueError(
                "Per-position risk exceeds total risk"
            )

        self.reset()

    def reset(self):
        self.cash_equity = float(
            self.initial_capital
        )

        self.positions = []
        self.trades = []
        self.equity_curve = []

        self.next_position_id = 1

        self.accepted_signals = 0
        self.rejected_signals = 0
        self.wait_signals = 0

        self.rejection_reasons = {}

        self.locked_days = set()
        self.daily_start_equity = {}

        self.total_entry_fees = 0.0
        self.total_exit_fees = 0.0

        self.max_open_positions = 0

    @staticmethod
    def day_key(timestamp):
        return datetime.fromtimestamp(
            timestamp,
            tz=timezone.utc
        ).strftime("%Y-%m-%d")

    def reject(self, reason):
        self.rejected_signals += 1

        self.rejection_reasons[reason] = (
            self.rejection_reasons.get(
                reason,
                0
            ) + 1
        )

    def position_pnl(self, position, price):
        sign = (
            1
            if position.direction == "LONG"
            else -1
        )

        return (
            sign
            * (price - position.entry_price)
            * position.quantity_btc
        )

    def equity(self, price):
        return (
            self.cash_equity
            + sum(
                self.position_pnl(p, price)
                for p in self.positions
            )
        )

    def gross_notional(self, price):
        return sum(
            price * p.quantity_btc
            for p in self.positions
        )

    def planned_risk(self):
        return sum(
            p.planned_risk_usdt
            for p in self.positions
        )

    def estimated_margin(self, price):
        # Simplified margin proxy only.
        # NOT an exchange margin model.

        return (
            self.gross_notional(price)
            / self.max_leverage
        )

    def entry_fill(self, price, direction):
        if direction == "LONG":
            return price * (
                1 + self.slippage_rate
            )

        return price * (
            1 - self.slippage_rate
        )

    def exit_fill(
        self,
        price,
        direction,
        stress=False
    ):
        slip = (
            self.stress_slippage_rate
            if stress
            else self.slippage_rate
        )

        if direction == "LONG":
            return price * (1 - slip)

        return price * (1 + slip)

    def start_day(self, candle):
        day = self.day_key(
            candle.timestamp
        )

        if day not in self.daily_start_equity:
            self.daily_start_equity[day] = (
                self.equity(candle.open)
            )

        return day

    def update_daily_lock(self, timestamp, price):
        day = self.day_key(timestamp)

        starting = self.daily_start_equity.get(
            day
        )

        if starting is None:
            return False

        current = self.equity(price)

        if current <= (
            starting
            * (1 - self.max_daily_loss)
        ):
            self.locked_days.add(day)

        return day in self.locked_days

    def open_hypothetical_position(
        self,
        signal,
        candle
    ):
        day = self.day_key(
            candle.timestamp
        )

        if signal.direction == "WAIT":
            self.wait_signals += 1
            return

        if day in self.locked_days:
            self.reject("DAILY_LOSS_LOCK")
            return

        if (
            self.position_count_limit is not None
            and len(self.positions)
            >= self.position_count_limit
        ):
            self.reject("POSITION_COUNT_LIMIT")
            return

        if (
            signal.max_holding_minutes
            > self.max_holding_minutes
        ):
            self.reject("HOLDING_TIME_LIMIT")
            return

        if (
            signal.risk_fraction
            > self.risk_per_position
        ):
            self.reject("PER_POSITION_RISK_LIMIT")
            return

        equity = self.equity(
            candle.open
        )

        if equity <= 0:
            self.reject("NONPOSITIVE_EQUITY")
            return

        entry = self.entry_fill(
            candle.open,
            signal.direction
        )

        if entry <= 0:
            self.reject("INVALID_ENTRY_PRICE")
            return

        if signal.direction == "LONG":
            stop = entry * (
                1 - signal.stop_loss_pct
            )

            target = entry * (
                1 + signal.take_profit_pct
            )

        else:
            stop = entry * (
                1 + signal.stop_loss_pct
            )

            target = entry * (
                1 - signal.take_profit_pct
            )

        if stop <= 0 or target <= 0:
            self.reject("INVALID_STOP_OR_TARGET")
            return

        # Risk sizing includes:
        # - stop distance
        # - estimated entry fee
        # - estimated exit fee
        # - adverse stop slippage stress
        #
        # This is still NOT a guaranteed loss cap.

        stressed_exit = self.exit_fill(
            stop,
            signal.direction,
            stress=True
        )

        loss_per_btc = abs(
            entry - stressed_exit
        )

        fee_per_btc = (
            entry * self.fee_rate
            + stressed_exit * self.fee_rate
        )

        estimated_risk_per_btc = (
            loss_per_btc
            + fee_per_btc
        )

        if estimated_risk_per_btc <= 0:
            self.reject("INVALID_RISK_ESTIMATE")
            return

        risk_budget = (
            equity
            * signal.risk_fraction
        )

        quantity = (
            risk_budget
            / estimated_risk_per_btc
        )

        if quantity <= 0:
            self.reject("INVALID_QUANTITY")
            return

        entry_notional = (
            entry * quantity
        )

        planned_risk = (
            quantity
            * estimated_risk_per_btc
        )

        if (
            self.planned_risk()
            + planned_risk
            > equity * self.max_total_risk
            + 1e-9
        ):
            self.reject("TOTAL_RISK_LIMIT")
            return

        if (
            self.gross_notional(candle.open)
            + entry_notional
            > equity * self.max_gross_ratio
            + 1e-9
        ):
            self.reject("GROSS_NOTIONAL_LIMIT")
            return

        # Simplified capital-reserve constraint.
        # Uses configured maximum leverage only
        # as a research margin approximation.

        projected_margin = (
            self.estimated_margin(candle.open)
            + entry_notional / self.max_leverage
        )

        entry_fee = (
            entry_notional * self.fee_rate
        )

        usable_capital = (
            equity
            * (1 - self.reserve_fraction)
        )

        if (
            projected_margin + entry_fee
            > usable_capital + 1e-9
        ):
            self.reject("CAPITAL_RESERVE_LIMIT")
            return

        position = SimulatedPosition(
            position_id=self.next_position_id,
            direction=signal.direction,
            entry_timestamp=candle.timestamp,
            entry_price=entry,
            quantity_btc=quantity,
            stop_price=stop,
            target_price=target,
            expiry_timestamp=(
                candle.timestamp
                + signal.max_holding_minutes * 60
            ),
            entry_fee_usdt=entry_fee,
            planned_risk_usdt=planned_risk,
        )

        self.cash_equity -= entry_fee

        self.total_entry_fees += entry_fee

        self.positions.append(position)

        self.next_position_id += 1

        self.accepted_signals += 1

        self.max_open_positions = max(
            self.max_open_positions,
            len(self.positions)
        )

    def close_hypothetical_position(
        self,
        position,
        timestamp,
        raw_price,
        reason,
        stress=False
    ):
        fill = self.exit_fill(
            raw_price,
            position.direction,
            stress=stress
        )

        gross_pnl = (
            self.position_pnl(
                position,
                fill
            )
        )

        exit_fee = (
            fill
            * position.quantity_btc
            * self.fee_rate
        )

        net_pnl = (
            gross_pnl
            - position.entry_fee_usdt
            - exit_fee
        )

        self.cash_equity += (
            gross_pnl - exit_fee
        )

        self.total_exit_fees += exit_fee

        self.positions.remove(position)

        self.trades.append({
            "position_id": position.position_id,
            "direction": position.direction,
            "entry_timestamp":
                position.entry_timestamp,
            "exit_timestamp": timestamp,
            "entry_price":
                round(position.entry_price, 8),
            "exit_price":
                round(fill, 8),
            "quantity_btc":
                position.quantity_btc,
            "stop_price":
                round(position.stop_price, 8),
            "target_price":
                round(position.target_price, 8),
            "gross_pnl_usdt":
                round(gross_pnl, 6),
            "entry_fee_usdt":
                round(
                    position.entry_fee_usdt,
                    6
                ),
            "exit_fee_usdt":
                round(exit_fee, 6),
            "net_pnl_usdt":
                round(net_pnl, 6),
            "exit_reason": reason,
            "stress_slippage_applied": stress,
            "research_only": True,
        })

    def check_position_exit(
        self,
        position,
        candle
    ):
        # First check opening-price gaps.
        # An adverse gap beyond the stop
        # exits at the worse opening price.

        if position.direction == "LONG":
            if candle.open <= position.stop_price:
                self.close_hypothetical_position(
                    position,
                    candle.timestamp,
                    candle.open,
                    "STOP_GAP",
                    stress=True
                )
                return

            if candle.open >= position.target_price:
                self.close_hypothetical_position(
                    position,
                    candle.timestamp,
                    position.target_price,
                    "TARGET_AT_OPEN"
                )
                return

            stop_hit = (
                candle.low <= position.stop_price
            )

            target_hit = (
                candle.high >= position.target_price
            )

        else:
            if candle.open >= position.stop_price:
                self.close_hypothetical_position(
                    position,
                    candle.timestamp,
                    candle.open,
                    "STOP_GAP",
                    stress=True
                )
                return

            if candle.open <= position.target_price:
                self.close_hypothetical_position(
                    position,
                    candle.timestamp,
                    position.target_price,
                    "TARGET_AT_OPEN"
                )
                return

            stop_hit = (
                candle.high >= position.stop_price
            )

            target_hit = (
                candle.low <= position.target_price
            )

        # Unknown intrabar sequence:
        # stop always takes priority.

        if stop_hit:
            reason = (
                "STOP_FIRST_AMBIGUOUS"
                if target_hit
                else "STOP_LOSS"
            )

            self.close_hypothetical_position(
                position,
                candle.timestamp,
                position.stop_price,
                reason,
                stress=True
            )
            return

        if target_hit:
            self.close_hypothetical_position(
                position,
                candle.timestamp,
                position.target_price,
                "TAKE_PROFIT"
            )
            return

        # A time exit occurs at candle close,
        # not at an unknown intrabar instant.

        if (
            candle.timestamp + 60
            >= position.expiry_timestamp
        ):
            self.close_hypothetical_position(
                position,
                candle.timestamp + 60,
                candle.close,
                "TIME_EXIT"
            )

    def record_equity(
        self,
        timestamp,
        price
    ):
        self.equity_curve.append({
            "timestamp": timestamp,
            "equity_usdt":
                round(self.equity(price), 6),
            "open_positions":
                len(self.positions),
        })

    def validate_inputs(
        self,
        candles,
        signals
    ):
        if not isinstance(candles, list):
            raise ValueError(
                "Candles must be a list"
            )

        if not isinstance(signals, list):
            raise ValueError(
                "Signals must be a list"
            )

        if len(candles) < 2:
            raise ValueError(
                "At least two candles are required"
            )

        for candle in candles:
            if not isinstance(candle, Candle):
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

        for index in range(
            1,
            len(candles)
        ):
            difference = (
                candles[index].timestamp
                - candles[index - 1].timestamp
            )

            if difference != 60:
                raise ValueError(
                    "Candles must be chronological, "
                    "unique and exactly one minute apart"
                )

        available_times = {
            candle.timestamp
            for candle in candles
        }

        seen_signals = set()

        for signal in signals:
            if signal.timestamp not in available_times:
                raise ValueError(
                    "Signal timestamp not found "
                    "in candle data"
                )

            if signal.timestamp in seen_signals:
                raise ValueError(
                    "Only one signal per candle "
                    "is supported in V24.5"
                )

            seen_signals.add(
                signal.timestamp
            )

    def run(self, candles, signals):
        verify_safety(self.config)

        self.validate_inputs(
            candles,
            signals
        )

        self.reset()

        signal_map = {
            signal.timestamp: signal
            for signal in signals
        }

        # A signal at timestamp T is assumed
        # to become available only AFTER
        # candle T has completed.
        #
        # It may enter at the open of T+60.
        #
        # IMPORTANT:
        # The engine cannot independently prove
        # that external signals avoid lookahead.

        for index, candle in enumerate(candles):
            day = self.start_day(candle)

            # Check daily risk using opening mark.
            # No new entries once locked.

            self.update_daily_lock(
                candle.timestamp,
                candle.open
            )

            # Handle opening gaps for positions
            # that were already open.
            #
            # The complete candle exit check
            # occurs after next-open entries.
            #
            # This ensures new positions are
            # exposed to the entry candle.

            for position in list(self.positions):
                if position.direction == "LONG":
                    gap_stop = (
                        candle.open
                        <= position.stop_price
                    )

                else:
                    gap_stop = (
                        candle.open
                        >= position.stop_price
                    )

                if gap_stop:
                    self.close_hypothetical_position(
                        position,
                        candle.timestamp,
                        candle.open,
                        "STOP_GAP",
                        stress=True
                    )

            self.update_daily_lock(
                candle.timestamp,
                candle.open
            )

            # Enter signals from the PREVIOUS
            # completed candle only.

            if index > 0:
                previous = candles[index - 1]

                signal = signal_map.get(
                    previous.timestamp
                )

                if signal is not None:
                    self.open_hypothetical_position(
                        signal,
                        candle
                    )

            # All positions, including those
            # entered at this candle's open,
            # are now checked against this
            # candle's high/low/close.

            for position in list(self.positions):
                self.check_position_exit(
                    position,
                    candle
                )

            # End-of-candle risk check.
            # Lock blocks subsequent entries.
            # It does NOT imply liquidation.

            self.update_daily_lock(
                candle.timestamp,
                candle.close
            )

            self.record_equity(
                candle.timestamp + 60,
                candle.close
            )

        # A signal on the final candle has
        # no next candle and cannot be filled.

        final_signal = signal_map.get(
            candles[-1].timestamp
        )

        if final_signal is not None:
            if final_signal.direction == "WAIT":
                self.wait_signals += 1
            else:
                self.reject(
                    "NO_NEXT_CANDLE"
                )

        # Mark remaining positions out
        # at the end of historical data.

        last = candles[-1]

        for position in list(self.positions):
            self.close_hypothetical_position(
                position,
                last.timestamp + 60,
                last.close,
                "END_OF_DATA"
            )

        # Ensure the final equity point
        # includes final exit fees/slippage.

        self.record_equity(
            last.timestamp + 60,
            last.close
        )

        return self.report()

    def report(self):
        wins = [
            trade
            for trade in self.trades
            if trade["net_pnl_usdt"] > 0
        ]

        losses = [
            trade
            for trade in self.trades
            if trade["net_pnl_usdt"] < 0
        ]

        total_trades = len(
            self.trades
        )

        total_profit = sum(
            trade["net_pnl_usdt"]
            for trade in wins
        )

        total_loss = abs(
            sum(
                trade["net_pnl_usdt"]
                for trade in losses
            )
        )

        profit_factor = (
            total_profit / total_loss
            if total_loss > 0
            else None
        )

        win_rate = (
            100 * len(wins) / total_trades
            if total_trades
            else 0.0
        )

        peak = float(
            self.initial_capital
        )

        max_drawdown = 0.0

        for point in self.equity_curve:
            value = point["equity_usdt"]

            peak = max(
                peak,
                value
            )

            if peak > 0:
                drawdown = (
                    peak - value
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

        long_trades = [
            trade
            for trade in self.trades
            if trade["direction"] == "LONG"
        ]

        short_trades = [
            trade
            for trade in self.trades
            if trade["direction"] == "SHORT"
        ]

        return {
            "version": VERSION,
            "status": STATUS,
            "symbol": "BTC",
            "reference_pair": "BTC-USD",

            "research_only": True,
            "trading_enabled": False,
            "model_approved": False,
            "execution_ready": False,
            "exchange_connected": False,

            "initial_capital_usdt":
                round(self.initial_capital, 2),

            "final_equity_usdt":
                round(final_equity, 2),

            "net_pnl_usdt":
                round(net_pnl, 2),

            "return_pct":
                round(
                    100
                    * net_pnl
                    / self.initial_capital,
                    4
                ),

            "closed_trades":
                total_trades,

            "long_trades":
                len(long_trades),

            "short_trades":
                len(short_trades),

            "winning_trades":
                len(wins),

            "losing_trades":
                len(losses),

            "win_rate_pct":
                round(win_rate, 2),

            "profit_factor":
                (
                    round(profit_factor, 4)
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

            "wait_signals":
                self.wait_signals,

            "rejection_reasons":
                self.rejection_reasons,

            "max_open_positions":
                self.max_open_positions,

            "locked_days":
                sorted(self.locked_days),

            "total_entry_fees_usdt":
                round(self.total_entry_fees, 6),

            "total_exit_fees_usdt":
                round(self.total_exit_fees, 6),

            "fee_rate_per_side":
                self.fee_rate,

            "base_slippage_per_side":
                self.slippage_rate,

            "stress_slippage_per_side":
                self.stress_slippage_rate,

            "assumptions": {
                "signal_timing":
                    "completed_candle",
                "entry_timing":
                    "next_candle_open",
                "intrabar_ambiguity":
                    "stop_first",
                "stop_gap":
                    "adverse_open",
                "daily_loss_lock":
                    "blocks_new_entries_only",
                "margin":
                    "simplified_proxy",
                "position_count":
                    "risk_and_capital_constrained",
            },

            "not_implemented": [
                "futures_liquidation",
                "funding_payments",
                "exchange_margin_rules",
                "dynamic_trailing_stop",
                "partial_profit_taking",
                "ATR_based_stop_generation",
                "automatic_signal_generation",
                "walk_forward_validation",
                "market_regime_analysis",
                "hedge_vs_net_comparison",
            ],

            "warnings": [
                "Spot candles are not futures fills.",
                "Stops do not guarantee maximum loss.",
                "Intrabar order is not observable.",
                "Slippage assumptions are estimates.",
                "Signals are not leakage-verified.",
                "No profitability guarantee.",
                "Not approved for real trading.",
            ],

            "trades": self.trades,
            "equity_curve": self.equity_curve,
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
                    direction=row[
                        "direction"
                    ],
                    stop_loss_pct=float(
                        row["stop_loss_pct"]
                    ),
                    take_profit_pct=float(
                        row["take_profit_pct"]
                    ),
                    max_holding_minutes=int(
                        row["max_holding_minutes"]
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
    config = load_config()

    engine = BTCBacktestEngine(
        config
    )

    print(
        json.dumps(
            {
                "version": VERSION,
                "status": STATUS,
                "symbol": "BTC",
                "config_valid": True,
                "initial_capital_usdt":
                    engine.initial_capital,
                "fee_rate_per_side":
                    engine.fee_rate,
                "slippage_rate_per_side":
                    engine.slippage_rate,
                "research_only": True,
                "trading_enabled": False,
                "note":
                    "Engine loaded. No historical "
                    "backtest has been executed."
            },
            indent=2
        )
    )


if __name__ == "__main__":
    main()
