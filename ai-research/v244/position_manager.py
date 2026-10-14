"""
V24.4 - BTC Multi-Position Research Ledger

Research-only accounting for hypothetical BTC futures positions.

SAFETY:
- NO EXCHANGE CONNECTIONS
- NO REAL ORDERS
- NO LIVE TRADING
- NO TRADING APPROVAL

This module:
- Imports hypothetical positions.
- Calculates mark-to-market PnL.
- Estimates margin and portfolio exposure.
- Reports research risk-limit violations.

This module DOES NOT:
- Open or close positions.
- Simulate liquidation.
- Simulate order execution.
- Guarantee stop-loss execution.
- Model funding, slippage or intrabar price paths.

All outputs are preliminary research estimates.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name(
    "research_config.json"
)


REQUIRED_SAFETY = {
    "research_only": True,
    "trading_enabled": False,
    "live_orders_enabled": False,
    "exchange_order_access": False,
    "model_approved": False,
}


def valid_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def require_positive(name, value):
    if not valid_number(value) or value <= 0:
        raise ValueError(
            f"{name} must be a positive finite number"
        )


def require_nonnegative(name, value):
    if not valid_number(value) or value < 0:
        raise ValueError(
            f"{name} must be a nonnegative finite number"
        )


def verify_research_safety(config):
    if not isinstance(config, dict):
        raise ValueError("Invalid research configuration")

    if config.get("status") != (
        "RESEARCH_ONLY_NOT_APPROVED"
    ):
        raise ValueError(
            "Research status is not locked"
        )

    safety = config.get("safety", {})

    if not isinstance(safety, dict):
        raise ValueError("Invalid safety configuration")

    for key, expected in REQUIRED_SAFETY.items():
        if safety.get(key) is not expected:
            raise ValueError(
                f"Research safety lock failed: {key}"
            )

    market = config.get("market", {})

    if market.get("symbol") != "BTC":
        raise ValueError(
            "V24.4 supports BTC only"
        )

    if market.get("market_type") != (
        "futures_simulation"
    ):
        raise ValueError(
            "Only futures simulation is supported"
        )

    risk = config.get("risk_management", {})

    required_positive = (
        "maximum_simulated_leverage",
        "max_gross_notional_to_equity",
        "max_total_planned_risk_fraction",
    )

    for key in required_positive:
        require_positive(
            key,
            risk.get(key),
        )

    if risk.get("independent_risk_engine") is not True:
        raise ValueError(
            "Independent risk engine must be enabled"
        )

    if risk.get("emergency_stop_enabled") is not True:
        raise ValueError(
            "Emergency stop must be enabled"
        )

    return True


def load_research_config(path=CONFIG_PATH):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    verify_research_safety(config)

    return config


@dataclass(frozen=True)
class SimulatedPosition:
    position_id: str
    direction: str
    entry_price: float
    quantity_btc: float
    leverage: float
    protective_stop: float
    entry_fee_usdt: float = 0.0
    planned_risk_usdt: float = 0.0

    def __post_init__(self):
        if (
            not isinstance(self.position_id, str)
            or not self.position_id.strip()
        ):
            raise ValueError(
                "Invalid position ID"
            )

        if self.direction not in (
            "LONG",
            "SHORT",
        ):
            raise ValueError(
                "Direction must be LONG or SHORT"
            )

        require_positive(
            "entry_price",
            self.entry_price,
        )

        require_positive(
            "quantity_btc",
            self.quantity_btc,
        )

        require_positive(
            "leverage",
            self.leverage,
        )

        require_positive(
            "protective_stop",
            self.protective_stop,
        )

        require_nonnegative(
            "entry_fee_usdt",
            self.entry_fee_usdt,
        )

        require_nonnegative(
            "planned_risk_usdt",
            self.planned_risk_usdt,
        )

        if (
            self.direction == "LONG"
            and self.protective_stop
            >= self.entry_price
        ):
            raise ValueError(
                "LONG stop must be below entry"
            )

        if (
            self.direction == "SHORT"
            and self.protective_stop
            <= self.entry_price
        ):
            raise ValueError(
                "SHORT stop must be above entry"
            )

    @property
    def entry_notional_usdt(self):
        return (
            self.entry_price
            * self.quantity_btc
        )

    @property
    def initial_margin_usdt(self):
        return (
            self.entry_notional_usdt
            / self.leverage
        )

    def unrealized_pnl(self, mark_price):
        require_positive(
            "mark_price",
            mark_price,
        )

        direction_sign = (
            1
            if self.direction == "LONG"
            else -1
        )

        return (
            direction_sign
            * (
                mark_price
                - self.entry_price
            )
            * self.quantity_btc
        )

    def planned_stop_loss_usdt(self):
        return (
            abs(
                self.entry_price
                - self.protective_stop
            )
            * self.quantity_btc
        )


class ResearchPositionLedger:
    """
    Accounting ledger for imported hypothetical positions.

    No open_position(), close_position(), or
    place_order() methods are provided.

    Imported positions are not approved executions.
    """

    def __init__(
        self,
        initial_capital_usdt,
        positions,
        config=None,
    ):
        self.config = (
            load_research_config()
            if config is None
            else config
        )

        verify_research_safety(
            self.config
        )

        require_positive(
            "initial_capital_usdt",
            initial_capital_usdt,
        )

        if not isinstance(positions, list):
            raise ValueError(
                "Positions must be a list"
            )

        if not all(
            isinstance(
                position,
                SimulatedPosition,
            )
            for position in positions
        ):
            raise ValueError(
                "All positions must be simulated"
            )

        ids = [
            position.position_id
            for position in positions
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate position IDs"
            )

        maximum_leverage = self.config[
            "risk_management"
        ]["maximum_simulated_leverage"]

        for position in positions:
            if position.leverage > maximum_leverage:
                raise ValueError(
                    "Simulated leverage limit exceeded"
                )

        self.initial_capital_usdt = (
            initial_capital_usdt
        )

        self.positions = tuple(positions)

    def snapshot(
        self,
        mark_price,
        realized_pnl_usdt=0.0,
        additional_costs_usdt=0.0,
    ):
        """
        Produce an accounting snapshot.

        realized_pnl_usdt:
            Externally supplied historical realized PnL.

        additional_costs_usdt:
            Externally supplied accumulated costs,
            excluding entry fees listed in positions.

        Margin values are simplified estimates.

        This method does not approve new positions.
        """

        verify_research_safety(
            self.config
        )

        require_positive(
            "mark_price",
            mark_price,
        )

        if not valid_number(
            realized_pnl_usdt
        ):
            raise ValueError(
                "Invalid realized PnL"
            )

        require_nonnegative(
            "additional_costs_usdt",
            additional_costs_usdt,
        )

        total_unrealized = 0.0
        total_margin = 0.0
        total_gross_notional = 0.0
        total_planned_risk = 0.0
        total_entry_fees = 0.0

        position_reports = []

        for position in self.positions:
            unrealized = (
                position.unrealized_pnl(
                    mark_price
                )
            )

            initial_margin = (
                position.initial_margin_usdt
            )

            current_notional = (
                mark_price
                * position.quantity_btc
            )

            stop_risk = (
                position.planned_stop_loss_usdt()
            )

            total_unrealized += unrealized
            total_margin += initial_margin
            total_gross_notional += (
                current_notional
            )
            total_planned_risk += stop_risk
            total_entry_fees += (
                position.entry_fee_usdt
            )

            position_reports.append({
                "position_id":
                    position.position_id,
                "direction":
                    position.direction,
                "entry_price":
                    position.entry_price,
                "mark_price":
                    mark_price,
                "quantity_btc":
                    position.quantity_btc,
                "leverage":
                    position.leverage,
                "entry_notional_usdt":
                    round(
                        position.entry_notional_usdt,
                        2,
                    ),
                "current_notional_usdt":
                    round(
                        current_notional,
                        2,
                    ),
                "initial_margin_usdt":
                    round(
                        initial_margin,
                        2,
                    ),
                "unrealized_pnl_usdt":
                    round(
                        unrealized,
                        2,
                    ),
                "protective_stop":
                    position.protective_stop,
                "planned_stop_loss_usdt":
                    round(
                        stop_risk,
                        2,
                    ),
                "entry_fee_usdt":
                    round(
                        position.entry_fee_usdt,
                        2,
                    ),
                "research_only": True,
            })

        cash_equity = (
            self.initial_capital_usdt
            + realized_pnl_usdt
            - total_entry_fees
            - additional_costs_usdt
        )

        marked_equity = (
            cash_equity
            + total_unrealized
        )

        free_margin_estimate = (
            cash_equity
            - total_margin
        )

        available_equity_estimate = (
            marked_equity
            - total_margin
        )

        risk_config = self.config[
            "risk_management"
        ]

        max_gross_ratio = risk_config[
            "max_gross_notional_to_equity"
        ]

        gross_limit = max(
            0.0,
            marked_equity
            * max_gross_ratio,
        )

        gross_limit_exceeded = (
            total_gross_notional
            > gross_limit
        )

        max_risk_fraction = risk_config[
            "max_total_planned_risk_fraction"
        ]

        risk_limit = max(
            0.0,
            marked_equity
            * max_risk_fraction,
        )

        risk_limit_exceeded = (
            total_planned_risk
            > risk_limit
        )

        insufficient_margin = (
            available_equity_estimate < 0
        )

        risk_alerts = []

        if gross_limit_exceeded:
            risk_alerts.append(
                "GROSS_EXPOSURE_LIMIT_EXCEEDED"
            )

        if risk_limit_exceeded:
            risk_alerts.append(
                "PLANNED_RISK_LIMIT_EXCEEDED"
            )

        if insufficient_margin:
            risk_alerts.append(
                "NEGATIVE_MARGIN_ESTIMATE"
            )

        if marked_equity <= 0:
            risk_alerts.append(
                "NONPOSITIVE_MARKED_EQUITY"
            )

        return {
            "version": "V24.4",
            "status":
                "RESEARCH_ONLY_NOT_APPROVED",
            "symbol": "BTC",
            "market_type":
                "futures_simulation",
            "research_only": True,
            "trading_enabled": False,
            "live_trading_allowed": False,
            "execution_ready": False,
            "accounting_only": True,
            "initial_capital_usdt":
                self.initial_capital_usdt,
            "mark_price":
                mark_price,
            "position_count":
                len(self.positions),
            "positions":
                position_reports,
            "realized_pnl_usdt":
                round(
                    realized_pnl_usdt,
                    2,
                ),
            "entry_fees_usdt":
                round(
                    total_entry_fees,
                    2,
                ),
            "additional_costs_usdt":
                round(
                    additional_costs_usdt,
                    2,
                ),
            "cash_equity_usdt":
                round(
                    cash_equity,
                    2,
                ),
            "unrealized_pnl_usdt":
                round(
                    total_unrealized,
                    2,
                ),
            "marked_equity_usdt":
                round(
                    marked_equity,
                    2,
                ),
            "initial_margin_used_usdt":
                round(
                    total_margin,
                    2,
                ),
            "free_margin_estimate_usdt":
                round(
                    free_margin_estimate,
                    2,
                ),
            "available_equity_estimate_usdt":
                round(
                    available_equity_estimate,
                    2,
                ),
            "gross_notional_usdt":
                round(
                    total_gross_notional,
                    2,
                ),
            "gross_limit_exceeded":
                gross_limit_exceeded,
            "planned_stop_risk_usdt":
                round(
                    total_planned_risk,
                    2,
                ),
            "planned_risk_limit_exceeded":
                risk_limit_exceeded,
            "insufficient_margin_estimate":
                insufficient_margin,
            "risk_alerts":
                risk_alerts,
            "risk_checks_complete": False,
            "liquidation_checked": False,
            "slippage_checked": False,
            "funding_checked": False,
            "exchange_connected": False,
        }


def main():
    config = load_research_config()

    capital = config[
        "portfolio"
    ]["initial_capital_usdt"]

    # Hypothetical examples only.
    # These are NOT trading recommendations.

    examples = [
        SimulatedPosition(
            position_id="BTC-LONG-001",
            direction="LONG",
            entry_price=100000,
            quantity_btc=0.03,
            leverage=2,
            protective_stop=99000,
            entry_fee_usdt=1.2,
        ),
        SimulatedPosition(
            position_id="BTC-SHORT-002",
            direction="SHORT",
            entry_price=101000,
            quantity_btc=0.02,
            leverage=2,
            protective_stop=102000,
            entry_fee_usdt=0.8,
        ),
    ]

    ledger = ResearchPositionLedger(
        initial_capital_usdt=capital,
        positions=examples,
        config=config,
    )

    report = ledger.snapshot(
        mark_price=100500
    )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
