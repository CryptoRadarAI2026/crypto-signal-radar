"""
V24.4 - BTC Multi-Position Research Ledger

Research-only accounting for simulated BTC positions.

NO EXCHANGE CONNECTIONS.
NO REAL ORDERS.
NO LIVE TRADING.
NO TRADING APPROVAL.

Important:
- This is an accounting ledger, not an execution engine.
- Positions are imported as hypothetical fills.
- No liquidation, funding, slippage, or intrabar execution.
- No position may be opened through this module.
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name(
    "research_config.json"
)


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


def load_research_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)

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
                f"Research safety lock failed: {key}"
            )

    if config.get("status") != (
        "RESEARCH_ONLY_NOT_APPROVED"
    ):
        raise ValueError(
            "Research status must remain locked"
        )

    if config.get("market", {}).get("symbol") != "BTC":
        raise ValueError(
            "V24.4 ledger supports BTC only"
        )

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
            raise ValueError("Invalid position ID")

        if self.direction not in ("LONG", "SHORT"):
            raise ValueError("Invalid direction")

        require_positive(
            "entry_price", self.entry_price
        )
        require_positive(
            "quantity_btc", self.quantity_btc
        )
        require_positive(
            "leverage", self.leverage
        )
        require_positive(
            "protective_stop", self.protective_stop
        )
        require_nonnegative(
            "entry_fee_usdt", self.entry_fee_usdt
        )
        require_nonnegative(
            "planned_risk_usdt",
            self.planned_risk_usdt
        )

        if (
            self.direction == "LONG"
            and self.protective_stop >= self.entry_price
        ):
            raise ValueError(
                "LONG stop must be below entry"
            )

        if (
            self.direction == "SHORT"
            and self.protective_stop <= self.entry_price
        ):
            raise ValueError(
                "SHORT stop must be above entry"
            )

    @property
    def entry_notional_usdt(self):
        return self.entry_price * self.quantity_btc

    @property
    def initial_margin_usdt(self):
        return (
            self.entry_notional_usdt / self.leverage
        )

    def unrealized_pnl(self, mark_price):
        require_positive("mark_price", mark_price)

        direction_sign = (
            1 if self.direction == "LONG" else -1
        )

        return (
            direction_sign
            * (mark_price - self.entry_price)
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

    def __init__(
        self,
        initial_capital_usdt,
        positions,
        config=None,
    ):
        """
        Import existing hypothetical positions.

        This class deliberately has no open_position()
        or place_order() method.

        All imported positions are accounting examples,
        NOT approved executions.
        """

        self.config = (
            load_research_config()
            if config is None
            else config
        )

        self._verify_safety()

        require_positive(
            "initial_capital_usdt",
            initial_capital_usdt
        )

        if not isinstance(positions, list):
            raise ValueError(
                "positions must be a list"
            )

        self.initial_capital_usdt = (
            initial_capital_usdt
        )

        self.positions = list(positions)

        if not all(
            isinstance(p, SimulatedPosition)
            for p in self.positions
        ):
            raise ValueError(
                "All positions must be simulated"
            )

        ids = [
            p.position_id
            for p in self.positions
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate position IDs"
            )

        max_leverage = self.config[
            "risk_management"
        ]["maximum_simulated_leverage"]

        for position in self.positions:
            if position.leverage > max_leverage:
                raise ValueError(
                    "Simulated leverage exceeds limit"
                )

    def _verify_safety(self):
        safety = self.config.get("safety", {})

        if (
            safety.get("research_only") is not True
            or safety.get("trading_enabled") is not False
            or safety.get("live_orders_enabled") is not False
            or safety.get("exchange_order_access")
            is not False
            or safety.get("model_approved") is not False
            or self.config.get("status")
            != "RESEARCH_ONLY_NOT_APPROVED"
        ):
            raise ValueError(
                "Research safety lock failed"
            )

        if self.config.get(
            "market", {}
        ).get("symbol") != "BTC":
            raise ValueError(
                "Only BTC is supported"
            )

    def snapshot(
        self,
        mark_price,
        realized_pnl_usdt=0.0,
        additional_costs_usdt=0.0,
    ):
        """
        Mark all imported positions to one BTC price.

        This is a simplified accounting snapshot.

        realized_pnl_usdt:
            External historical realized PnL.

        additional_costs_usdt:
            External accumulated fees/funding/etc.
            supplied by a trusted research dataset.

        Entry fees are deducted separately below.

        This does not model liquidation or execution.
        """

        self._verify_safety()

        require_positive(
            "mark_price", mark_price
        )

        if not valid_number(realized_pnl_usdt):
            raise ValueError(
                "Invalid realized PnL"
            )

        require_nonnegative(
            "additional_costs_usdt",
            additional_costs_usdt
        )

        entry_fees = sum(
            p.entry_fee_usdt
            for p in self.positions
        )

        total_unrealized = 0.0
        total_margin = 0.0
        total_gross_notional = 0.0
        total_planned_risk = 0.0

        position_reports = []

        for position in self.positions:
            pnl = position.unrealized_pnl(
                mark_price
            )

            margin = (
                position.initial_margin_usdt
            )

            current_notional = (
                mark_price
                * position.quantity_btc
            )

            stop_risk = (
                position.planned_stop_loss_usdt()
            )

            total_unrealized += pnl
            total_margin += margin
            total_gross_notional += (
                current_notional
            )
            total_planned_risk += stop_risk

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
                        2
                    ),
                "current_notional_usdt":
                    round(
                        current_notional, 2
                    ),
                "initial_margin_usdt":
                    round(margin, 2),
                "unrealized_pnl_usdt":
                    round(pnl, 2),
                "protective_stop":
                    position.protective_stop,
                "planned_stop_loss_usdt":
                    round(stop_risk, 2),
                "research_only": True,
            })

        cash_equity = (
            self.initial_capital_usdt
            + realized_pnl_usdt
            - entry_fees
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

        max_gross_ratio = self.config[
            "risk_management"
        ]["max_gross_notional_to_equity"]

        gross_limit = max(
            0.0,
            marked_equity * max_gross_ratio
        )

        gross_limit_exceeded = (
            total_gross_notional > gross_limit
        )

        max_risk_fraction = self.config[
            "risk_management"
        ]["max_total_planned_risk_fraction"]

        risk_limit = max(
            0.0,
            marked_equity * max_risk_fraction
        )

        risk_limit_exceeded = (
            total_planned_risk > risk_limit
        )

        return {
            "version": "V24.4",
            "status":
                "RESEARCH_ONLY_NOT_APPROVED",
            "symbol": "BTC",
            "research_only": True,
            "trading_enabled": False,
            "execution_ready": False,
            "accounting_only": True,
            "initial_capital_usdt":
                self.initial_capital_usdt,
            "mark_price": mark_price,
            "position_count":
                len(self.positions),
            "positions": position_reports,
            "cash_equity_usdt":
                round(cash_equity, 2),
            "unrealized_pnl_usdt":
                round(total_unrealized, 2),
            "marked_equity_usdt":
                round(marked_equity, 2),
            "initial_margin_used_usdt":
                round(total_margin, 2),
            "free_margin_estimate_usdt":
                round(free_margin_estimate, 2),
            "available_equity_estimate_usdt":
                round(
                    available_equity_estimate, 2
                ),
            "gross_notional_usdt":
                round(
                    total_gross_notional, 2
                ),
            "gross_limit_exceeded":
                gross_limit_exceeded,
            "planned_stop_risk_usdt":
                round(
                    total_planned_risk, 2
                ),
            "planned_risk_limit_exceeded":
                risk_limit_exceeded,
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

    # Example positions only.
    # These are
