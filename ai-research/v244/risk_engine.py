"""
V24.4 - Independent Research Risk Engine

Research-only position sizing and portfolio risk checks.

NO EXCHANGE CONNECTIONS.
NO REAL ORDERS.
NO TRADING APPROVAL.

This module is NOT a complete futures execution or
liquidation simulation.
"""

import json
import math
from pathlib import Path


CONFIG_PATH = Path(__file__).with_name("research_config.json")


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)

    safety = config["safety"]

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
                f"Research safety configuration invalid: {key}"
            )

    if config.get("status") != "RESEARCH_ONLY_NOT_APPROVED":
        raise ValueError("Research status is not locked")

    return config


def valid_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def reject(reason):
    return {
        "approved": False,
        "reason": reason,
        "position_notional_usdt": 0.0,
        "required_margin_usdt": 0.0,
        "planned_risk_usdt": 0.0,
    }


def calculate_position_size(
    config,
    equity_usdt,
    free_balance_usdt,
    entry_price,
    stop_price,
    direction,
    leverage,
    existing_positions=None,
    daily_realized_pnl_usdt=0.0,
):
    """
    Calculate a candidate simulated position size.

    existing_positions entries must contain:
      planned_risk_usdt
      position_notional_usdt

    These values must come from a trusted portfolio ledger.

    The returned approval is preliminary and research-only.
    """

    existing_positions = (
        [] if existing_positions is None else existing_positions
    )

    safety = config.get("safety", {})

    if (
        safety.get("research_only") is not True
        or safety.get("trading_enabled") is not False
        or safety.get("live_orders_enabled") is not False
        or safety.get("exchange_order_access") is not False
        or safety.get("model_approved") is not False
        or config.get("status") != "RESEARCH_ONLY_NOT_APPROVED"
    ):
        return reject("SAFETY_LOCK_FAILED")

    values = [
        equity_usdt,
        free_balance_usdt,
        entry_price,
        stop_price,
        leverage,
        daily_realized_pnl_usdt,
    ]

    if not all(valid_number(value) for value in values):
        return reject("INVALID_NUMERIC_INPUT")

    if (
        equity_usdt <= 0
        or free_balance_usdt < 0
        or free_balance_usdt > equity_usdt
        or entry_price <= 0
        or stop_price <= 0
    ):
        return reject("INVALID_ACCOUNT_OR_PRICE")

    if direction not in ("LONG", "SHORT"):
        return reject("INVALID_DIRECTION")

    if direction == "LONG" and stop_price >= entry_price:
        return reject("INVALID_LONG_STOP")

    if direction == "SHORT" and stop_price <= entry_price:
        return reject("INVALID_SHORT_STOP")

    risk = config["risk_management"]
    portfolio = config["portfolio"]

    max_leverage = risk["maximum_simulated_leverage"]

    if leverage < 1 or leverage > max_leverage:
        return reject("LEVERAGE_LIMIT_EXCEEDED")

    if not isinstance(existing_positions, list):
        return reject("INVALID_POSITION_LEDGER")

    existing_risk = 0.0
    existing_notional = 0.0

    for position in existing_positions:
        if not isinstance(position, dict):
            return reject("INVALID_POSITION_LEDGER")

        position_risk = position.get("planned_risk_usdt")
        position_notional = position.get(
            "position_notional_usdt"
        )

        if (
            not valid_number(position_risk)
            or not valid_number(position_notional)
            or position_risk < 0
            or position_notional < 0
        ):
            return reject("INVALID_POSITION_LEDGER")

        existing_risk += position_risk
        existing_notional += position_notional

    max_daily_loss = (
        equity_usdt * risk["max_daily_loss_fraction"]
    )

    if daily_realized_pnl_usdt <= -max_daily_loss:
        return reject("DAILY_LOSS_LIMIT_REACHED")

    max_position_risk = (
        equity_usdt * risk["risk_per_position_fraction"]
    )

    max_portfolio_risk = (
        equity_usdt
        * risk["max_total_planned_risk_fraction"]
    )

    remaining_portfolio_risk = (
        max_portfolio_risk - existing_risk
    )

    if remaining_portfolio_risk <= 0:
        return reject("PORTFOLIO_RISK_LIMIT_REACHED")

    allowed_risk = min(
        max_position_risk,
        remaining_portfolio_risk,
    )

    stop_distance_fraction = (
        abs(entry_price - stop_price) / entry_price
    )

    if stop_distance_fraction <= 0:
        return reject("INVALID_STOP_DISTANCE")

    candidate_notional = (
        allowed_risk / stop_distance_fraction
    )

    max_gross_notional = (
        equity_usdt
        * risk["max_gross_notional_to_equity"]
    )

    remaining_notional = (
        max_gross_notional - existing_notional
    )

    if remaining_notional <= 0:
        return reject("GROSS_EXPOSURE_LIMIT_REACHED")

    reserve_fraction = portfolio[
        "reserve_capital_fraction"
    ]

    if not (
        valid_number(reserve_fraction)
        and 0 <= reserve_fraction < 1
    ):
        return reject("INVALID_RESERVE_CONFIGURATION")

    required_reserve = equity_usdt * reserve_fraction

    available_margin = max(
        0.0,
        free_balance_usdt - required_reserve,
    )

    margin_limited_notional = (
        available_margin * leverage
    )

    final_notional = min(
        candidate_notional,
        remaining_notional,
        margin_limited_notional,
    )

    if final_notional <= 0:
        return reject("INSUFFICIENT_AVAILABLE_MARGIN")

    required_margin = final_notional / leverage

    planned_risk = (
        final_notional * stop_distance_fraction
    )

    return {
        "approved": True,
        "reason": "PRELIMINARY_RESEARCH_SIZE_ONLY",
        "direction": direction,
        "entry_price": round(entry_price, 8),
        "stop_price": round(stop_price, 8),
        "simulated_leverage": leverage,
        "position_notional_usdt": round(
            final_notional, 2
        ),
        "required_margin_usdt": round(
            required_margin, 2
        ),
        "planned_risk_usdt": round(
            planned_risk, 2
        ),
        "portfolio_planned_risk_after_usdt": round(
            existing_risk + planned_risk, 2
        ),
        "gross_notional_after_usdt": round(
            existing_notional + final_notional, 2
        ),
        "research_only": True,
        "live_trading_allowed": False,
        "execution_ready": False,
        "liquidation_checked": False,
        "fees_and_slippage_checked": False,
    }


def main():
    config = load_config()

    initial_capital = config[
        "portfolio"
    ]["initial_capital_usdt"]

    example = calculate_position_size(
        config=config,
        equity_usdt=initial_capital,
        free_balance_usdt=initial_capital,
        entry_price=100000,
        stop_price=99000,
        direction="LONG",
        leverage=2,
        existing_positions=[],
        daily_realized_pnl_usdt=0,
    )

    report = {
        "version": "V24.4",
        "status": "RESEARCH_ONLY_NOT_APPROVED",
        "trading_enabled": False,
        "example_only": True,
        "result": example,
    }

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
