"""
V24.4 Research Risk Engine Tests

All tests use simulated money.
No exchange connections.
No real orders.
"""

import copy
import unittest

from risk_engine import (
    load_config,
    calculate_position_size,
)


class TestResearchRiskEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def calculate(self, **overrides):
        params = {
            "config": self.config,
            "equity_usdt": 10000,
            "free_balance_usdt": 10000,
            "entry_price": 100000,
            "stop_price": 99000,
            "direction": "LONG",
            "leverage": 2,
            "existing_positions": [],
            "daily_realized_pnl_usdt": 0,
        }

        params.update(overrides)

        return calculate_position_size(**params)

    def test_research_safety_lock(self):
        self.assertTrue(
            self.config["safety"]["research_only"]
        )
        self.assertFalse(
            self.config["safety"]["trading_enabled"]
        )
        self.assertFalse(
            self.config["safety"]["live_orders_enabled"]
        )
        self.assertFalse(
            self.config["safety"]["exchange_order_access"]
        )

    def test_normal_position_sizing(self):
        result = self.calculate()

        self.assertTrue(result["approved"])
        self.assertEqual(
            result["position_notional_usdt"],
            5000.0,
        )
        self.assertEqual(
            result["required_margin_usdt"],
            2500.0,
        )
        self.assertEqual(
            result["planned_risk_usdt"],
            50.0,
        )
        self.assertFalse(
            result["live_trading_allowed"]
        )
        self.assertFalse(
            result["execution_ready"]
        )

    def test_excessive_leverage_rejected(self):
        result = self.calculate(leverage=20)

        self.assertFalse(result["approved"])
        self.assertEqual(
            result["reason"],
            "LEVERAGE_LIMIT_EXCEEDED",
        )

    def test_invalid_long_stop_rejected(self):
        result = self.calculate(
            stop_price=101000,
        )

        self.assertFalse(result["approved"])
        self.assertEqual(
            result["reason"],
            "INVALID_LONG_STOP",
        )

    def test_invalid_short_stop_rejected(self):
        result = self.calculate(
            direction="SHORT",
            stop_price=99000,
        )

        self.assertFalse(result["approved"])
        self.assertEqual(
            result["reason"],
            "INVALID_SHORT_STOP",
        )

    def test_portfolio_risk_limit_rejected(self):
        result = self.calculate(
            existing_positions=[
                {
                    "planned_risk_usdt": 200,
                    "position_notional_usdt": 5000,
                }
            ]
        )

        self.assertFalse(result["approved"])
        self.assertEqual(
            result["reason"],
            "PORTFOLIO_RISK_LIMIT_REACHED",
        )

    def test_daily_loss_limit_rejected(self):
        result = self.calculate(
            daily_realized_pnl_usdt=-200,
        )

        self.assertFalse(result["approved"])
        self.assertEqual(
            result["reason"],
            "DAILY_LOSS_LIMIT_REACHED",
        )

    def test_gross_exposure_limit_rejected(self):
        result = self.calculate(
            existing_positions=[
                {
                    "planned_risk_usdt": 50,
                    "position_notional_usdt": 20000,
                }
            ]
        )

        self.assertFalse(result["approved"])
        self.assertEqual(
            result["reason"],
            "GROSS_EXPOSURE_LIMIT_REACHED",
        )

    def test_invalid_position_ledger_rejected(self):
        result = self.calculate(
            existing_positions=[
                {
                    "planned_risk_usdt": -10,
                    "position_notional_usdt": 1000,
                }
            ]
        )

        self.assertFalse(result["approved"])
        self.assertEqual(
            result["reason"],
            "INVALID_POSITION_LEDGER",
        )

    def test_nan_input_rejected(self):
        result = self.calculate(
            entry_price=float("nan"),
        )

        self.assertFalse(result["approved"])
        self.assertEqual(
            result["reason"],
            "INVALID_NUMERIC_INPUT",
        )

    def test_safety_cannot_be_enabled(self):
        unsafe_config = copy.deepcopy(self.config)

        unsafe_config["safety"][
            "trading_enabled"
        ] = True

        result = self.calculate(
            config=unsafe_config,
        )

        self.assertFalse(result["approved"])
        self.assertEqual(
            result["reason"],
            "SAFETY_LOCK_FAILED",
        )

    def test_margin_reserve_respected(self):
        result = self.calculate(
            free_balance_usdt=2000,
        )

        self.assertFalse(result["approved"])
        self.assertEqual(
            result["reason"],
            "INSUFFICIENT_AVAILABLE_MARGIN",
        )

    def test_short_position_supported(self):
        result = self.calculate(
            direction="SHORT",
            stop_price=101000,
        )

        self.assertTrue(result["approved"])
        self.assertEqual(
            result["direction"],
            "SHORT",
        )

    def test_existing_positions_reduce_risk_budget(self):
        result = self.calculate(
            existing_positions=[
                {
                    "planned_risk_usdt": 180,
                    "position_notional_usdt": 5000,
                }
            ]
        )

        self.assertTrue(result["approved"])
        self.assertEqual(
            result["planned_risk_usdt"],
            20.0,
        )
        self.assertEqual(
            result["portfolio_planned_risk_after_usdt"],
            200.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
