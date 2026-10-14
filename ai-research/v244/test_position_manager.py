"""
V24.4 - BTC Position Manager Tests

Research-only tests.
No exchange connections or real orders.
"""

import copy
import unittest

from position_manager import (
    SimulatedPosition,
    ResearchPositionLedger,
    load_research_config,
)


class TestPositionManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = load_research_config()

    def make_position(
        self,
        position_id="BTC-LONG-001",
        direction="LONG",
        entry_price=100000,
        quantity_btc=0.03,
        leverage=2,
        protective_stop=99000,
        entry_fee_usdt=1.2,
    ):
        return SimulatedPosition(
            position_id=position_id,
            direction=direction,
            entry_price=entry_price,
            quantity_btc=quantity_btc,
            leverage=leverage,
            protective_stop=protective_stop,
            entry_fee_usdt=entry_fee_usdt,
        )

    def make_ledger(self, positions, config=None):
        return ResearchPositionLedger(
            initial_capital_usdt=10000,
            positions=positions,
            config=self.config if config is None else config,
        )

    def test_long_profit(self):
        position = self.make_position()

        self.assertAlmostEqual(
            position.unrealized_pnl(101000),
            30.0,
        )

    def test_long_loss(self):
        position = self.make_position()

        self.assertAlmostEqual(
            position.unrealized_pnl(99000),
            -30.0,
        )

    def test_short_profit(self):
        position = self.make_position(
            direction="SHORT",
            protective_stop=101000,
        )

        self.assertAlmostEqual(
            position.unrealized_pnl(99000),
            30.0,
        )

    def test_short_loss(self):
        position = self.make_position(
            direction="SHORT",
            protective_stop=101000,
        )

        self.assertAlmostEqual(
            position.unrealized_pnl(101000),
            -30.0,
        )

    def test_multiple_positions(self):
        long_position = self.make_position()

        short_position = self.make_position(
            position_id="BTC-SHORT-002",
            direction="SHORT",
            entry_price=101000,
            quantity_btc=0.02,
            protective_stop=102000,
            entry_fee_usdt=0.8,
        )

        ledger = self.make_ledger([
            long_position,
            short_position,
        ])

        report = ledger.snapshot(
            mark_price=100500
        )

        self.assertEqual(
            report["position_count"],
            2,
        )

        self.assertAlmostEqual(
            report["unrealized_pnl_usdt"],
            25.0,
        )

        self.assertAlmostEqual(
            report["initial_margin_used_usdt"],
            2510.0,
        )

        self.assertAlmostEqual(
            report["marked_equity_usdt"],
            10023.0,
        )

        self.assertAlmostEqual(
            report["gross_notional_usdt"],
            5025.0,
        )

    def test_duplicate_position_ids_rejected(self):
        position = self.make_position()

        with self.assertRaises(ValueError):
            self.make_ledger([
                position,
                position,
            ])

    def test_excessive_leverage_rejected(self):
        position = self.make_position(
            leverage=11,
        )

        with self.assertRaises(ValueError):
            self.make_ledger([
                position,
            ])

    def test_invalid_long_stop_rejected(self):
        with self.assertRaises(ValueError):
            self.make_position(
                protective_stop=101000,
            )

    def test_invalid_short_stop_rejected(self):
        with self.assertRaises(ValueError):
            self.make_position(
                direction="SHORT",
                protective_stop=99000,
            )

    def test_invalid_mark_price_rejected(self):
        ledger = self.make_ledger([
            self.make_position(),
        ])

        with self.assertRaises(ValueError):
            ledger.snapshot(
                mark_price=-1
            )

    def test_research_safety_lock(self):
        ledger = self.make_ledger([
            self.make_position(),
        ])

        report = ledger.snapshot(
            mark_price=100500
        )

        self.assertTrue(
            report["research_only"]
        )

        self.assertFalse(
            report["trading_enabled"]
        )

        self.assertFalse(
            report["execution_ready"]
        )

        self.assertFalse(
            report["exchange_connected"]
        )

    def test_unsafe_configuration_rejected(self):
        unsafe_config = copy.deepcopy(
            self.config
        )

        unsafe_config["safety"][
            "trading_enabled"
        ] = True

        with self.assertRaises(ValueError):
            self.make_ledger(
                [self.make_position()],
                config=unsafe_config,
            )

    def test_gross_exposure_warning(self):
        position = self.make_position(
            quantity_btc=0.25,
        )

        ledger = self.make_ledger([
            position,
        ])

        report = ledger.snapshot(
            mark_price=100000
        )

        self.assertTrue(
            report["gross_limit_exceeded"]
        )

        self.assertIn(
            "GROSS_EXPOSURE_LIMIT_EXCEEDED",
            report["risk_alerts"],
        )

    def test_planned_risk_warning(self):
        position = self.make_position(
            quantity_btc=0.25,
        )

        ledger = self.make_ledger([
            position,
        ])

        report = ledger.snapshot(
            mark_price=100000
        )

        self.assertTrue(
            report["planned_risk_limit_exceeded"]
        )

    def test_no_execution_methods(self):
        ledger = self.make_ledger([])

        self.assertFalse(
            hasattr(ledger, "place_order")
        )

        self.assertFalse(
            hasattr(ledger, "open_position")
        )

        self.assertFalse(
            hasattr(ledger, "close_position")
        )


if __name__ == "__main__":
    unittest.main()
