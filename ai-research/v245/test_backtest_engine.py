"""
V24.5 - BTC Research Backtest Engine Safety Tests

RESEARCH ONLY.
NO REAL ORDERS.
NO EXCHANGE CONNECTION.
NO LIVE TRADING.
NO TRADING APPROVAL.

These tests verify basic file integrity, Python syntax,
research configuration, and safety restrictions.

They do NOT prove backtest correctness, profitability,
or suitability for real trading.
"""

import ast
import json
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

ENGINE_PATH = BASE_DIR / "backtest_engine.py"
CONFIG_PATH = BASE_DIR / "research_config.json"

EXPECTED_VERSION = "V24.5"
EXPECTED_STATUS = "RESEARCH_ONLY_NOT_APPROVED"


class TestBTCBacktestEngineSafety(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not ENGINE_PATH.is_file():
            raise AssertionError(
                "Missing backtest_engine.py"
            )

        if not CONFIG_PATH.is_file():
            raise AssertionError(
                "Missing research_config.json"
            )

        cls.source = ENGINE_PATH.read_text(
            encoding="utf-8"
        )

        cls.config = json.loads(
            CONFIG_PATH.read_text(
                encoding="utf-8"
            )
        )

        cls.tree = ast.parse(
            cls.source,
            filename=str(ENGINE_PATH)
        )

    def test_engine_not_empty(self):
        self.assertTrue(
            self.source.strip(),
            "Backtest engine is empty"
        )

    def test_engine_valid_python(self):
        self.assertIsInstance(
            self.tree,
            ast.Module
        )

    def test_config_version(self):
        self.assertEqual(
            self.config.get("version"),
            EXPECTED_VERSION
        )

    def test_research_status(self):
        self.assertEqual(
            self.config.get("status"),
            EXPECTED_STATUS
        )

    def test_research_only_enabled(self):
        safety = self.config["safety"]

        self.assertIs(
            safety["research_only"],
            True
        )

    def test_real_trading_disabled(self):
        safety = self.config["safety"]

        self.assertIs(
            safety["trading_enabled"],
            False
        )

        self.assertIs(
            safety["live_orders_enabled"],
            False
        )

        self.assertIs(
            safety["exchange_order_access"],
            False
        )

        self.assertIs(
            safety["allow_real_capital"],
            False
        )

    def test_model_not_approved(self):
        self.assertIs(
            self.config["safety"]["model_approved"],
            False
        )

    def test_btc_only(self):
        market = self.config["market"]

        self.assertEqual(
            market["symbol"],
            "BTC"
        )

        self.assertEqual(
            market["reference_pair"],
            "BTC-USD"
        )

    def test_one_minute_analysis(self):
        market = self.config["market"]

        self.assertEqual(
            market["candle_interval_minutes"],
            1
        )

        self.assertEqual(
            market["analysis_interval_minutes"],
            1
        )

    def test_simulation_only(self):
        execution = self.config[
            "execution_assumptions"
        ]

        self.assertEqual(
            execution["mode"],
            "historical_simulation_only"
        )

        self.assertIs(
            execution["allow_execution_ready_claim"],
            False
        )

    def test_no_stop_widening(self):
        risk = self.config["risk_management"]

        position = self.config[
            "position_management"
        ]

        self.assertIs(
            risk["allow_widening_protective_stop"],
            False
        )

        self.assertIs(
            position["allow_stop_widening"],
            False
        )

    def test_risk_limits(self):
        risk = self.config["risk_management"]

        self.assertGreater(
            risk["risk_per_position_fraction"],
            0
        )

        self.assertLessEqual(
            risk["risk_per_position_fraction"],
            0.01
        )

        self.assertLessEqual(
            risk["max_total_planned_risk_fraction"],
            0.02
        )

        self.assertLessEqual(
            risk["max_daily_loss_fraction"],
            0.02
        )

    def test_emergency_stop(self):
        risk = self.config["risk_management"]

        self.assertIs(
            risk["emergency_stop_enabled"],
            True
        )

        self.assertIs(
            risk[
                "reject_new_positions_when_risk_limit_reached"
            ],
            True
        )

    def test_maximum_holding_time(self):
        position = self.config[
            "position_management"
        ]

        self.assertEqual(
            position["maximum_holding_minutes"],
            360
        )

    def test_conservative_same_candle_policy(self):
        execution = self.config[
            "execution_assumptions"
        ]

        self.assertEqual(
            execution[
                "same_candle_stop_and_target_policy"
            ],
            "stop_first"
        )

    def test_next_candle_entry(self):
        execution = self.config[
            "execution_assumptions"
        ]

        self.assertIs(
            execution["signal_at_completed_candle"],
            True
        )

        self.assertEqual(
            execution["earliest_entry"],
            "next_candle_open"
        )

    def test_funding_unverified(self):
        execution = self.config[
            "execution_assumptions"
        ]

        self.assertEqual(
            execution["funding_rate_source"],
            "not_available"
        )

    def test_liquidation_model_missing(self):
        execution = self.config[
            "execution_assumptions"
        ]

        self.assertEqual(
            execution["liquidation_model"],
            "not_implemented"
        )

    def test_no_automatic_approval(self):
        approval = self.config["approval"]

        self.assertIs(
            approval["automatic_approval"],
            False
        )

        self.assertIs(
            approval["trading_approval"],
            False
        )

        self.assertIs(
            approval["execution_ready"],
            False
        )

    def test_manual_review_required(self):
        approval = self.config["approval"]

        self.assertIs(
            approval["require_manual_research_review"],
            True
        )

    def test_no_future_data_allowed(self):
        data = self.config["data"]

        self.assertIs(
            data["allow_future_data_in_features"],
            False
        )

        self.assertIs(
            data["allow_future_data_in_signals"],
            False
        )

    def test_holdout_not_for_selection(self):
        validation = self.config["validation"]

        self.assertIs(
            validation["holdout_used_for_model_selection"],
            False
        )

        self.assertIs(
            validation["walk_forward_required"],
            True
        )

    def test_engine_has_main_function(self):
        functions = {
            node.name
            for node in ast.walk(self.tree)
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef)
            )
        }

        self.assertIn(
            "main",
            functions,
            "Engine must define main()"
        )

    def test_engine_has_backtest_class(self):
        classes = {
            node.name
            for node in ast.walk(self.tree)
            if isinstance(node, ast.ClassDef)
        }

        self.assertIn(
            "BTCBacktestEngine",
            classes,
            "BTCBacktestEngine class is missing"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
