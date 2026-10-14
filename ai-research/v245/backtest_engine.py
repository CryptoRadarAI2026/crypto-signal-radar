"""
V24.5 - BTC Research Backtest Engine Safety Tests

RESEARCH ONLY.
NO REAL ORDERS.
NO EXCHANGE CONNECTION.
NO LIVE TRADING.
NO TRADING APPROVAL.

These are initial smoke and safety tests.
They do not establish backtest correctness or profitability.
"""

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ENGINE_PATH = BASE_DIR / "backtest_engine.py"
CONFIG_PATH = BASE_DIR / "research_config.json"

EXPECTED_STATUS = "RESEARCH_ONLY_NOT_APPROVED"


class TestV245BacktestEngine(unittest.TestCase):

    def test_engine_file_exists(self):
        self.assertTrue(
            ENGINE_PATH.is_file(),
            "backtest_engine.py is missing",
        )

    def test_config_file_exists(self):
        self.assertTrue(
            CONFIG_PATH.is_file(),
            "research_config.json is missing",
        )

    def test_engine_python_syntax(self):
        source = ENGINE_PATH.read_text(
            encoding="utf-8"
        )

        ast.parse(
            source,
            filename=str(ENGINE_PATH),
        )

    def test_required_engine_classes(self):
        source = ENGINE_PATH.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(source)

        classes = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }

        self.assertIn(
            "BTCBacktestEngine",
            classes,
        )

    def test_required_main_function(self):
        source = ENGINE_PATH.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(source)

        functions = {
            node.name
            for node in tree.body
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
        }

        self.assertIn("main", functions)
        self.assertIn("load_config", functions)

    def test_config_is_valid_json(self):
        with CONFIG_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)

        self.assertIsInstance(config, dict)

    def test_research_status_locked(self):
        config = self.load_config()

        self.assertEqual(
            config.get("status"),
            EXPECTED_STATUS,
        )

    def test_research_safety_flags(self):
        config = self.load_config()
        safety = config.get("safety", {})

        self.assertIs(
            safety.get("research_only"),
            True,
        )

        for key in (
            "trading_enabled",
            "live_orders_enabled",
            "exchange_order_access",
            "model_approved",
            "allow_real_capital",
        ):
            with self.subTest(key=key):
                self.assertIs(
                    safety.get(key),
                    False,
                )

    def test_btc_only_market(self):
        config = self.load_config()
        market = config.get("market", {})

        self.assertEqual(
            market.get("symbol"),
            "BTC",
        )

        self.assertEqual(
            market.get("reference_pair"),
            "BTC-USD",
        )

    def test_historical_simulation_only(self):
        config = self.load_config()

        execution = config.get(
            "execution_assumptions",
            {},
        )

        self.assertEqual(
            execution.get("mode"),
            "historical_simulation_only",
        )

        self.assertIs(
            execution.get(
                "allow_execution_ready_claim"
            ),
            False,
        )

    def test_stop_widening_disabled(self):
        config = self.load_config()

        risk = config.get(
            "risk_management",
            {},
        )

        management = config.get(
            "position_management",
            {},
        )

        self.assertIs(
            risk.get(
                "allow_widening_protective_stop"
            ),
            False,
        )

        self.assertIs(
            management.get(
                "allow_stop_widening"
            ),
            False,
        )

    def test_conservative_same_candle_policy(self):
        config = self.load_config()

        execution = config.get(
            "execution_assumptions",
            {},
        )

        self.assertEqual(
            execution.get(
                "same_candle_stop_and_target_policy"
            ),
            "stop_first",
        )

    def test_liquidation_not_claimed(self):
        config = self.load_config()

        execution = config.get(
            "execution_assumptions",
            {},
        )

        self.assertEqual(
            execution.get("liquidation_model"),
            "not_implemented",
        )

    def test_approval_disabled(self):
        config = self.load_config()
        approval = config.get("approval", {})

        for key in (
            "automatic_approval",
            "trading_approval",
            "execution_ready",
        ):
            with self.subTest(key=key):
                self.assertIs(
                    approval.get(key),
                    False,
                )

    def test_engine_startup(self):
        """
        Runs the engine's main() as a smoke test.

        Expected behavior:
        Print configuration information only.

        This test does NOT validate trade simulation.
        """

        result = subprocess.run(
            [
                sys.executable,
                str(ENGINE_PATH),
            ],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=(
                "Engine startup failed:\n"
                + result.stderr[-3000:]
            ),
        )

        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            self.fail(
                "Engine startup did not return "
                f"valid JSON: {exc}\n"
                f"{result.stdout[-2000:]}"
            )

        self.assertIsInstance(report, dict)

        self.assertEqual(
            report.get("status"),
            EXPECTED_STATUS,
        )

        self.assertEqual(
            report.get("symbol"),
            "BTC",
        )

        self.assertIs(
            report.get("research_only"),
            True,
        )

        self.assertIs(
            report.get("trading_enabled"),
            False,
        )

    @staticmethod
    def load_config():
        with CONFIG_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
