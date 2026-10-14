"""
V24.5 - BTC Research Backtest Engine Safety Tests

RESEARCH ONLY.
NO REAL ORDERS.
NO EXCHANGE CONNECTION.
NO LIVE TRADING.
NO TRADING APPROVAL.

These are initial smoke and safety tests.
They do not establish backtest correctness.
"""

import ast
import json
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

ENGINE_PATH = BASE_DIR / "backtest_engine.py"
CONFIG_PATH = BASE_DIR / "research_config.json"

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

        cls.source =
