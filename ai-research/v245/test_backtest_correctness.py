"""
V24.5 - BTC Backtest Correctness Tests

Deterministic tests using synthetic candles.

RESEARCH ONLY.
NO REAL ORDERS.
NO EXCHANGE CONNECTION.
NO TRADING APPROVAL.
"""

import copy
import importlib.util
import json
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

ENGINE_PATH = BASE_DIR / "backtest_engine.py"
CONFIG_PATH = BASE_DIR / "research_config.json"


def load_engine_module():
    spec = importlib.util.spec_from_file_location(
        "v245_backtest_engine",
        ENGINE_PATH
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load backtest engine")

    module = importlib.util.module_from_spec(spec)

    import sys
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


engine_module = load_engine_module()

Candle = engine_module.Candle
ResearchSignal = engine_module.ResearchSignal
BTCBacktestEngine = engine_module.BTCBacktestEngine


def load_test_config():
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    return config


def make_candle(
    minute,
    open_price=100.0,
    high=None,
    low=None,
    close=None,
    volume=10.0
):
    if high is None:
        high = open_price

    if low is None:
        low = open_price

    if close is None:
        close = open_price

    return Candle(
        timestamp=minute * 60,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume
    )


def make_signal(
    minute=0,
    direction="LONG",
    stop=0.01,
    target=0.02,
    holding=5,
    risk=0.005
):
    return ResearchSignal(
        timestamp=minute * 60,
        direction=direction,
        stop_loss_pct=stop,
        take_profit_pct=target,
        max_holding_minutes=holding,
        risk_fraction=risk
    )


class TestBacktestCorrectness(unittest.TestCase):

    def setUp(self):
        self.config = load_test_config()

        # Isolate arithmetic tests from fee/slippage
        # unless a test explicitly enables them.
        self.config["execution_assumptions"][
            "market_order_fee_fraction_per_side"
        ] = 0.0

        self.config["execution_assumptions"][
            "base_slippage_fraction_per_side"
        ] = 0.0

        self.config["execution_assumptions"][
            "stress_slippage_fraction_per_side"
        ] = 0.0

        self.engine = BTCBacktestEngine(
            self.config
        )

    def test_long_take_profit(self):
        candles = [
            make_candle(0),
            make_candle(
                1,
                open_price=100,
                high=103,
                low=100,
                close=102
            )
        ]

        result = self.engine.run(
            candles,
            [make_signal()]
        )

        self.assertEqual(
            result["closed_trades"],
            1
        )

        trade = result["trades"][0]

        self.assertEqual(
            trade["exit_reason"],
            "TAKE_PROFIT"
        )

        self.assertGreater(
            trade["net_pnl_usdt"],
            0
        )

    def test_short_take_profit(self):
        candles = [
            make_candle(0),
            make_candle(
                1,
                open_price=100,
                high=100,
                low=97,
                close=98
            )
        ]

        result = self.engine.run(
            candles,
            [
                make_signal(
                    direction="SHORT"
                )
            ]
        )

        self.assertEqual(
            result["closed_trades"],
            1
        )

        self.assertEqual(
            result["trades"][0]["exit_reason"],
            "TAKE_PROFIT"
        )

        self.assertGreater(
            result["net_pnl_usdt"],
            0
        )

    def test_stop_first_when_both_hit(self):
        candles = [
            make_candle(0),
            make_candle(
                1,
                open_price=100,
                high=103,
                low=98,
                close=101
            )
        ]

        result = self.engine.run(
            candles,
            [make_signal()]
        )

        trade = result["trades"][0]

        self.assertEqual(
            trade["exit_reason"],
            "STOP_FIRST_AMBIGUOUS"
        )

        self.assertLess(
            trade["net_pnl_usdt"],
            0
        )

    def test_next_candle_entry(self):
        candles = [
            make_candle(
                0,
                open_price=100,
                high=110,
                low=90,
                close=100
            ),
            make_candle(
                1,
                open_price=105,
                high=105,
                low=105,
                close=105
            ),
            make_candle(
                2,
                open_price=105,
                high=105,
                low=105,
                close=105
            )
        ]

        result = self.engine.run(
            candles,
            [
                make_signal(
                    holding=1
                )
            ]
        )

        self.assertEqual(
            result["closed_trades"],
            1
        )

        self.assertEqual(
            result["trades"][0]["entry_price"],
            105.0
        )

        self.assertEqual(
            result["trades"][0]["entry_timestamp"],
            60
        )

    def test_final_candle_signal_rejected(self):
        candles = [
            make_candle(0),
            make_candle(1)
        ]

        result = self.engine.run(
            candles,
            [
                make_signal(
                    minute=1
                )
            ]
        )

        self.assertEqual(
            result["closed_trades"],
            0
        )

        self.assertEqual(
            result["rejection_reasons"].get(
                "NO_NEXT_CANDLE"
            ),
            1
        )

    def test_wait_does_not_open_position(self):
        candles = [
            make_candle(0),
            make_candle(1)
        ]

        result = self.engine.run(
            candles,
            [
                make_signal(
                    direction="WAIT"
                )
            ]
        )

        self.assertEqual(
            result["closed_trades"],
            0
        )

        self.assertEqual(
            result["wait_signals"],
            1
        )

    def test_duplicate_timestamps_rejected(self):
        candles = [
            make_candle(0),
            make_candle(0)
        ]

        with self.assertRaises(ValueError):
            self.engine.run(
                candles,
                []
            )

    def test_missing_minute_rejected(self):
        candles = [
            make_candle(0),
            make_candle(2)
        ]

        with self.assertRaises(ValueError):
            self.engine.run(
                candles,
                []
            )

    def test_fees_reduce_net_profit(self):
        config = copy.deepcopy(
            self.config
        )

        config["execution_assumptions"][
            "market_order_fee_fraction_per_side"
        ] = 0.0006

        fee_engine = BTCBacktestEngine(
            config
        )

        candles = [
            make_candle(0),
            make_candle(
                1,
                open_price=100,
                high=103,
                low=100,
                close=102
            )
        ]

        no_fee = self.engine.run(
            candles,
            [make_signal()]
        )

        with_fee = fee_engine.run(
            candles,
            [make_signal()]
        )

        self.assertGreater(
            no_fee["net_pnl_usdt"],
            with_fee["net_pnl_usdt"]
        )

        self.assertGreater(
            with_fee["total_entry_fees_usdt"],
            0
        )

        self.assertGreater(
            with_fee["total_exit_fees_usdt"],
            0
        )

    def test_accounting_matches_trade_pnl(self):
        candles = [
            make_candle(0),
            make_candle(
                1,
                open_price=100,
                high=103,
                low=100,
                close=102
            )
        ]

        result = self.engine.run(
            candles,
            [make_signal()]
        )

        trade_total = sum(
            trade["net_pnl_usdt"]
            for trade in result["trades"]
        )

        equity_change = (
            result["final_equity_usdt"]
            - result["initial_capital_usdt"]
        )

        self.assertAlmostEqual(
            trade_total,
            equity_change,
            places=2
        )

    def test_risk_fraction_limit(self):
        candles = [
            make_candle(0),
            make_candle(1)
        ]

        signal = make_signal(
            risk=0.02
        )

        result = self.engine.run(
            candles,
            [signal]
        )

        self.assertEqual(
            result["accepted_signals"],
            0
        )

        self.assertEqual(
            result["rejection_reasons"].get(
                "PER_POSITION_RISK_LIMIT"
            ),
            1
        )

    def test_research_safety_lock(self):
        config = copy.deepcopy(
            self.config
        )

        config["safety"][
            "trading_enabled"
        ] = True

        with self.assertRaises(ValueError):
            BTCBacktestEngine(
                config
            )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
            )
