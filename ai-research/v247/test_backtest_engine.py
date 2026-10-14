"""V24.7 regression tests; run: python -m unittest discover -s ai-research/v247 -p 'test_*.py'"""
import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backtest_engine as eng
from run_validation import validate_candles


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((HERE / 'research_config.json').read_text())

    def candles(self, count=10, price=100):
        return [eng.Candle(1800000000 + i * 60, price, price, price, price, 1) for i in range(count)]

    def signal(self, candles, direction='LONG'):
        return eng.ResearchSignal(candles[0].timestamp, direction, 0.01, 0.02, 2, 0.001)

    def test_safety_lock(self):
        bad = copy.deepcopy(self.config)
        bad['safety']['trading_enabled'] = True
        with self.assertRaises(ValueError):
            eng.BTCBacktestEngine(bad)

    def test_no_duplicate_final_equity_timestamp(self):
        candles = self.candles()
        result = eng.BTCBacktestEngine(self.config).run(candles, [self.signal(candles)])
        times = [p['timestamp'] for p in result['equity_curve']]
        self.assertEqual(len(times), len(set(times)))

    def test_duplicate_exposure_rejected(self):
        candles = self.candles()
        signals = [self.signal(candles), eng.ResearchSignal(candles[1].timestamp, 'LONG', 0.01, 0.02, 2, 0.001)]
        report = eng.BTCBacktestEngine(self.config).run(candles, signals)
        self.assertGreaterEqual(report['rejection_reasons'].get('DUPLICATE_EXPOSURE', 0), 1)

    def test_short_rejected(self):
        candles = self.candles()
        report = eng.BTCBacktestEngine(self.config).run(candles, [self.signal(candles, 'SHORT')])
        self.assertEqual(report['closed_trades'], 0)
        self.assertEqual(report['rejection_reasons'].get('SHORT_NOT_MODELED_WITH_SPOT_PROXY'), 1)

    def test_gapped_candles_rejected(self):
        candles = self.candles()
        with self.assertRaises(ValueError):
            validate_candles(candles[:3] + candles[4:])

    def test_next_candle_entry(self):
        candles = self.candles()
        report = eng.BTCBacktestEngine(self.config).run(candles, [self.signal(candles)])
        self.assertEqual(report['trades'][0]['entry_timestamp'], candles[1].timestamp)


if __name__ == '__main__':
    unittest.main()
