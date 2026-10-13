import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('research_v4', HERE / 'research_v4.py')
v4 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v4
spec.loader.exec_module(v4)


def result(n, pnl, pf):
    return {'signals': n, 'paper_pnl_usdt': pnl, 'profit_factor': pf}


class V4SafetyTests(unittest.TestCase):
    def test_reject_negative_aggregate(self):
        folds = [result(20, 2, 1.2)] * 3
        self.assertIn('NON_POSITIVE_TOTAL_PNL', v4.decision(folds, result(60, -1, .9)))

    def test_reject_bad_fold_even_when_aggregate_profitable(self):
        folds = [result(20, 3, 1.3), result(20, -1, .8), result(20, 3, 1.3)]
        self.assertIn('FOLD_2_NON_POSITIVE_PNL', v4.decision(folds, result(60, 5, 1.3)))

    def test_reject_too_few_trades(self):
        folds = [result(9, 1, 1.2)] * 3
        reasons = v4.decision(folds, result(27, 3, 1.2))
        self.assertIn('INSUFFICIENT_TOTAL_TRADES', reasons)
        self.assertIn('FOLD_1_TOO_FEW_TRADES', reasons)

    def test_pass_only_when_all_conditions_met(self):
        folds = [result(20, 2, 1.2)] * 3
        self.assertEqual(v4.decision(folds, result(60, 6, 1.2)), [])

    def test_validation_blocks_are_chronological(self):
        data = [SimpleNamespace(ts=i * 60, pair='BTC-USD') for i in range(300)]
        blocks = v4.validation_blocks(data)
        self.assertEqual([len(b) for b in blocks], [100, 100, 100])
        self.assertLess(blocks[0][-1].ts, blocks[1][0].ts)

    def test_no_trading_api_in_module(self):
        src = (HERE / 'research_v4.py').read_text(encoding='utf-8')
        self.assertNotIn('create_order(', src)
        self.assertNotIn('place_order(', src)


if __name__ == '__main__':
    unittest.main()
