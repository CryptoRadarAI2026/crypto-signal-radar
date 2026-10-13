import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from train_public_data import download, evaluate

class ResearchSafetyTests(unittest.TestCase):
    def test_evaluate_zero_signals(self):
        result = evaluate([(0,'BTC-USD',[0]*8,0,-0.01)], [0.1], 0.5)
        self.assertEqual(result['signals'], 0)
        self.assertIsNone(result['win_rate_pct'])

    def test_no_rewrite(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'existing.csv'
            p.write_text('keep', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'overwrite'):
                download('BTC-USD', 0, 60, p)
            self.assertEqual(p.read_text(encoding='utf-8'), 'keep')

    def test_data_sufficiency_fails(self):
        with tempfile.TemporaryDirectory() as folder, patch('train_public_data.fetch_page',return_value=[]), patch('train_public_data.time.sleep', return_value=None):
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                download('BTC-USD', 0, 600, Path(folder)/'test.csv')

if __name__ == '__main__': unittest.main()
