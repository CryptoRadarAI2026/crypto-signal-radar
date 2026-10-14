import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from audit import audit_csv, run


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_csv(self, name, rows):
        path = self.root / name
        with path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            writer.writerows(rows)
        return path

    def test_clean_and_gap(self):
        path = self.write_csv('ok.csv', [[60, 1, 2, 1, 2, 5], [180, 2, 3, 2, 3, 0]])
        result = audit_csv(path)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['missing_minutes_between_observed_bars'], 1)

    def test_invalid_ohlc(self):
        path = self.write_csv('bad.csv', [[60, 2, 1, 1, 2, 5]])
        self.assertTrue(audit_csv(path)['errors'])

    def test_duplicate(self):
        path = self.write_csv('dup.csv', [[60, 1, 2, 1, 2, 5], [60, 1, 2, 1, 2, 5]])
        self.assertTrue(audit_csv(path)['errors'])

    def test_manifest_and_cross_pair(self):
        manifest = []
        for pair, stem in [('BTC-USD', 'BTCUSD'), ('ETH-USD', 'ETHUSD'), ('SOL-USD', 'SOLUSD')]:
            path = self.write_csv(f'COINBASE_{stem}_1.csv', [[60, 1, 2, 1, 2, 5]])
            manifest.append({'pair': pair, 'bars': 1, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest_path = self.root / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        result = run(self.root, manifest_path)
        self.assertTrue(result['integrity_passed'])
        self.assertEqual(result['shared_timestamps'], 1)
        manifest[0]['sha256'] = '0' * 64
        manifest_path.write_text(json.dumps(manifest))
        self.assertFalse(run(self.root, manifest_path)['integrity_passed'])


if __name__ == '__main__':
    unittest.main()
