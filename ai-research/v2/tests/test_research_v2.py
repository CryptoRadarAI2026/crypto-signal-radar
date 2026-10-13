"""Offline unit tests with synthetic price series only; no network or exchange account."""
from __future__ import annotations
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import research_v2 as v2


class ConservativeFillTests(unittest.TestCase):
    def bars(self, n=32):
        return [np.full(n, 100.0) for _ in range(4)]

    def test_entry_on_next_open_and_costs(self):
        o, h, l, c = self.bars()
        o[1:] = 110.0  # Signal candle closes at 100. Entry must use 110.
        h[1:] = 110.0
        l[1:] = 109.95
        c[1:] = 110.0
        net, why, entry, _ = v2.simulate_long(o, h, l, c, 0, atr=0.1, horizon=2)
        self.assertEqual(why, 'TIME')
        self.assertAlmostEqual(entry, 110.0*(1+v2.SLIPPAGE_PER_SIDE))
        self.assertLess(net, 0)

    def test_stop_first_if_both_hit(self):
        o, h, l, c = self.bars()
        h[1] = 108.0
        l[1] = 90.0
        net, why, _, _ = v2.simulate_long(o, h, l, c, 0, atr=1., horizon=1)
        self.assertEqual(why, 'SL')
        self.assertLess(net, 0)

    def test_take_profit(self):
        o, h, l, c = self.bars()
        h[1] = 108.0
        l[1] = 99.0
        net, why, _, _ = v2.simulate_long(o, h, l, c, 0, atr=1., horizon=1)
        self.assertEqual(why, 'TP')
        self.assertGreater(net, 0)

    def test_gap_worse_than_stop(self):
        o, h, l, c = self.bars()
        l[1] = 99.0
        o[2], h[2], l[2], c[2] = 92., 93., 90., 92.
        net, why, _, exit_price = v2.simulate_long(o, h, l, c, 0, atr=1., horizon=2)
        self.assertEqual(why, 'SL_GAP')
        self.assertAlmostEqual(exit_price, 92 * (1-v2.SLIPPAGE_PER_SIDE))
        self.assertLess(net, -.03)


class SignalSelectionTests(unittest.TestCase):
    @staticmethod
    def sample(ts, pair, ret=.02):
        return v2.Sample(ts, pair, (0.,)*len(v2.FEATURES), int(ret>0), ret, 'TIME', 100, 101)

    def test_only_two_concurrent_and_pair_cooldown(self):
        xs = [self.sample(1800, 'BTC'), self.sample(1800,'ETH'),
              self.sample(1800,'SOL'), self.sample(1860,'BTC'),
              self.sample(3900,'BTC')]
        selected = v2.select_trades(xs, np.array([.9, .8, .7, .95, .99]), .6)
        self.assertEqual([s.pair for s,_ in selected[:2]], ['BTC','ETH'])
        self.assertFalse(any(s.ts == 1860 for s,_ in selected))
        self.assertTrue(any(s.ts == 3900 for s,_ in selected))

    def test_metrics_net_trading(self):
        xs = [self.sample(100,'BTC', .01),self.sample(200,'ETH', -.02)]
        m = v2.metrics([(xs[0], .9),(xs[1], .9)])
        self.assertEqual(m['signals'],2)
        self.assertAlmostEqual(m['paper_pnl_usdt'],-1.)
        self.assertAlmostEqual(m['win_rate_pct'],50.)
        self.assertAlmostEqual(m['profit_factor'],.5)

    def test_metrics_no_signal(self):
        result = v2.metrics([])
        self.assertEqual(result['signals'],0)
        self.assertIsNone(result['win_rate_pct'])
        self.assertEqual(result['ending_paper_equity_usdt'], v2.INITIAL_EQUITY)


class DataSafetyTests(unittest.TestCase):
    @staticmethod
    def synthetic(path, n=4000, gap_at=None):
        rng = np.random.default_rng(7)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, .0007, n)))
        with path.open('w', encoding='utf-8',newline='') as f:
            w = csv.writer(f)
            w.writerow(('timestamp','open','high','low','close','volume'))
            for i, c in enumerate(closes):
                opening = closes[i-1] if i else c
                stamp = (1_700_000_040 + i*60 + (60 if gap_at is not None and i>=gap_at else 0))
                w.writerow((stamp,opening,max(opening,c)*1.002,min(opening,c)*.998,c,100+i%25))

    def test_gap_not_interpolated(self):
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/'candles.csv';self.synthetic(f,gap_at=2000)
            out=v2.samples_from_csv(f,'BTC-USD')
            self.assertGreater(len(out),100)
            self.assertFalse(any(1700000040+2000*60-14*60 < s.ts < 1700000040+2000*60+240*60 for s in out))

    def test_signal_features_past_only_and_contiguous(self):
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/'candles.csv';self.synthetic(f)
            a=v2.samples_from_csv(f,'BTC-USD')
            self.assertGreater(len(a),150)
            self.assertEqual(len(a[0].x), len(v2.FEATURES))
            self.assertTrue(all(math.isfinite(z) for row in a for z in row.x))
            self.assertGreater(a[1].ts, a[0].ts)
            # Change a FUTURE candle; the previous signal features cannot change.
            with f.open('r',encoding='utf-8',newline='') as src:
                records=list(csv.reader(src))
            j=v2.LOOKBACK+1
            records[j+1][4]=str(float(records[j+1][4])*1.01)
            records[j+1][2]=str(max(float(records[j+1][2]),float(records[j+1][4])*1.001))
            records[j+1][3]=str(min(float(records[j+1][3]),float(records[j+1][4])*.999))
            with f.open('w',encoding='utf-8',newline='') as dest:
                csv.writer(dest).writerows(records)
            b=v2.samples_from_csv(f,'BTC-USD')
            self.assertEqual(a[0].x,b[0].x)

    def test_invalid_candle_rejected(self):
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/'candles.csv';self.synthetic(f)
            with f.open('a',encoding='utf-8') as d:
                d.write('1700300000,-1,1,-2,-1,1\n')
            with self.assertRaises(ValueError):v2.samples_from_csv(f,'BTC-USD')

    def test_purged_chronological_split(self):
        rows=[]
        for i in range(700):
            rows.append(self._sample(ts=1800000000+i*900,ret=.02 if i%2 else -.01))
        tr,va,te,_=v2.chronological_split(rows)
        self.assertGreater(min(map(len,(tr,va,te))), 100)
        self.assertGreater(va[0].ts-tr[-1].ts, v2.EMBARGO_SEC)
        self.assertGreater(te[0].ts-va[-1].ts, v2.EMBARGO_SEC)

    @staticmethod
    def _sample(ts,ret):
        return v2.Sample(ts,'BTC-USD',(0.,)*len(v2.FEATURES),int(ret>0),ret,'TIME',100.,100.)

    def test_model_rejected_when_no_edge(self):
        class BadModel:
            def fit(self,x,y): return self
            def predict_proba(self,x):
                return np.tile(np.array([[.9,.1]]), (len(x),1))
        data=[]
        for i in range(700):
            ret=-.005 if i%3 else .01
            data.append(v2.Sample(1800000000+i*900,'BTC-USD',
                (i%7/10.,)+(0.,)*(len(v2.FEATURES)-1),
                int(ret>0),ret,'TIME',100.,100.))
        result=v2.research(data,[],models=[('Rejector',BadModel())])
        self.assertEqual(result['status'],'REJECTED_ON_VALIDATION')
        self.assertNotIn('holdout',result)
        self.assertFalse(result['trading_enabled'])

    def test_dashboard_offline_without_external_code(self):
        report={'status':'REJECTED_ON_VALIDATION','rows':{'train':300,'validation':150,'holdout':150},
                'features':v2.FEATURES,'limitations':['No live orders']}
        with tempfile.TemporaryDirectory() as p:
            path=Path(p)/'report.html'
            v2.make_dashboard(report,path)
            data=path.read_text(encoding='utf-8')
            self.assertIn('REJECTED_ON_VALIDATION',data)
            self.assertIn('dir="rtl"', data)
            self.assertNotIn('<script',data)
            self.assertNotIn('https://',data)


if __name__=='__main__':unittest.main()
