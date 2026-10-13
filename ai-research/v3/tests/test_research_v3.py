"""No-network tests of V3 diagnostics, hard rejection and scenario path safety."""
from __future__ import annotations
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import research_v3 as v3


class FakeModel:
    def __init__(self, mode='reject'):
        self.mode=mode
        self.calls=[]
    def fit(self,x,y):
        self.calls.append(('fit',len(y)))
        return self
    def predict_proba(self,x):
        self.calls.append(('predict',len(x)))
        p = (np.full(len(x), .10) if self.mode=='reject'
             else np.where(x[:,0]>.5, .95, .05))
        return np.column_stack([1-p,p])


def samples(n=750, successful=False):
    ans=[]
    for i in range(n):
        positive=i%3 == 0
        ret=(.04 if positive else -.008) if successful else (.005 if positive else -.02)
        x=(float(positive),)+(0.,)*(len(v3.FEATURES)-1)
        ans.append(v3.v2.Sample(1700000000+i*900,'BTC-USD',x,int(ret>0),ret,'TIME',100.,100.))
    return ans


class Tests(unittest.TestCase):
    def test_fixed_settings_never_live(self):
        self.assertEqual(v3.v2.ORDER_USDT,100)
        self.assertEqual(v3.v2.MAX_SIMULTANEOUS,2)
        self.assertEqual(v3.MIN_SIGNALS,40)
        self.assertEqual(v3.EMBARGO_SEC,7200)

    def test_purge_separates_alternative_label_horizons(self):
        tr,va,te,_=v3.split_purged(samples())
        self.assertGreater(va[0].ts-tr[-1].ts,v3.EMBARGO_SEC)
        self.assertGreater(te[0].ts-va[-1].ts,v3.EMBARGO_SEC)
        self.assertGreater(len(te),100)

    def test_probability_diagnostics_zero_50pct(self):
        y=np.array([0,1,0,1])
        d=v3.probability_diagnostics(y,np.array([.1,.2,.2,.3]),label='validation')
        self.assertEqual(d['label_positive_rate_pct'],50)
        self.assertIsNone(d['precision_at_50pct_pct'])
        self.assertEqual(d['recall_at_50pct_pct'],0.0)
        self.assertAlmostEqual(d['probability_quantiles']['50pct'],.2)
        self.assertEqual(d['confusion_at_50pct']['fn'],2)

    def test_rejected_model_does_not_touch_holdout(self):
        m=FakeModel('reject')
        report=v3.evaluate(samples(),[],models=[('NoSignal',m)],enable_exploration=False)
        self.assertEqual(report['status'],'REJECTED_ON_VALIDATION')
        self.assertFalse(report['trading_enabled'])
        self.assertFalse(report['model_exported'])
        self.assertFalse(report['holdout_evaluated'])
        self.assertNotIn('holdout',report)
        self.assertEqual([c[0] for c in m.calls],['fit','predict'])
        self.assertIn('NO_SIGNALS',report['validation_candidates'][0]['rejection_reasons'])

    def test_passed_validation_remains_research_only_and_uses_holdout_once(self):
        m=FakeModel('oracle')
        report=v3.evaluate(samples(successful=True),[],models=[('SyntheticOracle',m)],enable_exploration=False)
        self.assertEqual(report['status'],'RESEARCH_ONLY_NOT_APPROVED')
        self.assertTrue(report['holdout_evaluated'])
        self.assertGreater(report['holdout']['signals'],0)
        self.assertFalse(report['trading_enabled'])
        self.assertEqual([c[0] for c in m.calls],['fit','predict','predict'])

    def test_conservative_simulation_matches_v2_primary(self):
        n=100
        o=np.full(n,100.)
        h=np.full(n,101.)
        l=np.full(n,99.)
        c=np.full(n,100.)
        ts=1700000000+np.arange(n,dtype=np.int64)*60
        b=v3.Bars(ts,o,h,l,c,{int(t):i for i,t in enumerate(ts)},np.zeros(n,dtype=int))
        # V2 and V3 primary scenarios use precisely the same fill rule.
        a=v3.v2.simulate_long(o,h,l,c,10,1.,horizon=15)
        z=v3.scenario_fill(b,10,1.,minutes=15,atr_mult=1.6,target_rr=1.8)
        self.assertEqual(a[1],z[1])
        self.assertAlmostEqual(a[0],z[0],places=12)
        self.assertAlmostEqual(a[2],z[2],places=12)

    def test_gap_scenario_not_interpolated(self):
        n=120
        ts=1700000000+np.arange(n,dtype=np.int64)*60
        ts[30:]+=60
        p=np.full(n,100.)
        b=v3.Bars(ts,p,p+1,p-1,p,{int(t):i for i,t in enumerate(ts)},
                  np.r_[0,np.cumsum(np.diff(ts)!=60)])
        self.assertIsNone(v3.scenario_fill(b,15,1.,minutes=30,atr_mult=1.6,target_rr=1.8))
        self.assertIsNotNone(v3.scenario_fill(b,40,1.,minutes=30,atr_mult=1.6,target_rr=1.8))

    def test_long_holding_capacity(self):
        xs=[]
        for i in range(10):
            s=v3.v2.Sample(1700000000+i*900,'BTC-USD' if i%2 else 'ETH-USD',
                           (0.,)*len(v3.FEATURES),1,.01,'TIME',100,101)
            xs.append((s,.9))
        out=v3.enforce_holding_capacity(xs,60)
        for ts in [s.ts for s,_ in xs]:
            active=[s for s,_ in out if s.ts<=ts<s.ts+61*60]
            self.assertLessEqual(len(active),2)

    def test_dashboard_static_and_escapes(self):
        r=v3.evaluate(samples(),[],models=[('bad<script>',FakeModel())],enable_exploration=False)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'dashboard.html'
            v3.dashboard(r,path)
            data=path.read_text(encoding='utf-8')
            self.assertIn('REJECTED_ON_VALIDATION',data)
            self.assertIn('&lt;script&gt;',data)
            self.assertNotIn('<script',data)
            self.assertNotIn('https://',data)

    def test_candidate_gate_flags_loss_and_low_count(self):
        reasons=v3.candidate_reasons({'signals':29,'paper_pnl_usdt':-5.62,'profit_factor':.51})
        self.assertIn('INSUFFICIENT_SIGNALS',reasons)
        self.assertIn('NON_POSITIVE_NET_PNL',reasons)
        self.assertIn('PROFIT_FACTOR_BELOW_MINIMUM',reasons)

if __name__=='__main__':unittest.main()
