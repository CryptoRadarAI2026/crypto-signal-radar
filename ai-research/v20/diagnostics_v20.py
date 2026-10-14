#!/usr/bin/env python3
"""V20: synthetic-only architecture and backtest integrity suite.

No market files, downloads, holdout access, training, model selection or orders.
V11-V19 remain untouched. Passing does NOT approve a trading model.
"""
from __future__ import annotations
import csv
import json
import math
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).resolve().parent.name == 'v20' else Path.cwd()
REPORT = ROOT / 'reports/v20/integrity.json'
LOOKBACK, HORIZON, EMBARGO = 240, 15, 3600
FEE, SLIPPAGE = .001, .0005

@dataclass(frozen=True)
class Trade:
    net_return: float
    reason: str
    entry_price: float
    exit_price: float
    exit_index: int

@dataclass(frozen=True)
class Window:
    timestamp: int
    features: tuple[float, ...]
    trade: Trade


def simulate_long(o, h, l, c, signal_index: int, atr: float, horizon: int = HORIZON) -> Trade:
    """Explicit five-field typed result; next open, stop-first, both-side costs."""
    i = signal_index
    if i < 0 or horizon < 1 or i + horizon >= len(o) or not math.isfinite(atr) or atr <= 0:
        raise ValueError('Invalid simulation inputs')
    entry = float(o[i+1]) * (1 + SLIPPAGE)
    stop_pct = min(.03, max(.003, 1.6 * atr / entry))
    sl, tp = entry * (1-stop_pct), entry * (1+1.8*stop_pct)
    price, reason, idx = float(c[i+horizon]), 'TIME', i+horizon
    for j in range(i+1, i+horizon+1):
        if j > i+1 and o[j] <= sl:
            price, reason, idx = float(o[j]), 'SL_GAP', j; break
        if j > i+1 and o[j] >= tp:
            price, reason, idx = float(o[j]), 'TP_GAP', j; break
        if l[j] <= sl:
            price, reason, idx = sl, 'SL', j; break
        if h[j] >= tp:
            price, reason, idx = tp, 'TP', j; break
    exit_price = price * (1-SLIPPAGE)
    net = exit_price*(1-FEE)/(entry*(1+FEE))-1
    return Trade(float(net), reason, entry, exit_price, idx)


def read_csv(path: Path):
    """Only call on explicitly authorized train/validation data, never holdout."""
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    required = ('timestamp','open','high','low','close','volume')
    if not rows or any(not all(k in row for k in required) for row in rows):
        raise ValueError('Missing data or required columns')
    ts = np.array([int(r['timestamp']) for r in rows], dtype=np.int64)
    arrays = [np.array([float(r[k]) for r in rows]) for k in required[1:]]
    o,h,l,c,v = arrays
    if np.any(ts % 60) or np.any(np.diff(ts) <= 0):
        raise ValueError('Timestamps must be unique, increasing, UTC minute aligned')
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError('Non-finite values')
    if not (np.all(l>0) and np.all(l<=np.minimum(o,c)) and np.all(h>=np.maximum(o,c)) and np.all(v>=0)):
        raise ValueError('Invalid OHLCV')
    return ts,o,h,l,c,v


def split_boundaries(start_utc: int, end_utc: int) -> tuple[int,int]:
    """Boundaries from fixed UTC clock interval, before CSV parsing/labels."""
    if start_utc % 60 or end_utc % 60 or end_utc <= start_utc:
        raise ValueError('Invalid UTC boundaries')
    span = end_utc-start_utc
    return (start_utc + (span*55//100//60)*60,
            start_utc + (span*80//100//60)*60)


def segment_rows(rows, start: int, end: int):
    """Filter raw rows before features and labels; end is exclusive."""
    ts = rows[0]
    mask = (ts >= start) & (ts < end)
    return tuple(a[mask] for a in rows)


def generate_segment(rows, *, start: int, end: int) -> list[Window]:
    """Only process one pre-separated segment; UTC anchored 15m sampling."""
    ts,o,h,l,c,v = segment_rows(rows,start,end)
    output = []
    if len(ts) <= LOOKBACK + HORIZON:
        return output
    for i in range(LOOKBACK, len(ts)-HORIZON):
        if ts[i] % (HORIZON*60) != 0:
            continue
        if ts[i+HORIZON]-ts[i-LOOKBACK] != (LOOKBACK+HORIZON)*60:
            continue
        if np.any(np.diff(ts[i-LOOKBACK:i+HORIZON+1]) != 60):
            continue
        # Simple past-only features for architecture checks; NOT a trained strategy.
        x = (float(c[i]/c[i-1]-1), float(np.mean(v[i-19:i+1])),
             float(np.std(np.diff(np.log(c[i-60:i+1])))))
        atr = float(np.mean(h[i-13:i+1]-l[i-13:i+1]))
        if not all(map(math.isfinite,x)) or atr <= 0:
            continue
        output.append(Window(int(ts[i]),x,simulate_long(o,h,l,c,i,atr)))
    return output


def prepare_non_holdout(rows, start: int, end: int):
    """Holdout is never materialized, labelled, scored or returned."""
    t1,t2 = split_boundaries(start,end)
    train = generate_segment(rows,start=start,end=t1-EMBARGO)
    validation = generate_segment(rows,start=t1+EMBARGO,end=t2-EMBARGO)
    return train,validation,(t1,t2)


def synthetic(n=6500, gap=None):
    ts = 1704067200 + np.arange(n,dtype=np.int64)*60
    if gap is not None: ts[gap:] += 60
    x = np.arange(n,dtype=float)
    c = 100 + .001*x + .12*np.sin(x/11)
    o = c+.01; h=np.maximum(o,c)+.2; l=np.minimum(o,c)-.2
    return ts,o,h,l,c,10+np.mod(x,7)


def test_utc_anchor():
    a=synthetic(); b=tuple(x[1:] for x in a)
    lo,hi=int(a[0][0]),int(a[0][-1])+60
    x=generate_segment(a,start=lo,end=hi)
    y=generate_segment(b,start=lo,end=hi)
    common=set(w.timestamp for w in x)&set(w.timestamp for w in y)
    assert common and all(t%(HORIZON*60)==0 for t in common)
    assert {w.timestamp for w in y}.issubset({w.timestamp for w in x})
    assert {w.timestamp:w.features for w in x if w.timestamp in common} == {w.timestamp:w.features for w in y if w.timestamp in common}
    return {'aligned_samples':len(x)}


def test_missing_minute():
    a=synthetic(); b=synthetic(gap=3000)
    lo=int(a[0][0]); hi=int(a[0][-1])+60
    baseline=generate_segment(a,start=lo,end=hi)
    gapped=generate_segment(b,start=lo,end=hi)
    assert len(gapped)<len(baseline)
    gap_left=int(b[0][2999]); gap_right=int(b[0][3000])
    assert all(not (w.timestamp-LOOKBACK*60<=gap_left and w.timestamp+HORIZON*60>=gap_right) for w in gapped)
    return {'baseline':len(baseline),'with_gap':len(gapped)}


def test_holdout_isolation():
    a=synthetic(); start=int(a[0][0]); end=int(a[0][-1])+60
    t1,t2=split_boundaries(start,end)
    train,val,_=prepare_non_holdout(a,start,end)
    modified=list(x.copy() for x in a)
    mask=modified[0]>=t2
    for arr in modified[1:]: arr[mask] *= 1.4
    train2,val2,_=prepare_non_holdout(tuple(modified),start,end)
    assert train==train2 and val==val2
    assert train and val and all(w.timestamp<t2-EMBARGO for w in val)
    return {'train':len(train),'validation':len(val),'holdout_processed':False}


def test_stop_first_and_fees():
    ts,o,h,l,c,v=synthetic(1000)
    i=500; o[i+1]=100; h[i+1]=104; l[i+1]=96; c[i+1]=100
    trade=simulate_long(o,h,l,c,i,1.0)
    assert isinstance(trade,Trade) and trade.reason=='SL' and trade.exit_index==i+1
    entry=100*(1+SLIPPAGE)
    stop=entry*(1-min(.03,max(.003,1.6/entry)))
    expected=stop*(1-SLIPPAGE)*(1-FEE)/(entry*(1+FEE))-1
    assert math.isclose(trade.net_return,expected,rel_tol=1e-12)
    return {'reason':trade.reason,'net_return':trade.net_return}


def test_csv_validation():
    a=synthetic(500)
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'synthetic.csv'
        with p.open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(('timestamp','open','high','low','close','volume'))
            writer.writerows(zip(*a))
        assert len(read_csv(p)[0])==500
        bad=list(x.copy() for x in a);bad[0][100]=bad[0][99]
        with p.open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(('timestamp','open','high','low','close','volume'))
            writer.writerows(zip(*bad))
        try: read_csv(p)
        except ValueError: pass
        else: raise AssertionError('Duplicate timestamp accepted')
    return {'duplicate_rejected':True}


def main():
    tests={}; failures=[]
    for fn in (test_utc_anchor,test_missing_minute,test_holdout_isolation,test_stop_first_and_fees,test_csv_validation):
        try: tests[fn.__name__]={'status':'PASS','details':fn()}
        except Exception as exc:
            tests[fn.__name__]={'status':'FAIL','error':repr(exc)}
            failures.append({'test':fn.__name__,'error':repr(exc),'trace':traceback.format_exc(limit=3)})
    report={'schema':'crypto-signal-radar-v20','status':'SYNTHETIC_PASS_NOT_APPROVED' if not failures else 'FAILED',
            'research_only':True,'trading_enabled':False,'market_data_accessed':False,
            'holdout_evaluated':False,'holdout':None,'model_trained':False,
            'validation_independent':False,'tests':tests,'failures':failures,
            'limitations':['V20 tests synthetic architecture only; no real-world profitability evidence.',
                           'Existing V11-V19 unchanged; production pipeline migration requires separate review.',
                           'No holdout data or labels are read; no live or paper orders are submitted.'],
            'generated_utc':datetime.now(timezone.utc).isoformat()}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'tests':tests,'failures':failures},indent=2))
    return int(bool(failures))

if __name__=='__main__':
    raise SystemExit(main())
