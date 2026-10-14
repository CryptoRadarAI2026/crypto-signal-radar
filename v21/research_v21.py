#!/usr/bin/env python3
"""V21 guarded, research-only CSV preparation. No model fitting or order execution.

Only explicitly supplied local CSV files are accepted. The reader stops at the
holdout boundary without parsing protected rows. No holdout labels or metrics.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import tempfile
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

MINUTE = 60
LOOKBACK = 240
HORIZON = 15
EMBARGO = 3600
GRID = HORIZON * MINUTE
FEE = .001
SLIPPAGE = .0005

@dataclass(frozen=True)
class Candle:
    ts: int
    o: float
    h: float
    l: float
    c: float
    v: float

@dataclass(frozen=True)
class PaperOutcome:
    net_return: float
    reason: str
    entry_price: float
    exit_price: float
    exit_index: int

@dataclass(frozen=True)
class Example:
    ts: int
    features: tuple[float, ...]
    outcome: PaperOutcome


def boundaries(start: int, end: int) -> tuple[int, int]:
    if start % 60 or end % 60 or end <= start or end - start < 12 * 3600:
        raise ValueError('Invalid fixed UTC time interval')
    duration = (end - start) // 60
    return start + (duration * 55 // 100) * 60, start + (duration * 80 // 100) * 60


def parse_candle(row: dict[str, str]) -> Candle:
    t = int(row['timestamp'])
    vals = [float(row[k]) for k in ('open','high','low','close','volume')]
    o,h,l,c,v = vals
    if t % MINUTE or not all(map(math.isfinite, vals)) or not (l > 0 and l <= min(o,c) and h >= max(o,c) and v >= 0):
        raise ValueError('Invalid UTC candle or OHLCV')
    return Candle(t,o,h,l,c,v)


def load_allowed(path: Path, *, start: int, holdout_boundary: int) -> tuple[list[Candle], dict]:
    """Stop at first protected timestamp; NEVER parse the holdout row itself.

    Requires monotonically increasing UTC timestamps; rejects disorder/duplicates
    within permitted rows. No automatic downloading or directory scanning.
    """
    if not path.is_file() or path.suffix.lower() != '.csv':
        raise ValueError('Explicit existing .csv file required')
    candles: list[Candle] = []
    stopped = False
    with path.open('r', newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'timestamp','open','high','low','close','volume'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Missing CSV columns')
        previous = None
        for row in reader:
            # Inspect ONLY timestamp to enforce the hard boundary; never parse OHLCV.
            ts = int(row['timestamp'])
            if ts >= holdout_boundary:
                stopped = True
                break
            if previous is not None and ts <= previous:
                raise ValueError('Non-increasing/duplicate permitted timestamp')
            previous = ts
            if ts >= start:
                candles.append(parse_candle(row))
    return candles, {'permitted_candles': len(candles), 'stopped_before_holdout': stopped,
                     'holdout_rows_parsed': 0}


def simulate_long(bars: list[Candle], i: int, atr: float) -> PaperOutcome:
    if i < 0 or i + HORIZON >= len(bars) or not math.isfinite(atr) or atr <= 0:
        raise ValueError('Invalid simulation window')
    entry = bars[i+1].o * (1 + SLIPPAGE)
    stop_pct = min(.03, max(.003, 1.6 * atr / entry))
    sl, tp = entry * (1-stop_pct), entry * (1+1.8*stop_pct)
    price, reason, exit_i = bars[i+HORIZON].c, 'TIME', i+HORIZON
    for j in range(i+1, i+HORIZON+1):
        b = bars[j]
        if j > i+1 and b.o <= sl:
            price,reason,exit_i = b.o,'SL_GAP',j; break
        if j > i+1 and b.o >= tp:
            price,reason,exit_i = b.o,'TP_GAP',j; break
        if b.l <= sl:
            price,reason,exit_i = sl,'SL',j; break
        if b.h >= tp:
            price,reason,exit_i = tp,'TP',j; break
    exit_fill = price * (1-SLIPPAGE)
    net = exit_fill * (1-FEE)/(entry*(1+FEE)) - 1
    return PaperOutcome(net, reason, entry, exit_fill, exit_i)


def segment_examples(candles: list[Candle], begin: int, end: int) -> list[Example]:
    """Slice FIRST; features and outcomes cannot cross a segment boundary."""
    bars = [b for b in candles if begin <= b.ts < end]
    out: list[Example] = []
    for i in range(LOOKBACK, len(bars)-HORIZON):
        b = bars[i]
        if b.ts % GRID:
            continue
        window = bars[i-LOOKBACK:i+HORIZON+1]
        if any(y.ts-x.ts != MINUTE for x,y in zip(window,window[1:])):
            continue
        closes = [x.c for x in bars[i-60:i+1]]
        returns = [math.log(y/x) for x,y in zip(closes,closes[1:])]
        mean = sum(returns)/len(returns)
        variance = sum((r-mean)**2 for r in returns)/len(returns)
        atr = sum(x.h-x.l for x in bars[i-13:i+1])/14
        features = (b.c/bars[i-1].c-1, sum(x.v for x in bars[i-19:i+1])/20, math.sqrt(variance))
        if atr > 0 and all(map(math.isfinite, features)):
            out.append(Example(b.ts, features, simulate_long(bars,i,atr)))
    return out


def prepare(candles: list[Candle], start: int, end: int):
    t1,t2 = boundaries(start,end)
    # Both segments end before holdout. An embargo is applied at every split.
    train = segment_examples(candles,start,t1-EMBARGO)
    validation = segment_examples(candles,t1+EMBARGO,t2-EMBARGO)
    return train,validation,(t1,t2)


def synthetic(n=6500, gap=None):
    base = 1704067200
    bars = []
    for i in range(n):
        t = base + (i + (1 if gap is not None and i >= gap else 0))*60
        c = 100 + .001*i + .12*math.sin(i/11)
        o = c+.01
        bars.append(Candle(t,o,max(o,c)+.2,min(o,c)-.2,c,10+i%7))
    return bars


def test_utc_alignment():
    a=synthetic(); start=a[0].ts; end=a[-1].ts+60
    first=segment_examples(a,start,end)
    shifted=segment_examples(a[1:],start,end)
    common={s.ts:s.features for s in first if s.ts in {z.ts for z in shifted}}
    assert len(first)>100 and common and all(s.ts%GRID==0 for s in first)
    assert common=={s.ts:s.features for s in shifted}
    return {'samples':len(first)}


def test_missing_minute():
    a=synthetic(); b=synthetic(gap=3000)
    start=a[0].ts; end=a[-1].ts+60
    x=segment_examples(a,start,end); y=segment_examples(b,start,end)
    assert len(y)<len(x)
    assert all(not (z.ts-LOOKBACK*60<=b[2999].ts and z.ts+HORIZON*60>=b[3000].ts) for z in y)
    return {'baseline':len(x),'with_gap':len(y)}


def test_split_and_holdout_mutation():
    a=synthetic(); start=a[0].ts; end=a[-1].ts+60
    tr,va,(_,t2)=prepare(a,start,end)
    changed=[Candle(b.ts,b.o*3,b.h*3,b.l*3,b.c*3,b.v) if b.ts>=t2 else b for b in a]
    tr2,va2,_=prepare(changed,start,end)
    assert tr and va and tr==tr2 and va==va2
    assert max(s.ts for s in va)<t2-EMBARGO
    return {'train':len(tr),'validation':len(va),'holdout_evaluated':False}


def test_reader_never_parses_holdout():
    a=synthetic(1000); t2=a[800].ts
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'data.csv'
        with p.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['timestamp','open','high','low','close','volume'])
            for b in a[:800]: w.writerow((b.ts,b.o,b.h,b.l,b.c,b.v))
            # Invalid protected OHLCV MUST NOT be parsed.
            w.writerow((t2,'SECRET_NOT_A_NUMBER','', '', '', ''))
        rows,info=load_allowed(p,start=a[0].ts,holdout_boundary=t2)
        assert len(rows)==800 and info['stopped_before_holdout'] and info['holdout_rows_parsed']==0
    return info


def test_csv_rejects_duplicate():
    a=synthetic(100)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'data.csv'
        with p.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['timestamp','open','high','low','close','volume'])
            for i,b in enumerate(a):
                w.writerow(((a[i-1].ts if i==40 else b.ts),b.o,b.h,b.l,b.c,b.v))
        try: load_allowed(p,start=a[0].ts,holdout_boundary=a[-1].ts+60)
        except ValueError: return {'duplicate_rejected':True}
        raise AssertionError('Duplicate accepted')


def test_stop_first_fees():
    a=synthetic(1000); i=500
    b=a[i+1]; a[i+1]=Candle(b.ts,100,104,96,100,b.v)
    result=simulate_long(a,i,1.0)
    entry=100*(1+SLIPPAGE); stop=entry*(1-min(.03,max(.003,1.6/entry)))
    expected=stop*(1-SLIPPAGE)*(1-FEE)/(entry*(1+FEE))-1
    assert result.reason=='SL' and result.exit_index==i+1
    assert math.isclose(result.net_return,expected,rel_tol=1e-12)
    return {'reason':result.reason,'net_return':result.net_return}


def main():
    p=argparse.ArgumentParser(description='V21 guarded paper-only data architecture')
    p.add_argument('--csv',type=Path,help='Optional explicit LOCAL CSV; never downloaded')
    p.add_argument('--start-utc',type=int)
    p.add_argument('--end-utc',type=int)
    p.add_argument('--report',type=Path,default=Path('reports/v21/integrity.json'))
    args=p.parse_args()
    tests={}; failures=[]
    for fn in (test_utc_alignment,test_missing_minute,test_split_and_holdout_mutation,
               test_reader_never_parses_holdout,test_csv_rejects_duplicate,test_stop_first_fees):
        try: tests[fn.__name__]={'status':'PASS','details':fn()}
        except Exception as e:
            tests[fn.__name__]={'status':'FAIL','error':repr(e)}
            failures.append({'test':fn.__name__,'error':repr(e),'trace':traceback.format_exc(limit=3)})
    data_summary=None
    if args.csv is not None:
        try:
            if args.start_utc is None or args.end_utc is None: raise ValueError('Both UTC boundaries required')
            _,t2=boundaries(args.start_utc,args.end_utc)
            candles,info=load_allowed(args.csv,start=args.start_utc,holdout_boundary=t2)
            train,validation,_=prepare(candles,args.start_utc,args.end_utc)
            data_summary={**info,'train_samples':len(train),'validation_samples':len(validation),
                          'holdout_samples':None,'validation_performance':None}
        except Exception as e:
            failures.append({'test':'explicit_csv_preparation','error':repr(e)})
    elif args.start_utc is not None or args.end_utc is not None:
        failures.append({'test':'arguments','error':'UTC boundaries require --csv'})
    report={'schema':'crypto-signal-radar-v21','status':'INTEGRITY_PASS_NOT_APPROVED' if not failures else 'FAILED',
            'research_only':True,'trading_enabled':False,'orders_submitted':False,
            'holdout_evaluated':False,'holdout':None,'model_trained':False,
            'validation_independent':False,'tests':tests,'failures':failures,
            'data_summary':data_summary,'market_data_accessed':args.csv is not None,
            'limitations':['Synthetic tests do not prove profitability or all-path absence of leakage.',
                           'V11-V20 remain unchanged; V21 is a separate preparation pipeline.',
                           'No model fitting, parameter tuning, holdout scoring or real orders.',
                           'Optional CSV must be chronologically ordered; protected OHLCV is never parsed.'],
            'generated_utc':datetime.now(timezone.utc).isoformat()}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'tests':tests,'failures':failures},indent=2))
    return bool(failures)

if __name__=='__main__':
    raise SystemExit(main())
