#!/usr/bin/env python3
"""V23: guarded, research-only pre-holdout BTC/ETH/SOL quality audit.

No network, model fitting, trading, holdout labels or performance scoring.
An independently approved holdout boundary is mandatory for real CSV audits.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ('timestamp', 'open', 'high', 'low', 'close', 'volume')
SYMBOLS = ('BTC', 'ETH', 'SOL')


def audit(path: Path, start: int, cutoff: int) -> dict:
    if start < 0 or cutoff <= start or start % 60 or cutoff % 60:
        raise ValueError('Invalid externally verified UTC boundaries')
    if not path.is_file() or path.suffix.lower() != '.csv':
        raise ValueError('An existing explicit CSV path is required')
    count = missing = 0
    first = last = prev = None
    stopped = False
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError('Missing/duplicate CSV headers')
        for row in reader:
            # Only inspect timestamp of the first protected row. Never parse its OHLCV.
            ts = int(row['timestamp'])
            if ts >= cutoff:
                stopped = True
                break
            if ts < start or ts % 60:
                raise ValueError('Out-of-range or non-minute UTC timestamp')
            if prev is not None:
                if ts <= prev:
                    raise ValueError('Duplicate/out-of-order permitted timestamp')
                missing += (ts-prev)//60 - 1
            values = [float(row[k]) for k in FIELDS[1:]]
            o,h,l,c,v = values
            if not all(map(math.isfinite, values)) or not (0 < l <= min(o,c) <= max(o,c) <= h and v >= 0):
                raise ValueError('Invalid permitted OHLCV')
            if first is None:
                first = ts
            prev = last = ts
            count += 1
    missing += (first-start)//60 + (cutoff-last)//60-1 if count else (cutoff-start)//60
    return {'permitted_rows': count, 'missing_minutes': missing,
            'first_permitted_utc': first, 'last_permitted_utc': last,
            'stopped_at_boundary': stopped, 'protected_ohlcv_parsed': False,
            'holdout_evaluated': False}


def write_fixture(path, timestamps, protected=None):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for ts in timestamps:
            writer.writerow((ts,100,102,99,101,10))
        if protected is not None:
            writer.writerow((protected,'SECRET','SECRET','SECRET','SECRET','SECRET'))


def tests():
    base = 1704067200
    results = {}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'fixture.csv'
        def check(name, fn):
            try:
                results[name] = {'status':'PASS','details':fn()}
            except Exception as exc:
                results[name] = {'status':'FAIL','error':repr(exc)}
        def complete():
            write_fixture(path,[base+60*i for i in range(20)],base+1200)
            r=audit(path,base,base+1200)
            assert r['missing_minutes']==0 and r['permitted_rows']==20 and r['stopped_at_boundary']
            return r
        check('complete_minutes',complete)
        def six_gaps():
            write_fixture(path,[base+60*i for i in range(20) if i not in (3,4,5,10,11,15)])
            r=audit(path,base,base+1200)
            assert r['missing_minutes']==6
            return {'missing_minutes':6}
        check('six_missing_minutes_detected',six_gaps)
        def protected():
            write_fixture(path,[base,base+60],base+120)
            r=audit(path,base,base+120)
            assert r['permitted_rows']==2 and r['protected_ohlcv_parsed'] is False
            return {'protected_ohlcv_parsed':False}
        check('protected_ohlcv_not_parsed',protected)
        def reject(name, times, start=base, end=base+1200):
            write_fixture(path,times)
            try:
                audit(path,start,end)
            except ValueError:
                return {'rejected':True}
            raise AssertionError(name+' accepted')
        check('duplicate_rejected',lambda:reject('duplicate',[base,base+60,base+60]))
        check('disorder_rejected',lambda:reject('disorder',[base,base+120,base+60]))
        check('unaligned_rejected',lambda:reject('unaligned',[base+1]))
        check('invalid_boundary_rejected',lambda:reject('boundary',[base],base,base))
        def bad_ohlcv():
            path.write_text(','.join(FIELDS)+'\n'+str(base)+',100,98,99,101,10\n')
            try: audit(path,base,base+120)
            except ValueError: return {'rejected':True}
            raise AssertionError('Invalid OHLCV accepted')
        check('invalid_ohlcv_rejected',bad_ohlcv)
    return results


def main():
    p=argparse.ArgumentParser(description='V23 research-only quality audit; never uses holdout OHLCV')
    for symbol in SYMBOLS:
        p.add_argument('--'+symbol.lower(),type=Path)
    p.add_argument('--start-utc',type=int)
    p.add_argument('--holdout-start-utc',type=int,
                   help='Externally verified UTC boundary, NOT inferred from CSV')
    p.add_argument('--report',type=Path,default=Path('reports/v23/quality.json'))
    a=p.parse_args()
    outcomes=tests()
    failures=[k for k,v in outcomes.items() if v['status']!='PASS']
    provided={s:getattr(a,s.lower()) for s in SYMBOLS if getattr(a,s.lower()) is not None}
    data=None
    if provided:
        data={}
        if a.start_utc is None or a.holdout_start_utc is None:
            failures.append('Real CSV requires independently verified --start-utc and --holdout-start-utc')
        else:
            for symbol,path in provided.items():
                try: data[symbol]=audit(path,a.start_utc,a.holdout_start_utc)
                except Exception as exc: failures.append(symbol+': '+repr(exc))
    elif a.start_utc is not None or a.holdout_start_utc is not None:
        failures.append('Boundaries without CSV files')
    report={'schema':'crypto-signal-radar-v23','status':'QUALITY_PASS_NOT_APPROVED' if not failures else 'FAILED',
            'research_only':True,'trading_enabled':False,'orders_submitted':False,
            'model_trained':False,'holdout_evaluated':False,'holdout':None,
            'validation_independent':False,'market_data_accessed':bool(provided),
            'tests':outcomes,'failures':failures,'data_quality':data,
            'limitations':['Workflow runs synthetic tests only; real CSV requires explicit paths and independently verified boundaries.',
                           'Does not verify that the provided boundary is authentic.',
                           'Only first protected timestamp may be inspected; no protected OHLCV parsed.',
                           'No model evaluation, profitability claim, network access or order submission.',
                           'V11-V22 remain unchanged.'],
            'generated_utc':datetime.now(timezone.utc).isoformat()}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'tests':outcomes,'failures':failures},indent=2))
    return 1 if failures else 0

if __name__=='__main__':
    raise SystemExit(main())
