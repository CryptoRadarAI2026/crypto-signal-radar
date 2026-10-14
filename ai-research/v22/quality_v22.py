#!/usr/bin/env python3
"""V22 research-only, guarded pre-holdout CSV quality audit. No trades or training.

CSV format: timestamp (UTC epoch seconds), open, high, low, close, volume.
Input must be explicitly supplied, chronologically ordered and one-minute bars.
Only the timestamp of the first protected row is inspected; protected OHLCV
is never parsed. An untrusted CSV cannot establish the holdout boundary.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MINUTE = 60
SYMBOLS = ('BTC', 'ETH', 'SOL')


def validate_bounds(start: int, holdout: int) -> None:
    if start < 0 or holdout <= start or start % 60 or holdout % 60:
        raise ValueError('Require aligned UTC start and protected holdout boundary')


def audit_csv(path: Path, start: int, holdout: int) -> dict:
    validate_bounds(start, holdout)
    if not path.is_file() or path.suffix.lower() != '.csv':
        raise ValueError('Explicit local .csv file required')
    required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    count = 0
    missing = 0
    first = last = previous = None
    stopped = False
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or not required.issubset(reader.fieldnames):
            raise ValueError('Missing or duplicate CSV headers')
        for row in reader:
            # Read ONLY timestamp to detect the first protected row.
            ts = int(row['timestamp'])
            if ts >= holdout:
                stopped = True
                break
            if ts < start:
                raise ValueError('Input contains timestamps before configured start')
            if ts % 60:
                raise ValueError('Non-minute UTC timestamp')
            if previous is not None:
                if ts <= previous:
                    raise ValueError('Duplicate or out-of-order timestamp')
                missing += (ts - previous) // 60 - 1
            vals = [float(row[k]) for k in ('open', 'high', 'low', 'close', 'volume')]
            o, h, l, c, v = vals
            if not all(math.isfinite(x) for x in vals) or not (0 < l <= min(o, c) <= max(o, c) <= h and v >= 0):
                raise ValueError('Invalid OHLCV')
            first = ts if first is None else first
            last, previous = ts, ts
            count += 1
    # Leading/trailing gaps inside the explicitly allowed interval.
    if count:
        missing += (first - start) // 60
        missing += (holdout - last) // 60 - 1
    else:
        missing = (holdout - start) // 60
    return {'permitted_rows': count, 'missing_minutes': missing,
            'first_permitted_utc': first, 'last_permitted_utc': last,
            'stopped_at_protected_boundary': stopped,
            'protected_ohlcv_parsed': False, 'holdout_evaluated': False}


def write_csv(path: Path, timestamps: list[int], *, invalid_at: int | None = None) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for i, ts in enumerate(timestamps):
            if invalid_at is not None and i == invalid_at:
                w.writerow([ts, 'PROTECTED_SECRET', '', '', '', ''])
            else:
                w.writerow([ts, 100, 102, 99, 101, 12])


def expect_rejection(path: Path, start: int, holdout: int) -> None:
    try:
        audit_csv(path, start, holdout)
    except ValueError:
        return
    raise AssertionError('Expected CSV rejection')


def synthetic_tests() -> dict:
    tests = {}
    base = 1704067200
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'candles.csv'
        def check(name, fn):
            try:
                details = fn()
                tests[name] = {'status': 'PASS', 'details': details}
            except Exception as exc:
                tests[name] = {'status': 'FAIL', 'error': repr(exc)}
        def complete():
            write_csv(p, [base + 60 * i for i in range(10)])
            r = audit_csv(p, base, base + 600)
            assert r['permitted_rows'] == 10 and r['missing_minutes'] == 0
            return r
        check('complete_utc_minutes', complete)
        def gap():
            write_csv(p, [base + 60 * i for i in range(10) if i != 4])
            r = audit_csv(p, base, base + 600)
            assert r['missing_minutes'] == 1 and r['permitted_rows'] == 9
            return {'missing_minutes': 1}
        check('missing_minute_detected', gap)
        def duplicate():
            write_csv(p, [base, base + 60, base + 60])
            expect_rejection(p, base, base + 600)
            return {'rejected': True}
        check('duplicate_rejected', duplicate)
        def disorder():
            write_csv(p, [base, base + 120, base + 60])
            expect_rejection(p, base, base + 600)
            return {'rejected': True}
        check('out_of_order_rejected', disorder)
        def invalid():
            write_csv(p, [base])
            lines = p.read_text().replace('100,102,99,101,12', '100,98,99,101,12')
            p.write_text(lines)
            expect_rejection(p, base, base + 600)
            return {'rejected': True}
        check('invalid_ohlcv_rejected', invalid)
        def protected():
            write_csv(p, [base, base + 60, base + 120], invalid_at=2)
            r = audit_csv(p, base, base + 120)
            assert r['permitted_rows'] == 2 and r['stopped_at_protected_boundary']
            assert not r['protected_ohlcv_parsed']
            return {'protected_rows_parsed': 0}
        check('protected_ohlcv_not_parsed', protected)
        def boundary():
            write_csv(p, [base])
            for start, end in ((base + 1, base + 600), (base, base), (base, base + 61)):
                expect_rejection(p, start, end)
            return {'invalid_boundaries_rejected': 3}
        check('invalid_boundaries_rejected', boundary)
        def no_csv():
            try:
                audit_csv(Path(d) / 'absent.csv', base, base + 600)
            except ValueError:
                return {'missing_file_rejected': True}
            raise AssertionError('Missing CSV accepted')
        check('missing_file_rejected', no_csv)
    return tests


def main() -> int:
    parser = argparse.ArgumentParser(description='V22 pre-holdout research-only quality audit')
    parser.add_argument('--btc', type=Path)
    parser.add_argument('--eth', type=Path)
    parser.add_argument('--sol', type=Path)
    parser.add_argument('--start-utc', type=int)
    parser.add_argument('--holdout-start-utc', type=int)
    parser.add_argument('--report', type=Path, default=Path('reports/v22/quality.json'))
    args = parser.parse_args()
    tests = synthetic_tests()
    failures = [name for name, result in tests.items() if result['status'] != 'PASS']
    files = {'BTC': args.btc, 'ETH': args.eth, 'SOL': args.sol}
    data = None
    if any(p is not None for p in files.values()):
        data = {}
        try:
            if args.start_utc is None or args.holdout_start_utc is None:
                raise ValueError('Both explicit UTC boundaries required for local CSV audit')
            validate_bounds(args.start_utc, args.holdout_start_utc)
            for symbol, path in files.items():
                if path is not None:
                    data[symbol] = audit_csv(path, args.start_utc, args.holdout_start_utc)
        except Exception as exc:
            failures.append('csv_audit: ' + repr(exc))
    elif args.start_utc is not None or args.holdout_start_utc is not None:
        failures.append('UTC boundaries supplied without CSV')
    report = {
        'schema': 'crypto-signal-radar-v22',
        'status': 'QUALITY_CHECK_PASS_NOT_APPROVED' if not failures else 'FAILED',
        'research_only': True, 'trading_enabled': False, 'orders_submitted': False,
        'model_trained': False, 'holdout_evaluated': False, 'holdout': None,
        'validation_independent': False, 'tests': tests, 'failures': failures,
        'market_data_accessed': data is not None, 'data_quality': data,
        'limitations': [
            'Workflow runs synthetic tests only; optional local CSV audit requires explicit files and UTC boundaries.',
            'No performance evaluation, training, tuning, labels, or order execution.',
            'Protected row timestamps are inspected only to stop reading; protected OHLCV is not parsed.',
            'CSV must be chronologically ordered; rows after the first protected timestamp are not inspected.',
            'A user-provided holdout boundary must be independently verified before use.',
            'V11-V21 are unchanged; no production pipeline migration is implied.'
        ],
        'generated_utc': datetime.now(timezone.utc).isoformat()
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'tests': tests, 'failures': failures}, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
