#!/usr/bin/env python3
"""V18 conservative OHLCV and historical validation audit.

Research only. No real orders. No holdout inspection, predictions or scoring.
The chronological validation slices are NOT a fresh independent holdout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PAIRS = ('BTC-USD', 'ETH-USD', 'SOL-USD')
SECONDS_PER_DAY = 86400
MINUTE = 60
DAYS = 90
FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0005


def iso(t):
    return datetime.fromtimestamp(int(t), timezone.utc).isoformat()


def scan_csv(path, expected_start, audit_end):
    """Read strictly before audit_end; never parse or assess holdout rows."""
    seen = set()
    first = last = None
    duplicates = 0
    invalid = []
    out_of_order = 0
    previous = None
    checked = 0
    digest = hashlib.sha256()
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing columns: {path}')
        for line_no, row in enumerate(reader, 2):
            t = int(row['timestamp'])
            if t >= audit_end:
                # Source downloader is expected to return ordered timestamps.
                # Stop before reading any holdout OHLCV values.
                break
            if t < expected_start:
                raise ValueError(f'Unexpected early timestamp in {path}: {t}')
            checked += 1
            digest.update((str(t) + '\n').encode())
            if previous is not None and t < previous:
                out_of_order += 1
            previous = t
            if t in seen:
                duplicates += 1
                continue
            seen.add(t)
            first = t if first is None else min(first, t)
            last = t if last is None else max(last, t)
            try:
                o, h, l, c, v = (float(row[k]) for k in ('open', 'high', 'low', 'close', 'volume'))
                if (not all(map(math.isfinite, (o, h, l, c, v))) or
                        l <= 0 or v < 0 or l > min(o, c) or h < max(o, c) or h < l):
                    raise ValueError('invalid OHLCV')
            except (ValueError, TypeError):
                invalid.append({'line': line_no, 'timestamp': t})
    expected = set(range(expected_start, audit_end, MINUTE))
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    return {
        'source_file': path.name,
        'rows_checked_pre_holdout': checked,
        'unique_timestamps': len(seen),
        'duplicate_timestamp_rows': duplicates,
        'out_of_order_rows': out_of_order,
        'invalid_ohlcv_rows': len(invalid),
        'invalid_ohlcv_examples': invalid[:10],
        'missing_minutes': len(missing),
        'missing_timestamp_examples_utc': [iso(t) for t in missing[:12]],
        'unexpected_minutes': len(unexpected),
        'first_timestamp_utc': iso(first) if first is not None else None,
        'last_timestamp_utc': iso(last) if last is not None else None,
        'timestamp_sha256': digest.hexdigest(),
    }, seen


def summary(rows):
    vals = [float(r['net_usdt']) for r in rows]
    gross = sum(float(r['gross_usdt']) for r in rows)
    slip = sum(float(r['slippage_usdt']) for r in rows)
    fees = sum(float(r['fees_usdt']) for r in rows)
    return {
        'trades': len(rows),
        'wins': sum(x > 0 for x in vals),
        'net_usdt': round(sum(vals), 6),
        'gross_usdt': round(gross, 6),
        'slippage_usdt': round(slip, 6),
        'fees_usdt': round(fees, 6),
        'avg_net_usdt': round(sum(vals) / len(vals), 6) if vals else None,
        'profit_factor': (round(sum(x for x in vals if x > 0) /
                                -sum(x for x in vals if x < 0), 6)
                          if any(x < 0 for x in vals) else None),
    }


def bootstrap_by_day(rows, iterations=2000, seed=18):
    """Exploratory cluster bootstrap; not an independent out-of-sample test."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['signal_utc'][:10]].append(row)
    days = sorted(grouped)
    if len(days) < 3:
        return {'status': 'INSUFFICIENT_DISTINCT_DAYS', 'days': len(days)}
    rng = random.Random(seed)
    pnl = []
    for _ in range(iterations):
        chosen = [rng.choice(days) for _ in days]
        pnl.append(sum(float(r['net_usdt']) for d in chosen for r in grouped[d]))
    pnl.sort()
    return {
        'status': 'EXPLORATORY_DEPENDENT_VALIDATION_ONLY',
        'distinct_days': len(days), 'iterations': iterations, 'seed': seed,
        'total_net_usdt_bootstrap_percentiles': {
            'p025': round(pnl[int(iterations * .025)], 6),
            'p50': round(pnl[int(iterations * .5)], 6),
            'p975': round(pnl[min(iterations - 1, int(iterations * .975))], 6),
        },
        'note': 'Resampling days of reused validation does not establish generalization.',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    p.add_argument('--v16-report', type=Path,
                   default=Path('reports/v16/trade_exit_diagnostics.json'))
    p.add_argument('--output', type=Path,
                   default=Path('reports/v18/data_integrity_and_validation.json'))
    p.add_argument('--end', type=int, default=None,
                   help='UTC-aligned exclusive download end; required to match downloader')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite existing report: {args.output}')
    if args.end is None or args.end % SECONDS_PER_DAY:
        raise ValueError('Pass --end as UTC-midnight Unix timestamp used by downloader')
    start = args.end - DAYS * SECONDS_PER_DAY
    # Audit 80% of the 90-day interval; last 20% is protected holdout.
    holdout_start = start + int(DAYS * SECONDS_PER_DAY * .8)
    if (holdout_start - start) % MINUTE:
        raise ValueError('Audit boundary not aligned to minute')
    checks, timestamp_sets = {}, {}
    for pair in PAIRS:
        path = args.data_dir / f"COINBASE_{pair.replace('-', '')}_90D_1M.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        checks[pair], timestamp_sets[pair] = scan_csv(path, start, holdout_start)
        print(pair, 'pre-holdout missing', checks[pair]['missing_minutes'], flush=True)

    btc, eth, sol = (timestamp_sets[x] for x in PAIRS)
    discrepancies = {
        'sol_missing_when_btc_and_eth_present': sorted((btc & eth) - sol),
        'btc_missing_when_eth_and_sol_present': sorted((eth & sol) - btc),
        'eth_missing_when_btc_and_sol_present': sorted((btc & sol) - eth),
    }
    comparison = {name: {
        'count': len(times), 'examples_utc': [iso(t) for t in times[:15]]
    } for name, times in discrepancies.items()}

    source = json.loads(args.v16_report.read_text(encoding='utf-8'))
    if (source.get('schema') != 'crypto-radar-ai-research-v16' or
            source.get('holdout_evaluated') is not False or
            source.get('holdout') is not None or
            source.get('trading_enabled') is not False or
            source.get('status') != 'EXPLORATORY_VALIDATION_ONLY_NOT_APPROVED'):
        raise ValueError('Unsafe or unexpected V16 report')
    rows = source['baseline_trades']
    if not isinstance(rows, list) or len(rows) != source['baseline_trade_count']:
        raise ValueError('Invalid baseline trade rows')
    rows = sorted(rows, key=lambda r: (r['signal_utc'], r['pair']))
    for r in rows:
        if r['pair'] not in PAIRS:
            raise ValueError('Unknown pair')
        if not all(math.isfinite(float(r[k])) for k in
                   ('net_usdt', 'gross_usdt', 'slippage_usdt', 'fees_usdt')):
            raise ValueError('Nonfinite PnL component')
        if not math.isclose(float(r['gross_usdt']) + float(r['slippage_usdt']) +
                            float(r['fees_usdt']), float(r['net_usdt']),
                            abs_tol=.00001):
            raise ValueError('Trade cost accounting mismatch')
        if float(r['slippage_usdt']) > 0 or float(r['fees_usdt']) > 0:
            raise ValueError('Unexpected positive trading cost')
        t = datetime.fromisoformat(r['signal_utc']).timestamp()
        if t >= holdout_start:
            raise ValueError('V16 trade entered holdout')
    overall = summary(rows)
    expected_net = source['scenarios']['A_baseline']['fixed_v15_entries']['net_usdt']
    if not math.isclose(overall['net_usdt'], float(expected_net), abs_tol=.0001):
        raise ValueError('Baseline PnL mismatch')
    # Chronological validation-only slices, not an untouched OOS estimate.
    midpoint = len(rows) // 2
    slices = {
        'earlier_validation_trades': summary(rows[:midpoint]),
        'later_validation_trades': summary(rows[midpoint:]),
    }
    report = {
        'schema': 'crypto-radar-ai-research-v18',
        'status': 'DATA_AUDIT_AND_REUSED_VALIDATION_ONLY_NOT_APPROVED',
        'research_only': True, 'trading_enabled': False,
        'holdout_evaluated': False, 'holdout': None,
        'download_window_utc': {'start': iso(start), 'end_exclusive': iso(args.end)},
        'audited_pre_holdout_window_utc': {
            'start': iso(start), 'end_exclusive': iso(holdout_start)},
        'holdout_policy': 'Last 20% of raw candle time range excluded from OHLCV parsing and diagnostics. No holdout evaluation.',
        'data_quality': checks,
        'cross_pair_missing_minute_comparison': comparison,
        'sol_six_candle_question': {
            'historical_observation': 'Previous downloader run showed SOL 129593 bars versus 129599 for BTC/ETH over full 90 days.',
            'pre_holdout_sol_missing_vs_both': comparison['sol_missing_when_btc_and_eth_present']['count'],
            'can_explain_full_window_six_without_holdout_access': False,
            'note': 'Full-window six-bar difference cannot be located or explained without accessing protected holdout; only pre-holdout is audited.',
        },
        'validation_paper_trades': {
            'model': source['model'], 'threshold': source['threshold'],
            'assumed_order_usdt': 100,
            'fee_per_side_assumed': FEE_PER_SIDE,
            'slippage_per_side_assumed': SLIPPAGE_PER_SIDE,
            'overall': overall, 'chronological_slices': slices,
            'day_cluster_bootstrap': bootstrap_by_day(rows),
        },
        'leakage_controls': {
            'no_new_model_training_or_tuning': True,
            'no_holdout_prediction_or_scoring': True,
            'trade_provenance': 'V16 uses next-candle entries, past-only features, embargo, and stop-first intrabar convention; V18 checks report accounting but does not formally prove V11 has no lookahead.',
            'raw_candles_restricted_to_pre_holdout': True,
        },
        'out_of_sample_assessment': {
            'status': 'NOT_PERFORMED',
            'reason': 'Independent holdout is protected and must not be evaluated. Chronological subsets of previously reused validation are descriptive, not independent OOS.',
        },
        'limitations': [
            'The validation data was repeatedly used in V11-V17; no fresh independent performance estimate is available.',
            'No strategy/model approval and no live orders.',
            'Costs are modeled assumptions, not exchange execution receipts.',
            'Quality scan checks duplicates, gaps, ordering, OHLCV sanity; it does not independently verify exchange prices.',
            'The six-bar full-window SOL discrepancy cannot be localized without touching protected holdout.',
            'Statistical bootstrap is exploratory and cannot repair selection bias or repeated validation reuse.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('V18 report:', args.output, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import sys
        print(f'V18_FAILED_SAFELY: {exc!r}', file=sys.stderr)
        raise SystemExit(2)
