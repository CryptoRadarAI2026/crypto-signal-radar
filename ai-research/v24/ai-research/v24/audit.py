#!/usr/bin/env python3
"""V24.1 research-only OHLCV audit. Standard library only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

PAIRS = {'BTC-USD': 'BTCUSD', 'ETH-USD': 'ETHUSD', 'SOL-USD': 'SOLUSD'}
COLUMNS = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}


def audit_csv(path):
    errors = []
    timestamps = []
    previous = None
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f'Missing OHLCV columns: {path}')
        for line, row in enumerate(reader, start=2):
            try:
                t = int(row['timestamp'])
                values = [float(row[k]) for k in ('open', 'high', 'low', 'close', 'volume')]
                o, h, l, c, v = values
                if t % 60 or not all(math.isfinite(x) for x in values):
                    raise ValueError('invalid timestamp alignment or nonfinite number')
                if min(o, h, l, c) <= 0 or v < 0 or h < max(o, c, l) or l > min(o, c, h):
                    raise ValueError('invalid OHLCV relationship')
                if previous is not None and t <= previous:
                    raise ValueError('duplicate or out-of-order timestamp')
                timestamps.append(t)
                previous = t
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(f'line {line}: {exc}')
                if len(errors) >= 20:
                    break
    gaps = sum(max(0, (b-a)//60 - 1) for a, b in zip(timestamps, timestamps[1:]))
    return {'bars': len(timestamps), 'first_timestamp': timestamps[0] if timestamps else None,
            'last_timestamp': timestamps[-1] if timestamps else None,
            'missing_minutes_between_observed_bars': gaps, 'errors': errors,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'timestamps': set(timestamps)}


def run(raw_dir, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    by_pair = {item['pair']: item for item in manifest}
    results = {}
    for pair, stem in PAIRS.items():
        path = raw_dir / f'COINBASE_{stem}_1.csv'
        if pair not in by_pair or not path.is_file():
            raise ValueError(f'Missing manifest entry or CSV: {pair}')
        result = audit_csv(path)
        expected = by_pair[pair]
        if result['sha256'] != expected['sha256']:
            result['errors'].append('SHA256 does not match manifest')
        if result['bars'] != expected['bars']:
            result['errors'].append('Bar count does not match manifest')
        results[pair] = result
    shared = set.intersection(*(r['timestamps'] for r in results.values()))
    output = {'status': 'RESEARCH_ONLY_NOT_APPROVED', 'trading_enabled': False,
              'shared_timestamps': len(shared), 'pairs': {}}
    failed = False
    for pair, result in results.items():
        result['bars_not_shared_with_all_pairs'] = len(result['timestamps'] - shared)
        result.pop('timestamps')
        output['pairs'][pair] = result
        failed |= bool(result['errors'])
    output['integrity_passed'] = not failed
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--manifest', default='reports/source_manifest.json')
    parser.add_argument('--output', default='reports/v24/audit_v241.json')
    args = parser.parse_args()
    result = run(Path(args.raw_dir), Path(args.manifest))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['integrity_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
