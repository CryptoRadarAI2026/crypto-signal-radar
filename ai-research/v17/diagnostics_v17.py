#!/usr/bin/env python3
"""V17: descriptive validation-only analysis of V16 baseline trade quality.

No filters are selected, tuned, backtested or approved. No holdout access.
Research only; never places orders.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V16 = ROOT / 'reports/v16/trade_exit_diagnostics.json'
DEFAULT_OUTPUT = ROOT / 'reports/v17/trade_quality_diagnostics.json'


def group_summary(rows):
    n = len(rows)
    net = sum(float(r['net_usdt']) for r in rows)
    gross = sum(float(r['gross_usdt']) for r in rows)
    return {
        'trades': n,
        'wins': sum(float(r['net_usdt']) > 0 for r in rows),
        'win_rate_pct': round(100 * sum(float(r['net_usdt']) > 0 for r in rows) / n, 3) if n else None,
        'gross_usdt': round(gross, 6),
        'net_usdt': round(net, 6),
        'avg_net_usdt': round(net / n, 6) if n else None,
        'exit_reasons': dict(sorted(Counter(r['exit_reason'] for r in rows).items())),
    }


def bins(rows, field, edges, labels):
    if len(labels) != len(edges) - 1:
        raise ValueError('Bin labels do not match edges')
    groups = {label: [] for label in labels}
    for r in rows:
        value = float(r[field])
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite {field}')
        for i, label in enumerate(labels):
            if edges[i] <= value < edges[i + 1]:
                groups[label].append(r)
                break
        else:
            raise ValueError(f'Out of range {field}: {value}')
    return {label: group_summary(group) for label, group in groups.items()}


def pearson(rows, field):
    if len(rows) < 3:
        return None
    xs = [float(r[field]) for r in rows]
    ys = [float(r['net_usdt']) for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return round(numerator / denominator, 6) if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v16-report', type=Path, default=DEFAULT_V16)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite: {args.output}')
    if args.output.resolve() == args.v16_report.resolve():
        raise ValueError('Input and output must differ')
    source = json.loads(args.v16_report.read_text(encoding='utf-8'))
    if (source.get('schema') != 'crypto-radar-ai-research-v16'
            or source.get('holdout_evaluated') is not False
            or source.get('holdout') is not None
            or source.get('trading_enabled') is not False
            or source.get('status') != 'EXPLORATORY_VALIDATION_ONLY_NOT_APPROVED'):
        raise ValueError('Unexpected or unsafe V16 source report')
    rows = source['baseline_trades']
    if not isinstance(rows, list) or len(rows) != source['baseline_trade_count']:
        raise ValueError('V16 baseline trade count mismatch')
    if len(rows) != 30:
        raise ValueError('Expected the 30-trade V16 report; refusing mismatched input')
    required = ('probability', 'atr_pct', 'rsi_14', 'net_usdt', 'gross_usdt',
                'slippage_usdt', 'fees_usdt', 'signal_utc', 'pair', 'exit_reason')
    for row in rows:
        if any(k not in row for k in required):
            raise ValueError('Missing V16 trade field')
        if any(not math.isfinite(float(row[k])) for k in
               ('probability', 'atr_pct', 'rsi_14', 'net_usdt', 'gross_usdt',
                'slippage_usdt', 'fees_usdt')):
            raise ValueError('Nonfinite trade field')
    baseline = source['scenarios']['A_baseline']['fixed_v15_entries']
    if not math.isclose(sum(float(r['net_usdt']) for r in rows),
                        float(baseline['net_usdt']), abs_tol=0.0001):
        raise ValueError('V16 baseline net does not reconcile')
    if sum(float(r['net_usdt']) > 0 for r in rows) != 11:
        raise ValueError('Unexpected V16 baseline win count')
    by_pair = {p: group_summary([r for r in rows if r['pair'] == p])
               for p in ('BTC-USD', 'ETH-USD', 'SOL-USD')}
    by_exit = {reason: group_summary([r for r in rows if r['exit_reason'] == reason])
               for reason in ('SL', 'TP', 'TIME', 'SL_GAP', 'TP_GAP')}
    by_day = {}
    for r in rows:
        day = datetime.fromisoformat(r['signal_utc']).astimezone(timezone.utc).date().isoformat()
        by_day.setdefault(day, []).append(r)
    result = {
        'schema': 'crypto-radar-ai-research-v17',
        'status': 'DESCRIPTIVE_VALIDATION_ONLY_NOT_APPROVED',
        'research_only': True,
        'trading_enabled': False,
        'holdout_evaluated': False,
        'holdout': None,
        'source': 'V16 baseline_trades; no new data downloads, model fitting or scoring',
        'source_trade_count': len(rows),
        'overall': group_summary(rows),
        'by_pair': by_pair,
        'by_exit_reason': by_exit,
        'by_probability': bins(rows, 'probability', [0, .55, .7, .85, 1.000001],
                               ['[0,.55)', '[.55,.70)', '[.70,.85)', '[.85,1]']),
        'by_atr_pct': bins(rows, 'atr_pct', [0, .3, .4, .5, float('inf')],
                           ['[0,.3)', '[.3,.4)', '[.4,.5)', '[.5,+inf)']),
        'by_rsi_14': bins(rows, 'rsi_14', [0, 30, 50, 70, 100.000001],
                          ['[0,30)', '[30,50)', '[50,70)', '[70,100]']),
        'by_signal_date_utc': {k: group_summary(v) for k, v in sorted(by_day.items())},
        'pearson_correlation_with_net_usdt': {
            field: pearson(rows, field) for field in ('probability', 'atr_pct', 'rsi_14')
        },
        'limitations': [
            'Only 30 selected trades: groups can be tiny or empty; no statistical confidence.',
            'Bins are descriptive and not a strategy, rule, selection, recommendation or test.',
            'Validation has been repeatedly examined in V11-V16; comparisons are exploratory.',
            'No new data, no holdout read, no model training, no optimization, no trading.',
            'V16 costs and exit assumptions are inherited unchanged.',
            'Correlations are descriptive and do not establish causality or predictive value.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('V17_OK: descriptive only; 30 trades; no holdout; no trading')
    print('Net USDT:', result['overall']['net_usdt'])
    print('Report:', args.output)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import sys
        print(f'V17_FAILED_SAFELY: {exc!r}', file=sys.stderr)
        raise SystemExit(2)
