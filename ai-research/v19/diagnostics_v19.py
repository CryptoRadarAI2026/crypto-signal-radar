#!/usr/bin/env python3
"""V19 deterministic, synthetic-only V11 backtest integrity checks.

No exchange downloads, no real candle files, no holdout data, no model fitting,
no live trading. A passing suite is NOT a certification of profitability.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
V11_FILE = REPO / 'ai-research' / 'v11' / 'research_v11.py'
OUT = REPO / 'reports' / 'v19' / 'backtest_integrity.json'


def load_v11():
    spec = importlib.util.spec_from_file_location('research_v11_for_v19', V11_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load V11')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic(n=3000, gap_at=None):
    """Synthetic bars only; optional one-minute missing timestamp."""
    ts = np.arange(n, dtype=np.int64) * 60 + 1704067200
    if gap_at is not None:
        ts[gap_at:] += 60
    t = np.arange(n, dtype=float)
    c = 100.0 + 0.002 * t + 0.18 * np.sin(t / 13)
    o = c + 0.01 * np.sin(t / 3)
    h = np.maximum(o, c) + 0.12
    l = np.minimum(o, c) - 0.12
    v = 10.0 + np.mod(t, 7)
    return ts, o, h, l, c, v


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_features_past_only(v11):
    ts, o, h, l, c, v = synthetic()
    i = 500
    x0, atr0 = v11.features_at(h, l, c, v, i)
    for arr in (o, h, l, c, v):
        arr[i+1:] *= 2.7
    x1, atr1 = v11.features_at(h, l, c, v, i)
    check(np.allclose(x0, x1, rtol=0, atol=1e-12), 'Future candle edits changed features')
    check(math.isclose(atr0, atr1, abs_tol=1e-12), 'Future candle edits changed ATR')
    check(len(x0) == 16 and all(map(math.isfinite, x0)), 'Unexpected feature vector')
    return {'signal_index': i, 'feature_count': len(x0)}


def test_next_open_and_stop_first(v11):
    ts, o, h, l, c, v = synthetic()
    i = 500
    o[i+1] = 100.0
    h[i+1] = 104.0
    l[i+1] = 96.0
    c[i+1] = 100.0
    ret, reason, entry, exit_fill, exit_idx = v11.simulate_long(o, h, l, c, i, 1.0)
    check(reason == 'SL' and exit_idx == i+1, 'Same-bar double touch did not stop first')
    check(math.isclose(entry, 100 * (1 + v11.SLIPPAGE_PER_SIDE), rel_tol=1e-12), 'Not next-open entry')
    stop_pct = min(.03, max(.003, 1.6 / entry))
    expected_exit = entry * (1 - stop_pct) * (1 - v11.SLIPPAGE_PER_SIDE)
    expected_net = expected_exit * (1 - v11.FEE_PER_SIDE) / (entry * (1 + v11.FEE_PER_SIDE)) - 1
    check(math.isclose(exit_fill, expected_exit, rel_tol=1e-12), 'Wrong exit fill')
    check(math.isclose(ret, expected_net, rel_tol=1e-12), 'Wrong net fee/slippage accounting')
    return {'exit_reason': reason, 'entry_index': i+1, 'exit_index': exit_idx}


def test_gaps_excluded(v11):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'synthetic.csv'
        import csv
        def write(gap):
            ts, o, h, l, c, v = synthetic(3600, gap_at=gap)
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                writer.writerows(zip(ts, o, h, l, c, v))
            return v11.samples_from_csv(path, 'SOL-USD'), ts
        normal, _ = write(None)
        gap_at = 1200
        gapped, ts = write(gap_at)
        gap_before = int(ts[gap_at-1])
        gap_after = int(ts[gap_at])
        check(len(gapped) < len(normal), 'Missing candle did not remove any samples')
        check(all(not (s.ts - v11.LOOKBACK * 60 <= gap_before and
                       s.ts + v11.HORIZON * 60 >= gap_after)
                  for s in gapped), 'Sample crosses missing minute')
        return {'baseline_samples': len(normal), 'gapped_samples': len(gapped),
                'missing_minutes': 1}


def test_embargo_and_sealed_scoring(v11):
    Sample = v11.Sample
    samples = []
    for k in range(1000):
        ts = 1704067200 + k * 900
        samples.append(Sample(ts, 'BTC-USD', (0.0,)*16, k % 2,
                              .01 if k % 2 else -.01, 'TIME', 100., 100.,
                              ts+60, ts+900))
    tr, va, te, (t1, t2) = v11.chronological_split(samples)
    check(all(s.ts < t1-v11.EMBARGO_SEC for s in tr), 'Train embargo failed')
    check(all(t1+v11.EMBARGO_SEC < s.ts < t2-v11.EMBARGO_SEC for s in va), 'Validation embargo failed')
    check(all(s.ts > t2+v11.EMBARGO_SEC for s in te), 'Holdout split embargo failed')
    # Inspect source structure, not protected market data.
    tree = ast.parse(V11_FILE.read_text(encoding='utf-8'))
    research = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'research')
    source = ast.get_source_segment(V11_FILE.read_text(encoding='utf-8'), research) or ''
    check('xte' not in source and 'yte' not in source and 'predict_proba(xva)' in source,
          'Manual review required: scoring structure changed')
    return {'train_count_synthetic': len(tr), 'validation_count_synthetic': len(va),
            'reserved_count_synthetic': len(te), 'note': 'Synthetic split only; no market holdout read'}


def test_known_limitations(v11):
    # These are review flags, not runtime failures: report limitations honestly.
    tree = ast.parse(V11_FILE.read_text(encoding='utf-8'))
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    src = V11_FILE.read_text(encoding='utf-8')
    sample_src = ast.get_source_segment(src, funcs['samples_from_csv']) or ''
    main_src = ast.get_source_segment(src, funcs['main']) or ''
    return {
        'full_csv_parsed_before_split': 'samples_from_csv(path, pair)' in main_src,
        'sample_grid_row_stride_not_absolute_utc': 'range(LOOKBACK, len(ts)-HORIZON, HORIZON)' in sample_src,
        'holdout_labels_built_before_split': 'simulate_long(' in sample_src,
        'type_hint_mismatch_simulate_long': 'tuple[float, str, float, float]' in src,
        'interpretation': 'Known architectural review flags; do not infer proven prediction leakage.'
    }


def main():
    results = {}
    failures = []
    try:
        v11 = load_v11()
    except Exception as exc:
        v11 = None
        failures.append({'test': 'load_v11', 'error': repr(exc), 'trace': traceback.format_exc(limit=3)})
    tests = [test_features_past_only, test_next_open_and_stop_first,
             test_gaps_excluded, test_embargo_and_sealed_scoring, test_known_limitations]
    if v11 is not None:
        for fn in tests:
            try:
                results[fn.__name__] = {'status': 'PASS', 'details': fn(v11)}
            except Exception as exc:
                results[fn.__name__] = {'status': 'FAIL', 'error': repr(exc)}
                failures.append({'test': fn.__name__, 'error': repr(exc),
                                 'trace': traceback.format_exc(limit=4)})
    report = {
        'schema': 'crypto-radar-ai-research-v19',
        'status': 'SYNTHETIC_TESTS_PASSED_NOT_APPROVED' if not failures else 'INTEGRITY_TESTS_FAILED',
        'research_only': True, 'trading_enabled': False,
        'market_data_accessed': False, 'holdout_evaluated': False, 'holdout': None,
        'tests': results, 'failures': failures,
        'limitations': [
            'Synthetic tests are not a formal proof of no lookahead in every code path.',
            'Existing V11 parses and labels full CSV before splitting; this audit does not change V11.',
            'V11 uses row-based sample stride; missing bars can shift sampling alignment.',
            'No new training, parameter tuning, exchange downloads or real orders.',
            'Historical validation is reused and not independent out-of-sample.'
        ],
        'generated_utc': datetime.now(timezone.utc).isoformat()
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'tests': results, 'failures': failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
