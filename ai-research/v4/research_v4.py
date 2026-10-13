#!/usr/bin/env python3
"""V4: multi-block validation, strict holdout gate, public data and paper only.

Depends on existing ai-research/v2/research_v2.py and v1 train_public_data.py.
Never sends orders or exports a live model.
"""
from __future__ import annotations
import argparse
import csv
import html
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'v2'))
import research_v2 as v2

THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
FOLDS = 3
EMBARGO_SEC = 2 * 3600
MIN_TOTAL_TRADES = 40
MIN_FOLD_TRADES = 10
MIN_PROFIT_FACTOR = 1.10


def verify_csv(path: Path):
    """Reject non-contiguous or malformed market data; v2 skips local gaps."""
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing OHLCV columns: {path}')
        last = None
        count = 0
        for row in reader:
            ts = int(row['timestamp'])
            if last is not None and ts <= last:
                raise ValueError(f'Unsorted or duplicate timestamps: {path}')
            last = ts
            count += 1
            vals = [float(row[k]) for k in ('open', 'high', 'low', 'close', 'volume')]
            if not all(math.isfinite(x) for x in vals):
                raise ValueError(f'Non-finite OHLCV: {path}')
        if count < 2000:
            raise ValueError(f'Insufficient candles: {path}')


def split_data(samples):
    ordered = sorted(samples, key=lambda s: (s.ts, s.pair))
    timestamps = sorted({s.ts for s in ordered})
    if len(timestamps) < 100:
        raise ValueError('Insufficient distinct timestamps')
    a = timestamps[int(len(timestamps) * .55)]
    b = timestamps[int(len(timestamps) * .80)]
    train = [s for s in ordered if s.ts < a - EMBARGO_SEC]
    validation = [s for s in ordered if a + EMBARGO_SEC < s.ts < b - EMBARGO_SEC]
    holdout = [s for s in ordered if s.ts > b + EMBARGO_SEC]
    if min(map(len, (train, validation, holdout))) < 300:
        raise ValueError('Insufficient purged samples')
    if len({s.label for s in train}) != 2:
        raise ValueError('One-class training labels')
    return train, validation, holdout, (a, b)


def validation_blocks(validation):
    """Time-based contiguous blocks, not random rows."""
    times = sorted({s.ts for s in validation})
    edges = [times[int(len(times) * i / FOLDS)] for i in range(1, FOLDS)]
    blocks = [[] for _ in range(FOLDS)]
    for s in validation:
        block = sum(s.ts >= edge for edge in edges)
        blocks[block].append(s)
    if any(len(b) < 100 for b in blocks):
        raise ValueError('Insufficient validation block')
    return blocks


def models():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    return [
        ('LogisticRegression', make_pipeline(RobustScaler(), LogisticRegression(C=.08, max_iter=700))),
        ('RandomForest', RandomForestClassifier(n_estimators=100, max_depth=6,
            min_samples_leaf=30, n_jobs=2, random_state=42)),
        ('HistGradientBoosting', HistGradientBoostingClassifier(max_iter=80,
            max_leaf_nodes=12, learning_rate=.04, l2_regularization=3, random_state=42)),
    ]


def decision(fold_metrics, aggregate):
    reasons = []
    if aggregate['signals'] < MIN_TOTAL_TRADES:
        reasons.append('INSUFFICIENT_TOTAL_TRADES')
    if aggregate['paper_pnl_usdt'] <= 0:
        reasons.append('NON_POSITIVE_TOTAL_PNL')
    if (aggregate['profit_factor'] or 0) < MIN_PROFIT_FACTOR:
        reasons.append('LOW_TOTAL_PROFIT_FACTOR')
    for i, m in enumerate(fold_metrics, 1):
        if m['signals'] < MIN_FOLD_TRADES:
            reasons.append(f'FOLD_{i}_TOO_FEW_TRADES')
        if m['paper_pnl_usdt'] <= 0:
            reasons.append(f'FOLD_{i}_NON_POSITIVE_PNL')
        if (m['profit_factor'] or 0) < MIN_PROFIT_FACTOR:
            reasons.append(f'FOLD_{i}_LOW_PROFIT_FACTOR')
    return reasons


def evaluate(samples, manifest, candidate_models=None):
    train, validation, holdout, boundaries = split_data(samples)
    blocks = validation_blocks(validation)
    xtr, ytr = v2._x_y(train)
    candidates = []
    best = None
    for name, model in (candidate_models if candidate_models is not None else models()):
        model.fit(xtr, ytr)
        fold_probs = [np.asarray(model.predict_proba(v2._x_y(block)[0])[:, 1], dtype=float)
                      for block in blocks]
        for probs, block in zip(fold_probs, blocks):
            if len(probs) != len(block) or not np.all(np.isfinite(probs)) or np.any((probs < 0) | (probs > 1)):
                raise ValueError('Invalid model probabilities')
        for threshold in THRESHOLDS:
            # Apply the portfolio capacity and cooldown to the entire validation
            # interval before slicing trades into folds. This avoids resets at folds.
            full_probs = np.concatenate(fold_probs)
            full_samples = [s for block in blocks for s in block]
            selected = v2.select_trades(full_samples, full_probs, threshold)
            fold_starts = [block[0].ts for block in blocks[1:]]
            chosen_folds = [[] for _ in range(FOLDS)]
            for trade in selected:
                ix = sum(trade[0].ts >= edge for edge in fold_starts)
                chosen_folds[ix].append(trade)
            fm = [v2.metrics(group) for group in chosen_folds]
            agg = v2.metrics(selected)
            reasons = decision(fm, agg)
            candidates.append({'model': name, 'threshold': threshold,
                'validation': {k: agg[k] for k in ('signals', 'paper_pnl_usdt', 'profit_factor', 'win_rate_pct')},
                'folds': [{k: m[k] for k in ('signals', 'paper_pnl_usdt', 'profit_factor')} for m in fm],
                'rejection_reasons': reasons})
            if not reasons:
                score = agg['paper_pnl_usdt']
                if best is None or score > best[0]:
                    best = (score, name, model, threshold, agg)
    report = {'schema': 'crypto-radar-ai-research-v4', 'research_only': True,
        'trading_enabled': False, 'model_exported': False,
        'status': 'REJECTED_ON_VALIDATION', 'holdout_evaluated': False,
        'data_source': 'Coinbase Exchange public 1-minute OHLCV',
        'markets': list(v2.PAIRS), 'features': list(v2.FEATURES),
        'rows': {'train': len(train), 'validation': len(validation), 'holdout': len(holdout)},
        'split_timestamps_utc': {'train_validation_boundary': boundaries[0],
            'validation_holdout_boundary': boundaries[1], 'embargo_seconds': EMBARGO_SEC},
        'validation_blocks': FOLDS, 'thresholds_tested': list(THRESHOLDS),
        'source_manifest': manifest, 'validation_candidates': candidates,
        'assumptions': {'fee_per_side': v2.FEE_PER_SIDE,
            'slippage_per_side': v2.SLIPPAGE_PER_SIDE, 'order_usdt': v2.ORDER_USDT,
            'horizon_minutes': v2.HORIZON, 'sample_interval_minutes': v2.HORIZON,
            'max_simultaneous': v2.MAX_SIMULTANEOUS,
            'pair_cooldown_seconds': v2.PAIR_COOLDOWN_SEC},
        'limitations': [
            'Historical research only; no orders, keys, leverage, or live model export.',
            '15-minute sampling retained from V2; this is not a high-frequency execution engine.',
            'Three validation blocks share one fixed training period; they are not independent walk-forward retrainings.',
            'Multiple threshold/model searches can overfit validation even with three-block checks.',
            'OHLCV cannot resolve intraminute paths; stop-first fill and assumed costs are approximations.',
            'Passing validation and holdout would not establish future profitability.'
        ]}
    if best is not None:
        _, name, model, threshold, val = best
        pred = np.asarray(model.predict_proba(v2._x_y(holdout)[0])[:, 1], dtype=float)
        report.update(status='RESEARCH_ONLY_NOT_APPROVED', selected_model=name,
            selected_threshold=threshold, validation=val,
            holdout=v2.metrics(v2.select_trades(holdout, pred, threshold)),
            holdout_evaluated=True)
    return report


def dashboard(report, path):
    rows = []
    for c in report['validation_candidates']:
        v = c['validation']
        reasons = ', '.join(c['rejection_reasons']) or 'PASS_VALIDATION_ONLY'
        rows.append('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in
            (c['model'], c['threshold'], v['signals'], v['paper_pnl_usdt'],
             v['profit_factor'], reasons)) + '</tr>')
    status = html.escape(report['status'])
    note = ('داده آزمون نهایی دست‌نخورده مانده است.' if not report['holdout_evaluated']
            else 'آزمون نهایی اجرا شد؛ معامله واقعی همچنان غیرفعال است.')
    content = f'''<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crypto Radar AI V4 — Paper Research</title>
<style>body{{background:#0c1625;color:#edf4ff;font:15px Tahoma,Arial;line-height:1.9;margin:0}}
main{{max-width:1100px;margin:auto;padding:20px}}section{{background:#18273c;padding:18px;border-radius:14px;margin:18px 0}}
.wrap{{overflow:auto}}table{{border-collapse:collapse;min-width:760px;width:100%}}
td,th{{padding:9px;border-bottom:1px solid #40516a;text-align:right}}</style>
<main><h1>گزارش پژوهشی V4</h1><section><b>وضعیت: {status}</b><p>{note}</p>
<p>فقط شبیه‌سازی؛ بدون سفارش واقعی و بدون اتصال حساب صرافی.</p></section>
<section><h2>اعتبارسنجی سه‌بخشی</h2><div class="wrap"><table><tr>
<th>مدل</th><th>آستانه</th><th>معامله</th><th>سود/زیان فرضی</th><th>ضریب سود</th><th>دلایل رد</th>
</tr>{''.join(rows)}</table></div></section>
<section><h2>محدودیت‌ها</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in report['limitations'])}</ul></section>
</main></html>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Public historical research only')
    parser.add_argument('--days', type=int, choices=(90, 120), default=90)
    parser.add_argument('--data-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--out-dir', type=Path, default=Path('reports/v4'))
    args = parser.parse_args()
    from train_public_data import download
    end = int(time.time()) // 86400 * 86400
    start = end - args.days * 86400
    samples, manifest = [], []
    for pair in v2.PAIRS:
        path = args.data_dir / f"COINBASE_{pair.replace('-', '')}_{args.days}D_1M.csv"
        manifest.append(download(pair, start, end, path))
        verify_csv(path)
        batch = v2.samples_from_csv(path, pair)
        samples.extend(batch)
        print(f'{pair}: {len(batch)} historical samples', flush=True)
    report = evaluate(samples, manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    dashboard(report, args.out_dir / 'dashboard.html')
    print(json.dumps({'status': report['status'], 'holdout_evaluated': report['holdout_evaluated']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'RESEARCH_V4_FAILED_SAFELY: {exc!r}', file=sys.stderr)
        raise SystemExit(2)
