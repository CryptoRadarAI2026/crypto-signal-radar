#!/usr/bin/env python3
"""V5 validation-only diagnostic; never places orders or exports models."""
import argparse
import csv
import importlib.util
import json
import math
import time
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / 'ai-research/v2/research_v2.py'

def load_v2():
    import sys
    spec = importlib.util.spec_from_file_location('radar_v2', V2)
    if not V2.is_file():
        raise FileNotFoundError(f'Missing V2 dependency: {V2}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def diagnose(chosen, v2):
    m = v2.metrics(chosen)
    by_reason = defaultdict(lambda: {'count': 0, 'net_pnl_usdt': 0.0})
    by_pair = defaultdict(lambda: {'count': 0, 'net_pnl_usdt': 0.0})
    gross = fees = slip = 0.0
    for s, _ in chosen:
        # Reconstruct an approximate pre-cost return from fee-adjusted fills.
        # Entry/exit fill prices already include slippage. Original raw opens/exit
        # prices are not stored in Sample, so exact slippage attribution is impossible.
        net = s.net_return
        fill_only = (1 + net) * (1 + v2.FEE_PER_SIDE) / (1 - v2.FEE_PER_SIDE) - 1
        fee_cost = (fill_only - net) * v2.ORDER_USDT
        gross += fill_only * v2.ORDER_USDT
        fees += fee_cost
        for bucket, key in ((by_reason, s.exit_reason), (by_pair, s.pair)):
            bucket[key]['count'] += 1
            bucket[key]['net_pnl_usdt'] += net * v2.ORDER_USDT
    def clean(d):
        return {k: {'count': v['count'], 'net_pnl_usdt': round(v['net_pnl_usdt'], 4)} for k,v in sorted(d.items())}
    return {'metrics': {k:v for k,v in m.items() if k != 'equity_curve'},
            'exit_reasons': clean(by_reason), 'markets': clean(by_pair),
            'after_slippage_before_fees_pnl_usdt': round(gross, 4),
            'estimated_fee_drag_usdt': round(fees, 4),
            'net_pnl_usdt': round(sum(s.net_return*v2.ORDER_USDT for s,_ in chosen), 4),
            'slippage_drag_usdt': None,
            'slippage_note': 'Cannot isolate exact slippage from V2 Sample; raw entry/exit reference prices are not stored.'}

def ensure_historical_csv(data_dir, days, pair, path):
    """Download only missing public historical data; never overwrite existing CSV."""
    if path.is_file():
        return {'pair': pair, 'source': 'existing_file', 'file': str(path),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    import public_data_v1
    end = int(time.time()) // 86400 * 86400
    start = end - days * 86400
    # Downloader refuses to overwrite. Its partial files must not be mistaken for complete data.
    try:
        return public_data_v1.download(pair, start, end, path)
    except Exception:
        if path.is_file():
            path.unlink()
        raise


def run(data_dir, days):
    v2 = load_v2()
    all_samples = []
    quality = {}
    manifest = []
    for pair in v2.PAIRS:
        path = data_dir / f"COINBASE_{pair.replace('-', '')}_{days}D_1M.csv"
        manifest.append(ensure_historical_csv(data_dir, days, pair, path))
        with path.open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        times = [int(r['timestamp']) for r in rows]
        quality[pair] = {'raw_rows': len(rows), 'duplicate_timestamps': len(times)-len(set(times)),
                         'missing_one_minute_intervals': sum(max(0, (b-a)//60-1) for a,b in zip(sorted(set(times)), sorted(set(times))[1:]) if b>a),
                         'out_of_order_rows': sum(b<=a for a,b in zip(times,times[1:]))}
        all_samples.extend(v2.samples_from_csv(path, pair))
    tr, va, _holdout_untouched, boundaries = v2.chronological_split(all_samples)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    models = [
        ('LogisticRegression', make_pipeline(RobustScaler(), LogisticRegression(C=.08, max_iter=700))),
        ('RandomForest', RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=30, n_jobs=2, random_state=42)),
        ('HistGradientBoosting', HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=12, learning_rate=.04, l2_regularization=3, random_state=42)),
    ]
    xtr = np.array([s.x for s in tr]); ytr = np.array([s.label for s in tr]); xva = np.array([s.x for s in va])
    candidates = []
    for name, model in models:
        model.fit(xtr,ytr)
        probs = model.predict_proba(xva)[:,1]
        for threshold in (.15,.20,.25,.30,.35,.40):
            chosen = v2.select_trades(va, probs, threshold)
            d = diagnose(chosen,v2)
            m = d['metrics']
            d.update({'model':name,'threshold':threshold,'passed_validation':bool(m['signals']>=40 and m['paper_pnl_usdt']>0 and (m['profit_factor'] or 0)>=1.10)})
            candidates.append(d)
    return {'schema':'crypto-radar-v5-diagnostic', 'status':'DIAGNOSTIC_ONLY_NOT_APPROVED',
            'research_only':True,'trading_enabled':False,'model_exported':False,
            'holdout_evaluated':False,'holdout_used_for_selection':False,
            'validation_scope':'V2 chronological validation only; not V4 three-fold walk-forward reproduction',
            'source_manifest':manifest, 'rows':{'train':len(tr),'validation':len(va),'holdout_reserved_not_evaluated':len(_holdout_untouched)},
            'split_boundaries_utc':list(boundaries), 'quality':quality, 'candidates':candidates,
            'warnings':['This is a diagnostic baseline, not a reproduction of V4 fold metrics.',
                        'V2 trade capacity assumes a fixed maximum holding time even for early exits.',
                        'Exact slippage attribution and mark-to-market drawdown require richer trade records.',
                        'Validation candidate selection does not imply trading approval.']}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data/raw'))
    p.add_argument('--days',type=int,choices=(90,120),default=90)
    p.add_argument('--out-dir',type=Path,default=Path('reports/v5'))
    args=p.parse_args()
    result=run(args.data_dir,args.days)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    import html
    lines=[]
    for c in result['candidates']:
        m=c['metrics']
        lines.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (c['model'],c['threshold'],m['signals'],m['paper_pnl_usdt'],m['profit_factor'],c['estimated_fee_drag_usdt'],c['passed_validation']))+'</tr>')
    page='''<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>گزارش تشخیصی V5</title><style>body{font-family:Tahoma,sans-serif;background:#101827;color:#f1f5f9;padding:18px}main{max-width:1050px;margin:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #475569;padding:10px;text-align:right}.scroll{overflow:auto}.warning{color:#fbbf24}</style><main><h1>Crypto Signal Radar AI — گزارش تشخیصی V5</h1><p class="warning">صرفاً تحقیقاتی؛ هیچ مدل یا معامله واقعی تأیید نشده است. Holdout ارزیابی نشده.</p><p>مقایسه اعتبارسنجی V2، نه بازتولید Walk-Forward نسخه V4</p><div class="scroll"><table><tr><th>مدل</th><th>آستانه</th><th>سیگنال</th><th>سود/زیان خالص USDT</th><th>Profit Factor</th><th>اثر کارمزد USDT</th><th>قبولی معیار پایه</th></tr>'''+''.join(lines)+'''</table></div><p>تفکیک دقیق لغزش قیمت با داده‌های ذخیره‌شده V2 ممکن نیست؛ جزئیات بازار و دلیل خروج در diagnostics.json است.</p></main></html>'''
    (args.out_dir/'dashboard.html').write_text(page,encoding='utf-8')
    print(json.dumps({'status':result['status'],'candidates':len(result['candidates']),'holdout_evaluated':False}))
if __name__=='__main__':main()
