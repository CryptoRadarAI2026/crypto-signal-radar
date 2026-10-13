#!/usr/bin/env python3
"""Research-only 1m OHLCV downloader + chronological ML evaluation.
Uses public Coinbase Exchange candles; no API keys, order execution or model promotion.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, math, os, random, sys, time
import urllib.error, urllib.parse, urllib.request
from pathlib import Path

SOURCE = 'https://api.exchange.coinbase.com'
PAIRS = ('BTC-USD', 'ETH-USD', 'SOL-USD')
FEATURES = ('r1', 'r5', 'r15', 'ema_gap', 'vol_ratio', 'range_pct', 'rsi14', 'atr14')
COST = 0.003  # Hypothetical round-trip fee+slippage; not execution fill estimates.
HORIZON = 15  # 15 x 1m candles.


def fetch_page(pair, start, end):
    params = urllib.parse.urlencode({'start': dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
                                     'end': dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat(),
                                     'granularity': 60})
    url = f'{SOURCE}/products/{pair}/candles?{params}'
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'CryptoRadarAI-Research/1.0', 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=35) as response:
                data = json.load(response)
            if not isinstance(data, list):
                raise RuntimeError(f'Unexpected API response for {pair}: {str(data)[:120]}')
            candles = {}
            for raw in data:
                if not isinstance(raw, list) or len(raw) < 6:
                    raise ValueError('Malformed OHLCV candle')
                t = int(raw[0]); low, high, opn, close, vol = map(float, raw[1:6])
                if not (start <= t < end):
                    continue
                if not (t % 60 == 0 and 0 < low <= min(opn, close) <= max(opn, close) <= high and vol >= 0 and all(map(math.isfinite, (opn, high, low, close, vol)))):
                    raise ValueError(f'Invalid candle for {pair} at {t}')
                value = (t, opn, high, low, close, vol)
                if t in candles and candles[t] != value:
                    raise ValueError('Conflicting duplicate candle')
                candles[t] = value
            return [candles[t] for t in sorted(candles)]
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403, 404):
                raise RuntimeError(f'Public data denied/unavailable: {pair} HTTP {e.code}') from e
            if attempt == 4:
                raise
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            if attempt == 4:
                raise RuntimeError(f'Public data unreachable: {pair}: {e}') from e
        time.sleep(min(45, 2 ** attempt + 1))
    raise RuntimeError('No data')


def download(pair, start, end, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    # GitHub Actions uses a fresh runner. Never overwrite any checked-in user dataset.
    if output.exists():
        raise RuntimeError(f'Refusing to overwrite existing file {output}')
    count = 0
    with output.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('timestamp', 'open', 'high', 'low', 'close', 'volume'))
        for page_start in range(start, end, 295 * 60):
            page_end = min(end, page_start + 295 * 60)
            rows = fetch_page(pair, page_start, page_end)
            writer.writerows(rows)
            count += len(rows)
            time.sleep(0.45)
    minimum = int((end - start) / 60 * 0.85)
    if count < minimum:
        raise RuntimeError(f'Dataset incomplete for {pair}: {count} bars, minimum {minimum}; refusing to train')
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    return {'pair': pair, 'bars': count, 'sha256': sha, 'source': SOURCE, 'file': str(output)}


def features_and_labels(path, pair):
    import numpy as np
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    rows = sorted(rows, key=lambda r: int(r['timestamp']))
    ts = np.array([int(r['timestamp']) for r in rows], dtype=np.int64)
    close = np.array([float(r['close']) for r in rows], dtype=np.float64)
    high = np.array([float(r['high']) for r in rows], dtype=np.float64)
    low = np.array([float(r['low']) for r in rows], dtype=np.float64)
    volume = np.array([float(r['volume']) for r in rows], dtype=np.float64)
    if len(ts) < 2000 or np.any(np.diff(ts) <= 0):
        raise ValueError(f'Missing or unordered source timestamps for {pair}')
    result = []
    # Each sample requires a strictly contiguous 65m interval. Never fill missing bars.
    # Sample every 15m to reduce overlapping future labels.
    for i in range(50, len(ts) - HORIZON, HORIZON):
        if ts[i + HORIZON] - ts[i - 49] != (HORIZON + 49) * 60:
            continue
        w = close[i - 49:i + 1]
        v = volume[i - 19:i + 1]
        changes = np.diff(w)
        up = np.maximum(changes[-14:], 0).mean()
        down = np.maximum(-changes[-14:], 0).mean()
        rsi = 100. if down == 0 else 100 - 100 / (1 + up / down)
        ema_short = np.average(w[-12:], weights=np.arange(1, 13))
        ema_long = np.average(w[-26:], weights=np.arange(1, 27))
        tr = np.maximum(high[i - 13:i + 1] - low[i - 13:i + 1], np.maximum(abs(high[i - 13:i + 1] - close[i - 14:i]), abs(low[i - 13:i + 1] - close[i - 14:i])))
        x = [(w[-1]/w[-2]-1), (w[-1]/w[-6]-1), (w[-1]/w[-16]-1),
             (ema_short/ema_long-1), (v[-1]/max(v.mean(), 1e-12)),
             ((high[i] - low[i])/w[-1]), rsi/100, float(tr.mean()/w[-1])]
        # Future close-only label, net of hypothetical costs. Not an execution simulator.
        ret = float(close[i + HORIZON]/close[i]-1-COST)
        if all(map(math.isfinite, x)):
            result.append((int(ts[i]), pair, x, int(ret > 0), ret))
    if len(result) < 100:
        raise ValueError(f'Insufficient contiguous samples for {pair}')
    return result


def evaluate(data, prob, threshold):
    import numpy as np
    returns = [row[4] for row, p in zip(data, prob) if p >= threshold]
    gross_gain = sum(max(r, 0) for r in returns)
    gross_loss = -sum(min(r, 0) for r in returns)
    pnl = 100. * sum(returns)
    eq = 10000.0
    peak = eq
    drawdown = 0.0
    for r in returns:
        eq += r * 100.0
        peak = max(peak, eq)
        drawdown = max(drawdown, (peak-eq)/peak)
    return {'signals': len(returns), 'paper_pnl_100usdt_per_signal': round(pnl, 2),
            'mean_net_return_pct': round(100*sum(returns)/len(returns), 5) if returns else None,
            'win_rate_pct': round(100*sum(r > 0 for r in returns)/len(returns), 2) if returns else None,
            'profit_factor': round(gross_gain/gross_loss, 3) if gross_loss else None,
            'closed_equity_drawdown_pct': round(100*drawdown, 3)}


def train(data, manifest, output):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss
    data.sort(key=lambda r: (r[0], r[1]))
    dates = sorted({r[0] for r in data})
    first = dates[int(len(dates)*0.6)]
    second = dates[int(len(dates)*0.8)]
    gap = HORIZON*60
    tr = [r for r in data if r[0] < first-gap]
    va = [r for r in data if first+gap < r[0] < second-gap]
    te = [r for r in data if r[0] > second+gap]
    if min(len(tr),len(va),len(te)) < 100 or len({r[3] for r in tr}) < 2:
        raise ValueError('Insufficient separated training, validation, holdout samples')
    xtr = np.array([r[2] for r in tr]); ytr = np.array([r[3] for r in tr])
    xva = np.array([r[2] for r in va]); yva = np.array([r[3] for r in va])
    xte = np.array([r[2] for r in te]); yte = np.array([r[3] for r in te])
    candidates = [make_pipeline(RobustScaler(), LogisticRegression(max_iter=600, C=0.1)),
                  RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=25, n_jobs=2, random_state=42)]
    best = None
    for model in candidates:
        model.fit(xtr, ytr)
        pv = model.predict_proba(xva)[:,1]
        for threshold in (0.5,0.55,0.60,0.65,0.70):
            perf = evaluate(va, pv, threshold)
            if perf['signals'] < 30:
                continue
            score = (perf['mean_net_return_pct'] or -100)
            if best is None or score > best[0]:
                best = (score, model, threshold, perf)
    if best is None:
        # A valid rejection, not a license to lower thresholds after looking at the test set.
        report = {'status':'REJECTED_NO_VALIDATION_CANDIDATE', 'source_manifest':manifest,
                  'rows':{'train':len(tr), 'validation':len(va), 'holdout':len(te)},
                  'research_only':True, 'trading_enabled':False}
    else:
        _,model,threshold,val_perf = best
        probabilities = model.predict_proba(xte)[:,1]
        perf = evaluate(te, probabilities, threshold)
        prevalence = float(ytr.mean())
        report = {'status':'RESEARCH_ONLY_NOT_APPROVED', 'source_manifest':manifest,
                  'data_timeframe':'1m', 'label_horizon_minutes':HORIZON,
                  'features':FEATURES, 'selected_model':type(model).__name__,
                  'threshold_selected_on_validation':threshold,
                  'rows':{'train':len(tr),'validation':len(va),'holdout':len(te)},
                  'chronological_split':{'train_before':first-gap,'validation_between':[first+gap,second-gap], 'holdout_after':second+gap},
                  'validation':val_perf,'holdout':perf,
                  'holdout_auc':float(roc_auc_score(yte,probabilities)) if len(set(yte))==2 else None,
                  'holdout_brier':float(brier_score_loss(yte,probabilities)),
                  'constant_baseline_brier':float(np.mean((yte-prevalence)**2)),
                  'research_only':True,'trading_enabled':False,
                  'limitations':'One-minute close-to-close approximation; no intrabar stops, liquidations, order-book, funding, news, outages, fill model, or live-forward validation.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'holdout':report.get('holdout'), 'rows':report['rows']},ensure_ascii=False))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--days',type=int,default=90)
    parser.add_argument('--out',default='reports/real_data_training.json')
    args=parser.parse_args()
    if not 90 <= args.days <= 365:
        raise SystemExit('days must be between 90 and 365')
    end = int(time.time())//86400 * 86400
    start = end - args.days * 86400
    raw = Path('data/raw'); raw.mkdir(parents=True,exist_ok=True)
    manifest=[]; data=[]
    for pair in PAIRS:
        dest = raw / f'COINBASE_{pair.replace("-", "")}_1.csv'
        meta = download(pair,start,end,dest)
        manifest.append(meta)
        samples = features_and_labels(dest,pair)
        print(pair, meta['bars'],len(samples),'samples')
        data.extend(samples)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path('reports/source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    train(data,manifest,Path(args.out))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('RESEARCH FAILED SAFELY:',repr(error),file=sys.stderr)
        sys.exit(2)
