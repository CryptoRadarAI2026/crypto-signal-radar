"""V24.8 hourly observational paper trading. NEVER places real orders."""
import json
import math
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research-output' / 'v248'
STATE = OUT / 'state.json'
REPORT = OUT / 'report.json'
FEE = 0.001
SLIPPAGE = 0.0005
HOLD_SECONDS = 5 * 3600
STARTING_CASH = 10000.0


def get_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'CryptoRadarAI2026-paper-research/2.0', 'Accept': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def candles():
    """Coinbase hourly OHLCV, ascending; require consecutive *completed* candles."""
    hour = int(time.time()) // 3600 * 3600
    start = hour - 60 * 24 * 3600
    rows = {}
    for a in range(start, hour, 299 * 3600):
        b = min(a + 299 * 3600, hour)
        q = urllib.parse.urlencode({'start': datetime.fromtimestamp(a, timezone.utc).isoformat(), 'end': datetime.fromtimestamp(b, timezone.utc).isoformat(), 'granularity': 3600})
        result = get_json('https://api.exchange.coinbase.com/products/BTC-USD/candles?' + q)
        if not isinstance(result, list):
            raise RuntimeError('Invalid candle response')
        for r in result:
            if isinstance(r, list) and len(r) >= 6 and a <= int(r[0]) < b and int(r[0]) % 3600 == 0:
                rows[int(r[0])] = [float(r[3]), float(r[2]), float(r[1]), float(r[4]), float(r[5])]
        time.sleep(0.15)
    stamps = sorted(rows)
    if len(stamps) < 650 or stamps[-1] != hour - 3600 or any(b-a != 3600 for a,b in zip(stamps, stamps[1:])):
        raise RuntimeError('Missing or stale completed candles: fail closed')
    data = np.array([rows[t] for t in stamps], dtype=float)
    if not np.isfinite(data).all() or (data[:, :4] <= 0).any() or (data[:, 4] < 0).any():
        raise RuntimeError('Invalid OHLCV data')
    return stamps, data


def model_probability(data):
    close, volume = data[:, 3], data[:, 4]
    feats = []
    for i in range(25, len(data)):
        rets = np.diff(np.log(close[i-23:i+1]))
        feats.append([math.log(close[i]/close[i-1]), math.log(close[i]/close[i-6]), math.log(close[i]/close[i-24]), float(np.std(rets)), float(volume[i]/max(1e-9, np.mean(volume[i-23:i+1]))), float((data[i,1]-data[i,2])/close[i])])
    x = np.asarray(feats)
    # Purge last 5 labels from training; latest feature is prediction-only.
    horizon = 5
    ntrain = len(x) - horizon
    if ntrain < 500:
        return None, 'INSUFFICIENT_HISTORY'
    y = np.array([int(close[i+horizon] > close[i]) for i in range(25, 25+ntrain)])
    if len(set(y)) < 2:
        return None, 'ONE_CLASS'
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
    model.fit(x[:ntrain], y)
    return float(model.predict_proba(x[-1:])[0,1]), 'OK_UNCALIBRATED'


def sentiment():
    try:
        row = get_json('https://api.alternative.me/fng/?limit=1&format=json')['data'][0]
        value, published = int(row['value']), int(row['timestamp'])
        if not 0 <= value <= 100 or not 0 <= time.time() - published <= 36*3600:
            return None, 'STALE_OR_INVALID'
        return value, 'OK_THIRD_PARTY'
    except (OSError, ValueError, KeyError, IndexError, TypeError, TimeoutError):
        return None, 'UNAVAILABLE'


def quote():
    """Observed spot bid/ask; fills are simulated at observation time, not past candle close."""
    x = get_json('https://api.exchange.coinbase.com/products/BTC-USD/ticker')
    bid, ask = float(x['bid']), float(x['ask'])
    if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
        raise RuntimeError('Invalid quote')
    return bid, ask


def initial():
    return {'version': 'V24.8', 'status': 'PAPER_TRADING_ONLY', 'cash': STARTING_CASH, 'btc': 0.0, 'position': None, 'last_bar': None, 'trades': [], 'events': []}


def read_state():
    if not STATE.exists():
        if os.environ.get('V248_RESTORE_REQUIRED') == '1':
            raise RuntimeError('State restoration required but missing; refusing reset')
        return initial()
    s = json.loads(STATE.read_text(encoding='utf-8'))
    if s.get('status') != 'PAPER_TRADING_ONLY' or s.get('version') != 'V24.8':
        raise RuntimeError('Unexpected state version/status')
    return s


def step(s, bar, now, bid, ask, p, model_status, mood, sentiment_status):
    """One decision per completed candle; existing position evaluated at current observed quote."""
    if s['last_bar'] is not None and bar <= s['last_bar']:
        return 'NO_NEW_BAR'
    action = 'WAIT'
    pos = s['position']
    if pos is not None:
        exit_reason = ('STOP_OBSERVED' if bid <= pos['stop'] else
                       'TARGET_OBSERVED' if bid >= pos['target'] else
                       'TIME_OBSERVED' if now >= pos['expires_at'] else None)
        if exit_reason:
            sell = bid * (1 - SLIPPAGE)
            proceeds = s['btc'] * sell
            fee = proceeds * FEE
            pnl = proceeds - fee - pos['total_entry_cost']
            s['cash'] += proceeds - fee
            s['trades'].append({'entry_observed_at': pos['opened_at'], 'exit_observed_at': now, 'entry_bar': pos['signal_bar'], 'exit_bar': bar, 'exit_reason': exit_reason, 'pnl_usd': round(pnl, 6), 'entry_fee_usd': pos['entry_fee'], 'exit_fee_usd': round(fee, 6), 'entry_price': pos['entry_price'], 'exit_price': sell})
            s['btc'] = 0.0
            s['position'] = None
            action = 'EXIT_' + exit_reason
    # Do not re-enter on same observation as an exit.
    if action == 'WAIT' and s['position'] is None and model_status == 'OK_UNCALIBRATED' and mood is not None and 25 <= mood <= 75 and p is not None and p >= 0.62:
        spend = s['cash'] * 0.10
        entry = ask * (1 + SLIPPAGE)
        fee = spend * FEE
        qty = (spend - fee) / entry
        if qty > 0:
            s['cash'] -= spend
            s['btc'] = qty
            s['position'] = {'opened_at': now, 'signal_bar': bar, 'expires_at': now + HOLD_SECONDS, 'entry_price': entry, 'entry_fee': fee, 'total_entry_cost': spend, 'stop': entry * 0.985, 'target': entry * 1.03, 'probability_uncalibrated': p, 'sentiment': mood}
            action = 'ENTER_PAPER'
    s['last_bar'] = bar
    s['events'].append({'observed_at': now, 'completed_bar': bar, 'action': action, 'probability_uncalibrated': p, 'model_status': model_status, 'sentiment': mood, 'sentiment_status': sentiment_status})
    s['events'] = s['events'][-500:]
    return action


def main():
    stamps, data = candles()
    s = read_state()
    if s['last_bar'] is not None and stamps[-1] <= s['last_bar']:
        print('No new completed candle; no state changes')
        return
    p, status = model_probability(data)
    mood, mood_status = sentiment()
    bid, ask = quote()
    now = int(time.time())
    action = step(s, stamps[-1], now, bid, ask, p, status, mood, mood_status)
    OUT.mkdir(parents=True, exist_ok=True)
    # Atomic replace prevents partially written state in a runner.
    temp = STATE.with_suffix('.json.tmp')
    temp.write_text(json.dumps(s, indent=2), encoding='utf-8')
    temp.replace(STATE)
    report = {'version': 'V24.8', 'status': 'PAPER_TRADING_ONLY', 'orders_sent': 0, 'observed_at_utc': datetime.fromtimestamp(now, timezone.utc).isoformat(), 'last_completed_bar_utc': datetime.fromtimestamp(stamps[-1], timezone.utc).isoformat(), 'action': action, 'model_probability_uncalibrated': p, 'model_status': status, 'fear_greed': mood, 'sentiment_status': mood_status, 'bid_observed': bid, 'ask_observed': ask, 'cash_usd': round(s['cash'], 2), 'btc_virtual': s['btc'], 'equity_usd_estimated_at_bid': round(s['cash'] + s['btc']*bid, 2), 'closed_trades': len(s['trades']), 'realized_pnl_usd': round(sum(t['pnl_usd'] for t in s['trades']), 2), 'position': s['position'], 'limitations': ['Paper fills only; no actual orders', 'Hourly observation can miss stop/target touches and gaps', 'Sentiment is a third-party proxy', 'Model uncalibrated and not proven profitable', 'Simulated fee/slippage']}
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
