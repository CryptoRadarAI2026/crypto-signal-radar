"""V24.8 BTC-USD observational paper trading. No exchange orders or secrets."""
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
FEE, SLIPPAGE = 0.001, 0.0005
HOLD_SECONDS = 5 * 3600
STARTING_CASH = 10000.0


def get_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'CryptoRadarAI2026-paper-research/3.0', 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def candles():
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
        raise RuntimeError('Missing/stale completed candles; fail closed')
    data = np.asarray([rows[t] for t in stamps], dtype=float)
    if not np.isfinite(data).all() or (data[:, :4] <= 0).any() or (data[:, 4] < 0).any():
        raise RuntimeError('Invalid OHLCV')
    if any(not (low <= min(open_, close) <= max(open_, close) <= high) for open_, high, low, close, _ in data):
        raise RuntimeError('Inconsistent OHLC')
    return stamps, data


def model_probability(data):
    close, volume = data[:, 3], data[:, 4]
    feats = []
    for i in range(25, len(data)):
        rets = np.diff(np.log(close[i-23:i+1]))
        feats.append([math.log(close[i]/close[i-1]), math.log(close[i]/close[i-6]), math.log(close[i]/close[i-24]), float(np.std(rets)), float(volume[i]/max(1e-9, np.mean(volume[i-23:i+1]))), float((data[i,1]-data[i,2])/close[i])])
    x = np.asarray(feats)
    horizon = 5
    ntrain = len(x) - horizon
    if ntrain < 500:
        return None, 'INSUFFICIENT_HISTORY'
    y = np.asarray([int(close[i+horizon] > close[i]) for i in range(25, 25+ntrain)])
    if len(set(y)) < 2:
        return None, 'ONE_CLASS'
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=42))
    model.fit(x[:ntrain], y)
    return float(model.predict_proba(x[-1:])[0, 1]), 'OK_UNCALIBRATED'


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
    x = get_json('https://api.exchange.coinbase.com/products/BTC-USD/ticker')
    bid, ask = float(x['bid']), float(x['ask'])
    if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
        raise RuntimeError('Invalid bid/ask')
    # Reject quotes whose exchange timestamp is absent or too old.
    observed = datetime.fromisoformat(x['time'].replace('Z', '+00:00')).timestamp()
    if not 0 <= time.time() - observed <= 120:
        raise RuntimeError('Stale quote')
    return bid, ask


def initial():
    return {'version': 'V24.8', 'status': 'PAPER_TRADING_ONLY', 'cash': STARTING_CASH, 'btc': 0.0, 'position': None, 'last_bar': None, 'trades': [], 'events': [], 'sequence': 0}


def validate_state(s):
    if not isinstance(s, dict) or s.get('version') != 'V24.8' or s.get('status') != 'PAPER_TRADING_ONLY':
        raise RuntimeError('Unexpected state version/status')
    if not isinstance(s.get('trades'), list) or not isinstance(s.get('events'), list):
        raise RuntimeError('Malformed state history')
    if type(s.get('sequence')) is not int or s['sequence'] < 0:
        raise RuntimeError('Invalid sequence')
    for key in ('cash', 'btc'):
        if not isinstance(s.get(key), (float, int)) or not math.isfinite(s[key]) or s[key] < 0:
            raise RuntimeError('Invalid balance: ' + key)
    if (s['position'] is None) != (s['btc'] == 0):
        raise RuntimeError('Position/btc mismatch')
    if s['position'] is not None:
        pos = s['position']
        for key in ('stop', 'target', 'total_entry_cost', 'opened_at', 'expires_at'):
            if key not in pos or not isinstance(pos[key], (int, float)) or not math.isfinite(pos[key]) or pos[key] <= 0:
                raise RuntimeError('Malformed position')
    if s['last_bar'] is not None and (type(s['last_bar']) is not int or s['last_bar'] <= 0):
        raise RuntimeError('Invalid last_bar')


def read_state():
    if not STATE.exists():
        if os.environ.get('V248_BOOTSTRAP') != '1':
            raise RuntimeError('State missing: restoration required. Refusing reset.')
        s = initial()
    else:
        if os.environ.get('V248_BOOTSTRAP') == '1':
            raise RuntimeError('Bootstrap forbidden: existing state present')
        s = json.loads(STATE.read_text(encoding='utf-8'))
    validate_state(s)
    return s


def step(s, bar, now, bid, ask, p, model_status, mood, sentiment_status):
    """Exits checked on every observation; new entries only on a new completed bar."""
    new_bar = s['last_bar'] is None or bar > s['last_bar']
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
            s['btc'], s['position'] = 0.0, None
            action = 'EXIT_' + exit_reason
    if (new_bar and action == 'WAIT' and s['position'] is None and model_status == 'OK_UNCALIBRATED' and mood is not None and 25 <= mood <= 75 and p is not None and p >= 0.62):
        spend = s['cash'] * 0.10
        entry = ask * (1 + SLIPPAGE)
        fee = spend * FEE
        qty = (spend - fee) / entry
        if qty > 0:
            s['cash'] -= spend
            s['btc'] = qty
            s['position'] = {'opened_at': now, 'signal_bar': bar, 'expires_at': now + HOLD_SECONDS, 'entry_price': entry, 'entry_fee': fee, 'total_entry_cost': spend, 'stop': entry * 0.985, 'target': entry * 1.03, 'probability_uncalibrated': p, 'sentiment': mood}
            action = 'ENTER_PAPER'
    if new_bar:
        s['last_bar'] = bar
    s['sequence'] += 1
    s['events'].append({'observed_at': now, 'completed_bar': bar, 'action': action, 'probability_uncalibrated': p, 'model_status': model_status, 'sentiment': mood, 'sentiment_status': sentiment_status})
    s['events'] = s['events'][-500:]
    validate_state(s)
    return action


def main():
    # Read and validate persisted state BEFORE any decision.
    s = read_state()
    stamps, data = candles()
    bar = stamps[-1]
    if s['last_bar'] is not None and bar < s['last_bar']:
        raise RuntimeError('Historical rollback detected')
    p, status = model_probability(data)
    mood, mood_status = sentiment()
    bid, ask = quote()
    now = int(time.time())
    if now - (bar + 3600) > 3600:
        raise RuntimeError('Completed candle too old for entry')
    action = step(s, bar, now, bid, ask, p, status, mood, mood_status)
    OUT.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix('.json.tmp')
    temp.write_text(json.dumps(s, indent=2), encoding='utf-8')
    temp.replace(STATE)
    report = {'version': 'V24.8', 'status': 'PAPER_TRADING_ONLY', 'orders_sent': 0, 'observed_at_utc': datetime.fromtimestamp(now, timezone.utc).isoformat(), 'last_completed_bar_utc': datetime.fromtimestamp(bar, timezone.utc).isoformat(), 'sequence': s['sequence'], 'action': action, 'model_probability_uncalibrated': p, 'model_status': status, 'fear_greed': mood, 'sentiment_status': mood_status, 'bid_observed': bid, 'ask_observed': ask, 'cash_usd': round(s['cash'], 2), 'btc_virtual': s['btc'], 'equity_usd_estimated_at_bid': round(s['cash'] + s['btc']*bid, 2), 'closed_trades': len(s['trades']), 'realized_pnl_usd': round(sum(t['pnl_usd'] for t in s['trades']), 2), 'position': s['position'], 'limitations': ['No actual orders', 'Hourly/manual observation misses intrahour stop/target touches', 'Uncalibrated model and no demonstrated edge', 'Fear & Greed third-party proxy', 'Assumed fees/slippage', 'Observed quote is not a guaranteed executable fill']}
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
