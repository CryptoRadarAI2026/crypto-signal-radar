#!/usr/bin/env python3
"""V10 independent synthetic integration test of V2-style long fills and event replay.

No downloads, exchange access, credentials, order execution, model training or holdout.
This does NOT patch research_v2.py and does NOT demonstrate profitability.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

TRADING_ENABLED = False
HOLDOUT_EVALUATED = False
FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0005
MAX_SIMULTANEOUS = 2
PAIR_COOLDOWN_SECONDS = 30 * 60
HORIZON = 15


@dataclass(frozen=True)
class Trade:
    pair: str
    signal_ts: int
    entry_ts: int
    exit_ts: int
    net_return: float
    reason: str


def simulate_long_with_exit(ohlc, signal_index: int, atr: float, horizon: int = HORIZON):
    """V2 fill conventions with an added actual exit bar index.

    ohlc is a sequence of (open, high, low, close) one-minute bars.
    All input bars are SYNTHETIC for this diagnostic.
    """
    if horizon < 1 or signal_index < 0 or signal_index + horizon >= len(ohlc):
        raise ValueError('Invalid signal index or horizon')
    if atr <= 0:
        raise ValueError('ATR must be positive')
    i = signal_index
    entry = float(ohlc[i + 1][0]) * (1 + SLIPPAGE_PER_SIDE)
    stop_pct = min(0.03, max(0.003, 1.6 * atr / entry))
    sl = entry * (1 - stop_pct)
    tp = entry * (1 + 1.8 * stop_pct)
    price, reason, exit_index = float(ohlc[i + horizon][3]), 'TIME', i + horizon
    for j in range(i + 1, i + horizon + 1):
        op, hi, lo, _ = map(float, ohlc[j])
        if j > i + 1 and op <= sl:
            price, reason, exit_index = op, 'SL_GAP', j
            break
        if j > i + 1 and op >= tp:
            price, reason, exit_index = op, 'TP_GAP', j
            break
        if lo <= sl:
            price, reason, exit_index = sl, 'SL', j
            break
        if hi >= tp:
            price, reason, exit_index = tp, 'TP', j
            break
    exit_fill = price * (1 - SLIPPAGE_PER_SIDE)
    net = exit_fill * (1 - FEE_PER_SIDE) / (entry * (1 + FEE_PER_SIDE)) - 1
    return net, reason, entry, exit_fill, exit_index


def make_trade(pair: str, signal_ts: int, bars, atr: float, signal_index: int = 0):
    net, reason, _, _, exit_index = simulate_long_with_exit(bars, signal_index, atr)
    entry_ts = signal_ts + 60  # next candle open, consistent with V2
    exit_ts = signal_ts + exit_index * 60  # synthetic bar boundary approximation
    if exit_ts < entry_ts:
        raise ValueError('Exit cannot precede entry')
    return Trade(pair, signal_ts, entry_ts, exit_ts, net, reason)


def replay(trades):
    """Chronological paper replay: only completed exits update cooldown state.

    Exit events are queued at entry; their effect is applied only when replay
    reaches that time. Candidate ranking uses only signal time and pair.
    """
    active = []  # heap: (exit_ts, pair)
    last_completed_exit = {}
    chosen = []
    for t in sorted(trades, key=lambda x: (x.entry_ts, x.pair)):
        if t.exit_ts < t.entry_ts:
            raise ValueError('Exit precedes entry')
        while active and active[0][0] <= t.entry_ts:
            completed_exit, completed_pair = heappop(active)
            last_completed_exit[completed_pair] = completed_exit
        if len(active) >= MAX_SIMULTANEOUS:
            continue
        if any(pair == t.pair for _, pair in active):
            continue
        if (t.pair in last_completed_exit and
                t.entry_ts < last_completed_exit[t.pair] + PAIR_COOLDOWN_SECONDS):
            continue
        chosen.append(t)
        heappush(active, (t.exit_ts, t.pair))
    return chosen


def flat_bars():
    return [(100.0, 100.1, 99.9, 100.0) for _ in range(HORIZON + 1)]


def test_exit_time_and_costs():
    bars = flat_bars()
    # SL at first entry bar: conservative stop-first if TP also hit.
    bars[1] = (100.0, 105.0, 95.0, 100.0)
    net, reason, _, _, exit_idx = simulate_long_with_exit(bars, 0, 0.2)
    assert reason == 'SL' and exit_idx == 1 and net < 0
    # Flat time exit still loses fees/slippage.
    net, reason, _, _, exit_idx = simulate_long_with_exit(flat_bars(), 0, 0.2)
    assert reason == 'TIME' and exit_idx == HORIZON and net < 0
    # Take profit at bar 3, not at fixed horizon.
    bars = flat_bars()
    bars[3] = (100.0, 102.0, 99.9, 101.0)
    _, reason, _, _, exit_idx = simulate_long_with_exit(bars, 0, 0.2)
    assert reason == 'TP' and exit_idx == 3


def test_replay_actual_exits():
    # Synthetic trades produced by the same integrated fill simulator.
    base = 1_800_000_000
    def trade(pair, entry_minute, exit_after_bars):
        bars = flat_bars()
        if exit_after_bars < HORIZON:
            bars[exit_after_bars] = (100.0, 102.0, 99.9, 101.0)
        return make_trade(pair, base + (entry_minute - 1) * 60, bars, 0.2)

    # BTC closes at minute 3, releasing capacity for SOL at minute 4.
    btc = trade('BTC', 0, 4)
    eth = trade('ETH', 1, 15)
    rejected = trade('SOL', 2, 15)
    sol = trade('SOL', 4, 15)
    chosen = replay([btc, eth, rejected, sol])
    assert [(t.pair, t.entry_ts) for t in chosen] == [
        (btc.pair, btc.entry_ts), (eth.pair, eth.entry_ts), (sol.pair, sol.entry_ts)]

    # BTC's cooldown starts at its completed exit (minute 3), not its entry.
    too_early = trade('BTC', 32, 3)
    boundary = trade('BTC', 33, 3)
    chosen = replay([btc, too_early, boundary])
    assert chosen == [btc, boundary], chosen


def test_invalid_and_boundary():
    base = 1_800_000_000
    try:
        replay([Trade('BTC', base, base + 60, base, 0, 'BAD')])
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid exit accepted')
    # Position ending exactly when another begins releases capacity.
    a = Trade('BTC', base - 60, base, base + 60, 0, 'TIME')
    b = Trade('ETH', base - 60, base, base + 60, 0, 'TIME')
    c = Trade('SOL', base, base + 60, base + 120, 0, 'TIME')
    assert replay([a, b, c]) == [a, b, c]


if __name__ == '__main__':
    test_exit_time_and_costs()
    test_replay_actual_exits()
    test_invalid_and_boundary()
    print('V10 SYNTHETIC INTEGRATION SELF-TEST PASSED')
    print('V2-style stop-first, TP, time exit, fees: PASSED')
    print('Actual-exit capacity and post-exit cooldown: PASSED')
    print('Invalid and boundary timestamps: PASSED')
    print(f'Trading enabled: {TRADING_ENABLED}')
    print(f'Holdout evaluated: {HOLDOUT_EVALUATED}')
