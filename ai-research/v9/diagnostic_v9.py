#!/usr/bin/env python3
"""V9 synthetic timing diagnostics only. No market access or real orders."""
from datetime import datetime, timedelta, timezone

TRADING_ENABLED = False
HOLDOUT_EVALUATED = False
MAX_SIMULTANEOUS = 2
PAIR_COOLDOWN_MINUTES = 30


def select_trades(candidates):
    """Select hypothetical trades using their known synthetic actual exit times.

    Diagnostic oracle only: future exits must NOT be used for model selection
    in a live or forward-looking backtest. For historical replay, exits become
    available only when the simulated clock reaches them.
    """
    accepted = []
    last_exit_by_pair = {}
    for trade in sorted(candidates, key=lambda t: (t['entry'], t['pair'])):
        pair, entry, exit_time = trade['pair'], trade['entry'], trade['exit']
        if not entry < exit_time:
            raise ValueError('Exit must follow entry')
        if any(t['pair'] == pair and t['exit'] > entry for t in accepted):
            continue
        active = sum(t['entry'] <= entry < t['exit'] for t in accepted)
        if active >= MAX_SIMULTANEOUS:
            continue
        last_exit = last_exit_by_pair.get(pair)
        if last_exit is not None and entry < last_exit + timedelta(minutes=PAIR_COOLDOWN_MINUTES):
            continue
        accepted.append(trade)
        last_exit_by_pair[pair] = exit_time
    return accepted


def test_early_exit_releases_capacity():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trades = [
        {'pair': 'BTC', 'entry': t, 'exit': t + timedelta(minutes=3)},
        {'pair': 'ETH', 'entry': t + timedelta(minutes=1), 'exit': t + timedelta(minutes=20)},
        {'pair': 'SOL', 'entry': t + timedelta(minutes=2), 'exit': t + timedelta(minutes=10)},
        {'pair': 'SOL', 'entry': t + timedelta(minutes=4), 'exit': t + timedelta(minutes=9)},
    ]
    chosen = select_trades(trades)
    assert [x['pair'] for x in chosen] == ['BTC', 'ETH', 'SOL'], chosen


def test_cooldown_begins_at_actual_exit():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trades = [
        {'pair': 'BTC', 'entry': t, 'exit': t + timedelta(minutes=5)},
        {'pair': 'BTC', 'entry': t + timedelta(minutes=30), 'exit': t + timedelta(minutes=31)},
        {'pair': 'BTC', 'entry': t + timedelta(minutes=34), 'exit': t + timedelta(minutes=36)},
        {'pair': 'BTC', 'entry': t + timedelta(minutes=35), 'exit': t + timedelta(minutes=40)},
    ]
    chosen = select_trades(trades)
    assert [x['entry'] for x in chosen] == [t, t + timedelta(minutes=35)], chosen


def test_boundary_and_invalid_times():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert len(select_trades([
        {'pair': 'BTC', 'entry': t, 'exit': t + timedelta(minutes=5)},
        {'pair': 'BTC', 'entry': t + timedelta(minutes=35), 'exit': t + timedelta(minutes=40)},
    ])) == 2
    try:
        select_trades([{'pair': 'BTC', 'entry': t, 'exit': t}])
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid exit was accepted')


if __name__ == '__main__':
    test_early_exit_releases_capacity()
    test_cooldown_begins_at_actual_exit()
    test_boundary_and_invalid_times()
    print('V9 SYNTHETIC SELF-TEST PASSED')
    print('Actual-exit capacity: PASSED')
    print('Cooldown from actual exit: PASSED')
    print('Boundary and invalid exit: PASSED')
    print(f'Trading enabled: {TRADING_ENABLED}')
    print(f'Holdout evaluated: {HOLDOUT_EVALUATED}')
