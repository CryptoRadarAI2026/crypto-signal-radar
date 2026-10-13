from datetime import datetime, timedelta

TRADING_ENABLED = False
HOLDOUT_EVALUATED = False
MAX_POSITIONS = 2
COOLDOWN_MINUTES = 30


def select_trades(candidates):
    accepted = []
    last_exit = {}

    for trade in sorted(candidates, key=lambda x: x["entry"]):
        entry = trade["entry"]
        exit_time = trade["exit"]
        pair = trade["pair"]

        assert exit_time > entry

        active = sum(
            t["entry"] <= entry < t["exit"]
            for t in accepted
        )

        if active >= MAX_POSITIONS:
            continue

        if pair in last_exit:
            if entry < last_exit[pair] + timedelta(
                minutes=COOLDOWN_MINUTES
            ):
                continue

        accepted.append(trade)
        last_exit[pair] = exit_time

    return accepted


def test_simulator():
    start = datetime(2026, 1, 1)

    def trade(pair, entry, exit_min):
        return {
            "pair": pair,
            "entry": start + timedelta(minutes=entry),
            "exit": start + timedelta(minutes=exit_min),
        }

    candidates = [
        trade("BTC", 0, 5),
        trade("ETH", 1, 20),
        trade("SOL", 2, 10),
        trade("SOL", 6, 12),
        trade("BTC", 10, 15),
        trade("BTC", 35, 40),
    ]

    accepted = select_trades(candidates)

    assert len(accepted) == 3
    assert [t["pair"] for t in accepted] == [
        "
