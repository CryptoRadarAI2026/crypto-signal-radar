
from math import isclose

TRADING_ENABLED = False
HOLDOUT_EVALUATED = False

FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
STOP_LOSS_RATE = 0.01
TAKE_PROFIT_RATE = 0.02


def simulate_long(entry_open, candles):
    assert entry_open > 0
    assert candles

    entry_price = entry_open * (1 + SLIPPAGE_RATE)
    stop_price = entry_price * (1 - STOP_LOSS_RATE)
    target_price = entry_price * (1 + TAKE_PROFIT_RATE)

    exit_reason = "TIME_EXIT"
    exit_reference = candles[-1]["close"]

    for candle in candles:
        low = candle["low"]
        high = candle["high"]

        assert 0 < low <= high
        assert low <= candle["close"] <= high

        if low <= stop_price:
            exit_reason = "STOP_LOSS"
            exit_reference = stop_price
            break

        if high >= target_price:
            exit_reason = "TAKE_PROFIT"
            exit_reference = target_price
            break

    exit_price = exit_reference * (1 - SLIPPAGE_RATE)

    gross_return = exit_price / entry_price - 1
    net_return = (
        (exit_price / entry_price)
        * (1 - FEE_RATE)
        / (1 + FEE_RATE)
        - 1
    )

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "gross_return": gross_return,
        "net_return": net_return,
    }


def test_simulator():
    stop = simulate_long(
        100,
        [{"low": 98, "high": 101, "close": 99}],
    )
    assert stop["exit_reason"] == "STOP_LOSS"
    assert stop["net_return"] < 0

    target = simulate_long(
        100,
        [{"low": 100, "high": 103, "close": 102}],
    )
    assert target["exit_reason"] == "TAKE_PROFIT"
    assert target["net_return"] > 0

    both = simulate_long(
        100,
        [{"low": 98, "high": 103, "close": 101}],
    )
    assert both["exit_reason"] == "STOP_LOSS"

    flat = simulate_long(
        100,
        [{"low": 100, "high": 101, "close": 100.5}],
    )
    assert flat["exit_reason"] == "TIME_EXIT"
    assert flat["net_return"] < 0

    no_cost_gross = flat["gross_return"]
    assert flat["net_return"] < no_cost_gross

    assert isclose(
        stop["entry_price"],
        100.05,
        rel_tol=1e-10,
    )

    assert TRADING_ENABLED is False
    assert HOLDOUT_EVALUATED is False

    print("V8 SYNTHETIC SELF-TEST PASSED")
    print("Stop loss: PASSED")
    print("Take profit: PASSED")
    print("Stop-first rule: PASSED")
    print("Fees and slippage: PASSED")
    print("Trading enabled:", TRADING_ENABLED)
    print("Holdout evaluated:", HOLDOUT_EVALUATED)


if __name__ == "__main__":
    test_simulator()
