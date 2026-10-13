
#!/usr/bin/env python3
"""V6 Stage 1: historical paper-trade accounting tests only."""

from dataclasses import dataclass
import json

FEE = 0.001
SLIPPAGE = 0.0005
ORDER_USDT = 100.0
MAX_SIMULTANEOUS = 2
COOLDOWN_SECONDS = 1800


@dataclass(frozen=True)
class PaperTrade:
    pair: str
    entry_ts: int
    exit_ts: int
    exit_reason: str
    entry_reference: float
    exit_reference: float
    entry_fill: float
    exit_fill: float
    gross_return: float
    after_slippage_return: float
    net_return: float


def simulate_long_v6(pair, timestamps, opens, highs, lows,
                     closes, signal_index, atr, horizon=15):
    """Next-candle entry; stop-first conservative paper fills."""
    i = signal_index

    if i < 0 or i + horizon >= len(timestamps):
        raise ValueError("Insufficient candles")

    if not all(len(a) == len(timestamps)
               for a in (opens, highs, lows, closes)):
        raise ValueError("Different array lengths")

    if any(timestamps[j + 1] - timestamps[j] != 60
           for j in range(i, i + horizon)):
        raise ValueError("Missing minute candles")

    entry_ref = float(opens[i + 1])
    if entry_ref <= 0 or atr < 0:
        raise ValueError("Invalid price or ATR")

    entry_fill = entry_ref * (1 + SLIPPAGE)

    stop_pct = min(
        0.03,
        max(0.003, 1.6 * atr / entry_fill)
    )
    stop = entry_fill * (1 - stop_pct)
    target = entry_fill * (1 + 1.8 * stop_pct)

    exit_index = i + horizon
    exit_ref = float(closes[exit_index])
    reason = "TIME"

    for j in range(i + 1, i + horizon + 1):
        if j > i + 1 and opens[j] <= stop:
            exit_index = j
            exit_ref = float(opens[j])
            reason = "SL_GAP"
            break

        if j > i + 1 and opens[j] >= target:
            exit_index = j
            exit_ref = float(opens[j])
            reason = "TP_GAP"
            break

        if lows[j] <= stop:
            exit_index = j
            exit_ref = stop
            reason = "SL"
            break

        if highs[j] >= target:
            exit_index = j
            exit_ref = target
            reason = "TP"
            break

    exit_fill = exit_ref * (1 - SLIPPAGE)

    gross = exit_ref / entry_ref - 1
    after_slippage = exit_fill / entry_fill - 1
    net = (
        exit_fill * (1 - FEE)
        / (entry_fill * (1 + FEE))
        - 1
    )

    return PaperTrade(
        pair=pair,
        entry_ts=int(timestamps[i + 1]),
        exit_ts=int(timestamps[exit_index]),
        exit_reason=reason,
        entry_reference=entry_ref,
        exit_reference=exit_ref,
        entry_fill=entry_fill,
        exit_fill=exit_fill,
        gross_return=gross,
        after_slippage_return=after_slippage,
        net_return=net,
    )


def select_trades_v6(trades, probabilities, threshold):
    """Historical capacity accounting using actual simulated exits."""
    if len(trades) != len(probabilities):
        raise ValueError("Length mismatch")

    eligible = [
        (trade, float(prob))
        for trade, prob in zip(trades, probabilities)
        if prob >= threshold
    ]

    eligible.sort(
        key=lambda item: (
            item[0].entry_ts,
            -item[1],
            item[0].pair
        )
    )

    selected = []
    active_exits = []
    last_pair_entry = {}

    for trade, probability in eligible:
        active_exits = [
            end for end in active_exits
            if end > trade.entry_ts
        ]

        if len(active_exits) >= MAX_SIMULTANEOUS:
            continue

        previous = last_pair_entry.get(trade.pair)

        if previous is not None:
            if trade.entry_ts - previous < COOLDOWN_SECONDS:
                continue

        selected.append((trade, probability))
        active_exits.append(trade.exit_ts)
        last_pair_entry[trade.pair] = trade.entry_ts

    return selected


def accounting_v6(selected):
    trades = [trade for trade, _ in selected]

    gross = sum(
        ORDER_USDT * t.gross_return for t in trades
    )
    after_slippage = sum(
        ORDER_USDT * t.after_slippage_return for t in trades
    )
    net = sum(
        ORDER_USDT * t.net_return for t in trades
    )

    equity = 10000.0
    peak = equity
    max_drawdown = 0.0

    # Realized PnL is booked at actual exit time.
    for trade in sorted(trades, key=lambda t: t.exit_ts):
        equity += ORDER_USDT * trade.net_return
        peak = max(peak, equity)
        max_drawdown = max(
            max_drawdown,
            (peak - equity) / peak
        )

    return {
        "signals": len(trades),
        "gross_pnl_usdt": round(gross, 6),
        "slippage_drag_usdt": round(
            gross - after_slippage, 6
        ),
        "fee_drag_usdt": round(
            after_slippage - net, 6
        ),
        "net_pnl_usdt": round(net, 6),
        "ending_paper_equity_usdt": round(equity, 6),
        "max_closed_equity_drawdown_pct": round(
            100 * max_drawdown, 6
        ),
    }


def self_test():
    timestamps = [
        1700000040 + 60 * i for i in range(20)
    ]

    opens = [100.0] * 20
    highs = [100.1] * 20
    lows = [99.9] * 20
    closes = [100.0] * 20

    # Trigger an early stop on candle 2.
    lows[2] = 99.0

    first = simulate_long_v6(
        "BTC-USD", timestamps, opens, highs,
        lows, closes, 0, 0.2
    )

    assert first.exit_reason == "SL"
    assert first.exit_ts == timestamps[2]

    second = simulate_long_v6(
        "ETH-USD", timestamps, opens, highs,
        lows, closes, 0, 0.2
    )

    third = simulate_long_v6(
        "SOL-USD", timestamps, opens, highs,
        lows, closes, 2, 0.2
    )

    selected = select_trades_v6(
        [first, second, third],
        [0.9, 0.8, 0.7],
        0.5
    )

    assert len(selected) == 3
    assert first.gross_return > first.after_slippage_return
    assert first.after_slippage_return > first.net_return

    result = accounting_v6(selected)

    difference = (
        result["gross_pnl_usdt"]
        - result["slippage_drag_usdt"]
        - result["fee_drag_usdt"]
        - result["net_pnl_usdt"]
    )
    assert abs(difference) < 0.00001

    return {
        "schema": "crypto-radar-v6-stage1",
        "status": "SELF_TEST_PASSED",
        "research_only": True,
        "trading_enabled": False,
        "holdout_evaluated": False,
        "real_market_validation": False,
        "example_accounting": result,
        "warnings": [
            "Synthetic candles only",
            "Mark-to-market drawdown not yet implemented",
            "Historical exits are for backtesting, not prediction",
            "No model approved for live trading"
        ]
    }


if __name__ == "__main__":
    print(json.dumps(
        self_test(),
        ensure_ascii=False,
        indent=2
    ))
