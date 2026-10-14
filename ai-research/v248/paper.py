"""V24.8 research-only hourly BTC paper simulator; no exchange credentials/orders."""

import csv
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

STATE = ROOT / "research-output/v248/state.json"
REPORT = ROOT / "research-output/v248/report.json"

CAPITAL = 10000.0

FEE = 0.001
SLIPPAGE = 0.0005

MAX_ALLOCATION = 0.10

HORIZON = 5
FEATURE_WARMUP = 25
MIN_ROWS = 650


def fetch_json(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CryptoRadarAI2026-V248-paper-research/1.0",
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(req, timeout=25) as response:
        return json.load(response)


def fetch_candles():
    # Fetch 60 days of hourly BTC-USD candles.
    # Only completed candles are accepted.

    end = (int(time.time()) // 3600) * 3600
    start = end - 60 * 24 * 3600

    rows = {}

    for a in range(start, end, 299 * 3600):
        b = min(a + 299 * 3600, end)

        query = urllib.parse.urlencode(
            {
                "start": datetime.fromtimestamp(
                    a, timezone.utc
                ).isoformat(),
                "end": datetime.fromtimestamp(
                    b, timezone.utc
                ).isoformat(),
                "granularity": 3600,
            }
        )

        result = fetch_json(
            "https://api.exchange.coinbase.com/"
            "products/BTC-USD/candles?" + query
        )

        if not isinstance(result, list):
            raise RuntimeError("Unexpected candle response")

        for x in result:
            if (
                len(x) >= 6
                and a <= int(x[0]) < b
                and int(x[0]) % 3600 == 0
            ):
                rows[int(x[0])] = [
                    float(x[3]),
                    float(x[2]),
                    float(x[1]),
                    float(x[4]),
                    float(x[5]),
                ]

        time.sleep(0.2)

    stamps = sorted(rows)

    if len(stamps) < MIN_ROWS or any(
        b - a != 3600
        for a, b in zip(stamps, stamps[1:])
    ):
        raise RuntimeError(
            "Incomplete hourly data: refuse to trade"
        )

    if stamps[-1] < end - 3600:
        raise RuntimeError(
            "Stale candles: refuse to trade"
        )

    return stamps, np.asarray(
        [rows[t] for t in stamps],
        dtype=float,
    )


def features(data):
    closes = data[:, 3]
    volumes = data[:, 4]

    X = []

    for i in range(FEATURE_WARMUP, len(data)):
        r1 = math.log(
            closes[i] / closes[i - 1]
        )

        r6 = math.log(
            closes[i] / closes[i - 6]
        )

        r24 = math.log(
            closes[i] / closes[i - 24]
        )

        past = np.diff(
            np.log(closes[i - 23:i + 1])
        )

        vol = float(np.std(past))

        volume_ratio = float(
            volumes[i] /
            max(
                1e-9,
                np.mean(volumes[i - 23:i + 1])
            )
        )

        range_ratio = float(
            (data[i, 1] - data[i, 2]) /
            closes[i]
        )

        X.append(
            [
                r1,
                r6,
                r24,
                vol,
                volume_ratio,
                range_ratio,
            ]
        )

    return np.asarray(X, dtype=float)


def predict(data):
    X = features(data)
    closes = data[:, 3]

    train_count = len(X) - HORIZON - 1

    if train_count < 500:
        return None, "INSUFFICIENT_TRAINING_DATA"

    y = np.asarray(
        [
            int(
                closes[i + HORIZON] >
                closes[i]
            )
            for i in range(
                FEATURE_WARMUP,
                FEATURE_WARMUP + train_count
            )
        ]
    )

    if len(set(y)) < 2:
        return None, "ONE_CLASS_TRAINING_DATA"

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500,
            random_state=42,
        ),
    )

    model.fit(
        X[:train_count],
        y,
    )

    probability = float(
        model.predict_proba(
            X[-1:]
        )[0, 1]
    )

    return probability, "OK"


def sentiment():
    # Fear & Greed is a third-party market proxy.
    # Missing or stale sentiment blocks new entries.

    try:
        result = fetch_json(
            "https://api.alternative.me/"
            "fng/?limit=1&format=json"
        )

        item = result["data"][0]

        age = (
            time.time() -
            int(item["timestamp"])
        )

        if not 0 <= age <= 36 * 3600:
            return None, "STALE_SENTIMENT"

        value = int(item["value"])

        if not 0 <= value <= 100:
            raise ValueError(
                "Invalid sentiment"
            )

        return value, "OK"

    except (
        KeyError,
        ValueError,
        IndexError,
        OSError,
        TimeoutError,
    ) as exc:
        return (
            None,
            "SENTIMENT_UNAVAILABLE_" +
            type(exc).__name__,
        )


def initial_state():
    return {
        "version": "V24.8",
        "status": "PAPER_TRADING_ONLY",
        "cash": CAPITAL,
        "btc": 0.0,
        "position": None,
        "last_bar": None,
        "trades": [],
        "events": [],
    }


def load_state():
    if not STATE.exists():
        return initial_state()

    s = json.loads(
        STATE.read_text(
            encoding="utf-8"
        )
    )

    if s.get("status") != "PAPER_TRADING_ONLY":
        raise RuntimeError(
            "Unsafe state status"
        )

    return s


def execute(
    stamps,
    data,
    s,
    p,
    reason,
    mood,
):
    t = stamps[-1]
    close = float(data[-1, 3])

    if (
        s["last_bar"] is not None
        and t <= s["last_bar"]
    ):
        return "DUPLICATE_OR_OLD_BAR"

    if s["position"]:
        pos = s["position"]

        later = [
            (int(stamps[i]), data[i])
            for i in range(len(stamps))
            if stamps[i] > pos["entry_bar"]
        ]

        exit_price = None
        exit_reason = None
        exit_bar = t

        for bt, bar in later:
            if bar[2] <= pos["stop"]:
                exit_price = pos["stop"]
                exit_reason = "STOP"
                exit_bar = bt
                break

            if bar[1] >= pos["target"]:
                exit_price = pos["target"]
                exit_reason = "TARGET"
                exit_bar = bt
                break

            if bt >= pos["expiry_bar"]:
                exit_price = float(bar[3])
                exit_reason = "TIME"
                exit_bar = bt
                break

        if exit_price is not None:
            fill = exit_price * (
                1 - SLIPPAGE
            )

            proceeds = s["btc"] * fill
            exit_fee = proceeds * FEE

            s["cash"] += (
                proceeds - exit_fee
            )

            pnl = (
                proceeds -
                exit_fee -
                pos["cost"]
            )

            s["trades"].append(
                {
                    "entry_bar": pos["entry_bar"],
                    "exit_bar": exit_bar,
                    "reason": exit_reason,
                    "pnl_usd": round(pnl, 4),
                    "entry_cost_usd": pos["cost"],
                    "exit_fee_usd": round(
                        exit_fee, 4
                    ),
                }
            )

            s["btc"] = 0.0
            s["position"] = None

    can_enter = (
        s["position"] is None
        and reason == "OK"
        and mood is not None
        and 25 <= mood <= 75
        and p is not None
        and p >= 0.62
    )

    if can_enter:
        allocation = (
            s["cash"] * MAX_ALLOCATION
        )

        fill = close * (
            1 + SLIPPAGE
        )

        fee = allocation * FEE

        qty = (
            allocation - fee
        ) / fill

        s["cash"] -= allocation
        s["btc"] = qty

        s["position"] = {
            "entry_bar": t,
            "expiry_bar": (
                t + HORIZON * 3600
            ),
            "entry_price": fill,
            "stop": fill * 0.985,
            "target": fill * 1.03,
            "cost": allocation,
            "entry_fee_usd": fee,
            "probability": p,
            "sentiment": mood,
        }

    s["last_bar"] = t

    s["events"].append(
        {
            "bar": t,
            "probability": p,
            "model_status": reason,
            "sentiment": mood,
            "position_open": (
                s["position"] is not None
            ),
        }
    )

    s["events"] = (
        s["events"][-500:]
    )

    return "PROCESSED"


def main():
    stamps, data = fetch_candles()

    state = load_state()

    if (
        state["last_bar"] is not None
        and stamps[-1] <= state["last_bar"]
    ):
        print(
            "No new completed candle"
        )
        return

    p, reason = predict(data)

    mood, sentiment_status = sentiment()

    action = execute(
        stamps,
        data,
        state,
        p,
        reason,
        mood,
    )

    STATE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATE.write_text(
        json.dumps(
            state,
            indent=2,
        ),
        encoding="utf-8",
    )

    price = float(data[-1, 3])

    report = {
        "version": "V24.8",
        "status": "PAPER_TRADING_ONLY",
        "trading_enabled": False,
        "orders_sent": 0,
        "bar_utc": datetime.fromtimestamp(
            stamps[-1],
            timezone.utc,
        ).isoformat(),
        "action": action,
        "model_probability_uncalibrated": p,
        "model_status": reason,
        "sentiment_index": mood,
        "sentiment_status": sentiment_status,
        "sentiment_is_third_party_proxy": True,
        "closed_trades": len(
            state["trades"]
        ),
        "cash_usd": round(
            state["cash"], 2
        ),
        "btc_virtual": state["btc"],
        "equity_usd": round(
            state["cash"] +
            state["btc"] * price,
            2,
        ),
        "realized_pnl_usd": round(
            sum(
                trade["pnl_usd"]
                for trade in state["trades"]
            ),
            2,
        ),
        "position": state["position"],
        "limitations": [
            "No live execution or exchange orders",
            "Uncalibrated model; no demonstrated out-of-sample edge",
            "Hourly polling and OHLC approximations; gaps between runs",
            "Fear and Greed is unverified third-party proxy",
            "Fees and slippage are assumptions",
        ],
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
