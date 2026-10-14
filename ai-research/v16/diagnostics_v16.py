#!/usr/bin/env python3
"""V16 validation-only trade provenance and exit scenario diagnostics.

Research only. No real orders, no holdout scoring, no model approval.
The five scenarios are exploratory; validation has been reused in V11-V15.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
V11_FILE = ROOT / "ai-research/v11/research_v11.py"
THRESHOLD = 0.50
SCENARIOS = (
    ("A_baseline", 1.6, 1.8, 15),
    ("B_longer", 1.6, 1.8, 30),
    ("C_wider_tp", 1.6, 2.2, 15),
    ("D_tighter_sl", 1.2, 1.8, 15),
    ("E_combined", 1.2, 2.2, 30),
)


def load_v11():
    spec = importlib.util.spec_from_file_location("research_v11", V11_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError("V11 module not found")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def split_validation(samples, embargo):
    samples = sorted(samples, key=lambda s: (s.ts, s.pair))
    timestamps = sorted({s.ts for s in samples})
    if len(timestamps) < 100:
        raise ValueError("Too few timestamps")
    t1 = timestamps[int(len(timestamps) * .55)]
    t2 = timestamps[int(len(timestamps) * .80)]
    train = [s for s in samples if s.ts < t1 - embargo]
    validation = [s for s in samples if t1 + embargo < s.ts < t2 - embargo]
    if min(len(train), len(validation)) < 100 or len({s.label for s in train}) != 2:
        raise ValueError("Invalid training/validation split")
    return train, validation, t2  # exits may enter embargo, never holdout


@dataclass
class Candles:
    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    index: dict


def read_candles(path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["timestamp"]))
    ts = np.asarray([int(r["timestamp"]) for r in rows], dtype=np.int64)
    if len(ts) < 2000 or np.any(np.diff(ts) <= 0):
        raise ValueError(f"Bad timestamps: {path}")
    arrays = [np.asarray([float(r[key]) for r in rows], dtype=float)
              for key in ("open", "high", "low", "close")]
    o, h, l, c = arrays
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError(f"Nonfinite candles: {path}")
    if not (np.all(l > 0) and np.all(l <= np.minimum(o, c)) and
            np.all(h >= np.maximum(o, c))):
        raise ValueError(f"Invalid OHLC: {path}")
    return Candles(ts, o, h, l, c, {int(t): i for i, t in enumerate(ts)})


def replay(sample, candles, v11, atr_mult, tp_ratio, horizon, validation_end):
    """Replay from signal candle, using only next candles within validation.

    ATR14 percent is feature 5; original V11 ATR was computed at signal close.
    """
    i = candles.index.get(sample.ts)
    if i is None or i + horizon >= len(candles.ts):
        return None
    if int(candles.ts[i + horizon]) >= validation_end:
        return None
    if int(candles.ts[i + horizon]) - int(candles.ts[i]) != horizon * 60:
        return None
    slip, fee = float(v11.SLIPPAGE_PER_SIDE), float(v11.FEE_PER_SIDE)
    entry = float(candles.o[i + 1]) * (1 + slip)
    atr = float(sample.x[5]) * float(candles.c[i])
    stop_pct = min(.03, max(.003, atr_mult * atr / entry))
    sl, tp = entry * (1 - stop_pct), entry * (1 + tp_ratio * stop_pct)
    price, reason, exit_i = float(candles.c[i + horizon]), "TIME", i + horizon
    for j in range(i + 1, i + horizon + 1):
        if j > i + 1 and candles.o[j] <= sl:
            price, reason, exit_i = float(candles.o[j]), "SL_GAP", j
            break
        if j > i + 1 and candles.o[j] >= tp:
            price, reason, exit_i = float(candles.o[j]), "TP_GAP", j
            break
        if candles.l[j] <= sl:
            price, reason, exit_i = sl, "SL", j
            break
        if candles.h[j] >= tp:
            price, reason, exit_i = tp, "TP", j
            break
    exit_fill = price * (1 - slip)
    net = exit_fill * (1 - fee) / (entry * (1 + fee)) - 1
    gross = (exit_fill / (1 - slip)) / (entry / (1 + slip)) - 1
    after_slip = exit_fill / entry - 1
    return replace(sample, net_return=float(net), label=int(net > 0),
                   exit_reason=reason, entry_price=entry, exit_price=exit_fill,
                   entry_ts=int(candles.ts[i + 1]), exit_ts=int(candles.ts[exit_i])), {
        "gross_usdt": v11.ORDER_USDT * gross,
        "slippage_usdt": v11.ORDER_USDT * (after_slip - gross),
        "fees_usdt": v11.ORDER_USDT * (net - after_slip),
        "net_usdt": v11.ORDER_USDT * net,
    }


def details(chosen, candlesticks, v11, validation_end):
    records = []
    for sample, probability in chosen:
        outcome = replay(sample, candlesticks[sample.pair], v11, 1.6, 1.8, 15,
                         validation_end)
        if outcome is None:
            raise ValueError("Baseline selected trade lacks validation-only candles")
        reproduced, costs = outcome
        if (reproduced.exit_reason != sample.exit_reason or
                reproduced.exit_ts != sample.exit_ts or
                not math.isclose(reproduced.net_return, sample.net_return,
                                 rel_tol=1e-8, abs_tol=1e-9)):
            raise ValueError("Baseline replay differs from V11; aborting")
        records.append({
            "pair": sample.pair,
            "signal_utc": datetime.fromtimestamp(sample.ts, timezone.utc).isoformat(),
            "entry_utc": datetime.fromtimestamp(sample.entry_ts, timezone.utc).isoformat(),
            "exit_utc": datetime.fromtimestamp(sample.exit_ts, timezone.utc).isoformat(),
            "probability": round(float(probability), 6),
            "atr_pct": round(float(sample.x[5]) * 100, 6),
            "rsi_14": round(float(sample.x[4]) * 100, 4),
            "exit_reason": sample.exit_reason,
            "holding_minutes": round((sample.exit_ts - sample.entry_ts) / 60, 3),
            "gross_usdt": round(costs["gross_usdt"], 6),
            "slippage_usdt": round(costs["slippage_usdt"], 6),
            "fees_usdt": round(costs["fees_usdt"], 6),
            "net_usdt": round(costs["net_usdt"], 6),
        })
    return records


def aggregate(chosen, candlesticks, v11, validation_end, scenario):
    _, atr_mult, tp_ratio, horizon = scenario
    totals = defaultdict(float)
    per_pair = defaultdict(lambda: {"trades": 0, "net_usdt": 0.0})
    exits = Counter()
    for sample, _ in chosen:
        outcome = replay(sample, candlesticks[sample.pair], v11, atr_mult,
                         tp_ratio, horizon, validation_end)
        if outcome is None:
            raise ValueError("Selected trade cannot be replayed")
        simulated, costs = outcome
        exits[simulated.exit_reason] += 1
        per_pair[sample.pair]["trades"] += 1
        per_pair[sample.pair]["net_usdt"] += costs["net_usdt"]
        for k, value in costs.items():
            totals[k] += value
    return {
        "trades": len(chosen),
        "gross_usdt": round(totals["gross_usdt"], 6),
        "slippage_usdt": round(totals["slippage_usdt"], 6),
        "fees_usdt": round(totals["fees_usdt"], 6),
        "net_usdt": round(totals["net_usdt"], 6),
        "exit_reasons": dict(exits),
        "per_pair": {k: {"trades": v["trades"],
                         "net_usdt": round(v["net_usdt"], 6)}
                     for k, v in sorted(per_pair.items())},
    }


def main():
    parser = argparse.ArgumentParser(description="V16 validation-only diagnostics")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v16/trade_exit_diagnostics.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing overwrite: {args.output}")
    v11 = load_v11()
    samples, candlesticks = [], {}
    for pair in v11.PAIRS:
        path = args.data_dir / f"COINBASE_{pair.replace('-', '')}_90D_1M.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        candlesticks[pair] = read_candles(path)
        samples.extend(v11.samples_from_csv(path, pair))
        print(f"Loaded {pair}", flush=True)
    train, validation, validation_end = split_validation(samples, v11.EMBARGO_SEC)
    model = make_pipeline(RobustScaler(), LogisticRegression(C=.08, max_iter=700))
    model.fit(np.asarray([s.x for s in train], dtype=float),
              np.asarray([s.label for s in train], dtype=int))
    probabilities = model.predict_proba(
        np.asarray([s.x for s in validation], dtype=float))[:, 1]
    baseline = v11.select_trades(validation, probabilities, THRESHOLD)
    print(f"V15-equivalent baseline trades: {len(baseline)}", flush=True)
    provenance = details(baseline, candlesticks, v11, validation_end)

    scenarios = {}
    for scenario in SCENARIOS:
        name, atr_mult, tp_ratio, horizon = scenario
        # Fixed original entries: isolates changes in exit rules.
        fixed = aggregate(baseline, candlesticks, v11, validation_end, scenario)
        # Full rescheduling: all eligible candidates, new exits and cooldowns.
        eligible = []
        missing = 0
        for sample, prob in zip(validation, probabilities):
            if not math.isfinite(float(prob)) or float(prob) < THRESHOLD:
                continue
            outcome = replay(sample, candlesticks[sample.pair], v11,
                             atr_mult, tp_ratio, horizon, validation_end)
            if outcome is None:
                missing += 1
                continue
            simulated, _ = outcome
            eligible.append((simulated, float(prob)))
        if missing:
            # Avoid a biased comparison due to unequal eligibility near the boundary.
            raise ValueError(f"Scenario {name} has {missing} unreplayable candidates")
        rescheduled = v11.select_trades(
            [s for s, _ in eligible], np.asarray([p for _, p in eligible]),
            THRESHOLD)
        full = aggregate(rescheduled, candlesticks, v11, validation_end, scenario)
        scenarios[name] = {
            "atr_multiplier": atr_mult, "tp_to_sl_ratio": tp_ratio,
            "max_holding_minutes": horizon,
            "fixed_v15_entries": fixed,
            "full_reselection_with_new_exits": full,
        }
        print(f"{name}: fixed={fixed['net_usdt']} rescheduled={full['net_usdt']}",
              flush=True)

    baseline_pnl = v11.metrics(baseline)["paper_pnl_usdt"]
    if not math.isclose(scenarios["A_baseline"]["fixed_v15_entries"]["net_usdt"],
                        baseline_pnl, abs_tol=.0011):
        raise ValueError("V16 baseline does not reconcile with V11")
    report = {
        "schema": "crypto-radar-ai-research-v16",
        "status": "EXPLORATORY_VALIDATION_ONLY_NOT_APPROVED",
        "research_only": True, "trading_enabled": False,
        "holdout_evaluated": False, "holdout": None,
        "model": "LogisticRegression", "threshold": THRESHOLD,
        "rows": {"train": len(train), "validation": len(validation)},
        "baseline_trade_count": len(baseline),
        "baseline_trades": provenance,
        "scenarios": scenarios,
        "limitations": [
            "Validation was reused in V11-V15; scenario rankings are exploratory.",
            "Scenario exits may extend into the pre-holdout embargo but never reach the holdout boundary.",
            "No holdout predictions, scoring or performance reporting.",
            "Baseline entries are fixed in one comparison; the other reselects with V11 cooldown rules.",
            "V11 stop-first intrabar convention, approximate gaps, fees and slippage retained.",
            "ATR is reconstructed from the signal candle's past-only atr_14_pct feature.",
            "No real orders, no model approval, no future profitability claim.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"V16_FAILED_SAFELY: {exc!r}", file=sys.stderr)
        raise SystemExit(2)
