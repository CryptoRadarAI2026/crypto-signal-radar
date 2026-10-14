#!/usr/bin/env python3
"""V15: validation-only historical paper-trade exit/cost diagnostics.

No orders, no holdout predictions/scoring, no model approval.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
V11_FILE = ROOT / "ai-research/v11/research_v11.py"
THRESHOLD = 0.50  # Fixed V11 diagnostic threshold, not optimized here.
EXIT_REASONS = ("SL", "SL_GAP", "TP", "TP_GAP", "TIME")


def load_v11():
    spec = importlib.util.spec_from_file_location("research_v11", V11_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot locate V11")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_models():
    return [
        ("LogisticRegression", make_pipeline(
            RobustScaler(), LogisticRegression(C=0.08, max_iter=700))),
        ("RandomForest", RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=30,
            n_jobs=2, random_state=42)),
        ("HistGradientBoosting", HistGradientBoostingClassifier(
            max_iter=80, max_leaf_nodes=12, learning_rate=0.04,
            l2_regularization=3, random_state=42)),
    ]


def validation_split(samples, embargo_seconds):
    """Mirror V11 boundaries without creating or using a holdout subset."""
    samples = sorted(samples, key=lambda s: (s.ts, s.pair))
    timestamps = sorted({s.ts for s in samples})
    if len(timestamps) < 100:
        raise ValueError("Insufficient timestamps")
    t1 = timestamps[int(len(timestamps) * 0.55)]
    t2 = timestamps[int(len(timestamps) * 0.80)]
    train = [s for s in samples if s.ts < t1 - embargo_seconds]
    validation = [s for s in samples
                  if t1 + embargo_seconds < s.ts < t2 - embargo_seconds]
    if min(len(train), len(validation)) < 100:
        raise ValueError("Insufficient samples")
    if len({s.label for s in train}) != 2:
        raise ValueError("Training requires two classes")
    return train, validation


def reconstruct_costs(sample, v11):
    """Sequential attribution that exactly reconciles to V11 net returns.

    V11 Sample entry_price and exit_price already include slippage.
    The gross baseline reverses both fills' slippage factors. Because
    V11 stops/targets depend on slipped entry, this is an accounting
    decomposition, NOT a counterfactual zero-slippage backtest.
    """
    entry_fill = float(sample.entry_price)
    exit_fill = float(sample.exit_price)
    fee = float(v11.FEE_PER_SIDE)
    slip = float(v11.SLIPPAGE_PER_SIDE)
    raw_entry = entry_fill / (1.0 + slip)
    raw_exit = exit_fill / (1.0 - slip)
    gross_return = raw_exit / raw_entry - 1.0
    after_slippage_return = exit_fill / entry_fill - 1.0
    net_return = exit_fill * (1.0 - fee) / (entry_fill * (1.0 + fee)) - 1.0
    if not math.isclose(net_return, float(sample.net_return),
                        rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Cost reconstruction does not match V11")
    order = float(v11.ORDER_USDT)
    return {
        "gross_pnl": order * gross_return,
        "slippage_impact": order * (after_slippage_return - gross_return),
        "fee_impact": order * (net_return - after_slippage_return),
        "net_pnl": order * net_return,
    }


def summarize_trades(chosen, v11):
    reasons = Counter()
    totals = {"gross_pnl": 0.0, "slippage_impact": 0.0,
              "fee_impact": 0.0, "net_pnl": 0.0}
    grouped = {
        reason: {
            "count": 0,
            "gross_pnl_usdt": 0.0,
            "slippage_impact_usdt": 0.0,
            "fee_impact_usdt": 0.0,
            "net_pnl_usdt": 0.0,
        }
        for reason in EXIT_REASONS
    }
    for sample, _probability in chosen:
        reason = sample.exit_reason
        if reason not in grouped:
            raise ValueError(f"Unknown V11 exit reason: {reason}")
        costs = reconstruct_costs(sample, v11)
        reasons[reason] += 1
        grouped[reason]["count"] += 1
        for key, value in costs.items():
            totals[key] += value
            grouped[reason][key + "_usdt"] += value
    if not math.isclose(
        totals["gross_pnl"] + totals["slippage_impact"] + totals["fee_impact"],
        totals["net_pnl"], rel_tol=1e-9, abs_tol=1e-8,
    ):
        raise ValueError("PnL components do not reconcile")
    v11_metrics = v11.metrics(chosen)
    if not math.isclose(totals["net_pnl"], v11_metrics["paper_pnl_usdt"],
                        abs_tol=0.0011):
        raise ValueError("Net PnL does not match V11 metrics")
    for group in grouped.values():
        for key in ("gross_pnl_usdt", "slippage_impact_usdt",
                    "fee_impact_usdt", "net_pnl_usdt"):
            group[key] = round(group[key], 6)
    return {
        "signals": len(chosen),
        "exit_reasons": {reason: reasons[reason] for reason in EXIT_REASONS},
        "exit_reason_breakdown": grouped,
        "gross_pnl_usdt": round(totals["gross_pnl"], 6),
        "slippage_impact_usdt": round(totals["slippage_impact"], 6),
        "fee_impact_usdt": round(totals["fee_impact"], 6),
        "net_pnl_usdt": round(totals["net_pnl"], 6),
        "win_rate_pct": v11_metrics["win_rate_pct"],
        "profit_factor": v11_metrics["profit_factor"],
        "max_closed_equity_drawdown_pct":
            v11_metrics["max_closed_equity_drawdown_pct"],
    }


def main():
    parser = argparse.ArgumentParser(description="V15 validation-only diagnostics")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/v15/exit_cost_diagnostics.json"))
    args = parser.parse_args()
    v11 = load_v11()
    samples = []
    for pair in v11.PAIRS:
        path = args.data_dir / f"COINBASE_{pair.replace('-', '')}_90D_1M.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Historical data not found: {path}")
        batch = v11.samples_from_csv(path, pair)
        samples.extend(batch)
        print(f"{pair}: {len(batch)} samples", flush=True)
    train, validation = validation_split(samples, v11.EMBARGO_SEC)
    x_train = np.asarray([s.x for s in train], dtype=float)
    y_train = np.asarray([s.label for s in train], dtype=int)
    x_validation = np.asarray([s.x for s in validation], dtype=float)
    results = {}
    for name, model in build_models():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_validation)[:, 1]
        chosen = v11.select_trades(validation, probabilities, THRESHOLD)
        results[name] = summarize_trades(chosen, v11)
        print(f"{name}: {len(chosen)} paper trades", flush=True)
    report = {
        "schema": "crypto-radar-ai-research-v15",
        "status": "DIAGNOSTIC_ONLY_NOT_APPROVED",
        "research_only": True,
        "trading_enabled": False,
        "holdout_evaluated": False,
        "holdout": None,
        "data_split": "validation_only",
        "threshold": THRESHOLD,
        "rows": {"train": len(train), "validation": len(validation)},
        "transaction_assumptions": {
            "order_usdt": v11.ORDER_USDT,
            "fee_per_side": v11.FEE_PER_SIDE,
            "slippage_per_side": v11.SLIPPAGE_PER_SIDE,
            "max_simultaneous": v11.MAX_SIMULTANEOUS,
            "pair_cooldown_seconds": v11.PAIR_COOLDOWN_SEC,
            "max_holding_minutes": v11.HORIZON,
        },
        "models": results,
        "limitations": [
            "Historical validation data only; reused in V11-V14.",
            "Fixed V11 diagnostic threshold 0.50; not a trading recommendation.",
            "No holdout predictions or scoring, no real trading or model approval.",
            "Sequential accounting attribution, not a no-cost counterfactual.",
            "Stop and take-profit levels depend on slipped entry price.",
            "Historical paper fills are approximations; no future profit guarantee.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite: {args.output}")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": report["rows"],
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"V15_FAILED_SAFELY: {error!r}", file=sys.stderr)
        raise SystemExit(2)
