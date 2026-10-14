#!/usr/bin/env python3
"""V14: exploratory economic analysis of validation predictions.

Historical research only.
No real orders, no model approval, no holdout evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)


ROOT = Path(__file__).resolve().parents[2]
V11_FILE = ROOT / "ai-research/v11/research_v11.py"

PROBABILITY_BINS = [
    (i / 10.0, (i + 1) / 10.0)
    for i in range(10)
]


def load_v11():
    """Import V11 functions without executing its main program."""

    spec = importlib.util.spec_from_file_location(
        "research_v11",
        V11_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load V11")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def split_validation_only(samples, embargo_seconds):
    """Match V11 chronological boundaries.

    Holdout samples are not returned or used for prediction.
    """

    dates = sorted({sample.ts for sample in samples})

    if len(dates) < 100:
        raise ValueError("Insufficient timestamps")

    t1 = dates[int(len(dates) * 0.55)]
    t2 = dates[int(len(dates) * 0.80)]

    train = [
        sample
        for sample in samples
        if sample.ts < t1 - embargo_seconds
    ]

    validation = [
        sample
        for sample in samples
        if (
            t1 + embargo_seconds
            < sample.ts
            < t2 - embargo_seconds
        )
    ]

    if min(len(train), len(validation)) < 100:
        raise ValueError("Insufficient train/validation samples")

    if len({sample.label for sample in train}) != 2:
        raise ValueError("Training needs both label classes")

    return train, validation


def build_models():
    """Same model configurations as V11."""

    return [
        (
            "LogisticRegression",
            make_pipeline(
                RobustScaler(),
                LogisticRegression(
                    C=0.08,
                    max_iter=700,
                ),
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=30,
                n_jobs=2,
                random_state=42,
            ),
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(
                max_iter=80,
                max_leaf_nodes=12,
                learning_rate=0.04,
                l2_regularization=3,
                random_state=42,
            ),
        ),
    ]


def rounded(value):
    if value is None:
        return None

    return round(float(value), 6)


def sample_statistics(samples, order_usdt):
    """Statistics without portfolio capacity or cooldown.

    These samples are NOT simultaneously executable trades.
    """

    count = len(samples)

    if count == 0:
        return {
            "count": 0,
            "positive_labels": 0,
            "positive_label_rate": None,
            "mean_net_return_pct": None,
            "mean_paper_pnl_usdt_per_sample": None,
            "sum_hypothetical_pnl_usdt": None,
        }

    returns = np.asarray(
        [sample.net_return for sample in samples],
        dtype=float,
    )

    labels = np.asarray(
        [sample.label for sample in samples],
        dtype=int,
    )

    return {
        "count": count,
        "positive_labels": int(labels.sum()),
        "positive_label_rate": rounded(labels.mean()),
        "mean_net_return_pct": rounded(
            100.0 * returns.mean()
        ),
        "mean_paper_pnl_usdt_per_sample": rounded(
            order_usdt * returns.mean()
        ),
        "sum_hypothetical_pnl_usdt": rounded(
            order_usdt * returns.sum()
        ),
    }


def bin_mask(probabilities, lower, upper):
    if upper == 1.0:
        return (
            (probabilities >= lower)
            & (probabilities <= upper)
        )

    return (
        (probabilities >= lower)
        & (probabilities < upper)
    )


def analyze_model(
    name,
    model,
    train,
    validation,
    v11,
):
    x_train = np.asarray(
        [sample.x for sample in train],
        dtype=float,
    )

    y_train = np.asarray(
        [sample.label for sample in train],
        dtype=int,
    )

    x_validation = np.asarray(
        [sample.x for sample in validation],
        dtype=float,
    )

    model.fit(x_train, y_train)

    probabilities = np.asarray(
        model.predict_proba(x_validation)[:, 1],
        dtype=float,
    )

    if len(probabilities) != len(validation):
        raise ValueError("Prediction length mismatch")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Non-finite probabilities")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Invalid probabilities")

    bins = []

    for lower, upper in PROBABILITY_BINS:
        mask = bin_mask(
            probabilities,
            lower,
            upper,
        )

        indices = np.flatnonzero(mask)

        group = [
            validation[int(index)]
            for index in indices
        ]

        group_probabilities = probabilities[mask]

        # Independent sample-level hypothetical outcomes.
        raw_stats = sample_statistics(
            group,
            v11.ORDER_USDT,
        )

        # Chronological paper-trade selection using V11 rules.
        # The filter selects only predictions within this bin.
        chosen = v11.select_trades(
            group,
            group_probabilities,
            lower,
        )

        executable = v11.metrics(chosen)

        bins.append({
            "probability_range": [
                rounded(lower),
                rounded(upper),
            ],
            "mean_predicted_probability": (
                rounded(group_probabilities.mean())
                if len(group_probabilities)
                else None
            ),
            "sample_statistics": raw_stats,
            "independent_bin_paper_selection": {
                "signals": executable["signals"],
                "paper_pnl_usdt": executable[
                    "paper_pnl_usdt"
                ],
                "profit_factor": executable[
                    "profit_factor"
                ],
                "win_rate_pct": executable[
                    "win_rate_pct"
                ],
                "max_closed_equity_drawdown_pct": executable[
                    "max_closed_equity_drawdown_pct"
                ],
            },
        })

    return {
        "model": name,
        "validation_predictions": len(probabilities),
        "minimum_probability": rounded(
            probabilities.min()
        ),
        "maximum_probability": rounded(
            probabilities.max()
        ),
        "mean_probability": rounded(
            probabilities.mean()
        ),
        "probability_bins": bins,
    }


def main():
    parser = argparse.ArgumentParser(
        description="V14 validation economic diagnostics"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/v14/economic_diagnostics.json"
        ),
    )

    args = parser.parse_args()

    v11 = load_v11()

    samples = []

    for pair in v11.PAIRS:
        filename = (
            f"COINBASE_"
            f"{pair.replace('-', '')}"
            f"_90D_1M.csv"
        )

        path = args.data_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Historical CSV not found: {path}"
            )

        batch = v11.samples_from_csv(
            path,
            pair,
        )

        samples.extend(batch)

        print(
            f"{pair}: {len(batch)} samples",
            flush=True,
        )

    train, validation = split_validation_only(
        samples,
        v11.EMBARGO_SEC,
    )

    results = {}

    for name, model in build_models():
        results[name] = analyze_model(
            name,
            model,
            train,
            validation,
            v11,
        )

        print(
            f"{name}: analysis completed",
            flush=True,
        )

    report = {
        "schema": "crypto-radar-ai-research-v14",
        "status": "DIAGNOSTIC_ONLY_NOT_APPROVED",
        "research_only": True,
        "trading_enabled": False,
        "holdout_evaluated": False,
        "holdout": None,
        "data_split": "validation_only",
        "rows": {
            "train": len(train),
            "validation": len(validation),
        },
        "models": results,
        "limitations": [
            "Exploratory analysis on reused validation data.",
            "No holdout predictions or scoring.",
            "No profitability-based model approval.",
            "Each probability bin is replayed independently.",
            "Bin paper PnLs must not be added together.",
            "Independent samples can overlap in time.",
            "Historical paper fills are approximations.",
            "No proof of future profitability.",
        ],
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.output.exists():
        raise RuntimeError(
            f"Refusing to overwrite: {args.output}"
        )

    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps({
            "status": report["status"],
            "rows": report["rows"],
            "output": str(args.output),
        }),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"V14_FAILED_SAFELY: {error!r}",
            file=sys.stderr,
        )
        raise SystemExit(2)
