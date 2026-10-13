#!/usr/bin/env python3
"""V13 probability calibration research.

Research only.
No live trading.
No holdout predictions, scoring, or model approval.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)


ROOT = Path(__file__).resolve().parents[2]

V11_FILE = (
    ROOT
    / "ai-research"
    / "v11"
    / "research_v11.py"
)


def load_v11():
    """Load V11 functions without executing its main program."""

    spec = importlib.util.spec_from_file_location(
        "research_v11",
        V11_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load V11 research module")

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    return module


def split_validation_only(samples, embargo_seconds):
    """Use chronological train/validation boundaries."""

    dates = sorted({
        sample.ts
        for sample in samples
    })

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

    if len(train) < 100:
        raise ValueError("Insufficient training samples")

    if len(validation) < 100:
        raise ValueError("Insufficient validation samples")

    if len({sample.label for sample in train}) != 2:
        raise ValueError(
            "Training must contain both label classes"
        )

    return train, validation


def build_models():
    """Use the same three model configurations as V11."""

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


def calibration_bins(y_true, probabilities):
    """Calculate statistics for ten fixed probability bins."""

    results = []

    for index in range(10):

        lower = index / 10.0
        upper = (index + 1) / 10.0

        if index == 9:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count > 0:

            mean_probability = float(
                probabilities[mask].mean()
            )

            observed_rate = float(
                y_true[mask].mean()
            )

        else:
            mean_probability = None
            observed_rate = None

        results.append({
            "range": [
                round(lower, 2),
                round(upper, 2),
            ],
            "count": count,
            "mean_predicted_probability": (
                round(mean_probability, 6)
                if mean_probability is not None
                else None
            ),
            "observed_positive_rate": (
                round(observed_rate, 6)
                if observed_rate is not None
                else None
            ),
        })

    return results


def calculate_ece(bins, total_count):
    """Calculate expected calibration error."""

    if total_count <= 0:
        raise ValueError("Empty validation data")

    weighted_error = 0.0

    for item in bins:

        count = item["count"]

        if count == 0:
            continue

        predicted = item[
            "mean_predicted_probability"
        ]

        observed = item[
            "observed_positive_rate"
        ]

        weighted_error += (
            count * abs(predicted - observed)
        )

    return weighted_error / total_count


def evaluate_model(
    model,
    x_train,
    y_train,
    x_validation,
    y_validation,
):
    """Train on training data and evaluate validation only."""

    model.fit(x_train, y_train)

    probabilities = np.asarray(
        model.predict_proba(x_validation)[:, 1],
        dtype=float,
    )

    if len(probabilities) != len(y_validation):
        raise ValueError("Prediction length mismatch")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Non-finite probabilities")

    if np.any(probabilities < 0):
        raise ValueError("Negative probabilities")

    if np.any(probabilities > 1):
        raise ValueError("Probabilities above 1")

    bins = calibration_bins(
        y_validation,
        probabilities,
    )

    ece = calculate_ece(
        bins,
        len(y_validation),
    )

    return {
        "prediction_count": int(len(probabilities)),
        "mean_predicted_probability": round(
            float(probabilities.mean()),
            6,
        ),
        "observed_positive_rate": round(
            float(y_validation.mean()),
            6,
        ),
        "brier_score": round(
            float(
                brier_score_loss(
                    y_validation,
                    probabilities,
                )
            ),
            6,
        ),
        "log_loss": round(
            float(
                log_loss(
                    y_validation,
                    probabilities,
                    labels=[0, 1],
                )
            ),
            6,
        ),
        "expected_calibration_error_10_bins": round(
            float(ece),
            6,
        ),
        "calibration_bins": bins,
    }


def main():

    parser = argparse.ArgumentParser(
        description="V13 validation calibration research"
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
            "reports/v13/calibration.json"
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
                f"Historical data not found: {path}"
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

    y_validation = np.asarray(
        [sample.label for sample in validation],
        dtype=int,
    )

    results = {}

    for name, model in build_models():

        results[name] = evaluate_model(
            model,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )

        print(
            f"{name}: calibration completed",
            flush=True,
        )

    report = {
        "schema": "crypto-radar-ai-research-v13",
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
            "Historical validation data only.",
            "No holdout predictions or scoring.",
            "No profitability test.",
            "No trading approval.",
            "Calibration is not proof of future performance.",
            "Validation period previously used in V11 and V12.",
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
            f"V13_FAILED_SAFELY: {error!r}",
            file=sys.stderr,
        )

        raise SystemExit(2)
