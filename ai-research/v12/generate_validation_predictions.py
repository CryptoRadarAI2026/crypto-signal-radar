#!/usr/bin/env python3
"""V12 validation probability generator.

Research only. No orders, no model approval, no holdout evaluation.
"""

from __future__ import annotations

import argparse
import csv
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


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v11.research_v11 import (
    PAIRS,
    EMBARGO_SEC,
    Sample,
    samples_from_csv,
)


def safe_split(samples):
    """Split chronologically without accessing holdout features."""

    dates = sorted({sample.ts for sample in samples})

    if len(dates) < 100:
        raise ValueError("Insufficient timestamps")

    t1 = dates[int(len(dates) * 0.55)]
    t2 = dates[int(len(dates) * 0.80)]

    train = [
        sample
        for sample in samples
        if sample.ts < t1 - EMBARGO_SEC
    ]

    validation = [
        sample
        for sample in samples
        if t1 + EMBARGO_SEC < sample.ts < t2 - EMBARGO_SEC
    ]

    if len(train) < 100 or len(validation) < 100:
        raise ValueError("Insufficient training or validation rows")

    if len({sample.label for sample in train}) != 2:
        raise ValueError("Training requires two label classes")

    return train, validation, t1, t2


def build_models():
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


def generate(samples, output):
    train, validation, t1, t2 = safe_split(samples)

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

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing file: {output}"
        )

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "split",
            "model",
            "probability",
        ])

        for name, model in build_models():
            model.fit(x_train, y_train)

            probabilities = model.predict_proba(
                x_validation
            )[:, 1]

            for probability in probabilities:
                writer.writerow([
                    "validation",
                    name,
                    f"{float(probability):.10f}",
                ])

            print(
                f"{name}: {len(probabilities)} "
                "validation probabilities",
                flush=True,
            )

    return {
        "status": "VALIDATION_PROBABILITIES_GENERATED",
        "research_only": True,
        "trading_enabled": False,
        "holdout_evaluated": False,
        "holdout": None,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_validation_boundary": t1,
        "validation_holdout_boundary": t2,
        "output": str(output),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/v12/validation_predictions.csv"
        ),
    )

    args = parser.parse_args()

    samples = []

    for pair in PAIRS:
        path = args.data_dir / (
            f"COINBASE_{pair.replace('-', '')}_90D_1M.csv"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Historical CSV not found: {path}"
            )

        batch = samples_from_csv(path, pair)
        samples.extend(batch)

        print(
            f"{pair}: {len(batch)} samples",
            flush=True,
        )

    result = generate(samples, args.output)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"V12_FAILED_SAFELY: {error!r}",
            file=sys.stderr,
        )
        raise SystemExit(2)
