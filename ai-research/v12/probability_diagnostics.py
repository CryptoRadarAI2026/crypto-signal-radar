#!/usr/bin/env python3
"""V12 probability diagnostics — research only.

Analyzes validation prediction probabilities.
Does not train models, trade, or access holdout data.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def analyze(path):
    models = defaultdict(list)

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required = {"split", "model", "probability"}

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "CSV requires split, model, probability columns"
            )

        for row in reader:
            if row["split"] != "validation":
                raise ValueError(
                    "Only validation predictions are permitted"
                )

            model = row["model"].strip()

            if not model:
                raise ValueError("Missing model name")

            probability = float(row["probability"])

            if not (
                math.isfinite(probability)
                and 0 <= probability <= 1
            ):
                raise ValueError("Invalid probability")

            models[model].append(probability)

    if not models:
        raise ValueError("No validation predictions found")

    results = {}

    for name, values in sorted(models.items()):
        values.sort()
        count = len(values)

        def percentile(q):
            index = round((count - 1) * q)
            return round(values[index], 6)

        results[name] = {
            "prediction_count": count,
            "minimum": round(values[0], 6),
            "maximum": round(values[-1], 6),
            "mean": round(sum(values) / count, 6),
            "p10": percentile(0.10),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p99": percentile(0.99),
            "counts_at_thresholds": {
                str(threshold): sum(
                    p >= threshold for p in values
                )
                for threshold in THRESHOLDS
            },
        }

    return {
        "schema": "crypto-radar-ai-v12-probability-diagnostics",
        "status": "DIAGNOSTIC_ONLY_NOT_APPROVED",
        "research_only": True,
        "trading_enabled": False,
        "holdout_evaluated": False,
        "holdout": None,
        "input": "validation predictions only",
        "models": results,
        "limitations": [
            "Counts are predictions, not executed trades.",
            "No profitability evaluation performed.",
            "No model approval or threshold optimization.",
            "Input must contain validation data only.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Validation predictions CSV",
    )

    parser.add_argument(
        "--output",
        default="reports/v12/probability_diagnostics.json",
    )

    args = parser.parse_args()

    source = Path(args.input)

    if not source.is_file():
        raise SystemExit(
            f"Validation predictions file not found: {source}"
        )

    result = analyze(source)

    destination = Path(args.output)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
