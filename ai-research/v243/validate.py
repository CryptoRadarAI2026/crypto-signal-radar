#!/usr/bin/env python3
"""Research-only validation of archived ML evaluation reports."""

import argparse
import ast
import json
import math
import sys
from pathlib import Path


STATUS = "RESEARCH_ONLY_NOT_APPROVED"


def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def check(audit, training, source):
    failures = []
    cautions = []

    # ---------------------------------
    # 1. Data integrity
    # ---------------------------------

    if audit.get("integrity_passed") is not True:
        failures.append(
            "Source data integrity not confirmed"
        )

    if audit.get("trading_enabled") is not False:
        failures.append(
            "Audit trading flag must be false"
        )

    if training.get("trading_enabled") is not False:
        failures.append(
            "Training trading flag must be false"
        )

    if training.get("research_only") is not True:
        failures.append(
            "Training must remain research-only"
        )

    shared = audit.get("shared_timestamps", 0)

    if (
        not isinstance(shared, int)
        or isinstance(shared, bool)
        or shared < 1000
    ):
        failures.append(
            "Insufficient shared timestamps"
        )

    # ---------------------------------
    # 2. Chronological split
    # ---------------------------------

    split = training.get("chronological_split")

    if not isinstance(split, dict):
        failures.append(
            "Missing chronological split"
        )
    else:
        try:
            a = int(split["train_before"])

            b, c = map(
                int,
                split["validation_between"]
            )

            d = int(split["holdout_after"])

            horizon = (
                int(training["label_horizon_minutes"])
                * 60
            )

            if horizon <= 0:
                failures.append(
                    "Invalid label horizon"
                )

            elif not (
                a < b < c < d
                and b - a >= 2 * horizon
                and d - c >= 2 * horizon
            ):
                failures.append(
                    "Invalid chronological boundaries "
                    "or purge gaps"
                )

        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ):
            failures.append(
                "Malformed chronological split"
            )

    # ---------------------------------
    # 3. Sample counts
    # ---------------------------------

    rows = training.get("rows", {})

    if not isinstance(rows, dict):
        failures.append(
            "Missing or malformed sample counts"
        )
        rows = {}

    for key in (
        "train",
        "validation",
        "holdout",
    ):
        value = rows.get(key)

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 100
        ):
            failures.append(
                f"Insufficient samples in {key}"
            )

    # ---------------------------------
    # 4. Static source inspection
    # ---------------------------------

    try:
        tree = ast.parse(source)

        functions = {
            node.name: node
            for node in tree.body
            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
        }

        required = {
            "features_and_labels",
            "train",
            "evaluate",
        }

        if not required.issubset(functions):
            failures.append(
                "Expected training functions missing"
            )

        else:
            train_code = (
                ast.get_source_segment(
                    source,
                    functions["train"],
                )
                or ""
            )

            markers = (
                "first - gap",
                "first + gap",
                "second - gap",
                "second + gap",
                "xtr",
                "xva",
                "xte",
            )

            for marker in markers:
                if marker not in train_code:
                    failures.append(
                        "Manual review required: "
                        + marker
                    )

            feature_code = (
                ast.get_source_segment(
                    source,
                    functions["features_and_labels"],
                )
                or ""
            )

            if (
                "ts[i + HORIZON] - ts[i - 49]"
                not in feature_code
            ):
                failures.append(
                    "Contiguous-window protection "
                    "not detected"
                )

    except SyntaxError as error:
        failures.append(
            "Training source syntax error: "
            + str(error)
        )

    cautions.append(
        "Static checks cannot prove absence "
        "of all future-data leakage."
    )

    # ---------------------------------
    # 5. Model performance
    # ---------------------------------

    holdout = training.get("holdout") or {}
    validation = training.get("validation") or {}

    if not isinstance(holdout, dict):
        failures.append(
            "Malformed holdout metrics"
        )
        holdout = {}

    if not isinstance(validation, dict):
        failures.append(
            "Malformed validation metrics"
        )
        validation = {}

    signals = holdout.get("signals", 0)

    mean = holdout.get(
        "mean_net_return_pct"
    )

    profit_factor = holdout.get(
        "profit_factor"
    )

    if (
        not isinstance(signals, int)
        or isinstance(signals, bool)
        or signals < 30
    ):
        cautions.append(
            "Fewer than 30 holdout signals."
        )

    if (
        not is_finite_number(mean)
        or mean <= 0
    ):
        cautions.append(
            "Holdout mean net return "
            "is not positive."
        )

    if (
        not is_finite_number(profit_factor)
        or profit_factor <= 1
    ):
        cautions.append(
            "Holdout profit factor "
            "does not exceed 1."
        )

    # ---------------------------------
    # 6. Final research-only report
    # ---------------------------------

    return {
        "status": STATUS,
        "trading_enabled": False,
        "model_approved": False,
        "checks_passed": not failures,
        "failures": failures,
        "cautions": cautions,
        "split_rows": rows,
        "split_boundaries": split,
        "validation": validation,
        "holdout": holdout,
        "holdout_auc": training.get(
            "holdout_auc"
        ),
        "holdout_brier": training.get(
            "holdout_brier"
        ),
        "constant_baseline_brier": training.get(
            "constant_baseline_brier"
        ),
    }


def read_json(path):
    with path.open(
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return data


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate archived ML research "
            "without enabling trading."
        )
    )

    parser.add_argument(
        "--training",
        type=Path,
        required=True,
        help="Training report JSON",
    )

    parser.add_argument(
        "--audit",
        type=Path,
        required=True,
        help="V24.2 audit report JSON",
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Training Python source",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Validation output JSON",
    )

    args = parser.parse_args()

    try:
        # Read all required inputs.
        training = read_json(
            args.training
        )

        audit = read_json(
            args.audit
        )

        source = args.source.read_text(
            encoding="utf-8"
        )

        # Run validation checks.
        result = check(
            audit,
            training,
            source
        )

        # Create output directory.
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Write JSON report.
        with args.output.open(
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                result,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

            file.write("\n")

        # Confirm report exists.
        if (
            not args.output.is_file()
            or args.output.stat().st_size == 0
        ):
            raise RuntimeError(
                "Validation report was not created"
            )

        # Display summary in Actions log.
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        )

        print(
            "\nValidation report saved to:",
            args.output
        )

        # Fail when integrity checks fail.
        if not result["checks_passed"]:
            print(
                "ERROR: Research validation "
                "checks failed.",
                file=sys.stderr,
            )

            return 1

        print(
            "Research validation checks passed."
        )

        print(
            "Model remains NOT APPROVED "
            "for live trading."
        )

        return 0

    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as error:
        print(
            "ERROR:",
            str(error),
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
