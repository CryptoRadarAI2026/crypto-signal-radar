#!/usr/bin/env python3
"""Research-only validation of archived ML evaluation reports."""

import argparse
import ast
import json
from pathlib import Path


def check(audit, training, source):
    failures = []
    cautions = []

    if audit.get("integrity_passed") is not True:
        failures.append("Source data integrity not confirmed")

    if audit.get("trading_enabled") is not False:
        failures.append("Audit trading flag must be false")

    if training.get("trading_enabled") is not False:
        failures.append("Training trading flag must be false")

    if training.get("research_only") is not True:
        failures.append("Training must remain research-only")

    if audit.get("shared_timestamps", 0) < 1000:
        failures.append("Insufficient shared timestamps")

    split = training.get("chronological_split")

    if not isinstance(split, dict):
        failures.append("Missing chronological split")
    else:
        try:
            a = int(split["train_before"])
            b, c = map(int, split["validation_between"])
            d = int(split["holdout_after"])

            horizon = (
                int(training["label_horizon_minutes"]) * 60
            )

            if not (
                a < b < c < d
                and b - a >= 2 * horizon
                and d - c >= 2 * horizon
            ):
                failures.append(
                    "Invalid chronological boundaries or purge gaps"
                )

        except (KeyError, TypeError, ValueError):
            failures.append("Malformed chronological split")

    rows = training.get("rows", {})

    for key in ("train", "validation", "holdout"):
        value = rows.get(key)

        if not isinstance(value, int) or value < 100:
            failures.append(
                f"Insufficient samples in {key}"
            )

    try:
        tree = ast.parse(source)

        functions = {
            node.name: node
            for node in tree.body
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
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
                        "Manual review required: " + marker
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
                    "Contiguous-window protection not detected"
                )

    except SyntaxError as error:
        failures.append(
            "Training source syntax error: " + str(error)
        )

    cautions.append(
        "Static checks cannot prove absence of all "
        "future-data leakage."
    )

    holdout = training.get("holdout") or {}
    validation = training.get("validation") or {}

    signals = holdout.get("signals", 0)
    mean = holdout.get("mean_net_return_pct")
    profit_factor = holdout.get("profit_factor")

    if not isinstance(signals, int) or signals < 30:
        cautions.append(
            "Fewer than 30 holdout signals."
        )

    if (
        not isinstance(mean, (int, float))
        or mean <= 0
    ):
        cautions.append(
            "Holdout mean net return is not positive."
        )

    if (
        not isinstance(profit_factor, (int, float))
        or profit_factor <= 1
    ):
        cautions.append(
            "Holdout profit factor does not exceed 1."
        )

    return {
        "status": "RESEARCH_ONLY_NOT_APPROVED",
        "trading_enabled": False,
        "checks_passed": not failures,
        "failures": failures,
        "cautions": cautions,
        "split_rows": rows,
        "split_boundaries": split,
        "validation": validation,
        "holdout": holdout,
        "holdout_auc": training.get("holdout_auc"),
        "holdout_brier": training.get("holdout_brier"),
        "constant_baseline_brier": training.get(
            "constant_baseline_brier"
        ),
        "model_approved": False,
    }


def main():
    parser = argparse.ArgumentParser
