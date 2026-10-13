#!/usr/bin/env python3
"""V12: Safe diagnostic analysis of V11 validation results.

Research only. No trading, no model training, no holdout access.
"""

import argparse
import json
from pathlib import Path


def analyze(report):
    if report.get("schema") != "crypto-radar-ai-research-v11":
        raise ValueError("Invalid V11 report schema")

    if report.get("trading_enabled") is not False:
        raise ValueError("Trading safety check failed")

    if report.get("holdout_evaluated") is not False:
        raise ValueError("Holdout safety check failed")

    if report.get("holdout") is not None:
        raise ValueError("Holdout must remain unevaluated")

    candidates = report.get("validation_candidates")

    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Missing validation candidates")

    models = {}

    for item in candidates:
        name = item["model"]
        signals = int(item["signals"])
        pnl = float(item["paper_pnl_usdt"])

        if name not in models:
            models[name] = {
                "thresholds_tested": 0,
                "thresholds_with_signals": 0,
                "total_signals_across_thresholds": 0,
                "positive_pnl_thresholds": 0,
                "best_pnl_usdt": None,
                "best_threshold": None,
            }

        result = models[name]

        result["thresholds_tested"] += 1
        result["total_signals_across_thresholds"] += signals

        if signals > 0:
            result["thresholds_with_signals"] += 1

        if pnl > 0:
            result["positive_pnl_thresholds"] += 1

        if (
            result["best_pnl_usdt"] is None
            or pnl > result["best_pnl_usdt"]
        ):
            result["best_pnl_usdt"] = pnl
            result["best_threshold"] = item["threshold"]

    findings = []

    for name, result in models.items():
        if result["thresholds_with_signals"] == 0:
            findings.append(
                f"{name}: no signals at tested thresholds"
            )
        elif result["positive_pnl_thresholds"] == 0:
            findings.append(
                f"{name}: all tested thresholds had "
                "non-positive paper PnL"
            )
        else:
            findings.append(
                f"{name}: some thresholds had positive "
                "paper PnL; further validation required"
            )

    return {
        "schema": "crypto-radar-ai-research-v12",
        "status": "DIAGNOSTIC_ONLY_NOT_APPROVED",
        "research_only": True,
        "trading_enabled": False,
        "holdout_evaluated": False,
        "holdout": None,
        "source": "V11 validation metrics only",
        "v11_validation_best": report.get("validation_best"),
        "models": models,
        "findings": findings,
        "limitations": [
            "Threshold results reuse overlapping signals.",
            "Signal counts across thresholds are not unique trades.",
            "No prediction probabilities are available in V11 metrics.",
            "No new model training or profitability test performed.",
            "No holdout data accessed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to V11 metrics.json",
    )

    parser.add_argument(
        "--output",
        default="reports/v12/diagnostics.json",
    )

    args = parser.parse_args()

    source = Path(args.input)
    destination = Path(args.output)

    if not source.is_file():
        raise SystemExit(
            f"V11 metrics file not found: {source}"
        )

    report = json.loads(
        source.read_text(encoding="utf-8")
    )

    result = analyze(report)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
