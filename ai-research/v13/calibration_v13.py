#!/usr/bin/env python3
"""V13 probability calibration research.

Validation only. No trading, no model approval,
no holdout predictions or scoring.
"""

from __future__ import annotations

import argparse
import csv
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
from sklearn.metrics import brier_score_loss, log_loss


ROOT = Path(__file__).resolve().parents[2]
V11_FILE = ROOT / "ai-research/v11/research_v11.py"


def load_v11():
    """Load V11 functions without running its main program."""
    spec = importlib.util.spec_from_file_location(
        "research_v11", V11_FILE
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load V11 research module")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def split_validation_only(samples, embargo_seconds):
    """Use V11 chronological boundaries; never construct holdout features."""

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
        if t1 + embargo_seconds < sample.ts < t2 - embargo_seconds
    ]

    if min(len(train), len(validation)) < 100:
        raise ValueError("Insufficient train/validation samples")

    if len({sample.label for sample in train}) != 2:
        raise ValueError("Training must contain both label classes")

    return train, validation


def build_models():
    """Use the same model configurations as V11."""

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
    """Summarize observed outcomes in fixed probability bins."""

    bins = []

    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10

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

        if count:
            mean_probability = float(
                probabilities[mask].mean()
            )
            observed_rate = float(
                y_true[mask].mean()
            )
        else:
            mean_probability = None
            observed_rate = None

        bins.append({
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

    return bins


def evaluate_model(model, x_train, y_train, x_validation, y_validation):
    model.fit(x_train, y_train)

    probabilities = np.asarray(
        model.predict_proba(x_validation)[:, 1],
        dtype=float,
    )

    if len(probabilities) != len(y_validation):
        raise ValueError("Prediction length mismatch")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Non-finite probabilities")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities outside [0, 1]")

    bins = calibration_bins(
        y_validation,
        probabilities,
    )

    ece = sum(
        item["
