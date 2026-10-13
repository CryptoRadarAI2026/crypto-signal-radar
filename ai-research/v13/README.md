# Crypto Signal Radar AI — Research V13

## Purpose

V13 investigates probability calibration for three models:

1. LogisticRegression
2. RandomForest
3. HistGradientBoosting

The goal is to compare predicted probabilities with observed
historical outcomes on validation data only.

## Previous findings

V11 rejected all models on validation.

V12 found the following maximum predicted probabilities:

- LogisticRegression: 0.981916
- RandomForest: 0.440255
- HistGradientBoosting: 0.452557

RandomForest and HistGradientBoosting produced no predictions
at or above the 0.50 threshold.

These findings do not establish model profitability.

## V13 research plan

1. Train the same three models using training data only.
2. Generate probabilities for validation samples only.
3. Compare probabilities against validation labels.
4. Calculate Brier score and calibration statistics.
5. Produce probability-bin summaries.
6. Report limitations without approving any model.

## Safety rules

- Research only.
- No real trading.
- No exchange credentials.
- No order execution.
- No model deployment.
- No changes to V2, V11, or V12.
- No holdout predictions.
- No holdout scoring.
- No holdout-based decisions.

## Approval status

NOT APPROVED FOR LIVE TRADING

Successful workflow execution does not imply profitability.

V13 is a diagnostic experiment, not a trading strategy.
