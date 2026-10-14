# Crypto Signal Radar AI — Research V14

## Purpose

V14 investigates the historical economic value of
validation predictions from three models:

1. LogisticRegression
2. RandomForest
3. HistGradientBoosting

The goal is to understand whether different predicted
probability ranges are associated with positive or
negative hypothetical trading returns.

## Previous findings

V11:
- No model passed validation approval requirements.
- LogisticRegression produced losing paper-trading results.
- RandomForest and HistGradientBoosting produced no signals
  at the tested thresholds of 0.50 to 0.75.

V12:
- RandomForest maximum predicted probability: 0.440255.
- HistGradientBoosting maximum predicted probability: 0.452557.
- LogisticRegression maximum predicted probability: 0.981916.

V13:
- RandomForest had the lowest Brier Score and Log Loss.
- HistGradientBoosting had the lowest 10-bin ECE.
- Good probability calibration did not establish profitability.

## V14 Research Plan

1. Reuse V11 historical feature and paper-fill definitions.
2. Train the same three models on training data only.
3. Generate predictions for validation data only.
4. Group predictions into fixed probability ranges.
5. Examine hypothetical net returns after fees and slippage.
6. Separately examine trade selection under V11 rules.
7. Report sample counts, paper PnL and risk limitations.
8. Keep all findings exploratory.

## Probability Ranges

Use fixed, non-overlapping ranges:

- 0.00 to below 0.10
- 0.10 to below 0.20
- 0.20 to below 0.30
- 0.30 to below 0.40
- 0.40 to below 0.50
- 0.50 to below 0.60
- 0.60 to below 0.70
- 0.70 to below 0.80
- 0.80 to below 0.90
- 0.90 to 1.00 inclusive

## Paper Trading Assumptions

Follow V11:

- LONG spot only.
- Fixed hypothetical order size: 100 USDT.
- Entry at next candle open.
- Trading fee: 0.1% per side.
- Slippage: 0.05% per side.
- Maximum two simultaneous positions.
- Maximum holding period: 15 minutes.
- Pair cooldown: 30 minutes after exit.
- Stop-loss first when stop and target touch
  during the same candle.

## Important Methodology

Probability-bin statistics and executable paper-trade
statistics must be reported separately.

Individual samples can overlap in time.

Trade capacity and cooldown restrictions must be applied
chronologically using V11 selection rules.

The validation period has already been examined in
V11, V12 and V13.

Therefore, V14 results are exploratory and must not be
used to approve a model or select a final trading threshold.

## Safety Rules

- Historical research only.
- No real orders.
- No exchange credentials.
- No live trading.
- No futures or leverage.
- No changes to V2 or V11-V13.
- No holdout predictions.
- No holdout scoring.
- No holdout-based model selection.

## Approval Status

DIAGNOSTIC_ONLY_NOT_APPROVED

No model is approved for live trading.

Successful workflow execution does not establish
profitability or future performance.
