# Crypto Signal Radar AI — Research V15

## Purpose

V15 investigates why historical paper trades lose money.

The goal is to separate trading costs, exit reasons,
and underlying price movements.

V15 is diagnostic research only.

## Previous Findings

V11:
- No model passed validation approval requirements.
- No profitable strategy was approved.

V12:
- RandomForest and HistGradientBoosting produced
  no predictions above 0.50.

V13:
- RandomForest had the best Brier Score and Log Loss.
- HistGradientBoosting had the lowest 10-bin ECE.
- Good calibration did not prove profitability.

V14:
- Most probability ranges produced negative paper PnL.
- RandomForest and HistGradientBoosting were unprofitable
  in every probability range containing trades.
- LogisticRegression showed a positive result in the
  0.90 to 1.00 range, but only four selected paper trades.
- That small result is not sufficient evidence
  of profitability.

## V15 Research Questions

1. How many paper trades exited by stop-loss?
2. How many exited by take-profit?
3. How many exited because of the time limit?
4. How much did entry and exit fees cost?
5. How much did simulated slippage cost?
6. What was the gross PnL before trading costs?
7. What was the net PnL after trading costs?
8. Which exit reasons contributed most to losses?

## Methodology

- Reuse V11 historical market data.
- Reuse V11 feature definitions.
- Reuse V11 model configurations.
- Reuse V11 chronological train/validation split.
- Apply the same paper-trading rules.
- Analyze validation data only.
- Do not evaluate holdout data.

## Paper Trading Assumptions

- LONG spot positions only.
- Hypothetical order size: 100 USDT.
- Entry at the next candle open.
- Trading fee: 0.1% per side.
- Simulated slippage: 0.05% per side.
- Maximum two simultaneous positions.
- Maximum holding period: 15 minutes.
- Pair cooldown: 30 minutes after exit.
- Stop-loss takes priority when stop-loss and
  take-profit are reached in the same candle.

## Exit Categories

- STOP_LOSS
- TAKE_PROFIT
- TIME_EXIT

Actual category names must be checked against
the V11 implementation before writing V15 code.

## Required Report

The V15 report should include:

- Number of selected paper trades.
- Number of trades by exit reason.
- Gross hypothetical PnL.
- Estimated trading fees.
- Estimated slippage costs.
- Net hypothetical PnL.
- Win rate.
- Profit factor.
- Limitations and assumptions.

Cost decomposition must be consistent with the
actual V11 execution and return calculations.

Do not assume that fees and slippage can be
reconstructed exactly without checking V11.

## Research Limitations

The validation period has already been examined
in V11 through V14.

V15 is exploratory and cannot independently
confirm profitability.

Historical paper fills are approximations.

No trading threshold or model may be approved
based on these diagnostics.

## Safety Rules

- Historical research only.
- No real trading.
- No exchange API credentials.
- No leverage or futures.
- No changes to V11 through V14.
- No holdout predictions.
- No holdout scoring.
- No holdout-based optimization.

## Approval Status

DIAGNOSTIC_ONLY_NOT_APPROVED

trading_enabled: false
holdout_evaluated: false

No model is approved for live trading.
