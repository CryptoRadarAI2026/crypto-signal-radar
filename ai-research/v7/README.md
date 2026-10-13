# V7 — Paper Trading Simulator Validation

Status: RESEARCH_ONLY
Trading enabled: false
Holdout evaluated: false

## Goal

Improve the accuracy of historical paper-trading simulation.

## Checks

1. Use actual trade exit times when checking overlapping positions.
2. Enforce a maximum of 2 simultaneous positions.
3. Apply the 30-minute cooldown after actual trade exits.
4. Calculate drawdown using both open and closed positions.
5. Prevent future data leakage.
6. Keep fees, slippage, and stop-first assumptions
