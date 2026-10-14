"""V24.7 conservative research engine. Requires unchanged V24.5 engine."""
import importlib.util
import sys
from pathlib import Path

OLD = Path(__file__).resolve().parents[1] / 'v245' / 'backtest_engine.py'
spec = importlib.util.spec_from_file_location('v247_v245_engine', OLD)
if spec is None or spec.loader is None:
    raise RuntimeError('V24.5 engine not found')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

Candle = base.Candle
ResearchSignal = base.ResearchSignal
load_candles_csv = base.load_candles_csv
save_report = base.save_report
verify_safety = base.verify_safety


class BTCBacktestEngine(base.BTCBacktestEngine):
    """Conservative single-position, unlevered proxy; retains V24.5 fill rules."""

    def __init__(self, config):
        super().__init__(config)
        if self.max_leverage != 1 or self.max_gross_ratio > 1:
            raise ValueError('V24.7 requires unlevered gross exposure <= 1')
        if self.position_count_limit != 1:
            raise ValueError('V24.7 requires position limit 1')

    def equity(self, price):
        # Mark-to-liquidation estimate including exit fee and adverse base slippage.
        total = self.cash_equity
        for position in self.positions:
            fill = self.exit_fill(price, position.direction)
            total += self.position_pnl(position, fill)
            total -= fill * position.quantity_btc * self.fee_rate
        return total

    def open_hypothetical_position(self, signal, candle):
        if signal.direction == 'WAIT':
            self.wait_signals += 1
            return
        if signal.direction == 'SHORT':
            self.reject('SHORT_NOT_MODELED_WITH_SPOT_PROXY')
            return
        if self.positions:
            self.reject('DUPLICATE_EXPOSURE')
            return
        super().open_hypothetical_position(signal, candle)

    def run(self, candles, signals):
        report = super().run(candles, signals)
        # V24.5 emits two end-of-data points with the same timestamp.
        if len(self.equity_curve) >= 2 and self.equity_curve[-1]['timestamp'] == self.equity_curve[-2]['timestamp']:
            self.equity_curve.pop(-2)
        report = self.report()
        report['version'] = 'V24.7'
        report['engine_notes'] = [
            'Unlevered spot-price proxy; SHORT entries rejected.',
            'Open equity includes estimated base exit fee/slippage.',
            'Intrabar sequence unknown; stop-first.',
            'No trailing stop, ATR stop, partial exit, funding or liquidation model.',
            'Daily lock blocks new entries, does not liquidate positions.'
        ]
        return report
