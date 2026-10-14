"""V24.7 causal EMA/RSI baseline signals; indicators use completed candles."""
import importlib.util
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / 'v246' / 'run_historical_backtest.py'
spec = importlib.util.spec_from_file_location('v247_v246_runner', RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError('V24.6 runner not found')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)

ema_series = old.ema_series
rsi_series = old.rsi_series


def generate_signals(candles, engine_module, settings, stop, target, holding):
    closes = [c.close for c in candles]
    fast = ema_series(closes, settings['fast_ema'])
    slow = ema_series(closes, settings['slow_ema'])
    rsi = rsi_series(closes, settings['rsi_period'])
    signals = []
    for i in range(settings['minimum_history'], len(candles) - 1):
        if rsi[i] is None:
            continue
        bullish = fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]
        if bullish and rsi[i] >= settings['long_rsi_min']:
            signals.append(engine_module.ResearchSignal(
                timestamp=candles[i].timestamp, direction='LONG',
                stop_loss_pct=stop, take_profit_pct=target,
                max_holding_minutes=holding, risk_fraction=settings['risk_fraction']))
    return signals
