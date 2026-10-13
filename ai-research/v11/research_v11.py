#!/usr/bin/env python3
"""Risk-aware research v11: public historical candles, conservative paper fills, no execution.

This module NEVER transmits orders, stores exchange credentials or approves live trading.
It consumes historical CSV from the existing v1 Coinbase downloader, builds past-only
features, uses chronological splits, selects thresholds on validation only, and
reports one untouched holdout. A successful process exit is NOT model approval.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAIRS = ("BTC-USD", "ETH-USD", "SOL-USD")
HORIZON = 15  # maximum time in market, 15 one-minute bars
LOOKBACK = 240
EMBARGO_SEC = 60 * 60  # greater than maximum label horizon
ORDER_USDT = 100.0
INITIAL_EQUITY = 10000.0
FEE_PER_SIDE = 0.001
SLIPPAGE_PER_SIDE = 0.0005
MAX_SIMULTANEOUS = 2
PAIR_COOLDOWN_SEC = 30 * 60
FEATURES = (
    "return_1m", "return_5m", "return_15m", "return_60m",
    "rsi_14", "atr_14_pct", "macd_hist_pct", "adx_14",
    "bollinger_z20", "bollinger_width20", "vwap_60_gap",
    "volume_z20", "fib_range_120", "range_pct", "trend_60_240",
    "realized_volatility_60",
)


@dataclass(frozen=True)
class Sample:
    ts: int
    pair: str
    x: tuple[float, ...]
    label: int
    net_return: float
    exit_reason: str
    entry_price: float
    exit_price: float
    entry_ts: int
    exit_ts: int


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    a = 2.0 / (period + 1)
    out = np.empty(len(values), dtype=float)
    out[0] = values[0]
    for j in range(1, len(values)):
        out[j] = a * values[j] + (1 - a) * out[j - 1]
    return out


def _adx_at(h: np.ndarray, l: np.ndarray, c: np.ndarray, i: int) -> float:
    # Rolling directional movement / true range. Only uses candles <= i.
    values = []
    for j in range(i - 13, i + 1):
        hp, lp = h[j - 14:j + 1], l[j - 14:j + 1]
        ups = np.diff(hp)
        downs = -np.diff(lp)
        plus = np.where((ups > downs) & (ups > 0), ups, 0).sum()
        minus = np.where((downs > ups) & (downs > 0), downs, 0).sum()
        prior = c[j - 14:j]
        tr = np.maximum(hp[1:] - lp[1:], np.maximum(abs(hp[1:] - prior), abs(lp[1:] - prior))).sum()
        plus_di, minus_di = plus / max(tr, 1e-12), minus / max(tr, 1e-12)
        values.append(100 * abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-12))
    return float(np.mean(values))


def features_at(h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray, i: int) -> tuple[tuple[float, ...], float]:
    """Close of signal candle i is the latest permitted information."""
    p = float(c[i]); eps = 1e-12
    changes = np.diff(c[i-14:i+1])
    up, down = float(np.maximum(changes, 0).mean()), float(np.maximum(-changes, 0).mean())
    rsi = 50. if up == down == 0 else 100 * up / max(up + down, eps)
    tr = np.maximum(h[i-13:i+1] - l[i-13:i+1],
                    np.maximum(abs(h[i-13:i+1] - c[i-14:i]), abs(l[i-13:i+1] - c[i-14:i])))
    atr = float(tr.mean())
    recent = c[i-80:i+1]
    macd = _ema(recent, 12) - _ema(recent, 26)
    hist = macd - _ema(macd, 9)
    m20, std20 = float(c[i-19:i+1].mean()), float(c[i-19:i+1].std())
    vol20, volstd = float(v[i-19:i+1].mean()), float(v[i-19:i+1].std())
    lo120, hi120 = float(l[i-119:i+1].min()), float(h[i-119:i+1].max())
    v60 = v[i-59:i+1]
    vw = float(np.dot(c[i-59:i+1], v60) / max(float(v60.sum()), eps))
    log_returns = np.diff(np.log(c[i-60:i+1]))
    feats = (
        p / c[i-1] - 1, p / c[i-5] - 1, p / c[i-15] - 1, p / c[i-60] - 1,
        rsi / 100, atr / p, float(hist[-1] / p), _adx_at(h, l, c, i) / 100,
        (p - m20) / max(2 * std20, eps), 4 * std20 / p,
        p / max(vw, eps) - 1, (float(v[i]) - vol20) / max(volstd, eps),
        (p - lo120) / max(hi120 - lo120, eps), (h[i] - l[i]) / p,
        c[i-60:i+1].mean() / c[i-239:i+1].mean() - 1,
        float(log_returns.std()),
    )
    return tuple(map(float, feats)), atr


def simulate_long(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                  signal_index: int, atr: float, horizon: int = HORIZON) -> tuple[float, str, float, float]:
    """Enter NEXT candle OPEN, conservatively stop first if TP and SL both touched.

    Uses long-only hypothetical fills, fees and slippage on both legs.
    Stops/gaps are approximations, not exchange execution or liquidation mechanics.
    """
    i = signal_index
    entry = float(o[i+1]) * (1 + SLIPPAGE_PER_SIDE)
    stop_pct = min(0.03, max(0.003, 1.6 * float(atr) / entry))
    sl = entry * (1 - stop_pct)
    tp = entry * (1 + 1.8 * stop_pct)
    price, reason, exit_index = float(c[i+horizon]), "TIME", i+horizon
    for j in range(i+1, i+horizon+1):
        if j > i+1 and o[j] <= sl:
            price, reason, exit_index = float(o[j]), "SL_GAP", j
            break
        if j > i+1 and o[j] >= tp:
            price, reason, exit_index = float(o[j]), "TP_GAP", j
            break
        # Same-minute double touch is treated as loss (not assumed profitable).
        if l[j] <= sl:
            price, reason, exit_index = sl, "SL", j
            break
        if h[j] >= tp:
            price, reason, exit_index = tp, "TP", j
            break
    exit_fill = price * (1 - SLIPPAGE_PER_SIDE)
    # Both entry and exit trading fees, charged on hypothetical notional.
    net = exit_fill * (1 - FEE_PER_SIDE) / (entry * (1 + FEE_PER_SIDE)) - 1
    return float(net), reason, entry, exit_fill, exit_index


def samples_from_csv(path: Path, pair: str) -> list[Sample]:
    with path.open(encoding="utf-8", newline="") as file:
        data = list(csv.DictReader(file))
    data.sort(key=lambda row: int(row["timestamp"]))
    ts = np.array([int(r["timestamp"]) for r in data], dtype=np.int64)
    o = np.array([float(r["open"]) for r in data])
    h = np.array([float(r["high"]) for r in data])
    l = np.array([float(r["low"]) for r in data])
    c = np.array([float(r["close"]) for r in data])
    v = np.array([float(r["volume"]) for r in data])
    if len(ts) < 2000 or np.any(np.diff(ts) <= 0):
        raise ValueError(f"Insufficient, duplicate or unordered history: {pair}")
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(h)) and
            np.all(np.isfinite(l)) and np.all(np.isfinite(c)) and np.all(np.isfinite(v))):
        raise ValueError(f"Non-finite market data: {pair}")
    if not (np.all(l > 0) and np.all(l <= np.minimum(o, c)) and
            np.all(h >= np.maximum(o, c)) and np.all(v >= 0)):
        raise ValueError(f"Invalid OHLCV data: {pair}")
    result = []
    # No interpolation. Only windows with exactly contiguous minute candles.
    for i in range(LOOKBACK, len(ts)-HORIZON, HORIZON):
        if ts[i+HORIZON] - ts[i-LOOKBACK] != (HORIZON+LOOKBACK)*60:
            continue
        x, atr = features_at(h, l, c, v, i)
        if not all(map(math.isfinite, x)):
            continue
        ret, reason, entry, exit_price, exit_index = simulate_long(o, h, l, c, i, atr)
        result.append(Sample(int(ts[i]), pair, x, int(ret > 0), ret, reason, entry, exit_price, int(ts[i+1]), int(ts[exit_index])))
    if len(result) < 100:
        raise ValueError(f"Too few contiguous samples: {pair}")
    return result


def chronological_split(samples: list[Sample]):
    samples = sorted(samples, key=lambda s: (s.ts, s.pair))
    dates = sorted({s.ts for s in samples})
    if len(dates) < 100:
        raise ValueError("Insufficient separate timestamps")
    t1, t2 = dates[int(len(dates)*.55)], dates[int(len(dates)*.80)]
    tr = [s for s in samples if s.ts < t1 - EMBARGO_SEC]
    va = [s for s in samples if t1 + EMBARGO_SEC < s.ts < t2 - EMBARGO_SEC]
    te = [s for s in samples if s.ts > t2 + EMBARGO_SEC]
    if min(map(len, (tr, va, te))) < 100 or len({s.label for s in tr}) != 2:
        raise ValueError("Insufficient train/validation/holdout or one label class")
    return tr, va, te, (t1, t2)


def select_trades(samples: list[Sample], probs: np.ndarray, threshold: float) -> list[tuple[Sample, float]]:
    """Replay entries chronologically; apply exits only once their timestamps arrive.

    A candidate's future exit is never used to decide its own entry. Exit timestamps
    for already accepted positions are consulted only after they have elapsed.
    """
    if len(samples) != len(probs):
        raise ValueError("Samples and probabilities length mismatch")
    eligible = [(s, float(prob)) for s, prob in zip(samples, probs)
                if math.isfinite(float(prob)) and float(prob) >= threshold]
    eligible.sort(key=lambda item: (item[0].entry_ts, -item[1], item[0].pair))
    chosen: list[tuple[Sample, float]] = []
    active: list[Sample] = []
    last_exit: dict[str, int] = {}
    for s, prob in eligible:
        if s.entry_ts <= s.ts or s.exit_ts < s.entry_ts:
            raise ValueError("Invalid entry/exit timestamp")
        still_active = []
        for position in active:
            if position.exit_ts <= s.entry_ts:
                last_exit[position.pair] = max(last_exit.get(position.pair, -1), position.exit_ts)
            else:
                still_active.append(position)
        active = still_active
        if len(active) >= MAX_SIMULTANEOUS:
            continue
        if any(position.pair == s.pair for position in active):
            continue
        if s.pair in last_exit and s.entry_ts < last_exit[s.pair] + PAIR_COOLDOWN_SEC:
            continue
        chosen.append((s, prob))
        active.append(s)
    return chosen


def metrics(chosen: list[tuple[Sample, float]]) -> dict:
    eq, peak, drawdown = INITIAL_EQUITY, INITIAL_EQUITY, 0.
    wins, losses, curve = 0., 0., [[None, round(eq, 2)]]
    per_pair: dict[str, dict] = {}
    reasons: dict[str, int] = {}
    for s, _ in sorted(chosen, key=lambda t: (t[0].exit_ts, t[0].pair)):
        pnl = ORDER_USDT * s.net_return
        eq += pnl
        peak = max(peak, eq)
        drawdown = max(drawdown, (peak - eq) / max(peak, 1e-12))
        wins += max(pnl, 0)
        losses += -min(pnl, 0)
        curve.append([s.exit_ts, round(eq, 5)])
        pp = per_pair.setdefault(s.pair, {"signals": 0, "paper_pnl_usdt": 0.})
        pp["signals"] += 1
        pp["paper_pnl_usdt"] += pnl
        reasons[s.exit_reason] = reasons.get(s.exit_reason, 0) + 1
    n = len(chosen)
    for pp in per_pair.values():
        pp["paper_pnl_usdt"] = round(pp["paper_pnl_usdt"], 3)
    return {
        "signals": n,
        "paper_pnl_usdt": round(eq - INITIAL_EQUITY, 3),
        "mean_net_return_pct": round(100 * sum(s.net_return for s, _ in chosen) / n, 5) if n else None,
        "win_rate_pct": round(100 * sum(s.net_return > 0 for s, _ in chosen) / n, 2) if n else None,
        "profit_factor": round(wins/losses, 4) if losses else None,
        "max_closed_equity_drawdown_pct": round(100 * drawdown, 4),
        "ending_paper_equity_usdt": round(eq, 3),
        "per_pair": per_pair, "exit_reasons": reasons, "equity_curve": curve,
    }


def _x_y(samples: list[Sample]):
    return np.asarray([s.x for s in samples], dtype=float), np.asarray([s.label for s in samples])


def research(samples: list[Sample], manifest: list[dict], *, models=None) -> dict:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss

    tr, va, te, (t1, t2) = chronological_split(samples)
    xtr, ytr = _x_y(tr)
    xva, yva = _x_y(va)
    # Holdout remains sealed: no feature arrays, prediction or scoring on holdout.
    candidates = models if models is not None else [
        ("LogisticRegression", make_pipeline(RobustScaler(), LogisticRegression(C=.08, max_iter=700))),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=30, n_jobs=2, random_state=42)),
        ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=12, learning_rate=.04, l2_regularization=3, random_state=42)),
    ]
    best = None
    checked = []
    for name, model in candidates:
        model.fit(xtr, ytr)
        pv = model.predict_proba(xva)[:, 1]
        for threshold in (.50, .55, .60, .65, .70, .75):
            perf = metrics(select_trades(va, pv, threshold))
            checked.append({"model": name, "threshold": threshold,
                            "signals": perf["signals"], "paper_pnl_usdt": perf["paper_pnl_usdt"],
                            "profit_factor": perf["profit_factor"]})
            if perf["signals"] < 40 or perf["paper_pnl_usdt"] <= 0 or (perf["profit_factor"] or 0) < 1.10:
                continue
            # Selection uses validation only and penalizes trivial low-count candidates.
            score = perf["paper_pnl_usdt"] * min(1.0, perf["signals"]/80)
            if best is None or score > best[0]:
                best = (score, name, model, threshold, perf)
    report = {
        "schema": "crypto-radar-ai-research-v2", "research_only": True,
        "trading_enabled": False, "model_exported": False,
        "status": "REJECTED_ON_VALIDATION", "data_source": "Coinbase Exchange public 1m historical OHLCV",
        "markets": list(PAIRS), "features": list(FEATURES),
        "rows": {"train": len(tr), "validation": len(va), "holdout": len(te)},
        "split_timestamps_utc": {"train_validation_boundary": t1, "validation_holdout_boundary": t2,
                                  "embargo_seconds": EMBARGO_SEC},
        "transaction_assumptions": {"side": "LONG_SPOT_ONLY", "fee_per_side": FEE_PER_SIDE,
          "slippage_per_side": SLIPPAGE_PER_SIDE, "order_usdt": ORDER_USDT,
          "initial_equity_usdt": INITIAL_EQUITY, "max_simultaneous": MAX_SIMULTANEOUS,
          "pair_cooldown_seconds": PAIR_COOLDOWN_SEC, "max_holding_minutes": HORIZON,
          "entry": "next_candle_open", "same_bar_stop_and_take_profit": "STOP_FIRST",
          "stop": "1.6x ATR14, clamped to 0.3%-3%", "take_profit": "1.8x stop_distance"},
        "source_manifest": manifest, "validation_candidates": checked,
        "limitations": ["Research only, never live execution or futures fills",
           "Uses only 1m OHLCV of 3 USD markets, not global top 100 or news/fundamentals/order books",
           "Intrabar candle path unknown; stop-first convention is conservative, gaps approximated",
           "No latency, exchange outage, borrow, margin, futures funding or liquidation model",
           "Equity assumes fixed 100 USDT trades, ignores portfolio margin/mark-to-market", 
           "One chronological holdout is not proof of future profitability"],
    }
    # V11 is a validation-only diagnostic. Holdout is never evaluated,
    # including when validation finds a qualifying candidate.
    report.update(schema="crypto-radar-ai-research-v11",
                  status="VALIDATION_ONLY_NOT_APPROVED",
                  holdout_evaluated=False,
                  holdout=None,
                  validation_best=(dict(model=best[1], threshold=best[3], metrics=best[4])
                                   if best is not None else None),
                  selection_rule="actual_exit_time_and_30m_post_exit_cooldown")
    return report


def make_dashboard(report: dict, destination: Path) -> None:
    """Self-contained responsive Persian report, no network, JS or external assets."""
    hold = report.get("holdout") or {}
    valid = report.get("validation") or {}
    def value(x, suffix="", digits=2):
        if x is None: return "—"
        if isinstance(x, (int, float)):
            return f"{x:,.{digits}f}{suffix}"
        return html.escape(str(x))
    metrics_cards = [
        ("سود/زیان دوره آزمون", value(hold.get("paper_pnl_usdt"), " USDT")),
        ("درصد معاملات سودده", value(hold.get("win_rate_pct"), "٪")),
        ("تعداد سیگنال", value(hold.get("signals"), digits=0)),
        ("ضریب سود به زیان", value(hold.get("profit_factor"), digits=3)),
        ("افت بسته‌شده سرمایه", value(hold.get("max_closed_equity_drawdown_pct"), "٪")),
        ("نتیجه بررسی اعتبارسنجی", value(valid.get("paper_pnl_usdt"), " USDT")),
    ]
    cards = "".join(f'<article class="metric"><div class="muted">{html.escape(label)}</div><strong>{v}</strong></article>' for label, v in metrics_cards)
    rows = "".join(f'<tr><td>{html.escape(k)}</td><td>{value(report["rows"].get(k), digits=0)}</td></tr>' for k in ("train","validation","holdout"))
    pair_rows = "".join(f'<tr><td>{html.escape(k)}</td><td>{v["signals"]}</td><td>{value(v["paper_pnl_usdt"], " USDT")}</td></tr>' for k,v in hold.get("per_pair", {}).items()) or '<tr><td colspan="3">مدل در اعتبارسنجی رد شد یا سیگنالی وجود ندارد.</td></tr>'
    points = hold.get("equity_curve", [])
    chart = '<p class="muted">هنوز منحنی آزمون نهایی وجود ندارد.</p>'
    if len(points) >= 2:
        ys = [float(p[1]) for p in points]
        mn, mx = min(ys), max(ys)
        pad = max(1, (mx - mn) * .1)
        low, high = mn - pad, mx + pad
        coords = " ".join(f"{35+870*i/max(1,len(ys)-1):.1f},{205-165*(y-low)/(high-low):.1f}" for i, y in enumerate(ys))
        chart = f'<svg viewBox="0 0 940 235" aria-label="منحنی سرمایه فرضی" role="img"><line x1="35" x2="905" y1="205" y2="205" stroke="#41516f"/><polyline fill="none" stroke="#31dfbd" stroke-width="3" points="{coords}"/></svg>'
    warnings = "".join(f'<li>{html.escape(t)}</li>' for t in report.get("limitations", []))
    status = html.escape(str(report.get("status", "UNKNOWN")))
    model = html.escape(str(report.get("selected_model", "هیچ مدل واجد شرایطی در اعتبارسنجی یافت نشد")))
    markup = f'''<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Crypto Signal Radar AI · Research V2</title><style>
:root{{color-scheme:dark;font-family:system-ui,"Vazirmatn",Tahoma,sans-serif}}*{{box-sizing:border-box}}body{{background:radial-gradient(circle at 20% 0%,#203556 0%,#090f1d 48%,#070c16 100%);color:#edf4ff;margin:0;line-height:1.9}}main{{max-width:1120px;margin:auto;padding:26px 16px 60px}}.tag{{display:inline-block;border:1px solid #3d6480;background:#13263c;border-radius:24px;padding:4px 14px;color:#a7d9f4}}h1{{font-size:clamp(26px,5vw,46px);line-height:1.3;margin:16px 0}}h2{{font-size:21px;margin:8px 0 16px}}.muted{{color:#aab9d0}}.sub{{font-size:15px}}.panel{{border:1px solid #29405e;border-radius:20px;background:#121e31e8;padding:22px;margin-top:22px;box-shadow:0 18px 55px #02081266}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.metric{{border:1px solid #324864;border-radius:14px;padding:15px;background:#1b2b42}}.metric strong{{display:block;font-size:clamp(19px,3vw,27px);direction:ltr;text-align:right}}.pill{{color:#ffcb83;background:#432b2633;border:1px solid #74533b;border-radius:10px;padding:11px 16px;margin-top:18px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:12px 9px;border-bottom:1px solid #30435e;text-align:right}}svg{{width:100%;height:auto}}li{{margin:6px 0}}footer{{color:#8193b0;font-size:13px;margin-top:30px}}@media(max-width:650px){{.grid{{grid-template-columns:repeat(2,1fr)}}.panel{{padding:15px}}.metric{{padding:11px}}}}</style></head><body><main><span class="tag">Crypto Signal Radar AI / V2 / PAPER ONLY</span><h1>داشبورد پژوهش بازار <span style="color:#5ce0c4">نسخه دوم</span></h1><p class="muted">گزارش تاریخی، ارزیابی زمانی و شبیه‌سازی محافظه‌کارانه معاملات؛ بدون اتصال به حساب صرافی</p><div class="pill"><b>وضعیت: {status}</b><div>این مدل برای معامله واقعی تأیید نشده است. موفقیت اجرای برنامه معادل سودآوری نیست.</div></div><section class="panel"><h2>خلاصه آزمون خارج از نمونه</h2><div class="grid">{cards}</div></section><section class="panel"><h2>منحنی سرمایه فرضی</h2>{chart}<p class="muted sub">مبنای اولیه ۱۰٬۰۰۰ USDT و هر سفارش فرضی ۱۰۰ USDT. هزینه و لغزش لحاظ شده؛ سود آینده تضمین نیست.</p></section><section class="panel"><h2>داده و تقسیم زمانی</h2><p>بازارها: BTC-USD · ETH-USD · SOL-USD | بازه: ۱ دقیقه | مدل انتخاب‌شده: <b>{model}</b></p><table><thead><tr><th>مجموعه</th><th>تعداد نمونه</th></tr></thead><tbody>{rows}</tbody></table></section><section class="panel"><h2>تحلیل بازارهای آزمون</h2><table><thead><tr><th>بازار</th><th>سیگنال</th><th>سود/زیان فرضی</th></tr></thead><tbody>{pair_rows}</tbody></table></section><section class="panel"><h2>محدودیت‌های مهم</h2><ul>{warnings}</ul></section><footer>Generated offline by Crypto Signal Radar AI Research V11 · No execution keys · No orders</footer></main></body></html>'''
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markup, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Research-only v2; no execution")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/v11"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if not 90 <= args.days <= 120:
        parser.error("Only 90 or 120 day research windows are supported")
    end = int(time.time()) // 86400 * 86400
    start = end - args.days * 86400
    from train_public_data import download  # Existing public Coinbase downloader in repository
    manifest, samples = [], []
    for pair in PAIRS:
        path = args.data_dir / f"COINBASE_{pair.replace('-', '')}_{args.days}D_1M.csv"
        meta = download(pair, start, end, path)  # refuses overwriting existing datasets
        manifest.append(meta)
        batch = samples_from_csv(path, pair)
        samples.extend(batch)
        print(f"{pair}: {meta['bars']} candles, {len(batch)} valid samples", flush=True)
    report = research(samples, manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_dashboard(report, args.out_dir / "dashboard.html")
    (args.out_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "holdout": report.get("holdout"),
                      "rows": report["rows"]}, ensure_ascii=False), flush=True)
    # Valid model rejection is a successful SAFETY outcome. No promotion or orders.


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RESEARCH_V11_FAILED_SAFELY: {exc!r}", file=sys.stderr)
        raise SystemExit(2)
