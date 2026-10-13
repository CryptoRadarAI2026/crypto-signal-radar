#!/usr/bin/env python3
"""Crypto Signal Radar research V3: explain V2 rejection, never place orders.

Uses the existing public-only V1 Coinbase downloader and tested V2 signal features.
All model/threshold acceptance uses validation alone; the time-separated holdout
is accessed for *performance* only after the locked primary candidate passes.
Alternative stops/targets/holding periods are exploratory validation-only.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]  # ai-research/
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "v2"))
import research_v2 as v2

PAIRS = v2.PAIRS
FEATURES = v2.FEATURES
PRIMARY_HORIZON = v2.HORIZON
MAX_EXPLORATORY_HORIZON = 60
EMBARGO_SEC = 120 * 60
THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
MIN_SIGNALS = 40
MIN_PROFIT_FACTOR = 1.10
EXPLORATORY_MODEL = "LogisticRegression"
EXPLORATORY_THRESHOLD = 0.45


@dataclass
class Bars:
    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    index: dict[int, int]
    gap_prefix: np.ndarray


def load_bars(path: Path) -> Bars:
    with path.open(encoding="utf-8", newline="") as fh:
        lines = sorted(csv.DictReader(fh), key=lambda r: int(r["timestamp"]))
    ts = np.asarray([int(r["timestamp"]) for r in lines], dtype=np.int64)
    if len(ts) < 2000 or np.any(np.diff(ts) <= 0):
        raise ValueError("Invalid or insufficient candle timestamps")
    arrays = [np.asarray([float(r[field]) for r in lines], dtype=float)
              for field in ("open", "high", "low", "close")]
    o, h, l, c = arrays
    if not (all(np.all(np.isfinite(a)) for a in arrays) and
            np.all(l > 0) and np.all(l <= np.minimum(o, c)) and
            np.all(h >= np.maximum(o, c))):
        raise ValueError("Invalid OHLC data")
    gaps = np.r_[0, np.cumsum(np.diff(ts) != 60)]
    return Bars(ts, o, h, l, c, {int(t): i for i, t in enumerate(ts)}, gaps)


def split_purged(samples: list[v2.Sample]):
    """Two-hour purge covers future alternative labels through 60m incl. entry."""
    samples = sorted(samples, key=lambda s: (s.ts, s.pair))
    dates = sorted({s.ts for s in samples})
    if len(dates) < 100:
        raise ValueError("Too few timestamps")
    a, b = dates[int(len(dates)*0.55)], dates[int(len(dates)*0.80)]
    train = [s for s in samples if s.ts < a-EMBARGO_SEC]
    valid = [s for s in samples if a+EMBARGO_SEC < s.ts < b-EMBARGO_SEC]
    holdout = [s for s in samples if s.ts > b+EMBARGO_SEC]
    if min(map(len, (train, valid, holdout))) < 100:
        raise ValueError("Insufficient purged time splits")
    if len({s.label for s in train}) < 2:
        raise ValueError("One-class training labels; refusing to fit")
    return train, valid, holdout, (a, b)


def candidate_reasons(result: dict) -> list[str]:
    reasons = []
    if result["signals"] == 0:
        reasons.append("NO_SIGNALS")
    if result["signals"] < MIN_SIGNALS:
        reasons.append("INSUFFICIENT_SIGNALS")
    if result["paper_pnl_usdt"] <= 0:
        reasons.append("NON_POSITIVE_NET_PNL")
    # A no-loss positive strategy has mathematically infinite profit factor.
    # The reporting layer uses None instead of infinity for JSON compatibility.
    pf = result["profit_factor"]
    effective_pf = (float("inf") if pf is None and result["signals"] > 0
                    and result["paper_pnl_usdt"] > 0 else (pf or 0))
    if effective_pf < MIN_PROFIT_FACTOR:
        reasons.append("PROFIT_FACTOR_BELOW_MINIMUM")
    return reasons


def probability_diagnostics(y: np.ndarray, p: np.ndarray, *, label: str) -> dict:
    from sklearn.metrics import roc_auc_score, brier_score_loss
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(y) != len(p) or len(y) == 0 or not np.all(np.isfinite(p)):
        raise ValueError("Invalid prediction probabilities")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Predictions must be probabilities in [0,1]")
    guess = p >= 0.50
    tp = int(np.sum(guess & (y == 1)))
    fp = int(np.sum(guess & (y == 0)))
    fn = int(np.sum((~guess) & (y == 1)))
    tn = int(np.sum((~guess) & (y == 0)))
    prevalence = float(y.mean())
    quantiles = np.quantile(p, [0, .10, .25, .50, .75, .90, .95, 1.0])
    bins = []
    for i, (lo, hi) in enumerate(zip((0., .2, .4, .6, .8), (.2, .4, .6, .8, 1.))):
        mask = (p >= lo) & ((p <= hi) if i == 4 else (p < hi))
        bins.append({"from": lo, "to": hi, "count": int(mask.sum()),
                     "mean_prediction": round(float(p[mask].mean()), 5) if mask.any() else None,
                     "realized_positive_rate": round(float(y[mask].mean()), 5) if mask.any() else None})
    return {
        "dataset": label, "samples": len(y), "label_positive_rate_pct": round(prevalence*100, 3),
        "roc_auc": round(float(roc_auc_score(y, p)), 5) if len(np.unique(y)) > 1 else None,
        "brier": round(float(brier_score_loss(y, p)), 6),
        "constant_base_rate_brier": round(float(np.mean((y - prevalence)**2)), 6),
        "confusion_at_50pct": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision_at_50pct_pct": round(100*tp/(tp+fp), 3) if tp+fp else None,
        "recall_at_50pct_pct": round(100*tp/(tp+fn), 3) if tp+fn else None,
        "probability_quantiles": {str(int(k*100))+"pct": round(float(q), 5) for k,q in
                                 zip((0,.10,.25,.50,.75,.90,.95,1.0), quantiles)},
        "fraction_above_threshold_pct": {str(th): round(float(np.mean(p >= th))*100, 3)
                                           for th in THRESHOLDS},
        "calibration_bins": bins,
    }


def scenario_fill(b: Bars, idx: int, atr: float, *, minutes: int,
                  atr_mult: float, target_rr: float):
    """Conservative stop-first historical simulation; no limit-order promises."""
    if idx < 1 or idx + minutes >= len(b.ts):
        return None
    if b.gap_prefix[idx+minutes] != b.gap_prefix[idx]:
        return None
    entry = float(b.o[idx+1])*(1+v2.SLIPPAGE_PER_SIDE)
    stop_pct = min(.03, max(.003, atr_mult*float(atr)/entry))
    stop = entry*(1-stop_pct)
    target = entry*(1+target_rr*stop_pct)
    raw, reason = float(b.c[idx+minutes]), "TIME"
    for j in range(idx+1, idx+minutes+1):
        if j > idx+1 and b.o[j] <= stop:
            raw, reason = float(b.o[j]), "SL_GAP"; break
        if j > idx+1 and b.o[j] >= target:
            raw, reason = float(b.o[j]), "TP_GAP"; break
        if b.l[j] <= stop:
            raw, reason = stop, "SL"; break
        if b.h[j] >= target:
            raw, reason = target, "TP"; break
    exit_fill = raw*(1-v2.SLIPPAGE_PER_SIDE)
    net = exit_fill*(1-v2.FEE_PER_SIDE)/(entry*(1+v2.FEE_PER_SIDE))-1
    return (float(net), reason, entry, exit_fill)


def scenario_samples(valid: list[v2.Sample], market_bars: dict[str, Bars], *,
                     minutes: int, atr_mult: float, target_rr: float):
    output = []
    kept_indices = []
    for k, s in enumerate(valid):
        bars = market_bars[s.pair]
        idx = bars.index.get(s.ts)
        if idx is None:
            continue
        atr = s.x[5]*float(bars.c[idx])  # V2 ATR percent: past-only at signal close
        filled = scenario_fill(bars, idx, atr, minutes=minutes,
                               atr_mult=atr_mult, target_rr=target_rr)
        if filled is None:
            continue
        net, why, entry, exited = filled
        output.append(replace(s, label=int(net > 0), net_return=net,
                              exit_reason=why, entry_price=entry, exit_price=exited))
        kept_indices.append(k)
    return output, np.asarray(kept_indices, dtype=int)


def choose_models():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    return [
        ("LogisticRegression", make_pipeline(RobustScaler(), LogisticRegression(C=.08, max_iter=700))),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=6,
                                                min_samples_leaf=30, n_jobs=2, random_state=42)),
        ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=80,
               max_leaf_nodes=12, learning_rate=.04, l2_regularization=3, random_state=42)),
    ]


def evaluate(samples: list[v2.Sample], manifest: list[dict], *, models=None,
             market_bars: dict[str, Bars] | None = None, enable_exploration=True) -> dict:
    train, valid, holdout, (a,b) = split_purged(samples)
    xtr, ytr = v2._x_y(train)
    xva, yva = v2._x_y(valid)
    candidates, diag, trained = [], [], []
    best = None
    for name, model in (models if models is not None else choose_models()):
        model.fit(xtr, ytr)
        pv = np.asarray(model.predict_proba(xva)[:, 1], dtype=float)
        diag.append({"model": name, "validation": probability_diagnostics(yva, pv, label="validation")})
        for threshold in THRESHOLDS:
            performance = v2.metrics(v2.select_trades(valid, pv, threshold))
            problems = candidate_reasons(performance)
            candidates.append({"model": name, "threshold": threshold,
                               "signals": performance["signals"],
                               "paper_pnl_usdt": performance["paper_pnl_usdt"],
                               "win_rate_pct": performance["win_rate_pct"],
                               "profit_factor": performance["profit_factor"],
                               "rejection_reasons": problems})
            if not problems:
                score = performance["paper_pnl_usdt"]*min(1.0, performance["signals"]/80)
                if best is None or score > best[0]:
                    best = (score, name, model, threshold, performance)
        trained.append((name, model, pv))
        print(f"Validation diagnostic {name}: min={float(pv.min()):.4f} median={float(np.median(pv)):.4f} max={float(pv.max()):.4f}",flush=True)

    report = {
        "schema": "crypto-radar-ai-research-v3", "research_only": True,
        "trading_enabled": False, "model_exported": False,
        "status": "REJECTED_ON_VALIDATION", "markets": list(PAIRS), "features": list(FEATURES),
        "data_source": "Coinbase Exchange public 1-minute OHLCV",
        "rows": {"train": len(train), "validation": len(valid), "holdout": len(holdout)},
        "split_timestamps_utc": {"train_validation_boundary": a,
                                 "validation_holdout_boundary": b, "embargo_seconds": EMBARGO_SEC},
        "train_positive_label_rate_pct": round(100*float(ytr.mean()), 3),
        "validation_positive_label_rate_pct": round(100*float(yva.mean()), 3),
        "primary_strategy": {"horizon_minutes": 15, "atr_multiplier": 1.6,
                             "take_profit_to_stop_ratio": 1.8,
                             "thresholds_tested": list(THRESHOLDS),
                             "validation_min_signals": MIN_SIGNALS,
                             "validation_min_profit_factor": MIN_PROFIT_FACTOR,
                             "validation_net_profit_must_be_positive": True},
        "transaction_assumptions": {"position_side": "LONG_SPOT_ONLY",
             "order_usdt": v2.ORDER_USDT, "initial_equity_usdt": v2.INITIAL_EQUITY,
             "fee_per_side": v2.FEE_PER_SIDE, "slippage_per_side": v2.SLIPPAGE_PER_SIDE,
             "maximum_simultaneous_trades": v2.MAX_SIMULTANEOUS,
             "pair_cooldown_minutes": v2.PAIR_COOLDOWN_SEC//60,
             "stop_loss_range_pct": [0.3, 3.0],
             "next_candle_open_entry": True, "double_hit_stop_first": True},
        "source_manifest": manifest,
        "model_probability_diagnostics": diag,
        "validation_candidates": candidates,
        "holdout_evaluated": False,
        "exploratory_sensitivity": [],
        "limitations": [
           "No live exchange integration, orders, API keys, futures, shorts or leverage.",
           "Only 3 USD spot markets, not global top-100.",
           "Alternative horizons/stops are exploratory on validation only; never used to pass promotion gate.",
           "Historical candle path within a minute is unknown: stop-first and imperfect gap estimates.",
           "Fees and slippage are assumptions; actual fills, outages and liquidity may differ.",
           "Fixed notional accounting ignores live portfolio margin/mark-to-market; drawdown is closed-trade approximation.",
           "Validation search over many candidates may overfit; one holdout never proves future profitability.",
        ]
    }
    # Only predetermined baseline at 15m can qualify for the holdout.
    if best is not None:
        _, name, model, threshold, performance = best
        xte, yte = v2._x_y(holdout)
        pred = model.predict_proba(xte)[:,1]
        report.update(status="RESEARCH_ONLY_NOT_APPROVED",
                      selected_model=name, selected_threshold=threshold,
                      validation=performance,
                      holdout=v2.metrics(v2.select_trades(holdout, pred, threshold)),
                      holdout_probability_diagnostics=probability_diagnostics(yte,pred,label="holdout"),
                      holdout_evaluated=True)
    # No leakage: sensitivity analysis has NO role in model/threshold selection.
    if enable_exploration and market_bars:
        explorer = next(((n, pv) for n,_,pv in trained if n == EXPLORATORY_MODEL), None)
        if explorer is not None:
            _, pval = explorer
            for minutes in (15, 30, 60):
                for atr_mult in (1.2, 1.6):
                    for rr in (1.5, 1.8):
                        scenario, indices = scenario_samples(valid, market_bars,
                                minutes=minutes, atr_mult=atr_mult, target_rr=rr)
                        eligible = [(s, float(p)) for s,p in zip(scenario, pval[indices])
                                    if p >= EXPLORATORY_THRESHOLD]
                        chosen = enforce_holding_capacity(eligible, minutes)
                        performance = v2.metrics(chosen)
                        report["exploratory_sensitivity"].append({
                            "model": EXPLORATORY_MODEL, "threshold": EXPLORATORY_THRESHOLD,
                            "holding_minutes": minutes, "atr_multiplier": atr_mult,
                            "tp_to_sl_ratio": rr, "eligible_candles": len(scenario),
                            "signals": performance["signals"],
                            "paper_pnl_usdt": performance["paper_pnl_usdt"],
                            "profit_factor": performance["profit_factor"],
                            "win_rate_pct": performance["win_rate_pct"],
                            "promotion_eligible": False,
                        })
    return report


def enforce_holding_capacity(chosen: list[tuple[v2.Sample, float]], minutes: int):
    """Conservative fixed-horizon occupancy, same pair cooldown, max 2 simultaneous."""
    active: list[int] = []
    last_pair: dict[str,int] = {}
    retained = []
    for s,p in sorted(chosen, key=lambda t:(t[0].ts, -t[1], t[0].pair)):
        active = [x for x in active if x > s.ts]
        if len(active) >= v2.MAX_SIMULTANEOUS:
            continue
        if s.pair in last_pair and s.ts-last_pair[s.pair] < v2.PAIR_COOLDOWN_SEC:
            continue
        retained.append((s,p))
        active.append(s.ts + (minutes+1)*60)
        last_pair[s.pair] = s.ts
    return retained


def dashboard(report: dict, outfile: Path):
    """Offline static Persian HTML with escaped labels, no scripts/external requests."""
    def esc(x): return html.escape(str(x))
    def num(x, digits=2):
        return "—" if x is None else f"{x:,.{digits}f}" if isinstance(x,(int,float)) else esc(x)
    rows = []
    for row in report.get("validation_candidates", []):
        why = ", ".join(row.get("rejection_reasons", [])) or "PASS ON VALIDATION"
        rows.append(f"<tr><td>{esc(row['model'])}</td><td>{num(row['threshold']*100,0)}٪</td>"
          f"<td>{num(row['signals'],0)}</td><td>{num(row['paper_pnl_usdt'],3)}</td>"
          f"<td>{num(row['profit_factor'],3)}</td><td>{esc(why)}</td></tr>")
    diag_rows = []
    for item in report.get("model_probability_diagnostics", []):
        d = item["validation"]
        q = d["probability_quantiles"]
        diag_rows.append(f"<tr><td>{esc(item['model'])}</td><td>{num(d['label_positive_rate_pct'])}٪</td>"
          f"<td>{num(q['50pct'],4)}</td><td>{num(q['95pct'],4)}</td>"
          f"<td>{num(d['roc_auc'],4)}</td><td>{num(d['brier'],5)}</td>"
          f"<td>{num(d['precision_at_50pct_pct'])}٪</td><td>{num(d['recall_at_50pct_pct'])}٪</td></tr>")
    scenario_rows = []
    for r in report.get('exploratory_sensitivity', []):
        scenario_rows.append(f"<tr><td>{r['holding_minutes']}</td><td>{r['atr_multiplier']}</td>"
          f"<td>{r['tp_to_sl_ratio']}</td><td>{num(r['signals'],0)}</td>"
          f"<td>{num(r['paper_pnl_usdt'],3)}</td><td>{num(r['profit_factor'],3)}</td></tr>")
    hold = report.get('holdout') or {}
    state = esc(report['status'])
    status_reason = "هیچ مدلِ واجد شرایط روی اعتبارسنجی پیدا نشد. داده آزمون نهایی برای تصمیم‌گیری استفاده نشده است." if not report.get('holdout_evaluated') else "مدل از شرط اعتبارسنجی عبور کرده، اما همچنان فقط پژوهشی و برای معامله واقعی تأییدنشده است."
    html_text = f'''<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Crypto Signal Radar AI — Research V3</title><style>
:root{{font-family:Tahoma,system-ui,sans-serif;color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;color:#e9f2ff;background:#0a1220;line-height:1.85}}main{{max-width:1150px;margin:auto;padding:24px 13px 65px}}.eyebrow{{color:#5ae0c5}}h1{{font-size:clamp(26px,6vw,42px)}}h2{{font-size:21px;margin:0 0 15px}}.panel{{background:#121e30;border:1px solid #30435d;border-radius:17px;padding:17px;margin:18px 0}}.warn{{border-inline-start:4px solid #ffd180;padding:12px;background:#352d24}}.muted{{color:#a9bdd1}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.metric{{padding:12px;border:1px solid #3c5069;border-radius:11px;background:#17273e}}strong.num{{display:block;font-size:25px;direction:ltr;text-align:right}}.scroll{{overflow:auto}}table{{border-collapse:collapse;min-width:650px;width:100%;direction:rtl;font-size:13px}}th,td{{border-bottom:1px solid #344961;padding:10px;text-align:right;white-space:nowrap}}th{{color:#b1ddeb}}footer{{font-size:12px;color:#91a5be}}</style></head><body><main><p class="eyebrow">Crypto Signal Radar AI · V3 · PAPER / RESEARCH ONLY</p><h1>گزارش تشخیصی نسخه سوم</h1><p class="muted">گزارش مستقل از اینترنت، بدون سفارش صرافی و بدون اتصال به حساب</p><div class="panel warn"><b>وضعیت: {state}</b><p>{status_reason}</p></div><section class="panel"><h2>خلاصه اعتبارسنجی</h2><div class="grid"><div class="metric">نمونه آموزش<strong class="num">{num(report['rows']['train'],0)}</strong></div><div class="metric">نمونه اعتبارسنجی<strong class="num">{num(report['rows']['validation'],0)}</strong></div><div class="metric">نرخ برچسب مثبت اعتبارسنجی<strong class="num">{num(report['validation_positive_label_rate_pct'])}٪</strong></div><div class="metric">سود/زیان آزمون نهایی<strong class="num">{num(hold.get('paper_pnl_usdt'),3)} USDT</strong></div></div></section><section class="panel"><h2>چرا مدل‌ها سیگنال نمی‌دهند؟</h2><p class="muted">برچسب مثبت: معامله خریدی که پس از هزینه‌های فرضی سود کرده است. Precision و Recall فقط معیار طبقه‌بندی‌اند، نه نرخ سود معاملاتی.</p><div class="scroll"><table><thead><tr><th>مدل</th><th>مثبت‌ها</th><th>میانه احتمال</th><th>صدک ۹۵ احتمال</th><th>AUC</th><th>Brier</th><th>Precision در ۵۰٪</th><th>Recall در ۵۰٪</th></tr></thead><tbody>{''.join(diag_rows)}</tbody></table></div></section><section class="panel"><h2>نتیجه همه آستانه‌های از قبل تعیین‌شده</h2><p class="muted">قواعد قبول: دست‌کم ۴۰ معامله، سود خالص مثبت، ضریب سود به زیان حداقل ۱٫۱۰. ریسک به‌صورت خودکار افزایش نمی‌یابد.</p><div class="scroll"><table><thead><tr><th>مدل</th><th>آستانه</th><th>تعداد معامله</th><th>سود/زیان تتر</th><th>ضریب سود</th><th>دلیل عدم پذیرش</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></section><section class="panel"><h2>آزمایش حساسیت — فقط پژوهشی</h2><p class="muted">فقط رگرسیون لجستیک با آستانه ثابت ۴۵٪. سناریوهای مدت نگهداری، فاصله حد ضرر و حد سود روی داده اعتبارسنجی مقایسه شده‌اند؛ هیچ‌کدام برای تأیید مدل استفاده نشده‌اند.</p><div class="scroll"><table><thead><tr><th>دقیقه</th><th>ضریب ATR</th><th>ضریب حد سود</th><th>معامله</th><th>سود/زیان</th><th>ضریب سود</th></tr></thead><tbody>{''.join(scenario_rows) or '<tr><td colspan="6">آزمایش حساسیت اجرا نشده است.</td></tr>'}</tbody></table></div></section><section class="panel"><h2>محدودیت‌ها</h2><ul>{''.join('<li>'+esc(t)+'</li>' for t in report['limitations'])}</ul></section><footer>نسخه ۳ آزمایشی است، کلید صرافی نیاز ندارد و هیچ مدلی را برای معامله واقعی تأیید یا صادر نمی‌کند.</footer></main></body></html>'''
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(html_text,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description="V3 safety-first diagnostics; public history only")
    parser.add_argument('--days',type=int,choices=(90,120),default=90)
    parser.add_argument('--data-dir',type=Path,default=Path('data/raw'))
    parser.add_argument('--out-dir',type=Path,default=Path('reports/v3'))
    args=parser.parse_args()
    from train_public_data import download
    today = int(time.time())//86400*86400
    start = today-args.days*86400
    manifest=[];samples=[];bars={}
    for pair in PAIRS:
        path=args.data_dir/f"COINBASE_{pair.replace('-','')}_{args.days}D_1M.csv"
        info=download(pair,start,today,path)
        manifest.append(info)
        batch=v2.samples_from_csv(path,pair)
        samples.extend(batch)
        bars[pair]=load_bars(path)
        print(f"{pair}: {info['bars']} real public candles, {len(batch)} valid 15m signals",flush=True)
    report=evaluate(samples,manifest,market_bars=bars)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.out_dir/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    dashboard(report,args.out_dir/'dashboard.html')
    print(json.dumps({'status':report['status'],'rows':report['rows'],
                      'holdout_evaluated':report['holdout_evaluated'],
                      'selected_model':report.get('selected_model')},ensure_ascii=False),flush=True)
    # A rejected model is an expected safe result, not a runtime failure.


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'RESEARCH_V3_FAILED_SAFELY: {exc!r}',file=sys.stderr)
        raise SystemExit(2)
