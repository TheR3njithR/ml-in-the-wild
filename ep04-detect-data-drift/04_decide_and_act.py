#!/usr/bin/env python3
"""Morning report for a given date (default: the hook day, 20 April 2026).

Reads the already-scored demand model and the drift monitor. Prints a plain-English
green / amber / red recommendation before the auto-purchase order goes out at 07:00.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import drift_tools as D
from model_common import HOOK_DAY, LABEL_LAG

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def main(day=None):
    day = pd.Timestamp(day or HOOK_DAY)
    scored = pd.read_csv(OUT / "02_scored_daily.csv", parse_dates=["date"])
    mon = pd.read_csv(OUT / "03_drift_daily.csv", parse_dates=["date"])
    base = pd.read_csv(OUT / "01_baseline_daily.csv", parse_dates=["date"])
    row = mon.loc[mon["date"] == day]
    sc = scored.loc[scored["date"] == day]
    if row.empty or sc.empty:
        raise SystemExit(f"no report for {day.date()}")
    r, s = row.iloc[0], sc.iloc[0]
    b = base.loc[base["date"] == day]
    delayed = float(b.iloc[0]["rolling_14_mape"]) if len(b) else float("nan")
    delayed_alert = int(b.iloc[0]["alert"]) if len(b) else 0

    action = {
        "GREEN": "Release the auto-purchase order. Keep watching tomorrow morning.",
        "AMBER": "Do not raise a hold yet. Open the feature view, tell planning a look is needed, and watch tomorrow.",
        "RED": "HOLD the auto-purchase order. Do not send today's predicted units to the warehouse. Investigate the features that moved, and fall back to last week's fulfilled volume until a human signs off.",
    }[r["status"]]

    feats = list(D.MONITOR_FEATURES)
    lines = [
        "QUICKBASKET  ·  BEVERAGES DEMAND  ·  MORNING DRIFT REPORT  ·  SYNTHETIC",
        f"Report date: {day.date()}   (labels are only in through {(day - pd.Timedelta(days=LABEL_LAG)).date()})",
        f"Model forecast for today: {int(s['pred']):,} units.",
        f"STATUS: {r['status']}",
        f"Domain-classifier AUC (14-day window vs training): {r['domain_auc']:.2f}",
        f"Top PSI: {r['max_psi']:.2f} on {r['top_feature']}   features above their 95th null: {int(r['n_red_features'])}",
        f"Delayed 14-day MAPE (labels already in): {delayed*100:.1f}%   baseline alert={'YES' if delayed_alert else 'NO'}.",
        "",
        "Feature PSI and K S, last 14 days vs the training distribution:",
    ]
    for f in feats:
        lines.append(f"  {f:20s}  PSI {r[f'psi_{f}']:.2f}  K S {r[f'ks_{f}']:.2f}")
    # If the day already has actuals (for the hook, we show what later arrived, clearly labelled).
    lines += [
        "",
        f"ACTION: {action}",
        "",
        "Note: today's actual units are not known this morning. They arrive in 7 days.",
        f"(For the case study, the actual that later arrived was {int(s['units']):,} units, "
        f"so the order would have been {int(s['err']):+} units off.)",
    ]
    text = "\n".join(lines) + "\n"
    out = OUT / f"04_report_{day.date()}.txt"
    out.write_text(text)
    (OUT / "04_decide.txt").write_text(text)  # default alias for the hook
    print(text, end="")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
