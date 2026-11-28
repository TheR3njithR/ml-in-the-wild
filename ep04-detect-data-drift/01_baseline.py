#!/usr/bin/env python3
"""Baseline: only watch a delayed rolling MAPE on the already-scored demand model.

Alert when the 14-day rolling MAPE (using only labels that have already arrived)
is at least twice the in-distribution test MAPE. Because actuals arrive with a
7-day lag, this check is late by design. Reads outputs/02_scored_daily.csv.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import drift_tools as D
from model_common import LABEL_LAG, DRIFT_START, HOOK_DAY, TEST_END

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def main():
    df = pd.read_csv(OUT / "02_scored_daily.csv", parse_dates=["date"])
    test = df["split"] == "test"
    test_mape = D.mape(df.loc[test, "units"], df.loc[test, "pred"])
    test_mae = D.mae(df.loc[test, "units"], df.loc[test, "pred"])
    # Calibrate the cut on in-distribution 14-day rolling MAPE (95th percentile),
    # and never go below twice the test MAPE.
    roll = []
    test_days = df.loc[test, "date"]
    for day in test_days:
        window = df[df["date"] <= day].tail(14)
        if len(window) < 14:
            continue
        if window["split"].eq("test").all() or window["date"].min() >= df.loc[test, "date"].min():
            roll.append(D.mape(window["units"], window["pred"]))
    p95 = float(pd.Series(roll).quantile(0.95)) if roll else test_mape * 2
    cut = max(2.0 * test_mape, p95)

    rows = []
    for day in df.loc[df["date"] > TEST_END, "date"]:
        available = df["date"] <= (day - pd.Timedelta(days=LABEL_LAG))
        window = df[available].tail(14)
        if len(window) < 14:
            continue
        rm = D.mape(window["units"], window["pred"])
        rows.append({"date": day, "rolling_14_mape": rm, "alert": int(rm >= cut)})
    mon = pd.DataFrame(rows)
    first_alert = mon.loc[mon["alert"] == 1, "date"].min()
    lag_days = (first_alert - DRIFT_START).days if pd.notna(first_alert) else None

    lines = [
        "BASELINE: delayed rolling MAPE only",
        f"In-distribution test MAPE {test_mape*100:.1f}%  MAE {test_mae:.0f} units",
        f"Alert if the 14-day rolling MAPE, using labels already in, is at least {cut*100:.1f}% (max of twice the test MAPE and the test 95th percentile of 14-day MAPE).",
        f"Label lag {LABEL_LAG} days.",
        f"Planted shift starts {DRIFT_START.date()}.",
    ]
    if pd.notna(first_alert):
        lines.append(f"FIRST MAPE ALERT {first_alert.date()}  ({lag_days} days after the shift started).")
    else:
        lines.append("FIRST MAPE ALERT none in the production window.")
    hook_row = mon.loc[mon["date"] == HOOK_DAY]
    if len(hook_row):
        lines.append(
            f"HOOK 20 Apr 2026: delayed 14-day MAPE {hook_row.iloc[0]['rolling_14_mape']*100:.1f}%  "
            f"alert={int(hook_row.iloc[0]['alert'])}."
        )
    text = "\n".join(lines) + "\n"
    (OUT / "01_baseline.txt").write_text(text)
    mon.to_csv(OUT / "01_baseline_daily.csv", index=False)
    print(text, end="")


if __name__ == "__main__":
    main()
