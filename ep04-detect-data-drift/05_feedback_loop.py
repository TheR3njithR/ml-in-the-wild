#!/usr/bin/env python3
"""Feedback: what the eight-day lead was worth, and why an automatic fallback is not enough.

The red flag on 14 April is not a magic new forecast. It is eight days of warning
before the delayed error pack breaks, so a planner can hold the auto-purchase
order and set a lower number. A naive swap to last week's volume barely helps,
because last week is still the old world. Known marketing changes do not silence
the alarm; they change the action from 'page someone' to 'use the fallback and
re-check the forecast'.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from model_common import DRIFT_START, HOOK_DAY, TEST_END

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def main():
    scored = pd.read_csv(OUT / "02_scored_daily.csv", parse_dates=["date"])
    mon = pd.read_csv(OUT / "03_drift_daily.csv", parse_dates=["date"])
    base = pd.read_csv(OUT / "01_baseline_daily.csv", parse_dates=["date"])
    first_red = mon.loc[mon["status"] == "RED", "date"].min()
    first_mape = base.loc[base["alert"] == 1, "date"].min()
    hook = scored.loc[scored["date"] == HOOK_DAY].iloc[0]
    win = scored[(scored["date"] >= DRIFT_START) & (scored["date"] <= HOOK_DAY)]
    model_err = float((win["pred"] - win["units"]).sum())
    n_calm = int((mon["date"] < DRIFT_START).sum())
    n_false_red = int(((mon["date"] < DRIFT_START) & (mon["status"] == "RED")).sum())
    n_false_amber = int(((mon["date"] < DRIFT_START) & (mon["status"] == "AMBER")).sum())
    lead = int((first_mape - first_red).days)
    # Naive automatic fallback: on red, send lag_7 instead of the model forecast.
    prod = scored[scored["date"] > TEST_END].copy()
    red = set(mon.loc[mon["status"] == "RED", "date"])
    naive = prod["pred"].copy()
    hold = prod["date"].isin(red)
    naive[hold] = prod.loc[hold, "lag_7"]
    hook_mask = (prod["date"] >= DRIFT_START) & (prod["date"] <= HOOK_DAY)
    naive_err = float((naive[hook_mask] - prod.loc[hook_mask, "units"]).sum())

    lines = [
        "FEEDBACK: eight days of warning, and what a hold is worth",
        f"Planted shift starts {DRIFT_START.date()}. First red flag {first_red.date()}.",
        f"First delayed-MAPE alert {first_mape.date()}. Lead time {lead} days.",
        f"Calm production days {n_calm}: false red {n_false_red}, false amber {n_false_amber}.",
        "",
        f"HOOK morning {HOOK_DAY.date()}: the model wanted {int(hook['pred']):,} units.",
        f"The actual that arrived a week later was {int(hook['units']):,}, so that one order would have been {int(hook['err']):+} units off.",
        f"SHIFT-TO-HOOK {DRIFT_START.date()} to {HOOK_DAY.date()} ({len(win)} days):",
        f"  Sending the stale forecast every morning: net over-order {model_err:+,.0f} units.",
        f"  Naive automatic fallback (on red, send last week's fulfilled volume): net over-order {naive_err:+,.0f} units.",
        f"  A last-week swap barely helps, because last week is still the old world.",
        "",
        "The value of the flag is the time: a planner who holds on 14 April has eight mornings",
        "to set a lower number before the error pack finally breaks on 22 April.",
        "Known-change calendar (Summer Sips promo, Sharjah Edge zone) is in data/known_changes.csv.",
        "An expected marketing change does not silence the red flag. It changes the action from",
        "'page someone in a panic' to 'hold the auto-order, use a conservative number, and re-check the forecast'.",
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "05_feedback.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
