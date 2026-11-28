#!/usr/bin/env python3
"""Generate the synthetic QuickBasket beverages-demand history for Episode 4.

The dataset is labelled SYNTHETIC. Seed 41. Nothing here is real customer data.

We plant a quiet input shift starting 11 April 2026 (a two-week app promo plus a
new delivery zone) so a demand model trained on the earlier world goes stale.
A hidden incident file is the answer key; the detectors never see it.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 41
START = "2024-12-01"
END = "2026-04-30"
DRIFT_START = pd.Timestamp("2026-04-11")  # Saturday, Summer Sips promo
NEW_ZONE_START = pd.Timestamp("2026-04-14")  # new Sharjah Edge zone, three days later
HOOK_DAY = pd.Timestamp("2026-04-20")  # Monday morning leadership pack
LABEL_LAG_DAYS = 7
TRAIN_END = pd.Timestamp("2025-11-30")
TEST_END = pd.Timestamp("2026-02-28")  # in-distribution holdout; March is calm production

HOLIDAYS = set(pd.to_datetime([
    "2024-12-02", "2024-12-03", "2024-12-25", "2025-01-01",
    "2025-03-30", "2025-03-31", "2025-04-01", "2025-06-06", "2025-06-07",
    "2025-12-02", "2025-12-03", "2025-12-25", "2026-01-01",
    "2026-03-20", "2026-03-21",
]).date)

# Quiet weekday multipliers (Mon=0). Friday is softer in this market; Saturday picks up.
DOW_MULT = {0: 1.00, 1: 0.97, 2: 0.95, 3: 0.98, 4: 1.08, 5: 1.18, 6: 0.88}

OUT = Path(__file__).resolve().parent / "data"


def main():
    rng = np.random.default_rng(SEED)
    days = pd.date_range(START, END, freq="D")
    n = len(days)
    dow = days.day_of_week
    is_weekend = (dow >= 4).astype(int)  # Fri-Sat
    is_holiday = np.array([d.date() in HOLIDAYS for d in days], dtype=int)
    # Mild seasonality and a slow growth trend (more customers over 17 months).
    t = np.arange(n)
    season = 0.08 * np.sin(2 * np.pi * (days.day_of_year - 15) / 365.25)  # warmer months slightly busier
    growth = 0.00012 * t

    # Temperature, C, Dubai-like, plus a little noise.
    temp = 28 + 10 * np.sin(2 * np.pi * (days.day_of_year - 200) / 365.25) + rng.normal(0, 1.2, n)
    # Competitor discount index: usually ~0.05, occasional spikes, then a lasting step at the promo.
    competitor = np.clip(0.05 + rng.normal(0, 0.015, n), 0, 0.4)

    # Pre-drift marketing and channel mix (stable world the model will learn).
    promo = np.clip(0.07 + 0.02 * is_weekend + rng.normal(0, 0.015, n), 0.01, 0.25)
    express = np.clip(0.18 + 0.03 * is_weekend + rng.normal(0, 0.02, n), 0.08, 0.35)
    app = np.clip(0.56 + 0.03 * is_weekend + rng.normal(0, 0.02, n), 0.40, 0.75)
    aov = 78 - 8 * promo + rng.normal(0, 2.0, n)
    new_zone = np.zeros(n)

    # Planned promo calendar (known in the morning) and realised mix after the shift.
    drift = days >= DRIFT_START
    # Ramp over 3 days so it is a shift, not a one-day spike.
    ramp = np.clip((days - DRIFT_START).days / 3.0, 0, 1)
    promo = np.where(drift, promo + ramp * 0.26, promo)          # ~0.07 -> ~0.33
    express = np.where(drift, express + ramp * 0.22, express)    # ~0.18 -> ~0.40
    app = np.where(drift, app + ramp * 0.16, app)
    aov = np.where(drift, aov - ramp * 16.0, aov)                # baskets get smaller
    competitor = np.where(drift, competitor + ramp * 0.12, competitor)
    zone = days >= NEW_ZONE_START
    zone_ramp = np.clip((days - NEW_ZONE_START).days / 4.0, 0, 1)
    new_zone = np.where(zone, 0.02 + zone_ramp * 0.10, 0.0)      # climbs toward ~12%

    promo = np.clip(promo, 0.01, 0.55)
    express = np.clip(express, 0.05, 0.60)
    app = np.clip(app, 0.35, 0.90)
    aov = np.clip(aov, 40, 120)
    new_zone = np.clip(new_zone, 0, 0.25)

    # True daily units. Before the shift, promo lifts demand; after the shift the
    # same promo_rate buys less lift (concept change), and smaller baskets plus a
    # new zone with different mix pull the level away from the training world.
    base = 12400.0
    holiday_mult = np.where(is_holiday == 1, 0.72, 1.0)
    dow_m = np.array([DOW_MULT[int(i)] for i in dow])
    promo_lift = 1.0 + 1.15 * promo
    # After the drift starts, the same promo rate buys far less lift (concept change),
    # express orders shrink the basket more than they used to, and a competitor
    # discount steals volume. Total units fall even while the promo flag is up.
    promo_lift = np.where(drift, 1.0 + 0.10 * promo, promo_lift)
    express_eff = np.where(drift, 1.0 - 0.45 * express, 1.0 - 0.20 * express)
    zone_eff = np.where(drift, 1.0 - 0.10 * new_zone, 1.0 + 0.15 * new_zone)
    comp_eff = np.where(drift, 1.0 - 0.80 * competitor, 1.0 - 0.40 * competitor)
    temp_eff = 1.0 + 0.006 * (temp - 28)
    noise = rng.lognormal(0, 0.035, n)  # ~3.5% multiplicative noise
    units = (base * (1 + growth + season) * dow_m * holiday_mult
             * promo_lift * express_eff * zone_eff * comp_eff * temp_eff * noise)
    units = np.round(units / 10) * 10  # warehouse ships in tens

    # Trailing features the morning model is allowed to see (no leakage of today's units).
    s = pd.Series(units, index=days)
    lag_1 = s.shift(1)
    lag_7 = s.shift(7)
    roll_7 = s.shift(1).rolling(7, min_periods=7).mean()
    roll_28 = s.shift(1).rolling(28, min_periods=14).mean()

    df = pd.DataFrame({
        "date": days,
        "dow": dow,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "temp_c": np.round(temp, 1),
        "promo_rate": np.round(promo, 4),
        "express_share": np.round(express, 4),
        "app_share": np.round(app, 4),
        "aov": np.round(aov, 2),
        "new_zone_share": np.round(new_zone, 4),
        "competitor_index": np.round(competitor, 4),
        "lag_1": np.round(lag_1, 0),
        "lag_7": np.round(lag_7, 0),
        "roll_7": np.round(roll_7, 1),
        "roll_28": np.round(roll_28, 1),
        "units": units.astype(int),
    })
    # Split flags (for the student's convenience; the model scripts recompute them).
    df["split"] = np.where(df["date"] <= TRAIN_END, "train",
                    np.where(df["date"] <= TEST_END, "test", "production"))

    # Actuals arrive with a 7-day lag. A row is "label_available" on date + 7.
    df["label_available_on"] = df["date"] + pd.Timedelta(days=LABEL_LAG_DAYS)

    OUT.mkdir(parents=True, exist_ok=True)
    daily_path = OUT / "daily_demand.csv"
    df.to_csv(daily_path, index=False)

    # Hidden answer key — never used at prediction time.
    inc = pd.DataFrame({
        "date": days[days >= DRIFT_START],
        "event": np.where(
            days[days >= DRIFT_START] < NEW_ZONE_START,
            "Summer Sips app promo: higher promo_rate, smaller baskets, more express and app orders; promo elasticity falls",
            "Summer Sips promo continues; Sharjah Edge zone starts contributing volume with a different mix",
        ),
        "kind": "planted_covariate_and_concept_shift",
    })
    inc.to_csv(OUT / "incident_log.csv", index=False)
    # Compact calendar used as an optional "known change" input in the feedback episode.
    calendar = pd.DataFrame({
        "date": [DRIFT_START, NEW_ZONE_START],
        "known_change": ["Summer Sips two-week app promo starts", "Sharjah Edge zone goes live"],
        "expected": [1, 1],
    })
    calendar.to_csv(OUT / "known_changes.csv", index=False)

    print(f"SYNTHETIC QuickBasket beverages demand  seed={SEED}")
    print(f"rows {len(df)}  {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"train <= {TRAIN_END.date()}  test <= {TEST_END.date()}  production after that")
    print(f"drift starts {DRIFT_START.date()}  new zone {NEW_ZONE_START.date()}  hook {HOOK_DAY.date()}")
    print(f"label lag {LABEL_LAG_DAYS} days")
    pre = df.loc[df["date"] < DRIFT_START, "units"]
    post = df.loc[df["date"] >= DRIFT_START, "units"]
    print(f"units mean pre {pre.mean():.0f}  post {post.mean():.0f}")
    print(f"promo mean pre {df.loc[df['date']<DRIFT_START,'promo_rate'].mean():.3f}  "
          f"post {df.loc[df['date']>=DRIFT_START,'promo_rate'].mean():.3f}")
    print(f"wrote {daily_path} ({daily_path.stat().st_size} bytes)")
    for p in sorted(OUT.glob("*.csv")):
        print(f"  {p.name:24s} {p.stat().st_size:7d} bytes")


if __name__ == "__main__":
    main()
