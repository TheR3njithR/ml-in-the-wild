"""Generate the SYNTHETIC dataset for ML in the Wild, Episode 1: "Can ML Detect When a KPI Dashboard Is Lying?"

Company: "QuickBasket", a fictional 10-minute grocery delivery app in five Indian cities.
Everything here is made up. No real company, customer or transaction data is used.

Writes:
  data/kpi_daily.csv       the table behind the daily KPI dashboard (city x day)
  data/event_calendar.csv  real business events (sales, festivals, heavy rain) that move the numbers for real
  data/incident_log.csv    ground truth: days where the DATA was wrong (only for evaluating the model;
                           in real life you never get this file up front)
Run: python make_data.py   (seeded, so every run gives the same data)
"""
import os
import numpy as np, pandas as pd

rng = np.random.default_rng(2026)
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)
DATES = pd.date_range("2024-09-01", "2026-10-31", freq="D")
CITIES = {  # base daily orders, base conversion, base average order value (Rs)
    "Bengaluru": (4200, 0.044, 560), "Mumbai": (5200, 0.041, 610), "Delhi": (4800, 0.039, 540),
    "Hyderabad": (2900, 0.046, 500), "Pune": (2100, 0.043, 520)}

# ---- real business events: the numbers move, but the data is TRUE --------------------------------
EVENTS = [  # date(s), name, order multiplier, session multiplier, AOV multiplier, cities (None = all)
    (["2024-10-31", "2024-11-01"], "Diwali", 2.3, 2.2, 1.12, None),
    (["2024-12-31"], "New Year's Eve", 1.9, 1.8, 1.15, None),
    (["2025-01-26"], "Republic Day Sale", 1.7, 1.75, 0.95, None),
    (["2025-03-14"], "Holi", 1.6, 1.55, 1.05, None),
    (["2025-05-16", "2025-05-17", "2025-05-18"], "Big Summer Sale", 2.0, 2.05, 0.93, None),
    (["2025-08-15"], "Independence Day Sale", 1.8, 1.85, 0.95, None),
    (["2025-08-27"], "Ganesh Chaturthi", 1.7, 1.6, 1.06, ["Mumbai", "Pune", "Hyderabad"]),
    (["2025-10-02"], "Dussehra", 1.5, 1.45, 1.05, None),
    (["2025-10-20", "2025-10-21"], "Diwali", 2.4, 2.3, 1.12, None),
    (["2025-12-31"], "New Year's Eve", 2.0, 1.9, 1.15, None),
    (["2026-01-26"], "Republic Day Sale", 1.8, 1.85, 0.95, None),
    (["2026-03-04"], "Holi", 1.6, 1.55, 1.05, None),
    (["2026-05-15", "2026-05-16", "2026-05-17"], "Big Summer Sale", 2.1, 2.15, 0.93, None),
    (["2026-08-15"], "Independence Day Sale", 1.9, 1.95, 0.95, None),
    (["2026-09-14"], "Ganesh Chaturthi", 1.7, 1.6, 1.06, ["Mumbai", "Pune", "Hyderabad"]),
    (["2026-10-20"], "Dussehra", 1.5, 1.45, 1.05, None),
]
RAIN = {"Mumbai": ["2025-07-07", "2025-07-08", "2025-07-25", "2026-07-06", "2026-07-16", "2026-07-17", "2026-08-03"],
        "Bengaluru": ["2025-10-14", "2026-10-07"], "Delhi": ["2025-07-31", "2026-07-29"]}

rows = []
for city, (o0, c0, a0) in CITIES.items():
    t = np.arange(len(DATES)) / 365.0
    trend = 1.0 + 0.22 * t                                   # +22% a year
    dow = np.array([0.95, 0.97, 0.98, 1.0, 1.06, 1.17, 1.14])[DATES.day_of_week]
    season = 1.0 + 0.04 * np.sin(2 * np.pi * (DATES.day_of_year - 30) / 365.0)
    orders = o0 * trend * dow * season * np.exp(rng.normal(0, 0.06, len(DATES)))
    conv = c0 * np.where(DATES.day_of_week >= 5, 1.04, 1.0) * np.exp(rng.normal(0, 0.035, len(DATES)))
    aov = a0 * (1 + 0.05 * t) * np.exp(rng.normal(0, 0.03, len(DATES)))
    refund_rate = 0.021 * np.exp(rng.normal(0, 0.12, len(DATES)))
    delivery = 21.5 + rng.normal(0, 1.2, len(DATES))
    df = pd.DataFrame({"date": DATES, "city": city, "orders": orders, "conv": conv, "aov": aov,
                       "refund_rate": refund_rate, "delivery_min": delivery})
    rows.append(df)
df = pd.concat(rows, ignore_index=True)
df["event"] = ""

cal = []
for dates, name, om, sm, am, cities in EVENTS:
    for d in dates:
        m = (df.date == d) & (df.city.isin(cities) if cities else True)
        df.loc[m, "orders"] *= om; df.loc[m, "conv"] *= om / sm; df.loc[m, "aov"] *= am; df.loc[m, "event"] = name
        df.loc[m, "delivery_min"] += 3.0
        cal.append({"date": d, "event": name, "cities": "All" if not cities else ", ".join(cities), "type": "sale/festival"})
for city, ds in RAIN.items():
    for d in ds:
        m = (df.date == d) & (df.city == city)
        df.loc[m, "orders"] *= 1.22; df.loc[m, "conv"] *= 1.08; df.loc[m, "delivery_min"] += 11; df.loc[m, "event"] = "Heavy rain"
        cal.append({"date": d, "event": "Heavy rain", "cities": city, "type": "weather"})

# integer dashboard columns
df["orders"] = df.orders.round().astype(int)
df["sessions"] = (df.orders / df.conv).round().astype(int)
df["revenue_inr"] = (df.orders * df.aov).round().astype(int)
df["refunds"] = rng.binomial(df.orders, df.refund_rate)
df["delivery_min"] = df.delivery_min.round(1)
df = df.sort_values(["date", "city"]).reset_index(drop=True)

# ---- data incidents: the business did NOT change, the DATA broke -----------------------------------
INCIDENTS = [  # start, days, type, cities (None = all), parameter
    ("2025-01-09", 1, "DUPLICATE_LOAD", ["Delhi"], 2.0),
    ("2025-02-18", 2, "TRACKING_TAG_BROKEN", None, 0.38),
    ("2025-04-03", 1, "PARTIAL_LOAD", ["Pune"], 0.62),
    ("2025-05-27", 1, "STALE_COPY", ["Hyderabad"], None),
    ("2025-06-19", 1, "TAX_DOUBLE_COUNTED", ["Mumbai"], 1.18),
    ("2025-09-10", 1, "REFUND_FEED_MISSING", None, None),
    # test year (Nov 2025 - Oct 2026)
    ("2025-11-12", 1, "DUPLICATE_LOAD", ["Bengaluru"], 2.0),
    ("2025-11-26", 1, "PARTIAL_LOAD", ["Mumbai"], 0.80),
    ("2025-12-09", 2, "TRACKING_TAG_BROKEN", ["Delhi", "Hyderabad"], 0.40),
    ("2026-01-06", 1, "STALE_COPY", ["Pune"], None),
    ("2026-01-21", 1, "TAX_DOUBLE_COUNTED", None, 1.18),
    ("2026-02-11", 1, "REFUND_FEED_MISSING", ["Bengaluru", "Mumbai"], None),
    ("2026-03-17", 1, "PARTIAL_LOAD", None, 0.70),
    ("2026-04-08", 1, "DUPLICATE_LOAD", ["Hyderabad"], 1.5),
    ("2026-04-22", 1, "TRACKING_DOUBLE_FIRE", ["Mumbai"], 1.65),
    ("2026-05-28", 2, "STALE_COPY", ["Delhi"], None),
    ("2026-06-10", 1, "TAX_DOUBLE_COUNTED", ["Bengaluru"], 1.18),
    ("2026-07-01", 1, "PARTIAL_LOAD", ["Delhi"], 0.83),
    ("2026-07-22", 1, "TRACKING_TAG_BROKEN", ["Pune"], 0.45),
    ("2026-08-26", 1, "DUPLICATE_LOAD", ["Mumbai", "Pune"], 2.0),
    ("2026-09-09", 1, "REFUND_FEED_MISSING", ["Hyderabad"], None),
    ("2026-09-30", 1, "STALE_COPY", ["Bengaluru"], None),
    ("2026-10-13", 1, "TRACKING_DOUBLE_FIRE", None, 1.55),
]
log = []
for start, days, kind, cities, p in INCIDENTS:
    for k in range(days):
        d = pd.Timestamp(start) + pd.Timedelta(days=k)
        for city in (cities or list(CITIES)):
            i = df.index[(df.date == d) & (df.city == city)][0]
            if kind == "DUPLICATE_LOAD":      # the same orders file loaded twice (or 1.5x: one batch twice)
                for c in ("orders", "revenue_inr", "refunds"): df.loc[i, c] = int(round(df.loc[i, c] * p))
            elif kind == "TRACKING_TAG_BROKEN":   # web/app analytics tag stopped firing for part of the day
                df.loc[i, "sessions"] = int(df.loc[i, "sessions"] * p)
            elif kind == "TRACKING_DOUBLE_FIRE":  # tag fires twice on some pages
                df.loc[i, "sessions"] = int(df.loc[i, "sessions"] * p)
            elif kind == "PARTIAL_LOAD":          # the nightly job ran before all orders landed
                for c in ("orders", "revenue_inr", "refunds"): df.loc[i, c] = int(round(df.loc[i, c] * p))
            elif kind == "STALE_COPY":            # pipeline failed silently; yesterday's numbers were copied forward
                j = df.index[(df.date == d - pd.Timedelta(days=1)) & (df.city == city)][0]
                for c in ("orders", "sessions", "revenue_inr", "refunds", "delivery_min"): df.loc[i, c] = df.loc[j, c]
            elif kind == "TAX_DOUBLE_COUNTED":    # revenue column now includes 18% GST on top of a GST-inclusive price
                df.loc[i, "revenue_inr"] = int(round(df.loc[i, "revenue_inr"] * p))
            elif kind == "REFUND_FEED_MISSING":   # refunds table did not arrive; join filled zeros
                df.loc[i, "refunds"] = 0
            log.append({"date": d.date().isoformat(), "city": city, "incident": kind})

out = df[["date", "city", "orders", "sessions", "revenue_inr", "refunds", "delivery_min"]].copy()
out["date"] = out.date.dt.date.astype(str)
out.to_csv(os.path.join(OUT, "kpi_daily.csv"), index=False)
pd.DataFrame(cal).to_csv(os.path.join(OUT, "event_calendar.csv"), index=False)
pd.DataFrame(log).to_csv(os.path.join(OUT, "incident_log.csv"), index=False)
print(f"SYNTHETIC data written to data/: {len(out):,} city-days ({out.date.min()} to {out.date.max()}), "
      f"{len(CITIES)} cities, {len(cal)} real event city-groups, {len(log)} incident city-days")
