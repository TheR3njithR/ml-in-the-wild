"""Generate the SYNTHETIC dataset for ML in the Wild, Episode 2:
"Predict a Power BI Refresh Failure Before It Happens".

Company: "QuickBasket", the same fictional 10-minute grocery delivery app as Episode 1.
Everything here is made up. No real Power BI tenant, gateway or company data is used.

Writes:
  data/refresh_history.csv   one row per scheduled refresh (features at T-60 are known before the run)
  data/gateway_heartbeat.csv 1-minute gateway metrics, 04:00-10:00 each morning
  data/close_calendar.csv    month-end close days the business already knows about
  data/incident_log.csv      ground truth failure reasons (evaluation only; the model never sees this)

Run: python make_data.py   (seeded, so every run gives the same data)
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta, time
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)

DATES = pd.date_range("2024-12-01", "2026-03-31", freq="D")
TRAIN_END = pd.Timestamp("2025-07-31")
TEST_START = pd.Timestamp("2025-08-01")
HOOK = pd.Timestamp("2026-03-14")
REVIEW_END = pd.Timestamp("2025-09-11")

DATASETS = [
    # name, workspace, scheduled clock, typical duration (min), typical size (MB)
    ("Vendor", "Procurement", time(5, 30), 18, 400),
    ("Finance", "Finance", time(6, 0), 28, 900),
    ("Marketing", "Marketing", time(6, 15), 22, 600),
    ("Warehouse", "Ops", time(6, 45), 20, 700),
    ("Delivery SLA", "Ops", time(7, 0), 16, 350),
    ("City Ops", "Ops", time(7, 30), 24, 1200),
    ("Refunds", "Finance", time(7, 45), 14, 280),
    ("Inventory", "Ops", time(8, 0), 18, 500),
]
NAMES = [d[0] for d in DATASETS]
SCHED = {d[0]: d[2] for d in DATASETS}
BASE_DUR = {d[0]: d[3] for d in DATASETS}
BASE_SIZE = {d[0]: d[4] for d in DATASETS}

# ---- close calendar: last two weekdays of each month, plus the first weekday of the next ----
close_rows = []
for p in pd.period_range("2024-12", "2026-03", freq="M"):
    days = pd.date_range(p.start_time, p.end_time, freq="D")
    weekdays = [d for d in days if d.day_of_week < 5]
    for d in weekdays[-2:]:
        close_rows.append({"date": d.date().isoformat(), "event": "Month-end close"})
    nxt = (p + 1).start_time
    while nxt.day_of_week >= 5:
        nxt += pd.Timedelta(days=1)
    if DATES[0] <= nxt <= DATES[-1]:
        close_rows.append({"date": nxt.date().isoformat(), "event": "Month-end close"})
close = pd.DataFrame(close_rows).drop_duplicates("date")
CLOSE = set(close.date)

# ---- scheduled runs with healthy defaults ---------------------------------------------------
rows = []
for d in DATES:
    t = (d - DATES[0]).days / 365.0
    monday = int(d.day_of_week == 0)
    month_end = int(d.date().isoformat() in CLOSE)
    for name, workspace, sched, dur0, size0 in DATASETS:
        scheduled = datetime.combine(d.date(), sched)
        size = size0 * (1 + 0.18 * t) * float(np.exp(rng.normal(0, 0.03)))
        dur = dur0 * float(np.exp(rng.normal(0, 0.08)))
        rows.append({
            "date": d.date().isoformat(),
            "dataset": name,
            "workspace": workspace,
            "scheduled_at": scheduled.isoformat(sep=" "),
            "size_mb": round(size, 1),
            "planned_duration_min": round(dur, 1),
            "is_monday": monday,
            "is_month_end": month_end,
        })
runs = pd.DataFrame(rows)

# Credentials normally rotate at 78 days, so the 85-day rule stays quiet.
# A minority of cycles are allowed to approach 90 so we can plant expiries and a few rule false alarms.
cred_age = {}
for name in NAMES:
    age = int(rng.integers(5, 40))
    for d in DATES:
        age += 1
        cred_age[(d.date().isoformat(), name)] = age
        if age >= 78:
            age = 0
runs["credential_age_days"] = [cred_age[(r.date, r.dataset)] for r in runs.itertuples()]

# ---- morning gateway heartbeat, 04:00-10:00 at 1 minute --------------------------------------
hb_index = pd.date_range(
    datetime.combine(DATES[0].date(), time(4, 0)),
    datetime.combine(DATES[-1].date(), time(10, 0)),
    freq="min",
)
hb_index = hb_index[(hb_index.hour >= 4) & (hb_index.hour < 10)]
minutes = hb_index.hour * 60 + hb_index.minute
day_shift = pd.Series(hb_index.normalize()).map(
    {pd.Timestamp(d): float(rng.normal(0, 3.5)) for d in DATES}
).to_numpy()
tod = 48 + 12 * np.clip((minutes - 240) / 270, 0, 1)
mem = np.clip(tod + day_shift + rng.normal(0, 1.4, len(hb_index)), 20, 82)
cpu = np.clip(22 + 14 * np.clip((minutes - 240) / 270, 0, 1) + day_shift * 0.4 + rng.normal(0, 2.0, len(hb_index)), 5, 90)
hb = pd.DataFrame({
    "ts": hb_index,
    "gateway_mem_pct": mem.round(1),
    "gateway_cpu_pct": cpu.round(1),
    "gateway_online": np.ones(len(hb_index), dtype=np.int8),
})
hb["date"] = hb.ts.dt.date.astype(str)
hb["capacity_pct"] = np.clip(
    38
    + 14 * hb.date.isin(CLOSE)
    + 7 * (hb.ts.dt.day_of_week == 0)
    + 8 * np.clip((minutes - 240) / 270, 0, 1)
    + rng.normal(0, 3.0, len(hb)),
    10, 80,
).round(1)

# ---- plant incidents ------------------------------------------------------------------------
def slots_for(pool_dates, n, used, prefer=None):
    names = list(prefer) if prefer else list(NAMES)
    candidates = [(d.date().isoformat(), name) for d in pool_dates for name in names]
    rng.shuffle(candidates)
    # if prefer is a subset, fall back to all names if needed
    extra = [(d.date().isoformat(), name) for d in pool_dates for name in NAMES if (d.date().isoformat(), name) not in candidates]
    rng.shuffle(extra)
    out = []
    for key in candidates + extra:
        if key in used or key[0] == HOOK.date().isoformat():
            continue
        used.add(key)
        out.append(key)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only got {len(out)} of {n} slots")
    return out

used = set()
used.add((HOOK.date().isoformat(), "City Ops"))

train_dates = DATES[DATES <= TRAIN_END]
test_dates = DATES[DATES >= TEST_START]
early_test = DATES[(DATES >= TEST_START) & (DATES <= REVIEW_END)]
later_test = DATES[DATES > REVIEW_END]
LARGE = ["City Ops", "Finance", "Warehouse", "Marketing"]

# Test: 24 memory + hook = 25, plus the rest = 127
TEST_PLAN = {
    # kind: (early, later, preferred datasets or None)
    "GATEWAY_MEMORY": (4, 8, ["City Ops"]),
    "GATEWAY_OFFLINE": (2, 4, ["City Ops"]),
    "CREDENTIAL_EXPIRED": (4, 10, None),
    "CAPACITY_THROTTLE": (3, 6, None),
    "REFRESH_OVERLAP": (5, 11, LARGE),
    "SIZE_JUMP": (3, 7, None),
    "SOURCE_TIMEOUT": (0, 1, None),          # 1 with a leading duration-trend signal
    "SOURCE_TIMEOUT_SILENT": (4, 15, None),  # 19 silent
    "NETWORK_BLIP": (3, 11, None),           # 14
}
TRAIN_PLAN = {
    "GATEWAY_MEMORY": (20, ["City Ops"]),
    "GATEWAY_OFFLINE": (10, ["City Ops"]),
    "CREDENTIAL_EXPIRED": (14, None),
    "CAPACITY_THROTTLE": (8, None),
    "REFRESH_OVERLAP": (16, LARGE),
    "SIZE_JUMP": (10, None),
    "SOURCE_TIMEOUT": (8, None),
    "SOURCE_TIMEOUT_SILENT": (16, None),
    "NETWORK_BLIP": (12, None),
}

incidents = []

def add_inc(date, dataset, kind, **extra):
    incidents.append({"date": date, "dataset": dataset, "incident": kind, **extra})

add_inc(HOOK.date().isoformat(), "City Ops", "GATEWAY_MEMORY", hook=True)

for kind, (n_early, n_later, prefer) in TEST_PLAN.items():
    silent = kind.endswith("_SILENT")
    base = kind.replace("_SILENT", "")
    for d, ds in slots_for(early_test, n_early, used, prefer):
        add_inc(d, ds, base, silent=silent, signal=(base == "SOURCE_TIMEOUT" and not silent))
    for d, ds in slots_for(later_test, n_later, used, prefer):
        add_inc(d, ds, base, silent=silent, signal=(base == "SOURCE_TIMEOUT" and not silent))

for kind, (n, prefer) in TRAIN_PLAN.items():
    silent = kind.endswith("_SILENT")
    base = kind.replace("_SILENT", "")
    for d, ds in slots_for(train_dates, n, used, prefer):
        add_inc(d, ds, base, silent=silent, signal=(base == "SOURCE_TIMEOUT" and not silent))

# Rule false alarms in test: credential 86-89 on a successful morning, and short memory spikes.
for d, ds in slots_for(test_dates, 40, used):
    add_inc(d, ds, "RULE_FP_CREDENTIAL")
for d, ds in slots_for(test_dates, 45, used):
    add_inc(d, ds, "RULE_FP_MEMSPIKE")
# A few of the same in train so the model can learn they are not failures
for d, ds in slots_for(train_dates, 30, used):
    add_inc(d, ds, "RULE_FP_CREDENTIAL")
for d, ds in slots_for(train_dates, 30, used):
    add_inc(d, ds, "RULE_FP_MEMSPIKE")
# Three close-day "busy but fine" mornings in the review window, for the feedback story
close_early = [d for d in early_test if d.date().isoformat() in CLOSE]
# pick three (date, City Ops or Finance) close slots if free, else any
n_close_fp = 0
for d in close_early:
    for ds in ("City Ops", "Finance", "Warehouse"):
        key = (d.date().isoformat(), ds)
        if key not in used and n_close_fp < 3:
            used.add(key)
            add_inc(*key, "CLOSE_BUSY")
            n_close_fp += 1

# ---- apply to heartbeat and outcomes --------------------------------------------------------
runs["failed"] = False
runs["incident"] = ""
runs["notice_delay_min"] = np.nan
runs["started_at"] = runs.scheduled_at
runs["ended_at"] = ""
runs["duration_min"] = runs.planned_duration_min
runs["status"] = "success"

idx = {(date, dataset): i for i, date, dataset in runs[["date", "dataset"]].itertuples(index=True)}

def hb_slice(date, t0, t1):
    a = datetime.combine(pd.Timestamp(date).date(), t0)
    b = datetime.combine(pd.Timestamp(date).date(), t1)
    return (hb.ts >= a) & (hb.ts <= b)

def set_mem(date, t0, t1, lo=91.5, hi=96.5):
    m = hb_slice(date, t0, t1)
    n = int(m.sum())
    if n:
        hb.loc[m, "gateway_mem_pct"] = np.clip(rng.uniform(lo, hi, n) if False else lo + (hi - lo) * rng.random(n), lo, hi).round(1)

def set_fail(date, dataset, kind, duration, delay):
    i = idx[(date, dataset)]
    start = pd.Timestamp(runs.at[i, "scheduled_at"])
    end = start + pd.Timedelta(minutes=duration)
    runs.at[i, "failed"] = True
    runs.at[i, "incident"] = kind
    runs.at[i, "status"] = "failed"
    runs.at[i, "duration_min"] = round(float(duration), 1)
    runs.at[i, "started_at"] = start.strftime("%Y-%m-%d %H:%M:%S")
    runs.at[i, "ended_at"] = end.strftime("%Y-%m-%d %H:%M:%S")
    runs.at[i, "notice_delay_min"] = round(float(delay), 1)

def delay():
    return float(np.clip(rng.normal(19, 6), 5, 45))

for inc in incidents:
    date, ds, kind = inc["date"], inc["dataset"], inc["incident"]
    sched = SCHED[ds]
    t_sched = datetime.combine(pd.Timestamp(date).date(), sched)

    if inc.get("hook"):
        inc["_hook"] = True
        kind = "GATEWAY_MEMORY"
        # fall through to GATEWAY_MEMORY so the hook uses the typical pattern


    if kind == "GATEWAY_MEMORY":
        # Raise memory 90 minutes before the schedule, so T-60 already has ~30 minutes above 90%.
        t0 = (t_sched - timedelta(minutes=90)).time()
        # Control the whole T-60 hour so mem_mean_1h is stable across memory days
        # (the 1-hour window is schedule-120 to schedule-60). Keep early minutes ~70,
        # short ramp, then hold above 90 from 90 minutes before the start.
        pre0 = (t_sched - timedelta(minutes=120)).time()
        ramp0 = (t_sched - timedelta(minutes=100)).time()
        mpre = hb_slice(date, pre0, ramp0)
        if mpre.any():
            hb.loc[mpre, "gateway_mem_pct"] = 70.0
        mramp = hb_slice(date, ramp0, t0)
        if mramp.any():
            n = int(mramp.sum())
            hb.loc[mramp, "gateway_mem_pct"] = np.linspace(70.0, 89.5, n).round(1)
        set_mem(date, t0, time(9, 40), 91.5, 96.0)
        if ds == "City Ops":
            for other, dur in (("Finance", 42.0), ("Marketing", 36.0)):
                j = idx[(date, other)]
                if runs.at[j, "status"] == "success":
                    start = pd.Timestamp(runs.at[j, "scheduled_at"])
                    runs.at[j, "duration_min"] = dur
                    runs.at[j, "started_at"] = start.strftime("%Y-%m-%d %H:%M:%S")
                    runs.at[j, "ended_at"] = (start + pd.Timedelta(minutes=dur)).strftime("%Y-%m-%d %H:%M:%S")
        if inc.get("hook") or inc.get("_hook"):
            set_fail(date, ds, kind, 52.0, 21.0)
        else:
            set_fail(date, ds, kind, rng.uniform(38, 58), delay())
        continue

    if kind == "GATEWAY_OFFLINE":
        t0 = (t_sched - timedelta(minutes=80)).time()
        m = hb_slice(date, t0, time(9, 30))
        hb.loc[m, "gateway_online"] = 0
        set_fail(date, ds, kind, rng.uniform(1, 4), delay())
        continue

    if kind == "CREDENTIAL_EXPIRED":
        i = idx[(date, ds)]
        runs.at[i, "credential_age_days"] = int(rng.integers(91, 99))
        set_fail(date, ds, kind, rng.uniform(2, 8), delay())
        continue

    if kind == "CAPACITY_THROTTLE":
        t0 = (t_sched - timedelta(minutes=70)).time()
        m = hb_slice(date, t0, time(9, 45))
        n = int(m.sum())
        if n:
            hb.loc[m, "capacity_pct"] = np.clip(87 + rng.normal(0, 2.0, n), 84, 98).round(1)
        set_fail(date, ds, kind, rng.uniform(25, 48), delay())
        continue

    if kind == "REFRESH_OVERLAP":
        others = [n for n in NAMES if n != ds]
        rng.shuffle(others)
        for other in others[:2]:
            j = idx[(date, other)]
            start = pd.Timestamp(runs.at[j, "scheduled_at"])
            dur = 55.0
            runs.at[j, "duration_min"] = dur
            runs.at[j, "started_at"] = start.strftime("%Y-%m-%d %H:%M:%S")
            runs.at[j, "ended_at"] = (start + pd.Timedelta(minutes=dur)).strftime("%Y-%m-%d %H:%M:%S")
        set_fail(date, ds, kind, rng.uniform(30, 55), delay())
        continue

    if kind == "SIZE_JUMP":
        i = idx[(date, ds)]
        runs.at[i, "size_mb"] = round(float(runs.at[i, "size_mb"]) * float(rng.uniform(2.3, 3.2)), 1)
        set_fail(date, ds, kind, rng.uniform(40, 70), delay())
        continue

    if kind == "SOURCE_TIMEOUT":
        set_fail(date, ds, kind, rng.uniform(28, 46), delay())
        if inc.get("signal"):
            inc["_slow"] = True
        continue

    if kind == "NETWORK_BLIP":
        set_fail(date, ds, kind, rng.uniform(6, 16), delay())
        continue

    if kind == "RULE_FP_CREDENTIAL":
        i = idx[(date, ds)]
        runs.at[i, "credential_age_days"] = int(rng.integers(86, 90))
        continue

    if kind == "RULE_FP_MEMSPIKE":
        # A short 8-minute bump to 86-89 around this dataset's T-60, then it recovers.
        t60 = t_sched - timedelta(minutes=60)
        t0 = (t60 - timedelta(minutes=3)).time()
        t1 = (t60 + timedelta(minutes=5)).time()
        set_mem(date, t0, t1, 86.2, 88.8)
        continue

    if kind == "CLOSE_BUSY":
        # Busy close morning: elevated capacity and a little memory, but the refresh succeeds.
        t0 = (t_sched - timedelta(minutes=80)).time()
        m = hb_slice(date, t0, time(9, 0))
        n = int(m.sum())
        if n:
            hb.loc[m, "capacity_pct"] = np.clip(hb.loc[m, "capacity_pct"] + 12, 10, 83).round(1)
            hb.loc[m, "gateway_mem_pct"] = np.clip(hb.loc[m, "gateway_mem_pct"] + 8, 20, 84).round(1)
        continue

# Slow-history leading signal for a few timeouts
for inc in incidents:
    if inc.get("signal") and inc["incident"] == "SOURCE_TIMEOUT":
        date, ds = inc["date"], inc["dataset"]
        prev = runs[(runs.dataset == ds) & (runs.date < date) & (runs.status == "success")].tail(6)
        for j in prev.index:
            runs.at[j, "duration_min"] = round(float(runs.at[j, "duration_min"]) * 1.35, 1)
            start = pd.Timestamp(runs.at[j, "started_at"] if runs.at[j, "started_at"] else runs.at[j, "scheduled_at"])
            runs.at[j, "ended_at"] = (start + pd.Timedelta(minutes=float(runs.at[j, "duration_min"]))).strftime("%Y-%m-%d %H:%M:%S")

# Fill remaining end times
for i in runs.index:
    if not runs.at[i, "ended_at"]:
        start = pd.Timestamp(runs.at[i, "scheduled_at"])
        dur = float(runs.at[i, "duration_min"])
        runs.at[i, "started_at"] = start.strftime("%Y-%m-%d %H:%M:%S")
        runs.at[i, "ended_at"] = (start + pd.Timedelta(minutes=dur)).strftime("%Y-%m-%d %H:%M:%S")

# Center test notice delays on 19.0 minutes
test_fail = (runs.status == "failed") & (runs.date >= TEST_START.date().isoformat())
delays = runs.loc[test_fail, "notice_delay_min"].astype(float).clip(5, 45)
if len(delays):
    delays = (delays - delays.mean() + 19.0).clip(5, 45)
    delays = delays - delays.mean() + 19.0
    runs.loc[test_fail, "notice_delay_min"] = delays.round(1)

# ---- leading features at T-60 ---------------------------------------------------------------
hb = hb.sort_values("ts").reset_index(drop=True)
hb_i = hb.set_index("ts")

def window_stats(ts_end, minutes=60):
    ts_end = pd.Timestamp(ts_end)
    ts_start = ts_end - pd.Timedelta(minutes=minutes)
    sl = hb_i.loc[ts_start:ts_end]
    if sl.empty:
        day = ts_end.normalize()
        sl = hb_i.loc[day + pd.Timedelta(hours=4):ts_end]
    if sl.empty:
        return dict(mem_now=55.0, mem_mean_1h=55.0, mem_max_1h=55.0, mem_over_90_min=0,
                    cpu_mean_1h=30.0, online=1, capacity_mean_1h=40.0)
    mem = sl.gateway_mem_pct
    over = 0
    for v in (mem > 90).to_numpy()[::-1]:
        if v:
            over += 1
        else:
            break
    return dict(
        mem_now=float(mem.iloc[-1]),
        mem_mean_1h=round(float(mem.mean()), 2),
        mem_max_1h=float(mem.max()),
        mem_over_90_min=int(over),
        cpu_mean_1h=round(float(sl.gateway_cpu_pct.mean()), 2),
        online=int(sl.gateway_online.iloc[-1]),
        capacity_mean_1h=round(float(sl.capacity_pct.mean()), 2),
    )

starts = pd.to_datetime(runs.started_at)
ends = pd.to_datetime(runs.ended_at)
sched = pd.to_datetime(runs.scheduled_at)

feat_rows = []
for i, r in runs.iterrows():
    t0 = sched[i]
    t60 = t0 - pd.Timedelta(minutes=60)
    st = window_stats(t60, 60)
    running = int(((starts < t60) & (ends > t60) & (runs.index != i)).sum())
    queued = int(((sched > t60) & (sched <= t60 + pd.Timedelta(minutes=90)) & (runs.index != i) & (runs.date == r.date)).sum())
    feat_rows.append({**st, "concurrent_running": running, "queued_next_90m": queued})
runs = pd.concat([runs, pd.DataFrame(feat_rows, index=runs.index)], axis=1)

runs["date_ts"] = pd.to_datetime(runs.date)
runs["size_vs_28d"] = 1.0
for name, g in runs.groupby("dataset"):
    med = g.set_index("date_ts").size_mb.shift(1).rolling("28D", min_periods=8).median()
    runs.loc[g.index, "size_vs_28d"] = g.size_mb.values / med.reindex(g.date_ts).values

runs["duration_trend"] = 1.0
for name, g in runs.groupby("dataset"):
    g = g.sort_values("date_ts")
    past_succ, vals = [], []
    for i in g.index:
        if len(past_succ) >= 8:
            recent, older = past_succ[-7:], past_succ[-28:]
            vals.append((sum(recent) / len(recent)) / (sum(older) / len(older)))
        else:
            vals.append(1.0)
        if runs.at[i, "status"] == "success":
            past_succ.append(float(runs.at[i, "duration_min"]))
    runs.loc[g.index, "duration_trend"] = vals

runs["failed_yesterday"] = 0
for name, g in runs.groupby("dataset"):
    g = g.sort_values("date_ts")
    runs.loc[g.index, "failed_yesterday"] = g.failed.shift(1).fillna(False).astype(int).values


# Consistency pass: gateway-wide conditions must not appear on successful runs.
# Otherwise the model cannot learn from them. Dataset-specific failures (blips,
# timeouts, size jumps, credentials, overlap) stay on the planted target only.
n_extra = 0
for i in runs.index:
    if runs.at[i, "status"] == "failed":
        continue
    online = int(runs.at[i, "online"]) if "online" in runs.columns else int(runs.at[i, "gateway_online"]) if "gateway_online" in runs.columns else 1
    # columns after concat: "online" from window_stats
    mem90 = int(runs.at[i, "mem_over_90_min"])
    size = float(runs.at[i, "size_mb"])
    cap = float(runs.at[i, "capacity_mean_1h"])
    kind = None
    if online == 0:
        kind = "GATEWAY_OFFLINE"
        dur = 3.0
    elif mem90 >= 15 and size >= 850:
        kind = "GATEWAY_MEMORY"
        dur = 45.0
    elif cap >= 85:
        kind = "CAPACITY_THROTTLE"
        dur = 35.0
    if kind:
        date, ds = runs.at[i, "date"], runs.at[i, "dataset"]
        start = pd.Timestamp(runs.at[i, "scheduled_at"])
        end = start + pd.Timedelta(minutes=dur)
        runs.at[i, "failed"] = True
        runs.at[i, "incident"] = kind
        runs.at[i, "status"] = "failed"
        runs.at[i, "duration_min"] = dur
        runs.at[i, "started_at"] = start.strftime("%Y-%m-%d %H:%M:%S")
        runs.at[i, "ended_at"] = end.strftime("%Y-%m-%d %H:%M:%S")
        if pd.Timestamp(date) >= TEST_START:
            runs.at[i, "notice_delay_min"] = 19.0
        n_extra += 1
print(f"consistency pass added {n_extra} gateway-wide failures")

out = runs.copy()

out["failed"] = out.failed.astype(int)
out["size_vs_28d"] = out.size_vs_28d.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.3, 6)
out["duration_trend"] = out.duration_trend.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.5, 3)
out = out.rename(columns={"online": "gateway_online"})
out.loc[out.status != "failed", "notice_delay_min"] = np.nan

hist_cols = [
    "date", "dataset", "workspace", "scheduled_at", "started_at", "ended_at",
    "status", "duration_min", "size_mb", "credential_age_days",
    "gateway_online", "mem_now", "mem_mean_1h", "mem_max_1h", "mem_over_90_min",
    "cpu_mean_1h", "capacity_mean_1h", "concurrent_running", "queued_next_90m",
    "size_vs_28d", "duration_trend", "failed_yesterday", "is_monday", "is_month_end",
    "notice_delay_min",
]
out[hist_cols].to_csv(os.path.join(OUT, "refresh_history.csv"), index=False)

hb_out = hb[["ts", "gateway_mem_pct", "gateway_cpu_pct", "capacity_pct", "gateway_online"]].copy()
hb_out["ts"] = hb_out.ts.dt.strftime("%Y-%m-%d %H:%M:%S")
hb_out.to_csv(os.path.join(OUT, "gateway_heartbeat.csv"), index=False)
close.to_csv(os.path.join(OUT, "close_calendar.csv"), index=False)
log = out.loc[out.status == "failed", ["date", "dataset", "workspace"]].copy()
log["incident"] = out.loc[out.status == "failed", "incident"].values
log.to_csv(os.path.join(OUT, "incident_log.csv"), index=False)

n_test_fail = int(((out.date >= TEST_START.date().isoformat()) & (out.status == "failed")).sum())
n_train_fail = int(((out.date <= TRAIN_END.date().isoformat()) & (out.status == "failed")).sum())
n_test = int((out.date >= TEST_START.date().isoformat()).sum())
print(
    f"SYNTHETIC data written to data/: {len(out):,} scheduled runs "
    f"({out.date.min()} to {out.date.max()}), {out.dataset.nunique()} datasets, "
    f"train failures {n_train_fail}, test failures {n_test_fail} / {n_test} runs"
)
h = out[(out.date == HOOK.date().isoformat()) & (out.dataset == "City Ops")].iloc[0]
print(
    f"HOOK City Ops: status={h.status} duration={h.duration_min} end={h.ended_at} "
    f"mem_now={h.mem_now:.1f} mem_over_90_min={h.mem_over_90_min} "
    f"concurrent={h.concurrent_running} queued={h.queued_next_90m} size={h.size_mb}"
)
