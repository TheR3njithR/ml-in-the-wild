"""Generate the synthetic pipeline history for ML in the Wild, Episode 3.

QuickBasket (fictional) runs eight daily data pipelines that feed the morning
dashboards. Almost every job finishes with status SUCCESS. The silent breaks
live in the *volume pattern*, not in the exit code.

The model never sees incident_log.csv at prediction time.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

import pipeline_guard as pg

SEED = 33
START = date(2024, 12, 1)
END = date(2026, 3, 31)
HOOK = date(2026, 3, 16)
HOOK_PIPE = "orders_facts"

# Eight pipelines that land before the 09:00 dashboards.
# base = typical Monday-Thursday usable rows.
PIPELINES = [
    # name, base, n_sources, weekend, holiday, promo, month_end
    ("orders_facts", 48_000, 8, 1.22, 0.55, 1.30, 1.04),
    ("inventory_snapshot", 12_000, 3, 0.98, 0.90, 1.00, 1.15),
    ("delivery_events", 45_000, 8, 1.18, 0.58, 1.22, 1.03),
    ("customer_signups", 820, 2, 1.10, 0.50, 1.40, 1.00),
    ("payments_ledger", 50_000, 4, 1.15, 0.60, 1.20, 1.18),
    ("marketing_spend", 240, 2, 0.70, 0.40, 1.80, 1.00),
    ("warehouse_picks", 30_000, 5, 0.55, 0.45, 1.15, 1.08),
    ("support_tickets", 460, 1, 0.75, 0.70, 1.10, 1.00),
]
# weekday multipliers Mon..Sun relative to the mid-week base (already in PIPELINES for weekend)
DOW = {0: 1.00, 1: 0.96, 2: 0.95, 3: 0.98, 4: 1.08, 5: 1.00, 6: 0.92}

# Zone volume shares for the orders_facts hook (three busiest sum to 0.41).
ZONE_SHARES = np.array([0.16, 0.14, 0.11, 0.13, 0.12, 0.10, 0.14, 0.10], dtype=float)
ZONE_NAMES = ["DXB-C", "DXB-N", "DXB-S", "AUH-C", "AUH-N", "SHJ", "AJM", "RAK"]
# Drop DXB-C, DXB-N, AJM = 0.16+0.14+0.14 = 0.44? That's 44%. Want ~41%.
# 0.16+0.14+0.11 = 0.41 (DXB-C, DXB-N, DXB-S). Good.
HOOK_MISSING = [0, 1, 2]

HOLIDAYS = set(pd.to_datetime([
    # UAE National Day and a few observed closures
    "2024-12-01", "2024-12-02", "2024-12-03",
    "2024-12-25", "2025-12-25", "2025-12-26",
    "2025-01-01", "2026-01-01",
    "2025-12-01", "2025-12-02", "2025-12-03",
    # Eid al-Fitr approximations
    "2025-03-30", "2025-03-31", "2025-04-01",
    "2026-03-20", "2026-03-21", "2026-03-22",
    # Eid al-Adha approximations
    "2025-06-06", "2025-06-07", "2025-06-08",
]).date)

# Promo bursts: orders and signups jump, then the next ordinary day looks like a "drop" to naive rules.
PROMOS = set()
for s in ["2025-01-16", "2025-02-13", "2025-05-08", "2025-08-14", "2025-11-13",
          "2025-12-18", "2026-01-15", "2026-02-12", "2025-09-11", "2026-03-05"]:
    d0 = date.fromisoformat(s)
    for k in range(3):
        PROMOS.add(d0 + timedelta(days=k))


def daterange(a, b):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def month_end(d: date) -> bool:
    return (d + timedelta(days=1)).month != d.month or (d + timedelta(days=2)).month != d.month


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(pg.DATA, exist_ok=True)

    # --- expected (healthy) daily volume ---------------------------------
    rows = []
    for d in daterange(START, END):
        is_hol = d in HOLIDAYS
        is_promo = d in PROMOS
        is_we = d.weekday() >= 5
        is_me = month_end(d)
        for i, (name, base, nsrc, we_m, hol_m, promo_m, me_m) in enumerate(PIPELINES):
            vol = base * DOW[d.weekday()]
            if is_we:
                vol *= we_m
            if is_hol:
                vol *= hol_m
            if is_promo:
                vol *= promo_m
            if is_me:
                vol *= me_m
            # Tight noise so a 41% hook shortfall cannot hide in ordinary variation.
            vol *= float(rng.lognormal(0, 0.035))
            rows.append({
                "date": d,
                "pipeline": name,
                "pipeline_id": i,
                "dow": d.weekday(),
                "week_of_month": (d.day - 1) // 7 + 1,
                "month": d.month,
                "is_weekend": int(is_we),
                "is_month_end": int(is_me),
                "is_holiday": int(is_hol),
                "is_promo": int(is_promo),
                "n_sources": nsrc,
                "healthy_rows": int(round(vol)),
            })
    df = pd.DataFrame(rows)

    # Stabilise the hook Monday: keep the pre-hook same-dow history close to the
    # typical Monday so lag features do not warp expected volume (cf. Ep2 hook).
    hook_mask = (df.pipeline == HOOK_PIPE) & (df.date == HOOK)
    hook_typical = int(df[(df.pipeline == HOOK_PIPE) & (df.dow == 0) & (df.is_holiday == 0)
                 & (df.is_promo == 0) & (df.date < HOOK) & (df.date > date(2026, 1, 1))]
              ["healthy_rows"].median())
    df.loc[hook_mask, "healthy_rows"] = hook_typical
    # previous 8 Mondays of orders_facts: pull toward typ so same_dow_median is stable
    mondays = df[(df.pipeline == HOOK_PIPE) & (df.dow == 0) & (df.date < HOOK) & (df.date >= HOOK - timedelta(days=56))]
    for idx in mondays.index:
        if df.loc[idx, "is_holiday"] or df.loc[idx, "is_promo"]:
            continue
        df.loc[idx, "healthy_rows"] = int(round(0.4 * df.loc[idx, "healthy_rows"] + 0.6 * hook_typical))

    df["rows"] = df["healthy_rows"]
    df["usable_rows"] = df["rows"]
    df["null_rate"] = rng.uniform(0.004, 0.018, len(df))
    df["sources_expected"] = df["n_sources"]
    df["sources_arrived"] = df["n_sources"]
    df["status"] = "SUCCESS"
    df["runtime_s"] = (8 + df["rows"] / 1800 * rng.lognormal(0, 0.08, len(df))).astype(int)
    df["is_silent"] = False
    df["is_hard_fail"] = False
    df["incident_type"] = ""

    incidents = []  # rows for incident_log

    def apply_break(idx, typ, *, row_mult=1.0, usable_mult=None, arrived=None, status="SUCCESS"):
        usable_mult = row_mult if usable_mult is None else usable_mult
        rec = df.loc[idx]
        if rec["is_silent"] or rec["is_hard_fail"] or rec["status"] != "SUCCESS":
            return False
        new_rows = max(0, int(round(rec["rows"] * row_mult)))
        new_usable = max(0, int(round(rec["healthy_rows"] * usable_mult)))
        if usable_mult < 1 and row_mult == 1.0:
            new_usable = max(0, int(round(rec["rows"] * usable_mult)))
        arrived_n = rec["sources_arrived"] if arrived is None else arrived
        df.at[idx, "rows"] = new_rows
        df.at[idx, "usable_rows"] = min(new_rows, new_usable) if typ != "NULL_SURGE" else new_usable
        if typ == "NULL_SURGE":
            df.at[idx, "rows"] = rec["rows"]
            df.at[idx, "usable_rows"] = new_usable
            df.at[idx, "null_rate"] = max(0.0, 1.0 - (new_usable / max(rec["rows"], 1)))
        df.at[idx, "sources_arrived"] = int(arrived_n)
        df.at[idx, "status"] = status
        df.at[idx, "runtime_s"] = max(4, int(rec["runtime_s"] * (0.65 if row_mult < 0.8 else 1.15 if row_mult > 1.4 else 1.0)))
        silent = typ != "HARD_FAIL"
        df.at[idx, "is_silent"] = silent
        df.at[idx, "is_hard_fail"] = not silent
        df.at[idx, "incident_type"] = typ
        incidents.append({
            "date": rec["date"],
            "pipeline": rec["pipeline"],
            "incident_type": typ,
            "status": status,
            "healthy_rows": int(rec["healthy_rows"]),
            "rows": int(df.at[idx, "rows"]),
            "usable_rows": int(df.at[idx, "usable_rows"]),
            "sources_arrived": int(df.at[idx, "sources_arrived"]),
            "sources_expected": int(rec["sources_expected"]),
            "note": typ,
        })
        return True

    # ---- Hook: 16 Mar 2026, orders_facts, three city zones silent, job green
    hook_idx = df.index[(df.pipeline == HOOK_PIPE) & (df.date == HOOK)][0]
    hook_share = float(ZONE_SHARES[HOOK_MISSING].sum())
    apply_break(
        hook_idx, "SOURCE_SILENT",
        row_mult=1.0 - hook_share,
        arrived=8 - len(HOOK_MISSING),
    )
    incidents[-1]["note"] = (
        "Three city zone extracts (DXB-C, DXB-N, DXB-S) returned empty files. "
        "The orchestrator still marked the job SUCCESS."
    )

    # ---- Plant other silent breaks (never on the hook cell; skip holidays for most)
    plan = [
        # (type, count_train, count_test, row_mult, usable_mult, arrived_fn, allow_holiday)
        ("SOURCE_SILENT", 8, 16, 0.59, 0.59, lambda n: max(1, n - 3), False),
        ("SOURCE_SILENT_MILD", 6, 12, 0.86, 0.86, lambda n: max(1, n - 1), False),
        ("FILTER_REGRESSION", 8, 16, 0.55, 0.55, None, False),
        ("LATE_ARRIVAL", 8, 18, 0.83, 0.83, None, False),
        ("DUP_LOAD", 6, 12, 2.02, 2.02, None, False),
        ("NULL_SURGE", 7, 14, 1.00, 0.42, None, False),
        ("TRUNCATED_EXTRACT", 7, 14, 0.50, 0.50, None, False),
        ("JOIN_COLLAPSE", 5, 10, 0.28, 0.28, None, False),
        ("HARD_FAIL", 10, 16, 0.00, 0.00, None, True),
    ]

    def pool(before_hook_end, typ_allow_hol):
        m = (~df["is_silent"]) & (~df["is_hard_fail"])
        m &= df["date"] != HOOK  # keep the hook morning clean for every pipeline
        m &= ~((df["pipeline"] == HOOK_PIPE) & (df["date"] == HOOK))
        if before_hook_end == "train":
            m &= df["date"] <= date(2025, 7, 31)
        else:
            m &= df["date"] > date(2025, 7, 31)
        if not typ_allow_hol:
            m &= df["is_holiday"] == 0
        return df.index[m].to_numpy()

    for typ, n_tr, n_te, rmul, umul, arr_fn, allow_hol in plan:
        kind = "HARD_FAIL" if typ == "HARD_FAIL" else typ.replace("_MILD", "")
        for split, n in (("train", n_tr), ("test", n_te)):
            cand = np.array(pool(split, allow_hol), copy=True)
            rng.shuffle(cand)
            placed = 0
            for idx in cand:
                if placed >= n:
                    break
                nsrc = int(df.at[idx, "n_sources"])
                arrived = arr_fn(nsrc) if arr_fn else None
                status = "FAILED" if kind == "HARD_FAIL" else "SUCCESS"
                # HARD_FAIL: keep a log-visible failure, not a silent break
                ok = apply_break(
                    idx, kind,
                    row_mult=rmul, usable_mult=umul, arrived=arrived, status=status,
                )
                if ok:
                    placed += 1
            if placed < n:
                print(f"WARN only placed {placed}/{n} {typ} in {split}")

    df["source_fill"] = df["sources_arrived"] / df["sources_expected"].clip(lower=1)
    # usable cannot exceed rows except we already handled NULL_SURGE (usable < rows)
    df.loc[df.incident_type != "NULL_SURGE", "usable_rows"] = np.minimum(
        df.loc[df.incident_type != "NULL_SURGE", "usable_rows"],
        df.loc[df.incident_type != "NULL_SURGE", "rows"],
    )
    df = pg.add_lags(df)

    # naive-rule helpers (raw rows vs recent medians) — computed here so baselines stay honest
    df = df.sort_values(["pipeline", "date"]).reset_index(drop=True)
    parts = []
    for _, g in df.groupby("pipeline", sort=False):
        g = g.copy()
        g["raw_med7"] = g["rows"].shift(1).rolling(7, min_periods=4).median()
        g["raw_same_dow"] = g["rows"].shift(7)
        g["usable_med7"] = g["usable_rows"].shift(1).rolling(7, min_periods=4).median()
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    fillr = df.groupby("pipeline")["rows"].transform("median")
    fillu = df.groupby("pipeline")["usable_rows"].transform("median")
    df["raw_med7"] = df["raw_med7"].fillna(fillr)
    df["raw_same_dow"] = df["raw_same_dow"].fillna(fillr)
    df["usable_med7"] = df["usable_med7"].fillna(fillu)

    # persist
    daily_cols = [
        "date", "pipeline", "pipeline_id", "dow", "week_of_month", "month",
        "is_weekend", "is_month_end", "is_holiday", "is_promo",
        "status", "rows", "usable_rows", "null_rate",
        "sources_expected", "sources_arrived", "source_fill", "runtime_s",
        "lag_7", "lag_14", "same_dow_median_8w",
        "raw_med7", "raw_same_dow", "usable_med7",
    ]
    # do not write labels into the model-facing history file
    hist = df[daily_cols].copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.strftime("%Y-%m-%d")
    hist.to_csv(os.path.join(pg.DATA, "pipeline_daily.csv"), index=False)

    inc = pd.DataFrame(incidents)
    inc["date"] = pd.to_datetime(inc["date"]).dt.strftime("%Y-%m-%d")
    inc.to_csv(os.path.join(pg.DATA, "incident_log.csv"), index=False)

    # labels file (answer key; the model does not read this at prediction time)
    lab = df[["date", "pipeline", "is_silent", "is_hard_fail", "incident_type", "healthy_rows"]].copy()
    lab["date"] = pd.to_datetime(lab["date"]).dt.strftime("%Y-%m-%d")
    lab.to_csv(os.path.join(pg.DATA, "incident_labels.csv"), index=False)

    cal = []
    for d in daterange(START, END):
        cal.append({"date": d.isoformat(), "is_holiday": int(d in HOLIDAYS), "is_promo": int(d in PROMOS)})
    pd.DataFrame(cal).to_csv(os.path.join(pg.DATA, "calendar.csv"), index=False)

    # Hook-morning timeline (illustrative cumulative load, labelled SYNTHETIC / ILLUSTRATIVE in the video)
    healthy = hook_typical
    arrived_share = 1.0 - hook_share
    # files land 05:10-06:05; missing zones never arrive; job "succeeds" 06:12
    times, cum = [], []
    t0 = datetime(2026, 3, 16, 5, 0)
    landed = 0.0
    present = [i for i in range(8) if i not in HOOK_MISSING]
    # stagger present zones
    land_at = {z: datetime(2026, 3, 16, 5, 10) + timedelta(minutes=8 * k) for k, z in enumerate(present)}
    for m in range(0, 241, 1):  # 05:00 to 09:00
        ts = t0 + timedelta(minutes=m)
        for z, at in land_at.items():
            if ts >= at and ts < at + timedelta(minutes=1):
                landed += ZONE_SHARES[z]
        times.append(ts)
        cum.append(int(round(healthy * landed)))
    hook_tl = pd.DataFrame({
        "ts": times,
        "rows_landed": cum,
        "expected_if_all_zones": healthy,
        "job_success_at": "2026-03-16 06:12",
        "meeting_at": "2026-03-16 09:00",
    })
    hook_tl.to_csv(os.path.join(pg.DATA, "hook_timeline.csv"), index=False)

    # size check
    print("wrote data/")
    for fn in sorted(os.listdir(pg.DATA)):
        p = os.path.join(pg.DATA, fn)
        mb = os.path.getsize(p) / 1e6
        print(f"  {fn:24s} {mb:.2f} MB")
        if mb > 5:
            print("    WARNING: over 5 MB")
    n = len(df)
    print(f"rows {n}: silent {int(df.is_silent.sum())}, hard fails {int(df.is_hard_fail.sum())}")
    hook = df[(df.pipeline == HOOK_PIPE) & (df.date == HOOK)].iloc[0]
    print("HOOK", hook["rows"], "usable", hook["usable_rows"], "healthy", hook["healthy_rows"],
          "ratio", hook["usable_rows"] / hook["healthy_rows"],
          "arrived", hook["sources_arrived"], "status", hook["status"])


if __name__ == "__main__":
    main()
