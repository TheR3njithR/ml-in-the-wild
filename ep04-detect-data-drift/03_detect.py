#!/usr/bin/env python3
"""Daily drift monitor: PSI and KS on each feature, plus a domain-classifier AUC.

For morning T we compare the last 14 complete feature-days to the training
distribution. Labels are not used. PSI uses three bins. Textbook PSI cuts of
0.10 / 0.25 assume hundreds of rows, so we also compute a small-sample null:
the 90th and 95th percentiles of 14-day PSI values drawn from training. A
feature is amber or red only when it beats that null. Domain-classifier AUC
is the multivariate check (amber 0.70, red 0.80).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import drift_tools as D
from model_common import DRIFT_START, HOOK_DAY, TEST_END, TRAIN_END

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

WINDOW = 14
BINS = 3
FEATS = list(D.MONITOR_FEATURES)


def status_of(n_red, n_amber, auc):
    """Red is a hold-the-orders call. Amber is a look.

    A single feature beating its 95th null is not enough for red: that will
    happen on a quiet week by chance. Red needs a multivariate move (domain
    AUC at least 0.80), three features at once, or two features plus an amber
    domain AUC.
    """
    if (auc >= D.AUC_RED) or (n_red >= 3) or (n_red >= 2 and auc >= D.AUC_AMBER):
        return "RED"
    if (auc >= D.AUC_AMBER) or (n_red >= 2):
        return "AMBER"
    return "GREEN"


def main():
    df = pd.read_csv(OUT / "02_scored_daily.csv", parse_dates=["date"])
    train = df[(df["date"] <= TRAIN_END) & df[FEATS].notna().all(axis=1)]

    nulls = {f: D.psi_null_quantiles(train[f].values, window=WINDOW, bins=BINS) for f in FEATS}

    recs = []
    prod_days = df.loc[df["date"] > TEST_END, "date"]
    for day in prod_days:
        recent = df[(df["date"] <= day) & (df["date"] > day - pd.Timedelta(days=WINDOW)) & df[FEATS].notna().all(axis=1)]
        if len(recent) < WINDOW:
            continue
        feat_psi, feat_ks = {}, {}
        n_red = n_amber = 0
        top, top_psi = None, -1
        for f in FEATS:
            feat_psi[f] = D.psi(train[f].values, recent[f].values, bins=BINS)
            feat_ks[f] = D.ks_stat(train[f].values, recent[f].values)
            if feat_psi[f] > top_psi:
                top, top_psi = f, feat_psi[f]
            # A previously constant feature (null p95 == 0) only flags when it actually moved.
            p95, p90 = nulls[f]["p95"], nulls[f]["p90"]
            if p95 <= 0:
                if feat_psi[f] > 0:
                    n_red += 1
            elif feat_psi[f] >= p95:
                n_red += 1
            elif feat_psi[f] >= max(p90, 1e-9):
                n_amber += 1
        auc = D.domain_auc(train[FEATS], recent[FEATS])
        max_ks = max(feat_ks.values())
        # How far is the top PSI above its own 95th null (1.0 = at the red cut).
        ratio = top_psi / max(nulls[top]["p95"], 1e-6)
        status = status_of(n_red, n_amber, auc)
        recs.append({
            "date": day, "n_window": int(len(recent)),
            "max_psi": top_psi, "top_feature": top, "top_psi": top_psi,
            "max_ks": max_ks, "domain_auc": auc, "status": status,
            "n_red_features": n_red, "n_amber_features": n_amber,
            **{f"psi_{f}": feat_psi[f] for f in FEATS},
            **{f"ks_{f}": feat_ks[f] for f in FEATS},
        })
    mon = pd.DataFrame(recs)

    first_amber = mon.loc[mon["status"].isin(["AMBER", "RED"]), "date"].min()
    first_red = mon.loc[mon["status"] == "RED", "date"].min()
    base = pd.read_csv(OUT / "01_baseline_daily.csv", parse_dates=["date"])
    mape_alert = base.loc[base["alert"] == 1, "date"].min()
    if pd.isna(mape_alert):
        mape_alert = None

    hook = mon.loc[mon["date"] == HOOK_DAY].iloc[0]
    after = mon[mon["date"] >= DRIFT_START]
    before = mon[mon["date"] < DRIFT_START]
    tp = int((after["status"].isin(["AMBER", "RED"])).sum())
    fn = int((after["status"] == "GREEN").sum())
    fp = int((before["status"].isin(["AMBER", "RED"])).sum())
    tn = int((before["status"] == "GREEN").sum())
    fp_red = int((before["status"] == "RED").sum())

    def lag(ts):
        return None if ts is None or pd.isna(ts) else int((pd.Timestamp(ts) - DRIFT_START).days)

    lines = [
        "DRIFT MONITOR  (14-day feature window vs training; no labels)",
        f"Features watched: {', '.join(FEATS)}.",
        "Textbook PSI cuts are 0.10 / 0.25, but those assume hundreds of rows.",
        "We flag a feature only when its 14-day, 3-bin PSI beats a training null:",
    ]
    for f in FEATS:
        q = nulls[f]
        lines.append(f"  {f:20s}  null p90 {q['p90']:.2f}  p95 {q['p95']:.2f}  p99 {q['p99']:.2f}")
    lines.append(
        f"Domain AUC amber {D.AUC_AMBER:.2f} red {D.AUC_RED:.2f}.  "
        f"K S amber {D.KS_AMBER:.2f} red {D.KS_RED:.2f}."
    )
    lines.append(f"Planted shift starts {DRIFT_START.date()}.")
    lines.append(f"FIRST AMBER {None if pd.isna(first_amber) else first_amber.date()}  (lag {lag(first_amber)} days).")
    lines.append(f"FIRST RED   {None if pd.isna(first_red) else first_red.date()}  (lag {lag(first_red)} days).")
    lines.append(
        f"FIRST DELAYED-MAPE ALERT {None if mape_alert is None else pd.Timestamp(mape_alert).date()}  "
        f"(lag {lag(mape_alert)} days)."
    )
    lines.append(
        f"HOOK {HOOK_DAY.date()}: status {hook['status']}  max PSI {hook['max_psi']:.2f} on {hook['top_feature']}  "
        f"max K S {hook['max_ks']:.2f}  domain AUC {hook['domain_auc']:.2f}."
    )
    lines.append("Hook feature PSI:")
    for f in FEATS:
        lines.append(f"  {f:20s}  PSI {hook[f'psi_{f}']:.2f}  K S {hook[f'ks_{f}']:.2f}")
    if not pd.isna(first_red) and mape_alert is not None:
        lead = (pd.Timestamp(mape_alert) - pd.Timestamp(first_red)).days
        lines.append(f"LEAD TIME: the red flag fired {lead} days before the delayed MAPE alert.")
    lines.append(
        f"From the shift to the end of the file: amber-or-red on {tp} of {tp+fn} planted days "
        f"(missed {fn}); false amber-or-red on {fp} of {fp+tn} pre-shift production days "
        f"(false red {fp_red})."
    )
    text = "\n".join(lines) + "\n"
    (OUT / "03_detect.txt").write_text(text)
    mon.to_csv(OUT / "03_drift_daily.csv", index=False)
    # store nulls for the report / charts
    pd.DataFrame(nulls).T.rename_axis("feature").reset_index().to_csv(OUT / "03_psi_nulls.csv", index=False)
    print(text, end="")


if __name__ == "__main__":
    main()
