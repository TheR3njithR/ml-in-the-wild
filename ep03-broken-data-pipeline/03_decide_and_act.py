"""Step 3: the morning risk report — a decision, not a score."""
import os, sys
import pandas as pd
import pipeline_guard as pg

day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-03-16")
hist = pd.read_csv(os.path.join(pg.DATA, "pipeline_daily.csv"), parse_dates=["date"])
lab = pd.read_csv(os.path.join(pg.DATA, "incident_labels.csv"), parse_dates=["date"])
df = hist.merge(lab, on=["date", "pipeline"], how="left")
train, _ = pg.split(df)
model = pg.fit_expected(train)
tr = train.copy()
tr["expected"] = pg.expected_rows(model, tr)
tr["ratio"] = pg.residual_ratio(tr["usable_rows"], tr["expected"])
amber, red = pg.threshold_from_train(tr)
scored = pg.score_frame(df[df.date == day].copy(), model, amber, red).sort_values("pipeline")

lines = []
lines.append(f"Pipeline volume report for {day.date()}, after the morning jobs (QuickBasket, synthetic data)")
lines.append("")
for _, r in scored.iterrows():
    reason = {
        "GREEN": "volume matches the expected pattern",
        "AMBER": "volume is off the usual pattern; check before you publish",
        "RED": "volume is far from the expected pattern, or sources did not all arrive",
    }[r.tier]
    if r.pipeline == "orders_facts" and r.tier == "RED":
        reason = (f"usable rows {int(r.usable_rows):,} vs expected {r.expected:,.0f} "
                  f"({r.ratio*100:.0f}% of pattern); sources {int(r.sources_arrived)}/"
                  f"{int(r.sources_expected)}; job status {r.status}")
    lines.append(
        f"{r.pipeline:20s}  {r.tier:5s}  ratio={r.ratio:4.2f}  "
        f"usable {int(r.usable_rows):7,}  expected {r.expected:7,.0f}  "
        f"sources {int(r.sources_arrived)}/{int(r.sources_expected)}  {reason}"
    )
reds = scored[scored.tier == "RED"]
ambers = scored[scored.tier == "AMBER"]
lines.append("")
if len(reds):
    lines.append("Decision: HOLD the dashboards that depend on RED pipelines; do not publish a business drop until the sources are checked. Page the data engineer.")
    focus = reds.iloc[0]
    lines.append(f"Focus, {focus.pipeline}: {focus.tier} (ratio {focus.ratio:.2f}).")
    lines.append("Action: keep the 9:00 meeting on the last good Orders numbers, inspect the zone file arrivals, and put a banner on the dashboard.")
elif len(ambers):
    lines.append("Decision: publish with a banner on AMBER pipelines; a person checks them after the meeting.")
else:
    lines.append("Decision: publish. Every pipeline matched its expected volume pattern.")

text = "\n".join(lines) + "\n"
os.makedirs(pg.OUTS, exist_ok=True)
open(os.path.join(pg.OUTS, "03_trust_report.txt"), "w").write(text)
scored.to_csv(os.path.join(pg.OUTS, "03_morning_scored.csv"), index=False)
print(text)
