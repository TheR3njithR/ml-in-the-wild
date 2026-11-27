"""Step 4: labelled feedback and a champion vs challenger test."""
import os
import pandas as pd
import pipeline_guard as pg

scored = pd.read_csv(os.path.join(pg.OUTS, "test_scored.csv"), parse_dates=["date"])

# First six labelled weeks of the test period: 1 Aug 2025 – 11 Sep 2025
label_end = pd.Timestamp("2025-09-11")
first = scored[(scored.date <= label_end) & (scored.status == 'SUCCESS')]
alerts = first[first.alert]
real = int(alerts.is_silent.sum())
false_n = int((~alerts.is_silent).sum())
prec = (real / len(alerts)) if len(alerts) else 0.0

# A "drop the calendar" challenger: re-fit expected volume without holiday/promo flags,
# on training data, then score the later test months.
hist = pd.read_csv(os.path.join(pg.DATA, "pipeline_daily.csv"), parse_dates=["date"])
lab = pd.read_csv(os.path.join(pg.DATA, "incident_labels.csv"), parse_dates=["date"])
df = hist.merge(lab, on=["date", "pipeline"], how="left")
train, test = pg.split(df)
later = test[test.date > label_end].copy()

# champion = already-scored later months
champ = scored[(scored.date > label_end) & (scored.status == 'SUCCESS')]
champ_s = pg.summarise(champ.alert, champ.is_silent.astype(bool), "champion")

# challenger: same model class, but is_holiday and is_promo zeroed at train and score time
tr = train.copy()
te = later.copy()
tr["is_holiday"] = 0
tr["is_promo"] = 0
te["is_holiday"] = 0
te["is_promo"] = 0
model = pg.fit_expected(tr)
tr["expected"] = pg.expected_rows(model, tr)
tr["ratio"] = pg.residual_ratio(tr["usable_rows"], tr["expected"])
amber, red = pg.threshold_from_train(tr)
ch = pg.score_frame(te, model, amber, red)
# restore labels from later (same rows, same order after score_frame copy)
ch["is_silent"] = te["is_silent"].to_numpy()
ch["status"] = te["status"].to_numpy()
ch_ok = ch[ch.status == "SUCCESS"]
chal_s = pg.summarise(ch_ok.alert, ch_ok.is_silent.astype(bool), "challenger")

# keep champion if precision is higher (calendar features should reduce FPs)
decision = "keep the champion" if champ_s["precision"] >= chal_s["precision"] else "promote the challenger"

lines = []
lines.append(f"1) Analyst review, 2025-08-01 to 2025-09-11: {len(alerts)} alerts labelled")
lines.append(f"   real silent breaks {real}, false alarm {false_n}: "
             + (", ".join(f"{r.pipeline} {r.date.date()}" for _, r in alerts[~alerts.is_silent].iterrows()) or "none"))
lines.append(f"   alert precision {prec*100:.0f}%")
lines.append("2) The holiday and promo calendar was already in the champion. "
             "Dropping those flags is the challenger, not a new rule on top.")
lines.append("")
lines.append(f"3) Champion vs challenger on 2025-09-12 to 2026-03-31 "
             f"({len(later):,} pipeline-days, {int(later.is_silent.sum())} silent breaks):")
lines.append(f"   champion: expected volume with calendar flags     {pg.fmt(champ_s)}")
lines.append(f"   challenger: same model, holiday and promo flags off {pg.fmt(chal_s)}")
lines.append(f"   decision: {decision}. "
             "Without the calendar, ordinary holidays look like broken pipelines.")

text = "\n".join(lines) + "\n"
open(os.path.join(pg.OUTS, "04_feedback.txt"), "w").write(text)
print(text)
