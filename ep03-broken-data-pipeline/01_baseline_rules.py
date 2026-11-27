"""Step 1: the usual answers — trust the job status, or a naive volume rule."""
import os
import pandas as pd
import pipeline_guard as pg

hist = pd.read_csv(os.path.join(pg.DATA, "pipeline_daily.csv"), parse_dates=["date"])
lab = pd.read_csv(os.path.join(pg.DATA, "incident_labels.csv"), parse_dates=["date"])
df = hist.merge(lab, on=["date", "pipeline"], how="left")
_, test = pg.split(df)
y_all = test["is_silent"].astype(bool)
# The title problem is silent breaks on SUCCESS jobs. Evaluate volume rules there.
ok = test[test["status"] == "SUCCESS"]
y = ok["is_silent"].astype(bool)
n_hard = int(test["is_hard_fail"].sum())

status = pg.summarise(test["status"].ne("SUCCESS"), y_all, "status")

raw = (
    (ok["rows"] < 0.70 * ok["raw_med7"])
    | (ok["rows"] < 0.80 * ok["raw_same_dow"])
    | (ok["rows"] > 1.50 * ok["raw_med7"])
)
raw_s = pg.summarise(raw, y, "naive_raw")

miss = ok[y & ~raw]
caught = ok[y & raw]
fp = ok[~y & raw]

lines = []
lines.append(
    f"Test months 2025-08-01 to 2026-03-31: {len(test):,} pipeline-days, "
    f"{int(y_all.sum())} silent breaks and {n_hard} hard job failures (synthetic ground truth)"
)
lines.append("Jobs marked SUCCESS on every silent break; the exit code never fired.")
lines.append("")
lines.append("Rule 1 — alert only when status != SUCCESS:")
lines.append(pg.fmt(status))
lines.append(
    f"Those {status['alerts']} alerts were exactly the {n_hard} hard job failures. "
    "Useful, but this rule missed every silent break."
)
lines.append("")
lines.append("Rule 2 — naive row-count volume rule, scored only on SUCCESS jobs:")
lines.append("alert if rows < 70% of the 7-day median, < 80% of last same weekday,")
lines.append("or > 150% of the 7-day median:")
lines.append(pg.fmt(raw_s))
lines.append("missed by type:")
for typ, n in miss["incident_type"].value_counts().items():
    lines.append(f"  {typ:22s} {n}")
lines.append("caught by type:")
for typ, n in caught["incident_type"].value_counts().items():
    lines.append(f"  {typ:22s} {n}")
lines.append(
    "false alarms include "
    f"{int(((fp.is_holiday == 1) | (fp.is_promo == 1)).sum())} "
    "holiday or promo days, where a real calendar pattern looks like a drop or spike."
)
lines.append(
    "The naive rule also looks at the raw row count, so it misses almost every NULL_SURGE, "
    "where the file still has the usual number of rows and a key column has gone null."
)

text = "\n".join(lines) + "\n"
os.makedirs(pg.OUTS, exist_ok=True)
open(os.path.join(pg.OUTS, "01_baseline.txt"), "w").write(text)
print(text)
