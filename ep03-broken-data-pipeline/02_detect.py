"""Step 2: expected-volume residual detector (volume-pattern anomaly detection)."""
import os
import pandas as pd
import pipeline_guard as pg

hist = pd.read_csv(os.path.join(pg.DATA, "pipeline_daily.csv"), parse_dates=["date"])
lab = pd.read_csv(os.path.join(pg.DATA, "incident_labels.csv"), parse_dates=["date"])
df = hist.merge(lab, on=["date", "pipeline"], how="left")
train, test = pg.split(df)

model = pg.fit_expected(train)
train = train.copy()
train["expected"] = pg.expected_rows(model, train)
train["ratio"] = pg.residual_ratio(train["usable_rows"], train["expected"])
amber, red = pg.threshold_from_train(train)

scored_tr = pg.score_frame(train, model, amber, red)
scored_te = pg.score_frame(test, model, amber, red)

# Evaluate the silent-break task on SUCCESS jobs only. Hard failures already
# show up in the status column; counting them as "false alarms" would hide
# how the volume pattern behaves on the days the logs called fine.
ok = scored_te[scored_te["status"] == "SUCCESS"]
oracle = pg.summarise(
    (ok["usable_rows"] / ok["healthy_rows"] <= 0.80)
    | (ok["usable_rows"] / ok["healthy_rows"] >= 1.25),
    ok["is_silent"].astype(bool),
    "oracle",
)
final = pg.summarise(ok["alert"], ok["is_silent"].astype(bool), "pattern")
reds = pg.summarise(ok["red_alert"], ok["is_silent"].astype(bool), "red")

os.makedirs(pg.OUTS, exist_ok=True)
keep = [
    "date", "pipeline", "status", "rows", "usable_rows", "expected", "ratio",
    "source_fill", "tier", "alert", "red_alert", "is_silent", "is_hard_fail",
    "incident_type", "is_holiday", "is_promo", "null_rate", "sources_arrived",
    "sources_expected", "runtime_s",
]
scored_te[keep].to_csv(os.path.join(pg.OUTS, "test_scored.csv"), index=False)

hook = scored_te[(scored_te.pipeline == "orders_facts") & (scored_te.date == pg.HOOK_DATE)].iloc[0]
miss = ok[ok.is_silent & ~ok.alert]
caught = ok[ok.is_silent & ok.alert]
fp = ok[~ok.is_silent & ok.alert]
red_fp = ok[~ok.is_silent & ok.red_alert]

lines = []
lines.append(
    f"trained on {len(train):,} pipeline-days (2024-12-01 to 2025-07-31); "
    f"tested on {len(test):,} pipeline-days, {int(test.is_silent.sum())} silent breaks"
)
lines.append("every test feature is taken from the finished job and the calendar — never from the log text")
lines.append("silent-break scores below are computed on SUCCESS jobs only")
lines.append("")
lines.append("[oracle, uses hidden healthy volume — NOT AVAILABLE IN PRODUCTION] " + pg.fmt(oracle))
lines.append("  this looks too good: nobody has a file that says what the row count *should* have been")
lines.append("")
lines.append("[final, expected-volume residual + source-fill check]")
lines.append("  expected usable rows = last-8 same weekdays, adjusted by a small gradient boosted")
lines.append("  regressor that learns holiday, promo and month-end multipliers")
lines.append(f"  thresholds from training ratios only: AMBER <= {amber:.3f} or >= {1/amber:.3f}, "
             f"RED <= {red:.3f} or >= {1/red:.3f}, or source_fill <= 0.70")
lines.append("  " + pg.fmt(final))
lines.append(f"  RED only: {pg.fmt(reds)}")
lines.append("  caught by type:")
for typ, n in caught["incident_type"].value_counts().items():
    lines.append(f"    {typ:22s} {n}")
lines.append("  missed by type:")
for typ, n in miss["incident_type"].value_counts().items():
    lines.append(f"    {typ:22s} {n}")
lines.append(
    f"  false alarms on holiday or promo days: {int(((fp.is_holiday==1)|(fp.is_promo==1)).sum())} of {int(len(fp))}"
)
lines.append(
    f"  red-band false alarms: {int(len(red_fp))}"
    + (f" ({', '.join(f'{r.pipeline} {r.date.date()}' for _, r in red_fp.iterrows())})" if len(red_fp) else "")
)
lines.append("")
lines.append(
    f"HOOK 16 Mar 2026 orders_facts: {hook.tier}  ratio={hook.ratio:.2f}  "
    f"usable={int(hook.usable_rows):,}  expected={hook.expected:,.0f}  "
    f"sources={int(hook.sources_arrived)}/{int(hook.sources_expected)}  status={hook.status}"
)

text = "\n".join(lines) + "\n"
open(os.path.join(pg.OUTS, "02_model.txt"), "w").write(text)
open(os.path.join(pg.OUTS, "thresholds.txt"), "w").write(f"AMBER={amber:.3f}\nRED={red:.3f}\n")
print(text)
