"""Step 4: feedback. The analyst labels every alert from the first three months (Nov 2025 - Jan 2026)
as 'data bug' or 'real event / normal day'. Three things happen with those labels:
  1. Alert precision is tracked: the alerting system gets its own KPI.
  2. Real events go into an event calendar; an alert on a calendar day is shown as EXPECTED instead of paging anyone.
  3. Champion vs challenger: a retrained model (all reviewed history, confirmed bug days removed) only replaces the
     current one if it does better on labelled months. Here we can check that on Feb - Oct 2026 using the
     synthetic ground truth; in real life you check it on each new month of labels."""
import os, pandas as pd
import kpi_guard as kg
df = kg.attach_truth(kg.build_features(kg.load())); tr, te = kg.split(df)
review_end = "2026-01-31"; reviewed = te[te.date <= review_end]; future = te[te.date > review_end]
cal = set(pd.read_csv(os.path.join(kg.DATA, "event_calendar.csv"), parse_dates=["date"]).date)
champion = kg.Guard().fit(tr)
def pages(g, d, use_cal):
    p = pd.Series(g.tier(g.score(d)) != "GREEN", index=d.index)
    return p & ~d.date.isin(cal) if use_cal else p
a = reviewed[pages(champion, reviewed, False)]
labels = a.assign(label=a.is_incident.map({True: "data bug", False: "real event / normal day"}))
labels[["date", "city", "label"]].assign(date=lambda x: x.date.dt.date, real_event=a.real_event.fillna("")).to_csv(os.path.join(kg.OUTS, "analyst_labels.csv"), index=False)
L = [f"1) Analyst review, {kg.TEST[0]} to {review_end}: {len(a)} alerts labelled",
     f"   data bug {int(a.is_incident.sum())}, false alarm {int((~a.is_incident).sum())}: " + ", ".join(f"{r.city} {r.date.date()} ({r.real_event})" for _, r in a[~a.is_incident].iterrows()),
     f"   alert precision {a.is_incident.mean():.0%}"]
tp, fp, fn, p, r = kg.prf(pages(champion, reviewed, True), reviewed.is_incident)
L += [f"2) Event calendar rule on the same months: false alarms {int((~a.is_incident).sum())} -> {fp}, bugs still caught {tp}", ""]
hist = pd.concat([tr, reviewed]); challenger = kg.Guard().fit(hist[~hist.is_incident])
L.append(f"3) Champion vs challenger on Feb - Oct 2026 ({len(future):,} city-days, {int(future.is_incident.sum())} with broken data), calendar rule on for both:")
res = {}
for name, g in (("champion: current model", champion), ("challenger: retrained on clean data", challenger)):
    tp, fp, fn, p, r = kg.prf(pages(g, future, True), future.is_incident); res[name] = p
    L.append(f"   {name:<38} caught {tp:>2}  false alarms {fp:>2}  missed {fn:>2}  precision {p:.0%}  recall {r:.0%}")
L.append("   decision: " + ("keep the champion. Cleaning the training data tightened 'normal', so the same alert budget now pages on ordinary noise."
                            if res["champion: current model"] >= res["challenger: retrained on clean data"] else "promote the challenger."))
open(os.path.join(kg.OUTS, "04_feedback.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
