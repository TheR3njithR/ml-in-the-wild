"""Step 4: feedback. The analyst labels every amber/red alert from the first six test weeks.
  1. Alert precision is tracked.
  2. Month-end close days go on a calendar; an amber alert on a close day is shown as expected, not a page.
  3. Champion vs challenger: a retrained, failure-emphasised model only replaces the current one if it wins
     on months nobody has reviewed yet.
"""
import os
import numpy as np
import pandas as pd
import refresh_guard as rg

df = rg.attach_truth(rg.load())
tr, te = rg.split(df)
reviewed = te[te.date <= rg.REVIEW_END]
future = te[te.date > rg.REVIEW_END]
champion = rg.Guard().fit(tr)

def pages(g, d, use_cal=False, include_amber=True):
    tier = pd.Series(g.tier(g.proba(d)), index=d.index)
    alert = tier.isin(["RED", "AMBER"] if include_amber else ["RED"])
    if use_cal:
        # Close-day amber is expected load, not a page. Red still pages.
        alert = (tier == "RED") | ((tier == "AMBER") & ~d.is_close)
    return alert

a = reviewed[pages(champion, reviewed, False)]
labels = a.assign(label=a.is_failure.map({True: "real risk", False: "false alarm"}))
labels[["date", "dataset", "label"]].assign(date=lambda x: x.date.dt.date).to_csv(
    os.path.join(rg.OUTS, "analyst_labels.csv"), index=False
)
n_false = int((~a.is_failure).sum())
false_desc = ", ".join(
    f"{r.dataset} {r.date.date()}{' (month-end close)' if r.is_close else ''}"
    for _, r in a[~a.is_failure].iterrows()
) or "none"
L = [
    f"1) Analyst review, {rg.TEST[0]} to {rg.REVIEW_END}: {len(a)} alerts labelled",
    f"   real risk {int(a.is_failure.sum())}, false alarm {n_false}: {false_desc}",
    f"   alert precision {a.is_failure.mean():.0%}" if len(a) else "   no alerts in the review window",
]
tp, fp, fn, p, r = rg.prf(pages(champion, reviewed, True), reviewed.is_failure)
L += [
    f"2) Close-day calendar rule on the same weeks: false pages {n_false} -> {fp}, real risks still paged or amber-flagged {tp}",
    "",
]
# Challenger: retrain on train + reviewed, with confirmed failures emphasised
hist = pd.concat([tr, reviewed], ignore_index=True)
w = np.where(hist.is_failure, 14.0, 1.0)
challenger = rg.Guard(fail_weight=14.0).fit(hist, sample_weight=w)
L.append(
    f"3) Champion vs challenger on {future.date.min().date()} to {rg.TEST[1]} "
    f"({len(future):,} runs, {int(future.is_failure.sum())} failures), calendar rule on for both:"
)
res = {}
for name, g in (("champion: current model", champion), ("challenger: retrained, failures emphasised", challenger)):
    tp, fp, fn, p, r = rg.prf(pages(g, future, True), future.is_failure)
    res[name] = (tp, fp, p, r)
    L.append(f"   {name:<46} caught {tp:>3}  false alarms {fp:>3}  missed {fn:>3}  precision {p:.0%}  recall {r:.0%}")
keep = res["champion: current model"][1] <= res["challenger: retrained, failures emphasised"][1] or \
       res["champion: current model"][2] >= res["challenger: retrained, failures emphasised"][2]
L.append(
    "   decision: " + (
        "keep the champion. The tighter fit started paging on ordinary busy mornings."
        if keep else "promote the challenger."
    )
)
open(os.path.join(rg.OUTS, "04_feedback.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
