"""Step 3: turn a probability into a decision at T-60, and write the action report.
Usage: python 03_decide_and_act.py [YYYY-MM-DD]   (default 2026-03-14)

GREEN  let the refresh run
AMBER  publish the previous successful refresh as a fallback; notify the dataset owner
RED    intervene: delay a competing refresh, or restart a saturated / offline gateway, and page the engineer
"""
import os, sys, pickle
import pandas as pd
import refresh_guard as rg

day = sys.argv[1] if len(sys.argv) > 1 else "2026-03-14"
g = pickle.load(open(os.path.join(rg.OUTS, "guard_v1.pkl"), "rb"))
df = rg.attach_truth(rg.load())
today = df[df.date == day].copy()
today["proba"] = g.proba(today)
today["tier"] = g.tier(today.proba)

rows, actions = [], []
for _, r in today.sort_values("scheduled_at").iterrows():
    if r.tier == "GREEN":
        why = "leading signals look normal"
        act = "let it run"
    else:
        f, why = g.explain(r)
        if r.tier == "RED":
            if f in ("mem_now", "mem_mean_1h", "mem_max_1h", "mem_over_90_min", "concurrent_running", "queued_next_90m"):
                act = "delay competing refreshes; page the engineer and restart the gateway"
            elif f == "gateway_online":
                act = "page the engineer and restart the gateway"
            elif f == "credential_age_days":
                act = "page the engineer and rotate the source credential"
            else:
                act = "page the engineer; hold a fallback refresh ready"
        else:
            act = "keep the last successful refresh as a fallback; notify the dataset owner"
    actions.append(act)
    rows.append(
        f"{r.scheduled_at.strftime('%H:%M')}  {r.dataset:<13} {r.tier:<6}  "
        f"p={r.proba:.2f}  mem {r.mem_now:4.0f}%  run {int(r.concurrent_running)}  {why}"
    )

reds = today[today.tier.values == "RED"] if hasattr(today.tier, 'values') else today
# today.tier is a numpy array from g.tier; attach it properly
today = today.copy()
n_red = int((today.tier == "RED").sum())
n_amber = int((today.tier == "AMBER").sum())
if n_red:
    decision = "INTERVENE on RED datasets; banner the AMBER ones. The 9:00 dashboard gets a short notice."
else:
    decision = ("Notify owners of AMBER datasets and keep yesterday's refresh as a fallback."
                if n_amber else "LET EVERY REFRESH RUN: nothing unusual in the leading signals.")

L = [f"Refresh risk report for {day}, decision at T-60 (QuickBasket, synthetic data)", ""] + rows + ["", "Decision: " + decision]
# Highlight City Ops if present
co = today[today.dataset == "City Ops"]
if len(co):
    r = co.iloc[0]
    L.append(f"Focus, City Ops: {r.tier} (p={r.proba:.2f}). " + (g.explain(r)[1] if r.tier != "GREEN" else "leading signals look normal") + ".")
    if r.tier == "RED":
        L.append("Action: delay the competing refreshes, page the data engineer, restart the gateway, and put a banner on the 9:00 dashboard.")

open(os.path.join(rg.OUTS, "03_trust_report.txt"), "w").write("\n".join(L) + "\n")
open(os.path.join(rg.REPORTS, f"action_report_{day}.md"), "w").write("# " + L[0] + "\n\n```\n" + "\n".join(rows) + "\n```\n\n**" + decision + "**\n")
print("\n".join(L))
