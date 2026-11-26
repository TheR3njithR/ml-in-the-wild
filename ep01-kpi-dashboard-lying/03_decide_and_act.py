"""Step 3: turn scores into a decision every morning, and write the trust report that goes out before the dashboard.
Usage: python 03_decide_and_act.py [YYYY-MM-DD]   (default 2026-01-21)
GREEN  publish as normal
AMBER  publish with a "data under review" banner on that city; analyst checks within 2 hours
RED    hold that city's tiles and page the data engineer
2+ cities flagged for the same reason on the same day = a systemic pipeline problem: hold the whole dashboard."""
import os, sys, pickle, pandas as pd
import kpi_guard as kg
day = sys.argv[1] if len(sys.argv) > 1 else "2026-01-21"
g = pickle.load(open(os.path.join(kg.OUTS, "guard_v1.pkl"), "rb"))
df = kg.build_features(kg.load()); today = df[df.date == day].copy()
today["score"] = g.score(today); today["tier"] = g.tier(today.score)
rows, reasons = [], []
for _, r in today.sort_values("score", ascending=False).iterrows():
    f, why = g.explain(r) if r.tier != "GREEN" else ("", "relationships look normal")
    reasons.append(f if r.tier != "GREEN" else None)
    vs = (r.revenue_inr / r.revenue_inr_base - 1) * 100
    rows.append(f"{r.city:<10} {r.tier:<6} {r.score:.2f}  revenue {r.revenue_inr/1e5:5.1f} L ({vs:+3.0f}%)  {why}")
flagged = [x for x in reasons if x]
systemic = max((flagged.count(x) for x in set(flagged)), default=0) >= 2
action = ("HOLD THE WHOLE DASHBOARD: one problem in several cities means the pipeline, not the business."
          if systemic else ("HOLD the RED cities, banner the AMBER ones." if flagged else "PUBLISH: nothing unusual in how the numbers relate."))
L = [f"Dashboard trust report for {day} (QuickBasket, synthetic data)", ""] + rows + ["", "Decision: " + action]
if systemic: L.append("Action: page the data engineer on call; post 'numbers under review' in #leadership before 9 AM.")
open(os.path.join(kg.OUTS, "03_trust_report.txt"), "w").write("\n".join(L) + "\n")
open(os.path.join(kg.REPORTS, f"trust_report_{day}.md"), "w").write("# " + L[0] + "\n\n```\n" + "\n".join(rows) + "\n```\n\n**" + action + "**\n")
print("\n".join(L))
