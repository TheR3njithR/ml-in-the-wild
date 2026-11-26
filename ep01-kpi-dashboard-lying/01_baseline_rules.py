"""Step 1: the alert most dashboards already have. One threshold per KPI tile."""
import os, pandas as pd
import kpi_guard as kg
df = kg.attach_truth(kg.build_features(kg.load())); tr, te = kg.split(df)
hit, which = kg.threshold_alerts(tr, te)
tp, fp, fn, p, r = kg.prf(hit, te.is_incident)
lines = [f"Test year {kg.TEST[0]} to {kg.TEST[1]}: {len(te):,} city-days, {int(te.is_incident.sum())} with broken data (synthetic ground truth)",
         f"Threshold rule: any of {', '.join(kg.HEADLINE)} more than 4 robust SDs from its usual level",
         f"alerts raised        {int(hit.sum())}",
         f"real data problems   {tp}   (precision {p:.0%})",
         f"false alarms         {fp}   of which on real events (sales, festivals, rain): {int((hit & ~te.is_incident & te.real_event.notna()).sum())}",
         f"problems missed      {fn}   (recall {r:.0%})",
         "missed by type:"]
miss = te[~hit & te.is_incident].incident.value_counts()
lines += [f"  {k:<22}{v}" for k, v in miss.items()]
lines += ["caught by type:"] + [f"  {k:<22}{v}" for k, v in te[hit & te.is_incident].incident.value_counts().items()]
open(os.path.join(kg.OUTS, "01_baseline.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
