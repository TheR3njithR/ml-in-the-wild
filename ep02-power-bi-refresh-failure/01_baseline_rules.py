"""Step 1: the usual process (email after failure) and the usual rules, scored at T-60."""
import os
import refresh_guard as rg

df = rg.attach_truth(rg.load())
tr, te = rg.split(df)
hit = rg.rule_alerts(te)
tp, fp, fn, p, r = rg.prf(hit, te.is_failure)
delay = te.loc[te.is_failure, "notice_delay_min"].mean()
lines = [
    f"Test months {rg.TEST[0]} to {rg.TEST[1]}: {len(te):,} scheduled runs, {int(te.is_failure.sum())} failures (synthetic ground truth)",
    f"Email-only process: average notice delay {delay:.0f} minutes after the run had already died",
    "",
    "Rule set, evaluated at T-60: gateway offline, OR memory > 85%, OR credential older than 85 days",
    f"alerts raised        {int(hit.sum())}",
    f"real failures        {tp}   (precision {p:.0%})",
    f"false alarms         {fp}",
    f"failures missed      {fn}   (recall {r:.0%})",
    "missed by type:",
]
miss = te[~hit & te.is_failure].incident.value_counts()
lines += [f"  {k:<22}{v}" for k, v in miss.items()]
lines += ["caught by type:"] + [f"  {k:<22}{v}" for k, v in te[hit & te.is_failure].incident.value_counts().items()]
open(os.path.join(rg.OUTS, "01_baseline.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
