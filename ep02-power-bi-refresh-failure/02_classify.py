"""Step 2: train a classifier on eight months, score the next eight.
Shows the leaked first attempt (duration of the run itself) and the honest leading-feature model."""
import os, pickle
import pandas as pd
import refresh_guard as rg

df = rg.attach_truth(rg.load())
tr, te = rg.split(df)
L = [
    f"trained on {len(tr):,} scheduled runs ({rg.TRAIN[0]} to {rg.TRAIN[1]}); "
    f"tested on {len(te):,} runs, {int(te.is_failure.sum())} failures",
    "every test feature is taken from before that run's scheduled start",
    "",
]
# Leaky first attempt
leak = rg.Guard(features=rg.FEATURES_LEAK).fit(tr)
pl = leak.proba(te)
flag_l = pd.Series(leak.tier(pl), index=te.index) != "GREEN"
tp, fp, fn, p, r = rg.prf(flag_l, te.is_failure)
L += [
    f"[first try, includes run duration — LEAKAGE] precision {p:.0%}  recall {r:.0%}  "
    f"alerts {int(flag_l.sum())}  caught {tp}  false alarms {fp}  missed {fn}",
    "  this looks too good: duration is only known after the refresh has finished",
    "",
]
# Honest model
g = rg.Guard(features=rg.FEATURES).fit(tr)
s = g.proba(te)
tier = pd.Series(g.tier(s), index=te.index)
flag = tier != "GREEN"
tp, fp, fn, p, r = rg.prf(flag, te.is_failure)
L += [
    f"[final, leading features only] {', '.join(rg.FEATURES)}",
    f"  alerts {int(flag.sum()):>3}   caught {tp:>3}   false alarms {fp:>3}   missed {fn:>3}   precision {p:.0%}   recall {r:.0%}",
    f"  thresholds from training probabilities: AMBER >= {g.amber:.3f}, RED >= {g.red:.3f}",
]
red = tier == "RED"
tpr, fpr, fnr, pr, rr = rg.prf(red, te.is_failure)
L += [
    f"  RED only: {int(red.sum())} alerts, {tpr} real, {fpr} false, precision {pr:.0%}",
    "  caught by type:",
]
L += [f"    {k:<22}{v}" for k, v in te[flag & te.is_failure].incident.value_counts().items()]
L += ["  missed by type:"]
L += [f"    {k:<22}{v}" for k, v in te[~flag & te.is_failure].incident.value_counts().items()]

te = te.copy()
te["proba"] = s
te["tier"] = tier
open(os.path.join(rg.OUTS, "02_model.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
te.to_csv(os.path.join(rg.OUTS, "test_scored.csv"), index=False)
pickle.dump(g, open(os.path.join(rg.OUTS, "guard_v1.pkl"), "wb"))
