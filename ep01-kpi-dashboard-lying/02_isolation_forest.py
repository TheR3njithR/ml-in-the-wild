"""Step 2: train an Isolation Forest on one year of history (no labels), score the next year day by day.
Compares the first attempt (six features) with the final one (four relationship features)."""
import os, pickle, pandas as pd
import kpi_guard as kg
df = kg.attach_truth(kg.build_features(kg.load())); tr, te = kg.split(df)
L = [f"trained on {len(tr):,} city-days ({kg.TRAIN[0]} to {kg.TRAIN[1]}), no labels used; tested on {len(te):,} city-days, {int(te.is_incident.sum())} with broken data", ""]
for name, feats in (("first try, 6 features", kg.FEATURES_V0), ("final, 4 features", kg.FEATURES)):
    g = kg.Guard(features=feats).fit(tr); s = g.score(te); tier = pd.Series(g.tier(s), index=te.index)
    flag = tier != "GREEN"; tp, fp, fn, p, r = kg.prf(flag, te.is_incident)
    L += [f"[{name}] {', '.join(feats)}",
          f"  alerts {int(flag.sum()):>3}   caught {tp:>2}   false alarms {fp:>2} (on real events {int((flag & ~te.is_incident & te.real_event.notna()).sum())})   missed {fn:>2}   precision {p:.0%}   recall {r:.0%}"]
    if feats == kg.FEATURES:
        te["score"] = s; te["tier"] = tier; final = g
        red = tier == "RED"; tpr, fpr_, fnr, pr, rr = kg.prf(red, te.is_incident)
        L += [f"  thresholds from training scores: AMBER >= {g.amber:.3f} (top 1.5%), RED >= {g.red:.3f} (top 0.5%)",
              f"  RED only: {int(red.sum())} alerts, {tpr} real, precision {pr:.0%}",
              "  caught by type:"] + [f"    {k:<22}{v}" for k, v in te[flag & te.is_incident].incident.value_counts().items()]
        L += ["  missed by type:"] + [f"    {k:<22}{v}" for k, v in te[~flag & te.is_incident].incident.value_counts().items()]
open(os.path.join(kg.OUTS, "02_model.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
te.to_csv(os.path.join(kg.OUTS, "test_scored.csv"), index=False)
pickle.dump(final, open(os.path.join(kg.OUTS, "guard_v1.pkl"), "wb"))
