#!/usr/bin/env python3
"""Train the demand model, report in-distribution quality, then score every later day."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import drift_tools as D
from model_common import FEATURES, load, train_model, predict, HOOK_DAY, DRIFT_START

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    df = load()
    model, info = train_model(df)
    df = predict(df, model)
    rows = []
    for split in ("train", "test", "production"):
        part = df[(df["split"] == split) & df["pred"].notna()]
        if split == "production":
            pre = part[part["date"] < DRIFT_START]
            post = part[part["date"] >= DRIFT_START]
            for name, g in (("production_pre_shift", pre), ("production_post_shift", post), ("production_all", part)):
                if len(g) == 0:
                    continue
                rows.append({
                    "split": name, "n": int(len(g)),
                    "mape": D.mape(g["units"], g["pred"]),
                    "mae": D.mae(g["units"], g["pred"]),
                    "mean_units": float(g["units"].mean()),
                    "mean_pred": float(g["pred"].mean()),
                    "mean_err": float(g["err"].mean()),
                })
        else:
            rows.append({
                "split": split, "n": int(len(part)),
                "mape": D.mape(part["units"], part["pred"]),
                "mae": D.mae(part["units"], part["pred"]),
                "mean_units": float(part["units"].mean()),
                "mean_pred": float(part["pred"].mean()),
                "mean_err": float(part["err"].mean()),
            })
    metrics = pd.DataFrame(rows)
    hook = df.loc[df["date"] == HOOK_DAY].iloc[0]
    # Units the model asked the warehouse to buy vs what actually moved, from shift to hook inclusive.
    win = df[(df["date"] >= DRIFT_START) & (df["date"] <= HOOK_DAY) & df["pred"].notna()]
    signed = float(win["err"].sum())  # pred - actual; positive = over-order
    short_or_over = -signed if signed < 0 else signed
    direction = "OVER" if signed > 0 else "SHORT"

    lines = [
        "DEMAND MODEL (HistGradientBoosting, training window only)",
        f"Trained on {info['n_train']} days  {info['train_start']} to {info['train_end']}.",
        "Features: calendar, temperature, planned promo rate, trailing channel mix and AOV,",
        "new-zone share, competitor index, and lagged units. The incident log is never used.",
    ]
    for r in rows:
        lines.append(
            f"{r['split']:24s} n={r['n']:3d}  MAPE {r['mape']*100:5.1f}%  MAE {r['mae']:6.0f}  "
            f"mean units {r['mean_units']:.0f}  mean pred {r['mean_pred']:.0f}  mean err {r['mean_err']:+.0f}"
        )
    lines.append(
        f"HOOK {HOOK_DAY.date()}: actual {int(hook['units'])}  predicted {int(hook['pred'])}  "
        f"error {int(hook['err']):+d}  APE {hook['ape']*100:.1f}%"
    )
    lines.append(
        f"SHIFT-TO-HOOK {DRIFT_START.date()} to {HOOK_DAY.date()}: "
        f"sum(pred-actual) {signed:+.0f}  warehouse {direction} by {short_or_over:.0f} units over {len(win)} days."
    )
    text = "\n".join(lines) + "\n"
    (OUT / "02_model.txt").write_text(text)
    metrics.to_csv(OUT / "02_model_metrics.csv", index=False)
    df.to_csv(OUT / "02_scored_daily.csv", index=False)
    print(text, end="")
    return metrics


if __name__ == "__main__":
    main()
