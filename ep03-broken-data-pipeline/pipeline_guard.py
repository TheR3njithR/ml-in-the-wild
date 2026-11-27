"""Shared paths, features and the expected-volume residual detector for Episode 3."""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, OUTS, CHARTS = [os.path.join(ROOT, d) for d in ("data", "outputs", "charts")]

TRAIN_END = pd.Timestamp("2025-07-31")
HOOK_DATE = pd.Timestamp("2026-03-16")

# Calendar features used to *adjust* the same-weekday volume pattern.
# The baseline expected volume is same_dow_median_8w (last 8 same weekdays).
# A small regressor then learns holiday / promo / month-end multipliers.
# Do NOT include today's usable_rows (the target) or source_fill
# (if the model saw missing sources, it would "expect" a shortfall
# and hide a SOURCE_SILENT break).
FEATURES = [
    "pipeline_id",
    "dow",
    "week_of_month",
    "month",
    "is_weekend",
    "is_month_end",
    "is_holiday",
    "is_promo",
]

# Decision thresholds are computed from training residuals only.
# AMBER: highest residual-ratio cut that still has >= 90% precision on train.
# RED: 100% precision cut on train, plus a 0.03 safety margin, floored at 0.55.
AMBER_PRECISION = 0.90
RED_MARGIN = 0.03
RED_FLOOR = 0.55


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """lag_7 / lag_14 are the same pipeline 7 and 14 days ago.
    same_dow_median_8w is the median of the last 8 same weekdays (d-7..d-56)."""
    df = df.sort_values(["pipeline", "date"]).copy()
    parts = []
    for _, g in df.groupby("pipeline", sort=False):
        g = g.copy()
        v = g["usable_rows"].to_numpy(dtype=float)
        n = len(g)
        lag7 = np.full(n, np.nan)
        lag14 = np.full(n, np.nan)
        med8 = np.full(n, np.nan)
        for i in range(n):
            if i >= 7:
                lag7[i] = v[i - 7]
            if i >= 14:
                lag14[i] = v[i - 14]
            past = [v[i - 7 * k] for k in range(1, 9) if i - 7 * k >= 0]
            if past:
                med8[i] = float(np.median(past))
        g["lag_7"] = lag7
        g["lag_14"] = lag14
        g["same_dow_median_8w"] = med8
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    fill = out.groupby("pipeline")["usable_rows"].transform("median")
    for c in ("lag_7", "lag_14", "same_dow_median_8w"):
        out[c] = out[c].fillna(fill)
    return out


def X_of(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES].astype(float)


def fit_expected(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    """Predict log(usable / same-dow median). Train on non-extreme days only
    so a few silent breaks cannot pull the usual pattern down."""
    base = train["same_dow_median_8w"].clip(lower=1)
    naive = train["usable_rows"].clip(lower=1) / base
    ok = train[(naive > 0.60) & (naive < 1.80)].copy()
    y = np.log(ok["usable_rows"].clip(lower=1) / ok["same_dow_median_8w"].clip(lower=1))
    model = HistGradientBoostingRegressor(
        max_depth=3,
        min_samples_leaf=15,
        max_iter=150,
        learning_rate=0.08,
        random_state=33,
    )
    model.fit(X_of(ok), y)
    return model


def expected_rows(model, df: pd.DataFrame) -> np.ndarray:
    base = df["same_dow_median_8w"].clip(lower=1).to_numpy(dtype=float)
    adj = np.exp(model.predict(X_of(df)))
    adj = np.clip(adj, 0.35, 2.5)
    return (base * adj).clip(min=1.0)


def residual_ratio(actual, expected) -> np.ndarray:
    return np.asarray(actual, dtype=float) / np.asarray(expected, dtype=float)


def threshold_from_train(train: pd.DataFrame, col="ratio"):
    """Cuts on actual/expected. Alerts when ratio is below the cut OR above 1/cut (too high).
    Precision is measured on train silent-break labels."""
    y = train["is_silent"].astype(bool).values
    ratios = train[col].values

    def prec_at(cut):
        pred = (ratios <= cut) | (ratios >= (1.0 / cut) if cut > 0 else False)
        if pred.sum() == 0:
            return 0.0, 0, 0
        tp = int((pred & y).sum())
        return tp / pred.sum(), int(pred.sum()), tp

    # search cuts from 0.95 down to 0.35
    cuts = np.round(np.linspace(0.95, 0.35, 61), 3)
    amber, red = 0.70, 0.55
    for c in cuts:
        p, n, _ = prec_at(c)
        if p >= AMBER_PRECISION and n >= 8:
            amber = c
        if p >= 1.0 - 1e-12 and n >= 4:
            red = c
            break
    red = min(max(red - RED_MARGIN, RED_FLOOR), amber - 0.02)
    return float(amber), float(red)


def decide(ratio, source_fill, amber, red):
    """Return (tier, reason_code). source_fill is a second, independent signal."""
    if ratio <= red or ratio >= (1.0 / red) or source_fill <= 0.70:
        return "RED", "volume_or_sources"
    if ratio <= amber or ratio >= (1.0 / amber) or source_fill < 0.999:
        return "AMBER", "volume_or_sources"
    return "GREEN", "within_pattern"


def score_frame(df: pd.DataFrame, model, amber, red) -> pd.DataFrame:
    out = df.copy()
    out["expected"] = expected_rows(model, out)
    out["ratio"] = residual_ratio(out["usable_rows"], out["expected"])
    tiers, reasons = [], []
    for r, sf in zip(out["ratio"], out["source_fill"]):
        t, k = decide(float(r), float(sf), amber, red)
        tiers.append(t)
        reasons.append(k)
    out["tier"] = tiers
    out["reason"] = reasons
    out["alert"] = out["tier"].isin(["AMBER", "RED"])
    out["red_alert"] = out["tier"].eq("RED")
    return out


def split(df: pd.DataFrame):
    train = df[df["date"] <= TRAIN_END].copy()
    test = df[df["date"] > TRAIN_END].copy()
    return train, test


def summarise(pred, y, name=""):
    pred = np.asarray(pred, dtype=bool)
    y = np.asarray(y, dtype=bool)
    alerts = int(pred.sum())
    caught = int((pred & y).sum())
    false = int((pred & ~y).sum())
    missed = int((~pred & y).sum())
    prec = (caught / alerts) if alerts else 0.0
    rec = (caught / y.sum()) if y.sum() else 0.0
    return {
        "name": name,
        "alerts": alerts,
        "caught": caught,
        "false": false,
        "missed": missed,
        "precision": prec,
        "recall": rec,
        "n_pos": int(y.sum()),
    }


def fmt(s):
    return (
        f"alerts {s['alerts']:4d}   caught {s['caught']:3d}   false alarms {s['false']:3d}   "
        f"missed {s['missed']:3d}   precision {s['precision']*100:.0f}%   recall {s['recall']*100:.0f}%"
    )
