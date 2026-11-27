"""refresh_guard: shared logic for Episode 2 (synthetic QuickBasket Power BI refresh data).

The idea in one line: a prediction may only use information known before the scheduled start.
Duration, error message and final row count are leakage. Leading signals at T-60 — gateway
health, contention, size, credential age, recent duration trend — are fair.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUTS = os.path.join(HERE, "outputs")
CHARTS = os.path.join(HERE, "charts")
REPORTS = os.path.join(HERE, "reports")
for d in (OUTS, CHARTS, REPORTS):
    os.makedirs(d, exist_ok=True)

TRAIN = ("2024-12-01", "2025-07-31")
TEST = ("2025-08-01", "2026-03-31")
REVIEW_END = "2025-09-11"  # first six weeks of the test window

# Leading features only — all timestamped at or before T-60.
FEATURES = [
    "gateway_online", "mem_now", "mem_mean_1h", "mem_max_1h", "mem_over_90_min",
    "cpu_mean_1h", "capacity_mean_1h", "concurrent_running", "queued_next_90m",
    "size_mb", "size_vs_28d", "credential_age_days", "duration_trend", "failed_yesterday",
    "is_monday", "is_month_end",
]
# First attempt (shown then discarded): adds the duration of the run we are predicting.
FEATURES_LEAK = FEATURES + ["duration_min"]

PLAIN = {
    "gateway_online": "the on-premises gateway is offline",
    "mem_now": "gateway memory is {x}% at T-60",
    "mem_mean_1h": "gateway memory has averaged {x}% over the last hour",
    "mem_max_1h": "gateway memory peaked at {x}% in the last hour",
    "mem_over_90_min": "gateway memory has sat above 90% for {x} minutes",
    "cpu_mean_1h": "gateway CPU has averaged {x}% over the last hour",
    "capacity_mean_1h": "workspace capacity has averaged {x}% over the last hour",
    "concurrent_running": "{x} other refresh(es) already running on the same gateway",
    "queued_next_90m": "{x} other refresh(es) scheduled in the next 90 minutes",
    "size_mb": "the dataset is {x} MB",
    "size_vs_28d": "dataset size is {x}x its usual 28-day level",
    "credential_age_days": "the source credential is {x} days old",
    "duration_trend": "recent successful runs are {x}x their usual duration",
    "failed_yesterday": "this dataset already failed yesterday",
    "is_monday": "Monday morning (the warehouse is usually busier)",
    "is_month_end": "month-end close morning (the warehouse is usually busier)",
}

def load():
    df = pd.read_csv(os.path.join(DATA, "refresh_history.csv"), parse_dates=["date", "scheduled_at", "started_at", "ended_at"])
    return df.sort_values(["scheduled_at", "dataset"]).reset_index(drop=True)

def attach_truth(df):
    inc = pd.read_csv(os.path.join(DATA, "incident_log.csv"), parse_dates=["date"])
    cal = pd.read_csv(os.path.join(DATA, "close_calendar.csv"), parse_dates=["date"])
    df = df.merge(inc[["date", "dataset", "incident"]], on=["date", "dataset"], how="left")
    df["is_failure"] = df.status.eq("failed") if "status" in df else df.incident.notna()
    df["is_close"] = df.date.isin(set(cal.date))
    return df

def split(df):
    tr = df[(df.date >= TRAIN[0]) & (df.date <= TRAIN[1])].copy()
    te = df[(df.date >= TEST[0]) & (df.date <= TEST[1])].copy()
    return tr, te

def prf(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return tp, fp, fn, p, r

def rule_alerts(df):
    """The usual team rules, evaluated at T-60: offline, memory > 85%, or credential older than 85 days."""
    return (df.gateway_online == 0) | (df.mem_now > 85) | (df.credential_age_days > 85)

class Guard:
    """Gradient-boosted classifier on leading features, with probability thresholds from training."""

    def __init__(self, features=FEATURES, amber=0.35, red=0.75, seed=42, fail_weight=6.0):
        self.features = list(features)
        self.amber, self.red, self.seed, self.fail_weight = amber, red, seed, fail_weight
        self.model = None

    def fit(self, train, sample_weight=None):
        X = train[self.features]
        y = train.is_failure.astype(int)
        if sample_weight is None:
            sample_weight = np.where(y == 1, self.fail_weight, 1.0)
        self.model = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=180,
            l2_regularization=0.2, min_samples_leaf=8,
            random_state=self.seed,
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        # Thresholds come from training probabilities only.
        # Red: the lowest cut with 100% precision on the training months (and at least 10 alerts).
        # Amber: the lowest cut with at least 90% precision on the training months.
        p = self.proba(train)
        yv = train.is_failure.to_numpy()
        def best_cut(min_prec, min_alerts=10):
            cuts = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 80)))
            chosen = None
            for t in cuts:
                pred = p >= t
                if int(pred.sum()) < min_alerts:
                    continue
                tp = int((pred & yv).sum()); fp = int((pred & ~yv).sum())
                prec = tp / max(1, tp + fp)
                if prec >= min_prec:
                    chosen = float(t)
                    break
            return chosen
        self.amber = best_cut(0.90) or float(np.quantile(p, 0.90))
        # A small margin above the 100%-precision training cut keeps the red band clean on new months.
        cut100 = best_cut(1.00) or float(np.quantile(p, 0.99))
        self.red = min(0.95, cut100 + 0.05)
        if self.red <= self.amber:
            self.red = min(0.99, self.amber + 0.1)
        return self

    def proba(self, df):
        return self.model.predict_proba(df[self.features])[:, 1]

    def tier(self, p):
        p = np.asarray(p)
        return np.where(p >= self.red, "RED", np.where(p >= self.amber, "AMBER", "GREEN"))

    def explain(self, row):
        """Plain-English reason: the leading feature furthest from a healthy training centre, in a bad direction."""
        # Hard rules first when they are obviously on
        if int(row.get("gateway_online", 1)) == 0:
            return "gateway_online", PLAIN["gateway_online"]
        # Score "badness" with simple, directed z-like ranks
        bad = {
            "mem_over_90_min": float(row.mem_over_90_min) / 10.0,
            "mem_now": (float(row.mem_now) - 70) / 10.0,
            "mem_mean_1h": (float(row.mem_mean_1h) - 70) / 10.0,
            "capacity_mean_1h": (float(row.capacity_mean_1h) - 70) / 10.0,
            "concurrent_running": float(row.concurrent_running),
            "queued_next_90m": float(row.queued_next_90m) / 2.0,
            "size_mb": (float(row.size_mb) - 800) / 400.0,
            "size_vs_28d": (float(row.size_vs_28d) - 1.0) * 3,
            "credential_age_days": (float(row.credential_age_days) - 80) / 8.0,
            "duration_trend": (float(row.duration_trend) - 1.0) * 4,
            "failed_yesterday": float(row.failed_yesterday) * 2,
        }
        f = max(bad, key=bad.get)
        v = row[f]
        if f == "size_mb":
            x = f"{v:.0f}"
        elif f in ("mem_now", "mem_mean_1h", "mem_max_1h", "cpu_mean_1h", "capacity_mean_1h"):
            x = f"{v:.0f}"
        elif f in ("size_vs_28d", "duration_trend"):
            x = f"{v:.2f}"
        else:
            x = f"{int(v)}"
        return f, PLAIN[f].format(x=x)
