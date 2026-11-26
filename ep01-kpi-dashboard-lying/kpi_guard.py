"""kpi_guard: shared logic for Episode 1 (synthetic QuickBasket KPI data).

The idea in one line: a dashboard rarely lies about ONE number; it lies about how numbers relate.
Orders up 2x with sessions up 2x is Diwali. Orders up 2x with sessions flat is a duplicate load.
So the model sees relationships (conversion, order value, refund rate, a city vs the market, "did anything change
since yesterday") measured against each city's own recent same-weekday baseline, not raw levels.
"""
import os
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); OUTS = os.path.join(HERE, "outputs"); CHARTS = os.path.join(HERE, "charts"); REPORTS = os.path.join(HERE, "reports")
for d in (OUTS, CHARTS, REPORTS): os.makedirs(d, exist_ok=True)
TRAIN = ("2024-11-01", "2025-10-31"); TEST = ("2025-11-01", "2026-10-31")
HEADLINE = ["orders", "sessions", "revenue_inr", "refunds"]          # the four tiles on the dashboard
FEATURES = ["conv_dev", "aov_dev", "refund_dev", "frozen"]          # four relationships (see 02: six features did worse)
FEATURES_V0 = ["conv_dev", "aov_dev", "refund_dev", "orders_vs_market", "sessions_vs_market", "frozen"]
PLAIN = {  # how each feature is explained to a human in the alert
    "conv_dev": "conversion {x} usual: orders and traffic disagree",
    "aov_dev": "order value {x} usual: revenue and orders disagree",
    "refund_dev": "refund rate {x} usual: refunds feed may be incomplete",
    "orders_vs_market": "orders {x} vs the other cities today",
    "sessions_vs_market": "sessions {x} vs the other cities today",
    "frozen": "identical to yesterday: the pipeline may not have run",
}

def load():
    df = pd.read_csv(os.path.join(DATA, "kpi_daily.csv"), parse_dates=["date"])
    return df.sort_values(["city", "date"]).reset_index(drop=True)

def baseline(s, weeks=8):
    """Median of the same weekday over the previous `weeks` weeks (today excluded)."""
    lags = pd.concat([s.shift(7 * k) for k in range(1, weeks + 1)], axis=1)
    return lags.median(axis=1, skipna=True).where(lags.notna().sum(axis=1) >= 4)

def build_features(df):
    df = df.copy()
    df["conv"] = df.orders / df.sessions
    df["aov"] = df.revenue_inr / df.orders.clip(lower=1)
    df["refund_rate"] = df.refunds / df.orders.clip(lower=1)
    g = df.groupby("city", group_keys=False)
    for c in HEADLINE + ["conv", "aov", "refund_rate"]:
        df[c + "_base"] = g[c].apply(baseline)
    for c in HEADLINE:
        df[c + "_dev"] = np.log((df[c] + 1) / (df[c + "_base"] + 1))      # log ratio to own baseline
    df["conv_dev"] = np.log(df.conv / df.conv_base)
    df["aov_dev"] = np.log(df.aov / df.aov_base)
    df["refund_dev"] = np.log((df.refund_rate + 0.002) / (df.refund_rate_base + 0.002))
    mkt = df.groupby("date")[["orders_dev", "sessions_dev"]].transform("median")
    df["orders_vs_market"] = df.orders_dev - mkt.orders_dev                # a city moving alone is suspicious
    df["sessions_vs_market"] = df.sessions_dev - mkt.sessions_dev
    prev = g[["orders", "sessions", "revenue_inr"]].shift(1)
    df["frozen"] = (df[["orders", "sessions", "revenue_inr"]].values == prev.values).sum(axis=1) / 3.0
    return df.dropna(subset=FEATURES_V0 + [c + "_dev" for c in HEADLINE]).reset_index(drop=True)

def split(df):
    tr = df[(df.date >= TRAIN[0]) & (df.date <= TRAIN[1])]; te = df[(df.date >= TEST[0]) & (df.date <= TEST[1])]
    return tr.copy(), te.copy()

def attach_truth(df):
    inc = pd.read_csv(os.path.join(DATA, "incident_log.csv"), parse_dates=["date"])
    ev = pd.read_csv(os.path.join(DATA, "event_calendar.csv"), parse_dates=["date"])
    df = df.merge(inc, on=["date", "city"], how="left")
    df["is_incident"] = df.incident.notna()
    rows = []
    for _, r in ev.iterrows():
        for c in (df.city.unique() if r.cities == "All" else [x.strip() for x in r.cities.split(",")]):
            rows.append({"date": r.date, "city": c, "real_event": r.event})
    df = df.merge(pd.DataFrame(rows).drop_duplicates(["date", "city"]), on=["date", "city"], how="left")
    return df

# ---------------------------------------------------------------- baseline: per-metric threshold alerts
def robust_scale(train_dev):
    return 1.4826 * (train_dev - train_dev.median()).abs().median()

def threshold_alerts(train, test, z=4.0):
    """The classic dashboard alert: flag a tile if it is more than z robust standard deviations from normal."""
    hit = pd.Series(False, index=test.index); which = pd.Series("", index=test.index)
    for c in HEADLINE:
        sc = train.groupby("city")[c + "_dev"].apply(robust_scale)
        zz = test[c + "_dev"] / test.city.map(sc)
        m = zz.abs() > z; hit |= m; which[m] = which[m] + c + " "
    return hit, which.str.strip()

# ---------------------------------------------------------------- the model
class Guard:
    """Isolation Forest on relationship features, scaled by robust spread learned from training data."""
    def __init__(self, features=FEATURES, n_estimators=500, seed=42, amber_q=0.985, red_q=0.995):
        self.features, self.n, self.seed, self.q = list(features), n_estimators, seed, (amber_q, red_q)
    def fit(self, train):
        X = train[self.features]
        # every tree sees the whole year: with only ~1% of days broken, a 256-row sample often contains none of them
        self.model = IsolationForest(n_estimators=self.n, max_samples=len(X), random_state=self.seed)
        self.center = X.median(); self.scale = (1.4826 * (X - self.center).abs().median()).replace(0, 1e-3)
        if "frozen" in self.scale: self.scale["frozen"] = 0.1                                    # frozen is 0 on normal days; any repeat is huge
        self.model.fit(self._z(X))
        s = self.score(train)
        self.amber, self.red = np.quantile(s, self.q[0]), np.quantile(s, self.q[1])   # alert budget: top 1.5% amber, top 0.5% red
        return self
    def _z(self, X): return ((X - self.center) / self.scale).clip(-25, 25)
    def score(self, df):  # higher = more anomalous
        return -self.model.score_samples(self._z(df[self.features]))
    def explain(self, row):
        z = ((row[self.features] - self.center) / self.scale).abs(); f = z.idxmax()
        v = row[f]
        if f in ("conv_dev", "aov_dev", "refund_dev"): x = f"{np.exp(v):.2f}x"
        elif f == "frozen": x = ""
        else: x = f"{(np.exp(v) - 1) * 100:+.0f}%"
        return f, PLAIN[f].format(x=x)
    def tier(self, s): return np.where(s >= self.red, "RED", np.where(s >= self.amber, "AMBER", "GREEN"))

def prf(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return tp, fp, fn, p, r
