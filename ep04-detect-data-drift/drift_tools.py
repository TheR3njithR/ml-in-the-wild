#!/usr/bin/env python3
"""Shared PSI, KS and domain-classifier helpers for Episode 4.

These functions are the teaching core. They never look at the hidden
incident labels. Windows and thresholds are chosen on the training
period only, then applied forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Standard population-stability thresholds (Siddiqi / credit-risk practice).
PSI_AMBER = 0.10
PSI_RED = 0.25
# Domain-classifier AUC: 0.5 means the two windows are indistinguishable.
# KS statistic (max vertical gap between two CDFs). Fixed cuts, not p-values,
# so a large sample cannot make a tiny shift look "significant".
KS_AMBER = 0.30
KS_RED = 0.45

MONITOR_FEATURES = [
    "promo_rate",
    "express_share",
    "app_share",
    "aov",
    "new_zone_share",
    "competitor_index",
]


def _edges(expected: np.ndarray, bins: int = 10) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, bins + 1)
    cuts = np.unique(np.quantile(expected, qs))
    if len(cuts) < 3:
        lo, hi = float(np.min(expected)), float(np.max(expected))
        if lo == hi:
            hi = lo + 1e-6
        cuts = np.linspace(lo, hi, 4)
    return cuts


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index of `actual` versus the `expected` baseline."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 5:
        return float("nan")
    # A feature that was (almost) constant in training and then moved is drifted.
    if np.nanstd(expected) < 1e-12:
        if abs(np.nanmean(actual) - np.nanmean(expected)) < 1e-9 and np.nanstd(actual) < 1e-12:
            return 0.0
        return 1.0
    cuts = _edges(expected, bins)
    inner = cuts[1:-1]
    bins_full = np.concatenate(([-np.inf], inner, [np.inf]))
    e_pct = np.histogram(expected, bins=bins_full)[0].astype(float)
    a_pct = np.histogram(actual, bins=bins_full)[0].astype(float)
    if e_pct.sum() <= 0 or a_pct.sum() <= 0:
        return float("nan")
    e_pct = np.clip(e_pct / e_pct.sum(), 1e-4, None)
    a_pct = np.clip(a_pct / a_pct.sum(), 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_stat(expected, actual) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 5:
        return float("nan")
    return float(ks_2samp(expected, actual, alternative="two-sided").statistic)


def domain_auc(train_x: pd.DataFrame, recent_x: pd.DataFrame, seed: int = 7) -> float:
    """AUC of a linear classifier that tries to tell training rows from recent rows.

    An AUC near 0.5 means the two windows look the same. An AUC well above 0.5
    means a model can tell them apart: the input world has moved.
    """
    if len(recent_x) < 8:
        return float("nan")
    x0 = train_x.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x1 = recent_x.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vary = [c for c in x0.columns if (x0[c].std() + x1[c].std()) > 1e-12]
    if len(vary) < 1:
        return 0.5
    X = pd.concat([x0[vary], x1[vary]], axis=0)
    y = np.r_[np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=400, random_state=seed),
    )
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, proba))


def status_from_scores(max_psi: float, max_ks: float, auc: float) -> str:
    if (max_psi >= PSI_RED) or (max_ks >= KS_RED) or (auc >= AUC_RED):
        return "RED"
    if (max_psi >= PSI_AMBER) or (max_ks >= KS_AMBER) or (auc >= AUC_AMBER):
        return "AMBER"
    return "GREEN"


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred) / np.clip(np.abs(y_true), 1.0, None)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def psi_null_quantiles(expected, window: int = 14, bins: int = 3, draws: int = 400, seed: int = 41) -> dict:
    """PSI of random `window`-day slices of the training series against the full series.

    Textbook 0.10 / 0.25 cuts assume hundreds of rows. A two-week window is much
    smaller, so the 95th percentile of this null is the fair red cut.
    """
    expected = np.asarray(expected, dtype=float)
    expected = expected[np.isfinite(expected)]
    n = len(expected)
    if n < window + 5:
        return {"p50": 0.10, "p90": 0.10, "p95": 0.25, "p99": 0.25}
    rng = np.random.default_rng(seed)
    ps = []
    for _ in range(draws):
        take = rng.choice(n, size=window, replace=False)
        ps.append(psi(expected, expected[take], bins=bins))
    qs = np.nanpercentile(ps, [50, 90, 95, 99])
    return {"p50": float(qs[0]), "p90": float(qs[1]), "p95": float(qs[2]), "p99": float(qs[3])}


# Domain-classifier cuts. 0.5 means the two windows are indistinguishable.
# A 14-day calm window can reach ~0.66 by chance, so amber starts at 0.70.
AUC_AMBER = 0.70
AUC_RED = 0.80
