#!/usr/bin/env python3
"""Shared load / train / predict for the QuickBasket beverages demand model."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "daily_demand.csv"

FEATURES = [
    "dow", "is_weekend", "is_holiday",
    "temp_c", "promo_rate", "express_share", "app_share",
    "aov", "new_zone_share", "competitor_index",
    "lag_7", "roll_28",
]
CAT = ["dow", "is_weekend", "is_holiday"]
TRAIN_END = pd.Timestamp("2025-11-30")
TEST_END = pd.Timestamp("2026-02-28")
DRIFT_START = pd.Timestamp("2026-04-11")
HOOK_DAY = pd.Timestamp("2026-04-20")
LABEL_LAG = 7
SEED = 41


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["date", "label_available_on"])
    df["split"] = np.where(df["date"] <= TRAIN_END, "train",
                    np.where(df["date"] <= TEST_END, "test", "production"))
    return df


def train_model(df: pd.DataFrame):
    tr = df[(df["split"] == "train") & df[FEATURES + ["units"]].notna().all(axis=1)]
    cat_idx = [FEATURES.index(c) for c in CAT]
    model = HistGradientBoostingRegressor(
        max_depth=3, max_iter=200, learning_rate=0.06,
        min_samples_leaf=20, l2_regularization=0.1,
        categorical_features=cat_idx, random_state=SEED,
    )
    model.fit(tr[FEATURES], tr["units"])
    return model, {"n_train": int(len(tr)), "train_start": str(tr["date"].min().date()),
                   "train_end": str(tr["date"].max().date())}


def predict(df: pd.DataFrame, model) -> pd.DataFrame:
    out = df.copy()
    ok = out[FEATURES].notna().all(axis=1)
    pred = np.full(len(out), np.nan)
    pred[ok.to_numpy()] = model.predict(out.loc[ok, FEATURES])
    out["pred"] = np.round(pred, 0)
    out["err"] = out["pred"] - out["units"]
    out["abs_err"] = out["err"].abs()
    out["ape"] = out["abs_err"] / out["units"].clip(lower=1)
    return out
