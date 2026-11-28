#!/usr/bin/env python3
"""Charts used in the Episode 4 screen recordings. Dark style to match the channel."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import drift_tools as D
from model_common import DRIFT_START, HOOK_DAY, TRAIN_END, TEST_END

ROOT = Path(__file__).resolve().parent
OUT, CHARTS = ROOT / "outputs", ROOT / "charts"
CHARTS.mkdir(exist_ok=True)

BG, FG, MUT, ACC, RED, AMB, GRN, BLU = (
    "#0a1222", "#e6edf7", "#8da2c0", "#34d399", "#f87171", "#facc15", "#38bdf8", "#a78bfa"
)
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": "#2a3a58",
    "axes.labelcolor": FG, "text.color": FG, "xtick.color": MUT, "ytick.color": MUT,
    "font.size": 15, "axes.titlesize": 20, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": "#1b2740", "legend.facecolor": "#111b30",
    "legend.edgecolor": "#2a3a58", "savefig.facecolor": BG,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(CHARTS / name, dpi=110)
    plt.close(fig)
    print("wrote", name)


def main():
    scored = pd.read_csv(OUT / "02_scored_daily.csv", parse_dates=["date"])
    mon = pd.read_csv(OUT / "03_drift_daily.csv", parse_dates=["date"])
    base = pd.read_csv(OUT / "01_baseline_daily.csv", parse_dates=["date"])
    hook = scored.loc[scored["date"] == HOOK_DAY].iloc[0]

    # 1) Hook: forecast vs actual around the shift
    win = scored[(scored["date"] >= "2026-03-15") & (scored["date"] <= "2026-04-30")]
    fig, ax = plt.subplots(figsize=(16, 6.8))
    ax.plot(win["date"], win["units"], color=GRN, lw=2.6, label="actual units (arrive 7 days later)")
    ax.plot(win["date"], win["pred"], color=AMB, lw=2.6, label="model forecast, sent to the warehouse")
    ax.axvline(DRIFT_START, color=RED, lw=1.8, ls="--", label="Summer Sips promo starts")
    ax.axvline(HOOK_DAY, color=ACC, lw=1.8)
    ax.annotate("hook 20 Apr", (HOOK_DAY, hook["pred"]),
                xytext=(15, 25), textcoords="offset points", color=ACC,
                arrowprops=dict(arrowstyle="->", color=ACC))
    ax.set_title("The forecast stayed in the old world. Actual demand had already dropped.")
    ax.set_ylabel("beverage units")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
    save(fig, "01_hook_forecast_vs_actual.png")

    # 2) Cumulative over-order from the shift to the hook
    hook_win = scored[(scored["date"] >= DRIFT_START) & (scored["date"] <= HOOK_DAY)].copy()
    hook_win["cum_err"] = hook_win["err"].cumsum()
    fig, ax = plt.subplots(figsize=(16, 6.4))
    ax.fill_between(hook_win["date"], 0, hook_win["cum_err"], color=RED, alpha=0.45)
    ax.plot(hook_win["date"], hook_win["cum_err"], color=RED, lw=3)
    last = hook_win.iloc[-1]
    ax.annotate(f"{int(last['cum_err']):,} units over-ordered",
                (last["date"], last["cum_err"]),
                xytext=(-220, -40), textcoords="offset points", color=FG,
                arrowprops=dict(arrowstyle="->", color=FG))
    ax.set_title("Ten mornings of a stale forecast: almost thirty thousand extra units")
    ax.set_ylabel("cumulative (forecast − actual) units")
    ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
    save(fig, "02_cumulative_overorder.png")

    # 3) Delayed MAPE vs the cut, with label lag
    fig, ax = plt.subplots(figsize=(16, 6.4))
    ax.plot(base["date"], base["rolling_14_mape"] * 100, color=AMB, lw=2.6, label="delayed 14-day MAPE")
    # cut is implied by first alert; draw 11.8 from the baseline text later if needed
    cut = 11.8
    ax.axhline(cut, color=RED, ls="--", lw=1.6, label=f"alert cut {cut:.1f}%")
    ax.axvline(DRIFT_START, color=RED, lw=1.6, ls=":")
    ax.axvline(HOOK_DAY, color=ACC, lw=1.6)
    ax.axvline(pd.Timestamp("2026-04-22"), color=GRN, lw=1.8, label="first MAPE alert, 22 Apr")
    ax.set_title("The error pack on 20 April still used labels from before the promo had landed")
    ax.set_ylabel("MAPE, percent")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
    save(fig, "03_delayed_mape.png")

    # 4) Domain AUC and status
    fig, ax = plt.subplots(figsize=(16, 6.4))
    ax.plot(mon["date"], mon["domain_auc"], color=BLU, lw=2.6, label="domain-classifier AUC")
    ax.axhline(D.AUC_AMBER, color=AMB, ls="--", lw=1.4, label=f"amber {D.AUC_AMBER:.2f}")
    ax.axhline(D.AUC_RED, color=RED, ls="--", lw=1.4, label=f"red {D.AUC_RED:.2f}")
    ax.axvline(DRIFT_START, color=RED, lw=1.6, ls=":")
    ax.axvline(pd.Timestamp("2026-04-14"), color=ACC, lw=1.8, label="first red, 14 Apr")
    colors = mon["status"].map({"GREEN": ACC, "AMBER": AMB, "RED": RED})
    ax.scatter(mon["date"], mon["domain_auc"], c=colors, s=36, zorder=3)
    ax.set_ylim(0.45, 1.05)
    ax.set_title("A classifier that tries to tell training rows from the last 14 days")
    ax.set_ylabel("AUC")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
    save(fig, "04_domain_auc.png")

    # 5) PSI for the features that moved
    fig, ax = plt.subplots(figsize=(16, 6.8))
    for f, col in [("promo_rate", AMB), ("express_share", GRN), ("app_share", BLU), ("new_zone_share", RED)]:
        ax.plot(mon["date"], mon[f"psi_{f}"], color=col, lw=2.2, label=f.replace("_", " "))
    ax.axhline(0.25, color=MUT, ls="--", lw=1.2, label="textbook red cut 0.25")
    ax.axvline(DRIFT_START, color=RED, lw=1.6, ls=":")
    ax.axvline(pd.Timestamp("2026-04-14"), color=ACC, lw=1.8)
    ax.set_title("Fourteen-day PSI on the input features (no labels required)")
    ax.set_ylabel("PSI")
    ax.legend(loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
    save(fig, "05_feature_psi.png")

    # 6) Train vs hook-window distributions for promo_rate
    train = scored[scored["date"] <= TRAIN_END]
    hook_w = scored[(scored["date"] > HOOK_DAY - pd.Timedelta(days=14)) & (scored["date"] <= HOOK_DAY)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, f, title in zip(
        axes,
        ["promo_rate", "express_share", "aov"],
        ["Promo rate", "Express share", "Average order value"],
    ):
        ax.hist(train[f], bins=18, density=True, color=GRN, alpha=0.55, label="training")
        ax.hist(hook_w[f], bins=10, density=True, color=RED, alpha=0.7, label="14 days to 20 Apr")
        ax.set_title(title)
        ax.legend(fontsize=11)
    fig.suptitle("The input world on 20 April was not the world the model had learned", fontsize=20, fontweight="bold")
    save(fig, "06_train_vs_hook_features.png")

    # 7) Lead-time strip: status vs delayed MAPE
    fig, ax = plt.subplots(figsize=(16, 4.2))
    color = mon["status"].map({"GREEN": ACC, "AMBER": AMB, "RED": RED})
    ax.bar(mon["date"], np.ones(len(mon)), color=color, width=0.9, align="center")
    ax.axvline(DRIFT_START, color="white", lw=1.6, ls="--")
    ax.axvline(HOOK_DAY, color="white", lw=1.6)
    ax.axvline(pd.Timestamp("2026-04-22"), color=GRN, lw=2.0)
    ax.set_yticks([])
    ax.set_title("Green / amber / red each morning, versus the delayed MAPE alert on 22 April")
    ax.xaxis.set_major_formatter(DateFormatter("%d %b"))
    save(fig, "07_status_strip.png")

    print("charts done")


if __name__ == "__main__":
    main()
