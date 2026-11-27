"""Step 5: the charts used in the video. Dark style to match the channel."""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import refresh_guard as rg

BG, FG, MUT, ACC, RED, AMB, GRN = "#0a1222", "#e6edf7", "#8da2c0", "#34d399", "#f87171", "#facc15", "#38bdf8"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": "#2a3a58",
    "axes.labelcolor": FG, "text.color": FG, "xtick.color": MUT, "ytick.color": MUT,
    "font.size": 15, "axes.titlesize": 20, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": "#1b2740", "legend.facecolor": "#111b30",
    "legend.edgecolor": "#2a3a58", "savefig.facecolor": BG,
})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(rg.CHARTS, name), dpi=110)
    plt.close(fig)

hb = pd.read_csv(os.path.join(rg.DATA, "gateway_heartbeat.csv"), parse_dates=["ts"])
te = pd.read_csv(os.path.join(rg.OUTS, "test_scored.csv"), parse_dates=["date", "scheduled_at", "started_at", "ended_at"])

# 1) Hook timeline: 14 March 2026 gateway memory, refresh, failure, meeting
day = hb[(hb.ts >= "2026-03-14 04:00") & (hb.ts <= "2026-03-14 10:00")]
fig, ax = plt.subplots(figsize=(16, 6.8))
ax.plot(day.ts, day.gateway_mem_pct, color=AMB, lw=3, label="gateway memory %")
ax.axhline(90, color=RED, ls="--", lw=1.5)
ax.axvline(pd.Timestamp("2026-03-14 07:30"), color=GRN, lw=2)
ax.axvline(pd.Timestamp("2026-03-14 08:22"), color=RED, lw=2)
ax.axvline(pd.Timestamp("2026-03-14 09:00"), color=ACC, lw=2)
ax.annotate("refresh starts\n07:30", (pd.Timestamp("2026-03-14 07:30"), 55),
            xytext=(-90, -40), textcoords="offset points", color=GRN,
            arrowprops=dict(arrowstyle="->", color=GRN))
ax.annotate("failed 08:22\nafter 52 minutes", (pd.Timestamp("2026-03-14 08:22"), 93),
            xytext=(30, -10), textcoords="offset points", color=RED,
            arrowprops=dict(arrowstyle="->", color=RED))
ax.annotate("meeting 09:00", (pd.Timestamp("2026-03-14 09:00"), 70),
            xytext=(25, 20), textcoords="offset points", color=ACC,
            arrowprops=dict(arrowstyle="->", color=ACC))
ax.set_ylim(40, 100)
ax.set_ylabel("gateway memory %")
ax.set_title("14 March 2026: the warning was on the gateway from 05:20")
ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
ax.legend(loc="lower right")
save(fig, "01_hook_timeline.png")

# 2) Leakage vs leading signals — duration of failed vs successful runs (the trap)
fig, ax = plt.subplots(figsize=(12, 7))
ok = te[~te.is_failure]; bad = te[te.is_failure]
ax.hist(ok.duration_min, bins=30, color="#3b4d6e", alpha=0.9, label="successful runs")
ax.hist(bad.duration_min, bins=20, color=RED, alpha=0.85, label="failed runs")
ax.set_xlabel("duration of this run (minutes)")
ax.set_ylabel("number of refreshes (test months)")
ax.set_title("Duration of the run you are predicting is leakage")
ax.legend()
save(fig, "02_leakage_duration.png")

# 3) Rules vs model
t1 = open(os.path.join(rg.OUTS, "01_baseline.txt")).read()
t2 = open(os.path.join(rg.OUTS, "02_model.txt")).read().split("[final")[1]
num = lambda s, k: int(re.search(k + r"\s+(\d+)", s).group(1))
A = [num(t1, "alerts raised"), num(t1, "real failures"), num(t1, "false alarms"), num(t1, "failures missed")]
B = [num(t2, "alerts"), num(t2, "caught"), num(t2, "false alarms"), num(t2, "missed")]
fig, ax = plt.subplots(figsize=(14, 6.5))
x = np.arange(4)
lab = ["alerts sent", "real failures caught", "false alarms", "failures missed"]
b1 = ax.bar(x - 0.2, A, 0.4, color="#64748b", label="rules at T-60")
b2 = ax.bar(x + 0.2, B, 0.4, color=ACC, label="classifier on leading signals")
for bars in (b1, b2):
    for bb in bars:
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.8, str(int(bb.get_height())),
                ha="center", fontsize=17, fontweight="bold")
ax.set_xticks(x, lab)
ax.set_title(f"Eight test months, {int(te.is_failure.sum())} failed refreshes (synthetic)")
ax.legend()
save(fig, "03_rules_vs_model.png")

# 4) Misses by type
fig, ax = plt.subplots(figsize=(12, 7))
miss = te[(te.tier == "GREEN") & te.is_failure].incident.value_counts().sort_values()
ax.barh(miss.index, miss.values, color=RED)
for y, v in enumerate(miss.values):
    ax.text(v + 0.15, y, str(v), va="center", fontsize=16, fontweight="bold")
ax.set_xlabel("missed failures (test months)")
ax.set_title("What the model missed: mostly failures with no leading signal")
save(fig, "04_misses_by_type.png")

# 5) Probability by outcome, with amber / red lines
import pickle
g = pickle.load(open(os.path.join(rg.OUTS, "guard_v1.pkl"), "rb"))
fig, ax = plt.subplots(figsize=(14, 6.8))
ax.hist(te.loc[~te.is_failure, "proba"], bins=40, color="#3b4d6e", alpha=0.9, label="successful runs")
ax.hist(te.loc[te.is_failure, "proba"], bins=20, color=RED, alpha=0.85, label="failed runs")
ax.axvline(g.amber, color=AMB, lw=2, ls="--")
ax.axvline(g.red, color=RED, lw=2, ls="--")
ax.text(g.amber + 0.01, ax.get_ylim()[1] * 0.85 if False else 1, " AMBER", color=AMB, va="bottom")
ax.set_xlabel("predicted failure probability at T-60")
ax.set_ylabel("number of refreshes (test months)")
ax.set_title("A probability, then a decision rule")
# annotate after ylim is known
ymin, ymax = ax.get_ylim()
ax.text(g.amber, ymax * 0.9, " AMBER", color=AMB, va="top")
ax.text(g.red, ymax * 0.9, " RED", color=RED, va="top")
ax.legend(loc="upper center")
save(fig, "05_probability_tiers.png")

# 6) Hook morning concurrent view — not required but useful
print("charts written to", rg.CHARTS)
