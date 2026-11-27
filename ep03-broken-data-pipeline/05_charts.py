"""Step 5: the charts used in the video. Dark style to match the channel."""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pipeline_guard as pg

BG, FG, MUT, ACC, RED, AMB, GRN = "#0a1222", "#e6edf7", "#8da2c0", "#34d399", "#f87171", "#facc15", "#38bdf8"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": "#2a3a58",
    "axes.labelcolor": FG, "text.color": FG, "xtick.color": MUT, "ytick.color": MUT,
    "font.size": 15, "axes.titlesize": 20, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": "#1b2740", "legend.facecolor": "#111b30",
    "legend.edgecolor": "#2a3a58", "savefig.facecolor": BG,
})

def save(fig, name):
    os.makedirs(pg.CHARTS, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(pg.CHARTS, name), dpi=110)
    plt.close(fig)

tl = pd.read_csv(os.path.join(pg.DATA, "hook_timeline.csv"), parse_dates=["ts"])
te = pd.read_csv(os.path.join(pg.OUTS, "test_scored.csv"), parse_dates=["date"])

# 1) Hook timeline
fig, ax = plt.subplots(figsize=(16, 6.8))
ax.plot(tl.ts, tl.rows_landed, color=AMB, lw=3, label="orders_facts rows landed")
ax.axhline(int(tl.expected_if_all_zones.iloc[0]), color=GRN, ls="--", lw=1.8, label="expected if all 8 zones arrive")
ax.axvline(pd.Timestamp("2026-03-16 06:12"), color=ACC, lw=2)
ax.axvline(pd.Timestamp("2026-03-16 09:00"), color=RED, lw=2)
ax.annotate("job SUCCESS\n06:12", (pd.Timestamp("2026-03-16 06:12"), tl.rows_landed.max() * 0.55),
            xytext=(25, -20), textcoords="offset points", color=ACC,
            arrowprops=dict(arrowstyle="->", color=ACC))
ax.annotate("meeting 09:00", (pd.Timestamp("2026-03-16 09:00"), tl.expected_if_all_zones.iloc[0] * 0.72),
            xytext=(-130, 15), textcoords="offset points", color=RED,
            arrowprops=dict(arrowstyle="->", color=RED))
ax.set_ylabel("rows landed")
ax.set_title("16 March 2026: the Orders job was green, with 41% of the volume missing")
ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
ax.legend(loc="upper left")
save(fig, "01_hook_timeline.png")

# 2) Status rule vs reality — silent breaks all look green
fig, ax = plt.subplots(figsize=(12, 6.5))
lab = ["silent breaks\n(the real problem)", "hard job failures\n(the log would catch)"]
vals = [int(te.is_silent.sum()), int(te.is_hard_fail.sum())]
colors = [RED, "#64748b"]
bars = ax.bar(lab, vals, color=colors, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", fontsize=20, fontweight="bold")
ax.set_ylabel("pipeline-days in the test months")
ax.set_title("Every silent break finished with status SUCCESS")
save(fig, "02_silent_vs_hard.png")

# 3) Rules vs pattern model
t1 = open(os.path.join(pg.OUTS, "01_baseline.txt")).read()
t2 = open(os.path.join(pg.OUTS, "02_model.txt")).read()

def grab(src, key):
    # find the first "alerts N caught N false alarms N missed N" after a marker, or named counts
    m = re.search(key + r".*?alerts\s+(\d+)\s+caught\s+(\d+)\s+false alarms\s+(\d+)\s+missed\s+(\d+)", src, re.S)
    return [int(x) for x in m.groups()]

A = grab(t1, "Rule 2")
B = grab(t2, r"\[final")
fig, ax = plt.subplots(figsize=(14, 6.5))
x = np.arange(4)
lab = ["alerts sent", "real breaks caught", "false alarms", "breaks missed"]
b1 = ax.bar(x - 0.2, A, 0.4, color="#64748b", label="naive row-count rule")
b2 = ax.bar(x + 0.2, B, 0.4, color=ACC, label="expected-volume residual")
for bars in (b1, b2):
    for bb in bars:
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.8, str(int(bb.get_height())),
                ha="center", fontsize=17, fontweight="bold")
ax.set_xticks(x, lab)
ax.set_title(f"Eight test months, {int(te.is_silent.sum())} silent pipeline breaks (synthetic)")
ax.legend()
save(fig, "03_rules_vs_model.png")

# 4) Ratio scatter / band on hook pipeline
of = te[te.pipeline == "orders_facts"].sort_values("date")
fig, ax = plt.subplots(figsize=(16, 6.5))
good = of[~of.is_silent]
bad = of[of.is_silent]
ax.scatter(good.date, good.ratio, s=22, c=GRN, alpha=0.85, label="usual days")
ax.scatter(bad.date, bad.ratio, s=42, c=RED, zorder=3, label="silent breaks")
# thresholds
th = open(os.path.join(pg.OUTS, "thresholds.txt")).read()
amber = float(re.search(r"AMBER=([0-9.]+)", th).group(1))
redc = float(re.search(r"RED=([0-9.]+)", th).group(1))
ax.axhline(amber, color=AMB, ls="--", lw=1.4, label=f"amber {amber:.2f}")
ax.axhline(redc, color=RED, ls="--", lw=1.4, label=f"red {redc:.2f}")
hook = of[of.date == pg.HOOK_DATE].iloc[0]
ax.scatter([hook.date], [hook.ratio], s=140, facecolors="none", edgecolors="#fff", lw=2, zorder=4)
ax.annotate("16 Mar hook", (hook.date, hook.ratio), xytext=(20, 18),
            textcoords="offset points", color=FG,
            arrowprops=dict(arrowstyle="->", color=FG))
ax.set_ylabel("usable rows / expected rows")
ax.set_title("orders_facts: the hook sits well below the expected volume band")
ax.legend(loc="upper right", ncol=2)
save(fig, "04_ratio_band.png")

# 5) Probability / tier counts
fig, ax = plt.subplots(figsize=(12, 6.5))
order = ["GREEN", "AMBER", "RED"]
cols = {"GREEN": GRN, "AMBER": AMB, "RED": RED}
xs = np.arange(3)
real, fake = [], []
for t in order:
    sub = te[te.tier == t]
    real.append(int(sub.is_silent.sum()))
    fake.append(int((~sub.is_silent).sum()))
b1 = ax.bar(xs - 0.2, fake, 0.4, color="#3b4d6e", label="usual days")
b2 = ax.bar(xs + 0.2, real, 0.4, color=RED, label="silent breaks")
ax.set_xticks(xs, order)
for bars in (b1, b2):
    for bb in bars:
        if bb.get_height() > 0:
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.8,
                    str(int(bb.get_height())), ha="center", fontsize=16, fontweight="bold")
ax.set_title("Green / amber / red on the test months")
ax.legend()
save(fig, "05_probability_tiers.png")
print("charts written:", os.listdir(pg.CHARTS))
