"""Step 5: the charts used in the video (and in your own report). Dark style to match the channel."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import kpi_guard as kg
BG, FG, MUT, ACC, RED, AMB, GRN = "#0a1222", "#e6edf7", "#8da2c0", "#34d399", "#f87171", "#facc15", "#38bdf8"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": "#2a3a58", "axes.labelcolor": FG, "text.color": FG,
                     "xtick.color": MUT, "ytick.color": MUT, "font.size": 15, "axes.titlesize": 20, "axes.titleweight": "bold",
                     "axes.grid": True, "grid.color": "#1b2740", "legend.facecolor": "#111b30", "legend.edgecolor": "#2a3a58", "savefig.facecolor": BG})
df = kg.attach_truth(kg.build_features(kg.load())); te = pd.read_csv(os.path.join(kg.OUTS, "test_scored.csv"), parse_dates=["date"])
def save(fig, name): fig.tight_layout(); fig.savefig(os.path.join(kg.CHARTS, name), dpi=110); plt.close(fig)

# 1) the hook: revenue looked fine, order value did not (Mumbai, January 2026)
m = df[(df.city == "Mumbai") & (df.date >= "2026-01-01") & (df.date <= "2026-01-31")]
fig, ax = plt.subplots(1, 2, figsize=(16, 6.5))
ax[0].plot(m.date, m.revenue_inr / 1e5, color=GRN, lw=3); ax[0].set_title("Dashboard tile: revenue (Rs lakh)")
ax[1].plot(m.date, m.aov, color=MUT, lw=3); ax[1].set_title("Model input: average order value (Rs)")
d = pd.Timestamp("2026-01-21"); r = m[m.date == d].iloc[0]
ax[0].scatter([d], [r.revenue_inr / 1e5], s=160, color=AMB, zorder=5); ax[0].annotate("21 Jan: +14%,\nlooks like a good day", (d, r.revenue_inr / 1e5), xytext=(-210, 60), textcoords="offset points", color=AMB, arrowprops=dict(arrowstyle="->", color=AMB))
ax[1].scatter([d], [r.aov], s=160, color=RED, zorder=5); ax[1].annotate("1.21x usual:\nrevenue and orders disagree", (d, r.aov), xytext=(-330, -30), textcoords="offset points", color=RED, arrowprops=dict(arrowstyle="->", color=RED))
for a in ax: a.tick_params(axis="x", rotation=30)
save(fig, "01_hook_revenue_vs_aov.png")

# 2) relationships: orders vs sessions change (test year)
fig, ax = plt.subplots(figsize=(11, 8.5))
n = te[~te.is_incident & te.real_event.isna()]; e = te[te.real_event.notna() & ~te.is_incident]; b = te[te.is_incident]
ax.scatter(np.exp(n.sessions_dev), np.exp(n.orders_dev), s=12, color="#3b4d6e", label="normal days")
ax.scatter(np.exp(e.sessions_dev), np.exp(e.orders_dev), s=70, color=ACC, label="real events: sales, festivals, rain")
ax.scatter(np.exp(b.sessions_dev), np.exp(b.orders_dev), s=90, color=RED, marker="X", label="broken data")
ax.plot([0.3, 2.6], [0.3, 2.6], color=MUT, ls="--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
for axis in (ax.xaxis, ax.yaxis): axis.set_major_locator(FixedLocator([0.4, 0.6, 0.8, 1, 1.5, 2, 2.5])); axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x")); axis.set_minor_locator(NullLocator())
ax.set_xlabel("sessions vs usual (x)"); ax.set_ylabel("orders vs usual (x)")
ax.set_title("Real events move together. Broken data moves alone."); ax.legend(loc="upper left")
save(fig, "02_orders_vs_sessions.png")

# 3) score timeline (test year)
fig, ax = plt.subplots(figsize=(16, 6.5)); g = __import__("pickle").load(open(os.path.join(kg.OUTS, "guard_v1.pkl"), "rb"))
ax.scatter(te.date, te.score, s=8, color="#3b4d6e", label="city-day")
ax.scatter(te[te.is_incident].date, te[te.is_incident].score, s=60, color=RED, marker="X", label="broken data (ground truth)")
ax.scatter(te[te.real_event.notna()].date, te[te.real_event.notna()].score, s=40, color=ACC, label="real event")
ax.axhline(g.amber, color=AMB, lw=2, ls="--"); ax.axhline(g.red, color=RED, lw=2, ls="--")
ax.text(te.date.min(), g.amber + 0.005, " AMBER: publish with banner", color=AMB, va="bottom"); ax.text(te.date.min(), g.red + 0.005, " RED: hold + page", color=RED, va="bottom")
ax.set_title("Anomaly score for every city, every day (Nov 2025 - Oct 2026)"); ax.set_ylabel("anomaly score"); ax.legend(loc="lower right", ncol=3, fontsize=12)
save(fig, "03_score_timeline.png")

# 4) thresholds vs isolation forest (numbers from outputs/01 and 02)
import re
t1 = open(os.path.join(kg.OUTS, "01_baseline.txt")).read(); t2 = open(os.path.join(kg.OUTS, "02_model.txt")).read().split("[final")[1]
num = lambda s, k: int(re.search(k + r"\s+(\d+)", s).group(1))
A = [num(t1, "alerts raised"), num(t1, "real data problems"), num(t1, "false alarms"), num(t1, "problems missed")]
B = [num(t2, "alerts"), num(t2, "caught"), num(t2, "false alarms"), num(t2, "missed")]
fig, ax = plt.subplots(figsize=(14, 6.5)); x = np.arange(4); lab = ["alerts sent", "real problems caught", "false alarms", "problems missed"]
b1 = ax.bar(x - 0.2, A, 0.4, color="#64748b", label="threshold per KPI tile"); b2 = ax.bar(x + 0.2, B, 0.4, color=ACC, label="Isolation Forest on relationships")
for bars in (b1, b2):
    for bb in bars: ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.8, str(int(bb.get_height())), ha="center", fontsize=17, fontweight="bold")
ax.set_xticks(x, lab); ax.set_title("One test year, 35 broken city-days (synthetic)"); ax.legend()
save(fig, "04_thresholds_vs_model.png")

# 5) the stale copy: Delhi orders, May 2026
s = df[(df.city == "Delhi") & (df.date >= "2026-05-14") & (df.date <= "2026-06-07")]
fig, ax = plt.subplots(figsize=(16, 6.5)); ax.plot(s.date, s.orders, color=GRN, lw=3, marker="o")
fz = s[s.frozen == 1]; ax.scatter(fz.date, fz.orders, s=220, color=RED, zorder=5)
ax.annotate("28 and 29 May: exactly the same as 27 May\nno spike, no dip, nothing for a threshold to catch", (fz.date.iloc[-1], fz.orders.iloc[-1]), xytext=(-120, 120), textcoords="offset points", color=RED, arrowprops=dict(arrowstyle="->", color=RED))
ax.set_title("Delhi orders: the dashboard that did not change"); ax.annotate("Big Summer Sale: real", (pd.Timestamp("2026-05-16"), s[s.date == "2026-05-16"].orders.iloc[0]), xytext=(25, -10), textcoords="offset points", color=ACC); ax.tick_params(axis="x", rotation=30)
save(fig, "05_stale_copy.png")

# 6) isolation intuition: random cuts isolate an odd point fast
rng = np.random.default_rng(7); P = rng.normal(0, 1, (220, 2)); odd = np.array([3.4, -2.6])
fig, axs = plt.subplots(1, 2, figsize=(16, 7))
for ax, target, ttl in ((axs[0], odd, "An odd day: isolated in a few cuts"), (axs[1], P[np.argmin((P ** 2).sum(1))], "A normal day: needs many cuts")):
    ax.scatter(P[:, 0], P[:, 1], s=14, color="#3b4d6e"); ax.scatter(*odd, s=120, color=RED, marker="X")
    ax.scatter(*target, s=160, facecolors="none", edgecolors=AMB, lw=2.5)
    box = [-4, 4, -4, 4]; pts = np.vstack([P, odd]); cuts = 0; r2 = np.random.default_rng(3)
    while ((pts[:, 0] >= box[0]) & (pts[:, 0] <= box[1]) & (pts[:, 1] >= box[2]) & (pts[:, 1] <= box[3])).sum() > 1 and cuts < 40:
        inb = pts[(pts[:, 0] >= box[0]) & (pts[:, 0] <= box[1]) & (pts[:, 1] >= box[2]) & (pts[:, 1] <= box[3])]
        k = r2.integers(2); lo, hi = inb[:, k].min(), inb[:, k].max(); c = r2.uniform(lo, hi); cuts += 1
        if k == 0:
            ax.plot([c, c], box[2:], color=AMB, lw=1, alpha=0.8); box = [box[0], c, box[2], box[3]] if target[0] < c else [c, box[1], box[2], box[3]]
        else:
            ax.plot(box[:2], [c, c], color=AMB, lw=1, alpha=0.8); box = [box[0], box[1], box[2], c] if target[1] < c else [box[0], box[1], c, box[3]]
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_title(f"{ttl}: {cuts} cuts"); ax.set_xticks([]); ax.set_yticks([])
save(fig, "06_isolation_intuition.png")
print("charts written:", ", ".join(sorted(os.listdir(kg.CHARTS))))
