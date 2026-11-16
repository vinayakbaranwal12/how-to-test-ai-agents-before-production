"""Build the two charts used in the article from the harness outputs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runcount import one_sample_power

BLUE, ORANGE, VERM, GREEN = "#0072B2", "#E69F00", "#D55E00", "#009E73"

# Chart 1: exact power of the one-sample gate, 0.95 -> 0.90, alpha 0.05
ns = list(range(20, 301))
power = [one_sample_power(0.95, 0.90, n, 0.05)[2] for n in ns]
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
ax.plot(ns, power, color=BLUE, linewidth=1.6, label="Exact power, 95% to 90% drop")
ax.axhline(0.80, color=VERM, linestyle="--", linewidth=1.2, label="80% target")
ax.axvline(169, color=GREEN, linestyle=":", linewidth=1.4)
ax.axvline(191, color=ORANGE, linestyle=":", linewidth=1.4)
ax.annotate("First n at 80%: 169", xy=(169, 0.80), xytext=(60, 0.90), arrowprops=dict(arrowstyle="->", color=GREEN), color=GREEN)
ax.annotate("Stays above 80% from n=191", xy=(191, 0.805), xytext=(205, 0.62), arrowprops=dict(arrowstyle="->", color=ORANGE), color="#8a5a00")
ax.set_xlabel("Runs per build (n)")
ax.set_ylabel("Chance the gate catches the drop")
ax.set_ylim(0, 1.0)
ax.set_title("Detection power is a sawtooth, not a smooth curve")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig("../results/power-curve.png")
plt.close(fig)

# Chart 2: false alarms vs detections per gate, 0.95 -> 0.85, from results/gates_sim.txt
labels = ["Exact match\n(1 run)", "Outcome check\n(1 run)", "Pass-rate gate\n(126 runs per arm)"]
false_alarm = [62.28, 5.24, 2.45]
detect = [66.23, 15.48, 80.18]
fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
x = range(len(labels))
w = 0.36
b1 = ax.bar([i - w / 2 for i in x], false_alarm, w, color=ORANGE, label="Fails a healthy build (false alarm)")
b2 = ax.bar([i + w / 2 for i in x], detect, w, color=BLUE, label="Fails a regressed build (detection)")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.2, f"{r.get_height():.1f}%", ha="center", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("Builds failed (%)")
ax.set_ylim(0, 100)
ax.set_title("Same hidden regression, three gates (10,000 simulated builds each)")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("../results/gate-comparison.png")
plt.close(fig)
print("charts written")
