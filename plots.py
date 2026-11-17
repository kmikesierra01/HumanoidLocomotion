"""README charts (light + dark PNGs) from results/*.json -> docs/images/.

Usage: python plots.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

from robustness import SWEEP  # noqa: E402

THEMES = {  # dataviz reference palette: surface, text primary/secondary, muted, gridline, baseline, series 1
    "light": dict(bg="#fcfcfb", t1="#0b0b0b", t2="#52514e", muted="#898781", grid="#e1e0d9", base="#c3c2b7", s1="#2a78d6"),
    "dark": dict(bg="#1a1a19", t1="#f0efec", t2="#c3c2b7", muted="#898781", grid="#2c2c2a", base="#383835", s1="#3987e5"),
}
DPI = 200  # README shows these at ~half size, so 2 image px ~ 1 display px
TITLES = {"push_n": "Push on torso, 0.2 s (N)", "friction": "Floor friction (x nominal)",
          "payload_kg": "Payload on torso (kg)", "delay_ms": "Control delay (ms)", "noise": "Sensor noise (x training)"}


def style(ax, c):
    ax.set_facecolor(c["bg"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(c["base"])
    ax.tick_params(colors=c["muted"], length=0, labelsize=8)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    ax.grid(axis="y", color=c["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def robustness(c, out):
    r = json.load(open("results/robustness.json"))
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.9), facecolor=c["bg"])
    for ax, (cond, levels) in zip(axes, SWEEP.items()):
        style(ax, c)
        succ = [100 - sum(e["fell"] for e in r[cond][str(lv)]) for lv in levels]
        x = range(len(levels))
        ax.plot(x, succ, color=c["s1"], linewidth=2, marker="o", markersize=8,
                markeredgecolor=c["bg"], markeredgewidth=2, solid_capstyle="round", zorder=3)
        ax.set_xticks(list(x), [f"{lv:g}" for lv in levels])
        ax.set_title(TITLES[cond], color=c["t1"], fontsize=9, loc="left", pad=8)
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
    fig.suptitle("Survival rate over 20 s as one condition gets worse (MuJoCo, 100 episodes per point)",
                 color=c["t2"], fontsize=9, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=DPI, facecolor=c["bg"])
    plt.close(fig)


def rounded_bar(ax, x, h, w, color, r_px=8):
    """One path: square base, rounded top corners (r_px image px), per the mark spec."""
    if h <= 0:
        return
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    rx = min(r_px * (ax.get_xlim()[1] - ax.get_xlim()[0]) / bb.width, w / 2)
    ry = min(r_px * (ax.get_ylim()[1] - ax.get_ylim()[0]) / bb.height, h)
    x0, x1 = x - w / 2, x + w / 2
    verts = [(x0, 0), (x0, h - ry), (x0, h), (x0 + rx, h), (x1 - rx, h), (x1, h), (x1, h - ry), (x1, 0), (x0, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
             Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def stairs_by_checkpoint(c, out):
    r = json.load(open("results/mujoco_stairs_by_checkpoint.json"))
    its = list(r)
    succ = [sum(e["outcome"] == "success" for e in r[it]) for it in its]
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=c["bg"])
    style(ax, c)
    ax.set_xlim(-0.6, len(its) - 0.4)
    ax.set_xticks(range(len(its)), its)
    ax.set_xlabel("training iteration (checkpoint)", color=c["muted"], fontsize=8)
    for i, s in enumerate(succ):
        rounded_bar(ax, i, s, 0.4, c["s1"])
    notes = {"100": "stops at\nthe stairs", "300": "falls on\nthe stairs", "400": "100% from here"}
    for i, it in enumerate(its):
        if it in notes:
            ax.annotate(f"{succ[i]}%  {notes[it]}" if it != "400" else notes[it], (i, succ[i]), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=c["t2"])
    ax.set_title("Stair-course success while training (MuJoCo, 100 episodes per checkpoint)",
                 color=c["t1"], fontsize=9, loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, facecolor=c["bg"])
    plt.close(fig)


if __name__ == "__main__":
    for mode, c in THEMES.items():
        robustness(c, f"docs/images/robustness_{mode}.png")
        stairs_by_checkpoint(c, f"docs/images/stairs_by_checkpoint_{mode}.png")
    print("wrote docs/images/{robustness,stairs_by_checkpoint}_{light,dark}.png")
