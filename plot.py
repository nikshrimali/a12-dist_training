"""Turn the sweep and the N=32 runs into the pictures the guide uses.

    uv run plot.py --runs runs --out plots

Reads runs/sweep/sweep.json, runs/sweep/s<stage>_n<N>/timeline.json and runs/s<stage>_n32/.
Writes, when the inputs exist:
    memory.png    what one rank holds at N=32, by stage: between steps and at the peak
    scaling.png   memory per rank against N, one line per stage
    comm.png      bytes each rank sends per step against N, and time spent in collectives
    work.png      optimizer work per rank against N, and where the time goes at N=32
    timeline.png  every allocation and free during one step, one panel per stage
    loss.png      the four N=32 loss curves, and how far each is from DDP

Same style rules as a10: two measures never share an axis (stacked panels instead), direct
labels next to lines, text in ink colors. Stage colors and memory-category colors are two
separate palettes, so blue always means DDP and violet always means weights.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLET, MAGENTA, GREEN, GREY = "#6250d6", "#e87ba4", "#008300", "#c3c2b7"
STAGE_COLORS = {0: BLUE, 1: ORANGE, 2: AQUA, 3: YELLOW}
STAGE_NAMES = {0: "DDP", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}
# Where stages give the same number their lines sit on top of each other, so each later stage is
# drawn thinner: identical lines show up as nested bands instead of hiding one another.
STAGE_WIDTHS = {0: 7, 1: 5, 2: 3.2, 3: 1.8}
CAT_COLORS = {"params": VIOLET, "grads": MAGENTA, "optimizer": GREEN}
CAT_NAMES = {"params": "weights", "grads": "gradients", "optimizer": "optimizer state"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def MB(b):
    return b / 1e6


def label_line_ends(ax, labels, min_gap=0.06):
    """Write each line's label right of its last point. Labels with the same value merge into one
    ("ZeRO-1, ZeRO-2 ..."), and the rest are nudged apart so no two overlap. Call after the limits are set.

    Args:
        labels: [(x, y, name, value_text)]. min_gap: smallest vertical gap, as a fraction of the axis height.
    """
    merged = {}
    for x, y, name, value in labels:
        merged.setdefault((x, value), [x, y, []])[2].append(name)
    lo, hi = ax.get_ylim()
    frac = (lambda v: (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo))) if ax.get_yscale() == "log" else (
        lambda v: (v - lo) / (hi - lo))
    height_pt = ax.get_window_extent().height * 72 / ax.figure.dpi
    last = None
    for (x, value), (_, y, names) in sorted(merged.items(), key=lambda kv: kv[1][1]):
        f = frac(y) if last is None else max(frac(y), last + min_gap)
        last = f
        ax.annotate(f"{', '.join(names)} {value}", (x, y), xytext=(8, (f - frac(y)) * height_pt),
                    textcoords="offset points", va="center", color=INK2, fontsize=9, annotation_clip=False)


def by_stage(sweep):
    out = {}
    for r in sweep:
        out.setdefault(r["stage"], []).append(r)
    return {s: sorted(rs, key=lambda r: r["world"]) for s, rs in sorted(out.items())}


def stacked_rows(ax, rows, key, activations):
    """One horizontal stacked bar per stage: weights, gradients, optimizer state, then activations."""
    names = [STAGE_NAMES[r["stage"]] for r in rows][::-1]
    left = np.zeros(len(rows))
    for cat in CAT_COLORS:
        vals = np.array([MB(r[key][cat]) for r in rows][::-1])
        ax.barh(names, vals, left=left, color=CAT_COLORS[cat], height=0.6, edgecolor=SURFACE, linewidth=2,
                label=CAT_NAMES[cat])
        left += vals
    act = np.array([MB(activations)] * len(rows))
    ax.barh(names, act, left=left, color=GREY, height=0.6, edgecolor=SURFACE, linewidth=2, hatch="//",
            label="activations (not sharded)")
    for i, (l, r) in enumerate(zip(left, rows[::-1])):
        ax.text(l + act[i] + 1, i, f"{l:.1f} MB + {act[i]:.1f} MB activations", va="center", color=INK, fontsize=9)
    ax.grid(axis="y", visible=False)


def plot_memory(sweep, out, world=32):
    rows = [r for r in sweep if r["world"] == world]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["stage"])
    act = rows[0]["activations"] or 0
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    stacked_rows(axes[0], rows, "steady", act)
    axes[0].set_title(f"What one rank holds between steps, N = {world} (the ledger matches the formula to the byte)")
    stacked_rows(axes[1], rows, "peak", act)
    axes[1].set_title("At the peak moment inside a step: ZeRO-2 and ZeRO-3 borrow full buffers one unit at a time")
    axes[1].set_xlabel("MB per rank")
    axes[0].legend(loc="lower right", ncol=2)
    xmax = max(sum(r["steady"].values()) for r in rows) / 1e6 + act / 1e6
    axes[1].set_xlim(0, xmax * 1.45)
    fig.tight_layout()
    fig.savefig(f"{out}/memory.png", dpi=130)
    plt.close(fig)


def plot_scaling(sweep, out):
    stages = by_stage(sweep)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    labels = ([], [])
    for s, rs in stages.items():
        n = [r["world"] for r in rs]
        steady = [MB(sum(r["steady"].values())) for r in rs]
        peak = [MB(r["peak_total"]) for r in rs]
        for ax, y, lab in ((axes[0], steady, labels[0]), (axes[1], peak, labels[1])):
            ax.plot(n, y, color=STAGE_COLORS[s], linewidth=STAGE_WIDTHS[s], marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                    label=STAGE_NAMES[s])
            lab.append((n[-1], y[-1], STAGE_NAMES[s], f"{y[-1]:.1f} MB"))
    axes[0].set_title("Memory per rank between steps: only DDP stays flat as you add GPUs")
    axes[1].set_title("Peak memory per rank during a step")
    for ax in axes:
        ax.set_ylabel("MB per rank")
        ax.set_ylim(0, None)
    axes[0].legend(loc="upper right")
    axes[1].set_xscale("log", base=2)
    ns = sorted({r["world"] for r in sweep})
    axes[1].set_xticks(ns, [str(v) for v in ns])
    axes[1].set_xlim(ns[0] * 0.8, ns[-1] * 2.2)
    axes[1].set_xlabel("N, number of ranks (virtual GPUs)")
    fig.tight_layout()
    for ax, lab in zip(axes, labels):
        label_line_ends(ax, lab)
    fig.savefig(f"{out}/scaling.png", dpi=130)
    plt.close(fig)


def plot_comm(sweep, out):
    stages = by_stage(sweep)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    labels = ([], [])
    for s, rs in stages.items():
        rs = [r for r in rs if r["world"] > 1]
        n = [r["world"] for r in rs]
        psi_bytes = [r["bytes_per_number"] * r["padded_params"] for r in rs]
        ratio = [sum(r["sent_per_step"].values()) / pb for r, pb in zip(rs, psi_bytes)]
        ms = [1000 * r["comm_time"] for r in rs]
        for ax, y, fmt, lab in ((axes[0], ratio, "{:.2f} Psi", labels[0]), (axes[1], ms, "{:.0f} ms", labels[1])):
            ax.plot(n, y, color=STAGE_COLORS[s], linewidth=STAGE_WIDTHS[s], marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                    label=STAGE_NAMES[s])
            lab.append((n[-1], y[-1], STAGE_NAMES[s], fmt.format(y[-1])))
    for level, text in ((2, "2 Psi: the paper's figure for DDP, ZeRO-1, ZeRO-2"), (3, "3 Psi: the paper's figure for ZeRO-3")):
        axes[0].axhline(level, color=INK2, linestyle="--", linewidth=1)
        axes[0].text(2, level + 0.04, " " + text, color=INK2, va="bottom", fontsize=9)
    axes[0].set_title("Bytes each rank sends per step, over the model's size in bytes (ring cost: 2 Psi x (N-1)/N)")
    axes[0].set_ylabel("sent / model size")
    axes[0].set_ylim(0, 3.4)
    axes[0].legend(loc="lower right")
    axes[1].set_title("Time inside collectives per step (gloo over local TCP, so only the trend means anything)")
    axes[1].set_ylabel("ms per step")
    axes[1].set_ylim(0, None)
    axes[1].set_xscale("log", base=2)
    ns = sorted({r["world"] for r in sweep if r["world"] > 1})
    axes[1].set_xticks(ns, [str(v) for v in ns])
    axes[1].set_xlim(ns[0] * 0.8, ns[-1] * 2.2)
    axes[1].set_xlabel("N, number of ranks (virtual GPUs)")
    fig.tight_layout()
    for ax, lab in zip(axes, labels):
        label_line_ends(ax, lab)
    fig.savefig(f"{out}/comm.png", dpi=130)
    plt.close(fig)


def plot_work(sweep, runs, out):
    stages = by_stage(sweep)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
    ax = axes[0]
    work_labels = []
    for s, rs in stages.items():
        n = [r["world"] for r in rs]
        y = [r["optimizer_numbers_per_rank"] / 1e6 for r in rs]
        ax.plot(n, y, color=STAGE_COLORS[s], linewidth=STAGE_WIDTHS[s], marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label=STAGE_NAMES[s])
        work_labels.append((n[-1], y[-1], STAGE_NAMES[s], f"{y[-1] * 1e3:,.0f}K"))
    fwd = stages[0][0]["fwd_bwd_flops_per_rank"]
    ax.set_title(f"Numbers each rank's AdamW updates per step\n(forward + backward: {fwd / 1e9:.1f} GFLOP per rank at every stage)")
    ax.set_ylabel("millions of numbers")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ns = sorted({r["world"] for r in sweep})
    ax.set_xticks(ns, [str(v) for v in ns])
    ax.set_xlim(ns[0] * 0.8, ns[-1] * 2.4)
    ax.set_xlabel("N, number of ranks (virtual GPUs)")
    ax.legend(loc="lower left")
    ax = axes[1]
    rows = []
    for s in range(4):
        path = f"{runs}/s{s}_n32/metrics.jsonl"
        if os.path.exists(path):
            tr = [r for r in read_jsonl(path) if r["type"] == "train"][1:]
            m = lambda k: 1000 * np.mean([r[k] for r in tr])
            rows.append((STAGE_NAMES[s], m("step_time") - m("comm_time") - m("opt_time"), m("comm_time"), m("opt_time")))
    if rows:
        names = [r[0] for r in rows][::-1]
        left = np.zeros(len(rows))
        for j, (label, color) in enumerate((("forward + backward", "#52514e"), ("collectives", "#a3a29c"),
                                            ("optimizer", "#d6d5d0")), start=1):
            vals = np.array([r[j] for r in rows][::-1])
            ax.barh(names, vals, left=left, color=color, height=0.6, edgecolor=SURFACE, linewidth=2, label=label)
            left += vals
        for i, (l, r) in enumerate(zip(left, rows[::-1])):
            ax.text(l + 50, i, f"{l:,.0f} ms  (collectives {r[2]:,.0f}, optimizer {r[3]:.1f})", va="center", color=INK, fontsize=9)
        ax.set_xlim(0, left.max() * 1.5)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")
        ax.set_xlabel("ms per step, mean of steps 2 onward")
        ax.set_title("Where one step's time goes at N = 32 (32 processes share 10 CPU cores)")
    fig.tight_layout()
    label_line_ends(axes[0], work_labels)
    fig.savefig(f"{out}/work.png", dpi=130)
    plt.close(fig)


def plot_timeline(runs, out, world):
    paths = [f"{runs}/sweep/s{s}_n{world}/timeline.json" for s in range(4)]
    if not all(os.path.exists(p) for p in paths):
        return
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharey=True)
    for s, (ax, path) in enumerate(zip(axes, paths)):
        ev = json.load(open(path))
        x = np.arange(len(ev))
        ys = [np.array([MB(e[c]) for e in ev]) for c in CAT_COLORS]
        ax.stackplot(x, *ys, colors=list(CAT_COLORS.values()), step="post", labels=[CAT_NAMES[c] for c in CAT_COLORS],
                     edgecolor=SURFACE, linewidth=0.5)
        for i, e in enumerate(ev):
            if e["what"].endswith("starts") and e["phase"] in ("forward", "backward", "optimizer step"):
                ax.axvline(i, color=INK2, linestyle="--", linewidth=1)
                ax.annotate(f" {e['phase']}", (i, 1), xycoords=("data", "axes fraction"), va="top", color=INK2, fontsize=9)
        total = sum(ys)
        ax.set_title(f"{STAGE_NAMES[s]}: memory on rank 0 through one step, N = {world}, peak {total.max():.1f} MB")
        ax.set_ylabel("MB")
        ax.set_xlim(0, len(ev) - 1)
        ax.grid(axis="x", visible=False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].legend(loc="center right")
    axes[-1].set_xlabel("allocation or free events, in order (not seconds)")
    fig.tight_layout()
    fig.savefig(f"{out}/timeline.png", dpi=130)
    plt.close(fig)


def plot_loss(runs, out):
    curves = {}
    for s in range(4):
        path = f"{runs}/s{s}_n32/metrics.jsonl"
        if os.path.exists(path):
            curves[s] = [r for r in read_jsonl(path) if r["type"] == "train"]
    if 0 not in curves:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    base = np.array([r["loss"] for r in curves[0]])
    for s, tr in curves.items():
        steps = [r["step"] for r in tr]
        loss = np.array([r["loss"] for r in tr])
        axes[0].plot(steps, loss, color=STAGE_COLORS[s], linewidth=3.5 if s == 0 else 1.6,
                     linestyle="-" if s == 0 else (0, (4, 2 + s)), label=STAGE_NAMES[s])
        if s:
            n = min(len(loss), len(base))
            axes[1].plot(steps[:n], loss[:n] - base[:n], color=STAGE_COLORS[s], marker="o", markersize=4,
                         label=STAGE_NAMES[s])
    axes[0].set_title("Training loss at N = 32: the four stages draw one line")
    axes[0].set_ylabel("loss (nats per token)")
    axes[0].legend(loc="upper right")
    axes[1].axhline(0, color=INK2, linewidth=1)
    axes[1].set_title("Each ZeRO stage's loss minus DDP's loss, same step")
    axes[1].set_ylabel("difference")
    if len(curves) > 1:
        axes[1].legend(loc="upper right")
    axes[-1].set_xlabel("optimizer step")
    fig.tight_layout()
    fig.savefig(f"{out}/loss.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--timeline-world", type=int, default=4, help="N of the sweep runs whose timelines to draw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sweep_path = f"{args.runs}/sweep/sweep.json"
    if os.path.exists(sweep_path):
        sweep = json.load(open(sweep_path))
        plot_memory(sweep, args.out)
        plot_scaling(sweep, args.out)
        plot_comm(sweep, args.out)
        plot_work(sweep, args.runs, args.out)
    plot_timeline(args.runs, args.out, args.timeline_world)
    plot_loss(args.runs, args.out)
    print(f"wrote plots to {args.out}/")


if __name__ == "__main__":
    main()
