#!/usr/bin/env python3
"""
Track B – Step 11: paper figures for the OOD ladder.

  fig_ladder.pdf/.png     accuracy across the ordered rungs (E39/E41)
  fig_ani_curve.pdf/.png  accuracy vs ANI to the nearest training genome (E40)
  fig_mechanism.pdf/.png  MT vs parameter-free k-mer NB, per rung (E42)

The rungs are ordered by increasing distance from training, so a line across
them reads as "how far does each method fall as the test organism gets further
from anything it saw" -- which is the question. Bars would need 20 of them to
say the same thing.

Colors are the validated 5-slot categorical order (see references/palette.md);
three of them sit below 3:1 on the light surface, so every series also carries a
direct label at its right end and the underlying numbers ship as TSV.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SERIES_COLOR = {
    "mt_250M": "#2a78d6",
    "mt_50M": "#eb6834",
    "mt_6mer": "#1baf7a",
    "nt_v9": "#eda100",
    "kraken2": "#e87ba4",
    "nb_13mer": "#4a3aa7",
    "nb_6mer": "#e34948",
}
SERIES_LABEL = {
    "mt_250M": "MT 13-mer 250M",
    "mt_50M": "MT 13-mer 50M",
    "mt_6mer": "MT 6-mer 50M",
    "nt_v9": "NT-v2 6-mer (RC-TTA)",
    "kraken2": "Kraken2 (1,535 DB)",
    "nb_13mer": "13-mer NB (no NN)",
    "nb_6mer": "6-mer NB (no NN)",
}
RUNG_ORDER = ["closed_set", "L1_strain", "L2_species", "L2_far"]
RUNG_LABEL = {
    "closed_set": "closed set\n(same genome)*",
    "L1_strain": "unseen strain\n(ANI 95–99)",
    "L2_species": "novel species\n(ANI 80–95)",
    "L2_far": "novel species\n(no measurable ANI)",
}

INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def style_axes(ax, ymax=100):
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)


def place_end_labels(ax, x, entries, ymax=100.0, min_gap=3.2):
    """Draw right-end value labels, nudged apart so near-tied series stay legible.

    Several methods converge to within a point of each other at the far end of
    the ladder -- which is the finding -- so the labels collide exactly where
    the reader most needs them.
    """
    entries = sorted((v for v in entries if v[0] is not None and not pd.isna(v[0])),
                     key=lambda e: e[0])
    placed = []
    for value, text, color in entries:
        y = value
        if placed and y - placed[-1] < min_gap:
            y = placed[-1] + min_gap
        placed.append(y)
    # Pull back down if the nudging pushed the stack past the top of the axes.
    if placed and placed[-1] > ymax:
        shift = placed[-1] - ymax
        placed = [p - shift for p in placed]
    for (value, text, color), y in zip(entries, placed):
        ax.annotate(text, xy=(x, y), xytext=(7, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8, color=INK, zorder=4)


def save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_dir / (stem + '.pdf')}")


def line_over_rungs(df: pd.DataFrame, models: list[str], title: str,
                    ylabel: str, out_dir: Path, stem: str, anchors: dict | None):
    """One line per model across the ordered rungs."""
    rungs = [r for r in RUNG_ORDER
             if r in set(df["level"]) or (anchors and r == "closed_set")]
    x = range(len(rungs))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    style_axes(ax)

    drawn = []
    for model in models:
        ys = []
        for rung in rungs:
            if rung == "closed_set":
                v = anchors.get(model) if anchors else None
                ys.append(v * 100 if v is not None else float("nan"))
            else:
                sub = df[(df["level"] == rung) & (df["model"] == model)]
                ys.append(float(sub["read_acc_all"].iloc[0]) * 100
                          if len(sub) else float("nan"))
        if all(pd.isna(v) for v in ys):
            continue
        color = SERIES_COLOR[model]
        ax.plot(list(x), ys, color=color, linewidth=2.0, marker="o",
                markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                zorder=3, label=SERIES_LABEL[model])
        drawn.append((model, ys))

    # Direct labels at the right end -- required relief for the low-contrast
    # slots, and it keeps the legend from being the only identity channel.
    place_end_labels(ax, len(rungs) - 1,
                     [(next((v for v in reversed(ys) if not pd.isna(v)), None),
                       f"{next((v for v in reversed(ys) if not pd.isna(v)), 0):.1f}%",
                       SERIES_COLOR[model]) for model, ys in drawn])

    ax.set_xticks(list(x))
    ax.set_xticklabels([RUNG_LABEL[r] for r in rungs], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left", pad=10)
    ax.set_xlim(-0.25, len(rungs) - 0.55)
    if anchors and "closed_set" in rungs:
        ax.axvline(0.5, color=AXIS, linewidth=0.8, linestyle=(0, (3, 3)),
                   zorder=1)
        ax.text(0.5, -0.20, "* closed-set column is the matched full-100K pool "
                            "(RESULTS_SUMMARY §D2), not a Track B pool",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=7, color=INK_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper right",
              handlelength=1.6, labelspacing=0.35)
    save(fig, out_dir, stem)


def ani_curve(df: pd.DataFrame, out_dir: Path):
    df = df.sort_values("bin_order")
    bins = list(dict.fromkeys(df["ani_bin"]))
    x = range(len(bins))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    style_axes(ax)

    ends = []
    for model in [m for m in SERIES_COLOR if m in set(df["model"])]:
        sub = df[df["model"] == model].set_index("ani_bin")
        ys = [float(sub.loc[b, "read_acc"]) * 100 if b in sub.index else float("nan")
              for b in bins]
        ax.plot(list(x), ys, color=SERIES_COLOR[model], linewidth=2.0,
                marker="o", markersize=5.5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3, label=SERIES_LABEL[model])
        last = next((v for v in reversed(ys) if not pd.isna(v)), None)
        if last is not None:
            ends.append((last, f"{last:.0f}%", SERIES_COLOR[model]))
    place_end_labels(ax, len(bins) - 1, ends)

    counts = (df.groupby("ani_bin")["n_reads"].max().reindex(bins).fillna(0))
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{b}\nn={int(counts[b]):,}" for b in bins], fontsize=7.5)
    ax.set_xlabel("ANI to the closest training genome of the same genus (%)")
    ax.set_ylabel("genus Top-1 accuracy (%)")
    ax.set_title("Generalisation radius: accuracy vs distance from training",
                 fontsize=10, loc="left", pad=10)
    ax.set_xlim(-0.25, len(bins) - 0.55)
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              handlelength=1.6, labelspacing=0.35)
    save(fig, out_dir, "fig_ani_curve")


def paired_dumbbell(df: pd.DataFrame, out_dir: Path):
    """Within-genus near/far comparison -- the composition-free view of E40.

    Same 45 genera on both ends of every row, so the gap is the effect of the
    test organism being a novel species, not of which genera landed in which
    ANI bin.
    """
    df = df.sort_values("mean_delta")
    fig, ax = plt.subplots(figsize=(6.8, 0.62 * len(df) + 1.6))
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    for i, (_, r) in enumerate(df.iterrows()):
        color = SERIES_COLOR.get(r["model"], "#2a78d6")
        near, far = r["mean_acc_near_ge95"] * 100, r["mean_acc_far_lt95"] * 100
        ax.plot([far, near], [i, i], color=color, linewidth=2.0,
                solid_capstyle="round", zorder=2)
        ax.scatter([near], [i], s=90, facecolor=color, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3)
        ax.scatter([far], [i], s=90, facecolor=SURFACE, edgecolor=color,
                   linewidth=2.0, zorder=3)
        ax.annotate(f"{r['mean_delta']*100:+.1f} pp   "
                    f"({r['n_genera_worse_when_far']}/{r['n_genera_paired']} worse)",
                    xy=(max(near, far), i), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([SERIES_LABEL.get(m, m) for m in df["model"]], fontsize=8.5)
    ax.set_xlabel("genus Top-1 accuracy (%), averaged over the paired genera")
    ax.set_xlim(0, 108)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_title("Same genera, near vs far: the long-k advantage is a "
                 "distance effect", fontsize=10, loc="left", pad=10)
    ax.scatter([], [], s=90, facecolor=INK_MUTED, edgecolor=SURFACE,
               linewidth=1.5, label="closest training genome ANI ≥ 95 (known species)")
    ax.scatter([], [], s=90, facecolor=SURFACE, edgecolor=INK_MUTED,
               linewidth=2.0, label="ANI < 95 (novel species)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", scatterpoints=1,
              bbox_to_anchor=(1.0, 1.02))
    save(fig, out_dir, "fig_paired_near_far")


def ood_accept_by_rung(df: pd.DataFrame, out_dir: Path, kraken: dict | None):
    """E44: acceptance per rung at one fixed in-distribution operating point.

    All neural detectors are pinned to the same 95% in-distribution acceptance,
    so the lines are comparable to each other. Kraken2 cannot be tuned -- it has
    a single operating point -- so it is drawn as isolated markers and labelled
    as such rather than as a fourth tunable curve.
    """
    rungs = ["L1_strain", "L2_species", "L2_far", "L3_genus"]
    labels = {"L1_strain": "unseen strain\n(should classify)",
              "L2_species": "novel species\n(should classify)",
              "L2_far": "novel species, far\n(should classify)",
              "L3_genus": "novel GENUS\n(should reject)"}
    have = [r for r in rungs if f"accept_rate_{r}" in df.columns]
    x = range(len(have))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    style_axes(ax)

    ends = []
    for model in [m for m in SERIES_COLOR if m in set(df["model"])]:
        row = df[df["model"] == model].iloc[0]
        ys = [float(row[f"accept_rate_{r}"]) * 100 for r in have]
        ax.plot(list(x), ys, color=SERIES_COLOR[model], linewidth=2.0, marker="o",
                markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                zorder=3, label=SERIES_LABEL[model])
        ends.append((ys[-1], f"{ys[-1]:.0f}%", SERIES_COLOR[model]))

    if kraken:
        ks = [kraken.get(r) for r in have]
        xs = [i for i, v in enumerate(ks) if v is not None]
        vs = [ks[i] * 100 for i in xs]
        ax.scatter(xs, vs, s=110, marker="D", facecolor=SURFACE,
                   edgecolor=SERIES_COLOR["kraken2"], linewidth=2.0, zorder=4,
                   label="Kraken2 — commits (single fixed operating point)")
        if vs:
            ends.append((vs[-1], f"{vs[-1]:.0f}%", SERIES_COLOR["kraken2"]))

    place_end_labels(ax, len(have) - 1, ends)
    ax.set_xticks(list(x))
    ax.set_xticklabels([labels[r] for r in have], fontsize=7.5)
    ax.set_ylabel("reads accepted / committed (%)")
    ax.set_title("No method tells a distant novel species from a novel genus",
                 fontsize=10, loc="left", pad=10)
    ax.set_xlim(-0.25, len(have) - 0.55)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", scatterpoints=1)
    ax.text(0.5, -0.30, "Neural curves are pinned to 95% in-distribution "
                        "acceptance (entropy score). The last two rungs differ by "
                        "<2 pp for every method,\nKraken2 included (8.7% vs 7.4%): "
                        "all of these signals track sequence distance, not whether "
                        "the true genus is in the label space.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7,
            color=INK_MUTED)
    save(fig, out_dir, "fig_ood_accept")


def main():
    args = parse_args()
    res = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    ladder_path = res / "ladder_accuracy.tsv"
    if not ladder_path.exists():
        raise SystemExit(f"missing {ladder_path} -- run tb_07 first")
    ladder = pd.read_csv(ladder_path, sep="\t")

    import json
    anchors = {}
    metrics_path = res / "metrics.json"
    if metrics_path.exists():
        anchors = json.loads(metrics_path.read_text()).get("closed_set_anchor", {})

    neural = ["mt_250M", "mt_50M", "mt_6mer", "nt_v9", "kraken2"]
    line_over_rungs(ladder, [m for m in neural if m in set(ladder["model"])],
                    "Genus classification falls off as the test species leaves training",
                    "genus Top-1 accuracy (%)", out_dir, "fig_ladder", anchors)

    mech = ["mt_250M", "nb_13mer", "mt_6mer", "nb_6mer"]
    present = [m for m in mech if m in set(ladder["model"])]
    if len([m for m in present if m.startswith("nb_")]) >= 1:
        line_over_rungs(ladder, present,
                        "Mechanism: long-k advantage tracks a parameter-free k-mer counter",
                        "genus Top-1 accuracy (%)", out_dir, "fig_mechanism", anchors)
    else:
        print("  (skipping fig_mechanism: no NB baseline in ladder_accuracy.tsv)")

    ani_path = res / "ani_curve.tsv"
    if ani_path.exists():
        ani_curve(pd.read_csv(ani_path, sep="\t"), out_dir)
    else:
        print(f"  (skipping fig_ani_curve: no {ani_path})")

    paired_path = res / "ani_paired_near_far.tsv"
    if paired_path.exists():
        paired_dumbbell(pd.read_csv(paired_path, sep="\t"), out_dir)
    else:
        print(f"  (skipping fig_paired_near_far: no {paired_path})")

    ood_path = res / "ood_detection.tsv"
    if ood_path.exists():
        ood = pd.read_csv(ood_path, sep="\t")
        ood = ood[ood["score"] == "entropy"]
        kraken = None
        ood_json_path = res / "ood_detection.json"
        if ood_json_path.exists():
            kraken = json.loads(ood_json_path.read_text()).get("kraken2_operating_point")
        ood_accept_by_rung(ood, out_dir, kraken)
    else:
        print(f"  (skipping fig_ood_accept: no {ood_path})")


if __name__ == "__main__":
    main()
