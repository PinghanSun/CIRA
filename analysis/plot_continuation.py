"""Render the continuation figures from the companion data."""

from argparse import ArgumentParser
from pathlib import Path

from recompute import (
    check_manifest,
    continuation,
    group,
    paired_interval,
    read,
    set_data_root,
)

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle
from matplotlib.transforms import blended_transform_factory


DARK = "#293B45"
SERIES = (
    ("lineage", "On deployment lineage", "#518CA6", "#DCEDF2"),
    ("retained", "Retained", "#659584", None),
    ("completed", "Completed", "#D4A53B", "#FFF1C4"),
)
BALL_SIZE = 6.0


def ball_layers(x, y, edge, highlight):
    dark = to_rgb(edge)
    light = to_rgb(highlight)
    radius = BALL_SIZE / 2
    layers = []
    for i in range(14):
        fraction = i / 14
        shade = 0.14 + 0.86 * (1 - (1 - fraction) ** 1.7)
        color = tuple((1 - shade) * k + shade * v for k, v in zip(dark, light))
        layers.append(
            Circle(
                (x - 0.30 * fraction, y + 0.35 * fraction),
                radius * (1 - fraction),
                facecolor=color,
                edgecolor="none",
            )
        )
    layers.append(
        Circle((x, y), radius, facecolor="none", edgecolor=edge, linewidth=0.68)
    )
    return layers


class BallLegend(HandlerBase):
    def __init__(self, edge, highlight):
        super().__init__()
        self.edge = edge
        self.highlight = highlight

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        markers = ball_layers(
            width / 2 - xdescent, height / 2 - ydescent, self.edge, self.highlight
        )
        for marker in markers:
            marker.set_transform(trans)
        return markers


def plot_s3(output, counts):
    data = [
        dict(
            zip(("completed", "retained", "lineage"), counts[f"{dataset}/{direction}"])
        )
        for dataset in ("ACDC", "ControlledFog")
        for direction in ("membership_to_configuration", "configuration_to_membership")
    ]

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Arial", "serif"],
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.65,
            "text.color": DARK,
            "axes.labelcolor": DARK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(5.8, 1.95))
    fig.subplots_adjust(left=0.162, right=0.983, bottom=0.198, top=0.863)
    positions = (3.25, 2.35, 1.05, 0.15)
    ax.set_yticks([])
    ax.set_ylim(-0.38, 4.00)
    ax.set_xlim(0, 190)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xlabel("Number of simulated cross-family events", labelpad=2.6)
    ax.tick_params(axis="both", width=0.65)
    ax.tick_params(axis="x", length=2.4, pad=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#81909A")
    ax.axhspan(1.70, 4.00, facecolor="#F0F5F6", linewidth=0, zorder=-1)
    ax.axhline(1.70, color="#CCD6DB", linewidth=0.55, zorder=1)

    for y, row in zip(positions, data):
        ax.hlines(y, 0, row["completed"], color="#A7B7BE", linewidth=0.75, zorder=2)
        for key, _, color, highlight in SERIES:
            value = row[key]
            if highlight is None:
                ax.plot(
                    value,
                    y,
                    linestyle="none",
                    marker="D",
                    color=color,
                    markersize=4.6,
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    zorder=3,
                )
            else:
                box = DrawingArea(BALL_SIZE, BALL_SIZE)
                for layer in ball_layers(
                    BALL_SIZE / 2, BALL_SIZE / 2, color, highlight
                ):
                    box.add_artist(layer)
                ax.add_artist(
                    AnnotationBbox(box, (value, y), frameon=False, pad=0, zorder=3)
                )
            ax.annotate(
                str(value),
                (value, y),
                xytext=(0, 5.0),
                textcoords="offset points",
                fontsize=8.4,
                color=DARK,
                ha="center",
                va="bottom",
                zorder=4,
            )

    text_transform = blended_transform_factory(fig.transFigure, ax.transData)
    for dataset, top, bottom in [
        ("ACDC", positions[0], positions[1]),
        ("ControlledFog", positions[2], positions[3]),
    ]:
        fig.text(
            0.016,
            (top + bottom) / 2,
            dataset,
            transform=text_transform,
            weight="bold",
            va="center",
            ha="left",
            fontsize=8.5,
        )

    handles = [
        Line2D(
            [],
            [],
            marker="D" if highlight is None else "o",
            linestyle="none",
            color=color,
            markersize=4.6,
        )
        for _, _, color, highlight in SERIES
    ]
    marker_handlers = {
        handle: BallLegend(color, highlight)
        for handle, (_, _, color, highlight) in zip(handles, SERIES)
        if highlight is not None
    }
    legend = fig.legend(
        handles,
        [label for _, label, _, _ in SERIES],
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=1,
        edgecolor="#CCD6DB",
        fontsize=8.4,
        borderpad=0.27,
        handletextpad=0.4,
        columnspacing=1.25,
        loc="center right",
        bbox_to_anchor=(0.983, 0.944),
        borderaxespad=0,
        handler_map=marker_handlers,
    )
    legend.get_frame().set_linewidth(0.55)

    output.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(
        output / "Figure_S3.pdf",
        metadata={
            "Title": "Simulated cross-family events for the Full policy",
            "Subject": "Descriptive counts in 16 simulated Full trajectories, eight per dataset",
        },
    )
    fig.savefig(output / "Figure_S3.svg")
    fig.savefig(output / "Figure_S3.png", dpi=600)
    plt.close(fig)


S2_NAME = "Figure_S2"
S2_GREEN = "#588D7B"
S2_DARK = "#33434C"


def plot_s2(output):
    data = group(read("yolov8n_continuation_preset_pairs.csv"), "dataset")
    values = {
        dataset: [
            float(r["full_seen_ap_pct"]) - float(r["strict_seen_ap_pct"])
            for r in sorted(data[dataset,], key=lambda x: int(x["seed"]))
        ]
        for dataset in ("ACDC", "ControlledFog")
    }
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Arial", "serif"],
            "font.size": 8.4,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": S2_DARK,
            "axes.labelcolor": S2_DARK,
            "axes.linewidth": 0.65,
            "xtick.color": S2_DARK,
            "ytick.color": S2_DARK,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.018,
        }
    )
    fig, ax = plt.subplots(figsize=(4.75, 1.43))
    fig.subplots_adjust(left=0.186, right=0.974, bottom=0.231, top=0.833)
    ax.set_xlim(-0.23, 1.72)
    ax.set_ylim(-0.30, 1.33)
    ax.set_xticks(
        [-0.2, 0, 0.4, 0.8, 1.2, 1.6], ["−0.2", "0", "0.4", "0.8", "1.2", "1.6"]
    )
    ax.set_yticks([1, 0], list(values), fontweight="bold")
    ax.tick_params(axis="both", length=2.3, pad=2.1, width=0.65)
    ax.set_xlabel(
        "Full − Strict No-cross in final SeenAP$_{50{:}95}$ (pp)", labelpad=2.3
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#75828C")

    ax.axhspan(0.5, ax.get_ylim()[1], color="#F1F5F7", linewidth=0, zorder=0)
    ax.axhline(0.5, color="#CCD6DD", linewidth=0.65, zorder=1)
    ax.grid(axis="x", color="#E5EAED", linewidth=0.55, zorder=1)
    ax.axvline(0, color="#60717D", linestyle=(0, (3, 2)), linewidth=0.8, zorder=2)

    for y, (dataset, values) in zip((1, 0), values.items()):
        estimate, low, high = paired_interval(values)
        half_width = high - estimate
        assert (
            ax.get_xlim()[0]
            < min(values + [low])
            < max(values + [high])
            < ax.get_xlim()[1]
        )
        ax.scatter(
            values,
            [y + 0.17] * len(values),
            s=11,
            c=S2_GREEN,
            alpha=0.76,
            edgecolors="white",
            linewidths=0.35,
            zorder=4,
        )
        ax.errorbar(
            estimate,
            y - 0.07,
            xerr=half_width,
            fmt="o",
            color=S2_GREEN,
            markerfacecolor=S2_GREEN,
            markeredgecolor=S2_DARK,
            markersize=4.5,
            markeredgewidth=0.55,
            linewidth=1.0,
            capsize=1.7,
            capthick=0.75,
            zorder=5,
        )
        print(
            f"{dataset}: {len(values)} ordered seeds, {estimate:+.6f} [{low:.6f}, {high:.6f}]"
        )

    fig.text(0.186, 0.909, "Numerical comparison", va="center", fontsize=8.2)
    seed_key = Line2D(
        [],
        [],
        color=S2_GREEN,
        linestyle="none",
        marker="o",
        markersize=3,
        markeredgecolor="white",
        markeredgewidth=0.35,
        alpha=0.76,
    )
    mean_key = Line2D(
        [],
        [],
        color=S2_GREEN,
        marker="o",
        markersize=4.0,
        markeredgecolor=S2_DARK,
        markeredgewidth=0.55,
        linewidth=1.0,
    )
    legend = fig.legend(
        [seed_key, mean_key],
        ["Paired seed", "Mean and 95% paired $t$ CI"],
        loc="center right",
        bbox_to_anchor=(0.974, 0.909),
        ncol=2,
        handlelength=1.5,
        handletextpad=0.42,
        columnspacing=0.9,
        borderaxespad=0,
        borderpad=0.25,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="#CCD6DD",
        fontsize=7.5,
    )
    legend.get_frame().set_linewidth(0.55)

    fig.canvas.draw()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output / f"{S2_NAME}.pdf",
        metadata={
            "Title": "Full vs. Strict No-cross final SeenAP differences",
            "Subject": "Numerical comparison across eight matched seeds per dataset",
        },
    )
    fig.savefig(output / f"{S2_NAME}.svg")
    fig.savefig(output / f"{S2_NAME}.png", dpi=600)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_data_root(args.data_root)
    check_manifest()
    counts = continuation()["figure_s3_completed_retained_lineage"]
    plot_s2(args.output)
    plot_s3(args.output, counts)


if __name__ == "__main__":
    main()
