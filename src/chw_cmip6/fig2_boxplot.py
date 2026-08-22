"""Reusable regional boxplot used by Figure 2 and Extended Data Figure 3."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D


BOX_COLORS = {"Daytime": "#E69F00", "Nighttime": "#56B4E9", "Compound": "#39AB74"}
OBSERVATION_MARKERS = ["o", "s", "D", "*", "^"]


def draw_regional_heatwave_boxplot(ax, payload, *, ylabel, ylim, title=None):
    """Draw the verified regional CMIP-member and observation comparison."""
    labels = list(np.asarray(payload["labels"], dtype=object))
    observation_names = list(np.asarray(payload["obs_names"], dtype=object))
    datasets = [
        ("Daytime", payload["cmip_all_day"], payload["obs_all_day"]),
        ("Nighttime", payload["cmip_all_night"], payload["obs_all_night"]),
        ("Compound", payload["cmip_all_compound"], payload["obs_all_compound"]),
    ]
    positions_base = np.arange(1, len(labels) + 1)
    for dataset_index, (kind, cmip, obs) in enumerate(datasets):
        values = np.asarray(cmip, dtype=float)
        observations = np.asarray(obs, dtype=float)
        positions = positions_base + (-0.2, 0.0, 0.2)[dataset_index]
        artists = ax.boxplot(values.T, positions=positions, widths=0.18, patch_artist=True,
                             showfliers=False, whis=(5, 95), zorder=1)
        color = BOX_COLORS[kind]
        for box in artists["boxes"]:
            box.set(facecolor=color, alpha=0.28, edgecolor=color, linewidth=1.0)
        for whisker in artists["whiskers"]:
            whisker.set(color=color, linewidth=0.8, linestyle="--", alpha=0.8)
        for cap in artists["caps"]:
            cap.set(color=color, linewidth=0.8)
        for median in artists["medians"]:
            median.set(color=color, linewidth=1.2)
        ax.scatter(positions, values.mean(axis=1), marker="x", s=110, linewidth=2, color=color, zorder=4)
        for obs_index, obs_name in enumerate(observation_names):
            for region_index in range(len(labels)):
                ax.scatter(positions[region_index], observations[obs_index, region_index],
                           marker=OBSERVATION_MARKERS[obs_index], s=95, facecolors="none",
                           edgecolor="black", alpha=0.65, zorder=4,
                           label=obs_name if dataset_index == 0 and region_index == 0 else None)

    ax.axvline(1.5, linestyle="--", alpha=0.25, color="gray")
    ax.axvline(6.5, linestyle="--", alpha=0.25, color="gray")
    ax.axhline(0, linestyle="--", linewidth=1, color="gray", alpha=0.6)
    ax.set_xticks(positions_base)
    ax.set_xticklabels(labels, fontsize=20.9)
    ax.set_xlim(0.35, len(labels) + 0.65)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=20.8)
    ax.tick_params(axis="y", labelsize=18.85)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    if title:
        ax.set_title(title, fontsize=22.7, pad=5, fontweight="normal")
    ax.text(4.0, ylim[1] - 0.07, "Overestimated", ha="center", fontsize=18.2, color="0.2")
    ax.text(8.0, ylim[1] - 0.07, "Underestimated", ha="center", fontsize=18.2, color="0.2")

    kind_legend = ax.legend(
        handles=[mpatches.Patch(facecolor=BOX_COLORS[name], edgecolor=BOX_COLORS[name], alpha=0.35,
                                label=f"{name} HW") for name in ("Daytime", "Nighttime", "Compound")],
        loc="upper left", bbox_to_anchor=(0.02, 1.03), frameon=False, fontsize=17.6)
    ax.add_artist(kind_legend)
    handles, legend_labels = ax.get_legend_handles_labels()
    legend_map = dict(zip(legend_labels, handles))
    legend_map["MME"] = Line2D([0], [0], marker="x", color="black", linestyle="None",
                                markersize=12.35, markeredgewidth=2, label="MME")
    order = ["MME", "ERA5", "MERRA2", "JRA-3Q", "CPC", "BEST"]
    present = [name for name in order if name in legend_map]
    observation_legend = ax.legend([legend_map[name] for name in present], present, frameon=False,
                                   fontsize=17.6, ncol=6, loc="upper center", bbox_to_anchor=(0.60, 0.082),
                                   columnspacing=1.0, handletextpad=0.4, markerscale=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    return kind_legend, observation_legend
