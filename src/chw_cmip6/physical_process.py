"""Shared map and correlation drawing primitives for Figs. 3–4 and ED Figs. 4 and 7."""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib.colors import LinearSegmentedColormap

from .scientific_constants import GIC_BOUNDS


PANEL_LABEL_PT = 23
PANEL_TITLE_PT = 19
AXIS_LABEL_PT = 16
AXIS_TICK_PT = 14.5
CBAR_TICK_PT = 14
CBAR_LABEL_PT = 14
MAP_TICK_PT = 14.5
ANNOTATION_PT = 14
PANEL_LABEL_X_WITH_TITLE = -0.06
PANEL_LABEL_Y_WITH_TITLE = 1.045
MAP_CBAR_LABEL_Y = 1
PHYSICS_RATIO_CMAP = LinearSegmentedColormap.from_list(
    "nature_percent_refined", ["#6FA3D8", "#F8F6F1", "#C96A3D"], N=256
)
PHYSICS_POSITIVE_BAR_COLOR = "#8DAE82"
PHYSICS_NEGATIVE_BAR_COLOR = "#C8A18C"
HOT_BOX_COLOR = "#BE4DD2"
NOTHOT_BOX_COLOR = "#1F9A8A"

HOT_REGIONS = (
    (-115, -97, 44, 59),
    (-14, 3, 12, 28),
    (54, 71, 56, 65),
    (69, 104, 27, 35),
    (100, 121, 54, 65),
)
UNDER_REGIONS = ((-106, -87, 30, 38), GIC_BOUNDS, (24, 38, 47, 66), (100, 119, 38, 52))
REGION_LABELS = ("NH", "NNA", "NAF", "EEU", "SAS", "ESB", "SNA", "GIC", "WEU", "ENA")
VARIABLE_TITLES = {
    "tcc": "TCC",
    "net": "NET",
    "q2m": "Q2M",
    "eddy_z500": "EDDY_Z500",
    "net_s": "NET_S",
    "net_l": "NET_L",
    "rlds": "DLR",
    "pr": "PR",
    "e": "E",
    "ef": "EF",
}


def configure_physical_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "font.size": AXIS_LABEL_PT,
            "axes.titlesize": PANEL_TITLE_PT,
            "axes.labelsize": AXIS_LABEL_PT,
            "xtick.labelsize": AXIS_TICK_PT,
            "ytick.labelsize": AXIS_TICK_PT,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def make_map(ax, box=(-180, 180, 0, 90), font_size=MAP_TICK_PT):
    ax.set_xticks(np.arange(box[0], box[1], 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(box[2], box[3] + 30, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis="both", which="major", labelsize=font_size, direction="out", length=5, width=1, pad=5)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis="both", which="minor", direction="out", length=2, width=0.4)
    ax.spines["geo"].set_linewidth(1)
    ax.set_extent(box, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, edgecolor="0.15", zorder=20)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", zorder=2)
    return ax


def apply_panel_header(ax, letter: str, title: str) -> None:
    ax.set_title(title, fontsize=PANEL_TITLE_PT, pad=5, loc="center", fontweight="normal")
    ax.text(
        PANEL_LABEL_X_WITH_TITLE,
        PANEL_LABEL_Y_WITH_TITLE,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_PT,
        fontweight="bold",
        color="0.1",
    )


def draw_rectangles(ax, regions, edgecolor: str) -> None:
    for lon_min, lon_max, lat_min, lat_max in regions:
        ax.add_patch(
            mpatches.Rectangle(
                (lon_min, lat_min),
                lon_max - lon_min,
                lat_max - lat_min,
                linewidth=2.0,
                edgecolor=edgecolor,
                facecolor="none",
                transform=ccrs.PlateCarree(),
                zorder=30,
            )
        )


def draw_ratio(ax, field, letter: str, title: str, *, hide_y: bool = False):
    make_map(ax)
    values = field.sortby("lat").sortby("lon")
    lon = np.asarray(values.lon)
    lat = np.asarray(values.lat)
    data = np.asarray(values, dtype=float)
    if len(lon) > 1 and np.isclose(lon[0], lon[-1]):
        lon, data = lon[:-1], data[:, :-1]
    data_cyclic, lon_cyclic = add_cyclic_point(np.ma.masked_invalid(data), coord=lon)
    lon2d, lat2d = np.meshgrid(lon_cyclic, lat)
    contour = ax.contourf(
        lon2d,
        lat2d,
        data_cyclic,
        levels=np.arange(0, 100, 1),
        cmap=PHYSICS_RATIO_CMAP,
        extend="both",
        transform=ccrs.PlateCarree(),
        transform_first=True,
        zorder=0,
    )
    draw_rectangles(ax, HOT_REGIONS, HOT_BOX_COLOR)
    draw_rectangles(ax, UNDER_REGIONS, NOTHOT_BOX_COLOR)
    apply_panel_header(ax, letter, title)
    if hide_y:
        ax.tick_params(labelleft=False)
    return contour


def draw_corr(ax, key: str, letter: str, title: str, dataset) -> None:
    r_values = np.asarray(dataset["r"].sel(variable=key), dtype=float)
    p_values = np.asarray(dataset["p"].sel(variable=key), dtype=float)
    x = np.arange(len(REGION_LABELS))
    colors = np.where(r_values >= 0, PHYSICS_POSITIVE_BAR_COLOR, PHYSICS_NEGATIVE_BAR_COLOR)
    ax.bar(x, r_values, color=colors, edgecolor="white", linewidth=0.5)
    for xi, (r_value, p_value) in enumerate(zip(r_values, p_values)):
        if np.isfinite(r_value) and np.isfinite(p_value) and p_value <= 0.05:
            y = r_value + 0.01 if r_value >= 0 else r_value - 0.15
            ax.text(xi, y, "*", ha="center", va="center", fontsize=ANNOTATION_PT + 12, color="0.2")
    ax.axhline(0, color="0.6", linewidth=0.9)
    ax.axvline(0.5, color="0.85", linestyle="--", linewidth=0.8)
    ax.axvline(5.5, color="0.85", linestyle="--", linewidth=0.8)
    apply_panel_header(ax, letter, title)
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_LABELS, rotation=40, fontsize=AXIS_TICK_PT)
    ax.set_ylim(-1, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(2.9, 0.91, "Overestimated", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=ANNOTATION_PT, color="0.2")
    ax.text(8.0, 0.91, "Underestimated", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=ANNOTATION_PT, color="0.2")
