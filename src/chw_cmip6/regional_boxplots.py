"""Extended Data Figs. 5–6 regional standardized-variable boxplots."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .config import ProjectConfig
from .data_io import MissingInputError, require_files, require_variables
from .plotting import save_figure
from .scientific_constants import (
    EXTENDED_DATA_5_6_VARIABLES,
    OVER_ESTIMATED_REGIONS,
    UNDER_ESTIMATED_REGIONS,
)


PAYLOAD_VERSION = 3
OBSERVATION_MARKERS = ("o", "s", "^", "D", "P")
VARIABLE_LABELS = {"rlds": "DLR", "eddy_z500": "EDDY_Z500"}


def _canonical_variable(name: str) -> str:
    lowered = str(name).lower()
    aliases = {"dlr": "rlds", "dtr": "tdurual"}
    return aliases.get(lowered, lowered)


def _sanitize_region(region_payload: dict) -> dict:
    """删除 DTR、去重变量，并把 EDDY_Z500 固定到末位。"""

    variables = [_canonical_variable(item) for item in region_payload["variables"]]
    cmip_values = list(region_payload["cmip_values"])
    obs_values = list(region_payload["obs_values"])
    if not (len(variables) == len(cmip_values) == len(obs_values)):
        raise ValueError("Regional boxplot payload has inconsistent variable/value lengths")
    by_variable: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for variable, cmip, obs in zip(variables, cmip_values, obs_values):
        if variable in {"tdurual", "dtr"}:
            continue
        by_variable[variable] = (np.asarray(cmip, dtype=float), np.asarray(obs, dtype=float))
    missing = [name for name in EXTENDED_DATA_5_6_VARIABLES if name not in by_variable]
    if missing:
        raise ValueError("Regional boxplot payload is missing required variable(s): " + ", ".join(missing))
    ordered = list(EXTENDED_DATA_5_6_VARIABLES)
    return {
        **region_payload,
        "variables": ordered,
        "cmip_values": [by_variable[name][0] for name in ordered],
        "obs_values": [by_variable[name][1] for name in ordered],
    }


def sanitize_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("Regional boxplot payload must be a dictionary")
    clean = dict(payload)
    for group_name in ("hot_group", "nonhot_group"):
        if group_name not in clean:
            raise ValueError(f"Regional boxplot payload is missing {group_name}")
        group = dict(clean[group_name])
        group["regions"] = [_sanitize_region(item) for item in group["regions"]]
        clean[group_name] = group
    clean["_version"] = PAYLOAD_VERSION
    return clean


def _standardize(cmip: np.ndarray, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.nanmean(cmip))
    standard_deviation = float(np.nanstd(cmip, ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        raise ValueError("Cannot standardize a variable with zero or invalid inter-model spread")
    return (cmip - mean) / standard_deviation, (observations - mean) / standard_deviation


def build_payload_from_analysis_ready(source: Path) -> dict:
    """从统一的 region-variable-member/observation 趋势文件生成可复用绘图缓存。"""

    require_files([source], figure="Extended Data Figures 5–6")
    dataset = xr.open_dataset(source)
    require_variables(dataset, ("cmip_trend", "obs_trend"), source=str(source))
    for dimension in ("region", "variable", "member", "observation"):
        if dimension not in dataset.coords and dimension not in dataset.dims:
            raise ValueError(f"{source} is missing required coordinate/dimension: {dimension}")

    available = {_canonical_variable(item): str(item) for item in dataset.variable.values}
    missing = [item for item in EXTENDED_DATA_5_6_VARIABLES if item not in available]
    if missing:
        raise ValueError(f"{source} is missing required variable(s): {', '.join(missing)}")
    observation_names = [str(item) for item in dataset.observation.values]

    def collect(region_names: tuple[str, ...]) -> dict:
        regions = []
        for region_name in region_names:
            cmip_values = []
            obs_values = []
            for canonical in EXTENDED_DATA_5_6_VARIABLES:
                raw_name = available[canonical]
                cmip = np.asarray(dataset["cmip_trend"].sel(region=region_name, variable=raw_name), dtype=float)
                obs = np.asarray(dataset["obs_trend"].sel(region=region_name, variable=raw_name), dtype=float)
                cmip_standardized, obs_standardized = _standardize(cmip, obs)
                cmip_values.append(cmip_standardized)
                obs_values.append(obs_standardized)
            regions.append(
                {
                    "region": region_name,
                    "variables": list(EXTENDED_DATA_5_6_VARIABLES),
                    "cmip_values": cmip_values,
                    "obs_values": obs_values,
                }
            )
        return {"region_names": list(region_names), "regions": regions}

    return {
        "_version": PAYLOAD_VERSION,
        "obs_names": observation_names,
        "obs_markers": list(OBSERVATION_MARKERS[: len(observation_names)]),
        "hot_group": collect(OVER_ESTIMATED_REGIONS),
        "nonhot_group": collect(UNDER_ESTIMATED_REGIONS),
    }


def load_or_build_payload(
    config: ProjectConfig,
    *,
    recompute: bool = False,
    plot_only: bool = False,
) -> dict:
    cache = config.cache_root / "hotspot_nonhot_region_boxplots_payload_v3.pkl"
    if cache.exists() and not recompute:
        with cache.open("rb") as handle:
            return sanitize_payload(pickle.load(handle))
    if plot_only:
        raise MissingInputError(
            f"Extended Data Figures 5–6 plot-only cache is missing: {cache}\n"
            "Run once without --plot-only or provide the cache described in DATA_REQUIREMENTS.md."
        )
    source = config.data_root / "physical_process" / "region_variable_trends.nc"
    payload = sanitize_payload(build_payload_from_analysis_ready(source))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return payload


def plot_group(payload: dict, group_name: str, output_root: Path, stem: str) -> list[Path]:
    group = payload[group_name]
    is_hot = group_name == "hot_group"
    figure_size = (16.0, 16.0) if is_hot else (15.0, 12.0)
    bottom, top = (0.12, 0.92) if is_hot else (0.16, 0.91)
    hspace = 0.26 if is_hot else 0.28
    fig, axes = plt.subplots(3 if is_hot else 2, 2, figsize=figure_size, dpi=300, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    legend_handles = legend_labels = None
    legend_seen: set[str] = set()
    for index, (region_name, region_payload) in enumerate(zip(group["region_names"], group["regions"])):
        ax = axes[index]
        variables = region_payload["variables"]
        positions = np.arange(len(variables), dtype=float) * 0.6 + 1.0
        boxplot = ax.boxplot(
            region_payload["cmip_values"],
            positions=positions,
            widths=0.35,
            patch_artist=True,
            whis=(5, 95),
            showfliers=False,
        )
        for box in boxplot["boxes"]:
            box.set(facecolor="#66c2a4", alpha=0.3, edgecolor="#66c2a4", linewidth=0.9)
        for median in boxplot["medians"]:
            median.set(color="#66c2a4", linewidth=1.1)
        for whisker in boxplot["whiskers"]:
            whisker.set(color="#66c2a4", linewidth=0.9, linestyle="--")
        for cap in boxplot["caps"]:
            cap.set(color="#66c2a4", linewidth=0.9)
        # 不同变量允许拥有不同数量的观测产品；例如 EDDY_Z500 只有三套再分析。
        for variable_index, values in enumerate(region_payload["obs_values"]):
            for obs_index, value in enumerate(np.asarray(values, dtype=float).reshape(-1)):
                if obs_index >= len(payload["obs_names"]) or obs_index >= len(payload["obs_markers"]):
                    raise ValueError("Observation metadata are shorter than the stored observation values")
                name = payload["obs_names"][obs_index]
                marker = payload["obs_markers"][obs_index]
                label = name if index == 0 and name not in legend_seen else None
                ax.scatter(
                    positions[variable_index],
                    value,
                    marker=marker,
                    s=60,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.4,
                    alpha=0.9,
                    zorder=5,
                    label=label,
                )
                if label is not None:
                    legend_seen.add(name)
        ax.axhline(0, linestyle="--", linewidth=0.8, color="gray")
        ax.set_xlim(positions[0] - 0.38, positions[-1] + 0.38)
        ax.set_xticks(positions)
        ax.set_xticklabels([VARIABLE_LABELS.get(item, item.upper()) for item in variables], rotation=35, fontsize=16, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="y", labelsize=16)
        ax.set_title(region_name, fontsize=21, pad=4)
        ax.text(-0.06, 1.045, "abcdefghijklmnopqrstuvwxyz"[index], transform=ax.transAxes, ha="left", va="bottom", fontsize=23, fontweight="bold")
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    for index in range(len(group["region_names"]), len(axes)):
        fig.delaxes(axes[index])
    if legend_handles:
        fig.legend(legend_handles, legend_labels, frameon=False, fontsize=16, loc="lower center", bbox_to_anchor=(0.5, 0.02 if is_hot else 0.03), ncol=5)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=bottom, top=top, wspace=0.16 if is_hot else 0.14, hspace=hspace)
    outputs = save_figure(fig, output_root, stem, dpi=300)
    plt.close(fig)
    return outputs
