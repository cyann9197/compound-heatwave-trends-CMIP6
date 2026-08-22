from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chw_cmip6.figure_cli import prepare_figure

ARGS, CONFIG = prepare_figure('fig1', 'Reproduce manuscript Figure 1.')
from chw_cmip6.figure_context import *  # noqa: E402,F403

FIG1_FIGSIZE = (18, 12)
FIG1_DPI = 300
FIG1_LINEWIDTH_MAIN = 3.6
FIG1_LINEWIDTH_FIT = 1.9
FIG1_BAR_MODEL_COLOR = "#5B9574"
FIG1_BAR_MME_COLOR = "#0C6B40"
FIG1_OBS_COLORS = {
    "CPC": "#C44E52",
    "BEST": "#B07AA1",
    "ERA5": "#64B5CD",
    "MERRA2": "#4C72B0",
    "JRA-3Q": "#DD8452",
}

FIG1_OBS_LINEWIDTH = 1.8
FIG1_LEGEND_PT = LEGEND_PT + 3
FIG1_TREND_TEXT_PT = ANNOTATION_PT + 1
FIG1_MME_LINEWIDTH = 4.8
FIG1_LEGEND_MME_LINEWIDTH = 2.6

ds_regional = xr.open_dataset(FILE_REGIONAL_MEANS)
fig1_ranking = ensure_fig1_ranking_cache(force=FORCE_RECOMPUTE_MINIMAL_CACHES)

fig = plt.figure(figsize=FIG1_FIGSIZE, dpi=FIG1_DPI)
gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.12], height_ratios=[1.0, 1.04], wspace=0.13, hspace=0.44)
axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 1]),
]

regional_specs = [
    ("compound_days", "Compound HW Days", "Days", "days yr$^{-1}$", (0, 16)),
    ("compound_cumulative_heat", "Compound HW CH", "°C", "°C yr$^{-1}$", (0, 30)),
]


def collect_obs_trend_info_regional(ds_regional, prefix, obs_order, mask_sub):
    obs_series_by_name = {}
    obs_fit_by_name = {}
    trend_info_full = []
    trend_info_sub = []
    for obs_name in obs_order:
        var_key = f"{prefix}_obs_{obs_name.lower().replace('-', '').replace(' ', '')}"
        if var_key not in ds_regional:
            if obs_name == "BEST":
                var_key = f"{prefix}_obs_best"
            elif obs_name in ("JRA-3Q", "JRA3Q"):
                var_key = f"{prefix}_obs_jra3q"
        obs_da = ds_regional[var_key]
        years_obs = np.asarray(obs_da["year"].values.astype(int), dtype=float)
        values_obs = np.asarray(obs_da.values, dtype=float)
        slope_full, fit_full, _p_full, star_full = fit_line_for_plot(years_obs, values_obs)
        slope_sub, _fit_sub, _p_sub, star_sub = fit_line_for_plot(years_obs[mask_sub], values_obs[mask_sub])
        obs_series_by_name[obs_name] = (years_obs, values_obs)
        obs_fit_by_name[obs_name] = (years_obs, fit_full)
        color = FIG1_OBS_COLORS[obs_name]
        trend_info_full.append((obs_name, slope_full, star_full, color))
        trend_info_sub.append((obs_name, slope_sub, star_sub, color))
    return (
        obs_series_by_name,
        obs_fit_by_name,
        sorted(trend_info_full, key=lambda x: x[1], reverse=True),
        sorted(trend_info_sub, key=lambda x: x[1], reverse=True),
    )


for panel_idx, (prefix, centered_title, ylabel_text, trend_unit_text, ylim) in enumerate(regional_specs):
    ax = axes[panel_idx]
    years = np.asarray(ds_regional[f"{prefix}_mme_245"]["year"].values.astype(int), dtype=float)
    mask_sub = years <= 2014
    mme245 = np.asarray(ds_regional[f"{prefix}_mme_245"].values, dtype=float)
    mme585 = np.asarray(ds_regional[f"{prefix}_mme_585"].values, dtype=float)
    p05_245 = np.asarray(ds_regional[f"{prefix}_p05_245"].values, dtype=float)
    p95_245 = np.asarray(ds_regional[f"{prefix}_p95_245"].values, dtype=float)
    p05_585 = np.asarray(ds_regional[f"{prefix}_p05_585"].values, dtype=float)
    p95_585 = np.asarray(ds_regional[f"{prefix}_p95_585"].values, dtype=float)

    mme245_slope_full, mme245_fit_full, _tmp1, mme245_star_full = fit_line_for_plot(years, mme245)
    mme585_slope_full, mme585_fit_full, _tmp2, mme585_star_full = fit_line_for_plot(years, mme585)
    mme_hist_slope_sub, _tmp3, _tmp4, mme_hist_star_sub = fit_line_for_plot(years[mask_sub], mme245[mask_sub])

    obs_order = ["CPC", "BEST", "ERA5", "MERRA2", "JRA-3Q"]
    obs_series_by_name, obs_fit_by_name, trend_info_full_sorted, trend_info_sub_sorted = collect_obs_trend_info_regional(ds_regional, prefix, obs_order, mask_sub)

    ax.fill_between(years, p05_585, p95_585, color="#A566D5", alpha=0.10, edgecolor="none")
    ax.fill_between(years, p05_245, p95_245, color="#438961F6", alpha=0.10, edgecolor="none")
    ax.plot(years, mme585, color="#A566D5", linewidth=FIG1_MME_LINEWIDTH, label="SSP585-MME")
    ax.plot(years, mme585_fit_full, color="#A566D5", linestyle="--", linewidth=FIG1_LINEWIDTH_FIT)
    ax.plot(years, mme245, color="#438961F6", linewidth=FIG1_MME_LINEWIDTH, label="SSP245-MME")
    ax.plot(years, mme245_fit_full, color="#438961F6", linestyle="--", linewidth=FIG1_LINEWIDTH_FIT)
    for obs_name, _slope, _star, color in trend_info_full_sorted:
        obs_years, obs_vals = obs_series_by_name[obs_name]
        fit_years, fit_vals = obs_fit_by_name[obs_name]
        ax.plot(obs_years, obs_vals, color=color, linewidth=FIG1_OBS_LINEWIDTH, label=canonical_dataset_label(obs_name))
        ax.plot(fit_years, fit_vals, color=color, linestyle="--", linewidth=0.9)
    ax.set_title(
        normalize_panel_title(centered_title),
        fontsize=PANEL_TITLE_PT+4,
        pad=6,
        loc="center",
        fontweight="normal",
    )
    add_panel_label(ax, "ab"[panel_idx], x=PANEL_LABEL_X_NO_TITLE, y=PANEL_LABEL_Y_NO_TITLE, fontsize=PANEL_LABEL_PT+4)
    ax.set_ylabel(ylabel_text, fontsize=AXIS_LABEL_PT+3)
    ax.set_xlabel("Year", fontsize=AXIS_LABEL_PT+3)
    ax.set_xlim(1981, 2023)
    ax.set_ylim(*ylim)
    xticks = sorted(set(list(range(1981, 2024, 10)) + [2014]))
    ax.set_xticks(xticks)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_PT+3)
    ax.minorticks_on()
    ax.axvline(2014, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)

    handles, labels = ax.get_legend_handles_labels()
    order_map = {label: handle for handle, label in zip(handles, labels)}
    desired_order = ["SSP245-MME", "SSP585-MME", "CPC", "BEST", "ERA5", "MERRA2", "JRA3Q"]
    ordered_labels = [label for label in desired_order if label in order_map]
    ordered_handles = []
    for label in ordered_labels:
        handle = order_map[label]
        if label == "SSP245-MME":
            handle = Line2D([0], [0], color="#438961F6", linewidth=FIG1_LEGEND_MME_LINEWIDTH)
        elif label == "SSP585-MME":
            handle = Line2D([0], [0], color="#A566D5", linewidth=FIG1_LEGEND_MME_LINEWIDTH)
        ordered_handles.append(handle)
    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper right",
        bbox_to_anchor=(0.992, 0.988),
        ncol=1,
        fontsize=FIG1_LEGEND_PT,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.8,
        labelspacing=0.42,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    left_header_y = 0.986
    left_col_x = 0.012
    right_col_x = 0.42
    line_step = 0.082
    ax.text(left_col_x, left_header_y, "1981-2023", color="0.1", fontsize=FIG1_TREND_TEXT_PT, ha="left", va="top", transform=ax.transAxes)
    ax.text(left_col_x, left_header_y - line_step, f"SSP245-MME: {mme245_slope_full:.3f}{mme245_star_full} {trend_unit_text}", color="#438961F6", fontsize=FIG1_TREND_TEXT_PT, ha="left", va="top", transform=ax.transAxes)
    ax.text(left_col_x, left_header_y - 2 * line_step, f"SSP585-MME: {mme585_slope_full:.3f}{mme585_star_full} {trend_unit_text}", color="#A566D5", fontsize=FIG1_TREND_TEXT_PT, ha="left", va="top", transform=ax.transAxes)
    for text_idx, (obs_name, slope, star, color) in enumerate(trend_info_full_sorted):
        ax.text(left_col_x, left_header_y - (text_idx + 3) * line_step, f"{canonical_dataset_label(obs_name)}: {slope:.3f}{star} {trend_unit_text}", color=color, fontsize=FIG1_TREND_TEXT_PT, ha="left", va="top", transform=ax.transAxes)
    ax.text(right_col_x, left_header_y, "1981-2014", color="0.1", fontsize=FIG1_TREND_TEXT_PT, ha="left", va="top", transform=ax.transAxes)
    ax.text(right_col_x, left_header_y - line_step, f"Hist-MME: {mme_hist_slope_sub:.3f}{mme_hist_star_sub} {trend_unit_text}", color="#438961F6", fontsize=FIG1_TREND_TEXT_PT, ha="left", va="top", transform=ax.transAxes)
    for text_idx, (obs_name, slope, star, color) in enumerate(trend_info_sub_sorted):
        ax.text(right_col_x, left_header_y - (text_idx + 2) * line_step, f"{canonical_dataset_label(obs_name)}: {slope:.3f}{star} {trend_unit_text}", color=color, fontsize=FIG1_TREND_TEXT_PT, ha="left", va="top", transform=ax.transAxes)

ranking_specs = [
    ("compound_days", "Compound HW days (Trend)", "c"),
    ("compound_ch", "Compound HW CH (Trend)", "d"),
]
for local_idx, (cache_key, ylabel_text, letter) in enumerate(ranking_specs):
    ax = axes[2 + local_idx]
    entries = list(fig1_ranking[cache_key])
    entries_sorted = sorted(entries, key=lambda x: x[1])
    display_names = [canonical_dataset_label(item[0]) for item in entries_sorted]
    values = [item[1] for item in entries_sorted]
    kinds = [item[2] for item in entries_sorted]
    colors = []
    for name, kind in zip(display_names, kinds):
        if kind == "mme":
            colors.append(FIG1_BAR_MME_COLOR)
        elif kind == "obs":
            colors.append("gray" if name in ["ERA5", "MERRA2", "JRA3Q"] else "black")
        else:
            colors.append(FIG1_BAR_MODEL_COLOR)
    x = np.arange(len(display_names), dtype=float)
    ax.bar(x, values, color=colors, edgecolor="none", alpha=0.6)
    ax.set_ylabel(normalize_panel_title(ylabel_text), fontsize=AXIS_LABEL_PT+3)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=AXIS_TICK_PT-2)
    ax.tick_params(axis="y", labelsize=AXIS_TICK_PT+1)
    add_panel_label(ax, letter, x=PANEL_LABEL_X_NO_TITLE, y=PANEL_LABEL_Y_NO_TITLE, fontsize=PANEL_LABEL_PT+4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.01)

fig.subplots_adjust(left=0.02, right=0.99, top=0.965, bottom=0.11)
save_figure_multi_format(fig, "Fig1", dpi=FIG1_DPI)
if display is not None:
    display(fig)
else:
    plt.show()
record_status("Fig1", "done", "严格最小修改版 Fig.1 已按最新 final notebook 修复并导出。")
