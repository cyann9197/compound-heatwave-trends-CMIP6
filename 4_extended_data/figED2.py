from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chw_cmip6.figure_cli import prepare_figure

ARGS, CONFIG = prepare_figure('figED2', 'Reproduce Extended Data Figure 2.')
from chw_cmip6.figure_context import *  # noqa: E402,F403

ds_trend = ensure_three_hw_trend_cache("mme", force=ARGS.recompute)

# 地图型空间图的行距、colorbar 与图的距离、以及 colorbar 尺寸，
# 统一由 cell 6 中的 MAP_* 参数控制。
def plot_three_hw_trend_group(ds_trend, source_title, stem_main, stem_legacy):
    fig = plt.figure(figsize=(16.6, 5.6), dpi=300)
    box = [-180, 180, 0, 90]
    levels_days = np.arange(-0.30, 0.3001, 0.03)
    levels_heat = np.arange(-0.5, 0.55, 0.05)
    row_defs = [
        ("Days", levels_days, [
            ("Daytime HW", "day_hw_days_trend", "day_hw_days_p"),
            ("Nighttime HW", "night_hw_days_trend", "night_hw_days_p"),
            ("Compound HW", "compound_hw_days_trend", "compound_hw_days_p"),
        ], "days yr$^{-1}$"),
        ("Cumulative Heat", levels_heat, [
            ("", "day_hw_ch_trend", "day_hw_ch_p"),
            ("", "night_hw_ch_trend", "night_hw_ch_p"),
            ("", "compound_hw_ch_trend", "compound_hw_ch_p"),
        ], "°C yr$^{-1}$"),
    ]
    panel_letters = "abcdef"
    row_meshes = {}
    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[1.0, MAP_CBAR_ROW_HEIGHT, 1.0, MAP_CBAR_ROW_HEIGHT],
        hspace=MAP_GRID_HSPACE - 0.03,
        wspace=MAP_GRID_WSPACE,
    )

    for row_idx, (row_label, levels, panel_defs, unit_label) in enumerate(row_defs):
        gs_row = 0 if row_idx == 0 else 2
        for col_idx, (title, trend_key, p_key) in enumerate(panel_defs):
            panel_idx = row_idx * 3 + col_idx
            ax = fig.add_subplot(gs[gs_row, col_idx], projection=PROJ)
            ax = make_map(ax, box)
            trend_np, lons_c = add_cyclic_point(np.asarray(ds_trend[trend_key]), coord=ds_trend[trend_key].lon.values)
            p_np, _ = add_cyclic_point(np.asarray(ds_trend[p_key]), coord=ds_trend[trend_key].lon.values)
            cf = ax.contourf(
                lons_c,
                ds_trend[trend_key].lat.values,
                trend_np,
                levels=levels,
                cmap=plt.cm.bwr,
                transform=ccrs.PlateCarree(),
                zorder=0,
                extend="both",
            )
            ax.contourf(
                lons_c,
                ds_trend[trend_key].lat.values,
                p_np,
                levels=[0.0, 0.05, 1.0],
                zorder=1,
                hatches=["///", None],
                colors="none",
                transform=ccrs.PlateCarree(),
            )
            if row_idx == 0:
                apply_panel_header(ax, panel_letters[panel_idx], title, title_pad=MAP_TITLE_PAD, has_title=True)
            else:
                apply_panel_header(ax, panel_letters[panel_idx], title=None, has_title=False)
            ax.tick_params(axis="both", labelsize=MAP_TICK_PT)
            if col_idx != 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(row_label, fontsize=AXIS_LABEL_PT)
            row_meshes[row_idx] = (cf, unit_label)

    fig.text(0.5, 0.96, source_title, ha="center", va="top", fontsize=PANEL_TITLE_PT, fontweight="normal")

    cb_ax1 = fig.add_subplot(gs[1, :])
    cb1 = plt.colorbar(row_meshes[0][0], cax=cb_ax1, orientation="horizontal")
    cb1.ax.tick_params(labelsize=CBAR_TICK_PT)
    cb1.set_ticks(np.linspace(-0.30, 0.30, 7))
    set_horizontal_cbar_label_top(cb1, row_meshes[0][1], y=MAP_CBAR_LABEL_Y)

    cb_ax2 = fig.add_subplot(gs[3, :])
    cb2 = plt.colorbar(row_meshes[1][0], cax=cb_ax2, orientation="horizontal")
    cb2.ax.tick_params(labelsize=CBAR_TICK_PT)
    cb2.set_ticks(np.linspace(-0.50, 0.50, 5))
    set_horizontal_cbar_label_top(cb2, row_meshes[1][1], y=MAP_CBAR_LABEL_Y)

    apply_map_figure_layout(fig, [cb_ax1, cb_ax2])
    extra_stems = [stem_legacy] if stem_legacy else []
    finalize_figure(fig, stem=stem_main, extra_stems=extra_stems, dpi=300, close_after=True)


plot_three_hw_trend_group(ds_trend, "MME", "Extended Data Fig2", None)
