from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chw_cmip6.figure_cli import prepare_figure

ARGS, CONFIG = prepare_figure('fig2', 'Reproduce manuscript Figure 2.')
from chw_cmip6.figure_context import *  # noqa: E402,F403

FIG2_DPI = 300
FIG2_MAP_FIGSIZE = (16, 7)
FIG2_GROUP_FIGSIZE = (14.6, 16)
FIG2_PERCENT_LEVELS = np.arange(0, 100, 1)
FIG2_CMAP = NATURE_PERCENT_CMAP
FIG2_BOX_COLORS = {
    "Daytime": "#E69F00",
    "Nighttime": "#56B4E9",
    "Compound": "#39AB74",
}
FIG2_OBS_MARKERS = ["o", "s", "D", "*", "^"]

# Fig.2 最终绘图数据缓存路径：FIG2_PREPARED_CACHE
# 只想重新绘图：保持 FIG2_FORCE_RECOMPUTE = False，直接运行下一格 Fig.2 绘图 cell 即可。
# 若需要强制重新计算 Fig.2 的最终处理后数据，请把 FIG2_FORCE_RECOMPUTE 改成 True 再运行下一格。
FIG2_FORCE_RECOMPUTE = False

# Fig2 下图累计热量版区域趋势缓存路径：FIG2_CUMHEAT_BOXPLOT_CACHE
# 只想重绘累计热量补图：保持 FIG2_CUMHEAT_FORCE_RECOMPUTE = False。
# 若需要强制重算累计热量补图的区域趋势缓存，请把 FIG2_CUMHEAT_FORCE_RECOMPUTE 改成 True。
FIG2_CUMHEAT_FORCE_RECOMPUTE = False

FIG2_OUTER_HSPACE = 0.10
FIG2_MAP_ROW_SPACING = 0.045
FIG2_MAP_COL_SPACING = 0.070
FIG2_CBAR_ROW_HEIGHT = 0.060 # colorbar 占整行的高度比例
FIG2_CBAR_WIDTH_FRAC = 0.60    # colorbar 占整列宽度的比例
FIG2_CBAR_HEIGHT_FRAC = 0.70 # colorbar 内部实际颜色条占 colorbar 区域的高度比例
FIG2_CBAR_Y_FRAC = 0.26 # colorbar 内部实际颜色条的底部距离 colorbar 区域底部的高度比例
FIG2_CBAR_DOWN_SHIFT=0.002 # colorbar 整体下移距离
FIG2_SUBPLOTS_LEFT = 0.082
FIG2_SUBPLOTS_RIGHT = 0.982
FIG2_SUBPLOTS_TOP = 0.972
FIG2_SUBPLOTS_BOTTOM = 0.055
FIG2_OVERUNDER_TOP_OFFSET = 0.07

# ============================================================
# Fig.2 统一字号放大系数
# 建议先用 1.15；如果还觉得小，可以改成 1.20 或 1.25
# ============================================================
FIG2_FONT_SCALE = 1.3

MAP_TICK_PT_FIG2 = MAP_TICK_PT * FIG2_FONT_SCALE
CBAR_TICK_PT_FIG2 = CBAR_TICK_PT * FIG2_FONT_SCALE
AXIS_TICK_PT_FIG2 = AXIS_TICK_PT * FIG2_FONT_SCALE
AXIS_LABEL_PT_FIG2 = AXIS_LABEL_PT * FIG2_FONT_SCALE
ANNOTATION_PT_FIG2 = ANNOTATION_PT * FIG2_FONT_SCALE
PANEL_LABEL_PT_FIG2 = PANEL_LABEL_PT + 1
LEGEND_PT_FIG2 = LEGEND_PT * FIG2_FONT_SCALE


FIG2_MAP_LABEL_ORDER = ["a", "b", "c", "d", "e", "f"]

# Fig.2 最终绘图数据缓存保存在 FIG2_PREPARED_CACHE。
# 默认 FIG2_FORCE_RECOMPUTE = False，只读取缓存并重新绘图。
# 若要强制重算最终处理后数据，请在上一格把 FIG2_FORCE_RECOMPUTE 改成 True。
assert FIG2_EXTREME_BASELINE_MODE == "fixed_jja_mean", "Fig.2 extreme 图必须使用 fixed_jja_mean。"
t0_fig2 = time.perf_counter()
fig2_payload = prepare_fig2_data(force_recompute=FIG2_FORCE_RECOMPUTE)
print(f"Fig2 prepared payload ready in {time.perf_counter() - t0_fig2:.2f} s")

labels = list(fig2_payload["labels"])
obs_names = list(fig2_payload["obs_names"])
fig2_box = fig2_payload["box_payload"]
ratio_payload = fig2_payload["map_payload"]
# ????????????
# ????a Compound HW Days?b Compound HW CH
# ????c TMAX?d TMIN
# ????e TMAX Extreme?f TMIN Extreme
for payload in ratio_payload:
    if payload.get("title") == "Compound CH":
        payload["title"] = "Compound HW CH"
ratio_payload_by_title = {payload["title"]: payload for payload in ratio_payload}
ratio_payload = [
    ratio_payload_by_title["Compound HW Days"],
    ratio_payload_by_title["Compound HW CH"],
    ratio_payload_by_title["TMAX"],
    ratio_payload_by_title["TMIN"],
    ratio_payload_by_title["TMAX Extreme"],
    ratio_payload_by_title["TMIN Extreme"],
]
map_label_order = FIG2_MAP_LABEL_ORDER


def draw_fig2_maps(fig_target, gs_target, map_payload):
    map_axes = []
    cf_local = None

    for idx, payload in enumerate(map_payload):
        row = idx // 2
        col = idx % 2

        ax = fig_target.add_subplot(gs_target[row, col], projection=PROJ)
        ax = make_map(ax, [-180, 180, 0, 90])

        cf_local = ax.contourf(
            payload["lon_cyclic"],
            payload["lat"],
            payload["data_cyclic"],
            levels=FIG2_PERCENT_LEVELS,
            cmap=FIG2_CMAP,
            transform=ccrs.PlateCarree(),
            zorder=0,
            extend="both",
        )

        draw_strict_boxes(ax)
        apply_panel_header(
            ax,
            map_label_order[idx],
            payload["title"],
            title_pad=MAP_TITLE_PAD,fsize=PANEL_TITLE_PT*FIG2_FONT_SCALE-2,
            has_title=True,
        )

        ax.tick_params(axis="both", labelsize=MAP_TICK_PT_FIG2-2)

        if col != 0:
            ax.tick_params(labelleft=False)

        map_axes.append(ax)

    cb_ax = fig_target.add_subplot(gs_target[3, :])
    cb = plt.colorbar(cf_local, cax=cb_ax, orientation="horizontal")

    cb.ax.tick_params(labelsize=CBAR_TICK_PT_FIG2-2)
    cb.set_ticks(np.arange(0, 101, 10))
    cb.set_ticklabels([str(x) for x in np.arange(0, 101, 10)])

    # 色标右上角百分号
    set_horizontal_cbar_percent_top(cb, "%")

    return map_axes, cb_ax


def draw_fig2_box_panel(
    ax_target,
    box_payload,
    add_obs_legend=True,
    ylabel="HW days trend (days yr$^{-1}$)",
    ylim=(-0.1, 0.4),
    yticks=np.arange(-0.1, 0.41, 0.1),
    panel_letter="g",
    show_panel_label=True,
    title_text=None,
):
    box_labels = list(np.asarray(box_payload.get("labels", labels), dtype=object))
    box_obs_names = list(np.asarray(box_payload.get("obs_names", obs_names), dtype=object))

    datasets = [
        ("Daytime", box_payload["cmip_all_day"], box_payload["obs_all_day"]),
        ("Nighttime", box_payload["cmip_all_night"], box_payload["obs_all_night"]),
        ("Compound", box_payload["cmip_all_compound"], box_payload["obs_all_compound"]),
    ]

    n_regions = len(box_labels)
    box_width = 0.2
    offsets = [-box_width, 0, box_width]

    for i, (hw_type, cmip_data, obs_data) in enumerate(datasets):
        cmip_data = np.asarray(cmip_data, dtype=float)
        obs_data = np.asarray(obs_data, dtype=float)

        positions = np.arange(1, n_regions + 1) + offsets[i]

        bp = ax_target.boxplot(
            cmip_data.T,
            positions=positions,
            widths=box_width * 0.9,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),     # 延伸线显示到5和95
            zorder=1,
        )

        for box in bp["boxes"]:
            box.set(
                facecolor=FIG2_BOX_COLORS[hw_type],
                alpha=0.28,
                edgecolor=FIG2_BOX_COLORS[hw_type],
                linewidth=1.0,
            )

        for whisker in bp["whiskers"]:
            whisker.set(
                color=FIG2_BOX_COLORS[hw_type],
                linewidth=0.8,
                linestyle="--",
                alpha=0.8,
            )

        for cap in bp["caps"]:
            cap.set(color=FIG2_BOX_COLORS[hw_type], linewidth=0.8)

        for median in bp["medians"]:
            median.set(color=FIG2_BOX_COLORS[hw_type], linewidth=1.2)

        means = cmip_data.mean(axis=1)

        ax_target.scatter(
            positions,
            means,
            marker="x",
            s=110,
            linewidth=2,
            color=FIG2_BOX_COLORS[hw_type],
            zorder=4,
        )

        for obs_i in range(len(box_obs_names)):
            for x_idx in range(n_regions):
                ax_target.scatter(
                    positions[x_idx],
                    obs_data[obs_i, x_idx],
                    marker=FIG2_OBS_MARKERS[obs_i],
                    s=95,
                    facecolors="none",
                    edgecolor="black",
                    alpha=0.65,
                    zorder=4,
                    label=box_obs_names[obs_i] if (add_obs_legend and i == 0 and x_idx == 0) else None,
                )

    ax_target.axvline(1.5, linestyle="--", alpha=0.25, color="gray")
    ax_target.axvline(1.5 + len(hotrgion_na), linestyle="--", alpha=0.25, color="gray")
    ax_target.axhline(0, linestyle="--", linewidth=1, color="gray", alpha=0.6)

    ax_target.set_xticks(np.arange(1, len(box_labels) + 1))
    ax_target.set_xticklabels(box_labels, fontsize=AXIS_TICK_PT_FIG2 + 2)
    ax_target.set_xlim(0.35, len(box_labels) + 0.65)

    ax_target.set_ylabel(ylabel, fontsize=AXIS_LABEL_PT_FIG2)
    ax_target.tick_params(axis="y", labelsize=AXIS_TICK_PT_FIG2)

    if title_text:
        ax_target.set_title(
            title_text,
            fontsize=PANEL_TITLE_PT * FIG2_FONT_SCALE-2,
            pad=MAP_TITLE_PAD,
            fontweight="normal",
        )

    if ylim is not None:
        ax_target.set_ylim(*ylim)
    if yticks is not None:
        ax_target.set_yticks(yticks)
    else:
        ax_target.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ylim_now = ax_target.get_ylim()

    ax_target.text(
        1 + len(hotrgion_na) / 2 + 0.5,
        ylim_now[1] - FIG2_OVERUNDER_TOP_OFFSET,
        "Overestimated",
        ha="center",
        fontsize=ANNOTATION_PT_FIG2,
        color="0.2",
    )

    ax_target.text(
        1 + len(hotrgion_na) + len(nothotrgion_na) / 2 + 0.5,
        ylim_now[1] - FIG2_OVERUNDER_TOP_OFFSET,
        "Underestimated",
        ha="center",
        fontsize=ANNOTATION_PT_FIG2,
        color="0.2",
    )

    if show_panel_label and panel_letter:
        add_panel_label(
            ax_target,
            panel_letter,
            x=PANEL_LABEL_X_NO_TITLE + 0.042,
            y=PANEL_LABEL_Y_NO_TITLE,
            fontsize=PANEL_LABEL_PT_FIG2,
        )

    hw_legend = [
        mpatches.Patch(
            facecolor=FIG2_BOX_COLORS["Daytime"],
            edgecolor=FIG2_BOX_COLORS["Daytime"],
            alpha=0.35,
            label="Daytime HW",
        ),
        mpatches.Patch(
            facecolor=FIG2_BOX_COLORS["Nighttime"],
            edgecolor=FIG2_BOX_COLORS["Nighttime"],
            alpha=0.35,
            label="Nighttime HW",
        ),
        mpatches.Patch(
            facecolor=FIG2_BOX_COLORS["Compound"],
            edgecolor=FIG2_BOX_COLORS["Compound"],
            alpha=0.35,
            label="Compound HW",
        ),
    ]

    legend1 = ax_target.legend(
        handles=hw_legend,
        loc="upper left",
        bbox_to_anchor=(0.02, 1.03),
        frameon=False,
        fontsize=LEGEND_PT_FIG2 + 2,
    )
    ax_target.add_artist(legend1)

    handles, labels_legend = ax_target.get_legend_handles_labels()
    unique = dict(zip(labels_legend, handles))

    mme_handle = Line2D(
        [0],
        [0],
        marker="x",
        color="black",
        linestyle="None",
        markersize=9.5 * FIG2_FONT_SCALE,
        markeredgewidth=2,
        label="MME",
    )

    desired_order = ["MME", "ERA5", "MERRA2", "JRA-3Q", "CPC", "BEST"]

    legend_map = {"MME": mme_handle}
    legend_map.update(unique)

    ordered_handles = [legend_map[name] for name in desired_order if name in legend_map]
    ordered_labels = [name for name in desired_order if name in legend_map]

    legend2 = ax_target.legend(
        ordered_handles,
        ordered_labels,
        frameon=False,
        fontsize=LEGEND_PT_FIG2 + 2,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.60, 0.082),
        columnspacing=1.0,
        handletextpad=0.4,
        markerscale=1.2,
    )

    ax_target.spines["top"].set_visible(False)
    ax_target.spines["right"].set_visible(False)
    ax_target.spines["left"].set_linewidth(1.2)
    ax_target.spines["bottom"].set_linewidth(1.2)

    return legend1, legend2


def plot_fig2(fig2_payload):
    fig_group = plt.figure(figsize=FIG2_GROUP_FIGSIZE, dpi=FIG2_DPI)

    outer = fig_group.add_gridspec(
        2,
        1,
        height_ratios=[1.50, 0.98],
        hspace=FIG2_OUTER_HSPACE,
    )

    top = outer[0].subgridspec(
        4,
        2,
        height_ratios=[1.0, 1.0, 1.0, FIG2_CBAR_ROW_HEIGHT],
        hspace=FIG2_MAP_ROW_SPACING,
        wspace=FIG2_MAP_COL_SPACING,
    )

    map_axes, cbax = draw_fig2_maps(
        fig_group,
        top,
        ratio_payload,
    )

    axg = fig_group.add_subplot(outer[1, 0])

    legend1, legend2 = draw_fig2_box_panel(
        axg,
        fig2_payload["box_payload"],
        add_obs_legend=True,
    )

    fig_group.subplots_adjust(
        left=FIG2_SUBPLOTS_LEFT,
        right=FIG2_SUBPLOTS_RIGHT,
        top=FIG2_SUBPLOTS_TOP,
        bottom=FIG2_SUBPLOTS_BOTTOM,
    )

    shrink_horizontal_cbar_axis(
        cbax,
        width_frac=FIG2_CBAR_WIDTH_FRAC,
        height_frac=FIG2_CBAR_HEIGHT_FRAC,
        y_frac=FIG2_CBAR_Y_FRAC,
    )

    # absolute downward adjustment for the upper colorbar
    pos = cbax.get_position()
    cbax.set_position([
        pos.x0,
        pos.y0 - FIG2_CBAR_DOWN_SHIFT,
        pos.width,
        pos.height,
    ])

    return fig_group


fig_group = plot_fig2(fig2_payload)

save_figure_multi_format(fig_group, "Fig2", dpi=FIG2_DPI)

if display is not None:
    display(fig_group)
else:
    plt.show()

plt.close(fig_group)

record_status("Fig2", "done", "Fig.2 仅导出组图，并优先读取最终绘图缓存。")
