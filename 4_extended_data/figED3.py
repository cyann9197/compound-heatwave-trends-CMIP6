from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chw_cmip6.figure_cli import prepare_figure

ARGS, CONFIG = prepare_figure('figED3', 'Reproduce Extended Data Figure 3.')
from chw_cmip6.figure_context import *  # noqa: E402,F403
from chw_cmip6.fig2_boxplot import draw_regional_heatwave_boxplot  # noqa: E402

# Fig2 下图累计热量版绘图数据缓存：FIG2_CUMHEAT_BOXPLOT_CACHE
# 默认 FIG2_CUMHEAT_FORCE_RECOMPUTE = False，只读取缓存并重绘该附图。
# 若需要强制重新计算累计热量版区域趋势，请把 FIG2_CUMHEAT_FORCE_RECOMPUTE 改成 True。
t0_fig2 = time.perf_counter()
fig2_payload = prepare_fig2_data(force_recompute=ARGS.recompute)
print(f"Fig2 prepared payload ready in {time.perf_counter() - t0_fig2:.2f} s")

labels = list(fig2_payload["labels"])
obs_names = list(fig2_payload["obs_names"])
fig2_box = fig2_payload["box_payload"]
ratio_payload = fig2_payload["map_payload"]
FIG2_CUMHEAT_SUPP_STEM = "Extended Data Fig3"
FIG2_DPI = 300
FIG2_CUMHEAT_SUPP_TITLE = "Cumulative heat trend"
FIG2_CUMHEAT_SUPP_FIGSIZE = (14.0, 6.8)
FIG2_CUMHEAT_SUPP_YLIM = (-0.1, 0.8)
FIG2_CUMHEAT_SUPP_LEFT = 0.075
FIG2_CUMHEAT_SUPP_RIGHT = 0.985
FIG2_CUMHEAT_SUPP_TOP = 0.98
FIG2_CUMHEAT_SUPP_BOTTOM = 0.1


# 这张附图沿用 Fig2 g 面板的区域、观测集、CMIP 成员和 1981-2014 趋势算法，只把分析对象改为累计热量。
t0_fig2_cumheat = time.perf_counter()
fig2_cumheat_box = ensure_fig2_cumheat_boxplot_cache(force=ARGS.recompute)
print(f"Fig2 cumulative-heat payload ready in {time.perf_counter() - t0_fig2_cumheat:.2f} s")

cumheat_labels = list(np.asarray(fig2_cumheat_box["labels"], dtype=object))
cumheat_obs_names = list(np.asarray(fig2_cumheat_box["obs_names"], dtype=object))
print("Fig2 cumulative-heat actual time range: 1981-2014")
assert cumheat_labels == labels, "累计热量补图的区域顺序必须与 Fig2 g 面板一致。"
assert cumheat_obs_names == obs_names, "累计热量补图的观测顺序必须与 Fig2 g 面板一致。"

cumheat_arrays = [
    np.asarray(fig2_cumheat_box["cmip_all_day"], dtype=float),
    np.asarray(fig2_cumheat_box["obs_all_day"], dtype=float),
    np.asarray(fig2_cumheat_box["cmip_all_night"], dtype=float),
    np.asarray(fig2_cumheat_box["obs_all_night"], dtype=float),
    np.asarray(fig2_cumheat_box["cmip_all_compound"], dtype=float),
    np.asarray(fig2_cumheat_box["obs_all_compound"], dtype=float),
]
cumheat_max = max(float(np.nanmax(arr)) for arr in cumheat_arrays if arr.size)
cumheat_ylim = [FIG2_CUMHEAT_SUPP_YLIM[0], FIG2_CUMHEAT_SUPP_YLIM[1]]
if np.isfinite(cumheat_max) and cumheat_max > FIG2_CUMHEAT_SUPP_YLIM[1] * 1.02:
    cumheat_ylim[1] = float(np.ceil((cumheat_max + 0.02) / 0.1) * 0.1)
print(f"Fig2 cumulative-heat y-range: {tuple(cumheat_ylim)}")

fig2_cumheat_fig = plt.figure(figsize=FIG2_CUMHEAT_SUPP_FIGSIZE, dpi=FIG2_DPI)
fig2_cumheat_ax = fig2_cumheat_fig.add_subplot(111)
draw_regional_heatwave_boxplot(
    fig2_cumheat_ax,
    fig2_cumheat_box,
    ylabel="HW CH trend (°C yr$^{-1}$)",
    ylim=(-0.2, 0.8),
    title=FIG2_CUMHEAT_SUPP_TITLE,
)

fig2_cumheat_fig.subplots_adjust(
    left=FIG2_CUMHEAT_SUPP_LEFT,
    right=FIG2_CUMHEAT_SUPP_RIGHT,
    top=FIG2_CUMHEAT_SUPP_TOP,
    bottom=FIG2_CUMHEAT_SUPP_BOTTOM,
)

finalize_figure(
    fig2_cumheat_fig,
    stem=FIG2_CUMHEAT_SUPP_STEM,
    dpi=FIG2_DPI,
    close_after=True,
)
record_status("Extended Data Fig3", "done", "Fig2 下图累计热量版附图，沿用 Fig2 g 面板区域趋势逻辑。")
