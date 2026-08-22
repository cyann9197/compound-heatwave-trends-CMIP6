from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chw_cmip6.figure_cli import prepare_figure

ARGS, CONFIG = prepare_figure('fig6', 'Reproduce manuscript Figure 6 using fixed-event counterfactual excess.')
from chw_cmip6.da_plot_style import (  # noqa: E402
    FIG567_ANNOTATION_PT,
    FIG567_AXIS_LABEL_PT,
    FIG567_AXIS_TICK_PT,
    FIG567_CBAR_LABEL_PT,
    FIG567_PANEL_LABEL_PT,
    FIG567_PANEL_TITLE_PT,
    FINAL_DA_RCPARAMS,
    PANEL_LABEL_X_WITH_TITLE,
    PANEL_LABEL_Y_WITH_TITLE,
)

CACHE_ROOT = CONFIG.cache_root

os.environ["DA_BASE_DIR"] = str(CACHE_ROOT / "dynamic_adjustment" / "fixed_event_counterfactual_excess")
os.environ["DA_VARIANT_TAGS"] = "chwcumheatcfexcess_chwcalendar31d"
# 若需要继续调整 Fig.5–Fig.7 与 linear budget 图的文字大小，请修改 cell 6 中的 FIG567_FONT_SCALE。
FIG7_PLOT_STYLE_CONFIG = {
    "title_template": "{var} | {component}",
    "var_labels": {"tmax": "TMAX_CH", "tmin": "TMIN_CH"},
    "component_labels": {"dyn": "Dynamical", "thermo": "Thermodynamic"},
    "ylabel": "Trend (°C yr$^{-1}$)",
    "band_ylabel": "Models >\nERA5 (%)",
    "over_label": "Overestimated",
    "under_label": "Underestimated",
    "member_key_title": "CMIP models",
    "era5_label": "ERA5",
    "mme_label": "MME",
    "title_pad": 10,
    "title_fontsize": FIG567_PANEL_TITLE_PT+1,
    "title_fontweight": "normal",
    "panel_label_x": PANEL_LABEL_X_WITH_TITLE,
    "panel_label_y": PANEL_LABEL_Y_WITH_TITLE,
    "panel_label_fontsize": FIG567_PANEL_LABEL_PT+1,
    "panel_label_fontweight": "bold",
    "annotation_fontsize": FIG567_ANNOTATION_PT + 2,
    "main_tick_fontsize": FIG567_AXIS_TICK_PT,
    "main_xtick_fontsize": FIG567_AXIS_TICK_PT,
    "ylabel_fontsize": FIG567_AXIS_LABEL_PT,
    "empty_member_label_fontsize": FIG567_ANNOTATION_PT + 2,
    "band_xtick_fontsize": FIG567_AXIS_TICK_PT,
    "band_value_fontsize": FIG567_AXIS_TICK_PT-3,
    "band_ylabel_fontsize": FIG567_CBAR_LABEL_PT-3,
    "legend_label_fontsize": FIG567_ANNOTATION_PT-2,
    "member_key_title_fontsize": FIG567_ANNOTATION_PT + 2,
    "member_label_fontsize": FIG567_ANNOTATION_PT -2,
    "output_dpi": 320,
    "rcParams": FINAL_DA_RCPARAMS,
}
os.environ["DA_HWC_AXIS_CONFIG_JSON"] = json.dumps({}, ensure_ascii=False)
os.environ["DA_HWC_PLOT_STYLE_JSON"] = json.dumps(FIG7_PLOT_STYLE_CONFIG, ensure_ascii=False)
DA_NOTEBOOK_OUT_NAME_MAP = {
    ("chwcumheatcfexcess_chwcalendar31d", "beeswarm"): "Fig6",
}
FIG7_REQUIRED_FILES = [
    str(CACHE_ROOT / "dynamic_adjustment" / "fixed_event_counterfactual_excess" / "outputs" / 'da_ridge_daily_chwcumheatcfexcess_chwcalendar31d_region_trends_ERA5_1981_2014_JJA.csv'),
    str(CACHE_ROOT / "dynamic_adjustment" / "fixed_event_counterfactual_excess" / "outputs" / 'da_ridge_daily_chwcumheatcfexcess_chwcalendar31d_region_trends_CMIP_members_JJA_1981_2014.csv'),
    str(CACHE_ROOT / "dynamic_adjustment" / "fixed_event_counterfactual_excess" / "outputs" / 'da_ridge_daily_chwcumheatcfexcess_chwcalendar31d_region_trends_MME_JJA_1981_2014.csv'),
]

_missing_fig7 = [path for path in FIG7_REQUIRED_FILES if not Path(path).exists()]
if _missing_fig7:
    print("Fig. 6 requires the following fixed-event counterfactual-excess CSV files:")
    for _path in _missing_fig7:
        print(" -", _path)
else:
    import json
    import os
    import sys

    import matplotlib
    if "ipykernel" not in sys.modules and "JPY_PARENT_PID" not in os.environ:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import FuncFormatter

    try:
        from IPython.display import display as _ipython_display
    except Exception:
        _ipython_display = None


    BASE_DIR = os.environ.get(
        "DA_BASE_DIR",
        str(CACHE_ROOT / "dynamic_adjustment" / "fixed_event_counterfactual_excess"),
    )
    OUT_DIR = os.path.join(BASE_DIR, "outputs")
    FIG_DIR = os.path.join(BASE_DIR, "figs")
    os.makedirs(FIG_DIR, exist_ok=True)


    def _env_list(name, default=None):
        value = os.environ.get(name)
        if value is None or not str(value).strip():
            return list(default or [])
        return [item.strip() for item in str(value).split(",") if item.strip()]


    VARIANT_TAGS = _env_list(
        "DA_VARIANT_TAGS",
        default=[
            "chwcumheatcf_alljja",
            "chwcumheatcf_chwcalendar31d",
        ],
    )

    hotrgion_na = ["NNA", "NAF", "EEU", "SAS", "ESB"]
    nothotrgion_na = ["SNA", "GIC", "WEU", "ENA"]
    region_order = hotrgion_na + nothotrgion_na
    hot_count = len(hotrgion_na)
    boundary_x = hot_count - 0.5
    x = np.arange(len(region_order))

    model_list = [
        "ACCESS-CM2", "ACCESS-ESM1-5", "AWI-CM-1-1-MR", "BCC-ESM1", "CanESM5",
        "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1", "E3SM-2-0", "E3SM-2-0-NARRM",
        "EC-Earth3", "EC-Earth3-AerChem", "EC-Earth3-CC", "EC-Earth3-Veg-LR",
        "FGOALS-f3-L", "FGOALS-g3", "HadGEM3-GC31-LL", "HadGEM3-GC31-MM",
        "IITM-ESM", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G",
        "MIROC6", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "NorESM2-LM", "NorESM2-MM",
        "TaiESM1", "UKESM1-0-LL",
    ]

    components = [
        ("dyn", "dyn_slope_per_year", "Dynamical"),
        ("thermo", "thermo_slope_per_year", "Thermodynamic"),
    ]
    vars_order = ["tmax", "tmin"]
    cmap = plt.cm.YlOrRd
    mme_color = "#088D27"
    generic_member_color = "#7fb9dd"
    marker_pool = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "p", "h", "H", "8", "d", "*", "+", "x"]
    color_pool = list(plt.cm.tab20.colors) + list(plt.cm.Dark2.colors) + list(plt.cm.Set2.colors)
    line_only_markers = {"+", "x", "1", "2", "3", "4", "|", "_"}

    default_model_styles = {
        "ACCESS-CM2": {"marker": "o", "color": "#8B0000"},
        "ACCESS-ESM1-5": {"marker": "s", "color": "#F08080"},
        "AWI-CM-1-1-MR": {"marker": "D", "color": "#8B4513"},
        "BCC-ESM1": {"marker": ">", "color": "#6B8E23"},
        "CanESM5": {"marker": "p", "color": "#008080"},
        "CMCC-ESM2": {"marker": "*", "color": "#000080"},
        "CNRM-CM6-1": {"marker": "h", "color": "#FF8C00"},
        "CNRM-ESM2-1": {"marker": "<", "color": "#F4A460"},
        "E3SM-2-0": {"marker": "H", "color": "#006400"},
        "E3SM-2-0-NARRM": {"marker": "+", "color": "#90EE90"},
        "EC-Earth3": {"marker": "x", "color": "#00008B"},
        "EC-Earth3-AerChem": {"marker": "X", "color": "#0000CD"},
        "EC-Earth3-CC": {"marker": "d", "color": "#6495ED"},
        "EC-Earth3-Veg-LR": {"marker": "v", "color": "#7ABBE4"},
        "FGOALS-f3-L": {"marker": "8", "color": "#9400D3"},
        "FGOALS-g3": {"marker": "P", "color": "#BA55D3"},
        "HadGEM3-GC31-LL": {"marker": "s", "color": "#FF1493"},
        "HadGEM3-GC31-MM": {"marker": "D", "color": "#FFB6C1"},
        "IITM-ESM": {"marker": "v", "color": "#32CD32"},
        "INM-CM4-8": {"marker": "<", "color": "#008B8B"},
        "INM-CM5-0": {"marker": "p", "color": "#72B5B5"},
        "IPSL-CM6A-LR": {"marker": "*", "color": "#800000"},
        "KACE-1-0-G": {"marker": "h", "color": "#FFD700"},
        "MIROC6": {"marker": "*", "color": "#4B0082"},
        "MPI-ESM1-2-HR": {"marker": "o", "color": "#A0522D"},
        "MPI-ESM1-2-LR": {"marker": "s", "color": "#DEB887"},
        "NorESM2-LM": {"marker": "*", "color": "#DAA520"},
        "NorESM2-MM": {"marker": "d", "color": "#F0E68C"},
        "TaiESM1": {"marker": "o", "color": "#FA8072"},
        "UKESM1-0-LL": {"marker": "s", "color": "#40E0D0"},
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 0.9,
    })


    def _load_style_override():
        raw = os.environ.get("DA_HWC_PLOT_STYLE_JSON", "").strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception as exc:
            print(f"绘图风格参数无法解析，将继续使用脚本默认设置：{exc}")
            return {}


    PLOT_STYLE = _load_style_override()
    if isinstance(PLOT_STYLE.get("rcParams"), dict) and PLOT_STYLE.get("rcParams"):
        plt.rcParams.update(PLOT_STYLE["rcParams"])


    def _style_value(key, default=None):
        return PLOT_STYLE.get(key, default)


    def _style_dict(key):
        value = PLOT_STYLE.get(key, {})
        return value if isinstance(value, dict) else {}


    def _panel_title(var_name, comp_name, default_title, panel_idx):
        panel_titles = _style_dict("panel_titles")
        override_key = f"{var_name}_{comp_name}"
        if override_key in panel_titles:
            return str(panel_titles[override_key])
        template = _style_value("title_template", "")
        if isinstance(template, str) and template:
            var_label = _style_dict("var_labels").get(var_name, f"{var_name.upper()}_CH")
            component_label = _style_dict("component_labels").get(comp_name, default_title)
            try:
                return template.format(
                    panel=chr(97 + panel_idx),
                    var=var_label,
                    component=component_label,
                    var_key=var_name,
                    component_key=comp_name,
                )
            except Exception:
                pass
        return f"{var_name.upper()}_CH | {default_title}"


    def _add_panel_label(ax, panel_idx):
        ax.text(
            _style_value("panel_label_x", -0.08),
            _style_value("panel_label_y", 1.04),
            chr(97 + panel_idx),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=_style_value("panel_label_fontsize", plt.rcParams.get("axes.titlesize", 18)),
            fontweight=_style_value("panel_label_fontweight", "bold"),
        )


    def _csv_path(tag, kind):
        mapping = {
            "era": f"da_ridge_daily_{tag}_region_trends_ERA5_1981_2014_JJA.csv",
            "mem": f"da_ridge_daily_{tag}_region_trends_CMIP_members_JJA_1981_2014.csv",
            "mme": f"da_ridge_daily_{tag}_region_trends_MME_JJA_1981_2014.csv",
        }
        return os.path.join(OUT_DIR, mapping[kind])


    def _fmt_y(value, _pos):
        if abs(value) < 5e-5:
            value = 0.0
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return "0" if text == "-0" else text


    def _ordered_series(df_indexed, col_name):
        values = []
        for region in region_order:
            if region in df_indexed.index:
                value = df_indexed.loc[region][col_name]
                if isinstance(value, pd.Series):
                    value = value.iloc[0]
                values.append(float(value))
            else:
                values.append(np.nan)
        return np.array(values, dtype=float)


    def _candidate_offsets(step=0.052, max_width=0.28):
        levels = int(np.floor(max_width / step))
        candidates = [0.0]
        for lev in range(1, levels + 1):
            candidates.extend([-lev * step, lev * step])
        return candidates


    def _beeswarm_offsets(values, step=0.052, max_width=0.28):
        values = np.asarray(values, dtype=float)
        offsets = np.zeros(values.size, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return offsets
        span = max(float(np.nanmax(finite) - np.nanmin(finite)), 1e-6)
        min_dy = max(0.012 * span, 0.015)
        candidates = _candidate_offsets(step=step, max_width=max_width)
        placed = []
        for idx in np.where(np.isfinite(values))[0][np.argsort(values[np.isfinite(values)], kind="mergesort")]:
            yi = float(values[idx])
            chosen = 0.0
            for cand in candidates:
                if all((abs(yi - yj) >= min_dy) or (abs(cand - xj) >= step * 0.95) for yj, xj in placed):
                    chosen = cand
                    break
            offsets[idx] = chosen
            placed.append((yi, chosen))
        return offsets


    def _draw_scatter_point(ax, xpos, ypos, style, size=105, zorder=2, transform=None):
        marker = style["marker"]
        color = style["color"]
        linewidth = style.get("linewidth", 1.45)
        alpha = style.get("alpha", 0.9)
        kwargs = {"zorder": zorder, "alpha": alpha}
        if transform is not None:
            kwargs["transform"] = transform
        if marker in line_only_markers:
            ax.scatter(xpos, ypos, s=size * 0.95, marker=marker, color=color, linewidths=linewidth, **kwargs)
        else:
            ax.scatter(xpos, ypos, s=size, marker=marker, facecolors="white", edgecolors=color, linewidths=linewidth, **kwargs)


    def _draw_reference_legend(ax, x_cols):
        line_len = 0.02
        legend_y = 0.925
        text_gap = 0.012
        ax.plot([x_cols[0], x_cols[0] + line_len], [legend_y, legend_y], color="k", linewidth=3.5, solid_capstyle="round", transform=ax.transAxes, clip_on=False)
        ax.text(
            x_cols[0] + line_len + text_gap,
            legend_y,
            _style_value("era5_label", "ERA5"),
            ha="left",
            va="center",
            fontsize=_style_value("legend_label_fontsize", 16),
            color="0.1",
            transform=ax.transAxes,
        )
        ax.plot([x_cols[1], x_cols[1] + line_len], [legend_y, legend_y], color=mme_color, linewidth=3.5, solid_capstyle="round", transform=ax.transAxes, clip_on=False)
        ax.text(
            x_cols[1] + line_len + text_gap,
            legend_y,
            _style_value("mme_label", "MME"),
            ha="left",
            va="center",
            fontsize=_style_value("legend_label_fontsize", 16),
            color="0.1",
            transform=ax.transAxes,
        )
        ax.plot([0.02, 0.98], [0.970, 0.970], color="0.85", linewidth=0.8, transform=ax.transAxes, clip_on=False)

    def _draw_member_key(ax, ordered_members, member_style_map):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ncols_key = 4
        nrows_key = int(np.ceil(len(ordered_members) / ncols_key)) if ordered_members else 1
        x_cols = np.linspace(0.03, 0.80, ncols_key)
        y_vals = np.linspace(0.80, 0.16, nrows_key)
        _draw_reference_legend(ax, x_cols)
        if not ordered_members:
            ax.text(
                0.02,
                0.88,
                _style_value("empty_member_label", "No member labels available"),
                ha="left",
                va="top",
                fontsize=_style_value("empty_member_label_fontsize", 10.6),
                transform=ax.transAxes,
            )
            return
        for idx, member_name in enumerate(ordered_members):
            col = idx // nrows_key
            row = idx % nrows_key
            px = x_cols[min(col, len(x_cols) - 1)]
            py = y_vals[row]
            style = member_style_map[member_name]
            _draw_scatter_point(ax, px, py, style, size=90, zorder=2, transform=ax.transAxes)
            ax.text(
                px + 0.050,
                py,
                member_name,
                ha="left",
                va="center",
                fontsize=_style_value("member_label_fontsize", 15),
                color="0.15",
                transform=ax.transAxes,
            )


    def _panel_ylim(values):
        finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)

        if finite.size == 0:
            ymin, ymax = -1.0, 1.0
        else:
            # 强制包含0
            vmin = min(float(np.nanmin(finite)), 0.0)
            vmax = max(float(np.nanmax(finite)), 0.0)

            span = vmax - vmin
            if span < 0.2:
                span = 0.2

            # 边距控制更紧凑
            padding_lower = span * 0.08
            padding_upper = span * 0.10

            ymin = vmin - padding_lower
            ymax = vmax + padding_upper

        # ===== 自动选择较美观刻度间距 =====
        rough_step = (ymax - ymin) / 4  # 目标约5个刻度
        candidates = np.array([
            0.01, 0.02, 0.05,
            0.1, 0.2, 0.25,
            0.5, 1.0, 2.0
        ])
        step = candidates[np.argmin(np.abs(candidates - rough_step))]

        # 向外取整
        ymin = np.floor(ymin / step) * step
        ymax = np.ceil(ymax / step) * step

        # 重新生成等距刻度
        ticks = np.arange(ymin, ymax + step * 0.5, step)

        # 确保0一定存在
        if not np.any(np.isclose(ticks, 0.0)):
            ticks = np.sort(np.append(ticks, 0.0))

        return ymin, ymax, ticks


    def _load_axis_override():
        raw = os.environ.get("DA_HWC_AXIS_CONFIG_JSON", "").strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception as exc:
            print(f"坐标轴覆盖参数无法解析，将使用自动范围：{exc}")
            return {}


    def _plot_member_points(ax, region_rows, col_name, xi, layout_mode, member_style_map, member_offset_map):
        if len(region_rows) == 0:
            return
        values = region_rows[col_name].values.astype(float)
        if layout_mode == "beeswarm":
            offsets = _beeswarm_offsets(values)
        else:
            offsets = np.array([member_offset_map.get(str(row_data["member"]), 0.0) for _, row_data in region_rows.iterrows()], dtype=float)
        for offset, (_, row_data) in zip(offsets, region_rows.iterrows()):
            member_name = str(row_data["member"]) if "member" in row_data.index else ""
            style = member_style_map.get(member_name, {"marker": "o", "color": generic_member_color, "linewidth": 1.35, "alpha": 0.9})
            _draw_scatter_point(ax, xi + float(offset), float(row_data[col_name]), style, size=92, zorder=2)


    def _build_member_styles(mem):
        member_names = [str(v) for v in mem["member"].dropna().unique()] if "member" in mem.columns else []
        ordered_members = [str(v) for v in model_list if str(v) in member_names]
        ordered_members += [v for v in member_names if v not in ordered_members]
        member_style_map = {}
        for idx, member in enumerate(ordered_members):
            if member in default_model_styles:
                marker = default_model_styles[member]["marker"]
                color = default_model_styles[member]["color"]
            else:
                marker = marker_pool[idx % len(marker_pool)]
                color = color_pool[idx % len(color_pool)]
            member_style_map[member] = {"marker": marker, "color": color, "linewidth": 1.50, "alpha": 0.8}
        if ordered_members:
            offsets = np.linspace(-0.27, 0.27, len(ordered_members))
            member_offset_map = {member: offsets[idx] for idx, member in enumerate(ordered_members)}
        else:
            member_offset_map = {}
        return ordered_members, member_style_map, member_offset_map


    def _draw_one_tag(tag, layout_mode, axis_override):
        era = pd.read_csv(_csv_path(tag, "era"))
        mem = pd.read_csv(_csv_path(tag, "mem"))
        mme = pd.read_csv(_csv_path(tag, "mme"))
        ordered_members, member_style_map, member_offset_map = _build_member_styles(mem)

        figsize = _style_value("figure_size", [21.8, 14.0])
        if not (isinstance(figsize, (list, tuple)) and len(figsize) == 2):
            figsize = [21.8, 14.0]
        fig = plt.figure(figsize=tuple(figsize))
        outer = GridSpec(
            nrows=2,
            ncols=2,
            width_ratios=[1.0, 1.0],
            height_ratios=[1.0, 1.0],
            hspace=0.30,
            wspace=0.10,
            figure=fig,
        )
        main_axes = np.empty((2, 2), dtype=object)
        band_axes = np.empty((2, 2), dtype=object)
        for row_idx in range(2):
            for col_idx in range(2):
                inner = outer[row_idx, col_idx].subgridspec(2, 1, height_ratios=[5.0, 0.78], hspace=0.03)
                main_axes[row_idx, col_idx] = fig.add_subplot(inner[0, 0])
                band_axes[row_idx, col_idx] = fig.add_subplot(inner[1, 0], sharex=main_axes[row_idx, col_idx])

        key_ax = fig.add_axes([0.055, 0.020, 0.89, 0.19])
        _draw_member_key(key_ax, ordered_members, member_style_map)

        era_halfwidth = 0.115
        mme_halfwidth = 0.115

        shared_component_ylim = {}
        for comp_name, col_name, _title in components:
            pooled_all = []
            for var_name in vars_order:
                era_v_tmp = era[era["var"] == var_name].set_index("region")
                mem_v_tmp = mem[mem["var"] == var_name]
                mme_v_tmp = mme[mme["var"] == var_name].set_index("region")
                era_vals_tmp = _ordered_series(era_v_tmp, col_name)
                pooled_all.extend(era_vals_tmp[np.isfinite(era_vals_tmp)].tolist())
                mme_vals_tmp = _ordered_series(mme_v_tmp, col_name)
                pooled_all.extend(mme_vals_tmp[np.isfinite(mme_vals_tmp)].tolist())
                for region in region_order:
                    region_rows_tmp = mem_v_tmp[mem_v_tmp["region"] == region].dropna(subset=[col_name])
                    if len(region_rows_tmp) > 0:
                        pooled_all.extend(region_rows_tmp[col_name].dropna().astype(float).tolist())
            shared_component_ylim[comp_name] = _panel_ylim(pooled_all)

        panel_idx = 0
        for row_idx, var_name in enumerate(vars_order):
            era_v = era[era["var"] == var_name].set_index("region")
            mem_v = mem[mem["var"] == var_name]
            mme_v = mme[mme["var"] == var_name].set_index("region")

            for col_idx, (comp_name, col_name, title) in enumerate(components):
                ax = main_axes[row_idx, col_idx]
                band_ax = band_axes[row_idx, col_idx]
                era_vals = _ordered_series(era_v, col_name)
                mme_vals = _ordered_series(mme_v, col_name)
                pooled = era_vals[np.isfinite(era_vals)].tolist() + mme_vals[np.isfinite(mme_vals)].tolist()

                exceed_pcts = []
                for xi, region in enumerate(region_order):
                    region_rows = mem_v[mem_v["region"] == region].dropna(subset=[col_name]).reset_index(drop=True)
                    pooled.extend(region_rows[col_name].dropna().astype(float).tolist())
                    _plot_member_points(ax, region_rows, col_name, xi, layout_mode, member_style_map, member_offset_map)
                    era_ref = era_vals[xi]
                    member_vals = region_rows[col_name].values.astype(float) if len(region_rows) > 0 else np.array([], dtype=float)
                    if len(member_vals) > 0 and np.isfinite(era_ref):
                        exceed_pct = 100.0 * np.sum(member_vals > era_ref) / len(member_vals)
                    else:
                        exceed_pct = np.nan
                    exceed_pcts.append(exceed_pct)

                for xi, mme_y in enumerate(mme_vals):
                    if np.isfinite(mme_y):
                        ax.plot([xi - mme_halfwidth - 0.1, xi + mme_halfwidth + 0.1], [mme_y, mme_y], color=mme_color, linewidth=3.0, solid_capstyle="round", zorder=8)
                for xi, era_y in enumerate(era_vals):
                    if np.isfinite(era_y):
                        ax.plot([xi - era_halfwidth - 0.1, xi + era_halfwidth + 0.1], [era_y, era_y], color="k", linewidth=2.8, solid_capstyle="round", zorder=7)

                override_key = f"{var_name}_{comp_name}"
                if override_key in axis_override:
                    cfg = axis_override[override_key]
                    panel_ymin, panel_ymax = cfg.get("ylim", shared_component_ylim[comp_name][:2])
                    panel_yticks = cfg.get("yticks", np.linspace(panel_ymin, panel_ymax, 5))
                else:
                    panel_ymin, panel_ymax, panel_yticks = shared_component_ylim[comp_name]

                ax.axhline(0.0, color="0.48", linewidth=1.0, zorder=1, linestyle="--")
                ax.axvline(boundary_x, color="0.62", linestyle="--", linewidth=1.0)
                ax.set_xlim(-0.62, len(region_order) - 0.38)
                ax.set_ylim(panel_ymin, panel_ymax)
                ax.set_yticks(panel_yticks)
                ax.yaxis.set_major_formatter(FuncFormatter(_fmt_y))
                ax.set_title(
                    _panel_title(var_name, comp_name, title, panel_idx),
                    loc="center",
                    pad=_style_value("title_pad", 14),
                    fontsize=_style_value("title_fontsize", plt.rcParams.get("axes.titlesize", 18)),
                    fontweight=_style_value("title_fontweight", "normal"),
                )
                _add_panel_label(ax, panel_idx)
                panel_idx += 1
                ax.tick_params(axis="both", labelsize=_style_value("main_tick_fontsize", plt.rcParams.get("xtick.labelsize", 14)))
                ax.tick_params(axis="x", labelbottom=False, labelsize=_style_value("main_xtick_fontsize", _style_value("main_tick_fontsize", plt.rcParams.get("xtick.labelsize", 14))))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if col_idx == 0:
                    ax.set_ylabel(_style_value("ylabel", "Trend (?C/yr)"), labelpad=_style_value("ylabel_pad", 18), fontsize=_style_value("ylabel_fontsize", plt.rcParams.get("axes.labelsize", 17)))
                ax.text(0.26, 0.955, _style_value("over_label", "Overestimated"), transform=ax.transAxes, ha="center", va="top", fontsize=_style_value("annotation_fontsize", 16.2), color="0.2")
                ax.text(0.77, 0.955, _style_value("under_label", "Underestimated"), transform=ax.transAxes, ha="center", va="top", fontsize=_style_value("annotation_fontsize", 16.2), color="0.2")

                band_ax.axvline(boundary_x, color="0.62", linestyle="--", linewidth=1.0)
                band_ax.set_ylim(0, 1)
                band_ax.set_yticks([])
                band_ax.set_xticks(x)
                band_ax.set_xticklabels(region_order, rotation=0, fontsize=_style_value("band_xtick_fontsize", 16))
                band_ax.set_xlim(-0.62, len(region_order) - 0.38)
                band_ax.spines["top"].set_visible(False)
                band_ax.spines["right"].set_visible(False)
                band_ax.spines["left"].set_visible(False)
                band_ax.spines["bottom"].set_linewidth(0.8)
                band_ax.tick_params(axis="x", pad=3)

                for xi, pct in enumerate(exceed_pcts):
                    color = cmap(0.08) if not np.isfinite(pct) else cmap(pct / 100.0)
                    band_ax.bar(xi, 0.78, width=0.82, bottom=0.10, color=color, edgecolor="0.78", linewidth=0.6)
                    pct_label = "NA" if not np.isfinite(pct) else f"{pct:.0f}%"
                    text_color = "white" if np.isfinite(pct) and pct >= 50 else "0.18"
                    band_ax.text(xi, 0.49, pct_label, ha="center", va="center", fontsize=_style_value("band_value_fontsize", 12), color=text_color, fontweight="semibold")

                if col_idx == 0:
                    band_ax.set_ylabel(_style_value("band_ylabel", "Models >\nERA5 (%)"), rotation=0, labelpad=_style_value("band_ylabel_pad", 25), fontsize=_style_value("band_ylabel_fontsize", 12.2), va="center")

        fig.subplots_adjust(left=_style_value("subplots_left", 0.055), right=_style_value("subplots_right", 0.978), top=_style_value("subplots_top", 0.96), bottom=_style_value("subplots_bottom", 0.24))
        out_name = f"DA_Ridge_daily_{tag}_region_trends_hwc_dyn_thermo_modelstyle_{layout_mode}.png"
        if "save_figure_multi_format" in globals():
            default_stem = os.path.splitext(out_name)[0]
            out_stem = globals().get("DA_NOTEBOOK_OUT_NAME_MAP", {}).get((tag, layout_mode), default_stem)
            out_fig = save_figure_multi_format(fig, out_stem, dpi=_style_value("output_dpi", 320))
        else:
            out_fig = os.path.join(FIG_DIR, out_name)
            fig.savefig(out_fig, dpi=_style_value("output_dpi", 320), bbox_inches="tight")
        if _ipython_display is not None:
            try:
                _ipython_display(fig)
            except Exception:
                pass
        plt.close(fig)
        print("Saved:", out_fig)
        return out_fig
    def main():
        axis_override = _load_axis_override()
        saved = []
        for tag in VARIANT_TAGS:
            for layout_mode in ["beeswarm"]:
                saved.append(_draw_one_tag(tag, layout_mode, axis_override))
        print("Generated figures:")
        for path in saved:
            print(" -", path)


    if __name__ == "__main__":
        main()
