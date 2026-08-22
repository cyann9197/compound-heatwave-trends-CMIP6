"""Preserved shared context used by the independently runnable figure scripts.

The numerical and plotting routines originate from the verified final-figure
notebook. Personal paths and notebook-only display behavior are replaced by the
portable runtime configuration.
"""

from chw_cmip6.runtime import get_runtime
from chw_cmip6.plotting import configure_fonts

RUNTIME = get_runtime()
CONFIG = RUNTIME.config
DATA_ROOT = CONFIG.data_root
CACHE_ROOT = CONFIG.cache_root
OUTPUT_ROOT = CONFIG.output_root
configure_fonts()

# GIC 权威区域合同：所有正式区域结果统一使用 55–22°W、71–83°N。
GIC_BOUNDS = (-55, -22, 71, 83)
GIC_REGION_BOUNDS_SIGNATURE = "GIC:-55:-22:71:83"


def region_bounds_signature():
    """返回当前正式 GIC 边界指纹，用于识别旧范围缓存。"""
    return GIC_REGION_BOUNDS_SIGNATURE


def validate_region_bounds_signature(value):
    """仅接受与当前正式 GIC 边界完全一致的缓存指纹。"""
    return str(value) == region_bounds_signature()


def validate_gic_dataframe(frame):
    """核验区域结果表中的 GIC 边界是否唯一为 71–83°N。"""
    required = {"region", "lon_min", "lon_max", "lat_min", "lat_max"}
    if frame is None or not required.issubset(frame.columns):
        return False
    rows = frame.loc[frame["region"].astype(str).eq("GIC")]
    if rows.empty:
        return False
    bounds = rows[["lon_min", "lon_max", "lat_min", "lat_max"]].drop_duplicates()
    if len(bounds) != 1:
        return False
    return tuple(float(v) for v in bounds.iloc[0]) == tuple(float(v) for v in GIC_BOUNDS)


import os
import json
import copy
import pickle
from pathlib import Path

try:
    from IPython.display import display, Image as IPImage
except Exception:
    display = None
    IPImage = None

import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy import stats
from sklearn.linear_model import LinearRegression

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import matplotlib.ticker as mticker
import geocat.viz as gv
import cmaps

STRICT_FINAL_OUTPUT_DIR = OUTPUT_ROOT
STRICT_FINAL_CACHE_DIR = STRICT_FINAL_OUTPUT_DIR / "strict_minimal_cache"
FINAL_FIG_CACHE_DIR = STRICT_FINAL_OUTPUT_DIR / "cache"
STRICT_FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STRICT_FINAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_FIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

FINAL_NATURE_BASE_DIR = DATA_ROOT
FINAL_NATURE_CACHE_DIR = CACHE_ROOT
PREPARED_PREREQ_PICKLE = CACHE_ROOT / "prepared_prereqs.pkl"

FILE_REGIONAL_MEANS = FINAL_NATURE_CACHE_DIR / "compound_regional_means_dual_scenarios.nc"
FILE_HEATWAVE_RATIO = FINAL_NATURE_CACHE_DIR / "heatwave_percentage_maps.nc"
FILE_PHYSICS_RATIO_ALLVARS = FINAL_NATURE_CACHE_DIR / "physical_percentage_allvars.nc"
FILE_CORR_DAYS_PHYSICS_ALLVARS = FINAL_NATURE_CACHE_DIR / "correlation_days_physics_allvars.nc"
FILE_CORR_HEAT_PHYSICS_ALLVARS = FINAL_NATURE_CACHE_DIR / "correlation_cumulative_heat_physics_allvars.nc"
STRICT_FONT_DIR = DATA_ROOT / "fonts"

STRICT_PREPARE_MINIMAL_CACHES = RUNTIME.recompute
FORCE_RECOMPUTE_MINIMAL_CACHES = RUNTIME.recompute

FIG1_RANKING_CACHE = STRICT_FINAL_CACHE_DIR / "fig1_compound_ranking_1981_2014.pkl"
FIG2_BOXPLOT_CACHE = STRICT_FINAL_CACHE_DIR / "fig2_boxplot_trends_1981_2014.npz"
FIG2_CUMHEAT_BOXPLOT_CACHE = STRICT_FINAL_CACHE_DIR / "fig2_cumheat_boxplot_trends_1981_2014.npz"
FIG2_PREPARED_CACHE = FINAL_FIG_CACHE_DIR / "fig2_prepared_plot_data_v1.pkl"
HOTSPOT_REGION_BOXPLOT_CACHE = FINAL_FIG_CACHE_DIR / "hotspot_nonhot_region_boxplots_payload_v1.pkl"
ERA5_TREND_CACHE = STRICT_FINAL_CACHE_DIR / "supp_era5_three_hw_trends_1981_2014.nc"
MME_TREND_CACHE = STRICT_FINAL_CACHE_DIR / "supp_mme_three_hw_trends_1981_2014.nc"
FINAL_MANIFEST = STRICT_FINAL_OUTPUT_DIR / "final_figure_manifest.json"

FIG2_EXTREME_BASELINE_MODE = "fixed_jja_mean"
FIGURE_STATUS = {}
EXPORTED_FIGURES = {}
_PREPARED_CONTEXT = None
_OBS_HEATWAVE_DAYS_CACHE = {}
_OBS_HEATWAVE_CH_CACHE = {}
_MEMBER_HEATWAVE_DAYS_CACHE = {}
_MEMBER_HEATWAVE_CH_CACHE = {}
_THREE_HW_TREND_CACHE_MEMORY = {}
_FIG2_BOXPLOT_MEMORY_CACHE = None
_FIG2_CUMHEAT_BOXPLOT_MEMORY_CACHE = None
_FIG2_PREPARED_MEMORY_CACHE = None
_HOTSPOT_REGION_BOXPLOT_MEMORY_CACHE = None


sc=1.2
FONT_PANEL = 23  #序号
FONT_TITLE = 19  #标题
FONT_LABEL = 16
FONT_TICK = 14.5
FONT_LEGEND = 12
FONT_CBAR = 14
FONT_ANNOTATION = 14



PANEL_LABEL_PT = FONT_PANEL
PANEL_TITLE_PT = FONT_TITLE
AXIS_LABEL_PT = FONT_LABEL
AXIS_TICK_PT = FONT_TICK
CBAR_TICK_PT = FONT_CBAR
CBAR_LABEL_PT = FONT_CBAR
LEGEND_PT = FONT_LEGEND
MAP_TICK_PT = FONT_TICK
ANNOTATION_PT = FONT_ANNOTATION
MONTHLY_TITLE_PAD = 10
PANEL_LABEL_X_WITH_TITLE = -0.06
PANEL_LABEL_Y_WITH_TITLE = 1.045
PANEL_LABEL_X_NO_TITLE = -0.06
PANEL_LABEL_Y_NO_TITLE = 1.035

def _scaled_pt(base_pt, scale):
    return round(float(base_pt) * float(scale), 1)

# ===== 地图型空间图统一版式参数 =====
# 若需要调整地图子图之间的行距，请修改 MAP_GRID_HSPACE。
# 若需要调整地图子图之间的列距，请修改 MAP_GRID_WSPACE。
# 若需要调整 colorbar 与地图之间的距离，请修改 MAP_CBAR_Y_FRAC（数值越大越靠近地图）以及 MAP_CBAR_LABEL_Y。
# 若需要调整 colorbar 尺寸，请修改 MAP_CBAR_WIDTH_FRAC / MAP_CBAR_HEIGHT_FRAC / MAP_CBAR_ROW_HEIGHT。
# 若需要调整标题与图框的距离，请修改 MAP_TITLE_PAD。
# 若需要调整 panel label 位置，请修改 PANEL_LABEL_X_WITH_TITLE / PANEL_LABEL_Y_WITH_TITLE / PANEL_LABEL_Y_NO_TITLE。
MAP_GRID_HSPACE = 0 ##
MAP_GRID_WSPACE = 0.070
MAP_CBAR_ROW_HEIGHT = 0.03
MAP_CBAR_WIDTH_FRAC = 0.56
MAP_CBAR_HEIGHT_FRAC = 1.5
MAP_CBAR_Y_FRAC = 0.35 ##
MAP_CBAR_LABEL_Y = 1
MAP_SUBPLOTS_LEFT = 0.070
MAP_SUBPLOTS_RIGHT = 0.985
MAP_SUBPLOTS_TOP = 0.934
MAP_SUBPLOTS_BOTTOM = 0.064
MAP_TITLE_PAD = 5

# 若需要继续调整 Fig.5–Fig.7 与 linear budget 图的文字大小，请修改 FIG567_FONT_SCALE。
FIG567_FONT_SCALE = 1.3
FIG567_PANEL_LABEL_PT = _scaled_pt(PANEL_LABEL_PT, FIG567_FONT_SCALE)
FIG567_PANEL_TITLE_PT = _scaled_pt(PANEL_TITLE_PT, FIG567_FONT_SCALE)
FIG567_AXIS_LABEL_PT = _scaled_pt(AXIS_LABEL_PT, FIG567_FONT_SCALE)
FIG567_AXIS_TICK_PT = _scaled_pt(AXIS_TICK_PT, FIG567_FONT_SCALE)
FIG567_LEGEND_PT = _scaled_pt(LEGEND_PT, FIG567_FONT_SCALE)
FIG567_CBAR_LABEL_PT = _scaled_pt(CBAR_LABEL_PT, FIG567_FONT_SCALE)
FIG567_ANNOTATION_PT = _scaled_pt(ANNOTATION_PT, FIG567_FONT_SCALE)

mpl.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": AXIS_LABEL_PT,
    "axes.titlesize": PANEL_TITLE_PT,
    "axes.labelsize": AXIS_LABEL_PT,
    "xtick.labelsize": AXIS_TICK_PT,
    "ytick.labelsize": AXIS_TICK_PT,
    "legend.fontsize": LEGEND_PT,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

FINAL_DA_RCPARAMS = {
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": FIG567_AXIS_LABEL_PT,
    "axes.titlesize": FIG567_PANEL_TITLE_PT,
    "axes.labelsize": FIG567_AXIS_LABEL_PT,
    "xtick.labelsize": FIG567_AXIS_TICK_PT,
    "ytick.labelsize": FIG567_AXIS_TICK_PT,
    "legend.fontsize": FIG567_LEGEND_PT,
    "axes.linewidth": 0.9,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}

OBS_COLOR_MAP = {
    "CPC": "#C44E52",
    "BEST": "#B07AA1",
    "ERA5": "#64B5CD",
    "MERRA2": "#4C72B0",
    "JRA3Q": "#DD8452",
    "JRA-3Q": "#DD8452",
}

HOT_BOX_COLOR = "#BE4DD2"
NOTHOT_BOX_COLOR = "#1F9A8A"

NATURE_PERCENT_CMAP = LinearSegmentedColormap.from_list(
    "nature_percent_refined",
    ["#6FA3D8", "#F8F6F1", "#C96A3D"],
    N=256,
)
PHYSICS_RATIO_CMAP = NATURE_PERCENT_CMAP
PHYSICS_POSITIVE_BAR_COLOR = "#8DAE82"
PHYSICS_NEGATIVE_BAR_COLOR = "#C8A18C"
PROJ = ccrs.PlateCarree()


def normalize_panel_title(text):
    text = str(text)
    text = text.replace("Daytime_HW", "Daytime HW")
    text = text.replace("Nighttime_HW", "Nighttime HW")
    text = text.replace("Compound_HW", "Compound HW")
    text = text.replace("/yr", " yr$^{-1}$")
    text = text.replace("Correlations (in CMIP6 models)", "Inter-model correlation")
    return text


def canonical_dataset_label(text):
    text = str(text)
    return {"JRA-3Q": "JRA3Q"}.get(text, text)


def save_figure_multi_format(fig, stem, dpi=300, bbox_inches="tight"):
    stem = str(stem)
    for ext in (".png", ".pdf", ".svg"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    out_base = STRICT_FINAL_OUTPUT_DIR / stem
    png_path = None
    for ext in ("png", "pdf", "svg"):
        out_path = out_base.with_suffix(f".{ext}")
        save_kwargs = {"bbox_inches": bbox_inches}
        if ext == "png":
            save_kwargs["dpi"] = dpi
            png_path = str(out_path)
        fig.savefig(out_path, **save_kwargs)
    EXPORTED_FIGURES[stem] = [
        str(out_base.with_suffix(".png")),
        str(out_base.with_suffix(".pdf")),
        str(out_base.with_suffix(".svg")),
    ]
    return png_path


def display_saved_png(stem):
    if display is None or IPImage is None:
        return
    png_path = STRICT_FINAL_OUTPUT_DIR / f"{stem}.png"
    if png_path.exists():
        display(IPImage(filename=str(png_path)))


def finalize_figure(fig, stem=None, extra_stems=None, dpi=300, bbox_inches="tight", close_after=False):
    if stem is not None:
        save_figure_multi_format(fig, stem, dpi=dpi, bbox_inches=bbox_inches)
    for extra_stem in (extra_stems or []):
        save_figure_multi_format(fig, extra_stem, dpi=dpi, bbox_inches=bbox_inches)
    if display is not None:
        display(fig)
    else:
        plt.show()
    if close_after:
        plt.close(fig)


def record_status(name, status, note=""):
    FIGURE_STATUS[name] = {"status": status, "note": note}


def add_panel_label(ax, letter, x=PANEL_LABEL_X_WITH_TITLE, y=PANEL_LABEL_Y_WITH_TITLE, fontsize=PANEL_LABEL_PT):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color="0.1",
    )


def apply_panel_header(ax, letter, title=None, title_pad=5,fsize=PANEL_TITLE_PT, has_title=True):
    if has_title and title:
        ax.set_title(
            normalize_panel_title(title),
            fontsize=fsize,
            pad=title_pad,
            loc="center",
            fontweight="normal",
        )
        add_panel_label(ax, letter, x=PANEL_LABEL_X_WITH_TITLE, y=PANEL_LABEL_Y_WITH_TITLE, fontsize=fsize+3)
    else:
        add_panel_label(ax, letter, x=PANEL_LABEL_X_NO_TITLE, y=PANEL_LABEL_Y_NO_TITLE, fontsize=fsize+3)


def reorder_legend_handles_rowwise(handles, labels, ncol=2):
    handles = list(handles)
    labels = list(labels)
    n_items = len(labels)
    if n_items <= 1 or ncol <= 1:
        return handles, labels
    nrows = int(np.ceil(n_items / ncol))
    order = []
    for col in range(ncol):
        for row in range(nrows):
            idx = row * ncol + col
            if idx < n_items:
                order.append(idx)
    return [handles[i] for i in order], [labels[i] for i in order]


def set_horizontal_cbar_label_top(cb, label, y=MAP_CBAR_LABEL_Y):
    cb.set_label("")
    cb.ax.text(0.5, y, label, transform=cb.ax.transAxes, ha="center", va="bottom", fontsize=CBAR_LABEL_PT)


def set_horizontal_cbar_percent_top(cb, label="%"):
    set_horizontal_cbar_label_top(cb, label, y=MAP_CBAR_LABEL_Y)


def shrink_horizontal_cbar_axis(cax, width_frac=MAP_CBAR_WIDTH_FRAC, height_frac=MAP_CBAR_HEIGHT_FRAC, y_frac=MAP_CBAR_Y_FRAC):
    pos = cax.get_position()
    new_width = pos.width * width_frac
    new_height = pos.height * height_frac
    new_x0 = pos.x0 + (pos.width - new_width) / 2
    new_y0 = pos.y0 + pos.height * y_frac
    cax.set_position([new_x0, new_y0, new_width, new_height])
    return cax


def apply_map_figure_layout(
    fig,
    cb_axes=None,
    left=MAP_SUBPLOTS_LEFT,
    right=MAP_SUBPLOTS_RIGHT,
    top=MAP_SUBPLOTS_TOP,
    bottom=MAP_SUBPLOTS_BOTTOM,
    cbar_width_frac=MAP_CBAR_WIDTH_FRAC,
    cbar_height_frac=MAP_CBAR_HEIGHT_FRAC,
    cbar_y_frac=MAP_CBAR_Y_FRAC,
):
    # 以后如果想继续调地图排版，优先改 cell 6 里的 MAP_* 参数：
    # - MAP_GRID_HSPACE / MAP_GRID_WSPACE 控制地图子图行距和列距
    # - MAP_CBAR_Y_FRAC / MAP_CBAR_LABEL_Y 控制 colorbar 与地图和文字的距离
    # - MAP_CBAR_WIDTH_FRAC / MAP_CBAR_HEIGHT_FRAC / MAP_CBAR_ROW_HEIGHT 控制 colorbar 尺寸
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    if cb_axes is None:
        return fig
    if not isinstance(cb_axes, (list, tuple)):
        cb_axes = [cb_axes]
    for cax in cb_axes:
        shrink_horizontal_cbar_axis(
            cax,
            width_frac=cbar_width_frac,
            height_frac=cbar_height_frac,
            y_frac=cbar_y_frac,
        )
    return fig


def draw_strict_boxes(ax):
    draw_rectangles(ax, hot_regions, edgecolor=HOT_BOX_COLOR)
    draw_rectangles(ax, nothot_regions, edgecolor=NOTHOT_BOX_COLOR)


import sys
# 已移除个人 Python 搜索路径；公开版只使用 environment.yml 中的依赖。
import pandas as pd
#from fnmatch import fnmatchc as easmatch
#import pyunicorn as pc
import xarray as xr
import numpy as np
import datetime as dt
import glob
from minisom import MiniSom
import datetime as dt
import warnings
warnings.filterwarnings(action='ignore')
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
# 原私有 pack.maskout 未被最终缓存绘图链调用，公开版不再导入。
import cartopy.crs as ccrs
from scipy.stats.mstats import ttest_ind
from scipy import signal
from matplotlib.patches import Rectangle
# 原私有 pack.maskout 未被最终缓存绘图链调用，公开版不再导入。
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeat
from cartopy.util import add_cyclic_point
import xarray as xr
from scipy import stats
import datetime as dt
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
import matplotlib.ticker as mticker
import geocat.viz as gv
import cmaps
from scipy.stats.mstats import ttest_ind
import matplotlib.patches as mpatches
import sacpy as scp
from netCDF4 import Dataset
import sacpy.Map
from fnmatch import fnmatchcase as match
import matplotlib as mpl
import metpy.calc as mpcalc
from metpy.units import units
import seaborn as sns
import xesmf as xe
import regionmask
#import xskillscore as xs
# 手动指定字体文件路径
#font_path = '/usr/share/fonts/'
##plt.rcParams['font.family'] = font_path
##plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.family'] = 'sans-serif'
##plt.rcParams['font.family']='Arial'
##plt.rcParams['font.size']=10
##plt.rcParams['font.weight']='bold'
SHP = None

def make_map(ax,box,fsize=16):#box:
    #加国界
    #ax.add_geometries(Reader(SHP).geometries(),ccrs.PlateCarree(),facecolor='none',edgecolor='k',linewidth=0.6,zorder=1)
    #标注坐标轴
    ax.set_xticks(np.arange(box[0],box[1],60),crs=ccrs.PlateCarree())#指定要显示的经纬度
    #ax.set_xticks(np.array([-80,-40,0,40,80,120,160]))
    ax.set_yticks(np.arange(box[2],box[3]+30,30),crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())#刻度格式转换为经纬度样式
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='both',which='major',labelsize=fsize,direction='out',length=5,width=1,pad=5)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))#刻度格式转换为经纬度样式
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis='both',which='minor',direction='out',length=2,width=0.4)
    ax.spines['geo'].set_linewidth(1)#调节边框粗细
    ax.set_extent([box[0],box[1],box[2],box[3]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),lw=0.6)#####添加海岸线#########
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white',zorder = 2)######添加海洋########

    #添加网格线
    #ax.gridlines(linestyle='--',alpha=0.4)
    return ax

#经度翻转函数
def filp_lon(ds,lon_name = 'longitude'):
    #0#360to-180~180
    if lon_name == 'lon':
        ds=ds.sortby('lat',ascending=True)
    else:
        ds=ds.sortby('latitude',ascending=True)
    ds['longitude_adjusted'] = xr.where(
        ds[lon_name] > 180,
        ds[lon_name] - 360,
        ds[lon_name])
    ds = (
        ds
        .swap_dims({lon_name: 'longitude_adjusted'})
        .sel(**{'longitude_adjusted': sorted(ds.longitude_adjusted)})
        .drop(lon_name))
    ds = ds.rename({'longitude_adjusted': 'lon'})
    if 'latitude' in ds.dims or 'latitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat'})
    # --- 3. 时间维度统一命名为 time ---
    if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})

    return ds

#掩膜函数
def mask_landsea(ds,lat_name='lat', label='land'): #所有数据都是-180到180
    landsea = xr.open_dataset(DATA_ROOT / "static" / "landsea.nc")
    landsea=filp_lon(landsea,lon_name = 'longitude')
    landsea = landsea['lsm'][0,:,:]
    rename_dict = {}
    if 'latitude' in landsea.coords:
        rename_dict['latitude'] = 'lat'
    if 'longitude' in landsea.coords:
        rename_dict['longitude'] = 'lon'
    if rename_dict:
        landsea = landsea.rename(rename_dict)
    #ds和地形数据分辨率不一致，需将地形数据插值
    rename_dict = {}
    if 'latitude' in ds.coords:
        rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        rename_dict['longitude'] = 'lon'
    if rename_dict:
        ds = ds.rename(rename_dict)
    if lat_name == 'lat':
        landsea = landsea.interp(lat=ds.lat.values, lon=ds.lon.values)
         #利用地形掩盖海陆数据
        ds.coords['mask'] = (('lat', 'lon'), landsea.values)
    # print(ds.mask)
    if label == 'land':
        ds = ds.where(ds.mask < 0.8) #可以尝试调整default：0.8
    elif label == 'ocean':
        ds = ds.where(ds.mask > 0.2) #可以尝试调整default：0.2

    ds=filp_lon(ds,lon_name = 'lon')

    return ds

#第一个代表计算的是mean还是trend，第二个代表以mean还是rc分类
def select_models(dataset, model_list, is_member=False):
    """根据变量类型选择模式"""
    if is_member:  # swcre, lwcre 特殊情况
        return dataset.sel(members=list(set(dataset.members.values) & set(model_list)))
    else:  # 其他变量直接 loc
        return dataset.loc[model_list]

from sklearn.linear_model import LinearRegression

def get_lr_stats(x, y, model):#显著性检验
    message0 = f'一元线性回归方程为: \ty={model.intercept_[0]:.3f} + {model.coef_[0][0]:.3f}*x'
    from scipy import stats
    n     = len(x)
    y_prd = model.predict(x)
    Regression = sum((y_prd - np.mean(y))**2) # 回归
    Residual   = sum((y - y_prd)**2)          # 残差
    R_square   = Regression / (Regression + Residual) # 相关性系数R^2
    F          = (Regression / 1) / (Residual / ( n - 2 ))  # F 分布
    pf         = stats.f.sf(F, 1, n-2)
    message1 = (f'相关系数(R^2): {R_square[0]:.3f}:\n' +
                f'回归分析(SSR): {Regression[0]:.3f}:\t残差(SSE):{Residual[0]:.3f}:\n' +
                f'           F : {F[0]:.3f}:\tpf : {pf[0]}')
    ## T
    L_xx  =  n * np.var(x)
    sigma =  np.sqrt(Residual / (n-2))
    t     =  model.coef_ * np.sqrt(L_xx) / sigma
    pt    =  stats.t.sf(t, n-2)
    message2 = f'           t : {t[0][0]:.3f}:\tpt : {pt[0][0]}'
    return  pt             #print(message0 +'\n' +message1 + '\n'+message2)
def Area_Mean(data, lat, lon):
    '''
    by XiaoMaFenJu
    data: 要进行区域加权平均的变量，支持2、3维  2D: [lat, lon]  3D：[time, lat, lon]
    lat: data2D对应的纬度 1D
    lon: data2D对应的经度 1D
    '''

    masked_data = np.ma.masked_invalid(data)
    if data.ndim == 2:
        y_weight2D = abs(np.cos(lat*np.pi/180))
        weight2D = np.expand_dims(y_weight2D, 1).repeat(len(lon), axis=1)
        # print(weight2D)
        new_data = np.ma.average(masked_data, weights=weight2D)
        return new_data
    elif data.ndim == 3:
        y_weight2D = abs(np.cos(lat*np.pi/180))
        weight2D = np.expand_dims(y_weight2D, 1).repeat(len(lon), axis=1)
        weight3D = np.expand_dims(weight2D,0).repeat(len(data[:,0,0]),axis=0)
        #print(weight3D.shape)
        new_data =  np.ma.average(masked_data, weights=weight3D, axis=(-1, -2))
        return new_data
    else:
        print('输入数据非2&3维')

import xarray as xr
import numpy as np

def area_weighted_mean(da, lat=1,lon=2,lon_dim='lon', lat_dim='lat', mask=None):
    """
    参数:
    da: xr.DataArray
        需要计算区域平均的数据，必须包含经纬度坐标。
    lon_dim: str, optional
        经度维度的名称，默认为 'lon'。
    lat_dim: str, optional
        纬度维度的名称，默认为 'lat'。

    返回:
    xr.DataArray
        计算了区域平均之后的结果，保留了其他所有维度。
    """
    # 1. 计算纬度权重 (cos(lat))
    #    纬度权重是根据数据本身的纬度坐标来计算的。
    weights = np.cos(np.deg2rad(da[lat_dim]))
    weights.name = "weights"

    # 2. 执行加权平均
    #    .weighted() 方法是xarray中进行加权计算的标准方法。
    #    .mean() 会作用在所有指定的维度上，这里是经度和纬度。
    #    由于没有预先选择子区域，平均值会在整个DataArray的空间范围上计算。

    if mask is not None:
        da = da.where(mask)

    return da.weighted(weights).mean(dim=(lon_dim, lat_dim))


def process_variable_CMIP6(var_name, model_list, target_months=[6, 7, 8]):
    """
    处理指定变量的 CMIP6 多模式数据，返回合并后的 Dataset
    """
    datasets_for_this_var = []
    print(f"\n处理变量: {var_name}...")
    for model in model_list:
        try:
            folder_path = f'/public/data3/yluo/CMIP6/hebin_1x1/historical/1x1/{var_name}/'#f'/public/data3/yluo/CMIP6/hebin_1x1/historical/1x1/{var_name}/'
            fiii = os.path.join(folder_path, f"*{model}_*.nc")
            files = glob.glob(fiii)
            if not files:
                raise FileNotFoundError(f"在 {folder_path} 中未找到模式 '{model}' 的 '{var_name}' 文件")

            file = files[0]
            model_var_ds = xr.open_dataset(file)

            # 坐标重命名
            def preprocess_model_data(ds):
                rename_dict = {}
                if 'latitude' in ds.coords:
                    rename_dict['latitude'] = 'lat'
                    rename_dict['longitude'] = 'lon'
                if rename_dict:
                    ds = ds.rename(rename_dict)
                return ds.sortby('lat')

            model_ds = preprocess_model_data(model_var_ds)

            # 日历标准化
            if hasattr(model_ds.time.to_index(), 'calendar'):
                calendar = model_ds.time.to_index().calendar
                #print(f"  模式 {model} 的日历为: {calendar}，正在标准化...")
                try:
                    if calendar == '360_day':
                        model_ds = model_ds.convert_calendar('proleptic_gregorian',
                                                             use_cftime=False, align_on="date")
                    else:
                        model_ds = model_ds.convert_calendar('proleptic_gregorian',
                                                             use_cftime=False)
                except AttributeError:
                    # 手动转换时间
                    new_time_values = [pd.Timestamp(year=t.year, month=t.month, day=t.day)
                                       for t in model_ds.time.values]
                    model_ds = model_ds.assign_coords(time=new_time_values)

            standardized_time = model_ds.time.to_index().to_period('M').to_timestamp()
            model_ds = model_ds.assign_coords(time=standardized_time)
            # 2. 为 Dataset 中的坐标添加标准属性
            # --- 为纬度（lat）添加属性 ---
            model_ds['lat'].attrs['units'] = 'degrees_north'
            model_ds['lat'].attrs['standard_name'] = 'latitude'
            model_ds['lat'].attrs['long_name'] = 'Latitude'
            model_ds['lat'].attrs['axis'] = 'Y'

            # --- 为经度（lon）添加属性 ---
            model_ds['lon'].attrs['units'] = 'degrees_east'
            model_ds['lon'].attrs['standard_name'] = 'longitude'
            model_ds['lon'].attrs['long_name'] = 'Longitude'
            model_ds['lon'].attrs['axis'] = 'X'


            # 时间筛选
            model_ds = model_ds.sel(time=slice('1981', '2023'))
            model_ds = model_ds.where(model_ds.time.dt.month.isin(target_months), drop=True)
            model_ds = model_ds.sel(lat=slice(0,90))
            # 删除无关变量
            vars_to_drop = ['time_bnds', 'lat_bnds', 'lon_bnds', 'plev_bnds', 'height']
            drop_vars = [v for v in vars_to_drop if v in model_ds.variables]
            if drop_vars:
                model_ds = model_ds.drop_vars(drop_vars)
            if 'plev' in model_ds.coords:
                # 1. 统一数据类型为 float64 确保是整数型
                model_ds = model_ds.assign_coords(plev=model_ds['plev'].round(0).astype('int'))

                # 2. 清空可能引起冲突的属性
                model_ds.plev.attrs = {}
                    # --- 标准化结束 ---
            datasets_for_this_var.append(model_ds)


        except Exception as e:
            print(f"处理模式 {model} 的变量 {var_name} 时出错: {e}，跳过该模式。")
            continue

    if not datasets_for_this_var:
        print(f"⚠️ 警告：未能成功加载任何模型的数据用于变量 {var_name}")
        return None



    print(f"\n正在合并所有模型的数据用于变量: {var_name}...")
    combined_var_ds = xr.concat(
    datasets_for_this_var,
    dim='members',)      # 覆盖不兼容的坐标变量
    combined_var_ds = combined_var_ds.assign_coords(members=model_list)
    # if 'plev' in combined_var_ds.coords:
    #     # 1.【最终修正】通过重建一个全新的 DataArray 来解决坐标问题
    #     # a. 识别有效数据层所在的索引位置
    #     sample_var = list(combined_var_ds.data_vars)[0]
    #     valid_counts = combined_var_ds[sample_var].count(dim=[d for d in combined_var_ds.dims if d != 'plev'])
    #     valid_indices = np.where(valid_counts > 0)[0]

    #     # b. 提取纯净的 NumPy 数据和坐标
    #     combined_var_ds_np_clean = combined_var_ds.isel(plev=valid_indices).values
    #     plev_np_clean = combined_var_ds.plev.isel(plev=valid_indices).values

    #     # c. 手动构建一个全新的、格式正确的 xarray.DataArray
    #     clean_coords = {
    #         'time': combined_var_ds.time,
    #         'plev': plev_np_clean,
    #         'lat': combined_var_ds.lat,
    #         'lon': combined_var_ds.lon
    #     }
    #     dims = ['time', 'plev', 'lat', 'lon']

    #     combined_var_ds = xr.DataArray(combined_var_ds_np_clean, coords=clean_coords, dims=dims)

    print(f"\n变量处理完成: {var_name}...")
    return combined_var_ds

import xarray as xr
import numpy as np
from scipy import stats

def calc_annual_trend(da, method="mean"):
    """
    计算多维数据的年际趋势 (K/year 或 mm/year)

    参数
    ----
    da : xr.DataArray
        输入数据，至少有 time 维度 (支持 members, lat, lon)
    method : str
        年聚合方式: "mean" 表示年平均, "sum" 表示年总和

    返回
    ----
    xr.Dataset，包含
        slope  : 年际趋势 (原始单位 / year)
        rvalue : 相关系数
        pvalue : 显著性水平
    """
    # ---- 1. 检查 time 维度类型并进行相应处理 ----
    # 检查 time 坐标是否为日期时间类型
    if np.issubdtype(da.time.dtype, np.datetime64):
        if method == "mean":
            da_year = da.groupby("time.year").mean("time")
        elif method == "sum":
            da_year = da.groupby("time.year").sum("time")
        else:
            raise ValueError("对于 datetime 类型的输入, method 必须是 'mean' 或 'sum'")
    else:
        # 如果 time 坐标不是日期时间类型 (例如，是整数年份)
        # 我们假设数据已经是年尺度，直接将 'time' 维度重命名为 'year'
        print("提示：检测到 'time' 维度为非日期时间格式，将直接作为年份处理。")
        da_year = da.rename({"time": "year"})

    # ---- 2. 获取自变量 (年份数组) ----
    years = da_year["year"].values

    # ---- 3. 定义回归函数 ----
    def linregress_func(y, x):
        mask = np.isfinite(y)
        if mask.sum() > 1:
            slope, intercept, r, p, stderr = stats.linregress(x[mask], y[mask])
            return slope, r, p
        else:
            return np.nan, np.nan, np.nan

    # ---- 4. 用 apply_ufunc 应用到空间维度 ----
    slope, rvalue, pvalue = xr.apply_ufunc(
        linregress_func,
        da_year,
        years,
        input_core_dims=[["year"], ["year"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float]
    )

    # ---- 5. 打包成 Dataset ----
    result = xr.Dataset(
        {
            "slope": slope,     # 趋势 (每年)
            "rvalue": rvalue,   # 相关系数
            "pvalue": pvalue    # 显著性水平
        }
    )
    return result

def compute_percentage(model_data, era_data):
    """
    model_data: (models, year, lat, lon)
    era_data:   (year, lat, lon)
    返回：百分比 (lat, lon) 0–100%
    """
    # 对模式和 ERA5 同一维度做平均
    model_mean = np.nanmean(model_data, axis=1)   # (models, lat, lon)
    era_mean   = np.nanmean(era_data, axis=0)     # (lat, lon)

    # 比较（30个模式是否 > 再分析）
    comp = model_mean > era_mean                  # (models, lat, lon)
    perc = 100 * np.mean(comp, axis=0)            # (lat, lon)

    return perc

def calc_trend_field(data, year):
    """
    data: (year, lat, lon)
    返回：trend(lat, lon)
    """
    ny, nlat, nlon = data.shape
    trend = np.zeros((nlat, nlon))

    for i in range(nlat):
        for j in range(nlon):
                ts=data[:, i, j]
                mask = ~np.isnan(ts)
                if mask.sum() >= 5:  # 至少5年数据才算趋势，按需调整
                     trend[i, j], intercept, r_value, p, std_err = stats.linregress(year[mask],ts[mask])
    return trend


def calc_trend_models(model_data, year):
    """
    model_data: (models, time, lat, lon) - 可以是 xarray 或 numpy
    year: (time,) - 必须是 numpy 数组
    """

    # -------------------------------------------------------
    # 步骤 1: 确保输入完全转为 Numpy 数组
    # -------------------------------------------------------
    if hasattr(model_data, 'values'):
        # 如果是 xarray，提取 values
        # ⚠️ 极其重要：确保维度顺序正确！
        # 假设你的数据顺序是 (model, time, lat, lon)
        # 如果不是，请先 model_data.transpose('model', 'time', 'lat', 'lon')
        data_np = model_data.values
    else:
        data_np = model_data

    # 检查 year 是否也是 numpy
    if hasattr(year, 'values'):
        year_np = year.values
    else:
        year_np = year

    # -------------------------------------------------------
    # 步骤 2: 获取维度并初始化
    # -------------------------------------------------------
    # 假设输入维度是 (models, time, lat, lon)
    nmodels, ny, nlat, nlon = data_np.shape

    trend_models = np.zeros((nmodels, nlat, nlon))

    # -------------------------------------------------------
    # 步骤 3: 循环计算
    # -------------------------------------------------------
    for k in range(nmodels):
        # 这里的 data_np[k] 现在已经是纯 numpy 数组了
        # calc_trend_field 必须是你之前定义的那个 Numba 函数
        trend_models[k] = calc_trend_field(data_np[k], year_np)

    return trend_models

def Auto_heatwave3(v, min_len=3):
    """
    v: 0/1 命中数组
    return vwave: 连续≥3天的热浪标记（与原逻辑完全一致）
    """
    import numpy as np

    # pad 前后可以让冲击检测更简单
    x = np.pad(v, ((0,0),(1,1),(0,0),(0,0)), constant_values=0)

    # 找 0→1 和 1→0 的突变位置
    diff = np.diff(x, axis=1)

    start = diff == 1   # 连续段开始
    end   = diff == -1  # 连续段结束

    # 取索引
    sidx = np.where(start)
    eidx = np.where(end)

    # 输出数组
    vwave = np.zeros_like(v)

    # zip 对应元素构造连续段
    for y, sy, lat, lon in zip(sidx[0], sidx[1], sidx[2], sidx[3]):
        # 查找匹配 end
        match = np.where(
            (eidx[0]==y)&(eidx[2]==lat)&(eidx[3]==lon)&(eidx[1]>sy)
        )[0][0]
        ey = eidx[1][match]

        length = ey - sy
        if length >= min_len:
            vwave[y, sy:ey, lat, lon] = 1

    return vwave


import numpy as np
from numba import jit
# 使用 @jit(nopython=True) 装饰器，强制编译为机器码
@jit(nopython=True)
def Auto_heatwave3(v):
    # 获取维度
    num_years, days_per_year, lat_size, lon_size = v.shape

    # 初始化结果数组 (数据类型与 v 保持一致，假设是 float 或 int)
    vwave = np.zeros_like(v)

    # 循环结构保持不变，但现在它们是编译过的机器码，速度极快
    for year in range(num_years):
        for lat in range(lat_size):
            for lon in range(lon_size):

                duration = 0
                # 遍历每一天
                for day in range(days_per_year):
                    val = v[year, day, lat, lon]

                    if val == 1.0:
                        duration += 1
                    else: # 当遇到 0 时，结算之前的热浪
                        if duration >= 3:
                            # 回溯赋值
                            vwave[year, day-duration:day, lat, lon] = 1
                        duration = 0

                # 检查每年结束时是否还有未结算的热浪
                if duration >= 3:
                    vwave[year, days_per_year - duration:days_per_year, lat, lon] = 1

    return vwave

def draw_rectangles(ax, regions, edgecolor, linewidth=2.0):
    """
    在地图上画矩形区域
    """
    for lon_min, lon_max, lat_min, lat_max in regions:
        rect = mpatches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor='none',
            transform=ccrs.PlateCarree(),
            zorder=10
        )
        ax.add_patch(rect)

model_list = ["ACCESS-CM2", "ACCESS-ESM1-5", "AWI-CM-1-1-MR", "BCC-ESM1",
                    "CanESM5", "CMCC-ESM2", "CNRM-CM6-1",  "CNRM-ESM2-1",
                    "E3SM-2-0", "E3SM-2-0-NARRM","EC-Earth3", "EC-Earth3-AerChem" ,"EC-Earth3-CC",
                    "EC-Earth3-Veg-LR", "FGOALS-f3-L","FGOALS-g3",
                     "HadGEM3-GC31-LL",
                    "HadGEM3-GC31-MM", "IITM-ESM", "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G",
                    "MIROC6", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR",
                    "NorESM2-LM", "NorESM2-MM", "TaiESM1", "UKESM1-0-LL"]#30

model_list23 = ["ACCESS-CM2", "ACCESS-ESM1-5", "AWI-CM-1-1-MR",
                    "CanESM5", "CMCC-ESM2", "CNRM-CM6-1",  "CNRM-ESM2-1",
                    "EC-Earth3","EC-Earth3-CC",
                    "EC-Earth3-Veg-LR", "FGOALS-g3",
                     "HadGEM3-GC31-LL",
                     "INM-CM4-8", "INM-CM5-0", "IPSL-CM6A-LR", "KACE-1-0-G",
                    "MIROC6", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR",
                    "NorESM2-LM", "NorESM2-MM", "TaiESM1", "UKESM1-0-LL"]#23

plevvalues=[100000,  92500,  85000,  70000,  60000,  50000,  40000,  30000,  25000,
        20000,  15000,  10000,   7000,   5000,   3000,   2000,   1000,    500,
          100]
plev=xr.DataArray(plevvalues,coords=[plevvalues],dims=['plev'])


import matplotlib.patches as mpatches
hot_regions = [
# (lon_min, lon_max, lat_min, lat_max)
(-115, -97, 44, 59), # NNA北美     (-115, -97, 44, 60),     # 北美
(-14, 3, 12, 28), # NAF非洲
(54,  71, 56, 65), # EEU东欧  
(69, 104, 27, 35), # SAS南亚
#(57,   69, 34, 50),      # 青藏高原
(100, 121, 54, 65), # ESB？ 东西伯利亚
]
# ================== 非热点区域 ==================
nothot_regions = [
(-106, -87, 30, 38),  # SNA北美
GIC_BOUNDS,  # GIC格陵兰附近
(24, 38, 47, 66), # WEU西偶 (24, 40, 55, 66)
#(99,  119, 38, 52),      # 东亚
(100, 119, 38, 52), # ENA东北亚
]

hotrgion_na=['NNA','NAF','EEU','SAS','ESB']
nothotrgion_na=['SNA','GIC','WEU','ENA']
def draw_rectangles(ax, regions, edgecolor, linewidth=2.0):
    """
    在地图上画矩形区域
    """
    for lon_min, lon_max, lat_min, lat_max in regions:
        rect = mpatches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor='none',
            transform=ccrs.PlateCarree(),
            zorder=10
        )
        ax.add_patch(rect)


import matplotlib.pyplot as plt
def get_ens_dim(ds):

    for d in ["model","member","members","ensemble"]:
        if d in ds.dims:
            return d

    raise ValueError("No ensemble dimension found!")

def calc_trend_annual(ds, regions, lat, lon):

    trends = []

    for lon1, lon2, lat1, lat2 in regions:

        sub = ds.sel(
            lon=slice(lon1, lon2),
            lat=slice(lat1, lat2)
        )

        area = area_weighted_mean(sub, lat, lon)

        # 直接对 year 回归
        slope = area.polyfit(
            dim="time",
            deg=1
        )["polyfit_coefficients"].sel(degree=1)

        trends.append(slope)

    return xr.concat(trends, dim="region")

    return slope   # 不乘10
def regional_interannual_trend(ds, regions, lat, lon):

    ds_annual = ds.groupby("time.year").mean("time")

    trends = []

    for lon1, lon2, lat1, lat2 in regions:

        sub = ds_annual.sel(
            lon=slice(lon1, lon2),
            lat=slice(lat1, lat2)
        )

        area = area_weighted_mean(sub, lat, lon)

        # ⭐⭐⭐⭐⭐ 正确polyfit方式
        slope = area.polyfit(dim="year", deg=1)["polyfit_coefficients"].sel(degree=1)

        trends.append(slope)

    return xr.concat(trends, dim="region")
def trend_with_mask(ds, mask):

    ds_annual = ds.groupby("time.year").mean("time")

    area = area_weighted_mean(ds_annual, mask=mask)

    slope = area.polyfit(
        dim="year",
        deg=1
    )["polyfit_coefficients"].sel(degree=1)

    return slope

def plot_trend_figure(
    datasets,
    reanalysis_dict,
    hot_regions,
    nothot_regions,
    hot_names,
    nothot_names,
    lat,
    lon,
    model_list
):

    variables = list(datasets.keys())

    for var in variables:

        print(f"Processing {var}...")

        ################################
        # CMIP
        ################################

        cmip, is_member = datasets[var]
        cmip = select_models(cmip, model_list, is_member)
        cmip = cmip.sel(time=slice("1981", "2014"))

        ######## NH ########
        nh_cmip = regional_interannual_trend(
            cmip,
            [(-180,180,0,90)],
            lat, lon
        )

        ######## regional ########

        hot_cmip = regional_interannual_trend(
            cmip, hot_regions, lat, lon
        )

        not_cmip = regional_interannual_trend(
            cmip, nothot_regions, lat, lon
        )

        ################################
        # OBS
        ################################

        obs_nh = []
        obs_hot = []
        obs_not = []

        obs_names_now=[]

        for name, ds in reanalysis_dict[var]:
            ds=ds.sel(time=slice("1981", "2014"))
            obs_names_now.append(name)

            obs_nh.append(
                regional_interannual_trend(
                    ds,[(-180,180,0,90)],lat,lon
                ).values
            )

            obs_hot.append(
                regional_interannual_trend(
                    ds, hot_regions, lat, lon
                ).values
            )

            obs_not.append(
                regional_interannual_trend(
                    ds, nothot_regions, lat, lon
                ).values
            )

        obs_nh = np.array(obs_nh)
        obs_hot = np.array(obs_hot)
        obs_not = np.array(obs_not)

        ################################
        # 合并（核心🔥）
        ################################

        cmip_all = np.concatenate([
            nh_cmip.values.reshape(1,-1),
            hot_cmip.values,
            not_cmip.values
        ], axis=0)

        labels = ["NH"] + hot_names + nothot_names

        obs_all = np.concatenate([
            obs_nh.reshape(len(obs_names_now),1),
            obs_hot,
            obs_not
        ], axis=1)

        ################################
        # PLOT
        ################################

        plt.figure(figsize=(8,6), dpi=300)


        spacing = 0.55   # ⭐ 控制密度（0.5~0.7最佳）

        positions = np.arange(len(labels)) * spacing + 1

        bp = plt.boxplot(
            cmip_all.T,
            positions=positions,
            widths=0.32,      # 可以略加宽一点
            patch_artist=True,
            showfliers=False,
            whis=1.5
        )
        ######## style ########

        for box in bp['boxes']:
            box.set(
                facecolor='#66c2a4',
                alpha=0.28,
                edgecolor='#66c2a4',
                linewidth=0.9
            )

        # ⭐ NH换颜色（高级感瞬间上来）
        bp['boxes'][0].set(
            facecolor='#66c2a4',#'#8da0cb',
            edgecolor='#66c2a4',
            alpha=0.35
        )

        for whisker in bp['whiskers']:
            whisker.set(
                color='#66c2a4',
                linewidth=0.8,
                alpha=0.8,
                linestyle='--'
            )
        for cap in bp['caps']:
            cap.set(
                color='#66c2a4',
                linewidth=0.8
            )

        for median in bp['medians']:
            median.set(color='#66c2a4', linewidth=1.1)

        ######## ensemble mean ########

        means = cmip_all.mean(axis=1)

        plt.scatter(
            positions,
            means,
            marker='x',
            s=55,
            color='#66c2a4',
            zorder=4
        )

        ######## OBS ########

        for obs_i in range(len(obs_names_now)):

            for x in range(len(labels)):
                plt.scatter(
                    positions[x],
                    obs_all[obs_i, x],
                    marker=obs_markers[obs_i],
                    s=45,
                    facecolors='none',
                    edgecolors='black',
                    linewidths=1.2,
                    alpha=0.8,
                    zorder=5,
                    label=obs_names[obs_i] if x==0 else None
                )

        ################################
        # decorations
        ################################

        plt.axvline(positions[0] + spacing/2, ls='--', alpha=0.25)
        plt.axvline(positions[len(hot_names)] + spacing/2, ls='--', alpha=0.25)


        plt.axhline(0, linestyle='--', linewidth=1, color='gray')

        # ymin = np.nanmin(cmip_all)
        # ymax = np.nanmax(cmip_all)
        # plt.ylim(ymin, ymax)

        ax = plt.gca()

        ax.text(positions[0]-0.1,0.92,"Northern\nHemisphere",
                transform=ax.get_xaxis_transform(),fontsize=12,ha='center')

        ax.text(positions[1:1+len(hot_names)].mean(),0.95,"Overestimated",
                transform=ax.get_xaxis_transform(),fontsize=13,
                ha='center')

        ax.text(
            positions[1+len(hot_names):].mean(),
            0.95,
            "Underestimated",
            transform=ax.get_xaxis_transform(),fontsize=13,
            ha='center'
        )


        plt.xticks(
            positions,
            labels,
            rotation=30,
            fontsize=13
        )


        plt.ylabel(f"{var.upper()} trend", fontsize=15)
        plt.legend(
            frameon=False,
            fontsize=8,
            ncol=5,
            loc='upper center',
            bbox_to_anchor=(0.76, 0.06),
            columnspacing=0.5,
            handletextpad=0.1,
            markerscale=0.8
        )
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()


def load_prepared_context():
    global _PREPARED_CONTEXT
    if _PREPARED_CONTEXT is None:
        if not PREPARED_PREREQ_PICKLE.exists():
            raise FileNotFoundError(f"缺少 prepared_prereqs.pkl：{PREPARED_PREREQ_PICKLE}")
        with PREPARED_PREREQ_PICKLE.open("rb") as f:
            _PREPARED_CONTEXT = pickle.load(f)
    return _PREPARED_CONTEXT


def context_item(name):
    ctx = load_prepared_context()
    if name not in ctx:
        raise KeyError(f"prepared_prereqs.pkl 中缺少变量：{name}")
    return ctx[name]


def fit_line_for_plot(years, values):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(years) & np.isfinite(values)
    if mask.sum() < 3:
        return np.nan, np.full_like(values, np.nan, dtype=float), np.nan, ""
    fit = stats.linregress(years[mask], values[mask])
    fitted = fit.slope * years + fit.intercept
    star = "*" if np.isfinite(fit.pvalue) and fit.pvalue <= 0.05 else ""
    return fit.slope, fitted, fit.pvalue, star


def _open_obs_heatwave_days(dataset_name, hw_type):
    key = (dataset_name, hw_type)
    if key in _OBS_HEATWAVE_DAYS_CACHE:
        return _OBS_HEATWAVE_DAYS_CACHE[key]
    file_map = {
        "compound": str(DATA_ROOT / "heatwaves_files" / f"{dataset_name}_heatwave_3.nc"),
        "day": str(DATA_ROOT / "heatwaves_files" / f"{dataset_name}_dayhw_3.nc"),
        "night": str(DATA_ROOT / "heatwaves_files" / f"{dataset_name}_nighthw_3.nc"),
    }
    ds = xr.open_dataset(file_map[hw_type]).sel(lat=slice(0, 90))
    ds = filp_lon(ds, lon_name="lon")
    values = np.nansum(ds["var"], axis=1)
    da = xr.DataArray(values, coords=[ds["year"], ds["lat"], ds["lon"]], dims=["time", "lat", "lon"])
    da = mask_landsea(da, lat_name="lat", label="ocean")
    da = da.sel(time=slice("1981", "2014"))
    _OBS_HEATWAVE_DAYS_CACHE[key] = da
    return da


def _open_obs_heatwave_ch(dataset_name, hw_type):
    key = (dataset_name, hw_type)
    if key in _OBS_HEATWAVE_CH_CACHE:
        return _OBS_HEATWAVE_CH_CACHE[key]
    file_map = {
        "compound": str(DATA_ROOT / "heatwaves_files" / f"{dataset_name}_compound_CumulativeHeat.nc"),
        "day": str(DATA_ROOT / "heatwaves_files" / f"{dataset_name}_day_CumulativeHeat.nc"),
        "night": str(DATA_ROOT / "heatwaves_files" / f"{dataset_name}_night_CumulativeHeat.nc"),
    }
    ds = xr.open_dataset(file_map[hw_type]).sel(lat=slice(0, 90))
    ds = filp_lon(ds, lon_name="lon")
    da = mask_landsea(ds["var"], lat_name="lat", label="ocean")
    da = da.sel(time=slice("1981", "2014"))
    _OBS_HEATWAVE_CH_CACHE[key] = da
    return da


def _open_member_heatwave_days(hw_type):
    if hw_type in _MEMBER_HEATWAVE_DAYS_CACHE:
        return _MEMBER_HEATWAVE_DAYS_CACHE[hw_type]
    file_map = {
        "compound": str(DATA_ROOT / "heatwaves_files" / "1x1" / "30member_heatwave_3.nc"),
        "day": str(DATA_ROOT / "heatwaves_files" / "1x1" / "30member_dayhw_3.nc"),
        "night": str(DATA_ROOT / "heatwaves_files" / "1x1" / "30member_nighthw_3.nc"),
    }
    ds = xr.open_dataset(file_map[hw_type]).sel(lat=slice(0, 90))
    ds = filp_lon(ds, lon_name="lon")
    da = mask_landsea(ds["var"].sel(time=slice("1981", "2014")), lat_name="lat", label="ocean")
    _MEMBER_HEATWAVE_DAYS_CACHE[hw_type] = da
    return da


def _open_member_heatwave_ch(hw_type):
    if hw_type in _MEMBER_HEATWAVE_CH_CACHE:
        return _MEMBER_HEATWAVE_CH_CACHE[hw_type]
    file_map = {
        "compound": str(DATA_ROOT / "heatwaves_files" / "1x1" / "30member_compound_CumulativeHeat.nc"),
        "day": str(DATA_ROOT / "heatwaves_files" / "1x1" / "30member_dayhw_CumulativeHeat.nc"),
        "night": str(DATA_ROOT / "heatwaves_files" / "1x1" / "30member_nighthw_CumulativeHeat.nc"),
    }
    ds = xr.open_dataset(file_map[hw_type]).sel(lat=slice(0, 90))
    ds = filp_lon(ds, lon_name="lon")
    da = mask_landsea(ds["var"].sel(time=slice("1981", "2014")), lat_name="lat", label="ocean")
    _MEMBER_HEATWAVE_CH_CACHE[hw_type] = da
    return da


def ensure_fig1_ranking_cache(force=False):
    cache_version = 2
    if FIG1_RANKING_CACHE.exists() and not force:
        with FIG1_RANKING_CACHE.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and payload.get("_version") == cache_version:
            print(f"Fig1 ranking cache hit: {FIG1_RANKING_CACHE}")
            return payload
        print("Fig1 ranking cache 版本过旧，正在按 1981-2014 重新生成。")
    payload = {"_version": cache_version}
    ds_regional = xr.open_dataset(FILE_REGIONAL_MEANS)
    for prefix, model_da, obs_var_map in [
        ("compound_days", _open_member_heatwave_days("compound"), {"CPC": "compound_days_obs_cpc", "BEST": "compound_days_obs_best", "ERA5": "compound_days_obs_era5", "MERRA2": "compound_days_obs_merra2", "JRA3Q": "compound_days_obs_jra3q"}),
        ("compound_ch", _open_member_heatwave_ch("compound"), {"CPC": "compound_cumulative_heat_obs_cpc", "BEST": "compound_cumulative_heat_obs_best", "ERA5": "compound_cumulative_heat_obs_era5", "MERRA2": "compound_cumulative_heat_obs_merra2", "JRA3Q": "compound_cumulative_heat_obs_jra3q"}),
    ]:
        member_dim = "members" if "members" in model_da.dims else model_da.dims[0]
        member_names = [str(v) for v in model_da[member_dim].values]
        slopes = []
        for member in model_da[member_dim].values:
            data = model_da.sel({member_dim: member}).sel(lat=slice(0, 90))
            a = Area_Mean(data, data.lat, data.lon)
            years = np.arange(1981, 2015).reshape(-1, 1)
            solver = LinearRegression()
            solver.fit(years, np.asarray(a).reshape(-1, 1))
            slopes.append(float(solver.coef_[0][0]))
        entries = [(name, value, "model") for name, value in zip(member_names, slopes)]
        for obs_name, obs_var in obs_var_map.items():
            obs_da = ds_regional[obs_var]
            obs_years = np.asarray(obs_da["year"].values, dtype=int)
            obs_mask = (obs_years >= 1981) & (obs_years <= 2014)
            years = obs_years[obs_mask].astype(float).reshape(-1, 1)
            values = np.asarray(obs_da.values, dtype=float)[obs_mask].reshape(-1, 1)
            solver = LinearRegression()
            solver.fit(years, values)
            entries.append((obs_name, float(solver.coef_[0][0]), "obs"))
        entries.append(("MME", float(np.nanmean(slopes)), "mme"))
        payload[prefix] = entries
    with FIG1_RANKING_CACHE.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Fig1 ranking cache rebuilt: {FIG1_RANKING_CACHE}")
    return payload


def ensure_fig2_boxplot_cache(force=False):
    global _FIG2_BOXPLOT_MEMORY_CACHE
    cache_version = 2
    if _FIG2_BOXPLOT_MEMORY_CACHE is not None and not force:
        print(f"Fig2 boxplot cache hit (memory): {FIG2_BOXPLOT_CACHE}")
        return _FIG2_BOXPLOT_MEMORY_CACHE
    if FIG2_BOXPLOT_CACHE.exists() and not force:
        with np.load(FIG2_BOXPLOT_CACHE, allow_pickle=True) as npz_obj:
            if ("cache_version" in npz_obj.files and int(np.asarray(npz_obj["cache_version"]).ravel()[0]) == cache_version and "region_bounds_signature" in npz_obj.files and validate_region_bounds_signature(np.asarray(npz_obj["region_bounds_signature"]).ravel()[0])):
                print(f"Fig2 boxplot cache hit (disk): {FIG2_BOXPLOT_CACHE}")
                _FIG2_BOXPLOT_MEMORY_CACHE = {key: npz_obj[key] for key in npz_obj.files}
                return _FIG2_BOXPLOT_MEMORY_CACHE
        print("Fig2 boxplot cache 版本过旧，正在重建。")
    print(f"Fig2 boxplot cache miss，开始生成：{FIG2_BOXPLOT_CACHE}")

    def prepare_hw_data(model_data, obs_data_list):
        hot_cmip = calc_trend_annual(model_data, hot_regions, model_data.lat, model_data.lon).transpose("region", "members")
        nothot_cmip = calc_trend_annual(model_data, nothot_regions, model_data.lat, model_data.lon).transpose("region", "members")
        nh_cmip = area_weighted_mean(model_data, model_data.lat, model_data.lon)
        nh_trend_cmip = nh_cmip.polyfit(dim="time", deg=1)["polyfit_coefficients"].sel(degree=1)
        obs_hot_trends = []
        obs_nothot_trends = []
        obs_nh_trends = []
        for obs in obs_data_list:
            obs_hot_trends.append(calc_trend_annual(obs, hot_regions, obs.lat, obs.lon).values)
            obs_nothot_trends.append(calc_trend_annual(obs, nothot_regions, obs.lat, obs.lon).values)
            nh = area_weighted_mean(obs, obs.lat, obs.lon)
            obs_nh_trends.append(nh.polyfit(dim="time", deg=1)["polyfit_coefficients"].sel(degree=1).values)
        cmip_all = np.concatenate([nh_trend_cmip.values.reshape(1, -1), hot_cmip.values, nothot_cmip.values], axis=0)
        obs_all = np.concatenate([np.array(obs_nh_trends).reshape(len(obs_data_list), 1), np.array(obs_hot_trends), np.array(obs_nothot_trends)], axis=1)
        return cmip_all, obs_all

    cmip_all_day, obs_all_day = prepare_hw_data(
        _open_member_heatwave_days("day"),
        [
            _open_obs_heatwave_days("ERA5", "day"),
            _open_obs_heatwave_days("MERRA2", "day"),
            _open_obs_heatwave_days("JRA-3Q", "day"),
            _open_obs_heatwave_days("CPC", "day"),
            _open_obs_heatwave_days("BerkeleyEarth", "day"),
        ],
    )
    cmip_all_night, obs_all_night = prepare_hw_data(
        _open_member_heatwave_days("night"),
        [
            _open_obs_heatwave_days("ERA5", "night"),
            _open_obs_heatwave_days("MERRA2", "night"),
            _open_obs_heatwave_days("JRA-3Q", "night"),
            _open_obs_heatwave_days("CPC", "night"),
            _open_obs_heatwave_days("BerkeleyEarth", "night"),
        ],
    )
    cmip_all_compound, obs_all_compound = prepare_hw_data(
        _open_member_heatwave_days("compound"),
        [
            _open_obs_heatwave_days("ERA5", "compound"),
            _open_obs_heatwave_days("MERRA2", "compound"),
            _open_obs_heatwave_days("JRA-3Q", "compound"),
            _open_obs_heatwave_days("CPC", "compound"),
            _open_obs_heatwave_days("BerkeleyEarth", "compound"),
        ],
    )
    np.savez_compressed(
        FIG2_BOXPLOT_CACHE,
        cache_version=np.array([cache_version], dtype=int),
        region_bounds_signature=np.array([region_bounds_signature()], dtype=object),
        cmip_all_day=cmip_all_day,
        obs_all_day=obs_all_day,
        cmip_all_night=cmip_all_night,
        obs_all_night=obs_all_night,
        cmip_all_compound=cmip_all_compound,
        obs_all_compound=obs_all_compound,
        labels=np.array(["NH"] + list(hotrgion_na) + list(nothotrgion_na), dtype=object),
        obs_names=np.array(["ERA5", "MERRA2", "JRA-3Q", "CPC", "BEST"], dtype=object),
    )
    with np.load(FIG2_BOXPLOT_CACHE, allow_pickle=True) as npz_obj:
        _FIG2_BOXPLOT_MEMORY_CACHE = {key: npz_obj[key] for key in npz_obj.files}
    print(f"Fig2 boxplot cache built: {FIG2_BOXPLOT_CACHE}")
    return _FIG2_BOXPLOT_MEMORY_CACHE


def ensure_fig2_cumheat_boxplot_cache(force=False):
    global _FIG2_CUMHEAT_BOXPLOT_MEMORY_CACHE
    cache_version = 1
    if _FIG2_CUMHEAT_BOXPLOT_MEMORY_CACHE is not None and not force:
        print(f"Fig2 cumulative-heat boxplot cache hit (memory): {FIG2_CUMHEAT_BOXPLOT_CACHE}")
        return _FIG2_CUMHEAT_BOXPLOT_MEMORY_CACHE
    if FIG2_CUMHEAT_BOXPLOT_CACHE.exists() and not force:
        with np.load(FIG2_CUMHEAT_BOXPLOT_CACHE, allow_pickle=True) as npz_obj:
            if ("cache_version" in npz_obj.files and int(np.asarray(npz_obj["cache_version"]).ravel()[0]) == cache_version and "region_bounds_signature" in npz_obj.files and validate_region_bounds_signature(np.asarray(npz_obj["region_bounds_signature"]).ravel()[0])):
                print(f"Fig2 cumulative-heat boxplot cache hit (disk): {FIG2_CUMHEAT_BOXPLOT_CACHE}")
                _FIG2_CUMHEAT_BOXPLOT_MEMORY_CACHE = {key: npz_obj[key] for key in npz_obj.files}
                return _FIG2_CUMHEAT_BOXPLOT_MEMORY_CACHE
        print("Fig2 cumulative-heat boxplot cache ??????????")

    print(f"Fig2 cumulative-heat boxplot cache miss??????{FIG2_CUMHEAT_BOXPLOT_CACHE}")
    print("Fig2 cumulative-heat sources are resolved under data_root/heatwaves_files.")
    print("Fig2 cumulative-heat sources are resolved under data_root/heatwaves_files.")
    print("Fig2 cumulative-heat time range: 1981-2014")

    def prepare_hw_data(model_data, obs_data_list):
        hot_cmip = calc_trend_annual(model_data, hot_regions, model_data.lat, model_data.lon).transpose("region", "members")
        nothot_cmip = calc_trend_annual(model_data, nothot_regions, model_data.lat, model_data.lon).transpose("region", "members")
        nh_cmip = area_weighted_mean(model_data, model_data.lat, model_data.lon)
        nh_trend_cmip = nh_cmip.polyfit(dim="time", deg=1)["polyfit_coefficients"].sel(degree=1)
        obs_hot_trends = []
        obs_nothot_trends = []
        obs_nh_trends = []
        for obs in obs_data_list:
            obs_hot_trends.append(calc_trend_annual(obs, hot_regions, obs.lat, obs.lon).values)
            obs_nothot_trends.append(calc_trend_annual(obs, nothot_regions, obs.lat, obs.lon).values)
            nh = area_weighted_mean(obs, obs.lat, obs.lon)
            obs_nh_trends.append(nh.polyfit(dim="time", deg=1)["polyfit_coefficients"].sel(degree=1).values)
        cmip_all = np.concatenate([nh_trend_cmip.values.reshape(1, -1), hot_cmip.values, nothot_cmip.values], axis=0)
        obs_all = np.concatenate([np.array(obs_nh_trends).reshape(len(obs_data_list), 1), np.array(obs_hot_trends), np.array(obs_nothot_trends)], axis=1)
        return cmip_all, obs_all

    cmip_all_day, obs_all_day = prepare_hw_data(
        _open_member_heatwave_ch("day"),
        [
            _open_obs_heatwave_ch("ERA5", "day"),
            _open_obs_heatwave_ch("MERRA2", "day"),
            _open_obs_heatwave_ch("JRA-3Q", "day"),
            _open_obs_heatwave_ch("CPC", "day"),
            _open_obs_heatwave_ch("BerkeleyEarth", "day"),
        ],
    )
    cmip_all_night, obs_all_night = prepare_hw_data(
        _open_member_heatwave_ch("night"),
        [
            _open_obs_heatwave_ch("ERA5", "night"),
            _open_obs_heatwave_ch("MERRA2", "night"),
            _open_obs_heatwave_ch("JRA-3Q", "night"),
            _open_obs_heatwave_ch("CPC", "night"),
            _open_obs_heatwave_ch("BerkeleyEarth", "night"),
        ],
    )
    cmip_all_compound, obs_all_compound = prepare_hw_data(
        _open_member_heatwave_ch("compound"),
        [
            _open_obs_heatwave_ch("ERA5", "compound"),
            _open_obs_heatwave_ch("MERRA2", "compound"),
            _open_obs_heatwave_ch("JRA-3Q", "compound"),
            _open_obs_heatwave_ch("CPC", "compound"),
            _open_obs_heatwave_ch("BerkeleyEarth", "compound"),
        ],
    )

    np.savez_compressed(
        FIG2_CUMHEAT_BOXPLOT_CACHE,
        cache_version=np.array([cache_version], dtype=int),
        region_bounds_signature=np.array([region_bounds_signature()], dtype=object),
        cmip_all_day=cmip_all_day,
        obs_all_day=obs_all_day,
        cmip_all_night=cmip_all_night,
        obs_all_night=obs_all_night,
        cmip_all_compound=cmip_all_compound,
        obs_all_compound=obs_all_compound,
        labels=np.array(["NH"] + list(hotrgion_na) + list(nothotrgion_na), dtype=object),
        obs_names=np.array(["ERA5", "MERRA2", "JRA-3Q", "CPC", "BEST"], dtype=object),
    )
    with np.load(FIG2_CUMHEAT_BOXPLOT_CACHE, allow_pickle=True) as npz_obj:
        _FIG2_CUMHEAT_BOXPLOT_MEMORY_CACHE = {key: npz_obj[key] for key in npz_obj.files}
    print(f"Fig2 cumulative-heat boxplot cache built: {FIG2_CUMHEAT_BOXPLOT_CACHE}")
    return _FIG2_CUMHEAT_BOXPLOT_MEMORY_CACHE


def prepare_fig2_data(force_recompute=False):
    global _FIG2_PREPARED_MEMORY_CACHE
    cache_version = 1
    if _FIG2_PREPARED_MEMORY_CACHE is not None and not force_recompute:
        print(f"Fig2 prepared cache hit (memory): {FIG2_PREPARED_CACHE}")
        return _FIG2_PREPARED_MEMORY_CACHE
    if FIG2_PREPARED_CACHE.exists() and not force_recompute:
        with FIG2_PREPARED_CACHE.open("rb") as f:
            payload = pickle.load(f)
        if (
            isinstance(payload, dict)
            and payload.get("_version") == cache_version
            and payload.get("baseline_mode") == FIG2_EXTREME_BASELINE_MODE
            and validate_region_bounds_signature(payload.get("_region_bounds_signature"))
        ):
            print(f"Fig2 prepared cache hit (disk): {FIG2_PREPARED_CACHE}")
            _FIG2_PREPARED_MEMORY_CACHE = payload
            return payload
        print("Fig2 prepared cache 版本过旧，正在重建。")

    print(f"Fig2 prepared cache miss，开始生成：{FIG2_PREPARED_CACHE}")
    t0 = time.perf_counter()
    ds_ratio = xr.open_dataset(FILE_HEATWAVE_RATIO)
    try:
        ratio_specs = [
            ("Compound HW Days", "compound_days"),
            ("TMAX", "tmax_clim"),
            ("TMAX Extreme", "extreme1_tmax_percent"),
            ("Compound CH", "compound_ch"),
            ("TMIN", "tmin_clim"),
            ("TMIN Extreme", "extreme1_tmin_percent"),
        ]
        map_payload = []
        for title, var_name in ratio_specs:
            field = ds_ratio[var_name]
            data_c, lons_c = add_cyclic_point(
                np.asarray(field.values, dtype=float),
                coord=np.asarray(field.lon.values, dtype=float),
            )
            map_payload.append(
                {
                    "title": title,
                    "lat": np.asarray(field.lat.values, dtype=float),
                    "lon_cyclic": np.asarray(lons_c, dtype=float),
                    "data_cyclic": np.asarray(data_c, dtype=float),
                }
            )
    finally:
        ds_ratio.close()

    box_payload = copy.deepcopy(ensure_fig2_boxplot_cache(force=force_recompute))
    payload = {
        "_version": cache_version,
        "baseline_mode": FIG2_EXTREME_BASELINE_MODE,
        "_region_bounds_signature": region_bounds_signature(),
        "map_payload": map_payload,
        "box_payload": box_payload,
        "labels": list(np.asarray(box_payload["labels"], dtype=object)),
        "obs_names": list(np.asarray(box_payload["obs_names"], dtype=object)),
    }
    with FIG2_PREPARED_CACHE.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    _FIG2_PREPARED_MEMORY_CACHE = payload
    print(f"Fig2 prepared cache built: {FIG2_PREPARED_CACHE} ({time.perf_counter() - t0:.2f} s)")
    return payload

def ensure_three_hw_trend_cache(kind="era5", force=False):
    target = ERA5_TREND_CACHE if kind == "era5" else MME_TREND_CACHE
    memory_key = (kind, force)
    if target.exists() and not force:
        if memory_key not in _THREE_HW_TREND_CACHE_MEMORY:
            _THREE_HW_TREND_CACHE_MEMORY[memory_key] = xr.open_dataset(target)
        return _THREE_HW_TREND_CACHE_MEMORY[memory_key]
    if kind == "era5":
        uu = [
            _open_obs_heatwave_days("ERA5", "day"),
            _open_obs_heatwave_days("ERA5", "night"),
            _open_obs_heatwave_days("ERA5", "compound"),
            _open_obs_heatwave_ch("ERA5", "day"),
            _open_obs_heatwave_ch("ERA5", "night"),
            _open_obs_heatwave_ch("ERA5", "compound"),
        ]
    else:
        uu = [
            _open_member_heatwave_days("day").mean(dim="members"),
            _open_member_heatwave_days("night").mean(dim="members"),
            _open_member_heatwave_days("compound").mean(dim="members"),
            _open_member_heatwave_ch("day").mean(dim="members"),
            _open_member_heatwave_ch("night").mean(dim="members"),
            _open_member_heatwave_ch("compound").mean(dim="members"),
        ]
    names = ["day_hw_days", "night_hw_days", "compound_hw_days", "day_hw_ch", "night_hw_ch", "compound_hw_ch"]
    ds_out = xr.Dataset()
    years = np.arange(1981, 2015)
    for name, da in zip(names, uu):
        varrr = da.sel(time=slice("1981", "2014"))
        trend = np.zeros((varrr.lat.size, varrr.lon.size), dtype=float)
        pval = np.zeros((varrr.lat.size, varrr.lon.size), dtype=float)
        ann = np.asarray(varrr)
        for m in range(varrr.lat.size):
            for n in range(varrr.lon.size):
                slope, _intercept, _r, p_tmp, _stderr = stats.linregress(years, ann[:, m, n])
                trend[m, n] = slope
                pval[m, n] = p_tmp
        ds_out[f"{name}_trend"] = xr.DataArray(trend, coords={"lat": varrr.lat, "lon": varrr.lon}, dims=("lat", "lon"))
        ds_out[f"{name}_p"] = xr.DataArray(pval, coords={"lat": varrr.lat, "lon": varrr.lon}, dims=("lat", "lon"))
    ds_out.to_netcdf(target)
    _THREE_HW_TREND_CACHE_MEMORY[memory_key] = ds_out
    return ds_out


def corr_records_from_dataset(ds):
    records = {}
    for var_name in ds["variable"].values:
        var_name = str(var_name)
        records[var_name] = {
            "r": np.asarray(ds["r"].sel(variable=var_name).values, dtype=float),
            "p": np.asarray(ds["p"].sel(variable=var_name).values, dtype=float),
        }
    return records
