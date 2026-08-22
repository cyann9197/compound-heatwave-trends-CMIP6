from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chw_cmip6.figure_cli import prepare_figure

ARGS, CONFIG = prepare_figure('fig5', 'Reproduce manuscript Figure 5.')
from chw_cmip6.figure_context import *  # noqa: E402,F403

import os
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

MONTHLY_DA_BASE_DIR = str(CACHE_ROOT / "dynamic_adjustment" / "monthly")
MONTHLY_DA_OUT_DIR = os.path.join(MONTHLY_DA_BASE_DIR, "outputs")
MONTHLY_DA_FIG_DIR = str(STRICT_FINAL_OUTPUT_DIR)
MONTHLY_DA_ERA5_REGION_CSV = os.path.join(MONTHLY_DA_OUT_DIR, "da_ridge_region_trends_ERA5_1981_2014_JJA.csv")
MONTHLY_DA_MEMBER_REGION_CSV = os.path.join(MONTHLY_DA_OUT_DIR, "da_ridge_region_trends_CMIP_members_JJA_1981_2014.csv")
MONTHLY_DA_MME_REGION_CSV = os.path.join(MONTHLY_DA_OUT_DIR, "da_ridge_region_trends_MME_JJA_1981_2014.csv")
MONTHLY_DA_ERA5_SPATIAL_NC = os.path.join(MONTHLY_DA_OUT_DIR, "da_ridge_spatial_trends_ERA5_JJA_1981_2014.nc")
MONTHLY_DA_MEMBER_SPATIAL_NC = os.path.join(MONTHLY_DA_OUT_DIR, "da_ridge_spatial_trends_CMIP_members_JJA_1981_2014.nc")
MONTHLY_DA_MME_SPATIAL_NC = os.path.join(MONTHLY_DA_OUT_DIR, "da_ridge_spatial_trends_MME_JJA_1981_2014.nc")
MONTHLY_DA_FORCE_RECOMPUTE = False
MONTHLY_DA_SAVE_PLOTS = True

MONTHLY_DA_BASE_DIR = os.environ.get("MONTHLY_DA_BASE_DIR", str(CACHE_ROOT / "dynamic_adjustment" / "monthly"))
MONTHLY_DA_OUT_DIR = os.path.join(MONTHLY_DA_BASE_DIR, "outputs")
MONTHLY_DA_FIG_DIR = os.path.join(MONTHLY_DA_BASE_DIR, "figs")
os.makedirs(MONTHLY_DA_OUT_DIR, exist_ok=True)
os.makedirs(MONTHLY_DA_FIG_DIR, exist_ok=True)

MONTHLY_DA_ERA5_REGION_CSV = os.path.join(MONTHLY_DA_OUT_DIR, "da_ridge_region_trends_ERA5_1981_2014_JJA.csv")
MONTHLY_DA_MEMBER_REGION_CSV = os.path.join(MONTHLY_DA_OUT_DIR, "da_ridge_region_trends_CMIP_members_JJA_1981_2014.csv")
MONTHLY_DA_MME_REGION_CSV = os.path.join(MONTHLY_DA_OUT_DIR, "da_ridge_region_trends_MME_JJA_1981_2014.csv")
MONTHLY_DA_ERA5_SPATIAL_NC = os.path.join(MONTHLY_DA_OUT_DIR, "da_ridge_spatial_trends_ERA5_JJA_1981_2014.nc")
MONTHLY_DA_MEMBER_SPATIAL_NC = os.path.join(MONTHLY_DA_OUT_DIR, "da_ridge_spatial_trends_CMIP_members_JJA_1981_2014.nc")
MONTHLY_DA_MME_SPATIAL_NC = os.path.join(MONTHLY_DA_OUT_DIR, "da_ridge_spatial_trends_MME_JJA_1981_2014.nc")
MONTHLY_DA_FINAL_FIG_DIR = str(OUTPUT_ROOT)
MONTHLY_DA_PAYLOAD_CACHE_DIR = os.path.join(MONTHLY_DA_FINAL_FIG_DIR, "cache")
os.makedirs(MONTHLY_DA_PAYLOAD_CACHE_DIR, exist_ok=True)
MONTHLY_DA_PAYLOAD_CACHE_PKL = os.path.join(MONTHLY_DA_PAYLOAD_CACHE_DIR, "monthly_da_plot_payload_v1.pkl")

MONTHLY_DA_FORCE_RECOMPUTE = False
MONTHLY_DA_SAVE_PLOTS = False

MONTHLY_DA_HOT_REGIONS = [
    (-115, -97, 44, 59),
    (-14, 3, 12, 28),
    (54, 71, 56, 65),
    (69, 104, 27, 35),
    (100, 121, 54, 65),
]
MONTHLY_DA_NOTHOT_REGIONS = [
    (-106, -87, 30, 38),
    GIC_BOUNDS,
    (24, 38, 47, 66),
    (100, 119, 38, 52),
]
MONTHLY_DA_HOT_NAMES = ["NNA", "NAF", "EEU", "SAS", "ESB"]
MONTHLY_DA_NOTHOT_NAMES = ["SNA", "GIC", "WEU", "ENA"]


def monthly_da_region_settings():
    return list(zip(MONTHLY_DA_HOT_NAMES, MONTHLY_DA_HOT_REGIONS)) + list(zip(MONTHLY_DA_NOTHOT_NAMES, MONTHLY_DA_NOTHOT_REGIONS))


def monthly_da_infer_member_dim(da):
    for dim in ("member", "members", "model", "models"):
        if dim in da.dims:
            return dim
    for dim in da.dims:
        if dim not in ("time", "lat", "lon", "plev"):
            return dim
    raise ValueError("无法识别成员维度：{}".format(da.dims))


def monthly_da_slice_region(da, box):
    lon_min, lon_max, lat_min, lat_max = box
    return da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))


def monthly_da_trend_map_jja(monthly_da):
    annual = monthly_da.groupby("time.year").mean("time", skipna=True)
    fit = annual.polyfit(dim="year", deg=1, skipna=True)
    return fit["polyfit_coefficients"].sel(degree=1)


def monthly_da_area_weighted_mean(da, lon_dim="lon", lat_dim="lat"):
    weights = np.cos(np.deg2rad(da[lat_dim]))
    weights = xr.where(np.isfinite(weights), weights, np.nan)
    weighted = da.weighted(weights)
    return weighted.mean(dim=(lat_dim, lon_dim), skipna=True)


def monthly_da_calc_annual_trend(ts, method="mean"):
    if "time" not in ts.dims:
        raise ValueError("输入时间序列缺少 time 维度。")
    annual = ts.groupby("time.year").mean("time", skipna=True) if method == "mean" else ts.groupby("time.year").sum("time", skipna=True)
    fit = annual.polyfit(dim="year", deg=1, skipna=True)
    slope = fit["polyfit_coefficients"].sel(degree=1)
    pvalue = xr.full_like(slope, np.nan, dtype=float)
    try:
        years = annual["year"].values.astype(float)
        values = annual.values.astype(float)
        valid = np.isfinite(years) & np.isfinite(values)
        if valid.sum() >= 3:
            from scipy import stats
            pvalue[...] = stats.linregress(years[valid], values[valid]).pvalue
    except Exception:
        pass
    return xr.Dataset({"slope": slope, "pvalue": pvalue})


def monthly_da_make_map(ax, box):
    ax.set_xticks(np.arange(box[0], box[1], 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(box[2], box[3] + 30, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis="both", which="major", labelsize=16, direction="out", length=5, width=1, pad=5)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis="both", which="minor", direction="out", length=2, width=0.4)
    ax.spines["geo"].set_linewidth(1)
    ax.set_extent([box[0], box[1], box[2], box[3]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", zorder=2)
    return ax


def monthly_da_draw_rectangles(ax, boxes, edgecolor="green", linewidth=2.0):
    for lon_min, lon_max, lat_min, lat_max in boxes:
        rect = mpatches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor="none",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        ax.add_patch(rect)


def monthly_da_missing_globals(names):
    return [name for name in names if name not in globals()]


def monthly_da_import_ridge_functions():
    if "subtract_reference_mean" in globals() and "dynamical_adjustment_ridge" in globals():
        return globals()["subtract_reference_mean"], globals()["dynamical_adjustment_ridge"]
    sys.path.insert(0, os.getcwd())
    from da_ridge import subtract_reference_mean, dynamical_adjustment_ridge
    return subtract_reference_mean, dynamical_adjustment_ridge


def monthly_da_compute_era5_products():
    required = ["mera5_tmax", "mera5_tmin", "era5_500zg_2014"]
    missing = monthly_da_missing_globals(required)
    if missing:
        raise NameError(
            "月尺度 ERA5 动力调整缺少上游变量：{}。如只需重绘，请保持 MONTHLY_DA_FORCE_RECOMPUTE=False；"
            "如需重算，请先运行月值温度和 Z500 的读取部分。".format(", ".join(missing))
        )

    subtract_reference_mean, dynamical_adjustment_ridge = monthly_da_import_ridge_functions()

    TMAX = globals()["mera5_tmax"].sel(time=slice("1981-06-01", "2014-08-31"))
    TMIN = globals()["mera5_tmin"].sel(time=slice("1981-06-01", "2014-08-31"))
    Z500 = globals()["era5_500zg_2014"].sel(time=slice("1981-06-01", "2014-08-31"))
    Z500_REL = subtract_reference_mean(Z500, reference="nh")

    template = xr.full_like(TMAX.isel(time=0, drop=True), np.nan)
    rows = []
    spatial_out = {}

    for var_name, target in [("tmax", TMAX), ("tmin", TMIN)]:
        total_list, dyn_list, thermo_list = [], [], []
        thermo_direct_list, thermo_diff_list = [], []

        for region_name, box in monthly_da_region_settings():
            y_region = monthly_da_slice_region(target, box)
            ds_da = dynamical_adjustment_ridge(
                z500_full=Z500_REL,
                y_region=y_region,
                subtract_mean_reference=None,
                predictor_expand_deg=15.0,
                months=(6, 7, 8),
                standardize_X="zscore",
                cv_year_folds=5,
                max_targets_for_cv=800,
                random_state=0,
            )

            slope_total_reg = monthly_da_trend_map_jja(ds_da["y_sel"])
            slope_dyn_reg = monthly_da_trend_map_jja(ds_da["y_dyn_anom"])
            slope_thermo_reg = slope_total_reg - slope_dyn_reg
            slope_thermo_direct_reg = monthly_da_trend_map_jja(ds_da["y_adj"])
            slope_thermo_diff_reg = slope_thermo_reg - slope_thermo_direct_reg

            def _place(reg_da):
                full = template.copy(deep=True)
                full.loc[dict(lat=reg_da.lat, lon=reg_da.lon)] = reg_da
                return full

            total_list.append(_place(slope_total_reg).astype("float32").assign_coords(region=region_name).expand_dims("region"))
            dyn_list.append(_place(slope_dyn_reg).astype("float32").assign_coords(region=region_name).expand_dims("region"))
            thermo_list.append(_place(slope_thermo_reg).astype("float32").assign_coords(region=region_name).expand_dims("region"))
            thermo_direct_list.append(_place(slope_thermo_direct_reg).astype("float32").assign_coords(region=region_name).expand_dims("region"))
            thermo_diff_list.append(_place(slope_thermo_diff_reg).astype("float32").assign_coords(region=region_name).expand_dims("region"))

            slope_total_map = float(monthly_da_area_weighted_mean(slope_total_reg, lon_dim="lon", lat_dim="lat").values)
            slope_dyn_map = float(monthly_da_area_weighted_mean(slope_dyn_reg, lon_dim="lon", lat_dim="lat").values)
            slope_thermo_map = float(monthly_da_area_weighted_mean(slope_thermo_reg, lon_dim="lon", lat_dim="lat").values)

            y_total_ts = monthly_da_area_weighted_mean(ds_da["y_sel"], lon_dim="lon", lat_dim="lat")
            y_dyn_ts = monthly_da_area_weighted_mean(ds_da["y_dyn_anom"], lon_dim="lon", lat_dim="lat")
            y_thermo_ts = y_total_ts - y_dyn_ts

            tr_total = monthly_da_calc_annual_trend(y_total_ts, method="mean")
            tr_dyn = monthly_da_calc_annual_trend(y_dyn_ts, method="mean")
            tr_thermo = monthly_da_calc_annual_trend(y_thermo_ts, method="mean")

            slope_total_ts = float(tr_total["slope"].values)
            p_total_ts = float(tr_total["pvalue"].values)
            slope_dyn_ts = float(tr_dyn["slope"].values)
            p_dyn_ts = float(tr_dyn["pvalue"].values)
            slope_thermo_ts = float(tr_thermo["slope"].values)
            p_thermo_ts = float(tr_thermo["pvalue"].values)

            thermo_diff_abs_max = float(np.nanmax(np.abs(slope_thermo_diff_reg.values)))
            lon_min, lon_max, lat_min, lat_max = box
            rows.append({
                "var": var_name,
                "region": region_name,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "alpha": float(ds_da.attrs.get("selected_alpha", np.nan)),
                "total_slope_per_year": slope_total_map,
                "total_slope_per_decade": slope_total_map * 10.0,
                "dyn_slope_per_year": slope_dyn_map,
                "dyn_slope_per_decade": slope_dyn_map * 10.0,
                "thermo_slope_per_year": slope_thermo_map,
                "thermo_slope_per_decade": slope_thermo_map * 10.0,
                "total_slope_ts_per_year": slope_total_ts,
                "dyn_slope_ts_per_year": slope_dyn_ts,
                "thermo_slope_ts_per_year": slope_thermo_ts,
                "total_pvalue_ts": p_total_ts,
                "dyn_pvalue_ts": p_dyn_ts,
                "thermo_pvalue_ts": p_thermo_ts,
                "total_delta_map_minus_ts_per_year": slope_total_map - slope_total_ts,
                "dyn_delta_map_minus_ts_per_year": slope_dyn_map - slope_dyn_ts,
                "thermo_delta_map_minus_ts_per_year": slope_thermo_map - slope_thermo_ts,
                "thermo_diff_abs_max_per_year": thermo_diff_abs_max,
                "predictor_bounds_lat_min": float(ds_da.attrs.get("predictor_bounds_lat_min", np.nan)),
                "predictor_bounds_lat_max": float(ds_da.attrs.get("predictor_bounds_lat_max", np.nan)),
                "predictor_bounds_lon_min": float(ds_da.attrs.get("predictor_bounds_lon_min", np.nan)),
                "predictor_bounds_lon_max": float(ds_da.attrs.get("predictor_bounds_lon_max", np.nan)),
                "standardize_X": ds_da.attrs.get("standardize_X", ""),
            })

        spatial_out["{}_total_slope_per_year".format(var_name)] = xr.concat(total_list, dim="region")
        spatial_out["{}_dyn_slope_per_year".format(var_name)] = xr.concat(dyn_list, dim="region")
        spatial_out["{}_thermo_slope_per_year".format(var_name)] = xr.concat(thermo_list, dim="region")
        spatial_out["{}_thermo_direct_slope_per_year".format(var_name)] = xr.concat(thermo_direct_list, dim="region")
        spatial_out["{}_thermo_diff_slope_per_year".format(var_name)] = xr.concat(thermo_diff_list, dim="region")

    era_region_df = pd.DataFrame(rows)
    era_region_df.to_csv(MONTHLY_DA_ERA5_REGION_CSV, index=False)
    era_spatial_ds = xr.Dataset(spatial_out)
    era_spatial_ds.to_netcdf(
        MONTHLY_DA_ERA5_SPATIAL_NC,
        encoding={name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in era_spatial_ds.data_vars},
    )
    print("[MONTHLY_DA] Saved ERA5 region csv:", MONTHLY_DA_ERA5_REGION_CSV)
    print("[MONTHLY_DA] Saved ERA5 spatial nc:", MONTHLY_DA_ERA5_SPATIAL_NC)
    return era_region_df, era_spatial_ds


def monthly_da_compute_cmip_products():
    required = ["mall_cimp_tmax", "mall_cimp_tmin", "all_cimp_zg"]
    missing = monthly_da_missing_globals(required)
    if missing:
        raise NameError(
            "月尺度 CMIP6 动力调整缺少上游变量：{}。如只需重绘，请保持 MONTHLY_DA_FORCE_RECOMPUTE=False；"
            "如需重算，请先运行月值模式温度和 Z500 读取部分。".format(", ".join(missing))
        )

    subtract_reference_mean, dynamical_adjustment_ridge = monthly_da_import_ridge_functions()

    tmax_m = globals()["mall_cimp_tmax"].sel(time=slice("1981-06-01", "2014-08-31"))
    tmin_m = globals()["mall_cimp_tmin"].sel(time=slice("1981-06-01", "2014-08-31"))
    z500_m = globals()["all_cimp_zg"].sel(plev=50000, time=slice("1981-06-01", "2014-08-31"))

    member_dim = monthly_da_infer_member_dim(tmax_m)
    all_members = [str(value) for value in tmax_m[member_dim].values]
    if "model_list" in globals():
        members = [member for member in globals()["model_list"] if member in all_members]
    else:
        members = all_members

    template = xr.full_like(tmax_m.isel({member_dim: 0, "time": 0}, drop=True), np.nan)
    rows = []
    member_maps = {
        "tmax": {"total": [], "dyn": [], "thermo": [], "thermo_direct": [], "thermo_diff": []},
        "tmin": {"total": [], "dyn": [], "thermo": [], "thermo_direct": [], "thermo_diff": []},
    }

    for member in members:
        z500_member = z500_m.sel({member_dim: member})
        z500_rel_member = subtract_reference_mean(z500_member, reference="nh")

        for var_name, target in [("tmax", tmax_m), ("tmin", tmin_m)]:
            t_member = target.sel({member_dim: member})
            reg_total, reg_dyn, reg_thermo, reg_thermo_direct, reg_thermo_diff = [], [], [], [], []

            for region_name, box in monthly_da_region_settings():
                y_region = monthly_da_slice_region(t_member, box)
                ds_da = dynamical_adjustment_ridge(
                    z500_full=z500_rel_member,
                    y_region=y_region,
                    subtract_mean_reference=None,
                    predictor_expand_deg=15.0,
                    months=(6, 7, 8),
                    standardize_X="zscore",
                    cv_year_folds=5,
                    max_targets_for_cv=400,
                    random_state=0,
                )

                slope_total_reg = monthly_da_trend_map_jja(ds_da["y_sel"])
                slope_dyn_reg = monthly_da_trend_map_jja(ds_da["y_dyn_anom"])
                slope_thermo_reg = slope_total_reg - slope_dyn_reg
                slope_thermo_direct_reg = monthly_da_trend_map_jja(ds_da["y_adj"])
                slope_thermo_diff_reg = slope_thermo_reg - slope_thermo_direct_reg

                def _place(reg_da):
                    full = template.copy(deep=True)
                    full.loc[dict(lat=reg_da.lat, lon=reg_da.lon)] = reg_da
                    return full

                reg_total.append(_place(slope_total_reg).astype("float32").assign_coords(region=region_name).expand_dims("region"))
                reg_dyn.append(_place(slope_dyn_reg).astype("float32").assign_coords(region=region_name).expand_dims("region"))
                reg_thermo.append(_place(slope_thermo_reg).astype("float32").assign_coords(region=region_name).expand_dims("region"))
                reg_thermo_direct.append(_place(slope_thermo_direct_reg).astype("float32").assign_coords(region=region_name).expand_dims("region"))
                reg_thermo_diff.append(_place(slope_thermo_diff_reg).astype("float32").assign_coords(region=region_name).expand_dims("region"))

                slope_total_map = float(monthly_da_area_weighted_mean(slope_total_reg, lon_dim="lon", lat_dim="lat").values)
                slope_dyn_map = float(monthly_da_area_weighted_mean(slope_dyn_reg, lon_dim="lon", lat_dim="lat").values)
                slope_thermo_map = float(monthly_da_area_weighted_mean(slope_thermo_reg, lon_dim="lon", lat_dim="lat").values)

                y_total_ts = monthly_da_area_weighted_mean(ds_da["y_sel"], lon_dim="lon", lat_dim="lat")
                y_dyn_ts = monthly_da_area_weighted_mean(ds_da["y_dyn_anom"], lon_dim="lon", lat_dim="lat")
                y_thermo_ts = y_total_ts - y_dyn_ts

                tr_total = monthly_da_calc_annual_trend(y_total_ts, method="mean")
                tr_dyn = monthly_da_calc_annual_trend(y_dyn_ts, method="mean")
                tr_thermo = monthly_da_calc_annual_trend(y_thermo_ts, method="mean")

                slope_total_ts = float(tr_total["slope"].values)
                p_total_ts = float(tr_total["pvalue"].values)
                slope_dyn_ts = float(tr_dyn["slope"].values)
                p_dyn_ts = float(tr_dyn["pvalue"].values)
                slope_thermo_ts = float(tr_thermo["slope"].values)
                p_thermo_ts = float(tr_thermo["pvalue"].values)

                thermo_diff_abs_max = float(np.nanmax(np.abs(slope_thermo_diff_reg.values)))
                lon_min, lon_max, lat_min, lat_max = box
                rows.append({
                    "var": var_name,
                    "region": region_name,
                    "member": str(member),
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                    "alpha": float(ds_da.attrs.get("selected_alpha", np.nan)),
                    "total_slope_per_year": slope_total_map,
                    "total_slope_per_decade": slope_total_map * 10.0,
                    "dyn_slope_per_year": slope_dyn_map,
                    "dyn_slope_per_decade": slope_dyn_map * 10.0,
                    "thermo_slope_per_year": slope_thermo_map,
                    "thermo_slope_per_decade": slope_thermo_map * 10.0,
                    "total_slope_ts_per_year": slope_total_ts,
                    "dyn_slope_ts_per_year": slope_dyn_ts,
                    "thermo_slope_ts_per_year": slope_thermo_ts,
                    "total_pvalue_ts": p_total_ts,
                    "dyn_pvalue_ts": p_dyn_ts,
                    "thermo_pvalue_ts": p_thermo_ts,
                    "total_delta_map_minus_ts_per_year": slope_total_map - slope_total_ts,
                    "dyn_delta_map_minus_ts_per_year": slope_dyn_map - slope_dyn_ts,
                    "thermo_delta_map_minus_ts_per_year": slope_thermo_map - slope_thermo_ts,
                    "thermo_diff_abs_max_per_year": thermo_diff_abs_max,
                })

            member_maps[var_name]["total"].append(xr.concat(reg_total, dim="region").expand_dims(member=[str(member)]))
            member_maps[var_name]["dyn"].append(xr.concat(reg_dyn, dim="region").expand_dims(member=[str(member)]))
            member_maps[var_name]["thermo"].append(xr.concat(reg_thermo, dim="region").expand_dims(member=[str(member)]))
            member_maps[var_name]["thermo_direct"].append(xr.concat(reg_thermo_direct, dim="region").expand_dims(member=[str(member)]))
            member_maps[var_name]["thermo_diff"].append(xr.concat(reg_thermo_diff, dim="region").expand_dims(member=[str(member)]))

    member_region_df = pd.DataFrame(rows)
    member_region_df.to_csv(MONTHLY_DA_MEMBER_REGION_CSV, index=False)

    spatial_vars = {}
    for var_name in ["tmax", "tmin"]:
        spatial_vars["{}_total_slope_per_year".format(var_name)] = xr.concat(member_maps[var_name]["total"], dim="member")
        spatial_vars["{}_dyn_slope_per_year".format(var_name)] = xr.concat(member_maps[var_name]["dyn"], dim="member")
        spatial_vars["{}_thermo_slope_per_year".format(var_name)] = xr.concat(member_maps[var_name]["thermo"], dim="member")
        spatial_vars["{}_thermo_direct_slope_per_year".format(var_name)] = xr.concat(member_maps[var_name]["thermo_direct"], dim="member")
        spatial_vars["{}_thermo_diff_slope_per_year".format(var_name)] = xr.concat(member_maps[var_name]["thermo_diff"], dim="member")

    member_spatial_ds = xr.Dataset(spatial_vars)
    member_spatial_ds.to_netcdf(
        MONTHLY_DA_MEMBER_SPATIAL_NC,
        encoding={name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in member_spatial_ds.data_vars},
    )

    mme_spatial_ds = member_spatial_ds.mean("member", skipna=True)
    mme_spatial_ds.to_netcdf(
        MONTHLY_DA_MME_SPATIAL_NC,
        encoding={name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in mme_spatial_ds.data_vars},
    )

    mme_region_df = member_region_df.groupby(["var", "region"], as_index=False).mean(numeric_only=True)
    mme_region_df.to_csv(MONTHLY_DA_MME_REGION_CSV, index=False)

    print("[MONTHLY_DA] Saved CMIP member region csv:", MONTHLY_DA_MEMBER_REGION_CSV)
    print("[MONTHLY_DA] Saved CMIP member spatial nc:", MONTHLY_DA_MEMBER_SPATIAL_NC)
    print("[MONTHLY_DA] Saved MME region csv:", MONTHLY_DA_MME_REGION_CSV)
    print("[MONTHLY_DA] Saved MME spatial nc:", MONTHLY_DA_MME_SPATIAL_NC)
    return member_region_df, member_spatial_ds, mme_region_df, mme_spatial_ds


def monthly_da_load_or_compute(force=False):
    if os.path.exists(MONTHLY_DA_PAYLOAD_CACHE_PKL) and not force:
        print("[MONTHLY_DA] 检查已有 payload 缓存：", MONTHLY_DA_PAYLOAD_CACHE_PKL)
        with open(MONTHLY_DA_PAYLOAD_CACHE_PKL, "rb") as f:
            payload = pickle.load(f)
        region_tables = [payload.get(key) for key in ("era_region", "member_region", "mme_region")]
        if all(validate_gic_dataframe(frame) for frame in region_tables):
            return payload
        print("[MONTHLY_DA] payload 中的 GIC 边界不是 71–83°N，拒绝复用并转入重算。")
        force = True

    cache_files = [
        MONTHLY_DA_ERA5_REGION_CSV,
        MONTHLY_DA_MEMBER_REGION_CSV,
        MONTHLY_DA_MME_REGION_CSV,
        MONTHLY_DA_ERA5_SPATIAL_NC,
        MONTHLY_DA_MME_SPATIAL_NC,
    ]
    missing_cache_files = [path for path in cache_files if not os.path.exists(path)]

    if not missing_cache_files and not force:
        print("[MONTHLY_DA] ????????????????")
        print("[MONTHLY_DA] ERA5 region csv ->", MONTHLY_DA_ERA5_REGION_CSV)
        print("[MONTHLY_DA] CMIP member csv ->", MONTHLY_DA_MEMBER_REGION_CSV)
        print("[MONTHLY_DA] MME region csv ->", MONTHLY_DA_MME_REGION_CSV)
        print("[MONTHLY_DA] ERA5 spatial nc ->", MONTHLY_DA_ERA5_SPATIAL_NC)
        print("[MONTHLY_DA] MME spatial nc ->", MONTHLY_DA_MME_SPATIAL_NC)
        payload = {
            "era_region": pd.read_csv(MONTHLY_DA_ERA5_REGION_CSV),
            "member_region": pd.read_csv(MONTHLY_DA_MEMBER_REGION_CSV),
            "mme_region": pd.read_csv(MONTHLY_DA_MME_REGION_CSV),
            "era_spatial": xr.open_dataset(MONTHLY_DA_ERA5_SPATIAL_NC).load(),
            "mme_spatial": xr.open_dataset(MONTHLY_DA_MME_SPATIAL_NC).load(),
        }
        with open(MONTHLY_DA_PAYLOAD_CACHE_PKL, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("[MONTHLY_DA] Saved payload cache:", MONTHLY_DA_PAYLOAD_CACHE_PKL)
        return payload

    if not force:
        raise FileNotFoundError(
            "??????????{}???????????????????????????? MONTHLY_DA_FORCE_RECOMPUTE=True ??????????? Z500 ?????".format(
                ", ".join(missing_cache_files)
            )
        )

    print("[MONTHLY_DA] MONTHLY_DA_FORCE_RECOMPUTE=True?????????????")
    era_region_df, era_spatial_ds = monthly_da_compute_era5_products()
    member_region_df, member_spatial_ds, mme_region_df, mme_spatial_ds = monthly_da_compute_cmip_products()
    payload = {
        "era_region": era_region_df,
        "member_region": member_region_df,
        "mme_region": mme_region_df,
        "era_spatial": era_spatial_ds,
        "member_spatial": member_spatial_ds,
        "mme_spatial": mme_spatial_ds,
    }
    with open(MONTHLY_DA_PAYLOAD_CACHE_PKL, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("[MONTHLY_DA] Saved payload cache:", MONTHLY_DA_PAYLOAD_CACHE_PKL)
    return payload


if "monthly_da_era_region" not in globals():
    MONTHLY_DA_RESULTS = monthly_da_load_or_compute(force=False)
    monthly_da_era_region = MONTHLY_DA_RESULTS["era_region"]
    monthly_da_member_region = MONTHLY_DA_RESULTS["member_region"]
    monthly_da_mme_region = MONTHLY_DA_RESULTS["mme_region"]
    monthly_da_era_spatial = MONTHLY_DA_RESULTS["era_spatial"]
    monthly_da_mme_spatial = MONTHLY_DA_RESULTS["mme_spatial"]

print("[MONTHLY_DA] 绘图阶段读取 monthly 文件：")
print("  -", MONTHLY_DA_ERA5_REGION_CSV)
print("  -", MONTHLY_DA_MEMBER_REGION_CSV)
print("  -", MONTHLY_DA_MME_REGION_CSV)
print("  -", MONTHLY_DA_ERA5_SPATIAL_NC)
print("  -", MONTHLY_DA_MME_SPATIAL_NC)


# 原始月尺度代码里有中间 cell 会把保存开关改回 False，这里强制恢复为最终导出模式。
MONTHLY_DA_SAVE_PLOTS = True


region_order = MONTHLY_DA_HOT_NAMES + MONTHLY_DA_NOTHOT_NAMES
hot_count = len(MONTHLY_DA_HOT_NAMES)
boundary_x = hot_count - 0.5
x = np.arange(len(region_order))

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

plt.rcParams.update(FINAL_DA_RCPARAMS)

FIG567_TITLE_SIZE = FIG567_PANEL_TITLE_PT
FIG567_LABEL_SIZE = FIG567_AXIS_LABEL_PT
FIG567_TICK_SIZE = FIG567_AXIS_TICK_PT
FIG567_LEGEND_SIZE = FIG567_LEGEND_PT
FIG567_CBAR_SIZE = FIG567_CBAR_LABEL_PT
FIG567_ANNOTATION_SIZE = FIG567_ANNOTATION_PT + 2
FIG567_PANEL_LABEL_SIZE = FIG567_PANEL_LABEL_PT


def _fmt_y(xval, pos):
    if abs(xval) < 5e-4:
        xval = 0.0
    text = "{:.2f}".format(xval).rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


AXIS_CONFIG = {
    ("tmax", "dyn"): {"ylim": (-0.04, 0.12), "yticks": np.arange(-0.04, 0.101, 0.02)},
    ("tmax", "thermo"): {"ylim": (-0.01, 0.12), "yticks": np.arange(0.00, 0.101, 0.02)},
    ("tmin", "dyn"): {"ylim": (-0.04, 0.12), "yticks": np.arange(-0.04, 0.101, 0.02)},
    ("tmin", "thermo"): {"ylim": (-0.01, 0.12), "yticks": np.arange(0.00, 0.101, 0.02)},
}


def _ordered_series(df_indexed, col_name):
    vals = []
    for region in region_order:
        if region in df_indexed.index:
            value = df_indexed.loc[region][col_name]
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            vals.append(float(value))
        else:
            vals.append(np.nan)
    return np.array(vals, dtype=float)


def _candidate_offsets(step=0.052, max_width=0.28):
    levels = int(np.floor(max_width / step))
    candidates = [0.0]
    for lev in range(1, levels + 1):
        candidates.extend([-lev * step, lev * step])
    return candidates


def _beeswarm_offsets(values, step=0.052, max_width=0.28, min_dy=0.050):
    values = np.asarray(values, dtype=float)
    offsets = np.zeros(values.size, dtype=float)
    finite_mask = np.isfinite(values)
    finite_idx = np.where(finite_mask)[0]
    if finite_idx.size == 0:
        return offsets
    candidates = _candidate_offsets(step=step, max_width=max_width)
    placed = []
    for idx in finite_idx[np.argsort(values[finite_idx], kind="mergesort")]:
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
    ax.text(x_cols[0] + line_len + text_gap, legend_y, "ERA5", ha="left", va="center", fontsize=FIG567_ANNOTATION_SIZE-4, color="0.1", transform=ax.transAxes)
    ax.plot([x_cols[1], x_cols[1] + line_len], [legend_y, legend_y], color=mme_color, linewidth=3.5, solid_capstyle="round", transform=ax.transAxes, clip_on=False)
    ax.text(x_cols[1] + line_len + text_gap, legend_y, "MME", ha="left", va="center", fontsize=FIG567_ANNOTATION_SIZE-4, color="0.1", transform=ax.transAxes)
    ax.plot([0.02, 0.98], [0.970, 0.970], color="0.85", linewidth=0.8, transform=ax.transAxes, clip_on=False)

def _draw_member_key(ax, ordered_members, member_style_map):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ncols_key = 4
    nrows_key = int(np.ceil(len(ordered_members) / ncols_key)) if len(ordered_members) else 1
    x_cols = np.linspace(0.03, 0.80, ncols_key)
    y_vals = np.linspace(0.80, 0.16, nrows_key)
    _draw_reference_legend(ax, x_cols)
    if len(ordered_members) == 0:
        ax.text(0.02, 0.88, "No member labels available", ha="left", va="top", fontsize=FIG567_ANNOTATION_SIZE-2, transform=ax.transAxes)
        return
    for idx, member_name in enumerate(ordered_members):
        col = idx // nrows_key
        row = idx % nrows_key
        px = x_cols[min(col, len(x_cols) - 1)]
        py = y_vals[row]
        style = member_style_map[member_name]
        _draw_scatter_point(ax, px, py, style, size=90, zorder=2, transform=ax.transAxes)
        ax.text(px + 0.050, py, member_name, ha="left", va="center", fontsize=FIG567_ANNOTATION_SIZE-4, color="0.15", transform=ax.transAxes)


def _plot_member_points(ax, region_rows, col_name, xi, layout_mode, member_style_map, member_offset_map):
    if len(region_rows) == 0:
        return
    values = region_rows[col_name].values.astype(float)
    offsets = _beeswarm_offsets(values) if layout_mode == "beeswarm" else np.array([
        member_offset_map.get(str(row_data["member"]), 0.0)
        for _, row_data in region_rows.iterrows()
    ], dtype=float)
    for offset, (_, row_data) in zip(offsets, region_rows.iterrows()):
        member_name = str(row_data["member"]) if "member" in row_data.index else ""
        style = member_style_map.get(member_name, {"marker": "o", "color": generic_member_color, "linewidth": 1.35, "alpha": 0.9})
        _draw_scatter_point(ax, xi + float(offset), float(row_data[col_name]), style, size=92, zorder=2)


member_names = [str(v) for v in monthly_da_member_region["member"].dropna().unique()] if "member" in monthly_da_member_region.columns else []
if "model_list" in globals():
    ordered_members = [str(v) for v in model_list if str(v) in member_names]
    ordered_members += [v for v in member_names if v not in ordered_members]
else:
    ordered_members = sorted(member_names)

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

member_style_map = {}
for idx, member in enumerate(ordered_members):
    if member in default_model_styles:
        marker = default_model_styles[member]["marker"]
        color = default_model_styles[member]["color"]
    else:
        marker = marker_pool[idx % len(marker_pool)]
        color = color_pool[idx % len(color_pool)]
    member_style_map[member] = {"marker": marker, "color": color, "linewidth": 1.50, "alpha": 0.8}

member_offset_map = {}
if len(ordered_members) > 0:
    offsets = np.linspace(-0.27, 0.27, len(ordered_members))
    member_offset_map = {member: offsets[idx] for idx, member in enumerate(ordered_members)}

version_specs = [
    {"layout_mode": "beeswarm", "out_name": "Fig5.png", "legacy": False},
]
era_halfwidth = 0.115
mme_halfwidth = 0.115
panel_counter = 0
for spec in version_specs:
    fig = plt.figure(figsize=(21.6, 13.8))
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

    for row_idx, var_name in enumerate(vars_order):
        era_v = monthly_da_era_region[monthly_da_era_region["var"] == var_name].set_index("region")
        mem_v = monthly_da_member_region[monthly_da_member_region["var"] == var_name]
        mme_v = monthly_da_mme_region[monthly_da_mme_region["var"] == var_name].set_index("region")

        for col_idx, (_, col_name, title) in enumerate(components):
            ax = main_axes[row_idx, col_idx]
            band_ax = band_axes[row_idx, col_idx]

            era_vals = _ordered_series(era_v, col_name)
            mme_vals = _ordered_series(mme_v, col_name)
            axis_cfg = AXIS_CONFIG[(var_name, components[col_idx][0])]

            exceed_pcts = []
            for xi, region in enumerate(region_order):
                region_rows = mem_v[mem_v["region"] == region].dropna(subset=[col_name]).reset_index(drop=True)
                _plot_member_points(ax, region_rows, col_name, xi, spec["layout_mode"], member_style_map, member_offset_map)
                era_ref = era_vals[xi]
                member_vals = region_rows[col_name].values.astype(float) if len(region_rows) > 0 else np.array([], dtype=float)
                if len(member_vals) > 0 and np.isfinite(era_ref):
                    exceed_pcts.append(100.0 * np.sum(member_vals > era_ref) / len(member_vals))
                else:
                    exceed_pcts.append(np.nan)

            for xi, mme_y in enumerate(mme_vals):
                if np.isfinite(mme_y):
                    ax.plot(
                        [xi - mme_halfwidth - 0.1, xi + mme_halfwidth + 0.1],
                        [mme_y, mme_y],
                        color=mme_color,
                        linewidth=3.3,
                        solid_capstyle="round",
                        zorder=8,
                    )

            for xi, era_y in enumerate(era_vals):
                if np.isfinite(era_y):
                    ax.plot(
                        [xi - era_halfwidth - 0.1, xi + era_halfwidth + 0.1],
                        [era_y, era_y],
                        color="k",
                        linewidth=2.8,
                        solid_capstyle="round",
                        zorder=7,
                    )

            ax.axhline(0.0, color="0.48", linewidth=1.0, zorder=1, linestyle="--")
            ax.axvline(boundary_x, color="0.62", linestyle="--", linewidth=1.0)
            ax.set_xlim(-0.62, len(region_order) - 0.38)
            ax.set_ylim(axis_cfg["ylim"])
            ax.set_yticks(axis_cfg["yticks"])
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_y))
            panel_letter = chr(97 + panel_counter)
            ax.set_title("{} | {}".format(var_name.upper(), title), loc="center", pad=MONTHLY_TITLE_PAD, fontsize=FIG567_TITLE_SIZE+1, fontweight="normal")
            ax.text(PANEL_LABEL_X_WITH_TITLE, PANEL_LABEL_Y_WITH_TITLE, panel_letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=FIG567_PANEL_LABEL_SIZE+1, fontweight="bold")
            panel_counter = (panel_counter + 1) % 4
            ax.tick_params(axis="both", labelsize=FIG567_TICK_SIZE)
            ax.tick_params(axis="x", labelbottom=False, labelsize=FIG567_TICK_SIZE)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel("Trend (?C yr$^{-1}$)", labelpad=18, fontsize=FIG567_LABEL_SIZE)

            ax.text(0.26, 0.955, "Overestimated", transform=ax.transAxes, ha="center", va="top", fontsize=FIG567_ANNOTATION_SIZE, color="0.2")
            ax.text(0.77, 0.955, "Underestimated", transform=ax.transAxes, ha="center", va="top", fontsize=FIG567_ANNOTATION_SIZE, color="0.2")

            band_ax.axvline(boundary_x, color="0.62", linestyle="--", linewidth=1.0)
            band_ax.set_ylim(0, 1)
            band_ax.set_yticks([])
            band_ax.set_xticks(x)
            band_ax.set_xticklabels(region_order, rotation=0, fontsize=FIG567_TICK_SIZE)
            band_ax.set_xlim(-0.62, len(region_order) - 0.38)
            band_ax.spines["top"].set_visible(False)
            band_ax.spines["right"].set_visible(False)
            band_ax.spines["left"].set_visible(False)
            band_ax.spines["bottom"].set_linewidth(0.8)
            band_ax.tick_params(axis="x", pad=3, labelsize=FIG567_TICK_SIZE)

            for xi, pct in enumerate(exceed_pcts):
                color = cmap(0.08) if not np.isfinite(pct) else cmap(pct / 100.0)
                band_ax.bar(xi, 0.78, width=0.82, bottom=0.10, color=color, edgecolor="0.78", linewidth=0.6)
                pct_label = "NA" if not np.isfinite(pct) else "{:.0f}%".format(pct)
                text_color = "white" if np.isfinite(pct) and pct >= 50 else "0.18"
                band_ax.text(xi, 0.49, pct_label, ha="center", va="center", fontsize=FIG567_TICK_SIZE-3, color=text_color, fontweight="semibold")

            if col_idx == 0:
                band_ax.set_ylabel("Models >\nERA5 (%)", rotation=0, labelpad=25, fontsize=FIG567_CBAR_SIZE-3, va="center")


    fig.subplots_adjust(left=0.055, right=0.978, top=0.96, bottom=0.24)

    if MONTHLY_DA_SAVE_PLOTS:
        out_stem = spec["out_name"].replace(".png", "")
        saved_png = save_figure_multi_format(fig, out_stem, dpi=320)
        print("[MONTHLY_DA] Saved:", saved_png)
        if spec["legacy"]:
            legacy_png = save_figure_multi_format(fig, "DA_Ridge_region_trends_dyn_thermo_only", dpi=320)
            print("[MONTHLY_DA] Saved:", legacy_png)
    plt.show()
