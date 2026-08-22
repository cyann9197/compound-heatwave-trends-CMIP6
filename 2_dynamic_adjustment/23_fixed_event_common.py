import importlib.util
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed


CODE_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("CHW_DATA_ROOT", "data"))
CACHE_ROOT = Path(os.environ.get("CHW_CACHE_ROOT", "cache"))
DEFAULT_BASE_DIR = str(CACHE_ROOT / "dynamic_adjustment" / "fixed_event")
ERA5_THRESHOLD_FILE = DATA_ROOT / "thresholds" / "percent_ERA5.nc"
MODEL_THRESHOLD_ROOT = DATA_ROOT / "heatwaves_files" / "1x1" / "model" / "percent"
TARGET_N_JOBS = max(1, int(os.environ.get("DA_TARGET_NJOBS", min(os.cpu_count() or 1, 24))))
YEARCHW_MIN_VALID_YEARS = max(2, int(os.environ.get("DA_YEARCHW_MIN_YEARS", "10")))


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(name, default=None):
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        return list(default or [])
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _import_script(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHW_MOD = _import_script(CODE_DIR / "22_daily_da_common.py", "daily_chwonly_base")


def _load_daily_namespace():
    # 公开版直接导入已提取的日尺度动力调整函数，不再读取大型 notebook。
    import daily_da_core as core
    return {name: value for name, value in vars(core).items() if not name.startswith("__")}


def _result_base_dir():
    return os.environ.get("DA_BASE_DIR", DEFAULT_BASE_DIR)


def _variant_specs():
    default_tags = [
        "chwcumheatcf_alljja",
        "chwcumheatcf_chwcalendar31d",
    ]
    tags = [tag.lower() for tag in _env_list("DA_VARIANT_TAGS", default=default_tags)]
    mapping = {
        "chwcumheatcf_alljja": {
            "tag": "chwcumheatcf_alljja",
            "training_mode": "all_jja",
            "top_n": None,
            "scheme_label": "fixed-event | JJA 全部日值训练",
        },
        "chwcumheatcf_hottest15": {
            "tag": "chwcumheatcf_hottest15",
            "training_mode": "topn",
            "top_n": 15,
            "scheme_label": "fixed-event | 每年最强 15 天训练",
        },
        "chwcumheatcf_hottest30": {
            "tag": "chwcumheatcf_hottest30",
            "training_mode": "topn",
            "top_n": 30,
            "scheme_label": "fixed-event | 每年最强 30 天训练",
        },
        "chwcumheatcf_chwcenter31d": {
            "tag": "chwcumheatcf_chwcenter31d",
            "training_mode": "chwcenter_window",
            "top_n": None,
            "scheme_label": "fixed-event | CHW 日同年前后 15 天窗口训练",
        },
        "chwcumheatcf_chwcalendar31d": {
            "tag": "chwcumheatcf_chwcalendar31d",
            "training_mode": "chwcalendar_window",
            "top_n": None,
            "scheme_label": "fixed-event | CHW 日所有年份同日序前后 15 天窗口训练",
        },
        "chwcumheatcf_chwdaily": {
            "tag": "chwcumheatcf_chwdaily",
            "training_mode": "chwdaily",
            "top_n": None,
            "scheme_label": "fixed-event | CHW 日逐日训练",
        },
    }
    out = []
    for tag in tags:
        if tag not in mapping:
            raise ValueError(f"不支持的变体标签：{tag}")
        out.append(dict(mapping[tag]))
    return out


def _build_variant_paths(spec):
    base_dir = Path(_result_base_dir())
    out_dir = base_dir / "outputs"
    fig_dir = base_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tag = spec["tag"]
    member_cache_dir = out_dir / f"{tag}_member_cache"
    daily_cache_dir = out_dir / f"{tag}_daily_cache"
    member_cache_dir.mkdir(parents=True, exist_ok=True)
    daily_cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "fig_dir": str(fig_dir),
        "tag": tag,
        "training_mode": spec["training_mode"],
        "top_n": spec["top_n"],
        "reuse_from_tag": spec.get("reuse_from_tag"),
        "scheme_label": spec["scheme_label"],
        "member_sources": str(out_dir / f"da_ridge_daily_{tag}_z500_member_availability_1981_2014.csv"),
        "era5_region": str(out_dir / f"da_ridge_daily_{tag}_region_trends_ERA5_1981_2014_JJA.csv"),
        "era5_spatial": str(out_dir / f"da_ridge_daily_{tag}_spatial_trends_ERA5_JJA_1981_2014.nc"),
        "era5_annual": str(out_dir / f"da_ridge_daily_{tag}_annual_fields_ERA5_JJA_1981_2014.nc"),
        "members_region": str(out_dir / f"da_ridge_daily_{tag}_region_trends_CMIP_members_JJA_1981_2014.csv"),
        "mme_region": str(out_dir / f"da_ridge_daily_{tag}_region_trends_MME_JJA_1981_2014.csv"),
        "members_spatial": str(out_dir / f"da_ridge_daily_{tag}_spatial_trends_CMIP_members_JJA_1981_2014.nc"),
        "mme_spatial": str(out_dir / f"da_ridge_daily_{tag}_spatial_trends_MME_JJA_1981_2014.nc"),
        "members_annual": str(out_dir / f"da_ridge_daily_{tag}_annual_fields_CMIP_members_JJA_1981_2014.nc"),
        "mme_annual": str(out_dir / f"da_ridge_daily_{tag}_annual_fields_MME_JJA_1981_2014.nc"),
        "member_cache_dir": str(member_cache_dir),
        "daily_cache_dir": str(daily_cache_dir),
    }


def _member_region_path(paths, member):
    return os.path.join(paths["member_cache_dir"], f"{member}_region.csv")


def _member_spatial_path(paths, member):
    return os.path.join(paths["member_cache_dir"], f"{member}_spatial.nc")


def _member_annual_path(paths, member):
    return os.path.join(paths["member_cache_dir"], f"{member}_annual.nc")


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))


def _daily_region_path(paths, owner, var_name, region_name):
    owner_dir = Path(paths["daily_cache_dir"]) / _safe_name(owner)
    owner_dir.mkdir(parents=True, exist_ok=True)
    return str(owner_dir / f"{_safe_name(var_name)}_{_safe_name(region_name)}_daily_fixed_event.nc")


def _years_from_time(time_values):
    years = []
    for value in time_values:
        if hasattr(value, "year"):
            years.append(int(value.year))
        else:
            years.append(int(pd.Timestamp(value).year))
    return np.asarray(years, dtype=int)


def _build_year_indexer(time_values):
    years = _years_from_time(time_values)
    unique_years = np.unique(years).astype(int)
    year_row_index = [np.where(years == year)[0] for year in unique_years]
    return years, unique_years, year_row_index


def _build_gridwise_all_jja_mask(y_total_values):
    mask = np.isfinite(y_total_values)
    sample_counts = mask.sum(axis=0).astype(int)
    return mask, sample_counts


def _build_gridwise_topn_mask(y_total_values, year_row_index, top_n_days):
    return CHW_MOD._build_gridwise_topn_mask(y_total_values, year_row_index, top_n_days)


def _build_gridwise_chw_daily_mask(y_total_values, chw_mask):
    mask = np.isfinite(y_total_values) & np.asarray(chw_mask, dtype=bool)
    sample_counts = mask.sum(axis=0).astype(int)
    return mask, sample_counts


def _build_row_year_and_day_index(year_row_index, n_time):
    row_year_index = np.full(n_time, -1, dtype=int)
    row_day_index = np.full(n_time, -1, dtype=int)
    for year_idx, rows in enumerate(year_row_index):
        row_year_index[rows] = int(year_idx)
        row_day_index[rows] = np.arange(len(rows), dtype=int)
    return row_year_index, row_day_index


def _build_calendar_window_rows(year_row_index, max_day_index, half_window=15):
    rows_by_center = {}
    for center_day in range(int(max_day_index) + 1):
        pieces = []
        for rows in year_row_index:
            if center_day >= len(rows):
                continue
            start = max(0, center_day - int(half_window))
            stop = min(len(rows), center_day + int(half_window) + 1)
            pieces.append(rows[start:stop])
        rows_by_center[center_day] = np.concatenate(pieces).astype(int) if pieces else np.array([], dtype=int)
    return rows_by_center


def _build_gridwise_chw_center_window_mask(y_total_values, chw_mask, year_row_index, half_window=15):
    chw_bool = np.asarray(chw_mask, dtype=bool)
    finite_y = np.isfinite(y_total_values)
    n_time, n_targets = y_total_values.shape
    row_year_index, row_day_index = _build_row_year_and_day_index(year_row_index, n_time)
    mask = np.zeros((n_time, n_targets), dtype=bool)

    for target_idx in range(n_targets):
        event_rows = np.where(chw_bool[:, target_idx] & finite_y[:, target_idx])[0]
        if event_rows.size == 0:
            continue
        for row in event_rows:
            year_idx = row_year_index[row]
            day_idx = row_day_index[row]
            if year_idx < 0 or day_idx < 0:
                continue
            rows = year_row_index[year_idx]
            start = max(0, int(day_idx) - int(half_window))
            stop = min(len(rows), int(day_idx) + int(half_window) + 1)
            mask[rows[start:stop], target_idx] = True

    mask &= finite_y
    sample_counts = mask.sum(axis=0).astype(int)
    return mask, sample_counts


def _build_gridwise_chw_calendar_window_mask(y_total_values, chw_mask, year_row_index, half_window=15):
    chw_bool = np.asarray(chw_mask, dtype=bool)
    finite_y = np.isfinite(y_total_values)
    n_time, n_targets = y_total_values.shape
    _, row_day_index = _build_row_year_and_day_index(year_row_index, n_time)
    max_day_index = int(np.nanmax(row_day_index)) if np.any(row_day_index >= 0) else 0
    rows_by_center = _build_calendar_window_rows(year_row_index, max_day_index, half_window=half_window)
    mask = np.zeros((n_time, n_targets), dtype=bool)

    for target_idx in range(n_targets):
        event_rows = np.where(chw_bool[:, target_idx] & finite_y[:, target_idx])[0]
        if event_rows.size == 0:
            continue
        for center_day in np.unique(row_day_index[event_rows]):
            if center_day < 0:
                continue
            rows = rows_by_center.get(int(center_day), np.array([], dtype=int))
            if rows.size:
                mask[rows, target_idx] = True

    mask &= finite_y
    sample_counts = mask.sum(axis=0).astype(int)
    return mask, sample_counts


def _build_daily_training_mask(training_mode, y_total_full, chw_mask, year_row_index, top_n_days=None):
    if training_mode == "all_jja":
        return _build_gridwise_all_jja_mask(y_total_full)
    if training_mode == "topn":
        if top_n_days is None:
            raise ValueError("topn 训练模式必须提供 top_n_days。")
        return _build_gridwise_topn_mask(y_total_full, year_row_index, top_n_days)
    if training_mode == "chwdaily":
        return _build_gridwise_chw_daily_mask(y_total_full, chw_mask)
    if training_mode == "chwcenter_window":
        return _build_gridwise_chw_center_window_mask(y_total_full, chw_mask, year_row_index, half_window=15)
    if training_mode == "chwcalendar_window":
        return _build_gridwise_chw_calendar_window_mask(y_total_full, chw_mask, year_row_index, half_window=15)
    raise ValueError(f"不支持的逐日训练模式：{training_mode}")


def _vector_slope(values, years, min_count=5):
    return CHW_MOD._vector_slope(values, years, min_count=min_count)


def _choose_shared_alpha_gridwise(ns, X_full, Y_full, train_mask, years, **kwargs):
    return CHW_MOD._choose_shared_alpha_gridwise(ns, X_full, Y_full, train_mask, years, **kwargs)


def _choose_shared_alpha_yearchw(
    ns,
    annual_x_list,
    annual_y_anom_all,
    valid_year_mask_all,
    unique_years,
    standardize_X="zscore",
    alphas=None,
    cv_year_folds=5,
    max_targets_for_cv=24,
    random_state=0,
):
    if alphas is None:
        alphas = np.logspace(-2, 4, 13)

    valid_targets = [
        idx
        for idx in range(valid_year_mask_all.shape[1])
        if int(valid_year_mask_all[:, idx].sum()) >= YEARCHW_MIN_VALID_YEARS
    ]
    if not valid_targets:
        return np.nan, np.nan

    rng = np.random.RandomState(random_state)
    if len(valid_targets) > max_targets_for_cv:
        valid_targets = sorted(rng.choice(valid_targets, size=max_targets_for_cv, replace=False).tolist())

    best_alpha = float(alphas[0])
    best_score = np.inf
    for alpha in alphas:
        fold_scores = []
        for target_idx in valid_targets:
            rows = np.where(valid_year_mask_all[:, target_idx])[0]
            n_splits = min(int(cv_year_folds), int(rows.size))
            if n_splits < 2:
                continue
            X_target = annual_x_list[target_idx][rows, :]
            Y_target = annual_y_anom_all[rows, target_idx]
            X_std, _ = ns["daily_da_standardize_matrix"](X_target, mode=standardize_X)
            groups = unique_years[rows]
            splitter = ns["GroupKFold"](n_splits=n_splits)
            for train_idx, test_idx in splitter.split(X_std, Y_target, groups):
                model = ns["Ridge"](alpha=float(alpha), fit_intercept=True)
                model.fit(X_std[train_idx], Y_target[train_idx])
                pred = model.predict(X_std[test_idx])
                fold_scores.append(float(np.mean((pred - Y_target[test_idx]) ** 2)))
        if fold_scores:
            mean_score = float(np.mean(fold_scores))
            if mean_score < best_score:
                best_score = mean_score
                best_alpha = float(alpha)

    if not np.isfinite(best_score):
        return np.nan, np.nan
    return float(best_alpha), float(best_score)


def _put_annual_on_template(region_da, template_2d):
    years = region_da["year"].values
    full = xr.DataArray(
        np.full((len(years), template_2d.sizes["lat"], template_2d.sizes["lon"]), np.nan, dtype=np.float32),
        dims=("year", "lat", "lon"),
        coords={"year": years, "lat": template_2d["lat"], "lon": template_2d["lon"]},
    )
    full.loc[dict(lat=region_da["lat"], lon=region_da["lon"])] = region_da.astype(np.float32)
    return full


def _write_dataset(dataset_or_mapping, path):
    ds = dataset_or_mapping if isinstance(dataset_or_mapping, xr.Dataset) else xr.Dataset(dataset_or_mapping)
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    return ds


def _write_daily_region_dataset(paths, owner, var_name, region_name, fit_result):
    needed = ("daily_dyn_pred", "daily_total_excess", "daily_residual_fixed", "daily_chw_mask")
    if not all(name in fit_result for name in needed):
        return None
    out_path = _daily_region_path(paths, owner, var_name, region_name)
    ds = xr.Dataset(
        {
            f"{var_name}_hwc_daily_dyn_pred": fit_result["daily_dyn_pred"].astype("float32"),
            f"{var_name}_hwc_daily_total_excess": fit_result["daily_total_excess"].astype("float32"),
            f"{var_name}_hwc_daily_residual_fixed": fit_result["daily_residual_fixed"].astype("float32"),
            "chw_mask_daily": fit_result["daily_chw_mask"].astype("float32"),
        }
    )
    _write_dataset(ds, out_path)
    return out_path


def _standardize_threshold_da(ns, da):
    da = ns["daily_da_standardize_coords"](da)
    if float(da["lon"].max()) > 180.0:
        da = ns["daily_da_filp_lon"](da, lon_name="lon")
        da = ns["daily_da_standardize_coords"](da)
    da = da.sel(lat=slice(0, 90))
    if "mask" in da.coords:
        da = da.reset_coords("mask", drop=True)
    mean_value = float(da.mean(skipna=True).values)
    if mean_value > 150.0:
        da = da - 273.15
    return da.astype("float32")


def _open_era5_thresholds(ns):
    ds = xr.open_dataset(ERA5_THRESHOLD_FILE)
    return {
        "tmax": _standardize_threshold_da(ns, ds["percent_Tmax"]),
        "tmin": _standardize_threshold_da(ns, ds["percent_Tmin"]),
    }


def _open_member_thresholds(ns, member):
    path = MODEL_THRESHOLD_ROOT / f"{member}_HW_thresholds_1981-2010.nc"
    if not path.exists():
        raise FileNotFoundError(f"未找到模式阈值文件：{path}")
    ds = xr.open_dataset(path)
    return {
        "tmax": _standardize_threshold_da(ns, ds["tmax_th90"]),
        "tmin": _standardize_threshold_da(ns, ds["tmin_th90"]),
    }


def _align_threshold_to_region(threshold_da, y_sel_full):
    th = threshold_da
    if "mask" in th.coords:
        th = th.reset_coords("mask", drop=True)
    try:
        th = th.sel(lat=y_sel_full["lat"], lon=y_sel_full["lon"])
    except Exception:
        th = th.interp(lat=y_sel_full["lat"], lon=y_sel_full["lon"], method="nearest")
    return th


def _jja_day_positions(time_values, n_threshold_days):
    years = _years_from_time(time_values)
    positions = np.zeros(len(years), dtype=int)
    for year in np.unique(years):
        rows = np.where(years == year)[0]
        if rows.size > n_threshold_days:
            raise ValueError(
                f"{year} 年 JJA 日数为 {rows.size}，但阈值 day 维为 {n_threshold_days}，请检查日历或阈值文件。"
            )
        # 少数 360-day 模式的 CHW 掩膜在既有代码里按 Gregorian 日期展开，
        # 与温度场对齐后每年会少最后 1 天；这里按对齐后的实际日序取阈值。
        positions[rows] = np.arange(rows.size, dtype=int)
    return positions


def _threshold_to_time(threshold_region, y_sel_full):
    th = _align_threshold_to_region(threshold_region, y_sel_full)
    day_dim = "day"
    if day_dim not in th.dims:
        raise ValueError("阈值文件必须包含 day 维。")
    th = th.transpose(day_dim, "lat", "lon")
    positions = _jja_day_positions(y_sel_full["time"].values, th.sizes[day_dim])
    values = th.values.astype(np.float32)[positions, :, :]
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": y_sel_full["time"], "lat": y_sel_full["lat"], "lon": y_sel_full["lon"]},
        name="threshold",
    )


def _prepare_cumheat_daily_matrices(ns, z500_rel, y_region, chw_region, threshold_region, predictor_expand_deg=15.0):
    z500_rel = ns["daily_da_standardize_coords"](z500_rel)
    y_region = ns["daily_da_standardize_coords"](y_region)
    chw_region = ns["daily_da_standardize_coords"](chw_region)
    threshold_region = ns["daily_da_standardize_coords"](threshold_region)
    if "mask" in y_region.coords:
        y_region = y_region.reset_coords("mask", drop=True)
    if "mask" in chw_region.coords:
        chw_region = chw_region.reset_coords("mask", drop=True)

    z_sel = ns["daily_da_select_jja"](z500_rel)
    y_sel_full = ns["daily_da_select_jja"](y_region)
    chw_sel_full = ns["daily_da_select_jja"](chw_region)
    z_sel, y_sel_full, chw_sel_full = xr.align(z_sel, y_sel_full, chw_sel_full, join="inner")
    threshold_time = _threshold_to_time(threshold_region, y_sel_full)
    y_sel_full, chw_sel_full, threshold_time = xr.align(y_sel_full, chw_sel_full, threshold_time, join="inner")

    y_box = (
        float(y_sel_full["lon"].min()),
        float(y_sel_full["lon"].max()),
        float(y_sel_full["lat"].min()),
        float(y_sel_full["lat"].max()),
    )
    pred_box = ns["daily_da_expand_box"](y_box, expand_deg=predictor_expand_deg)
    z_dom_full = ns["daily_da_select_box_wrap"](z_sel, pred_box)
    z_dom_full, y_sel_full, chw_sel_full, threshold_time = xr.align(
        z_dom_full, y_sel_full, chw_sel_full, threshold_time, join="inner"
    )

    z_anom_full = ns["daily_da_calendar_day_anomaly"](z_dom_full)
    y_anom_full = ns["daily_da_calendar_day_anomaly"](y_sel_full)

    X_full, _ = ns["daily_da_stack_feature_matrix"](z_anom_full)
    Y_anom_full, y_stack = ns["daily_da_stack_target_matrix"](y_anom_full)
    Y_total_full = (
        y_sel_full.stack(target=("lat", "lon"))
        .transpose("time", "target")
        .sel(target=y_stack["target"])
        .values.astype(float)
    )
    threshold_full = (
        threshold_time.stack(target=("lat", "lon"))
        .transpose("time", "target")
        .sel(target=y_stack["target"])
        .values.astype(float)
    )
    CHW_full = (
        chw_sel_full.stack(target=("lat", "lon"))
        .transpose("time", "target")
        .sel(target=y_stack["target"])
        .values.astype(float)
    )
    CHW_mask = np.isfinite(CHW_full) & (CHW_full > 0.5)
    valid_excess = CHW_mask & np.isfinite(Y_total_full) & np.isfinite(threshold_full)
    excess_full = np.where(valid_excess, np.maximum(Y_total_full - threshold_full, 0.0), 0.0)
    Y_target_full = excess_full.astype(float)

    years, unique_years, year_row_index = _build_year_indexer(y_sel_full["time"].values)
    target_valid = np.isfinite(Y_total_full).any(axis=0) & np.isfinite(threshold_full).any(axis=0)
    return {
        "pred_box": pred_box,
        "X_full": X_full,
        "Y_anom_full": Y_anom_full,
        "Y_target_full": Y_target_full,
        "Y_total_full": Y_total_full,
        "threshold_full": threshold_full,
        "CHW_mask": CHW_mask,
        "valid_excess": valid_excess,
        "excess_full": excess_full,
        "target_valid": target_valid,
        "years": years,
        "unique_years": unique_years,
        "year_row_index": year_row_index,
        "y_stack": y_stack,
        "time_values": y_sel_full["time"].values,
    }


def _annual_totals_and_days(excess_full, chw_mask, target_valid, year_row_index):
    n_years = len(year_row_index)
    n_targets = excess_full.shape[1]
    annual_total = np.full((n_years, n_targets), np.nan, dtype=float)
    annual_chw_days = np.full((n_years, n_targets), np.nan, dtype=float)
    for target_idx in range(n_targets):
        if not target_valid[target_idx]:
            continue
        for year_idx, rows in enumerate(year_row_index):
            chw_days = int(np.sum(chw_mask[rows, target_idx]))
            annual_chw_days[year_idx, target_idx] = float(chw_days)
            annual_total[year_idx, target_idx] = float(np.nansum(excess_full[rows, target_idx])) if chw_days > 0 else 0.0
    return annual_total, annual_chw_days


def _fit_gridwise_cumheat_daily_region(
    ns,
    z500_rel,
    y_region,
    chw_region,
    threshold_region,
    training_mode="all_jja",
    top_n_days=None,
    predictor_expand_deg=15.0,
    standardize_X="zscore",
    alphas=None,
    cv_year_folds=5,
    max_targets_for_cv=24,
    random_state=0,
):
    if alphas is None:
        alphas = np.logspace(-2, 4, 13)

    prepared = _prepare_cumheat_daily_matrices(
        ns,
        z500_rel,
        y_region,
        chw_region,
        threshold_region,
        predictor_expand_deg=predictor_expand_deg,
    )
    X_full = prepared["X_full"]
    Y_target_full = prepared["Y_target_full"]
    Y_total_full = prepared["Y_total_full"]
    threshold_full = prepared["threshold_full"]
    CHW_mask = prepared["CHW_mask"]
    valid_excess = prepared["valid_excess"]
    excess_full = prepared["excess_full"]
    target_valid = prepared["target_valid"]
    years = prepared["years"]
    unique_years = prepared["unique_years"]
    year_row_index = prepared["year_row_index"]
    y_stack = prepared["y_stack"]
    pred_box = prepared["pred_box"]

    if X_full.shape[1] < 2:
        raise ValueError("有效预测格点过少，无法进行 Ridge 回归。")
    if Y_target_full.shape[1] < 1:
        raise ValueError("有效目标格点为空。")

    train_mask, train_sample_counts = _build_daily_training_mask(
        training_mode,
        Y_total_full,
        CHW_mask,
        year_row_index,
        top_n_days=top_n_days,
    )
    display_mask = valid_excess
    display_sample_counts = display_mask.sum(axis=0).astype(int)
    annual_total, annual_chw_days = _annual_totals_and_days(excess_full, CHW_mask, target_valid, year_row_index)

    alpha, alpha_score = _choose_shared_alpha_gridwise(
        ns,
        X_full,
        Y_target_full,
        train_mask,
        years,
        standardize_X=standardize_X,
        alphas=alphas,
        cv_year_folds=cv_year_folds,
        max_targets_for_cv=max_targets_for_cv,
        random_state=random_state,
    )

    n_years = unique_years.size
    n_targets = Y_target_full.shape[1]
    annual_dyn = np.full((n_years, n_targets), np.nan, dtype=float)
    annual_thermo = np.full((n_years, n_targets), np.nan, dtype=float)
    daily_dyn_pred = np.full_like(Y_target_full, np.nan, dtype=np.float32)
    daily_total_excess = np.full_like(Y_target_full, np.nan, dtype=np.float32)
    daily_residual_fixed = np.full_like(Y_target_full, np.nan, dtype=np.float32)
    daily_chw_mask = np.where(display_mask, 1.0, 0.0).astype(np.float32)
    r2 = np.full(n_targets, np.nan, dtype=float)

    def _solve_one_target(target_idx):
        total_col = annual_total[:, target_idx].copy()
        chw_days_col = annual_chw_days[:, target_idx]
        dyn_col = np.full(n_years, np.nan, dtype=float)
        adj_col = np.full(n_years, np.nan, dtype=float)
        daily_dyn_col = np.full(Y_target_full.shape[0], np.nan, dtype=np.float32)
        daily_total_col = np.full(Y_target_full.shape[0], np.nan, dtype=np.float32)
        daily_residual_col = np.full(Y_target_full.shape[0], np.nan, dtype=np.float32)
        r2_val = np.nan
        if not target_valid[target_idx]:
            return target_idx, dyn_col, adj_col, r2_val, daily_dyn_col, daily_total_col, daily_residual_col

        no_chw_years = np.isfinite(chw_days_col) & (chw_days_col == 0)
        dyn_col[no_chw_years] = 0.0
        adj_col[no_chw_years] = 0.0

        train_rows = np.where(train_mask[:, target_idx] & np.isfinite(Y_target_full[:, target_idx]))[0]
        if train_rows.size < 2 or not np.isfinite(alpha):
            return target_idx, dyn_col, adj_col, r2_val, daily_dyn_col, daily_total_col, daily_residual_col

        X_train = X_full[train_rows, :]
        Y_train = Y_target_full[train_rows, target_idx]
        X_train_std, scaler = ns["daily_da_standardize_matrix"](X_train, mode=standardize_X)
        model = ns["Ridge"](alpha=float(alpha), fit_intercept=True)
        model.fit(X_train_std, Y_train)
        pred_train = model.predict(X_train_std)
        ss_res = float(np.sum((Y_train - pred_train) ** 2))
        ss_tot = float(np.sum((Y_train - np.mean(Y_train)) ** 2))
        if ss_tot > 0:
            r2_val = 1.0 - ss_res / ss_tot

        display_rows = np.where(display_mask[:, target_idx])[0]
        if display_rows.size == 0:
            return target_idx, dyn_col, adj_col, r2_val, daily_dyn_col, daily_total_col, daily_residual_col

        X_display = X_full[display_rows, :]
        X_display_std = scaler.transform(X_display) if scaler is not None else X_display
        pred_display = model.predict(X_display_std)
        total_excess_display = excess_full[display_rows, target_idx]
        residual_display = total_excess_display - pred_display
        daily_dyn_col[display_rows] = pred_display.astype(np.float32)
        daily_total_col[display_rows] = total_excess_display.astype(np.float32)
        daily_residual_col[display_rows] = residual_display.astype(np.float32)
        display_years = years[display_rows]
        for year_idx, year in enumerate(unique_years):
            if not np.isfinite(chw_days_col[year_idx]) or chw_days_col[year_idx] == 0:
                continue
            year_mask = display_years == year
            if np.any(year_mask):
                residual_sum = float(np.nansum(residual_display[year_mask]))
                dyn_col[year_idx] = float(np.nansum(pred_display[year_mask]))
                adj_col[year_idx] = residual_sum
        return target_idx, dyn_col, adj_col, r2_val, daily_dyn_col, daily_total_col, daily_residual_col

    n_jobs = max(1, min(TARGET_N_JOBS, n_targets))
    results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=1)(
        delayed(_solve_one_target)(target_idx) for target_idx in range(n_targets)
    )
    for target_idx, dyn_col, adj_col, r2_val, daily_dyn_col, daily_total_col, daily_residual_col in results:
        annual_dyn[:, target_idx] = dyn_col
        annual_thermo[:, target_idx] = adj_col
        r2[target_idx] = r2_val
        daily_dyn_pred[:, target_idx] = daily_dyn_col
        daily_total_excess[:, target_idx] = daily_total_col
        daily_residual_fixed[:, target_idx] = daily_residual_col

    return _pack_fit_result(
        annual_total,
        annual_dyn,
        annual_thermo,
        annual_chw_days,
        r2,
        unique_years,
        y_stack,
        pred_box,
        alpha,
        alpha_score,
        training_mode,
        top_n_days,
        standardize_X,
        train_sample_counts,
        display_sample_counts,
        time_values=prepared["time_values"],
        daily_dyn_pred=daily_dyn_pred,
        daily_total_excess=daily_total_excess,
        daily_residual_fixed=daily_residual_fixed,
        daily_chw_mask=daily_chw_mask,
    )


def _fit_gridwise_cumheat_calendar_window_region(
    ns,
    z500_rel,
    y_region,
    chw_region,
    threshold_region,
    predictor_expand_deg=15.0,
    standardize_X="zscore",
    alphas=None,
    cv_year_folds=5,
    max_targets_for_cv=24,
    random_state=0,
    half_window=15,
):
    if alphas is None:
        alphas = np.logspace(-2, 4, 13)

    prepared = _prepare_cumheat_daily_matrices(
        ns,
        z500_rel,
        y_region,
        chw_region,
        threshold_region,
        predictor_expand_deg=predictor_expand_deg,
    )
    X_full = prepared["X_full"]
    Y_target_full = prepared["Y_target_full"]
    Y_total_full = prepared["Y_total_full"]
    threshold_full = prepared["threshold_full"]
    CHW_mask = prepared["CHW_mask"]
    valid_excess = prepared["valid_excess"]
    excess_full = prepared["excess_full"]
    target_valid = prepared["target_valid"]
    years = prepared["years"]
    unique_years = prepared["unique_years"]
    year_row_index = prepared["year_row_index"]
    y_stack = prepared["y_stack"]
    pred_box = prepared["pred_box"]

    if X_full.shape[1] < 2:
        raise ValueError("有效预测格点过少，无法进行 Ridge 回归。")
    if Y_target_full.shape[1] < 1:
        raise ValueError("有效目标格点为空。")

    train_mask, train_sample_counts = _build_gridwise_chw_calendar_window_mask(
        Y_total_full,
        CHW_mask,
        year_row_index,
        half_window=half_window,
    )
    display_mask = valid_excess
    display_sample_counts = display_mask.sum(axis=0).astype(int)
    annual_total, annual_chw_days = _annual_totals_and_days(excess_full, CHW_mask, target_valid, year_row_index)

    alpha, alpha_score = _choose_shared_alpha_gridwise(
        ns,
        X_full,
        Y_target_full,
        train_mask,
        years,
        standardize_X=standardize_X,
        alphas=alphas,
        cv_year_folds=cv_year_folds,
        max_targets_for_cv=max_targets_for_cv,
        random_state=random_state,
    )

    n_time = Y_target_full.shape[0]
    n_years = unique_years.size
    n_targets = Y_target_full.shape[1]
    row_year_index, row_day_index = _build_row_year_and_day_index(year_row_index, n_time)
    max_day_index = int(np.nanmax(row_day_index)) if np.any(row_day_index >= 0) else 0
    rows_by_center = _build_calendar_window_rows(year_row_index, max_day_index, half_window=half_window)

    annual_dyn = np.full((n_years, n_targets), np.nan, dtype=float)
    annual_thermo = np.full((n_years, n_targets), np.nan, dtype=float)
    daily_dyn_pred = np.full_like(Y_target_full, np.nan, dtype=np.float32)
    daily_total_excess = np.full_like(Y_target_full, np.nan, dtype=np.float32)
    daily_residual_fixed = np.full_like(Y_target_full, np.nan, dtype=np.float32)
    daily_chw_mask = np.where(display_mask, 1.0, 0.0).astype(np.float32)
    r2 = np.full(n_targets, np.nan, dtype=float)

    def _solve_one_target(target_idx):
        total_col = annual_total[:, target_idx].copy()
        chw_days_col = annual_chw_days[:, target_idx]
        dyn_col = np.full(n_years, np.nan, dtype=float)
        adj_col = np.full(n_years, np.nan, dtype=float)
        daily_dyn_col = np.full(n_time, np.nan, dtype=np.float32)
        daily_total_col = np.full(n_time, np.nan, dtype=np.float32)
        daily_residual_col = np.full(n_time, np.nan, dtype=np.float32)
        r2_scores = []
        if not target_valid[target_idx]:
            return target_idx, dyn_col, adj_col, np.nan, daily_dyn_col, daily_total_col, daily_residual_col

        no_chw_years = np.isfinite(chw_days_col) & (chw_days_col == 0)
        dyn_col[no_chw_years] = 0.0
        adj_col[no_chw_years] = 0.0

        event_rows = np.where(display_mask[:, target_idx])[0]
        if event_rows.size == 0 or not np.isfinite(alpha):
            return target_idx, dyn_col, adj_col, np.nan, daily_dyn_col, daily_total_col, daily_residual_col

        residual_sum = np.zeros(n_years, dtype=float)
        predicted_counts = np.zeros(n_years, dtype=int)
        event_center_days = np.unique(row_day_index[event_rows])

        for center_day in event_center_days:
            if center_day < 0:
                continue
            train_rows = rows_by_center.get(int(center_day), np.array([], dtype=int))
            if train_rows.size == 0:
                continue
            train_rows = train_rows[np.isfinite(Y_target_full[train_rows, target_idx])]
            if train_rows.size < 2:
                continue

            X_train = X_full[train_rows, :]
            Y_train = Y_target_full[train_rows, target_idx]
            X_train_std, scaler = ns["daily_da_standardize_matrix"](X_train, mode=standardize_X)
            model = ns["Ridge"](alpha=float(alpha), fit_intercept=True)
            model.fit(X_train_std, Y_train)

            pred_train = model.predict(X_train_std)
            ss_res = float(np.sum((Y_train - pred_train) ** 2))
            ss_tot = float(np.sum((Y_train - np.mean(Y_train)) ** 2))
            if ss_tot > 0:
                r2_scores.append(1.0 - ss_res / ss_tot)

            predict_rows = event_rows[row_day_index[event_rows] == center_day]
            if predict_rows.size == 0:
                continue
            X_predict = X_full[predict_rows, :]
            X_predict_std = scaler.transform(X_predict) if scaler is not None else X_predict
            pred_event = model.predict(X_predict_std)
            total_excess_event = excess_full[predict_rows, target_idx]
            residual_event = total_excess_event - pred_event
            daily_dyn_col[predict_rows] = pred_event.astype(np.float32)
            daily_total_col[predict_rows] = total_excess_event.astype(np.float32)
            daily_residual_col[predict_rows] = residual_event.astype(np.float32)
            for row, resid_value in zip(predict_rows, residual_event):
                year_idx = row_year_index[row]
                if year_idx >= 0:
                    residual_sum[year_idx] += float(resid_value)
                    predicted_counts[year_idx] += 1

        for year_idx in range(n_years):
            if not np.isfinite(chw_days_col[year_idx]) or chw_days_col[year_idx] == 0:
                continue
            expected_count = int(round(float(chw_days_col[year_idx])))
            if predicted_counts[year_idx] == expected_count:
                dyn_col[year_idx] = float(np.nansum(daily_dyn_col[year_row_index[year_idx]]))
                adj_col[year_idx] = float(residual_sum[year_idx])

        r2_val = float(np.nanmean(r2_scores)) if r2_scores else np.nan
        return target_idx, dyn_col, adj_col, r2_val, daily_dyn_col, daily_total_col, daily_residual_col

    n_jobs = max(1, min(TARGET_N_JOBS, n_targets))
    results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=1)(
        delayed(_solve_one_target)(target_idx) for target_idx in range(n_targets)
    )
    for target_idx, dyn_col, adj_col, r2_val, daily_dyn_col, daily_total_col, daily_residual_col in results:
        annual_dyn[:, target_idx] = dyn_col
        annual_thermo[:, target_idx] = adj_col
        r2[target_idx] = r2_val
        daily_dyn_pred[:, target_idx] = daily_dyn_col
        daily_total_excess[:, target_idx] = daily_total_col
        daily_residual_fixed[:, target_idx] = daily_residual_col

    return _pack_fit_result(
        annual_total,
        annual_dyn,
        annual_thermo,
        annual_chw_days,
        r2,
        unique_years,
        y_stack,
        pred_box,
        alpha,
        alpha_score,
        "chwcalendar_window",
        None,
        standardize_X,
        train_sample_counts,
        display_sample_counts,
        time_values=prepared["time_values"],
        daily_dyn_pred=daily_dyn_pred,
        daily_total_excess=daily_total_excess,
        daily_residual_fixed=daily_residual_fixed,
        daily_chw_mask=daily_chw_mask,
    )


def _fit_gridwise_cumheat_yearchw_region(
    ns,
    z500_rel,
    y_region,
    chw_region,
    threshold_region,
    predictor_expand_deg=15.0,
    standardize_X="zscore",
    alphas=None,
    cv_year_folds=5,
    max_targets_for_cv=24,
    random_state=0,
):
    if alphas is None:
        alphas = np.logspace(-2, 4, 13)

    prepared = _prepare_cumheat_daily_matrices(
        ns,
        z500_rel,
        y_region,
        chw_region,
        threshold_region,
        predictor_expand_deg=predictor_expand_deg,
    )
    X_full = prepared["X_full"]
    Y_anom_full = prepared["Y_anom_full"]
    CHW_mask = prepared["CHW_mask"]
    excess_full = prepared["excess_full"]
    target_valid = prepared["target_valid"]
    unique_years = prepared["unique_years"]
    year_row_index = prepared["year_row_index"]
    y_stack = prepared["y_stack"]
    pred_box = prepared["pred_box"]

    if X_full.shape[1] < 2:
        raise ValueError("有效预测格点过少，无法进行 Ridge 回归。")

    display_mask = CHW_mask & np.isfinite(Y_anom_full)
    display_sample_counts = display_mask.sum(axis=0).astype(int)
    annual_total, annual_chw_days = _annual_totals_and_days(excess_full, CHW_mask, target_valid, year_row_index)
    n_years = unique_years.size
    n_targets = Y_anom_full.shape[1]

    annual_y_anom_all = np.full((n_years, n_targets), np.nan, dtype=float)
    annual_x_list = []
    valid_year_mask_all = np.zeros((n_years, n_targets), dtype=bool)
    train_sample_counts = np.zeros(n_targets, dtype=int)

    for target_idx in range(n_targets):
        x_year = np.full((n_years, X_full.shape[1]), np.nan, dtype=np.float32)
        y_year = np.full(n_years, np.nan, dtype=float)
        valid_year_mask = np.zeros(n_years, dtype=bool)
        if target_valid[target_idx]:
            for year_idx, year_rows in enumerate(year_row_index):
                selected_rows = year_rows[display_mask[year_rows, target_idx]]
                if selected_rows.size == 0:
                    continue
                y_year[year_idx] = float(np.nanmean(Y_anom_full[selected_rows, target_idx]))
                x_mean = np.nanmean(X_full[selected_rows, :], axis=0)
                x_year[year_idx, :] = x_mean.astype(np.float32)
                valid_year_mask[year_idx] = np.isfinite(y_year[year_idx]) and np.all(np.isfinite(x_year[year_idx, :]))
        annual_y_anom_all[:, target_idx] = y_year
        annual_x_list.append(x_year)
        valid_year_mask_all[:, target_idx] = valid_year_mask
        train_sample_counts[target_idx] = int(valid_year_mask.sum())

    alpha, alpha_score = _choose_shared_alpha_yearchw(
        ns,
        annual_x_list,
        annual_y_anom_all,
        valid_year_mask_all,
        unique_years,
        standardize_X=standardize_X,
        alphas=alphas,
        cv_year_folds=cv_year_folds,
        max_targets_for_cv=max_targets_for_cv,
        random_state=random_state,
    )

    annual_dyn = np.full((n_years, n_targets), np.nan, dtype=float)
    annual_adj = np.full((n_years, n_targets), np.nan, dtype=float)
    r2 = np.full(n_targets, np.nan, dtype=float)

    def _solve_one_target(target_idx):
        total_col = annual_total[:, target_idx].copy()
        chw_days_col = annual_chw_days[:, target_idx]
        dyn_col = np.full(n_years, np.nan, dtype=float)
        adj_col = np.full(n_years, np.nan, dtype=float)
        r2_val = np.nan
        if not target_valid[target_idx]:
            return target_idx, dyn_col, adj_col, r2_val

        no_chw_years = np.isfinite(chw_days_col) & (chw_days_col == 0)
        dyn_col[no_chw_years] = 0.0
        adj_col[no_chw_years] = 0.0

        valid_rows = np.where(valid_year_mask_all[:, target_idx])[0]
        if valid_rows.size < YEARCHW_MIN_VALID_YEARS or not np.isfinite(alpha):
            return target_idx, dyn_col, adj_col, r2_val

        X_train = annual_x_list[target_idx][valid_rows, :]
        Y_train = annual_y_anom_all[valid_rows, target_idx]
        X_train_std, scaler = ns["daily_da_standardize_matrix"](X_train, mode=standardize_X)
        model = ns["Ridge"](alpha=float(alpha), fit_intercept=True)
        model.fit(X_train_std, Y_train)
        pred_mean_anom = model.predict(X_train_std)

        ss_res = float(np.sum((Y_train - pred_mean_anom) ** 2))
        ss_tot = float(np.sum((Y_train - np.mean(Y_train)) ** 2))
        if ss_tot > 0:
            r2_val = 1.0 - ss_res / ss_tot

        dyn_col[valid_rows] = pred_mean_anom * chw_days_col[valid_rows]
        adj_col[valid_rows] = total_col[valid_rows] - dyn_col[valid_rows]
        return target_idx, dyn_col, adj_col, r2_val

    n_jobs = max(1, min(TARGET_N_JOBS, n_targets))
    results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=1)(
        delayed(_solve_one_target)(target_idx) for target_idx in range(n_targets)
    )
    for target_idx, dyn_col, adj_col, r2_val in results:
        annual_dyn[:, target_idx] = dyn_col
        annual_adj[:, target_idx] = adj_col
        r2[target_idx] = r2_val

    return _pack_fit_result(
        annual_total,
        annual_dyn,
        annual_adj,
        annual_chw_days,
        r2,
        unique_years,
        y_stack,
        pred_box,
        alpha,
        alpha_score,
        "yearchw",
        None,
        standardize_X,
        train_sample_counts,
        display_sample_counts,
    )


def _pack_fit_result(
    annual_total,
    annual_dyn,
    annual_thermo,
    annual_chw_days,
    r2,
    unique_years,
    y_stack,
    pred_box,
    alpha,
    alpha_score,
    training_mode,
    top_n_days,
    standardize_X,
    train_sample_counts,
    display_sample_counts,
    time_values=None,
    daily_dyn_pred=None,
    daily_total_excess=None,
    daily_residual_fixed=None,
    daily_chw_mask=None,
):
    annual_total_da = xr.DataArray(
        annual_total,
        dims=("year", "target"),
        coords={"year": unique_years, "target": y_stack["target"]},
    ).unstack("target").transpose("year", "lat", "lon")
    annual_dyn_da = xr.DataArray(
        annual_dyn,
        dims=("year", "target"),
        coords={"year": unique_years, "target": y_stack["target"]},
    ).unstack("target").transpose("year", "lat", "lon")
    annual_thermo_da = xr.DataArray(
        annual_thermo,
        dims=("year", "target"),
        coords={"year": unique_years, "target": y_stack["target"]},
    ).unstack("target").transpose("year", "lat", "lon")
    annual_chw_days_da = xr.DataArray(
        annual_chw_days,
        dims=("year", "target"),
        coords={"year": unique_years, "target": y_stack["target"]},
    ).unstack("target").transpose("year", "lat", "lon")

    slope_total = xr.DataArray(_vector_slope(annual_total, unique_years), dims=("target",), coords={"target": y_stack["target"]}).unstack("target").transpose("lat", "lon")
    slope_dyn = xr.DataArray(_vector_slope(annual_dyn, unique_years), dims=("target",), coords={"target": y_stack["target"]}).unstack("target").transpose("lat", "lon")
    slope_thermo_direct = xr.DataArray(_vector_slope(annual_thermo, unique_years), dims=("target",), coords={"target": y_stack["target"]}).unstack("target").transpose("lat", "lon")
    r2_map = xr.DataArray(r2, dims=("target",), coords={"target": y_stack["target"]}).unstack("target").transpose("lat", "lon")

    slope_thermo = slope_thermo_direct
    slope_thermo_diff = (slope_total - slope_dyn) - slope_thermo_direct

    out = {
        "annual_total": annual_total_da,
        "annual_dyn": annual_dyn_da,
        "annual_thermo": annual_thermo_da,
        "annual_adj": annual_thermo_da,
        "annual_dyn_contrib_fixed": annual_dyn_da,
        "annual_residual_fixed": annual_thermo_da,
        "annual_chw_days": annual_chw_days_da,
        "slope_total": slope_total,
        "slope_dyn": slope_dyn,
        "slope_thermo": slope_thermo,
        "slope_dyn_contrib_fixed": slope_dyn,
        "slope_residual_fixed": slope_thermo_direct,
        "slope_thermo_direct": slope_thermo_direct,
        "slope_thermo_diff": slope_thermo_diff,
        "r2": r2_map,
        "r2_train": r2_map,
        "selected_alpha": float(alpha) if np.isfinite(alpha) else np.nan,
        "alpha_cv_score": float(alpha_score) if np.isfinite(alpha_score) else np.nan,
        "predictor_bounds_lon_min": float(pred_box[0]),
        "predictor_bounds_lon_max": float(pred_box[1]),
        "predictor_bounds_lat_min": float(pred_box[2]),
        "predictor_bounds_lat_max": float(pred_box[3]),
        "standardize_X": standardize_X,
        "training_mode": training_mode,
        "top_n_warm_days_per_year": int(top_n_days) if top_n_days is not None else np.nan,
        "train_sample_counts": train_sample_counts,
        "display_sample_counts": display_sample_counts,
        "years": unique_years,
    }
    if time_values is not None:
        def _daily_to_da(values):
            return (
                xr.DataArray(
                    values,
                    dims=("time", "target"),
                    coords={"time": time_values, "target": y_stack["target"]},
                )
                .unstack("target")
                .transpose("time", "lat", "lon")
            )

        if daily_dyn_pred is not None:
            out["daily_dyn_pred"] = _daily_to_da(daily_dyn_pred)
            out["daily_dyn_raw"] = out["daily_dyn_pred"]
        if daily_total_excess is not None:
            out["daily_total_excess"] = _daily_to_da(daily_total_excess)
            out["daily_total_raw"] = out["daily_total_excess"]
        if daily_residual_fixed is not None:
            out["daily_residual_fixed"] = _daily_to_da(daily_residual_fixed)
            out["daily_thermo_raw"] = out["daily_residual_fixed"]
        if daily_chw_mask is not None:
            out["daily_chw_mask"] = _daily_to_da(daily_chw_mask)
    return out


def _fit_region_by_mode(ns, z500_rel, y_region, chw_region, threshold_region, training_mode, top_n_days, max_targets_for_cv):
    if training_mode == "yearchw":
        return _fit_gridwise_cumheat_yearchw_region(
            ns,
            z500_rel,
            y_region,
            chw_region,
            threshold_region,
            predictor_expand_deg=15.0,
            standardize_X="zscore",
            cv_year_folds=5,
            max_targets_for_cv=max_targets_for_cv,
            random_state=0,
        )
    if training_mode == "chwcalendar_window":
        return _fit_gridwise_cumheat_calendar_window_region(
            ns,
            z500_rel,
            y_region,
            chw_region,
            threshold_region,
            predictor_expand_deg=15.0,
            standardize_X="zscore",
            cv_year_folds=5,
            max_targets_for_cv=max_targets_for_cv,
            random_state=0,
            half_window=15,
        )
    return _fit_gridwise_cumheat_daily_region(
        ns,
        z500_rel,
        y_region,
        chw_region,
        threshold_region,
        training_mode=training_mode,
        top_n_days=top_n_days,
        predictor_expand_deg=15.0,
        standardize_X="zscore",
        cv_year_folds=5,
        max_targets_for_cv=max_targets_for_cv,
        random_state=0,
    )


def _compute_rows_spatial_and_annual(
    ns,
    paths,
    temp_dict,
    z500_rel,
    chw_mask,
    threshold_dict,
    template_2d,
    training_mode,
    top_n_days,
    member_name=None,
    max_targets_for_cv=400,
):
    rows = []
    spatial_out = {}
    annual_out = {}
    chw_days_region_list = []
    regions = ns["daily_da_get_region_settings"]()
    owner = "ERA5" if member_name is None else str(member_name)

    for var_name, temp_da in temp_dict.items():
        total_list, dyn_list, thermo_list = [], [], []
        thermo_direct_list, thermo_diff_list = [], []
        annual_total_list, annual_dyn_list, annual_thermo_list = [], [], []
        annual_chw_days_list = []

        for region_name, box in regions:
            print(f"[{paths['tag']}] {owner} | {var_name} | {region_name} 开始计算累计热量", flush=True)
            y_region = ns["daily_da_select_box_wrap"](temp_da, box)
            chw_region = ns["daily_da_select_box_wrap"](chw_mask, box)
            threshold_region = ns["daily_da_select_box_wrap"](threshold_dict[var_name], box)
            fit_result = _fit_region_by_mode(
                ns,
                z500_rel,
                y_region,
                chw_region,
                threshold_region,
                training_mode=training_mode,
                top_n_days=top_n_days,
                max_targets_for_cv=max_targets_for_cv,
            )
            daily_cache_path = _write_daily_region_dataset(paths, owner, var_name, region_name, fit_result)

            slope_total_reg = fit_result["slope_total"]
            slope_dyn_reg = fit_result["slope_dyn"]
            slope_thermo_reg = fit_result["slope_thermo"]
            slope_thermo_direct_reg = fit_result["slope_thermo_direct"]
            slope_thermo_diff_reg = fit_result["slope_thermo_diff"]
            annual_total_reg = fit_result["annual_total"]
            annual_dyn_reg = fit_result["annual_dyn"]
            annual_thermo_reg = fit_result["annual_thermo"]
            annual_chw_days_reg = fit_result["annual_chw_days"]
            thermo_diff_abs_max = (
                float(np.nanmax(np.abs(slope_thermo_diff_reg.values)))
                if np.isfinite(slope_thermo_diff_reg.values).any()
                else np.nan
            )

            total_list.append(ns["daily_da_put_on_template"](slope_total_reg, template_2d).astype("float32").expand_dims(region=[region_name]))
            dyn_list.append(ns["daily_da_put_on_template"](slope_dyn_reg, template_2d).astype("float32").expand_dims(region=[region_name]))
            thermo_list.append(ns["daily_da_put_on_template"](slope_thermo_reg, template_2d).astype("float32").expand_dims(region=[region_name]))
            thermo_direct_list.append(ns["daily_da_put_on_template"](slope_thermo_direct_reg, template_2d).astype("float32").expand_dims(region=[region_name]))
            thermo_diff_list.append(ns["daily_da_put_on_template"](slope_thermo_diff_reg, template_2d).astype("float32").expand_dims(region=[region_name]))

            annual_total_list.append(_put_annual_on_template(annual_total_reg, template_2d).expand_dims(region=[region_name]))
            annual_dyn_list.append(_put_annual_on_template(annual_dyn_reg, template_2d).expand_dims(region=[region_name]))
            annual_thermo_list.append(_put_annual_on_template(annual_thermo_reg, template_2d).expand_dims(region=[region_name]))
            annual_chw_days_list.append(_put_annual_on_template(annual_chw_days_reg, template_2d).expand_dims(region=[region_name]))

            slope_total_map = float(ns["daily_da_area_mean"](slope_total_reg).values)
            slope_dyn_map = float(ns["daily_da_area_mean"](slope_dyn_reg).values)
            slope_thermo_map = float(ns["daily_da_area_mean"](slope_thermo_reg).values)
            y_total_ts = ns["daily_da_area_mean"](annual_total_reg)
            y_dyn_ts = ns["daily_da_area_mean"](annual_dyn_reg)
            y_thermo_ts = ns["daily_da_area_mean"](annual_thermo_reg)
            tr_total = ns["daily_da_series_trend"](y_total_ts)
            tr_dyn = ns["daily_da_series_trend"](y_dyn_ts)
            tr_thermo = ns["daily_da_series_trend"](y_thermo_ts)

            row = {
                "var": var_name,
                "region": region_name,
                "lon_min": box[0],
                "lon_max": box[1],
                "lat_min": box[2],
                "lat_max": box[3],
                "training_mode": training_mode,
                "scheme_label": paths["scheme_label"],
                "alpha": float(fit_result.get("selected_alpha", np.nan)),
                "top_n_warm_days_per_year": fit_result.get("top_n_warm_days_per_year", np.nan),
                "n_train_samples_mean": float(np.nanmean(fit_result["train_sample_counts"])),
                "n_train_samples_min": int(np.nanmin(fit_result["train_sample_counts"])),
                "n_train_samples_max": int(np.nanmax(fit_result["train_sample_counts"])),
                "n_chw_display_samples_mean": float(np.nanmean(fit_result["display_sample_counts"])),
                "n_chw_display_samples_min": int(np.nanmin(fit_result["display_sample_counts"])),
                "n_chw_display_samples_max": int(np.nanmax(fit_result["display_sample_counts"])),
                "total_slope_per_year": slope_total_map,
                "total_slope_per_decade": slope_total_map * 10.0,
                "dyn_slope_per_year": slope_dyn_map,
                "dyn_slope_per_decade": slope_dyn_map * 10.0,
                "thermo_slope_per_year": slope_thermo_map,
                "thermo_slope_per_decade": slope_thermo_map * 10.0,
                "dyn_contrib_fixed_slope_per_year": slope_dyn_map,
                "dyn_contrib_fixed_slope_per_decade": slope_dyn_map * 10.0,
                "residual_fixed_slope_per_year": slope_thermo_map,
                "residual_fixed_slope_per_decade": slope_thermo_map * 10.0,
                "total_slope_ts_per_year": tr_total["slope"],
                "dyn_slope_ts_per_year": tr_dyn["slope"],
                "thermo_slope_ts_per_year": tr_thermo["slope"],
                "dyn_contrib_fixed_slope_ts_per_year": tr_dyn["slope"],
                "residual_fixed_slope_ts_per_year": tr_thermo["slope"],
                "total_pvalue_ts": tr_total["pvalue"],
                "dyn_pvalue_ts": tr_dyn["pvalue"],
                "thermo_pvalue_ts": tr_thermo["pvalue"],
                "dyn_contrib_fixed_pvalue_ts": tr_dyn["pvalue"],
                "residual_fixed_pvalue_ts": tr_thermo["pvalue"],
                "total_delta_map_minus_ts_per_year": slope_total_map - tr_total["slope"],
                "dyn_delta_map_minus_ts_per_year": slope_dyn_map - tr_dyn["slope"],
                "thermo_delta_map_minus_ts_per_year": slope_thermo_map - tr_thermo["slope"],
                "dyn_contrib_fixed_delta_map_minus_ts_per_year": slope_dyn_map - tr_dyn["slope"],
                "residual_fixed_delta_map_minus_ts_per_year": slope_thermo_map - tr_thermo["slope"],
                "thermo_diff_abs_max_per_year": thermo_diff_abs_max,
                "predictor_bounds_lat_min": float(fit_result.get("predictor_bounds_lat_min", np.nan)),
                "predictor_bounds_lat_max": float(fit_result.get("predictor_bounds_lat_max", np.nan)),
                "predictor_bounds_lon_min": float(fit_result.get("predictor_bounds_lon_min", np.nan)),
                "predictor_bounds_lon_max": float(fit_result.get("predictor_bounds_lon_max", np.nan)),
                "standardize_X": fit_result.get("standardize_X", ""),
                "alpha_cv_score": float(fit_result.get("alpha_cv_score", np.nan)),
                "unit": "degC_days_per_year",
                "daily_cache_path": "" if daily_cache_path is None else daily_cache_path,
            }
            if member_name is not None:
                row["member"] = str(member_name)
            rows.append(row)
            print(f"[{paths['tag']}] {owner} | {var_name} | {region_name} 完成", flush=True)

        spatial_out[f"{var_name}_hwc_total_slope_per_year"] = xr.concat(total_list, dim="region")
        spatial_out[f"{var_name}_hwc_dyn_slope_per_year"] = xr.concat(dyn_list, dim="region")
        spatial_out[f"{var_name}_hwc_thermo_slope_per_year"] = xr.concat(thermo_list, dim="region")
        spatial_out[f"{var_name}_hwc_dyn_contrib_fixed_slope_per_year"] = xr.concat(dyn_list, dim="region")
        spatial_out[f"{var_name}_hwc_residual_fixed_slope_per_year"] = xr.concat(thermo_list, dim="region")
        spatial_out[f"{var_name}_hwc_thermo_direct_slope_per_year"] = xr.concat(thermo_direct_list, dim="region")
        spatial_out[f"{var_name}_hwc_thermo_diff_slope_per_year"] = xr.concat(thermo_diff_list, dim="region")

        annual_total_da = xr.concat(annual_total_list, dim="region").transpose("year", "region", "lat", "lon").astype("float32")
        annual_dyn_da = xr.concat(annual_dyn_list, dim="region").transpose("year", "region", "lat", "lon").astype("float32")
        annual_thermo_da = xr.concat(annual_thermo_list, dim="region").transpose("year", "region", "lat", "lon").astype("float32")
        annual_chw_days_da = xr.concat(annual_chw_days_list, dim="region").transpose("year", "region", "lat", "lon").astype("float32")
        annual_out[f"{var_name}_hwc_annual_total"] = annual_total_da
        annual_out[f"{var_name}_hwc_annual_dyn"] = annual_dyn_da
        annual_out[f"{var_name}_hwc_annual_thermo"] = annual_thermo_da
        annual_out[f"{var_name}_hwc_annual_adj"] = annual_thermo_da
        annual_out[f"{var_name}_hwc_annual_dyn_contrib_fixed"] = annual_dyn_da
        annual_out[f"{var_name}_hwc_annual_residual_fixed"] = annual_thermo_da
        if var_name == "tmax":
            chw_days_region_list = annual_chw_days_list

    if chw_days_region_list:
        annual_out["chw_days"] = xr.concat(chw_days_region_list, dim="region").transpose("year", "region", "lat", "lon").astype("float32")
    return rows, spatial_out, annual_out


def _compute_era5_variant(ns, paths, force=False):
    outputs_exist = all(os.path.exists(paths[key]) for key in ("era5_region", "era5_spatial", "era5_annual"))
    if outputs_exist and not force:
        print(f"ERA5 {paths['tag']} 缓存已存在，跳过重算。", flush=True)
        return {"skipped": True, "region_csv": paths["era5_region"], "spatial_nc": paths["era5_spatial"], "annual_nc": paths["era5_annual"]}

    tmax, tmin = ns["daily_da_open_era5_daily_temps"]()
    z500_rel = ns["daily_da_remove_nh_mean"](ns["daily_da_open_era5_daily_z500"]())
    chw_mask = CHW_MOD._open_era5_chw_mask(ns)
    thresholds = _open_era5_thresholds(ns)
    template = xr.full_like(tmax.isel(time=0, drop=True), np.nan)
    rows, spatial_out, annual_out = _compute_rows_spatial_and_annual(
        ns,
        paths,
        {"tmax": tmax, "tmin": tmin},
        z500_rel,
        chw_mask,
        thresholds,
        template,
        training_mode=paths["training_mode"],
        top_n_days=paths["top_n"],
        member_name=None,
        max_targets_for_cv=24,
    )
    pd.DataFrame(rows).to_csv(paths["era5_region"], index=False)
    _write_dataset(spatial_out, paths["era5_spatial"])
    _write_dataset(annual_out, paths["era5_annual"])
    print(f"已保存 ERA5 {paths['tag']} 累计热量结果。", flush=True)
    return {"skipped": False, "n_rows": len(rows), "region_csv": paths["era5_region"], "spatial_nc": paths["era5_spatial"], "annual_nc": paths["era5_annual"]}


def _compute_member_cache_variant(ns, paths, member, source_table=None, force=False):
    region_path = _member_region_path(paths, member)
    spatial_path = _member_spatial_path(paths, member)
    annual_path = _member_annual_path(paths, member)
    outputs_exist = os.path.exists(region_path) and os.path.exists(spatial_path) and os.path.exists(annual_path)
    if outputs_exist and not force:
        print(f"{member} {paths['tag']} 缓存已存在，跳过重算。", flush=True)
        return {"member": member, "skipped": True, "region_csv": region_path, "spatial_nc": spatial_path, "annual_nc": annual_path}

    tmax = ns["daily_da_open_cmip_daily_temperature"](member, "tasmax")
    tmin = ns["daily_da_open_cmip_daily_temperature"](member, "tasmin")
    z500_rel = ns["daily_da_remove_nh_mean"](ns["daily_da_open_cmip_daily_z500"](member, source_table=source_table))
    chw_mask = CHW_MOD._open_member_chw_mask(ns, member)
    thresholds = _open_member_thresholds(ns, member)
    template = xr.full_like(tmax.isel(time=0, drop=True), np.nan)
    rows, spatial_out, annual_out = _compute_rows_spatial_and_annual(
        ns,
        paths,
        {"tmax": tmax, "tmin": tmin},
        z500_rel,
        chw_mask,
        thresholds,
        template,
        training_mode=paths["training_mode"],
        top_n_days=paths["top_n"],
        member_name=member,
        max_targets_for_cv=16,
    )
    pd.DataFrame(rows).to_csv(region_path, index=False)
    _write_dataset(spatial_out, spatial_path)
    _write_dataset(annual_out, annual_path)
    print(f"已保存 {member} {paths['tag']} 成员累计热量缓存。", flush=True)
    return {"member": member, "skipped": False, "region_csv": region_path, "spatial_nc": spatial_path, "annual_nc": annual_path}


def _aggregate_member_caches_variant(paths, model_list=None):
    if model_list is None:
        raise ValueError("汇总成员缓存时必须提供成员列表。")

    region_frames = []
    spatial_list = []
    annual_list = []
    valid_members = []
    for member in model_list:
        region_path = _member_region_path(paths, member)
        spatial_path = _member_spatial_path(paths, member)
        annual_path = _member_annual_path(paths, member)
        if os.path.exists(region_path) and os.path.exists(spatial_path) and os.path.exists(annual_path):
            df = pd.read_csv(region_path)
            if "member" not in df.columns:
                df["member"] = member
            region_frames.append(df)
            spatial_list.append(xr.open_dataset(spatial_path).expand_dims(member=[member]))
            annual_list.append(xr.open_dataset(annual_path).expand_dims(member=[member]))
            valid_members.append(member)

    if not region_frames:
        raise FileNotFoundError(f"当前没有可汇总的 {paths['tag']} 成员缓存。")

    df_members = pd.concat(region_frames, ignore_index=True)
    df_members.to_csv(paths["members_region"], index=False)
    df_mme = df_members.groupby(["var", "region"], as_index=False).mean(numeric_only=True)
    df_mme["training_mode"] = paths["training_mode"]
    df_mme["scheme_label"] = paths["scheme_label"]
    df_mme["top_n_warm_days_per_year"] = np.nan if paths["top_n"] is None else int(paths["top_n"])
    df_mme["unit"] = "degC_days_per_year"
    df_mme.to_csv(paths["mme_region"], index=False)

    members_spatial_ds = xr.concat(spatial_list, dim="member")
    _write_dataset(members_spatial_ds, paths["members_spatial"])
    mme_spatial_ds = members_spatial_ds.mean("member", skipna=True)
    _write_dataset(mme_spatial_ds, paths["mme_spatial"])

    members_annual_ds = xr.concat(annual_list, dim="member")
    _write_dataset(members_annual_ds, paths["members_annual"])
    mme_annual_ds = members_annual_ds.mean("member", skipna=True)
    _write_dataset(mme_annual_ds, paths["mme_annual"])

    print(f"已完成 {paths['tag']} 成员汇总、MME 汇总和逐年网格场汇总。", flush=True)
    return {"valid_members": valid_members, "n_members": len(valid_members)}


def _paths_for_existing_tag(paths, source_tag):
    out_dir = Path(paths["out_dir"])
    return {
        "era5_region": str(out_dir / f"da_ridge_daily_{source_tag}_region_trends_ERA5_1981_2014_JJA.csv"),
        "era5_spatial": str(out_dir / f"da_ridge_daily_{source_tag}_spatial_trends_ERA5_JJA_1981_2014.nc"),
        "era5_annual": str(out_dir / f"da_ridge_daily_{source_tag}_annual_fields_ERA5_JJA_1981_2014.nc"),
        "members_region": str(out_dir / f"da_ridge_daily_{source_tag}_region_trends_CMIP_members_JJA_1981_2014.csv"),
        "mme_region": str(out_dir / f"da_ridge_daily_{source_tag}_region_trends_MME_JJA_1981_2014.csv"),
        "members_spatial": str(out_dir / f"da_ridge_daily_{source_tag}_spatial_trends_CMIP_members_JJA_1981_2014.nc"),
        "mme_spatial": str(out_dir / f"da_ridge_daily_{source_tag}_spatial_trends_MME_JJA_1981_2014.nc"),
        "members_annual": str(out_dir / f"da_ridge_daily_{source_tag}_annual_fields_CMIP_members_JJA_1981_2014.nc"),
        "mme_annual": str(out_dir / f"da_ridge_daily_{source_tag}_annual_fields_MME_JJA_1981_2014.nc"),
    }


def _copy_csv_with_scheme(src, dst, paths, force=False):
    if os.path.exists(dst) and not force:
        return False
    df = pd.read_csv(src)
    df["training_mode"] = paths["training_mode"]
    df["scheme_label"] = paths["scheme_label"]
    df["top_n_warm_days_per_year"] = np.nan if paths["top_n"] is None else int(paths["top_n"])
    df.to_csv(dst, index=False)
    return True


def _reuse_from_existing_tag(paths, force=False):
    source_tag = paths.get("reuse_from_tag")
    if not source_tag:
        return None
    source_paths = _paths_for_existing_tag(paths, source_tag)
    required = list(source_paths.values())
    if not all(os.path.exists(path) for path in required):
        print(f"{paths['tag']} 未找到完整可复用的 {source_tag} 输出，将改为正常计算。", flush=True)
        return None

    csv_keys = ["era5_region", "members_region", "mme_region"]
    nc_keys = ["era5_spatial", "era5_annual", "members_spatial", "mme_spatial", "members_annual", "mme_annual"]
    changed = []
    for key in csv_keys:
        if _copy_csv_with_scheme(source_paths[key], paths[key], paths, force=force):
            changed.append(key)
    for key in nc_keys:
        if os.path.exists(paths[key]) and not force:
            continue
        shutil.copy2(source_paths[key], paths[key])
        changed.append(key)
    print(f"{paths['tag']} 已复用 {source_tag} 输出。", flush=True)
    return {"source_tag": source_tag, "changed": changed}


def compute_variant(
    ns,
    spec,
    run_era5=True,
    run_cmip=True,
    force_era5=False,
    force_members=False,
    force_summary=False,
    member_subset=None,
    skip_summary=False,
):
    paths = _build_variant_paths(spec)
    source_table = ns["daily_da_build_member_source_table"](output_csv=paths["member_sources"], force=force_summary)
    available_members = source_table.loc[source_table["available"], "member"].astype(str).tolist()
    missing_members = source_table.loc[~source_table["available"], "member"].astype(str).tolist()
    if member_subset is not None:
        available_members = [m for m in available_members if m in member_subset]

    out = {
        "tag": paths["tag"],
        "training_mode": paths["training_mode"],
        "top_n": paths["top_n"],
        "scheme_label": paths["scheme_label"],
        "available_members": available_members,
        "missing_members": missing_members,
        "paths": paths,
    }
    reused = _reuse_from_existing_tag(paths, force=(force_era5 or force_members or force_summary))
    if reused is not None:
        out["reused"] = reused
        return out
    if run_era5:
        out["era5"] = _compute_era5_variant(ns, paths, force=force_era5)
    if run_cmip:
        member_results = []
        for member in available_members:
            print(f"[{paths['tag']}] 成员开始：{member}", flush=True)
            member_results.append(_compute_member_cache_variant(ns, paths, member, source_table=source_table, force=force_members))
            print(f"[{paths['tag']}] 成员完成：{member}", flush=True)
        out["member_results"] = member_results

    need_summary = False
    if not skip_summary:
        need_summary = (
            force_summary
            or run_cmip
            or (not os.path.exists(paths["members_region"]))
            or (not os.path.exists(paths["mme_region"]))
            or (not os.path.exists(paths["members_spatial"]))
            or (not os.path.exists(paths["mme_spatial"]))
            or (not os.path.exists(paths["members_annual"]))
            or (not os.path.exists(paths["mme_annual"]))
        )
    if need_summary and available_members:
        out["summary"] = _aggregate_member_caches_variant(paths, model_list=available_members)
    return out


def smoke_test_variant(ns, spec, member_name=None):
    paths = _build_variant_paths(spec)
    source_table = ns["daily_da_build_member_source_table"](output_csv=paths["member_sources"], force=False)
    available_members = source_table.loc[source_table["available"], "member"].astype(str).tolist()
    if member_name is None:
        member_name = available_members[0]
    test_region_name, test_box = ns["daily_da_get_region_settings"]()[0]

    tmax_era, _ = ns["daily_da_open_era5_daily_temps"]()
    z_era = ns["daily_da_remove_nh_mean"](ns["daily_da_open_era5_daily_z500"]())
    chw_era = CHW_MOD._open_era5_chw_mask(ns)
    th_era = _open_era5_thresholds(ns)["tmax"]
    fit_era = _fit_region_by_mode(
        ns,
        z_era,
        ns["daily_da_select_box_wrap"](tmax_era, test_box),
        ns["daily_da_select_box_wrap"](chw_era, test_box),
        ns["daily_da_select_box_wrap"](th_era, test_box),
        training_mode=paths["training_mode"],
        top_n_days=paths["top_n"],
        max_targets_for_cv=12,
    )

    t_test = ns["daily_da_open_cmip_daily_temperature"](member_name, "tasmax")
    z_test = ns["daily_da_remove_nh_mean"](ns["daily_da_open_cmip_daily_z500"](member_name, source_table=source_table))
    chw_test = CHW_MOD._open_member_chw_mask(ns, member_name)
    th_test = _open_member_thresholds(ns, member_name)["tmax"]
    fit_test = _fit_region_by_mode(
        ns,
        z_test,
        ns["daily_da_select_box_wrap"](t_test, test_box),
        ns["daily_da_select_box_wrap"](chw_test, test_box),
        ns["daily_da_select_box_wrap"](th_test, test_box),
        training_mode=paths["training_mode"],
        top_n_days=paths["top_n"],
        max_targets_for_cv=12,
    )

    era_total = fit_era["annual_total"].values
    era_dyn = fit_era["annual_dyn"].values
    era_thermo = fit_era["annual_thermo"].values
    residual_max = float(np.nanmax(np.abs((era_total - era_dyn) - era_thermo))) if np.isfinite(era_total).any() else np.nan

    return {
        "tag": paths["tag"],
        "training_mode": paths["training_mode"],
        "top_n": paths["top_n"],
        "test_member": member_name,
        "test_region": test_region_name,
        "era5_threshold_mean_c": float(th_era.mean(skipna=True).values),
        "member_threshold_mean_c": float(th_test.mean(skipna=True).values),
        "era5_train_samples_min": int(np.nanmin(fit_era["train_sample_counts"])),
        "era5_train_samples_max": int(np.nanmax(fit_era["train_sample_counts"])),
        "era5_chw_samples_min": int(np.nanmin(fit_era["display_sample_counts"])),
        "era5_chw_samples_max": int(np.nanmax(fit_era["display_sample_counts"])),
        "member_train_samples_min": int(np.nanmin(fit_test["train_sample_counts"])),
        "member_train_samples_max": int(np.nanmax(fit_test["train_sample_counts"])),
        "member_chw_samples_min": int(np.nanmin(fit_test["display_sample_counts"])),
        "member_chw_samples_max": int(np.nanmax(fit_test["display_sample_counts"])),
        "era5_alpha": float(fit_era.get("selected_alpha", np.nan)),
        "member_alpha": float(fit_test.get("selected_alpha", np.nan)),
        "era5_thermo_residual_abs_max": residual_max,
    }


def main():
    ns = _load_daily_namespace()
    specs = _variant_specs()
    run_era5 = _env_flag("DA_RUN_ERA5", default=True)
    run_cmip = _env_flag("DA_RUN_CMIP", default=False)
    force_era5 = _env_flag("DA_FORCE_ERA5", default=False)
    force_members = _env_flag("DA_FORCE_MEMBERS", default=False)
    force_summary = _env_flag("DA_FORCE_SUMMARY", default=False)
    skip_summary = _env_flag("DA_SKIP_SUMMARY", default=False)
    smoke_only = _env_flag("DA_SMOKE_ONLY", default=False)
    member_subset = _env_list("DA_MEMBER_SUBSET", default=None) or None

    print(
        {
            "base_dir": _result_base_dir(),
            "variant_specs": specs,
            "run_era5": run_era5,
            "run_cmip": run_cmip,
            "force_era5": force_era5,
            "force_members": force_members,
            "force_summary": force_summary,
            "skip_summary": skip_summary,
            "smoke_only": smoke_only,
            "member_subset": member_subset,
            "yearchw_min_valid_years": YEARCHW_MIN_VALID_YEARS,
        },
        flush=True,
    )

    outputs = []
    for spec in specs:
        print(f"开始处理 {spec['tag']} 版本。", flush=True)
        if smoke_only:
            outputs.append(smoke_test_variant(ns, spec, member_name=member_subset[0] if member_subset else None))
        else:
            outputs.append(
                compute_variant(
                    ns,
                    spec,
                    run_era5=run_era5,
                    run_cmip=run_cmip,
                    force_era5=force_era5,
                    force_members=force_members,
                    force_summary=force_summary,
                    member_subset=member_subset,
                    skip_summary=skip_summary,
                )
            )
    print("全部完成。", flush=True)
    print(outputs, flush=True)


if __name__ == "__main__":
    main()
