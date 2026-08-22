import json
import os
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
DEFAULT_BASE_DIR = str(CACHE_ROOT / "dynamic_adjustment" / "chwonly")
ERA5_CHW_FILE = str(DATA_ROOT / "heatwaves_files" / "ERA5_heatwave_3.nc")
MODEL_CHW_ROOT = DATA_ROOT / "heatwaves_files" / "1x1" / "model"
MODEL_CHW_ROOT_FALLBACK = DATA_ROOT / "heatwaves_files" / "1x1"
TARGET_N_JOBS = max(1, int(os.environ.get("DA_TARGET_NJOBS", min(os.cpu_count() or 1, 24))))


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


def _load_daily_namespace():
    # 公开版直接导入已提取的日尺度动力调整函数，不再读取大型 notebook。
    import daily_da_core as core
    return {name: value for name, value in vars(core).items() if not name.startswith("__")}


def _result_base_dir():
    return os.environ.get("DA_BASE_DIR", DEFAULT_BASE_DIR)


def _variant_specs():
    default_tags = ["chwonly", "hottest15_chwonly", "hottest30_chwonly"]
    tags = [tag.lower() for tag in _env_list("DA_VARIANT_TAGS", default=default_tags)]
    mapping = {
        "chwonly": {"tag": "chwonly", "training_mode": "all_jja", "top_n": None},
        "hottest15_chwonly": {"tag": "hottest15_chwonly", "training_mode": "topn", "top_n": 15},
        "hottest30_chwonly": {"tag": "hottest30_chwonly", "training_mode": "topn", "top_n": 30},
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
    if tag == "chwonly":
        member_cache_dir = out_dir / "member_cache"
    else:
        member_cache_dir = out_dir / f"{tag}_member_cache"
    member_cache_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base_dir": str(base_dir),
        "out_dir": str(out_dir),
        "fig_dir": str(fig_dir),
        "tag": tag,
        "training_mode": spec["training_mode"],
        "top_n": spec["top_n"],
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
    }


def _member_region_path(paths, member):
    return os.path.join(paths["member_cache_dir"], f"{member}_region.csv")


def _member_spatial_path(paths, member):
    return os.path.join(paths["member_cache_dir"], f"{member}_spatial.nc")


def _member_annual_path(paths, member):
    return os.path.join(paths["member_cache_dir"], f"{member}_annual.nc")


def _build_year_indexer(time_values):
    years = pd.DatetimeIndex(time_values).year.astype(int)
    unique_years = np.unique(years)
    year_row_index = [np.where(years == year)[0] for year in unique_years]
    return years, unique_years.astype(int), year_row_index


def _build_gridwise_all_jja_mask(y_total_values):
    mask = np.isfinite(y_total_values)
    sample_counts = mask.sum(axis=0).astype(int)
    return mask, sample_counts


def _build_gridwise_topn_mask(y_total_values, year_row_index, top_n_days):
    n_time, n_target = y_total_values.shape
    mask = np.zeros((n_time, n_target), dtype=bool)
    sample_counts = np.zeros(n_target, dtype=int)

    for target_idx in range(n_target):
        col = y_total_values[:, target_idx]
        for year_rows in year_row_index:
            year_vals = col[year_rows]
            finite_local = np.where(np.isfinite(year_vals))[0]
            if finite_local.size == 0:
                continue
            pick_n = min(int(top_n_days), int(finite_local.size))
            finite_rows = year_rows[finite_local]
            if pick_n == finite_local.size:
                chosen_rows = finite_rows
            else:
                top_local = np.argpartition(year_vals[finite_local], -pick_n)[-pick_n:]
                chosen_rows = finite_rows[top_local]
            mask[np.sort(chosen_rows), target_idx] = True
        sample_counts[target_idx] = int(mask[:, target_idx].sum())

    return mask, sample_counts


def _vector_slope(values, years, min_count=5):
    arr = np.asarray(values, dtype=float)
    years = np.asarray(years, dtype=float)
    x = years[:, None]
    mask = np.isfinite(arr)
    n = mask.sum(axis=0)
    xsum = np.where(mask, x, 0.0).sum(axis=0)
    ysum = np.where(mask, arr, 0.0).sum(axis=0)
    xmean = np.divide(xsum, n, out=np.zeros_like(xsum, dtype=float), where=n > 0)
    ymean = np.divide(ysum, n, out=np.zeros_like(ysum, dtype=float), where=n > 0)
    x_centered = x - xmean
    y_centered = np.where(mask, arr - ymean, 0.0)
    cov = np.where(mask, x_centered * y_centered, 0.0).sum(axis=0)
    var = np.where(mask, x_centered * x_centered, 0.0).sum(axis=0)
    slope = np.divide(cov, var, out=np.full(arr.shape[1], np.nan, dtype=float), where=(var > 0) & (n >= min_count))
    slope = np.where(n >= min_count, slope, np.nan)
    return slope


def _choose_shared_alpha_gridwise(ns, X_full, Y_full, train_mask, years, standardize_X="zscore", alphas=None, cv_year_folds=5, max_targets_for_cv=24, random_state=0):
    if alphas is None:
        alphas = np.logspace(-2, 4, 13)

    valid_targets = []
    for target_idx in range(train_mask.shape[1]):
        rows = np.where(train_mask[:, target_idx] & np.isfinite(Y_full[:, target_idx]))[0]
        if rows.size == 0:
            continue
        unique_groups = np.unique(years[rows])
        if unique_groups.size >= 2:
            valid_targets.append(target_idx)

    if not valid_targets:
        raise ValueError("没有可用于共享 alpha 交叉验证的目标格点。")

    rng = np.random.RandomState(random_state)
    if len(valid_targets) > max_targets_for_cv:
        valid_targets = sorted(rng.choice(valid_targets, size=max_targets_for_cv, replace=False).tolist())

    best_alpha = float(alphas[0])
    best_score = np.inf
    for alpha in alphas:
        fold_scores = []
        for target_idx in valid_targets:
            rows = np.where(train_mask[:, target_idx] & np.isfinite(Y_full[:, target_idx]))[0]
            groups = years[rows]
            unique_groups = np.unique(groups)
            n_splits = min(int(cv_year_folds), int(unique_groups.size))
            if n_splits < 2:
                continue
            X_target = X_full[rows, :]
            Y_target = Y_full[rows, target_idx]
            X_std, _ = ns["daily_da_standardize_matrix"](X_target, mode=standardize_X)
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

    return float(best_alpha), float(best_score)


def _candidate_member_chw_paths(member):
    primary = MODEL_CHW_ROOT / f"{member}_heatwave_3.nc"
    fallback = MODEL_CHW_ROOT_FALLBACK / f"{member}_heatwave_3.nc"
    candidates = [primary, fallback]
    extra = sorted(MODEL_CHW_ROOT.rglob(f"{member}_heatwave_3.nc"))
    for path in extra:
        if path not in candidates:
            candidates.append(path)
    return candidates


def _locate_member_chw_file(member):
    for path in _candidate_member_chw_paths(member):
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"未找到 {member} 的逐日复合型热浪文件。")


def _year_day_to_daily_time(da):
    if "year" not in da.dims or "day" not in da.dims:
        raise ValueError("热浪文件必须包含 year 和 day 维度。")
    stacked = da.stack(time=("year", "day")).transpose("time", "lat", "lon")
    multi_index = stacked["time"].to_index()
    years = multi_index.get_level_values("year").astype(int)
    days = multi_index.get_level_values("day").astype(int)
    times = pd.DatetimeIndex(
        [pd.Timestamp(year=int(year), month=6, day=1) + pd.Timedelta(days=int(day) - 1) for year, day in zip(years, days)],
        name="time",
    )
    return stacked.assign_coords(time=times)


def _open_era5_chw_mask(ns):
    ds = xr.open_dataset(ERA5_CHW_FILE).sel(lat=slice(0, 90))
    ds = ns["daily_da_filp_lon"](ds, lon_name="lon")
    da = ds["var"].sel(year=slice(1981, 2014))
    da = ns["daily_da_mask_landsea"](da, lat_name="lat", label="ocean")
    if "mask" in da.coords:
        da = da.reset_coords("mask", drop=True)
    da = _year_day_to_daily_time(da)
    da = ns["daily_da_standardize_coords"](da)
    da = ns["daily_da_select_jja"](da)
    return da.fillna(0.0)


def _open_member_chw_mask(ns, member):
    ds = xr.open_dataset(_locate_member_chw_file(member)).sel(lat=slice(0, 90))
    ds = ns["daily_da_filp_lon"](ds, lon_name="lon")
    da = ds["var"].sel(year=slice(1981, 2014))
    da = ns["daily_da_mask_landsea"](da, lat_name="lat", label="ocean")
    if "mask" in da.coords:
        da = da.reset_coords("mask", drop=True)
    da = _year_day_to_daily_time(da)
    da = ns["daily_da_standardize_coords"](da)
    da = ns["daily_da_select_jja"](da)
    return da.fillna(0.0)


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


def _fit_gridwise_conditioned_region(
    ns,
    z500_rel,
    y_region,
    chw_region,
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

    z500_rel = ns["daily_da_standardize_coords"](z500_rel)
    y_region = ns["daily_da_standardize_coords"](y_region)
    chw_region = ns["daily_da_standardize_coords"](chw_region)
    if "mask" in y_region.coords:
        y_region = y_region.reset_coords("mask", drop=True)
    if "mask" in chw_region.coords:
        chw_region = chw_region.reset_coords("mask", drop=True)

    z_sel = ns["daily_da_select_jja"](z500_rel)
    y_sel_full = ns["daily_da_select_jja"](y_region)
    chw_sel_full = ns["daily_da_select_jja"](chw_region)
    z_sel, y_sel_full, chw_sel_full = xr.align(z_sel, y_sel_full, chw_sel_full, join="inner")

    y_box = (
        float(y_sel_full["lon"].min()),
        float(y_sel_full["lon"].max()),
        float(y_sel_full["lat"].min()),
        float(y_sel_full["lat"].max()),
    )
    pred_box = ns["daily_da_expand_box"](y_box, expand_deg=predictor_expand_deg)
    z_dom_full = ns["daily_da_select_box_wrap"](z_sel, pred_box)
    z_dom_full, y_sel_full, chw_sel_full = xr.align(z_dom_full, y_sel_full, chw_sel_full, join="inner")

    z_anom_full = ns["daily_da_calendar_day_anomaly"](z_dom_full)
    y_anom_full = ns["daily_da_calendar_day_anomaly"](y_sel_full)

    X_full, _ = ns["daily_da_stack_feature_matrix"](z_anom_full)
    Y_anom_full, y_stack = ns["daily_da_stack_target_matrix"](y_anom_full)
    Y_total_full = y_sel_full.stack(target=("lat", "lon")).transpose("time", "target").sel(target=y_stack["target"]).values.astype(float)
    CHW_full = chw_sel_full.stack(target=("lat", "lon")).transpose("time", "target").sel(target=y_stack["target"]).values.astype(float)
    CHW_mask = np.isfinite(CHW_full) & (CHW_full > 0.5)

    if X_full.shape[1] < 2:
        raise ValueError("有效预测格点过少，无法进行 Ridge 回归。")
    if Y_anom_full.shape[1] < 1:
        raise ValueError("有效目标格点为空。")

    years, unique_years, year_row_index = _build_year_indexer(y_sel_full["time"].values)
    if training_mode == "all_jja":
        train_mask, train_sample_counts = _build_gridwise_all_jja_mask(Y_total_full)
    elif training_mode == "topn":
        if top_n_days is None:
            raise ValueError("topn 训练模式必须提供 top_n_days。")
        train_mask, train_sample_counts = _build_gridwise_topn_mask(Y_total_full, year_row_index, top_n_days)
    else:
        raise ValueError(f"不支持的训练模式：{training_mode}")
    display_mask = CHW_mask & np.isfinite(Y_total_full)
    display_sample_counts = display_mask.sum(axis=0).astype(int)

    alpha, alpha_score = _choose_shared_alpha_gridwise(
        ns,
        X_full,
        Y_anom_full,
        train_mask,
        years,
        standardize_X=standardize_X,
        alphas=alphas,
        cv_year_folds=cv_year_folds,
        max_targets_for_cv=max_targets_for_cv,
        random_state=random_state,
    )

    n_years = unique_years.size
    n_targets = Y_anom_full.shape[1]
    annual_total = np.full((n_years, n_targets), np.nan, dtype=float)
    annual_dyn = np.full((n_years, n_targets), np.nan, dtype=float)
    annual_adj = np.full((n_years, n_targets), np.nan, dtype=float)
    r2 = np.full(n_targets, np.nan, dtype=float)

    def _solve_one_target(target_idx):
        train_rows = np.where(train_mask[:, target_idx] & np.isfinite(Y_anom_full[:, target_idx]))[0]
        annual_total_col = np.full(n_years, np.nan, dtype=float)
        annual_dyn_col = np.full(n_years, np.nan, dtype=float)
        annual_adj_col = np.full(n_years, np.nan, dtype=float)
        r2_val = np.nan

        if train_rows.size == 0:
            return target_idx, annual_total_col, annual_dyn_col, annual_adj_col, r2_val

        X_train = X_full[train_rows, :]
        Y_train = Y_anom_full[train_rows, target_idx]
        X_train_std, scaler = ns["daily_da_standardize_matrix"](X_train, mode=standardize_X)
        model = ns["Ridge"](alpha=float(alpha), fit_intercept=True)
        model.fit(X_train_std, Y_train)
        pred_train = model.predict(X_train_std)

        ss_res = float(np.sum((Y_train - pred_train) ** 2))
        ss_tot = float(np.sum((Y_train - np.mean(Y_train)) ** 2))
        if ss_tot > 0:
            r2_val = 1.0 - ss_res / ss_tot

        display_rows = np.where(display_mask[:, target_idx] & np.isfinite(Y_anom_full[:, target_idx]))[0]
        if display_rows.size == 0:
            return target_idx, annual_total_col, annual_dyn_col, annual_adj_col, r2_val

        X_display = X_full[display_rows, :]
        if scaler is not None:
            X_display_std = scaler.transform(X_display)
        else:
            X_display_std = X_display
        pred_display = model.predict(X_display_std)
        Y_total_display = Y_total_full[display_rows, target_idx]
        display_years = years[display_rows]

        for year_idx, year in enumerate(unique_years):
            year_mask = display_years == year
            if np.any(year_mask):
                annual_total_col[year_idx] = float(np.nanmean(Y_total_display[year_mask]))
                annual_dyn_col[year_idx] = float(np.nanmean(pred_display[year_mask]))
                annual_adj_col[year_idx] = float(np.nanmean(Y_total_display[year_mask] - pred_display[year_mask]))

        return target_idx, annual_total_col, annual_dyn_col, annual_adj_col, r2_val

    n_jobs = max(1, min(TARGET_N_JOBS, n_targets))
    results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=1)(
        delayed(_solve_one_target)(target_idx) for target_idx in range(n_targets)
    )
    for target_idx, annual_total_col, annual_dyn_col, annual_adj_col, r2_val in results:
        annual_total[:, target_idx] = annual_total_col
        annual_dyn[:, target_idx] = annual_dyn_col
        annual_adj[:, target_idx] = annual_adj_col
        r2[target_idx] = r2_val

    annual_total_da = xr.DataArray(annual_total, dims=("year", "target"), coords={"year": unique_years, "target": y_stack["target"]}).unstack("target").transpose("year", "lat", "lon")
    annual_dyn_da = xr.DataArray(annual_dyn, dims=("year", "target"), coords={"year": unique_years, "target": y_stack["target"]}).unstack("target").transpose("year", "lat", "lon")
    annual_adj_da = xr.DataArray(annual_adj, dims=("year", "target"), coords={"year": unique_years, "target": y_stack["target"]}).unstack("target").transpose("year", "lat", "lon")

    slope_total = xr.DataArray(_vector_slope(annual_total, unique_years), dims=("target",), coords={"target": y_stack["target"]}).unstack("target").transpose("lat", "lon")
    slope_dyn = xr.DataArray(_vector_slope(annual_dyn, unique_years), dims=("target",), coords={"target": y_stack["target"]}).unstack("target").transpose("lat", "lon")
    slope_adj = xr.DataArray(_vector_slope(annual_adj, unique_years), dims=("target",), coords={"target": y_stack["target"]}).unstack("target").transpose("lat", "lon")
    r2_map = xr.DataArray(r2, dims=("target",), coords={"target": y_stack["target"]}).unstack("target").transpose("lat", "lon")

    return {
        "annual_total": annual_total_da,
        "annual_dyn": annual_dyn_da,
        "annual_adj": annual_adj_da,
        "slope_total": slope_total,
        "slope_dyn": slope_dyn,
        "slope_thermo": slope_total - slope_dyn,
        "slope_thermo_direct": slope_adj,
        "slope_thermo_diff": (slope_total - slope_dyn) - slope_adj,
        "r2": r2_map,
        "selected_alpha": float(alpha),
        "alpha_cv_score": float(alpha_score),
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


def _compute_rows_spatial_and_annual(ns, paths, temp_dict, z500_rel, chw_mask, template_2d, training_mode, top_n_days, member_name=None, max_targets_for_cv=400):
    rows = []
    spatial_out = {}
    annual_out = {}
    regions = ns["daily_da_get_region_settings"]()
    owner = "ERA5" if member_name is None else str(member_name)

    for var_name, temp_da in temp_dict.items():
        total_list, dyn_list, thermo_list = [], [], []
        thermo_direct_list, thermo_diff_list = [], []
        annual_total_list, annual_dyn_list, annual_adj_list = [], [], []

        for region_name, box in regions:
            print(
                f"[{paths['tag']}] {owner} | {var_name} | {region_name} 开始计算",
                flush=True,
            )
            y_region = ns["daily_da_select_box_wrap"](temp_da, box)
            chw_region = ns["daily_da_select_box_wrap"](chw_mask, box)
            fit_result = _fit_gridwise_conditioned_region(
                ns,
                z500_rel,
                y_region,
                chw_region,
                training_mode=training_mode,
                top_n_days=top_n_days,
                predictor_expand_deg=15.0,
                standardize_X="zscore",
                cv_year_folds=5,
                max_targets_for_cv=max_targets_for_cv,
                random_state=0,
            )

            slope_total_reg = fit_result["slope_total"]
            slope_dyn_reg = fit_result["slope_dyn"]
            slope_thermo_reg = fit_result["slope_thermo"]
            slope_thermo_direct_reg = fit_result["slope_thermo_direct"]
            slope_thermo_diff_reg = fit_result["slope_thermo_diff"]
            annual_total_reg = fit_result["annual_total"]
            annual_dyn_reg = fit_result["annual_dyn"]
            annual_adj_reg = fit_result["annual_adj"]
            thermo_diff_abs_max = float(np.nanmax(np.abs(slope_thermo_diff_reg.values))) if np.isfinite(slope_thermo_diff_reg.values).any() else np.nan

            full_total = ns["daily_da_put_on_template"](slope_total_reg, template_2d).astype("float32").expand_dims(region=[region_name])
            full_dyn = ns["daily_da_put_on_template"](slope_dyn_reg, template_2d).astype("float32").expand_dims(region=[region_name])
            full_thermo = ns["daily_da_put_on_template"](slope_thermo_reg, template_2d).astype("float32").expand_dims(region=[region_name])
            full_thermo_direct = ns["daily_da_put_on_template"](slope_thermo_direct_reg, template_2d).astype("float32").expand_dims(region=[region_name])
            full_thermo_diff = ns["daily_da_put_on_template"](slope_thermo_diff_reg, template_2d).astype("float32").expand_dims(region=[region_name])

            total_list.append(full_total)
            dyn_list.append(full_dyn)
            thermo_list.append(full_thermo)
            thermo_direct_list.append(full_thermo_direct)
            thermo_diff_list.append(full_thermo_diff)

            annual_total_list.append(_put_annual_on_template(annual_total_reg, template_2d).expand_dims(region=[region_name]))
            annual_dyn_list.append(_put_annual_on_template(annual_dyn_reg, template_2d).expand_dims(region=[region_name]))
            annual_adj_list.append(_put_annual_on_template(annual_adj_reg, template_2d).expand_dims(region=[region_name]))

            slope_total_map = float(ns["daily_da_area_mean"](slope_total_reg).values)
            slope_dyn_map = float(ns["daily_da_area_mean"](slope_dyn_reg).values)
            slope_thermo_map = float(ns["daily_da_area_mean"](slope_thermo_reg).values)
            y_total_ts = ns["daily_da_area_mean"](annual_total_reg)
            y_dyn_ts = ns["daily_da_area_mean"](annual_dyn_reg)
            y_thermo_ts = y_total_ts - y_dyn_ts
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
                "total_slope_ts_per_year": tr_total["slope"],
                "dyn_slope_ts_per_year": tr_dyn["slope"],
                "thermo_slope_ts_per_year": tr_thermo["slope"],
                "total_pvalue_ts": tr_total["pvalue"],
                "dyn_pvalue_ts": tr_dyn["pvalue"],
                "thermo_pvalue_ts": tr_thermo["pvalue"],
                "total_delta_map_minus_ts_per_year": slope_total_map - tr_total["slope"],
                "dyn_delta_map_minus_ts_per_year": slope_dyn_map - tr_dyn["slope"],
                "thermo_delta_map_minus_ts_per_year": slope_thermo_map - tr_thermo["slope"],
                "thermo_diff_abs_max_per_year": thermo_diff_abs_max,
                "predictor_bounds_lat_min": float(fit_result.get("predictor_bounds_lat_min", np.nan)),
                "predictor_bounds_lat_max": float(fit_result.get("predictor_bounds_lat_max", np.nan)),
                "predictor_bounds_lon_min": float(fit_result.get("predictor_bounds_lon_min", np.nan)),
                "predictor_bounds_lon_max": float(fit_result.get("predictor_bounds_lon_max", np.nan)),
                "standardize_X": fit_result.get("standardize_X", ""),
                "alpha_cv_score": float(fit_result.get("alpha_cv_score", np.nan)),
            }
            if member_name is not None:
                row["member"] = str(member_name)
            rows.append(row)
            print(
                f"[{paths['tag']}] {owner} | {var_name} | {region_name} 完成",
                flush=True,
            )

        spatial_out[f"{var_name}_total_slope_per_year"] = xr.concat(total_list, dim="region")
        spatial_out[f"{var_name}_dyn_slope_per_year"] = xr.concat(dyn_list, dim="region")
        spatial_out[f"{var_name}_thermo_slope_per_year"] = xr.concat(thermo_list, dim="region")
        spatial_out[f"{var_name}_thermo_direct_slope_per_year"] = xr.concat(thermo_direct_list, dim="region")
        spatial_out[f"{var_name}_thermo_diff_slope_per_year"] = xr.concat(thermo_diff_list, dim="region")

        annual_out[f"{var_name}_annual_total"] = xr.concat(annual_total_list, dim="region").transpose("year", "region", "lat", "lon").astype("float32")
        annual_out[f"{var_name}_annual_dyn"] = xr.concat(annual_dyn_list, dim="region").transpose("year", "region", "lat", "lon").astype("float32")
        annual_out[f"{var_name}_annual_adj"] = xr.concat(annual_adj_list, dim="region").transpose("year", "region", "lat", "lon").astype("float32")

    return rows, spatial_out, annual_out


def _compute_era5_variant(ns, paths, force=False):
    outputs_exist = all(os.path.exists(paths[key]) for key in ("era5_region", "era5_spatial", "era5_annual"))
    if outputs_exist and not force:
        print(f"ERA5 {paths['tag']} 缓存已存在，跳过重算。", flush=True)
        return {
            "region_csv": paths["era5_region"],
            "spatial_nc": paths["era5_spatial"],
            "annual_nc": paths["era5_annual"],
            "skipped": True,
        }

    tmax, tmin = ns["daily_da_open_era5_daily_temps"]()
    z500_rel = ns["daily_da_remove_nh_mean"](ns["daily_da_open_era5_daily_z500"]())
    chw_mask = _open_era5_chw_mask(ns)
    template = xr.full_like(tmax.isel(time=0, drop=True), np.nan)
    rows, spatial_out, annual_out = _compute_rows_spatial_and_annual(
        ns,
        paths,
        {"tmax": tmax, "tmin": tmin},
        z500_rel,
        chw_mask,
        template,
        training_mode=paths["training_mode"],
        top_n_days=paths["top_n"],
        member_name=None,
        max_targets_for_cv=24,
    )
    pd.DataFrame(rows).to_csv(paths["era5_region"], index=False)
    _write_dataset(spatial_out, paths["era5_spatial"])
    _write_dataset(annual_out, paths["era5_annual"])
    print(f"已保存 ERA5 {paths['tag']} 条件化日尺度结果。", flush=True)
    return {
        "region_csv": paths["era5_region"],
        "spatial_nc": paths["era5_spatial"],
        "annual_nc": paths["era5_annual"],
        "skipped": False,
        "n_rows": len(rows),
    }


def _compute_member_cache_variant(ns, paths, member, source_table=None, force=False):
    region_path = _member_region_path(paths, member)
    spatial_path = _member_spatial_path(paths, member)
    annual_path = _member_annual_path(paths, member)
    outputs_exist = os.path.exists(region_path) and os.path.exists(spatial_path) and os.path.exists(annual_path)
    if outputs_exist and not force:
        print(f"{member} {paths['tag']} 缓存已存在，跳过重算。", flush=True)
        return {"member": member, "region_csv": region_path, "spatial_nc": spatial_path, "annual_nc": annual_path, "skipped": True}

    tmax = ns["daily_da_open_cmip_daily_temperature"](member, "tasmax")
    tmin = ns["daily_da_open_cmip_daily_temperature"](member, "tasmin")
    z500_rel = ns["daily_da_remove_nh_mean"](ns["daily_da_open_cmip_daily_z500"](member, source_table=source_table))
    chw_mask = _open_member_chw_mask(ns, member)
    template = xr.full_like(tmax.isel(time=0, drop=True), np.nan)
    rows, spatial_out, annual_out = _compute_rows_spatial_and_annual(
        ns,
        paths,
        {"tmax": tmax, "tmin": tmin},
        z500_rel,
        chw_mask,
        template,
        training_mode=paths["training_mode"],
        top_n_days=paths["top_n"],
        member_name=member,
        max_targets_for_cv=16,
    )
    pd.DataFrame(rows).to_csv(region_path, index=False)
    _write_dataset(spatial_out, spatial_path)
    _write_dataset(annual_out, annual_path)
    print(f"已保存 {member} {paths['tag']} 条件化成员缓存。", flush=True)
    return {"member": member, "region_csv": region_path, "spatial_nc": spatial_path, "annual_nc": annual_path, "skipped": False}


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
    df_mme["top_n_warm_days_per_year"] = np.nan if paths["top_n"] is None else int(paths["top_n"])
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


def compute_chw_variant(ns, spec, run_era5=True, run_cmip=True, force_era5=False, force_members=False, force_summary=False, member_subset=None, skip_summary=False):
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
        "available_members": available_members,
        "missing_members": missing_members,
        "paths": paths,
    }
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


def smoke_test_chw_variant(ns, spec, member_name=None):
    paths = _build_variant_paths(spec)
    source_table = ns["daily_da_build_member_source_table"](output_csv=paths["member_sources"], force=False)
    available_members = source_table.loc[source_table["available"], "member"].astype(str).tolist()
    missing_members = source_table.loc[~source_table["available"], "member"].astype(str).tolist()
    if member_name is None:
        member_name = available_members[0]

    test_region_name, test_box = ns["daily_da_get_region_settings"]()[0]
    tmax_era, _ = ns["daily_da_open_era5_daily_temps"]()
    z_era = ns["daily_da_remove_nh_mean"](ns["daily_da_open_era5_daily_z500"]())
    chw_era = _open_era5_chw_mask(ns)
    fit_era = _fit_gridwise_conditioned_region(
        ns,
        z_era,
        ns["daily_da_select_box_wrap"](tmax_era, test_box),
        ns["daily_da_select_box_wrap"](chw_era, test_box),
        training_mode=paths["training_mode"],
        top_n_days=paths["top_n"],
        max_targets_for_cv=12,
    )

    t_test = ns["daily_da_open_cmip_daily_temperature"](member_name, "tasmax")
    z_test = ns["daily_da_remove_nh_mean"](ns["daily_da_open_cmip_daily_z500"](member_name, source_table=source_table))
    chw_test = _open_member_chw_mask(ns, member_name)
    fit_test = _fit_gridwise_conditioned_region(
        ns,
        z_test,
        ns["daily_da_select_box_wrap"](t_test, test_box),
        ns["daily_da_select_box_wrap"](chw_test, test_box),
        training_mode=paths["training_mode"],
        top_n_days=paths["top_n"],
        max_targets_for_cv=12,
    )

    return {
        "tag": paths["tag"],
        "training_mode": paths["training_mode"],
        "top_n": paths["top_n"],
        "available_members": available_members,
        "missing_members": missing_members,
        "test_member": member_name,
        "test_region": test_region_name,
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
        },
        flush=True,
    )

    outputs = []
    for spec in specs:
        print(f"开始处理 {spec['tag']} 版本。", flush=True)
        if smoke_only:
            outputs.append(smoke_test_chw_variant(ns, spec, member_name=member_subset[0] if member_subset else None))
        else:
            outputs.append(
                compute_chw_variant(
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
