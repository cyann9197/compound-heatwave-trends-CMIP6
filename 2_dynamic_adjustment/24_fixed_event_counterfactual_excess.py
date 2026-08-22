import importlib.util
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
DEFAULT_BASE_DIR = str(CACHE_ROOT / "dynamic_adjustment" / "fixed_event_counterfactual_excess")
TARGET_N_JOBS = max(1, int(os.environ.get("DA_TARGET_NJOBS", "8")))


def _import_script(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHW_MOD = _import_script(CODE_DIR / "22_daily_da_common.py", "daily_chwonly_base")
FE_MOD = _import_script(
    CODE_DIR / "23_fixed_event_common.py",
    "daily_fixed_event_base",
)


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


def _result_base_dir():
    return os.environ.get("DA_BASE_DIR", DEFAULT_BASE_DIR)


def _variant_specs():
    default_tags = ["chwcumheatcfexcess_chwcalendar31d"]
    tags = [tag.lower() for tag in _env_list("DA_VARIANT_TAGS", default=default_tags)]
    mapping = {
        "chwcumheatcfexcess_alljja": {
            "tag": "chwcumheatcfexcess_alljja",
            "training_mode": "all_jja",
            "top_n": None,
            "scheme_label": "fixed-event cfexcess | JJA 全部日值训练",
        },
        "chwcumheatcfexcess_chwcalendar31d": {
            "tag": "chwcumheatcfexcess_chwcalendar31d",
            "training_mode": "chwcalendar_window",
            "top_n": None,
            "scheme_label": "fixed-event cfexcess | CHW 日所有年份同日序前后 15 天窗口训练",
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
    return str(owner_dir / f"{_safe_name(var_name)}_{_safe_name(region_name)}_daily_cfexcess.nc")


def _annual_sum_from_mask(values, mask, target_valid, year_row_index, require_all_finite=False):
    n_years = len(year_row_index)
    n_targets = values.shape[1]
    out = np.full((n_years, n_targets), np.nan, dtype=float)
    for target_idx in range(n_targets):
        if not target_valid[target_idx]:
            continue
        for year_idx, year_rows in enumerate(year_row_index):
            year_mask = mask[year_rows, target_idx]
            count = int(np.sum(year_mask))
            if count == 0:
                out[year_idx, target_idx] = 0.0
                continue
            sel = values[year_rows, target_idx][year_mask]
            if require_all_finite and not np.all(np.isfinite(sel)):
                out[year_idx, target_idx] = np.nan
            else:
                out[year_idx, target_idx] = float(np.nansum(sel))
    return out


def _prepare_cfexcess_daily_matrices(ns, z500_rel, y_region, chw_region, threshold_region, predictor_expand_deg=15.0):
    prepared = FE_MOD._prepare_cumheat_daily_matrices(
        ns,
        z500_rel,
        y_region,
        chw_region,
        threshold_region,
        predictor_expand_deg=predictor_expand_deg,
    )
    y_anom_full = prepared["Y_anom_full"]
    y_total_full = prepared["Y_total_full"]
    threshold_full = prepared["threshold_full"]
    chw_mask = prepared["CHW_mask"]
    tclim_full = (y_total_full - y_anom_full).astype(np.float32)
    valid_display = (
        chw_mask
        & np.isfinite(y_total_full)
        & np.isfinite(threshold_full)
        & np.isfinite(y_anom_full)
    )
    excess_full = np.where(valid_display, np.maximum(y_total_full - threshold_full, 0.0), 0.0).astype(np.float32)
    target_valid = prepared["target_valid"] & np.isfinite(y_anom_full).any(axis=0)
    prepared.update(
        {
            "tclim_full": tclim_full,
            "valid_display": valid_display,
            "excess_full": excess_full,
            "target_valid": target_valid,
        }
    )
    return prepared


def _pack_cfexcess_fit_result(
    annual_total,
    annual_dyn_contrib_fixed,
    annual_residual_fixed,
    annual_chw_days,
    r2,
    unique_years,
    y_stack,
    pred_box,
    alpha,
    alpha_score,
    standardize_X,
    train_sample_counts,
    display_sample_counts,
    time_values,
    daily_delta_dyn_temp,
    daily_total_excess,
    daily_residual_fixed,
    daily_dyn_contrib_fixed,
    daily_chw_mask,
    daily_raw_full,
    daily_q90_time,
    daily_tclim_raw,
):
    def _annual_to_da(values):
        return (
            xr.DataArray(
                values,
                dims=("year", "target"),
                coords={"year": unique_years, "target": y_stack["target"]},
            )
            .unstack("target")
            .transpose("year", "lat", "lon")
            .astype("float32")
        )

    def _slope_to_da(values):
        return (
            xr.DataArray(
                FE_MOD._vector_slope(values, unique_years),
                dims=("target",),
                coords={"target": y_stack["target"]},
            )
            .unstack("target")
            .transpose("lat", "lon")
            .astype("float32")
        )

    def _daily_to_da(values):
        return (
            xr.DataArray(
                values,
                dims=("time", "target"),
                coords={"time": time_values, "target": y_stack["target"]},
            )
            .unstack("target")
            .transpose("time", "lat", "lon")
            .astype("float32")
        )

    annual_total_da = _annual_to_da(annual_total)
    annual_dyn_da = _annual_to_da(annual_dyn_contrib_fixed)
    annual_residual_da = _annual_to_da(annual_residual_fixed)
    annual_chw_days_da = _annual_to_da(annual_chw_days)

    slope_total = _slope_to_da(annual_total)
    slope_dyn = _slope_to_da(annual_dyn_contrib_fixed)
    slope_residual = _slope_to_da(annual_residual_fixed)
    r2_map = (
        xr.DataArray(r2, dims=("target",), coords={"target": y_stack["target"]})
        .unstack("target")
        .transpose("lat", "lon")
        .astype("float32")
    )

    daily_delta_dyn_da = _daily_to_da(daily_delta_dyn_temp)
    daily_total_da = _daily_to_da(daily_total_excess)
    daily_residual_da = _daily_to_da(daily_residual_fixed)
    daily_dyn_contrib_da = _daily_to_da(daily_dyn_contrib_fixed)
    daily_chw_mask_da = _daily_to_da(daily_chw_mask)

    display_bool = daily_chw_mask.astype(bool)

    def _masked_mean(values):
        if not np.any(display_bool):
            return np.nan
        sel = values[display_bool]
        return float(np.nanmean(sel)) if np.isfinite(sel).any() else np.nan

    diagnostics = {
        "mean_T_raw": _masked_mean(daily_raw_full),
        "mean_T_clim": _masked_mean(daily_tclim_raw),
        "mean_delta_dyn": _masked_mean(daily_delta_dyn_temp),
        "mean_Q90": _masked_mean(daily_q90_time),
        "mean_total_excess": _masked_mean(daily_total_excess),
        "mean_residual_fixed": _masked_mean(daily_residual_fixed),
        "mean_dyn_contrib_fixed": _masked_mean(daily_dyn_contrib_fixed),
    }

    return {
        "annual_total": annual_total_da,
        "annual_dyn_contrib_fixed": annual_dyn_da,
        "annual_residual_fixed": annual_residual_da,
        "annual_dyn": annual_dyn_da,
        "annual_thermo": annual_residual_da,
        "annual_chw_days": annual_chw_days_da,
        "slope_total": slope_total,
        "slope_dyn_contrib_fixed": slope_dyn,
        "slope_residual_fixed": slope_residual,
        "slope_dyn": slope_dyn,
        "slope_thermo": slope_residual,
        "r2_train": r2_map,
        "selected_alpha": float(alpha) if np.isfinite(alpha) else np.nan,
        "alpha_cv_score": float(alpha_score) if np.isfinite(alpha_score) else np.nan,
        "predictor_bounds_lon_min": float(pred_box[0]),
        "predictor_bounds_lon_max": float(pred_box[1]),
        "predictor_bounds_lat_min": float(pred_box[2]),
        "predictor_bounds_lat_max": float(pred_box[3]),
        "standardize_X": standardize_X,
        "top_n_warm_days_per_year": np.nan,
        "train_sample_counts": train_sample_counts.astype(int),
        "display_sample_counts": display_sample_counts.astype(int),
        "daily_delta_dyn_temp": daily_delta_dyn_da,
        "daily_total_excess": daily_total_da,
        "daily_residual_fixed": daily_residual_da,
        "daily_dyn_contrib_fixed": daily_dyn_contrib_da,
        "daily_chw_mask": daily_chw_mask_da,
        "diagnostics": diagnostics,
    }


def _fit_gridwise_cfexcess_daily_region(
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

    prepared = _prepare_cfexcess_daily_matrices(
        ns,
        z500_rel,
        y_region,
        chw_region,
        threshold_region,
        predictor_expand_deg=predictor_expand_deg,
    )
    X_full = prepared["X_full"]
    Y_anom_full = prepared["Y_anom_full"]
    Y_total_full = prepared["Y_total_full"]
    threshold_full = prepared["threshold_full"]
    CHW_mask = prepared["CHW_mask"]
    valid_display = prepared["valid_display"]
    excess_full = prepared["excess_full"]
    target_valid = prepared["target_valid"]
    years = prepared["years"]
    unique_years = prepared["unique_years"]
    year_row_index = prepared["year_row_index"]
    y_stack = prepared["y_stack"]
    pred_box = prepared["pred_box"]

    if X_full.shape[1] < 2:
        raise ValueError("有效预测格点过少，无法进行 Ridge 回归。")
    if Y_anom_full.shape[1] < 1:
        raise ValueError("有效目标格点为空。")

    train_mask, train_sample_counts = FE_MOD._build_daily_training_mask(
        training_mode,
        Y_total_full,
        CHW_mask,
        year_row_index,
        top_n_days=top_n_days,
    )
    display_sample_counts = valid_display.sum(axis=0).astype(int)
    annual_total, annual_chw_days = FE_MOD._annual_totals_and_days(
        excess_full,
        CHW_mask,
        target_valid,
        year_row_index,
    )

    alpha, alpha_score = FE_MOD._choose_shared_alpha_gridwise(
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
    n_time = Y_anom_full.shape[0]
    annual_dyn = np.full((n_years, n_targets), np.nan, dtype=float)
    annual_residual = np.full((n_years, n_targets), np.nan, dtype=float)
    daily_delta_dyn = np.full((n_time, n_targets), np.nan, dtype=np.float32)
    daily_total_excess = np.full((n_time, n_targets), np.nan, dtype=np.float32)
    daily_residual_fixed = np.full((n_time, n_targets), np.nan, dtype=np.float32)
    daily_dyn_contrib = np.full((n_time, n_targets), np.nan, dtype=np.float32)
    daily_chw_mask = valid_display.astype(np.float32)
    r2 = np.full(n_targets, np.nan, dtype=float)

    def _solve_one_target(target_idx):
        dyn_col = np.full(n_years, np.nan, dtype=float)
        residual_col = np.full(n_years, np.nan, dtype=float)
        delta_dyn_col = np.full(n_time, np.nan, dtype=np.float32)
        total_col = np.full(n_time, np.nan, dtype=np.float32)
        residual_daily_col = np.full(n_time, np.nan, dtype=np.float32)
        dyn_daily_col = np.full(n_time, np.nan, dtype=np.float32)
        r2_val = np.nan

        if not target_valid[target_idx]:
            return target_idx, dyn_col, residual_col, r2_val, delta_dyn_col, total_col, residual_daily_col, dyn_daily_col

        chw_days_col = annual_chw_days[:, target_idx]
        no_chw_years = np.isfinite(chw_days_col) & (chw_days_col == 0)
        dyn_col[no_chw_years] = 0.0
        residual_col[no_chw_years] = 0.0

        train_rows = np.where(train_mask[:, target_idx] & np.isfinite(Y_anom_full[:, target_idx]))[0]
        if train_rows.size < 2 or not np.isfinite(alpha):
            return target_idx, dyn_col, residual_col, r2_val, delta_dyn_col, total_col, residual_daily_col, dyn_daily_col

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

        display_rows = np.where(valid_display[:, target_idx])[0]
        if display_rows.size == 0:
            return target_idx, dyn_col, residual_col, r2_val, delta_dyn_col, total_col, residual_daily_col, dyn_daily_col

        X_display = X_full[display_rows, :]
        X_display_std = scaler.transform(X_display) if scaler is not None else X_display
        pred_display = model.predict(X_display_std)
        total_excess_display = np.maximum(
            Y_total_full[display_rows, target_idx] - threshold_full[display_rows, target_idx],
            0.0,
        )
        residual_display = np.maximum(
            Y_total_full[display_rows, target_idx] - pred_display - threshold_full[display_rows, target_idx],
            0.0,
        )
        dyn_contrib_display = total_excess_display - residual_display

        delta_dyn_col[display_rows] = pred_display.astype(np.float32)
        total_col[display_rows] = total_excess_display.astype(np.float32)
        residual_daily_col[display_rows] = residual_display.astype(np.float32)
        dyn_daily_col[display_rows] = dyn_contrib_display.astype(np.float32)

        display_years = years[display_rows]
        for year_idx, year in enumerate(unique_years):
            if not np.isfinite(chw_days_col[year_idx]) or chw_days_col[year_idx] == 0:
                continue
            year_mask = display_years == year
            if np.any(year_mask):
                dyn_col[year_idx] = float(np.nansum(dyn_contrib_display[year_mask]))
                residual_col[year_idx] = float(np.nansum(residual_display[year_mask]))
        return target_idx, dyn_col, residual_col, r2_val, delta_dyn_col, total_col, residual_daily_col, dyn_daily_col

    n_jobs = max(1, min(TARGET_N_JOBS, n_targets))
    results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=1)(
        delayed(_solve_one_target)(target_idx) for target_idx in range(n_targets)
    )
    for target_idx, dyn_col, residual_col, r2_val, delta_dyn_col, total_col, residual_daily_col, dyn_daily_col in results:
        annual_dyn[:, target_idx] = dyn_col
        annual_residual[:, target_idx] = residual_col
        r2[target_idx] = r2_val
        daily_delta_dyn[:, target_idx] = delta_dyn_col
        daily_total_excess[:, target_idx] = total_col
        daily_residual_fixed[:, target_idx] = residual_daily_col
        daily_dyn_contrib[:, target_idx] = dyn_daily_col

    return _pack_cfexcess_fit_result(
        annual_total=annual_total,
        annual_dyn_contrib_fixed=annual_dyn,
        annual_residual_fixed=annual_residual,
        annual_chw_days=annual_chw_days,
        r2=r2,
        unique_years=unique_years,
        y_stack=y_stack,
        pred_box=pred_box,
        alpha=alpha,
        alpha_score=alpha_score,
        standardize_X=standardize_X,
        train_sample_counts=train_sample_counts,
        display_sample_counts=display_sample_counts,
        time_values=prepared["time_values"],
        daily_delta_dyn_temp=daily_delta_dyn,
        daily_total_excess=daily_total_excess,
        daily_residual_fixed=daily_residual_fixed,
        daily_dyn_contrib_fixed=daily_dyn_contrib,
        daily_chw_mask=daily_chw_mask,
        daily_raw_full=Y_total_full.astype(np.float32),
        daily_q90_time=threshold_full.astype(np.float32),
        daily_tclim_raw=prepared["tclim_full"].astype(np.float32),
    )


def _fit_gridwise_cfexcess_calendar_window_region(
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

    prepared = _prepare_cfexcess_daily_matrices(
        ns,
        z500_rel,
        y_region,
        chw_region,
        threshold_region,
        predictor_expand_deg=predictor_expand_deg,
    )
    X_full = prepared["X_full"]
    Y_anom_full = prepared["Y_anom_full"]
    Y_total_full = prepared["Y_total_full"]
    threshold_full = prepared["threshold_full"]
    CHW_mask = prepared["CHW_mask"]
    valid_display = prepared["valid_display"]
    excess_full = prepared["excess_full"]
    target_valid = prepared["target_valid"]
    years = prepared["years"]
    unique_years = prepared["unique_years"]
    year_row_index = prepared["year_row_index"]
    y_stack = prepared["y_stack"]
    pred_box = prepared["pred_box"]

    if X_full.shape[1] < 2:
        raise ValueError("有效预测格点过少，无法进行 Ridge 回归。")
    if Y_anom_full.shape[1] < 1:
        raise ValueError("有效目标格点为空。")

    train_mask, train_sample_counts = FE_MOD._build_gridwise_chw_calendar_window_mask(
        Y_total_full,
        CHW_mask,
        year_row_index,
        half_window=half_window,
    )
    display_sample_counts = valid_display.sum(axis=0).astype(int)
    annual_total, annual_chw_days = FE_MOD._annual_totals_and_days(
        excess_full,
        CHW_mask,
        target_valid,
        year_row_index,
    )

    alpha, alpha_score = FE_MOD._choose_shared_alpha_gridwise(
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

    n_time = Y_anom_full.shape[0]
    n_years = unique_years.size
    n_targets = Y_anom_full.shape[1]
    row_year_index, row_day_index = FE_MOD._build_row_year_and_day_index(year_row_index, n_time)
    max_day_index = int(np.nanmax(row_day_index)) if np.any(row_day_index >= 0) else 0
    rows_by_center = FE_MOD._build_calendar_window_rows(year_row_index, max_day_index, half_window=half_window)

    annual_dyn = np.full((n_years, n_targets), np.nan, dtype=float)
    annual_residual = np.full((n_years, n_targets), np.nan, dtype=float)
    daily_delta_dyn = np.full((n_time, n_targets), np.nan, dtype=np.float32)
    daily_total_excess = np.full((n_time, n_targets), np.nan, dtype=np.float32)
    daily_residual_fixed = np.full((n_time, n_targets), np.nan, dtype=np.float32)
    daily_dyn_contrib = np.full((n_time, n_targets), np.nan, dtype=np.float32)
    daily_chw_mask = valid_display.astype(np.float32)
    r2 = np.full(n_targets, np.nan, dtype=float)

    def _solve_one_target(target_idx):
        dyn_col = np.full(n_years, np.nan, dtype=float)
        residual_col = np.full(n_years, np.nan, dtype=float)
        delta_dyn_col = np.full(n_time, np.nan, dtype=np.float32)
        total_col = np.full(n_time, np.nan, dtype=np.float32)
        residual_daily_col = np.full(n_time, np.nan, dtype=np.float32)
        dyn_daily_col = np.full(n_time, np.nan, dtype=np.float32)
        r2_scores = []

        if not target_valid[target_idx]:
            return target_idx, dyn_col, residual_col, np.nan, delta_dyn_col, total_col, residual_daily_col, dyn_daily_col

        chw_days_col = annual_chw_days[:, target_idx]
        no_chw_years = np.isfinite(chw_days_col) & (chw_days_col == 0)
        dyn_col[no_chw_years] = 0.0
        residual_col[no_chw_years] = 0.0

        event_rows = np.where(valid_display[:, target_idx] & np.isfinite(Y_anom_full[:, target_idx]))[0]
        if event_rows.size == 0 or not np.isfinite(alpha):
            return target_idx, dyn_col, residual_col, np.nan, delta_dyn_col, total_col, residual_daily_col, dyn_daily_col

        predicted_counts = np.zeros(n_years, dtype=int)
        for center_day in np.unique(row_day_index[event_rows]):
            if center_day < 0:
                continue
            train_rows = rows_by_center.get(int(center_day), np.array([], dtype=int))
            if train_rows.size == 0:
                continue
            train_rows = train_rows[np.isfinite(Y_anom_full[train_rows, target_idx])]
            if train_rows.size < 2:
                continue

            X_train = X_full[train_rows, :]
            Y_train = Y_anom_full[train_rows, target_idx]
            X_train_std, scaler = ns["daily_da_standardize_matrix"](X_train, mode=standardize_X)
            model = ns["Ridge"](alpha=float(alpha), fit_intercept=True)
            model.fit(X_train_std, Y_train)

            pred_train = model.predict(X_train_std)
            ss_res = float(np.sum((Y_train - pred_train) ** 2))
            ss_tot = float(np.sum((Y_train - np.mean(Y_train)) ** 2))
            if ss_tot > 0:
                r2_scores.append(1.0 - ss_res / ss_tot)

            predict_rows = np.where(valid_display[:, target_idx] & (row_day_index == center_day))[0]
            if predict_rows.size == 0:
                continue
            X_predict = X_full[predict_rows, :]
            X_predict_std = scaler.transform(X_predict) if scaler is not None else X_predict
            pred_event = model.predict(X_predict_std)
            total_excess_event = np.maximum(
                Y_total_full[predict_rows, target_idx] - threshold_full[predict_rows, target_idx],
                0.0,
            )
            residual_event = np.maximum(
                Y_total_full[predict_rows, target_idx] - pred_event - threshold_full[predict_rows, target_idx],
                0.0,
            )
            dyn_contrib_event = total_excess_event - residual_event

            delta_dyn_col[predict_rows] = pred_event.astype(np.float32)
            total_col[predict_rows] = total_excess_event.astype(np.float32)
            residual_daily_col[predict_rows] = residual_event.astype(np.float32)
            dyn_daily_col[predict_rows] = dyn_contrib_event.astype(np.float32)

            for row in predict_rows:
                year_idx = row_year_index[row]
                if year_idx >= 0:
                    predicted_counts[year_idx] += 1

        for year_idx, year_rows in enumerate(year_row_index):
            if not np.isfinite(chw_days_col[year_idx]) or chw_days_col[year_idx] == 0:
                continue
            expected_count = int(np.sum(valid_display[year_rows, target_idx]))
            if predicted_counts[year_idx] == expected_count:
                dyn_col[year_idx] = float(np.nansum(dyn_daily_col[year_rows]))
                residual_col[year_idx] = float(np.nansum(residual_daily_col[year_rows]))

        r2_val = float(np.nanmean(r2_scores)) if r2_scores else np.nan
        return target_idx, dyn_col, residual_col, r2_val, delta_dyn_col, total_col, residual_daily_col, dyn_daily_col

    n_jobs = max(1, min(TARGET_N_JOBS, n_targets))
    results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=1)(
        delayed(_solve_one_target)(target_idx) for target_idx in range(n_targets)
    )
    for target_idx, dyn_col, residual_col, r2_val, delta_dyn_col, total_col, residual_daily_col, dyn_daily_col in results:
        annual_dyn[:, target_idx] = dyn_col
        annual_residual[:, target_idx] = residual_col
        r2[target_idx] = r2_val
        daily_delta_dyn[:, target_idx] = delta_dyn_col
        daily_total_excess[:, target_idx] = total_col
        daily_residual_fixed[:, target_idx] = residual_daily_col
        daily_dyn_contrib[:, target_idx] = dyn_daily_col

    return _pack_cfexcess_fit_result(
        annual_total=annual_total,
        annual_dyn_contrib_fixed=annual_dyn,
        annual_residual_fixed=annual_residual,
        annual_chw_days=annual_chw_days,
        r2=r2,
        unique_years=unique_years,
        y_stack=y_stack,
        pred_box=pred_box,
        alpha=alpha,
        alpha_score=alpha_score,
        standardize_X=standardize_X,
        train_sample_counts=train_sample_counts,
        display_sample_counts=display_sample_counts,
        time_values=prepared["time_values"],
        daily_delta_dyn_temp=daily_delta_dyn,
        daily_total_excess=daily_total_excess,
        daily_residual_fixed=daily_residual_fixed,
        daily_dyn_contrib_fixed=daily_dyn_contrib,
        daily_chw_mask=daily_chw_mask,
        daily_raw_full=Y_total_full.astype(np.float32),
        daily_q90_time=threshold_full.astype(np.float32),
        daily_tclim_raw=prepared["tclim_full"].astype(np.float32),
    )


def _write_daily_region_dataset(paths, owner, var_name, region_name, fit_result):
    out_path = _daily_region_path(paths, owner, var_name, region_name)
    ds = xr.Dataset(
        {
            f"{var_name}_hwc_daily_delta_dyn_temp": fit_result["daily_delta_dyn_temp"].astype("float32"),
            f"{var_name}_hwc_daily_total_excess": fit_result["daily_total_excess"].astype("float32"),
            f"{var_name}_hwc_daily_residual_fixed": fit_result["daily_residual_fixed"].astype("float32"),
            f"{var_name}_hwc_daily_dyn_contrib_fixed": fit_result["daily_dyn_contrib_fixed"].astype("float32"),
            "chw_mask_daily": fit_result["daily_chw_mask"].astype("float32"),
        }
    )
    for key, value in fit_result["diagnostics"].items():
        ds.attrs[key] = value
    ds.attrs["formula_total"] = "I_CHW * max(T_raw - Q90, 0)"
    ds.attrs["formula_residual_fixed"] = "I_CHW * max(T_raw - delta_T_dyn - Q90, 0)"
    ds.attrs["formula_dyn"] = "total - residual_fixed"
    FE_MOD._write_dataset(ds, out_path)
    return out_path


def _fit_region_by_mode(ns, z500_rel, y_region, chw_region, threshold_region, training_mode, top_n_days, max_targets_for_cv):
    if training_mode == "chwcalendar_window":
        return _fit_gridwise_cfexcess_calendar_window_region(
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
    return _fit_gridwise_cfexcess_daily_region(
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
    regions = ns["daily_da_get_region_settings"]()
    owner = "ERA5" if member_name is None else str(member_name)

    for var_name, temp_da in temp_dict.items():
        total_list, dyn_list, residual_list = [], [], []
        annual_total_list, annual_dyn_list, annual_residual_list = [], [], []
        annual_chw_days_list = []

        for region_name, box in regions:
            print(f"[{paths['tag']}] {owner} | {var_name} | {region_name} 开始计算 fixed-event cfexcess", flush=True)
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
            slope_dyn_reg = fit_result["slope_dyn_contrib_fixed"]
            slope_residual_reg = fit_result["slope_residual_fixed"]
            annual_total_reg = fit_result["annual_total"]
            annual_dyn_reg = fit_result["annual_dyn_contrib_fixed"]
            annual_residual_reg = fit_result["annual_residual_fixed"]
            annual_chw_days_reg = fit_result["annual_chw_days"]

            total_list.append(ns["daily_da_put_on_template"](slope_total_reg, template_2d).astype("float32").expand_dims(region=[region_name]))
            dyn_list.append(ns["daily_da_put_on_template"](slope_dyn_reg, template_2d).astype("float32").expand_dims(region=[region_name]))
            residual_list.append(ns["daily_da_put_on_template"](slope_residual_reg, template_2d).astype("float32").expand_dims(region=[region_name]))

            annual_total_list.append(FE_MOD._put_annual_on_template(annual_total_reg, template_2d).expand_dims(region=[region_name]))
            annual_dyn_list.append(FE_MOD._put_annual_on_template(annual_dyn_reg, template_2d).expand_dims(region=[region_name]))
            annual_residual_list.append(FE_MOD._put_annual_on_template(annual_residual_reg, template_2d).expand_dims(region=[region_name]))
            annual_chw_days_list.append(FE_MOD._put_annual_on_template(annual_chw_days_reg, template_2d).expand_dims(region=[region_name]))

            slope_total_map = float(ns["daily_da_area_mean"](slope_total_reg).values)
            slope_dyn_map = float(ns["daily_da_area_mean"](slope_dyn_reg).values)
            slope_residual_map = float(ns["daily_da_area_mean"](slope_residual_reg).values)
            y_total_ts = ns["daily_da_area_mean"](annual_total_reg)
            y_dyn_ts = ns["daily_da_area_mean"](annual_dyn_reg)
            y_residual_ts = ns["daily_da_area_mean"](annual_residual_reg)
            tr_total = ns["daily_da_series_trend"](y_total_ts)
            tr_dyn = ns["daily_da_series_trend"](y_dyn_ts)
            tr_residual = ns["daily_da_series_trend"](y_residual_ts)

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
                "thermo_slope_per_year": slope_residual_map,
                "thermo_slope_per_decade": slope_residual_map * 10.0,
                "dyn_contrib_fixed_slope_per_year": slope_dyn_map,
                "dyn_contrib_fixed_slope_per_decade": slope_dyn_map * 10.0,
                "residual_fixed_slope_per_year": slope_residual_map,
                "residual_fixed_slope_per_decade": slope_residual_map * 10.0,
                "total_slope_ts_per_year": tr_total["slope"],
                "dyn_slope_ts_per_year": tr_dyn["slope"],
                "thermo_slope_ts_per_year": tr_residual["slope"],
                "dyn_contrib_fixed_slope_ts_per_year": tr_dyn["slope"],
                "residual_fixed_slope_ts_per_year": tr_residual["slope"],
                "total_pvalue_ts": tr_total["pvalue"],
                "dyn_pvalue_ts": tr_dyn["pvalue"],
                "thermo_pvalue_ts": tr_residual["pvalue"],
                "dyn_contrib_fixed_pvalue_ts": tr_dyn["pvalue"],
                "residual_fixed_pvalue_ts": tr_residual["pvalue"],
                "total_delta_map_minus_ts_per_year": slope_total_map - tr_total["slope"],
                "dyn_delta_map_minus_ts_per_year": slope_dyn_map - tr_dyn["slope"],
                "thermo_delta_map_minus_ts_per_year": slope_residual_map - tr_residual["slope"],
                "dyn_contrib_fixed_delta_map_minus_ts_per_year": slope_dyn_map - tr_dyn["slope"],
                "residual_fixed_delta_map_minus_ts_per_year": slope_residual_map - tr_residual["slope"],
                "n_chw_days_mean": float(np.nanmean(annual_chw_days_reg.values)),
                "mean_T_raw": fit_result["diagnostics"]["mean_T_raw"],
                "mean_T_clim": fit_result["diagnostics"]["mean_T_clim"],
                "mean_delta_dyn": fit_result["diagnostics"]["mean_delta_dyn"],
                "mean_Q90": fit_result["diagnostics"]["mean_Q90"],
                "mean_total_excess": fit_result["diagnostics"]["mean_total_excess"],
                "mean_residual_fixed": fit_result["diagnostics"]["mean_residual_fixed"],
                "mean_dyn_contrib_fixed": fit_result["diagnostics"]["mean_dyn_contrib_fixed"],
                "formula_total": "I_CHW * max(T_raw - Q90, 0)",
                "formula_residual_fixed": "I_CHW * max(T_raw - delta_T_dyn - Q90, 0)",
                "formula_dyn": "total - residual_fixed",
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
        spatial_out[f"{var_name}_hwc_dyn_contrib_fixed_slope_per_year"] = xr.concat(dyn_list, dim="region")
        spatial_out[f"{var_name}_hwc_residual_fixed_slope_per_year"] = xr.concat(residual_list, dim="region")
        spatial_out[f"{var_name}_hwc_dyn_slope_per_year"] = spatial_out[f"{var_name}_hwc_dyn_contrib_fixed_slope_per_year"]
        spatial_out[f"{var_name}_hwc_thermo_slope_per_year"] = spatial_out[f"{var_name}_hwc_residual_fixed_slope_per_year"]

        annual_out[f"{var_name}_hwc_annual_total"] = xr.concat(annual_total_list, dim="region").transpose("year", "region", "lat", "lon").astype("float32")
        annual_out[f"{var_name}_hwc_annual_dyn_contrib_fixed"] = xr.concat(annual_dyn_list, dim="region").transpose("year", "region", "lat", "lon").astype("float32")
        annual_out[f"{var_name}_hwc_annual_residual_fixed"] = xr.concat(annual_residual_list, dim="region").transpose("year", "region", "lat", "lon").astype("float32")
        annual_out[f"{var_name}_hwc_annual_dyn"] = annual_out[f"{var_name}_hwc_annual_dyn_contrib_fixed"]
        annual_out[f"{var_name}_hwc_annual_thermo"] = annual_out[f"{var_name}_hwc_annual_residual_fixed"]
        annual_out[f"{var_name}_hwc_annual_chw_days"] = xr.concat(annual_chw_days_list, dim="region").transpose("year", "region", "lat", "lon").astype("float32")

    return rows, spatial_out, annual_out


def _compute_era5_variant(ns, paths, force=False):
    outputs_exist = all(os.path.exists(paths[key]) for key in ("era5_region", "era5_spatial", "era5_annual"))
    if outputs_exist and not force:
        print(f"ERA5 {paths['tag']} 缓存已存在，跳过重算。", flush=True)
        return {"skipped": True, "region_csv": paths["era5_region"], "spatial_nc": paths["era5_spatial"], "annual_nc": paths["era5_annual"]}

    tmax, tmin = ns["daily_da_open_era5_daily_temps"]()
    z500_rel = ns["daily_da_remove_nh_mean"](ns["daily_da_open_era5_daily_z500"]())
    chw_mask = CHW_MOD._open_era5_chw_mask(ns)
    thresholds = FE_MOD._open_era5_thresholds(ns)
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
    FE_MOD._write_dataset(spatial_out, paths["era5_spatial"])
    FE_MOD._write_dataset(annual_out, paths["era5_annual"])
    print(f"已保存 ERA5 {paths['tag']} 结果。", flush=True)
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
    thresholds = FE_MOD._open_member_thresholds(ns, member)
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
    FE_MOD._write_dataset(spatial_out, spatial_path)
    FE_MOD._write_dataset(annual_out, annual_path)
    print(f"已保存 {member} {paths['tag']} 成员缓存。", flush=True)
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
    df_mme["top_n_warm_days_per_year"] = np.nan
    df_mme["formula_total"] = "I_CHW * max(T_raw - Q90, 0)"
    df_mme["formula_residual_fixed"] = "I_CHW * max(T_raw - delta_T_dyn - Q90, 0)"
    df_mme["formula_dyn"] = "total - residual_fixed"
    df_mme["unit"] = "degC_days_per_year"
    df_mme.to_csv(paths["mme_region"], index=False)

    members_spatial_ds = xr.concat(spatial_list, dim="member")
    FE_MOD._write_dataset(members_spatial_ds, paths["members_spatial"])
    mme_spatial_ds = members_spatial_ds.mean("member", skipna=True)
    FE_MOD._write_dataset(mme_spatial_ds, paths["mme_spatial"])

    members_annual_ds = xr.concat(annual_list, dim="member")
    FE_MOD._write_dataset(members_annual_ds, paths["members_annual"])
    mme_annual_ds = members_annual_ds.mean("member", skipna=True)
    FE_MOD._write_dataset(mme_annual_ds, paths["mme_annual"])

    print(f"已完成 {paths['tag']} 成员汇总、MME 汇总和逐年场汇总。", flush=True)
    return {"valid_members": valid_members, "n_members": len(valid_members)}


def preflight_variant(ns, spec):
    paths = _build_variant_paths(spec)
    test_region_name, test_box = ns["daily_da_get_region_settings"]()[0]
    tmax_era, _ = ns["daily_da_open_era5_daily_temps"]()
    z_era = ns["daily_da_remove_nh_mean"](ns["daily_da_open_era5_daily_z500"]())
    chw_era = CHW_MOD._open_era5_chw_mask(ns)
    th_era = FE_MOD._open_era5_thresholds(ns)["tmax"]

    y_region = ns["daily_da_select_box_wrap"](tmax_era, test_box)
    chw_region = ns["daily_da_select_box_wrap"](chw_era, test_box)
    th_region = ns["daily_da_select_box_wrap"](th_era, test_box)

    prepared = _prepare_cfexcess_daily_matrices(ns, z_era, y_region, chw_region, th_region)
    fit = _fit_region_by_mode(
        ns,
        z_era,
        y_region,
        chw_region,
        th_region,
        training_mode=paths["training_mode"],
        top_n_days=paths["top_n"],
        max_targets_for_cv=12,
    )

    delta_dyn = (
        fit["daily_delta_dyn_temp"]
        .stack(target=("lat", "lon"))
        .transpose("time", "target")
        .sel(target=prepared["y_stack"]["target"])
        .values
    )
    total_out = (
        fit["daily_total_excess"]
        .stack(target=("lat", "lon"))
        .transpose("time", "target")
        .sel(target=prepared["y_stack"]["target"])
        .values
    )
    residual_out = (
        fit["daily_residual_fixed"]
        .stack(target=("lat", "lon"))
        .transpose("time", "target")
        .sel(target=prepared["y_stack"]["target"])
        .values
    )
    dyn_contrib_out = (
        fit["daily_dyn_contrib_fixed"]
        .stack(target=("lat", "lon"))
        .transpose("time", "target")
        .sel(target=prepared["y_stack"]["target"])
        .values
    )

    display_mask = prepared["valid_display"]
    expected_total = np.where(
        display_mask,
        np.maximum(prepared["Y_total_full"] - prepared["threshold_full"], 0.0),
        np.nan,
    )
    pred_mask = display_mask & np.isfinite(delta_dyn)
    expected_residual = np.where(
        pred_mask,
        np.maximum(prepared["Y_total_full"] - delta_dyn - prepared["threshold_full"], 0.0),
        np.nan,
    )
    expected_dyn = expected_total - expected_residual

    def _max_abs_diff(lhs, rhs, mask):
        use = mask & np.isfinite(lhs) & np.isfinite(rhs)
        if not np.any(use):
            return np.nan
        return float(np.nanmax(np.abs(lhs[use] - rhs[use])))

    summary = {
        "tag": paths["tag"],
        "test_region": test_region_name,
        "max_abs_total_formula_diff": _max_abs_diff(total_out, expected_total, display_mask),
        "max_abs_residual_formula_diff": _max_abs_diff(residual_out, expected_residual, pred_mask),
        "max_abs_dyn_formula_diff": _max_abs_diff(dyn_contrib_out, expected_dyn, pred_mask),
        "mean_T_raw": fit["diagnostics"]["mean_T_raw"],
        "mean_T_clim": fit["diagnostics"]["mean_T_clim"],
        "mean_delta_dyn": fit["diagnostics"]["mean_delta_dyn"],
        "mean_Q90": fit["diagnostics"]["mean_Q90"],
        "mean_total_excess": fit["diagnostics"]["mean_total_excess"],
        "mean_residual_fixed": fit["diagnostics"]["mean_residual_fixed"],
        "train_samples_min": int(np.nanmin(fit["train_sample_counts"])),
        "train_samples_max": int(np.nanmax(fit["train_sample_counts"])),
        "display_samples_min": int(np.nanmin(fit["display_sample_counts"])),
        "display_samples_max": int(np.nanmax(fit["display_sample_counts"])),
        "selected_alpha": float(fit.get("selected_alpha", np.nan)),
    }
    print("Preflight summary:", summary, flush=True)

    tol = 1e-5
    for key in ("max_abs_total_formula_diff", "max_abs_residual_formula_diff", "max_abs_dyn_formula_diff"):
        if np.isfinite(summary[key]) and summary[key] > tol:
            raise RuntimeError(f"Preflight 失败：{key} 过大 -> {summary[key]}")
    return summary


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
    if run_era5:
        out["era5"] = _compute_era5_variant(ns, paths, force=force_era5)
    if run_cmip:
        member_results = []
        for member in available_members:
            print(f"[{paths['tag']}] 成员开始：{member}", flush=True)
            member_results.append(_compute_member_cache_variant(ns, paths, member, source_table=source_table, force=force_members))
            print(f"[{paths['tag']}] 成员完成：{member}", flush=True)
        out["member_results"] = member_results

    has_any_member_cache = any(
        os.path.exists(_member_region_path(paths, member))
        and os.path.exists(_member_spatial_path(paths, member))
        and os.path.exists(_member_annual_path(paths, member))
        for member in available_members
    )
    need_summary = False
    if not skip_summary and available_members:
        need_summary = (
            force_summary
            or run_cmip
            or (
                has_any_member_cache
                and (
                    (not os.path.exists(paths["members_region"]))
                    or (not os.path.exists(paths["mme_region"]))
                    or (not os.path.exists(paths["members_spatial"]))
                    or (not os.path.exists(paths["mme_spatial"]))
                    or (not os.path.exists(paths["members_annual"]))
                    or (not os.path.exists(paths["mme_annual"]))
                )
            )
        )
    if need_summary and available_members:
        out["summary"] = _aggregate_member_caches_variant(paths, model_list=available_members)
    return out


def main():
    ns = FE_MOD._load_daily_namespace()
    specs = _variant_specs()
    run_era5 = _env_flag("DA_RUN_ERA5", default=True)
    run_cmip = _env_flag("DA_RUN_CMIP", default=True)
    force_era5 = _env_flag("DA_FORCE_ERA5", default=False)
    force_members = _env_flag("DA_FORCE_MEMBERS", default=False)
    force_summary = _env_flag("DA_FORCE_SUMMARY", default=False)
    skip_summary = _env_flag("DA_SKIP_SUMMARY", default=False)
    member_subset = _env_list("DA_MEMBER_SUBSET", default=None) or None
    skip_preflight = _env_flag("DA_SKIP_PREFLIGHT", default=False)

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
            "member_subset": member_subset,
            "skip_preflight": skip_preflight,
            "target_n_jobs": TARGET_N_JOBS,
        },
        flush=True,
    )

    outputs = []
    for spec in specs:
        print(f"开始处理 {spec['tag']} 版本。", flush=True)
        if not skip_preflight:
            outputs.append({"preflight": preflight_variant(ns, spec)})
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
