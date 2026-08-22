"""
Ridge-regression dynamical adjustment (DA) using Z500 as circulation predictor.

This module is designed to mirror the *idea* described in Vautard et al. (2023)
(Nat. Commun.; see the PMC mirror for the quoted Methods text) while being
practical for a Northern Hemisphere (NH)-only dataset and for running the same
workflow across many regions.

Key design choices (explicitly configurable):
1) Z500 "thermal expansion" removal:
   - Paper: subtract global-mean Z500 at each time step over the circulation domain,
     so only *relative spatial* changes remain.
   - Here: you can subtract the **NH mean** at each time step (lat >= 0), which is
     appropriate when your data domain is NH-only.

2) Regional workflow:
   - You can precompute the NH-mean-subtracted Z500 once for the full NH field,
     then for each target region select a circulation domain that is the target box
     expanded by +/-15 degrees in lat/lon (clamped to data bounds).

3) Standardization:
   - Ridge penalty is scale-sensitive (feature scaling changes the effective penalty).
     The paper does not explicitly mention standardization; in practice it is common
     to standardize predictors. This module defaults to standardizing predictors (X)
     but allows turning it off to mimic a "raw predictors" workflow.

Dependencies: numpy, pandas, xarray, scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

MeanRef = Literal["global", "nh", "sh"]
Standardize = Literal["zscore", "center", "none"]


def _as_months(months: Sequence[int]) -> Tuple[int, ...]:
    months = tuple(int(m) for m in months)
    if any(m < 1 or m > 12 for m in months):
        raise ValueError(f"months must be 1..12, got {months}")
    return months


def _infer_lat_lon_names(
    da: xr.DataArray, lat_name: Optional[str], lon_name: Optional[str]
) -> Tuple[str, str]:
    if lat_name is None:
        for cand in ("lat", "latitude", "LAT", "nav_lat"):
            if cand in da.dims or cand in da.coords:
                lat_name = cand
                break
    if lon_name is None:
        for cand in ("lon", "longitude", "LON", "nav_lon"):
            if cand in da.dims or cand in da.coords:
                lon_name = cand
                break
    if lat_name is None or lon_name is None:
        raise ValueError(
            "Could not infer lat/lon names; pass lat_name/lon_name. "
            f"dims={da.dims}, coords={list(da.coords)}"
        )
    return lat_name, lon_name


def _require_time(da: xr.DataArray, time_name: str) -> None:
    if time_name not in da.dims:
        raise ValueError(f"Expected {time_name!r} in dims, got {da.dims}")
    if not hasattr(da[time_name].dt, "month"):
        raise ValueError(f"{time_name!r} must be datetime-like")


def _coslat_weights(lat: xr.DataArray) -> xr.DataArray:
    """Area weights proportional to cos(latitude) for regular lat/lon grids."""
    w = np.cos(np.deg2rad(lat.astype(float)))
    # clip: numerical safety near poles
    return xr.DataArray(w.clip(min=0.0), dims=(lat.dims[0],), coords={lat.dims[0]: lat})


def select_months(da: xr.DataArray, months: Sequence[int], time_name: str = "time") -> xr.DataArray:
    """Select specified months from a time series."""
    _require_time(da, time_name)
    months = _as_months(months)
    return da.sel({time_name: da[time_name].dt.month.isin(months)})


def monthly_anomalies(da: xr.DataArray, time_name: str = "time") -> xr.DataArray:
    """
    Monthly anomalies: remove the monthly climatology.

    For monthly mean data (your case), this is the standard "anomaly" definition:
    da(t) - mean_{years}( da(month=month(t)) ).
    """
    _require_time(da, time_name)
    clim = da.groupby(f"{time_name}.month").mean(time_name, skipna=True)
    return da.groupby(f"{time_name}.month") - clim


def _year_groups(time_index: xr.DataArray) -> np.ndarray:
    """Groups for CV: year of each sample (prevents leakage across same-year months)."""
    dt = pd.DatetimeIndex(time_index.values)
    return dt.year.astype(int).to_numpy()


def _lon_convention(lon: xr.DataArray) -> Literal["-180_180", "0_360", "other"]:
    """Rudimentary inference of lon convention."""
    vmin = float(lon.min())
    vmax = float(lon.max())
    if vmin >= 0.0 and vmax <= 360.0:
        return "0_360"
    if vmin >= -180.0 and vmax <= 180.0:
        return "-180_180"
    return "other"


def _wrap_lon(value: float, convention: Literal["-180_180", "0_360", "other"]) -> float:
    """Wrap a longitude into the dataset's convention."""
    if convention == "0_360":
        v = value % 360.0
        return v if v >= 0.0 else v + 360.0
    if convention == "-180_180":
        v = ((value + 180.0) % 360.0) - 180.0
        # map -180 to 180 if dataset uses 180 endpoint; keep as-is (xarray slice ok)
        return v
    return value


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))


@dataclass(frozen=True)
class SpatialBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def bounds_from_data(da: xr.DataArray, lat_name: str, lon_name: str) -> SpatialBounds:
    """Compute bounds from a DataArray's coordinates (assumes contiguous region)."""
    lat_min = float(da[lat_name].min())
    lat_max = float(da[lat_name].max())
    lon_min = float(da[lon_name].min())
    lon_max = float(da[lon_name].max())
    return SpatialBounds(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


def expand_bounds(
    b: SpatialBounds,
    expand_deg: float,
    lat_limit: Tuple[float, float],
    lon_limit: Tuple[float, float],
    lon_convention: Literal["-180_180", "0_360", "other"],
) -> SpatialBounds:
    """
    Expand bounds by +/- expand_deg and clamp to the available data limits.

    Note: if the expansion causes lon_min > lon_max under wrapping conventions,
    you will need a wrap-aware selection. `select_box()` handles that.
    """
    lat_min = _clamp(b.lat_min - expand_deg, lat_limit[0], lat_limit[1])
    lat_max = _clamp(b.lat_max + expand_deg, lat_limit[0], lat_limit[1])

    lon_min = _wrap_lon(b.lon_min - expand_deg, lon_convention)
    lon_max = _wrap_lon(b.lon_max + expand_deg, lon_convention)

    # clamp within numeric lon limits of the dataset (after wrapping)
    lon_min = _clamp(lon_min, lon_limit[0], lon_limit[1])
    lon_max = _clamp(lon_max, lon_limit[0], lon_limit[1])

    return SpatialBounds(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


def select_box(
    da: xr.DataArray,
    bounds: SpatialBounds,
    lat_name: str,
    lon_name: str,
) -> xr.DataArray:
    """
    Select a lat/lon rectangle, supporting dateline-crossing lon ranges.

    If bounds.lon_min <= bounds.lon_max: simple slice.
    Else (crosses wrap seam): concatenate two slices.
    """
    da = da.sel({lat_name: slice(bounds.lat_min, bounds.lat_max)})
    if bounds.lon_min <= bounds.lon_max:
        return da.sel({lon_name: slice(bounds.lon_min, bounds.lon_max)})

    # dateline-crossing: [lon_min .. max] U [min .. lon_max]
    lon = da[lon_name]
    lo = float(lon.min())
    hi = float(lon.max())
    part1 = da.sel({lon_name: slice(bounds.lon_min, hi)})
    part2 = da.sel({lon_name: slice(lo, bounds.lon_max)})
    return xr.concat([part1, part2], dim=lon_name)


def subtract_reference_mean(
    z500: xr.DataArray,
    *,
    reference: MeanRef = "nh",
    time_name: str = "time",
    lat_name: Optional[str] = None,
    lon_name: Optional[str] = None,
) -> xr.DataArray:
    """
    Subtract a large-scale mean Z500 at each time step (scalar) from the full field.

    Paper idea: subtract global mean to remove the part of Z500 changes driven by
    thermal expansion (a near-uniform thickness increase), leaving relative patterns.

    For NH-only datasets, subtracting the NH mean is a practical analogue.

    Implementation details:
    - Uses cos(lat) area-weighting.
    - For "nh": uses lat >= 0
    - For "sh": uses lat <= 0
    - For "global": uses all available latitudes in the dataset
    """
    _require_time(z500, time_name)
    lat_name, lon_name = _infer_lat_lon_names(z500, lat_name, lon_name)

    z = z500.transpose(time_name, lat_name, lon_name)
    lat = z[lat_name]

    if reference == "nh":
        z = z.sel({lat_name: lat >= 0})
    elif reference == "sh":
        z = z.sel({lat_name: lat <= 0})
    elif reference == "global":
        pass
    else:
        raise ValueError(f"Unknown reference={reference!r}")

    w = _coslat_weights(z[lat_name])
    mean_ts = z.weighted(w).mean((lat_name, lon_name))  # (time,)

    # subtract from the *original* (not lat-subset) field so output keeps full domain
    return (z500 - mean_ts).rename(z500.name or "z500_rel")


def standardize_matrix(X: np.ndarray, mode: Standardize) -> Tuple[np.ndarray, Optional[StandardScaler]]:
    """
    Standardize predictors for ridge regression.

    - "zscore": (X - mean)/std  (recommended for ridge)
    - "center": (X - mean)      (keeps original variance; penalty depends on scale)
    - "none": no scaling
    """
    if mode == "none":
        return X, None
    if mode == "center":
        scaler = StandardScaler(with_mean=True, with_std=False)
        return scaler.fit_transform(X), scaler
    if mode == "zscore":
        scaler = StandardScaler(with_mean=True, with_std=True)
        return scaler.fit_transform(X), scaler
    raise ValueError(f"Unknown standardize mode: {mode!r}")


def choose_ridge_alpha(
    X: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: Sequence[float],
    n_splits: int = 5,
    max_targets_for_cv: int = 4000,
    random_state: int = 0,
) -> float:
    """
    Select ridge alpha by GroupKFold CV (grouped by year).

    Why grouping by year?
      For monthly JJA, samples within the same year are correlated; splitting them
      across folds can overstate out-of-sample skill.

    Why sample targets?
      If Y is a full grid, it can have 10^5 targets; CV over all targets is slow.
      For alpha selection, sampling a few thousand targets is usually enough.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays.")
    if X.shape[0] != Y.shape[0] or X.shape[0] != groups.shape[0]:
        raise ValueError("X, Y, groups must share n_samples (time).")

    rng = np.random.default_rng(random_state)
    if Y.shape[1] > max_targets_for_cv:
        idx = rng.choice(Y.shape[1], size=max_targets_for_cv, replace=False)
        Y_cv = Y[:, idx]
    else:
        Y_cv = Y

    unique_groups = np.unique(groups)
    splits = int(min(n_splits, unique_groups.size))
    if splits < 2:
        raise ValueError("Not enough years/groups for CV (need >=2).")

    gkf = GroupKFold(n_splits=splits)

    best_alpha: Optional[float] = None
    best_mse = np.inf

    for a in alphas:
        fold_mse = []
        for tr, te in gkf.split(X, Y_cv, groups=groups):
            m = Ridge(alpha=float(a), fit_intercept=True)
            m.fit(X[tr], Y_cv[tr])
            pred = m.predict(X[te])
            fold_mse.append(np.mean((Y_cv[te] - pred) ** 2))
        mse = float(np.mean(fold_mse))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(a)

    if best_alpha is None:
        raise RuntimeError("alpha selection failed")
    return best_alpha


def dynamical_adjustment_ridge(
    *,
    z500_full: xr.DataArray,
    y_region: xr.DataArray,
    months: Sequence[int] = (6, 7, 8),
    time_name: str = "time",
    z_lat_name: Optional[str] = None,
    z_lon_name: Optional[str] = None,
    y_lat_name: Optional[str] = None,
    y_lon_name: Optional[str] = None,
    # preprocessing
    subtract_mean_reference: Optional[MeanRef] = "nh",
    predictor_expand_deg: float = 15.0,
    # regression
    standardize_X: Standardize = "zscore",
    alphas: Optional[Sequence[float]] = None,
    cv_year_folds: int = 5,
    max_targets_for_cv: int = 4000,
    random_state: int = 0,
) -> xr.Dataset:
    """
    Perform ridge-regression DA for one target region.

    Inputs:
      z500_full: Z500 field over (at least) NH domain, dims (time, lat, lon)
      y_region: Target temperature field for a region, dims (time, lat, lon) or (time,)

    Regional circulation domain:
      Derived from y_region bounds expanded by +/- predictor_expand_deg in each direction,
      then clamped to z500_full's bounds.

    Returns (Dataset):
      - y_sel, y_anom
      - y_dyn_anom (circulation-induced part of anomalies)
      - y_adj (y_sel - y_dyn_anom) and y_adj_anom (y_anom - y_dyn_anom)
      - r2 (per grid cell if y is gridded)
      - attrs: selected_alpha, bounds used, etc.
    """
    if alphas is None:
        alphas = np.logspace(-2, 4, 25)

    months = _as_months(months)
    _require_time(z500_full, time_name)
    _require_time(y_region, time_name)

    z_lat_name, z_lon_name = _infer_lat_lon_names(z500_full, z_lat_name, z_lon_name)
    if y_region.ndim >= 3:
        y_lat_name, y_lon_name = _infer_lat_lon_names(y_region, y_lat_name, y_lon_name)

    # --- 1) Z500 preprocessing: subtract NH mean at each timestep (or global/sh)
    # If you want to preprocess once outside and reuse across many regions, pass
    # an already-processed z500_full and set subtract_mean_reference=None here.
    if subtract_mean_reference is None:
        z500_rel = z500_full
    else:
        z500_rel = subtract_reference_mean(
            z500_full,
            reference=subtract_mean_reference,
            time_name=time_name,
            lat_name=z_lat_name,
            lon_name=z_lon_name,
        )

    # --- 2) Build circulation domain from y_region bounds expanded by +/-15°
    # Determine y bounds (in the same lon convention as y_region)
    if y_region.ndim >= 3:
        yb = bounds_from_data(y_region, y_lat_name, y_lon_name)
    else:
        # 1D target series: cannot infer region box; user should pass gridded y_region
        # (or you can modify this function to accept explicit region bounds)
        raise ValueError("y_region must be gridded (time, lat, lon) to infer bounds for predictor domain.")

    z_lon = z500_rel[z_lon_name]
    lon_conv = _lon_convention(z_lon)
    # wrap y bounds into z500 lon convention (if needed)
    yb_wrapped = SpatialBounds(
        lat_min=yb.lat_min,
        lat_max=yb.lat_max,
        lon_min=_wrap_lon(yb.lon_min, lon_conv),
        lon_max=_wrap_lon(yb.lon_max, lon_conv),
    )

    z_lat_lim = (float(z500_rel[z_lat_name].min()), float(z500_rel[z_lat_name].max()))
    z_lon_lim = (float(z500_rel[z_lon_name].min()), float(z500_rel[z_lon_name].max()))

    pred_bounds = expand_bounds(
        yb_wrapped,
        expand_deg=float(predictor_expand_deg),
        lat_limit=z_lat_lim,
        lon_limit=z_lon_lim,
        lon_convention=lon_conv,
    )

    z500_dom = select_box(z500_rel, pred_bounds, z_lat_name, z_lon_name)

    # --- 3) Select months and align times
    z_sel = select_months(z500_dom, months, time_name=time_name)
    y_sel = select_months(y_region, months, time_name=time_name)
    z_sel, y_sel = xr.align(z_sel, y_sel, join="inner")

    # --- 4) Convert to anomalies (monthly climatology removed)
    z_anom = monthly_anomalies(z_sel, time_name=time_name)
    y_anom = monthly_anomalies(y_sel, time_name=time_name)

    # --- 5) Build regression matrices
    # X: (time, features) where features are circulation grid cells in the expanded domain
    z_stack = z_anom.transpose(time_name, z_lat_name, z_lon_name).stack(feature=(z_lat_name, z_lon_name))
    good_x = np.isfinite(z_stack).all(time_name)
    z_stack = z_stack.sel(feature=good_x)
    X = z_stack.values
    X, scaler = standardize_matrix(X, standardize_X)

    # Y: (time, targets) where targets are y grid cells in the region
    y_stack = y_anom.transpose(time_name, y_lat_name, y_lon_name).stack(target=(y_lat_name, y_lon_name))
    good_y = np.isfinite(y_stack).all(time_name)
    y_stack = y_stack.sel(target=good_y)
    Y = y_stack.values
    target_index = y_stack["target"].values

    if Y.shape[1] < 1:
        raise ValueError("No valid target grid cells (finite across time).")
    if X.shape[1] < 2:
        raise ValueError("Too few predictor grid cells after masking.")

    # --- 6) Choose alpha and fit ridge
    groups = _year_groups(y_sel[time_name])
    alpha = choose_ridge_alpha(
        X,
        Y,
        groups,
        alphas=alphas,
        n_splits=cv_year_folds,
        max_targets_for_cv=max_targets_for_cv,
        random_state=random_state,
    )

    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(X, Y)
    Y_hat = model.predict(X)  # (time, targets) => circulation-induced anomalies

    # --- 7) Diagnostics: R^2 on anomalies
    ss_res = np.sum((Y - Y_hat) ** 2, axis=0)
    ss_tot = np.sum((Y - np.mean(Y, axis=0, keepdims=True)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    r2 = np.where(np.isfinite(r2), r2, np.nan)

    # --- 8) Unstack back to grids and compute adjusted fields
    # y_dyn_anom in full region grid
    full_target = y_sel.transpose(time_name, y_lat_name, y_lon_name).stack(target=(y_lat_name, y_lon_name))
    y_dyn_stack = xr.full_like(full_target, np.nan).rename("y_dyn_anom_stack")
    y_dyn_stack.loc[dict(target=target_index)] = Y_hat
    y_dyn_anom = (
        y_dyn_stack.unstack("target")
        .transpose(time_name, y_lat_name, y_lon_name)
        .rename("y_dyn_anom")
    )

    y_adj = (y_sel.transpose(time_name, y_lat_name, y_lon_name) - y_dyn_anom).rename("y_adj")
    y_adj_anom = (y_anom.transpose(time_name, y_lat_name, y_lon_name) - y_dyn_anom).rename("y_adj_anom")

    r2_stack = xr.full_like(full_target.isel({time_name: 0}, drop=True), np.nan).rename("r2_stack")
    r2_stack.loc[dict(target=target_index)] = r2
    r2_da = r2_stack.unstack("target").transpose(y_lat_name, y_lon_name).rename("r2")

    ds = xr.Dataset(
        {
            "y_sel": y_sel.rename("y_sel"),
            "y_anom": y_anom.rename("y_anom"),
            "y_dyn_anom": y_dyn_anom,
            "y_adj": y_adj,
            "y_adj_anom": y_adj_anom,
            "r2": r2_da,
        }
    )
    ds.attrs.update(
        {
            "selected_alpha": float(alpha),
            "months": ",".join(str(m) for m in months),
            "subtract_mean_reference": subtract_mean_reference if subtract_mean_reference is not None else "none",
            "predictor_expand_deg": float(predictor_expand_deg),
            "predictor_bounds_lat_min": float(pred_bounds.lat_min),
            "predictor_bounds_lat_max": float(pred_bounds.lat_max),
            "predictor_bounds_lon_min": float(pred_bounds.lon_min),
            "predictor_bounds_lon_max": float(pred_bounds.lon_max),
            "standardize_X": standardize_X,
            "notes": "DA via ridge: y_adj = y_sel - y_dyn_anom; computed on monthly anomalies for selected months.",
        }
    )
    return ds
