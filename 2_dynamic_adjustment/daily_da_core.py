"""Daily ridge dynamic-adjustment functions extracted from the verified analysis notebook."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chw_cmip6.scientific_constants import GIC_BOUNDS

DATA_ROOT = Path(os.environ.get("CHW_DATA_ROOT", "data"))
CACHE_ROOT = Path(os.environ.get("CHW_CACHE_ROOT", "cache"))


import os
import warnings
import calendar
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.ticker import FuncFormatter
from scipy.stats import linregress
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', message='.*All-NaN slice encountered.*')

DAILY_DA_HIST_START = '1981-01-01'
DAILY_DA_HIST_END = '2014-12-31'
DAILY_DA_JJA_MONTHS = (6, 7, 8)

DAILY_DA_DEFAULT_MODELS = [
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'BCC-ESM1', 'CanESM5',
    'CMCC-ESM2', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'E3SM-2-0', 'E3SM-2-0-NARRM',
    'EC-Earth3', 'EC-Earth3-AerChem', 'EC-Earth3-CC', 'EC-Earth3-Veg-LR',
    'FGOALS-f3-L', 'FGOALS-g3', 'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM',
    'IITM-ESM', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G',
    'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'NorESM2-LM', 'NorESM2-MM',
    'TaiESM1', 'UKESM1-0-LL'
]

DAILY_DA_HOT_REGIONS = [(-115, -97, 44, 59), (-14, 3, 12, 28), (54, 71, 56, 65), (69, 104, 27, 35), (100, 121, 54, 65)]
DAILY_DA_NOTHOT_REGIONS = [(-106, -87, 30, 38), GIC_BOUNDS, (24, 38, 47, 66), (100, 119, 38, 52)]
DAILY_DA_HOT_NAMES = ['NNA', 'NAF', 'EEU', 'SAS', 'ESB']
DAILY_DA_NOTHOT_NAMES = ['SNA', 'GIC', 'WEU', 'ENA']


def daily_da_make_map(ax, box):
    ax.set_xticks(np.arange(box[0], box[1], 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(box[2], box[3] + 30, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='both', which='major', labelsize=16, direction='out', length=5, width=1, pad=5)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis='both', which='minor', direction='out', length=2, width=0.4)
    ax.spines['geo'].set_linewidth(1)
    ax.set_extent([box[0], box[1], box[2], box[3]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.6)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white', zorder=2)
    return ax


def daily_da_draw_rectangles(ax, boxes, edgecolor='green', linewidth=2.0):
    for lon_min, lon_max, lat_min, lat_max in boxes:
        rect = mpatches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor='none',
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        ax.add_patch(rect)


def daily_da_filp_lon(ds, lon_name='longitude'):
    if lon_name == 'lon':
        ds = ds.sortby('lat', ascending=True)
    else:
        ds = ds.sortby('latitude', ascending=True)
    ds['longitude_adjusted'] = xr.where(ds[lon_name] > 180, ds[lon_name] - 360, ds[lon_name])
    ds = (
        ds
        .swap_dims({lon_name: 'longitude_adjusted'})
        .sel(**{'longitude_adjusted': sorted(ds.longitude_adjusted)})
        .drop(lon_name)
    )
    ds = ds.rename({'longitude_adjusted': 'lon'})
    if 'latitude' in ds.dims or 'latitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat'})
    if 'valid_time' in ds.dims or 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    return ds


def daily_da_standardize_coords(obj):
    rename = {}
    if 'latitude' in obj.dims or 'latitude' in obj.coords:
        rename['latitude'] = 'lat'
    if 'longitude' in obj.dims or 'longitude' in obj.coords:
        rename['longitude'] = 'lon'
    if rename:
        obj = obj.rename(rename)
    if 'lon' in obj.coords:
        lon = obj['lon'].astype(float)
        if float(lon.max()) > 180.0:
            lon = xr.where(lon > 180.0, lon - 360.0, lon)
            obj = obj.assign_coords(lon=lon)
        obj = obj.sortby('lon')
    if 'lat' in obj.coords and obj['lat'].ndim == 1 and obj['lat'].size > 1:
        if float(obj['lat'][0]) > float(obj['lat'][-1]):
            obj = obj.sortby('lat')
    if 'time' in obj.coords:
        raw_time = obj['time'].values
        try:
            time_values = pd.to_datetime(raw_time)
        except Exception:
            try:
                time_index = obj['time'].to_index()
                time_calendar = getattr(time_index, 'calendar', None)
                if time_calendar == '360_day':
                    obj = obj.convert_calendar('proleptic_gregorian', use_cftime=False, align_on='date')
                else:
                    obj = obj.convert_calendar('proleptic_gregorian', use_cftime=False)
                raw_time = obj['time'].values
                time_values = pd.to_datetime(raw_time)
            except Exception:
                time_values = []
                for t in raw_time:
                    year = int(t.year)
                    month = int(t.month)
                    day = int(t.day)
                    last_day = calendar.monthrange(year, month)[1]
                    safe_day = min(day, last_day)
                    time_values.append(pd.Timestamp(year=year, month=month, day=safe_day))
        obj = obj.assign_coords(time=pd.DatetimeIndex(time_values).normalize())
    return obj


def daily_da_subset_time_fast(obj, start=DAILY_DA_HIST_START, end=DAILY_DA_HIST_END):
    if 'time' not in obj.coords:
        return obj
    try:
        return obj.sel(time=slice(start, end))
    except Exception:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        raw_time = obj['time'].values
        mask = []
        for t in raw_time:
            y = int(t.year)
            m = int(t.month)
            d = int(t.day)
            keep = (y, m, d) >= (start_ts.year, start_ts.month, start_ts.day) and (y, m, d) <= (end_ts.year, end_ts.month, end_ts.day)
            mask.append(keep)
        mask = np.asarray(mask, dtype=bool)
        return obj.isel(time=np.where(mask)[0])


def daily_da_get_var_name(ds, preferred=None):
    if preferred and preferred in ds.data_vars:
        return preferred
    for name in ('t2m', 'tasmax', 'tasmin', 'zg', 'z'):
        if name in ds.data_vars:
            return name
    return list(ds.data_vars)[0]


def daily_da_select_jja(da):
    da = daily_da_standardize_coords(da)
    return da.where(da['time'].dt.month.isin(DAILY_DA_JJA_MONTHS), drop=True)


def daily_da_calendar_day_anomaly(da):
    da = daily_da_standardize_coords(da)
    key = xr.DataArray(
        da['time'].dt.month.astype(int) * 100 + da['time'].dt.day.astype(int),
        dims=('time',),
        coords={'time': da['time']},
        name='month_day_key',
    )
    clim = da.groupby(key).mean('time', skipna=True)
    return da.groupby(key) - clim


def daily_da_coslat_weights(lat):
    values = np.cos(np.deg2rad(lat.astype(float)))
    return xr.DataArray(values.clip(min=0.0), dims=(lat.dims[0],), coords={lat.dims[0]: lat})


def daily_da_area_mean(da):
    da = daily_da_standardize_coords(da)
    weights = daily_da_coslat_weights(da['lat'])
    return da.weighted(weights).mean(('lat', 'lon'), skipna=True)


def daily_da_remove_nh_mean(zg):
    zg = daily_da_standardize_coords(zg)
    nh = zg.sel(lat=slice(0, 90))
    weights = daily_da_coslat_weights(nh['lat'])
    nh_mean = nh.weighted(weights).mean(('lat', 'lon'), skipna=True)
    return (zg - nh_mean).rename(zg.name or 'zg_rel')


def daily_da_wrap_lon(lon):
    return ((float(lon) + 180.0) % 360.0) - 180.0


def daily_da_select_box_wrap(da, box):
    da = daily_da_standardize_coords(da)
    lon_min, lon_max, lat_min, lat_max = box
    lat_sel = da.sel(lat=slice(lat_min, lat_max))
    if lon_min <= lon_max:
        return lat_sel.sel(lon=slice(lon_min, lon_max))
    left = lat_sel.sel(lon=slice(lon_min, float(lat_sel['lon'].max())))
    right = lat_sel.sel(lon=slice(float(lat_sel['lon'].min()), lon_max))
    return xr.concat([left, right], dim='lon')


def daily_da_expand_box(box, expand_deg=15.0, lat_bounds=(0.0, 90.0)):
    lon_min, lon_max, lat_min, lat_max = box
    lat_min_new = max(lat_bounds[0], lat_min - expand_deg)
    lat_max_new = min(lat_bounds[1], lat_max + expand_deg)
    lon_min_new = daily_da_wrap_lon(lon_min - expand_deg)
    lon_max_new = daily_da_wrap_lon(lon_max + expand_deg)
    return (lon_min_new, lon_max_new, lat_min_new, lat_max_new)


def daily_da_put_on_template(region_da, template):
    full = template.copy(deep=True)
    full.loc[dict(lat=region_da['lat'], lon=region_da['lon'])] = region_da
    return full


def daily_da_annual_mean(da):
    da = daily_da_standardize_coords(da)
    if 'time' not in da.dims:
        return da
    da = daily_da_select_jja(da)
    return da.groupby('time.year').mean('time', skipna=True)


def daily_da_trend_map_from_annual(da):
    annual = daily_da_annual_mean(da) if 'time' in da.dims else da
    fit = annual.polyfit(dim='year', deg=1, skipna=True)
    return fit['polyfit_coefficients'].sel(degree=1)


def daily_da_series_trend(ts):
    annual = daily_da_annual_mean(ts) if 'time' in ts.dims else ts
    years = np.asarray(annual['year'].values, dtype=float)
    values = np.asarray(annual.values, dtype=float)
    mask = np.isfinite(years) & np.isfinite(values)
    if mask.sum() < 3:
        return {'slope': np.nan, 'pvalue': np.nan}
    fit = linregress(years[mask], values[mask])
    return {'slope': float(fit.slope), 'pvalue': float(fit.pvalue)}


def daily_da_standardize_matrix(X, mode='zscore'):
    if mode == 'none':
        return X, None
    if mode == 'center':
        scaler = StandardScaler(with_mean=True, with_std=False)
        return scaler.fit_transform(X), scaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(X), scaler


def daily_da_choose_alpha(X, Y, groups, alphas=None, n_splits=5, max_targets_for_cv=400, random_state=0):
    if alphas is None:
        alphas = np.logspace(-2, 4, 25)
    rng = np.random.default_rng(random_state)
    if Y.shape[1] > max_targets_for_cv:
        idx = rng.choice(Y.shape[1], size=max_targets_for_cv, replace=False)
        Y_cv = Y[:, idx]
    else:
        Y_cv = Y
    unique_groups = np.unique(groups)
    n_splits = int(min(n_splits, unique_groups.size))
    if n_splits < 2:
        raise ValueError('可用于交叉验证的年份不足。')
    gkf = GroupKFold(n_splits=n_splits)
    best_alpha = None
    best_mse = np.inf
    for alpha in alphas:
        mse_list = []
        for train_idx, test_idx in gkf.split(X, Y_cv, groups=groups):
            model = Ridge(alpha=float(alpha), fit_intercept=True)
            model.fit(X[train_idx], Y_cv[train_idx])
            pred = model.predict(X[test_idx])
            mse_list.append(np.mean((Y_cv[test_idx] - pred) ** 2))
        mse = float(np.mean(mse_list))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    return best_alpha


def daily_da_stack_feature_matrix(z_anom):
    z_stack = z_anom.transpose('time', 'lat', 'lon').stack(feature=('lat', 'lon'))
    good = np.isfinite(z_stack).all('time')
    z_stack = z_stack.sel(feature=good)
    return z_stack.values, z_stack


def daily_da_stack_target_matrix(y_anom):
    y_stack = y_anom.transpose('time', 'lat', 'lon').stack(target=('lat', 'lon'))
    good = np.isfinite(y_stack).all('time')
    y_stack = y_stack.sel(target=good)
    return y_stack.values, y_stack


def daily_da_mask_landsea(ds, lat_name='lat', label='land'):
    landsea = xr.open_dataset(DATA_ROOT / "static" / "landsea.nc")
    landsea = daily_da_filp_lon(landsea, lon_name='longitude')
    landsea = landsea['lsm'][0, :, :]
    rename_dict = {}
    if 'latitude' in landsea.coords:
        rename_dict['latitude'] = 'lat'
    if 'longitude' in landsea.coords:
        rename_dict['longitude'] = 'lon'
    if rename_dict:
        landsea = landsea.rename(rename_dict)
    rename_dict = {}
    if 'latitude' in ds.coords:
        rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        rename_dict['longitude'] = 'lon'
    if rename_dict:
        ds = ds.rename(rename_dict)
    if lat_name == 'lat':
        landsea = landsea.interp(lat=ds.lat.values, lon=ds.lon.values)
        ds.coords['mask'] = (('lat', 'lon'), landsea.values)
    if label == 'land':
        ds = ds.where(ds.mask < 0.8)
    elif label == 'ocean':
        ds = ds.where(ds.mask > 0.2)
    ds = daily_da_filp_lon(ds, lon_name='lon')
    return ds



def daily_da_get_region_settings():
    return list(zip(DAILY_DA_HOT_NAMES, DAILY_DA_HOT_REGIONS)) + list(zip(DAILY_DA_NOTHOT_NAMES, DAILY_DA_NOTHOT_REGIONS))


def daily_da_get_model_list():
    return list(DAILY_DA_DEFAULT_MODELS)


def daily_da_open_era5_daily_temps():
    if 'mera5_tmax' in globals() and 'mera5_tmin' in globals():
        tmax = daily_da_standardize_coords(globals()['mera5_tmax'])
        tmin = daily_da_standardize_coords(globals()['mera5_tmin'])
        tmax = tmax.sel(time=slice(DAILY_DA_HIST_START, DAILY_DA_HIST_END), lat=slice(0, 90))
        tmin = tmin.sel(time=slice(DAILY_DA_HIST_START, DAILY_DA_HIST_END), lat=slice(0, 90))
    else:
        path_tmax = str(DATA_ROOT / "observations" / "ERA5" / "daily" / "surface" / "t2m_max" / "era5_tmax_1980_2023_1x1.nc")
        path_tmin = str(DATA_ROOT / "observations" / "ERA5" / "daily" / "surface" / "t2m_min" / "era5_tmin_1980_2023_1x1.nc")
        ds_tmax = xr.open_dataset(path_tmax)
        ds_tmin = xr.open_dataset(path_tmin)
        ds_tmax = daily_da_standardize_coords(daily_da_subset_time_fast(ds_tmax)).sel(lat=slice(0, 90))
        ds_tmin = daily_da_standardize_coords(daily_da_subset_time_fast(ds_tmin)).sel(lat=slice(0, 90))
        tmax = ds_tmax[daily_da_get_var_name(ds_tmax, 't2m')] - 273.15
        tmin = ds_tmin[daily_da_get_var_name(ds_tmin, 't2m')] - 273.15
        tmax = daily_da_mask_landsea(daily_da_select_jja(tmax), label='ocean')
        tmin = daily_da_mask_landsea(daily_da_select_jja(tmin), label='ocean')
        if 'mask' in tmax.coords:
            tmax = tmax.reset_coords('mask', drop=True)
        if 'mask' in tmin.coords:
            tmin = tmin.reset_coords('mask', drop=True)
    return daily_da_select_jja(tmax), daily_da_select_jja(tmin)


def daily_da_open_era5_daily_z500():
    ds = xr.open_dataset(DATA_ROOT / "observations" / "ERA5" / "daily" / "pressure" / "hgt1980_2023jja.nc")
    ds = daily_da_standardize_coords(ds).sel(time=slice(DAILY_DA_HIST_START, DAILY_DA_HIST_END))
    level_name = 'level' if 'level' in ds.coords else 'plev'
    level_value = 500 if level_name == 'level' else 50000
    zg = ds[daily_da_get_var_name(ds, 'z')].sel({level_name: level_value}) / 9.80665
    zg = zg.sel(lat=slice(0, 90))
    return daily_da_select_jja(zg)


def daily_da_locate_cmip_temp_file(model_name, var_name):
    root = DATA_ROOT / "CMIP6" / "historical" / "1x1" / var_name
    matches = sorted(root.glob(f'*_{model_name}_historical_*.nc'))
    if not matches:
        raise FileNotFoundError(f'未找到 {model_name} 的 {var_name} 日值文件。')
    return str(matches[0])


def daily_da_open_cmip_daily_temperature(model_name, var_name):
    path = daily_da_locate_cmip_temp_file(model_name, var_name)
    ds = xr.open_dataset(path)
    ds = daily_da_standardize_coords(daily_da_subset_time_fast(ds)).sel(lat=slice(0, 90))
    da = ds[daily_da_get_var_name(ds, var_name)] - 273.15
    da = daily_da_select_jja(da)
    da = daily_da_mask_landsea(da, label='ocean')
    if 'mask' in da.coords:
        da = da.reset_coords('mask', drop=True)
    return da


def daily_da_score_z500_path(path):
    path = str(path)
    score = 0
    if '/1x1_all/' in path:
        score += 100
    if 'historical' in path:
        score += 20
    if 'zg_day' in path:
        score += 10
    return score


def daily_da_build_member_source_table(output_csv=None, force=False, model_list=None):
    if output_csv and os.path.exists(output_csv) and not force:
        return pd.read_csv(output_csv)
    if model_list is None:
        model_list = daily_da_get_model_list()
    root = DATA_ROOT / "CMIP6" / "historical_zg_day"
    all_files = [p for p in root.rglob('*.nc') if 'historical' in p.name and 'zg_day' in p.name]
    rows = []
    for model_name in model_list:
        matches = [str(p) for p in all_files if model_name in p.name]
        matches = sorted(matches, key=daily_da_score_z500_path, reverse=True)
        best = matches[0] if matches else ''
        rows.append({
            'member': model_name,
            'available': bool(best),
            'path': best,
            'n_matches': len(matches),
            'source_dir': str(Path(best).parent) if best else '',
        })
    df = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def daily_da_open_cmip_daily_z500(model_name, source_table=None):
    if source_table is None:
        source_table = daily_da_build_member_source_table(model_list=[model_name])
    row = source_table.loc[source_table['member'] == model_name]
    if row.empty or not bool(row['available'].iloc[0]):
        raise FileNotFoundError(f'??? {model_name} ??? Z500 ???')
    path = str(row['path'].iloc[0])
    ds = xr.open_dataset(path)
    ds = daily_da_standardize_coords(daily_da_subset_time_fast(ds))
    level_name = 'plev' if 'plev' in ds.coords else 'level'
    level_coord = ds[level_name].astype(float)
    target_level = 50000.0 if float(level_coord.max()) > 2000.0 else 500.0
    nearest_level = float(level_coord.sel({level_name: target_level}, method='nearest').values)
    zg = ds[daily_da_get_var_name(ds, 'zg')].sel({level_name: nearest_level}) / 9.80665
    zg.attrs['selected_level_value'] = nearest_level
    zg = zg.sel(lat=slice(0, 90))
    return daily_da_select_jja(zg)


def daily_da_ridge_da_one_region(z500_rel, y_region, predictor_expand_deg=15.0, standardize_X='zscore', alphas=None, cv_year_folds=5, max_targets_for_cv=400, random_state=0):
    if alphas is None:
        alphas = np.logspace(-2, 4, 25)
    z500_rel = daily_da_standardize_coords(z500_rel)
    y_region = daily_da_standardize_coords(y_region)
    if 'mask' in y_region.coords:
        y_region = y_region.reset_coords('mask', drop=True)
    z_sel = daily_da_select_jja(z500_rel)
    y_sel = daily_da_select_jja(y_region)
    z_sel, y_sel = xr.align(z_sel, y_sel, join='inner')
    y_box = (
        float(y_sel['lon'].min()),
        float(y_sel['lon'].max()),
        float(y_sel['lat'].min()),
        float(y_sel['lat'].max()),
    )
    pred_box = daily_da_expand_box(y_box, expand_deg=predictor_expand_deg)
    z_dom = daily_da_select_box_wrap(z_sel, pred_box)
    z_anom = daily_da_calendar_day_anomaly(z_dom)
    X, _ = daily_da_stack_feature_matrix(z_anom)
    Y, y_stack = daily_da_stack_target_matrix(y_sel)
    if X.shape[1] < 2:
        raise ValueError('有效预测格点过少，无法进行 Ridge 回归。')
    if Y.shape[1] < 1:
        raise ValueError('有效目标格点为空。')
    X_std, _ = daily_da_standardize_matrix(X, mode=standardize_X)
    groups = pd.DatetimeIndex(y_sel['time'].values).year.astype(int)
    alpha = daily_da_choose_alpha(
        X_std,
        Y,
        groups,
        alphas=alphas,
        n_splits=cv_year_folds,
        max_targets_for_cv=max_targets_for_cv,
        random_state=random_state,
    )
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(X_std, Y)
    Y_pred = model.predict(X_std)
    ss_res = np.sum((Y - Y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    dyn_stack = xr.DataArray(Y_pred, dims=('time', 'target'), coords={'time': y_sel['time'], 'target': y_stack['target']})
    dyn_map = dyn_stack.unstack('target').transpose('time', 'lat', 'lon')
    thermo_map = y_sel - dyn_map
    r2_map = xr.DataArray(r2, dims=('target',), coords={'target': y_stack['target']}).unstack('target').transpose('lat', 'lon')
    ds_out = xr.Dataset({
        'y_raw': y_sel,
        'y_dyn_raw': dyn_map,
        'y_thermo_raw': thermo_map,
        'r2': r2_map,
    })
    ds_out.attrs.update({
        'selected_alpha': float(alpha),
        'predictor_bounds_lon_min': float(pred_box[0]),
        'predictor_bounds_lon_max': float(pred_box[1]),
        'predictor_bounds_lat_min': float(pred_box[2]),
        'predictor_bounds_lat_max': float(pred_box[3]),
        'standardize_X': standardize_X,
        'target_space': 'raw',
    })
    return ds_out



DAILY_DA_BASE_DIR = os.environ.get("DA_BASE_DIR", str(CACHE_ROOT / "dynamic_adjustment"))
DAILY_DA_OUT_DIR = os.path.join(DAILY_DA_BASE_DIR, 'outputs')
DAILY_DA_FIG_DIR = os.path.join(DAILY_DA_BASE_DIR, 'figs')
DAILY_DA_MEMBER_CACHE_DIR = os.path.join(DAILY_DA_OUT_DIR, 'da_ridge_daily_member_cache')

os.makedirs(DAILY_DA_OUT_DIR, exist_ok=True)
os.makedirs(DAILY_DA_FIG_DIR, exist_ok=True)
os.makedirs(DAILY_DA_MEMBER_CACHE_DIR, exist_ok=True)

FILE_DAILY_DA_MEMBER_SOURCES = os.path.join(DAILY_DA_OUT_DIR, 'da_ridge_daily_z500_member_availability_1981_2014.csv')
FILE_DAILY_DA_ERA5_REGION = os.path.join(DAILY_DA_OUT_DIR, 'da_ridge_daily_region_trends_ERA5_1981_2014_JJA.csv')
FILE_DAILY_DA_ERA5_SPATIAL = os.path.join(DAILY_DA_OUT_DIR, 'da_ridge_daily_spatial_trends_ERA5_JJA_1981_2014.nc')
FILE_DAILY_DA_MEMBERS_REGION = os.path.join(DAILY_DA_OUT_DIR, 'da_ridge_daily_region_trends_CMIP_members_JJA_1981_2014.csv')
FILE_DAILY_DA_MME_REGION = os.path.join(DAILY_DA_OUT_DIR, 'da_ridge_daily_region_trends_MME_JJA_1981_2014.csv')
FILE_DAILY_DA_MEMBERS_SPATIAL = os.path.join(DAILY_DA_OUT_DIR, 'da_ridge_daily_spatial_trends_CMIP_members_JJA_1981_2014.nc')
FILE_DAILY_DA_MME_SPATIAL = os.path.join(DAILY_DA_OUT_DIR, 'da_ridge_daily_spatial_trends_MME_JJA_1981_2014.nc')



def daily_da_member_region_path(member):
    return os.path.join(DAILY_DA_MEMBER_CACHE_DIR, f'{member}_region.csv')


def daily_da_member_spatial_path(member):
    return os.path.join(DAILY_DA_MEMBER_CACHE_DIR, f'{member}_spatial.nc')


def daily_da_write_spatial_dataset(spatial_out, path):
    ds = xr.Dataset(spatial_out)
    encoding = {name: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} for name in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    return ds


def daily_da_compute_rows_and_spatial(temp_dict, z500_rel, template, member_name=None, max_targets_for_cv=400):
    rows = []
    spatial_out = {}
    regions = daily_da_get_region_settings()
    for var_name, temp_da in temp_dict.items():
        total_list, dyn_list, thermo_list = [], [], []
        thermo_direct_list, thermo_diff_list = [], []
        for region_name, box in regions:
            y_region = daily_da_select_box_wrap(temp_da, box)
            ds_da = daily_da_ridge_da_one_region(
                z500_rel,
                y_region,
                predictor_expand_deg=15.0,
                standardize_X='zscore',
                cv_year_folds=5,
                max_targets_for_cv=max_targets_for_cv,
                random_state=0,
            )
            slope_total_reg = daily_da_trend_map_from_annual(ds_da['y_raw'])
            slope_dyn_reg = daily_da_trend_map_from_annual(ds_da['y_dyn_raw'])
            slope_thermo_reg = slope_total_reg - slope_dyn_reg
            slope_thermo_direct_reg = daily_da_trend_map_from_annual(ds_da['y_thermo_raw'])
            slope_thermo_diff_reg = slope_thermo_reg - slope_thermo_direct_reg
            thermo_diff_abs_max = float(np.nanmax(np.abs(slope_thermo_diff_reg.values))) if np.isfinite(slope_thermo_diff_reg.values).any() else np.nan
            full_total = daily_da_put_on_template(slope_total_reg, template).astype('float32').expand_dims(region=[region_name])
            full_dyn = daily_da_put_on_template(slope_dyn_reg, template).astype('float32').expand_dims(region=[region_name])
            full_thermo = daily_da_put_on_template(slope_thermo_reg, template).astype('float32').expand_dims(region=[region_name])
            full_thermo_direct = daily_da_put_on_template(slope_thermo_direct_reg, template).astype('float32').expand_dims(region=[region_name])
            full_thermo_diff = daily_da_put_on_template(slope_thermo_diff_reg, template).astype('float32').expand_dims(region=[region_name])
            total_list.append(full_total)
            dyn_list.append(full_dyn)
            thermo_list.append(full_thermo)
            thermo_direct_list.append(full_thermo_direct)
            thermo_diff_list.append(full_thermo_diff)
            slope_total_map = float(daily_da_area_mean(slope_total_reg).values)
            slope_dyn_map = float(daily_da_area_mean(slope_dyn_reg).values)
            slope_thermo_map = float(daily_da_area_mean(slope_thermo_reg).values)
            y_total_ts = daily_da_area_mean(ds_da['y_raw'])
            y_dyn_ts = daily_da_area_mean(ds_da['y_dyn_raw'])
            y_thermo_ts = daily_da_area_mean(ds_da['y_thermo_raw'])
            tr_total = daily_da_series_trend(y_total_ts)
            tr_dyn = daily_da_series_trend(y_dyn_ts)
            tr_thermo = daily_da_series_trend(y_thermo_ts)
            row = {
                'var': var_name,
                'region': region_name,
                'lon_min': box[0],
                'lon_max': box[1],
                'lat_min': box[2],
                'lat_max': box[3],
                'alpha': float(ds_da.attrs.get('selected_alpha', np.nan)),
                'total_slope_per_year': slope_total_map,
                'total_slope_per_decade': slope_total_map * 10.0,
                'dyn_slope_per_year': slope_dyn_map,
                'dyn_slope_per_decade': slope_dyn_map * 10.0,
                'thermo_slope_per_year': slope_thermo_map,
                'thermo_slope_per_decade': slope_thermo_map * 10.0,
                'total_slope_ts_per_year': tr_total['slope'],
                'dyn_slope_ts_per_year': tr_dyn['slope'],
                'thermo_slope_ts_per_year': tr_thermo['slope'],
                'total_pvalue_ts': tr_total['pvalue'],
                'dyn_pvalue_ts': tr_dyn['pvalue'],
                'thermo_pvalue_ts': tr_thermo['pvalue'],
                'total_delta_map_minus_ts_per_year': slope_total_map - tr_total['slope'],
                'dyn_delta_map_minus_ts_per_year': slope_dyn_map - tr_dyn['slope'],
                'thermo_delta_map_minus_ts_per_year': slope_thermo_map - tr_thermo['slope'],
                'thermo_diff_abs_max_per_year': thermo_diff_abs_max,
                'predictor_bounds_lat_min': float(ds_da.attrs.get('predictor_bounds_lat_min', np.nan)),
                'predictor_bounds_lat_max': float(ds_da.attrs.get('predictor_bounds_lat_max', np.nan)),
                'predictor_bounds_lon_min': float(ds_da.attrs.get('predictor_bounds_lon_min', np.nan)),
                'predictor_bounds_lon_max': float(ds_da.attrs.get('predictor_bounds_lon_max', np.nan)),
                'standardize_X': ds_da.attrs.get('standardize_X', ''),
            }
            if member_name is not None:
                row['member'] = str(member_name)
            rows.append(row)
        spatial_out[f'{var_name}_total_slope_per_year'] = xr.concat(total_list, dim='region')
        spatial_out[f'{var_name}_dyn_slope_per_year'] = xr.concat(dyn_list, dim='region')
        spatial_out[f'{var_name}_thermo_slope_per_year'] = xr.concat(thermo_list, dim='region')
        spatial_out[f'{var_name}_thermo_direct_slope_per_year'] = xr.concat(thermo_direct_list, dim='region')
        spatial_out[f'{var_name}_thermo_diff_slope_per_year'] = xr.concat(thermo_diff_list, dim='region')
    return rows, spatial_out


def daily_da_compute_era5(force=False):
    if os.path.exists(FILE_DAILY_DA_ERA5_REGION) and os.path.exists(FILE_DAILY_DA_ERA5_SPATIAL) and not force:
        print('ERA5 日尺度缓存已存在，跳过重算。')
        return {'region_csv': FILE_DAILY_DA_ERA5_REGION, 'spatial_nc': FILE_DAILY_DA_ERA5_SPATIAL, 'skipped': True}
    tmax, tmin = daily_da_open_era5_daily_temps()
    z500 = daily_da_open_era5_daily_z500()
    z500_rel = daily_da_remove_nh_mean(z500)
    template = xr.full_like(tmax.isel(time=0, drop=True), np.nan)
    rows, spatial_out = daily_da_compute_rows_and_spatial({'tmax': tmax, 'tmin': tmin}, z500_rel, template, member_name=None, max_targets_for_cv=800)
    df = pd.DataFrame(rows)
    df.to_csv(FILE_DAILY_DA_ERA5_REGION, index=False)
    daily_da_write_spatial_dataset(spatial_out, FILE_DAILY_DA_ERA5_SPATIAL)
    print('已保存 ERA5 日尺度结果。')
    return {'region_csv': FILE_DAILY_DA_ERA5_REGION, 'spatial_nc': FILE_DAILY_DA_ERA5_SPATIAL, 'skipped': False, 'n_rows': len(df)}


def daily_da_compute_member_cache(member, source_table=None, force=False):
    region_path = daily_da_member_region_path(member)
    spatial_path = daily_da_member_spatial_path(member)
    if os.path.exists(region_path) and os.path.exists(spatial_path) and not force:
        print(f'{member} 缓存已存在，跳过重算。')
        return {'member': member, 'region_csv': region_path, 'spatial_nc': spatial_path, 'skipped': True}
    tmax = daily_da_open_cmip_daily_temperature(member, 'tasmax')
    tmin = daily_da_open_cmip_daily_temperature(member, 'tasmin')
    z500 = daily_da_open_cmip_daily_z500(member, source_table=source_table)
    z500_rel = daily_da_remove_nh_mean(z500)
    template = xr.full_like(tmax.isel(time=0, drop=True), np.nan)
    rows, spatial_out = daily_da_compute_rows_and_spatial({'tmax': tmax, 'tmin': tmin}, z500_rel, template, member_name=member, max_targets_for_cv=400)
    pd.DataFrame(rows).to_csv(region_path, index=False)
    daily_da_write_spatial_dataset(spatial_out, spatial_path)
    print(f'已保存 {member} 日尺度成员缓存。')
    return {'member': member, 'region_csv': region_path, 'spatial_nc': spatial_path, 'skipped': False}


def daily_da_aggregate_member_caches(model_list=None):
    if model_list is None:
        model_list = daily_da_get_model_list()
    region_frames = []
    spatial_list = []
    valid_members = []
    for member in model_list:
        region_path = daily_da_member_region_path(member)
        spatial_path = daily_da_member_spatial_path(member)
        if os.path.exists(region_path) and os.path.exists(spatial_path):
            df = pd.read_csv(region_path)
            if 'member' not in df.columns:
                df['member'] = member
            region_frames.append(df)
            spatial_list.append(xr.open_dataset(spatial_path).expand_dims(member=[member]))
            valid_members.append(member)
    if not region_frames:
        raise FileNotFoundError('当前没有可汇总的成员缓存。')
    df_members = pd.concat(region_frames, ignore_index=True)
    df_members.to_csv(FILE_DAILY_DA_MEMBERS_REGION, index=False)
    df_mme = df_members.groupby(['var', 'region'], as_index=False).mean(numeric_only=True)
    df_mme.to_csv(FILE_DAILY_DA_MME_REGION, index=False)
    members_ds = xr.concat(spatial_list, dim='member')
    members_encoding = {name: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} for name in members_ds.data_vars}
    members_ds.to_netcdf(FILE_DAILY_DA_MEMBERS_SPATIAL, encoding=members_encoding)
    mme_ds = members_ds.mean('member', skipna=True)
    mme_encoding = {name: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} for name in mme_ds.data_vars}
    mme_ds.to_netcdf(FILE_DAILY_DA_MME_SPATIAL, encoding=mme_encoding)
    print('已完成成员汇总和 MME 汇总。')
    return {'valid_members': valid_members, 'n_members': len(valid_members)}


def daily_da_compute_and_save(run_era5=True, run_cmip=True, force_era5=False, force_members=False, force_summary=False, member_subset=None):
    source_table = daily_da_build_member_source_table(output_csv=FILE_DAILY_DA_MEMBER_SOURCES, force=force_summary)
    available_members = source_table.loc[source_table['available'], 'member'].astype(str).tolist()
    missing_members = source_table.loc[~source_table['available'], 'member'].astype(str).tolist()
    if member_subset is not None:
        available_members = [m for m in available_members if m in member_subset]
    out = {'available_members': available_members, 'missing_members': missing_members}
    if run_era5:
        out['era5'] = daily_da_compute_era5(force=force_era5)
    if run_cmip:
        member_results = []
        for member in available_members:
            member_results.append(daily_da_compute_member_cache(member, source_table=source_table, force=force_members))
        out['member_results'] = member_results
    need_summary = force_summary or run_cmip or (not os.path.exists(FILE_DAILY_DA_MEMBERS_REGION)) or (not os.path.exists(FILE_DAILY_DA_MME_REGION))
    if need_summary and available_members:
        out['summary'] = daily_da_aggregate_member_caches(model_list=available_members)
    return out


def daily_da_run_smoke_test():
    source_table = daily_da_build_member_source_table(output_csv=FILE_DAILY_DA_MEMBER_SOURCES, force=False)
    available_members = source_table.loc[source_table['available'], 'member'].astype(str).tolist()
    missing_members = source_table.loc[~source_table['available'], 'member'].astype(str).tolist()
    print('当前可用模式数：', len(available_members))
    print('当前缺失模式：', missing_members)
    tmax_era, tmin_era = daily_da_open_era5_daily_temps()
    z_era = daily_da_open_era5_daily_z500()
    print('ERA5 Tmax 维度：', tmax_era.dims, '时间范围：', str(tmax_era.time.values[0]), str(tmax_era.time.values[-1]))
    print('ERA5 Tmin 维度：', tmin_era.dims, '时间范围：', str(tmin_era.time.values[0]), str(tmin_era.time.values[-1]))
    print('ERA5 Z500 维度：', z_era.dims, '时间范围：', str(z_era.time.values[0]), str(z_era.time.values[-1]))
    era5_result = daily_da_compute_era5(force=False)
    test_member = available_members[0]
    test_region_name, test_box = daily_da_get_region_settings()[0]
    z_test = daily_da_remove_nh_mean(daily_da_open_cmip_daily_z500(test_member, source_table=source_table))
    t_test = daily_da_open_cmip_daily_temperature(test_member, 'tasmax')
    ds_test = daily_da_ridge_da_one_region(z_test, daily_da_select_box_wrap(t_test, test_box), max_targets_for_cv=100)
    dyn_slope = float(daily_da_area_mean(daily_da_trend_map_from_annual(ds_test['y_dyn_raw'])).values)
    return {
        'available_members': available_members,
        'missing_members': missing_members,
        'era5_result': era5_result,
        'test_member': test_member,
        'test_region': test_region_name,
        'test_alpha': float(ds_test.attrs.get('selected_alpha', np.nan)),
        'test_dyn_slope_per_year': dyn_slope,
    }
