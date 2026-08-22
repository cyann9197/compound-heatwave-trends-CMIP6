"""Small statistical functions shared by the released figure code."""

from __future__ import annotations

import numpy as np
from scipy import stats


def linear_trend(values, years) -> tuple[float, float]:
    """返回含缺测值一维序列的最小二乘趋势和双侧 p 值。"""

    y = np.asarray(values, dtype=float)
    x = np.asarray(years, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan"), float("nan")
    result = stats.linregress(x[valid], y[valid])
    return float(result.slope), float(result.pvalue)


def area_weighted_mean(data, latitude_name: str = "lat"):
    """按纬度余弦权重计算区域平均。"""

    lat = data[latitude_name]
    weights = np.cos(np.deg2rad(lat))
    spatial_dims = [dim for dim in (latitude_name, "lon") if dim in data.dims]
    return data.weighted(weights).mean(spatial_dims, skipna=True)
