# Data requirements

This repository begins with analysis-ready files. It does not download the complete CMIP6, ERA5, MERRA-2, CPC, or Berkeley Earth archives and it does not redistribute any data.

## General conventions

- Main analysis period: 1981–2014.
- Season: June–August (`month = 6, 7, 8`).
- Spatial domain: Northern Hemisphere land, normally on a 1° × 1° latitude–longitude grid.
- Temperature and cumulative-heat trends: °C yr⁻¹.
- Heatwave-day trends: days yr⁻¹.
- EDDY_Z500 trends: m yr⁻¹ after removing the contemporaneous zonal mean.
- Missing values must be represented as NaN rather than numeric sentinels.
- Longitude may be supplied in either convention but is standardized to −180° to 180° before regional selection.

## Official source portals

- CMIP6 model output: [Earth System Grid Federation CMIP6 search](https://esgf-node.llnl.gov/search/cmip6/).
- ERA5: [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-complete?tab=overview).
- MERRA-2: [NASA GMAO MERRA-2](https://gmao.gsfc.nasa.gov/gmao-products/merra-2/) and the linked GES DISC collections.
- JRA-3Q: [JMA JRA-3Q access guide](https://www.data.jma.go.jp/jra/html/JRA-3Q/index_ja.html); the monthly statistics used by `11_prepare_jra3q_eddy_z500.sh` are also distributed as NCAR GDEX dataset d640002.
- CPC Global Unified Temperature: [NOAA Physical Sciences Laboratory](https://psl.noaa.gov/data/gridded/data.cpc.globaltemp.html).
- Berkeley Earth temperature: [Berkeley Earth data](https://berkeleyearth.org/data/).

Users are responsible for complying with each provider’s license, access rules, and citation guidance.

## Prepared cache files

The following files belong under `cache_root`.

### `compound_regional_means_dual_scenarios.nc`

- Dimension: `year = 1981…2023`.
- Variables: `compound_days_*` and `compound_cumulative_heat_*` for SSP245 MME, SSP585 MME, 5th–95th percentiles, CPC, Berkeley Earth, ERA5, MERRA-2, and JRA-3Q.
- Used by Fig. 1.

### `heatwave_percentage_maps.nc`

- Dimensions: `lat`, `lon`.
- Required variables include `compound_days`, `compound_ch`, `tmax_clim`, `tmin_clim`, `extreme1_tmax_percent`, `extreme1_tmin_percent`, `extreme2_tmax_percent`, and `extreme2_tmin_percent`.
- Used by Fig. 2.

### `physical_percentage_allvars.nc`

- Dimensions: `lat`, `lon`.
- Required variables for the released figures: `tcc`, `net`, `q2m`, `eddy_z500`, `net_s`, `net_l`, `rlds`, `pr`, `e`, and `ef`.
- Values are the percentage of CMIP6 models whose historical trend exceeds the ERA5 trend at each grid cell.
- Used by Fig. 3 and Extended Data Fig. 4.

### `correlation_days_physics_allvars.nc`

- Dimensions: `variable`, `region`.
- Variables: `r` and `p`.
- Region order: `NH`, `NNA`, `NAF`, `EEU`, `SAS`, `ESB`, `SNA`, `GIC`, `WEU`, `ENA`.
- Used by Fig. 4 and Extended Data Fig. 7.

### `correlation_cumulative_heat_physics_allvars.nc`

- Same schema as the heatwave-day correlation file, calculated for cumulative heat.
- Retained as a validated physical-process analysis product.

### `hotspot_nonhot_region_boxplots_payload_v3.pkl`

- Reusable plotting cache for Extended Data Figs. 5–6.
- The code can build it from `data_root/physical_process/region_variable_trends.nc`.
- DTR (`dtr` or `tdurual`) is always discarded. `dlr` is canonicalized to `rlds`, displayed as DLR, and `eddy_z500` is stored exactly once in the final position.

## Analysis-ready regional physical-process file

`data_root/physical_process/region_variable_trends.nc` is used when rebuilding Extended Data Figs. 5–6. It must contain:

- `cmip_trend(region, variable, member)` for 30 model members.
- `obs_trend(region, variable, observation)` for the reference products.
- Regions `NNA`, `NAF`, `EEU`, `SAS`, `ESB`, `SNA`, `GIC`, `WEU`, and `ENA`.
- Variables, in the desired display order: `tmax`, `tmin`, `q2m`, `e`, `ef`, `pr`, `tcc`, `net`, `net_s`, `net_l`, `rlds`, and `eddy_z500`.

The script standardizes each observational/reanalysis trend using the corresponding 30-member CMIP6 mean and sample standard deviation.

## Heatwave files used for recomputation

The extracted analysis code expects event masks and cumulative-heat files under `data_root/heatwaves_files`:

- `{dataset}_heatwave_3.nc`, `{dataset}_dayhw_3.nc`, `{dataset}_nighthw_3.nc`.
- `{dataset}_compound_CumulativeHeat.nc`, `{dataset}_day_CumulativeHeat.nc`, `{dataset}_night_CumulativeHeat.nc`.
- CMIP6 member equivalents under `heatwaves_files/1x1` and `heatwaves_files/1x1/model`.

Event-mask files use `year`, `day`, `lat`, and `lon`; daily dynamic-adjustment inputs use `time`, `lat`, and `lon`.

## Dynamic-adjustment inputs and outputs

The scripts in `2_dynamic_adjustment` read daily TMAX, TMIN, Z500, heatwave masks, and percentile thresholds from `data_root`. Their exact directory layout is encoded relative to `CHW_DATA_ROOT` in `daily_da_core.py` and the common fixed-event script.

Generated files belong under `cache_root/dynamic_adjustment` and are not committed. The principal Fig. 6 inputs are:

- `da_ridge_daily_chwcumheatcfexcess_chwcalendar31d_region_trends_ERA5_1981_2014_JJA.csv`.
- `da_ridge_daily_chwcumheatcfexcess_chwcalendar31d_region_trends_CMIP_members_JJA_1981_2014.csv`.
- `da_ridge_daily_chwcumheatcfexcess_chwcalendar31d_region_trends_MME_JJA_1981_2014.csv`.

The principal Extended Data Fig. 8 inputs use the same three suffixes with the `linearbudget_chwcalendar31d` tag. Tables contain the variable and region identifiers, region bounds, training scheme, ridge parameter, sample counts, component slopes, p values, closure diagnostics, units, and optional member-level cache references.

## JRA-3Q EDDY_Z500 preparation

`1_analysis/11_prepare_jra3q_eddy_z500.sh` downloads 102 JJA monthly JRA-3Q files for 1981–2014, selects 500 hPa, remaps to the user-supplied 1° grid with CDO, validates the time axis, and writes the result beneath `CHW_DATA_ROOT`. Set `JRA3Q_GRID_FILE` if the grid description is not at `data_root/static/grid1x1.txt`.
