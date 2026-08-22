# Compound heatwave trends in CMIP6

This repository contains the minimum code needed to reproduce the main and Extended Data figures associated with the manuscript *Dynamical bias drive CMIP6 regional discrepancies in simulations of compound heatwave changes over Northern Hemisphere*.

The workflow evaluates historical (1981–2014) June–August compound daytime–nighttime heatwave trends over Northern Hemisphere land using 30 CMIP6 models and multiple observational or reanalysis products. Data are not distributed with this repository.

## Repository organization

```text
1_analysis/             preparation and physical-process analysis
2_dynamic_adjustment/   ridge dynamic adjustment and fixed-event decompositions
3_main_figures/         one Python entry point for each main figure
4_extended_data/        one Python entry point for each Extended Data figure
src/chw_cmip6/          shared I/O, statistics, plotting, and scientific contracts
tests/                  repository, CLI, and scientific-contract tests
```

The numbered organization follows the workflow style of `FenyingCai/heatwave_trends-AT-network`, while keeping this release limited to manuscript-essential code.

## Environment

```bash
conda env create -f environment.yml
conda activate compound-heatwave-trends
```

The environment records the versions used by the original `lyh3.9` analysis environment. Arial is used when installed; otherwise the plotting code falls back to Liberation Sans or DejaVu Sans. No proprietary font files are bundled.

## Configuration

Copy the example configuration and replace all paths with locations outside the repository:

```bash
cp config/paths.example.json config/paths.json
```

The four settings are `data_root`, `cache_root`, `output_root`, and `n_jobs`. Resolution order is command-line options, environment variables, then JSON values. The supported environment variables are `CHW_DATA_ROOT`, `CHW_CACHE_ROOT`, `CHW_OUTPUT_ROOT`, and `CHW_N_JOBS`.

All figure entry points support the same controls:

```bash
python 3_main_figures/fig3.py --config config/paths.json
python 4_extended_data/figED5.py --config config/paths.json --recompute
python 3_main_figures/fig6.py --config config/paths.json --plot-only
```

By default, an existing reusable cache is read first and a missing cache is computed when the required analysis-ready inputs are available. `--recompute` forces rebuilding. `--plot-only` never performs the expensive computation and reports every missing prerequisite.

## Figure entry points

| Figure | Command | Primary prepared input or computation |
|---|---|---|
| Fig. 1 | `python 3_main_figures/fig1.py --config config/paths.json` | `compound_regional_means_dual_scenarios.nc`; ranking cache or heatwave files |
| Fig. 2 | `python 3_main_figures/fig2.py --config config/paths.json` | `heatwave_percentage_maps.nc`; prepared regional trend payload |
| Fig. 3 | `python 3_main_figures/fig3.py --config config/paths.json` | `physical_percentage_allvars.nc` |
| Fig. 4 | `python 3_main_figures/fig4.py --config config/paths.json` | `correlation_days_physics_allvars.nc` |
| Fig. 5 | `python 3_main_figures/fig5.py --config config/paths.json` | monthly TMAX/TMIN dynamic-adjustment tables |
| Fig. 6 | `python 3_main_figures/fig6.py --config config/paths.json` | fixed-event counterfactual-excess tables, tag `chwcumheatcfexcess_chwcalendar31d` |
| ED Fig. 1 | `python 4_extended_data/figED1.py --config config/paths.json` | ERA5 daytime, nighttime, and compound heatwave trend cache |
| ED Fig. 2 | `python 4_extended_data/figED2.py --config config/paths.json` | CMIP6 MME daytime, nighttime, and compound heatwave trend cache |
| ED Fig. 3 | `python 4_extended_data/figED3.py --config config/paths.json` | regional cumulative-heat trend payload |
| ED Fig. 4 | `python 4_extended_data/figED4.py --config config/paths.json` | `physical_percentage_allvars.nc` |
| ED Fig. 5 | `python 4_extended_data/figED5.py --config config/paths.json` | standardized overestimated-region variable payload |
| ED Fig. 6 | `python 4_extended_data/figED6.py --config config/paths.json` | standardized underestimated-region variable payload |
| ED Fig. 7 | `python 4_extended_data/figED7.py --config config/paths.json` | `correlation_days_physics_allvars.nc` |
| ED Fig. 8 | `python 4_extended_data/figED8.py --config config/paths.json` | linear-budget tables, tag `linearbudget_chwcalendar31d` |

Detailed schemas, units, and source portals are listed in [DATA_REQUIREMENTS.md](DATA_REQUIREMENTS.md).

## Scientific contracts in this release

- The analysis period is 1981–2014 and the target season is June–August.
- The CMIP6 ensemble contains 30 models.
- The Greenland/Iceland region uses 55–22°W and 71–83°N.
- Fig. 3 uses `TCC`, `NET`, `Q2M`, and `EDDY_Z500` in that order.
- Extended Data Figs. 4 and 7 use `NET_S`, `NET_L`, `DLR`, `PR`, `E`, and `EF`.
- Extended Data Figs. 5 and 6 exclude DTR and place `EDDY_Z500` last.
- Fig. 6 uses only the fixed-event counterfactual-excess product.
- Extended Data Fig. 8 uses the linear-budget decomposition.

These requirements are executable tests in `tests/test_scientific_contracts.py`.

## Dynamic-adjustment workflow

The dynamic-adjustment scripts preserve the verified ridge-regression implementation but no longer load functions from the large development notebook.

```bash
export CHW_DATA_ROOT=/path/to/analysis-ready-data
export CHW_CACHE_ROOT=/path/to/reusable-cache
python 2_dynamic_adjustment/22_daily_da_common.py
python 2_dynamic_adjustment/23_fixed_event_common.py
python 2_dynamic_adjustment/24_fixed_event_counterfactual_excess.py
python 2_dynamic_adjustment/25_linear_budget.py
```

Long-running computations save member-level and daily caches before creating aggregate tables. See `DATA_REQUIREMENTS.md` before running these scripts.

## Tests

The test suite uses only the Python standard library:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src 1_analysis 2_dynamic_adjustment 3_main_figures 4_extended_data
```

## Data and code availability

No data, generated figures, cached results, logs, manuscript files, backup scripts, or proprietary fonts are tracked. Users must obtain the source datasets under their original licenses and prepare the inputs described in `DATA_REQUIREMENTS.md`.

## License

The code is released under the MIT License. Dataset licenses and citation requirements remain those of the original data providers.
