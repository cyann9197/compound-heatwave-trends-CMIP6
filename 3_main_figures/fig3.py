from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chw_cmip6.figure_cli import prepare_figure

ARGS, CONFIG = prepare_figure("fig3", "Reproduce manuscript Figure 3.")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from chw_cmip6.data_io import require_files, require_variables
from chw_cmip6.physical_process import CBAR_LABEL_PT, CBAR_TICK_PT, VARIABLE_TITLES, configure_physical_style, draw_ratio
from chw_cmip6.plotting import save_figure
from chw_cmip6.scientific_constants import FIG3_VARIABLES


def main() -> None:
    configure_physical_style()
    source = CONFIG.cache_root / "physical_percentage_allvars.nc"
    require_files([source], figure="Figure 3")
    dataset = xr.open_dataset(source)
    require_variables(dataset, FIG3_VARIABLES, source=str(source))
    fig = plt.figure(figsize=(13.2, 9.0), dpi=300)
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.035], hspace=0.07, wspace=0.06)
    contour = None
    for index, key in enumerate(FIG3_VARIABLES):
        contour = draw_ratio(
            fig.add_subplot(grid[index // 2, index % 2], projection=ccrs.PlateCarree()),
            dataset[key],
            "abcd"[index],
            VARIABLE_TITLES[key],
            hide_y=index % 2 != 0,
        )
    color_axis = fig.add_subplot(grid[2, :])
    colorbar = plt.colorbar(contour, cax=color_axis, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=CBAR_TICK_PT)
    colorbar.set_ticks(np.arange(0, 101, 10))
    colorbar.ax.text(0.5, 1.0, "%", transform=colorbar.ax.transAxes, ha="center", va="bottom", fontsize=CBAR_LABEL_PT)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.934, bottom=0.064)
    save_figure(fig, CONFIG.output_root, "Fig3", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
