from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chw_cmip6.figure_cli import prepare_figure

ARGS, CONFIG = prepare_figure("fig4", "Reproduce manuscript Figure 4.")

import matplotlib.pyplot as plt
import xarray as xr

from chw_cmip6.data_io import require_files, require_variables
from chw_cmip6.physical_process import AXIS_LABEL_PT, VARIABLE_TITLES, configure_physical_style, draw_corr
from chw_cmip6.plotting import save_figure
from chw_cmip6.scientific_constants import FIG3_VARIABLES


def main() -> None:
    configure_physical_style()
    source = CONFIG.cache_root / "correlation_days_physics_allvars.nc"
    require_files([source], figure="Figure 4")
    dataset = xr.open_dataset(source)
    require_variables(dataset, ("r", "p"), source=str(source))
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), dpi=300, sharey=True)
    for index, key in enumerate(FIG3_VARIABLES):
        draw_corr(axes.flat[index], key, "abcd"[index], VARIABLE_TITLES[key], dataset)
    fig.text(0.0, 0.5, "Inter-model correlation", rotation=90, va="center", ha="center", fontsize=AXIS_LABEL_PT + 2)
    fig.tight_layout()
    save_figure(fig, CONFIG.output_root, "Fig4", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
