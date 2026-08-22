from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chw_cmip6.data_io import require_files
from chw_cmip6.figure_cli import prepare_figure
from chw_cmip6.regional_boxplots import build_payload_from_analysis_ready, sanitize_payload


ARGS, CONFIG = prepare_figure(
    "physical-process-analysis",
    "Validate physical-process caches and build the ED Figs. 5–6 reusable payload.",
)


def main() -> None:
    # Fig. 3、Fig. 4、ED Fig. 4 和 ED Fig. 7 直接读取这三个核验后的缓存。
    required = [
        CONFIG.cache_root / "physical_percentage_allvars.nc",
        CONFIG.cache_root / "correlation_days_physics_allvars.nc",
        CONFIG.cache_root / "correlation_cumulative_heat_physics_allvars.nc",
    ]
    require_files(required, figure="Physical-process analysis")

    source = CONFIG.data_root / "physical_process" / "region_variable_trends.nc"
    payload = sanitize_payload(build_payload_from_analysis_ready(source))
    destination = CONFIG.cache_root / "hotspot_nonhot_region_boxplots_payload_v3.pkl"
    with destination.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"READY {destination}")


if __name__ == "__main__":
    main()
