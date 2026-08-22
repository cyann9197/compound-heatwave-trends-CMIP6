from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chw_cmip6.figure_cli import prepare_figure
ARGS, CONFIG = prepare_figure("figED5", "Reproduce Extended Data Figure 5 without DTR.")

from chw_cmip6.regional_boxplots import load_or_build_payload, plot_group


def main() -> None:
    payload = load_or_build_payload(CONFIG, recompute=ARGS.recompute, plot_only=ARGS.plot_only)
    plot_group(payload, "hot_group", CONFIG.output_root, "Extended Data Fig5")


if __name__ == "__main__":
    main()
