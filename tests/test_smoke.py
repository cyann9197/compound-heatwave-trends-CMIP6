from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chw_cmip6.config import ConfigError, load_config
from chw_cmip6.figure_cli import build_parser
from chw_cmip6.scientific_constants import EXTENDED_DATA_5_6_VARIABLES


class SmokeTest(unittest.TestCase):
    def test_config_precedence_cli_over_environment_over_file(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "paths.json"
            config_path.write_text(
                json.dumps(
                    {
                        "data_root": "file-data",
                        "cache_root": "file-cache",
                        "output_root": "file-output",
                        "n_jobs": 2,
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CHW_DATA_ROOT": "env-data", "CHW_N_JOBS": "4"}, clear=False):
                config = load_config(config_path, output_root="cli-output", n_jobs=8)

        self.assertEqual(config.data_root, Path("env-data"))
        self.assertEqual(config.cache_root, Path("file-cache"))
        self.assertEqual(config.output_root, Path("cli-output"))
        self.assertEqual(config.n_jobs, 8)

    def test_missing_required_roots_raise_actionable_error(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "empty.json"
            config_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "data_root"):
                load_config(config_path)

    def test_recompute_and_plot_only_are_mutually_exclusive(self) -> None:
        parser = build_parser("fig1", "Figure 1")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--recompute", "--plot-only"])

    def test_regional_boxplot_accepts_variable_specific_observation_counts(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from tempfile import TemporaryDirectory

        from chw_cmip6.regional_boxplots import plot_group

        observation_values = [np.arange(5, dtype=float) for _ in EXTENDED_DATA_5_6_VARIABLES]
        observation_values[-1] = np.arange(3, dtype=float)
        region = {
            "region": "NNA",
            "variables": list(EXTENDED_DATA_5_6_VARIABLES),
            "cmip_values": [np.linspace(-1, 1, 30) for _ in EXTENDED_DATA_5_6_VARIABLES],
            "obs_values": observation_values,
        }
        payload = {
            "obs_names": ["ERA5", "MERRA2", "JRA-3Q", "CPC", "BEST"],
            "obs_markers": ["o", "s", "^", "D", "P"],
            "hot_group": {"region_names": ["NNA"], "regions": [region]},
            "nonhot_group": {"region_names": ["SNA"], "regions": [region]},
        }
        with TemporaryDirectory() as directory:
            outputs = plot_group(payload, "hot_group", Path(directory), "test-figure")
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in outputs))
