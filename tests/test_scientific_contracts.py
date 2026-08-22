from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chw_cmip6.scientific_constants import (
    ANALYSIS_MONTHS,
    ANALYSIS_PERIOD,
    EXTENDED_DATA_4_7_VARIABLES,
    EXTENDED_DATA_5_6_VARIABLES,
    FIG3_VARIABLES,
    FIG6_PRODUCT_TAG,
    GIC_BOUNDS,
    MODEL_NAMES,
    N_MODELS,
)


class ScientificContractsTest(unittest.TestCase):
    def test_core_analysis_contract(self) -> None:
        self.assertEqual(ANALYSIS_PERIOD, (1981, 2014))
        self.assertEqual(ANALYSIS_MONTHS, (6, 7, 8))
        self.assertEqual(N_MODELS, 30)
        self.assertEqual(len(MODEL_NAMES), N_MODELS)
        self.assertEqual(len(set(MODEL_NAMES)), N_MODELS)
        self.assertEqual(GIC_BOUNDS, (-55.0, -22.0, 71.0, 83.0))

    def test_physical_process_variable_order(self) -> None:
        self.assertEqual(FIG3_VARIABLES, ("tcc", "net", "q2m", "eddy_z500"))
        self.assertEqual(EXTENDED_DATA_4_7_VARIABLES, ("net_s", "net_l", "rlds", "pr", "e", "ef"))

    def test_extended_data_5_6_excludes_dtr_and_ends_with_eddy_z500(self) -> None:
        lowered = tuple(item.lower() for item in EXTENDED_DATA_5_6_VARIABLES)
        self.assertNotIn("dtr", lowered)
        self.assertNotIn("tdurual", lowered)
        self.assertEqual(lowered.count("eddy_z500"), 1)
        self.assertEqual(lowered[-1], "eddy_z500")

    def test_fig6_uses_only_fixed_event_counterfactual_excess(self) -> None:
        self.assertEqual(FIG6_PRODUCT_TAG, "chwcumheatcfexcess_chwcalendar31d")
        self.assertNotIn("cumheat_da", FIG6_PRODUCT_TAG)
