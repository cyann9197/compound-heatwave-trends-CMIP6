from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_FIGURES = [f"fig{i}.py" for i in range(1, 7)]
EXTENDED_FIGURES = [f"figED{i}.py" for i in range(1, 9)]


class RepositoryContractsTest(unittest.TestCase):
    def test_expected_figure_entrypoints_exist(self) -> None:
        self.assertEqual(sorted(p.name for p in (ROOT / "3_main_figures").glob("fig*.py")), MAIN_FIGURES)
        self.assertEqual(sorted(p.name for p in (ROOT / "4_extended_data").glob("figED*.py")), EXTENDED_FIGURES)

    def test_every_figure_entrypoint_exposes_help(self) -> None:
        scripts = [
            *(ROOT / "3_main_figures" / name for name in MAIN_FIGURES),
            *(ROOT / "4_extended_data" / name for name in EXTENDED_FIGURES),
        ]
        for script in scripts:
            completed = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("--config", completed.stdout)
            self.assertIn("--recompute", completed.stdout)
            self.assertIn("--plot-only", completed.stdout)

    def test_repository_contains_no_data_or_generated_outputs(self) -> None:
        forbidden_suffixes = {".nc", ".nc4", ".csv", ".npy", ".npz", ".pkl", ".joblib", ".png", ".jpg", ".tif", ".tiff", ".pdf", ".docx", ".pyc"}
        tracked_candidates = [
            p for p in ROOT.rglob("*")
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
        ]
        offenders = [p.relative_to(ROOT) for p in tracked_candidates if p.suffix.lower() in forbidden_suffixes]
        self.assertEqual(offenders, [])

    def test_repository_contains_no_private_absolute_paths(self) -> None:
        forbidden = (
            "/public/home/" + "yluo",
            "E:" + "\\\\" + "暂时汇报",
            "C:" + "\\\\" + "Users" + "\\\\" + "87272",
        )
        offenders: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(token in text for token in forbidden):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_repository_has_no_file_larger_than_10_mb(self) -> None:
        offenders = [
            (p.relative_to(ROOT), p.stat().st_size)
            for p in ROOT.rglob("*")
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and p.stat().st_size > 10 * 1024 * 1024
        ]
        self.assertEqual(offenders, [])

    def test_repository_has_no_runtime_dependency_on_development_notebooks(self) -> None:
        offenders = []
        forbidden = ("newest" + ".ipynb", "final_figures_nature_geoscience" + ".ipynb")
        for path in ROOT.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in forbidden):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_fig6_entrypoint_contains_no_legacy_product_name(self) -> None:
        text = (ROOT / "3_main_figures" / "fig6.py").read_text(encoding="utf-8")
        self.assertIn("chwcumheatcfexcess_chwcalendar31d", text)
        self.assertNotIn("Fig7_fixed_event", text)
        self.assertNotIn("Fig6_cumheat_da_chwcalendar31d_beeswarm", text)

    def test_prepared_dynamic_figures_do_not_load_unrelated_data_context(self) -> None:
        for relative_path in ("3_main_figures/fig6.py", "4_extended_data/figED8.py"):
            text = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("chw_cmip6.da_plot_style", text)
            self.assertNotIn("chw_cmip6.figure_context", text)

    def test_shared_figure_context_has_no_import_time_observation_reads(self) -> None:
        text = (ROOT / "src/chw_cmip6/figure_context.py").read_text(encoding="utf-8")
        self.assertNotIn("f1=xr.open_dataset", text)
        self.assertNotIn("f2=xr.open_dataset", text)

    def test_extended_data_figure_3_uses_the_shared_recompute_flag(self) -> None:
        text = (ROOT / "4_extended_data/figED3.py").read_text(encoding="utf-8")
        self.assertIn("force_recompute=ARGS.recompute", text)
        self.assertIn("force=ARGS.recompute", text)
        self.assertNotIn("force_recompute=FIG2_FORCE_RECOMPUTE", text)
        self.assertNotIn("force=FIG2_CUMHEAT_FORCE_RECOMPUTE", text)
