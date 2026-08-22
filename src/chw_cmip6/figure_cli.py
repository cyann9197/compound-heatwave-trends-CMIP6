"""Common command-line interface used by all fourteen figure entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ProjectConfig, load_config
from .runtime import FigureRuntime, set_runtime


def build_parser(figure_id: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=figure_id, description=description)
    parser.add_argument("--config", default="config/paths.json", help="Path to the private JSON path configuration.")
    parser.add_argument("--data-root", help="Override data_root from the configuration file.")
    parser.add_argument("--cache-root", help="Override cache_root from the configuration file.")
    parser.add_argument("--output-root", help="Override output_root from the configuration file.")
    parser.add_argument("--n-jobs", type=int, help="Override the configured worker count.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--recompute", action="store_true", help="Force rebuilding reusable intermediate results before plotting.")
    mode.add_argument("--plot-only", action="store_true", help="Use existing intermediate results and fail if any are missing.")
    return parser


def prepare_figure(figure_id: str, description: str) -> tuple[argparse.Namespace, ProjectConfig]:
    """解析参数、创建新输出目录，并为后续模块注册运行设置。"""

    args = build_parser(figure_id, description).parse_args()
    config = load_config(
        args.config,
        data_root=args.data_root,
        cache_root=args.cache_root,
        output_root=args.output_root,
        n_jobs=args.n_jobs,
    )
    config.cache_root.mkdir(parents=True, exist_ok=True)
    config.output_root.mkdir(parents=True, exist_ok=True)
    set_runtime(FigureRuntime(config=config, recompute=args.recompute, plot_only=args.plot_only))
    return args, config


def bootstrap_source_tree(script_file: str) -> Path:
    """允许从仓库根目录直接运行脚本，而无需先安装包。"""

    return Path(script_file).resolve().parents[1] / "src"
