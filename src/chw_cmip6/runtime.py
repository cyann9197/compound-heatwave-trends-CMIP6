"""Small process-local runtime shared by the extracted figure scripts."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ProjectConfig


@dataclass(frozen=True)
class FigureRuntime:
    config: ProjectConfig
    recompute: bool = False
    plot_only: bool = False


_RUNTIME: FigureRuntime | None = None


def set_runtime(runtime: FigureRuntime) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def get_runtime() -> FigureRuntime:
    if _RUNTIME is None:
        raise RuntimeError("Figure runtime has not been configured. Call prepare_figure() first.")
    return _RUNTIME
