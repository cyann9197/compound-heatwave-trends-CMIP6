"""Input validation helpers for analysis-ready files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class MissingInputError(FileNotFoundError):
    """一个或多个分析就绪输入文件不存在。"""


def require_files(paths: Iterable[str | Path], *, figure: str) -> list[Path]:
    resolved = [Path(path) for path in paths]
    missing = [path for path in resolved if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise MissingInputError(
            f"{figure} is missing required analysis-ready input(s):\n{formatted}\n"
            "See DATA_REQUIREMENTS.md for variables, dimensions, units, and preparation notes."
        )
    return resolved


def require_variables(dataset, variables: Iterable[str], *, source: str) -> None:
    missing = [name for name in variables if name not in dataset]
    if missing:
        raise KeyError(f"{source} is missing required variable(s): {', '.join(missing)}")
