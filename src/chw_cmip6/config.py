"""Portable path configuration for the public figure code."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """配置文件缺失或字段无效。"""


@dataclass(frozen=True)
class ProjectConfig:
    """统一保存分析就绪数据、缓存、输出和并行设置。"""

    data_root: Path
    cache_root: Path
    output_root: Path
    n_jobs: int = 1


_ENV_KEYS = {
    "data_root": "CHW_DATA_ROOT",
    "cache_root": "CHW_CACHE_ROOT",
    "output_root": "CHW_OUTPUT_ROOT",
    "n_jobs": "CHW_N_JOBS",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(
            f"Configuration file not found: {path}. Copy config/paths.example.json "
            "to config/paths.json and set the four required values."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read configuration file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Configuration root must be a JSON object: {path}")
    return payload


def load_config(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    output_root: str | Path | None = None,
    n_jobs: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProjectConfig:
    """按“命令行 > 环境变量 > JSON 文件”的顺序合并配置。"""

    values = _read_json(Path(path))
    env = os.environ if environ is None else environ
    cli_values: dict[str, Any] = {
        "data_root": data_root,
        "cache_root": cache_root,
        "output_root": output_root,
        "n_jobs": n_jobs,
    }
    for key, env_key in _ENV_KEYS.items():
        if env.get(env_key) not in (None, ""):
            values[key] = env[env_key]
        if cli_values[key] is not None:
            values[key] = cli_values[key]

    missing = [key for key in ("data_root", "cache_root", "output_root") if values.get(key) in (None, "")]
    if missing:
        raise ConfigError("Missing required configuration value(s): " + ", ".join(missing))
    try:
        jobs = int(values.get("n_jobs", 1))
    except (TypeError, ValueError) as exc:
        raise ConfigError("n_jobs must be a positive integer") from exc
    if jobs < 1:
        raise ConfigError("n_jobs must be a positive integer")

    return ProjectConfig(
        data_root=Path(values["data_root"]).expanduser(),
        cache_root=Path(values["cache_root"]).expanduser(),
        output_root=Path(values["output_root"]).expanduser(),
        n_jobs=jobs,
    )
