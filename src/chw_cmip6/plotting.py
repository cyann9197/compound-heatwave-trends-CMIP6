"""Shared publication plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


def configure_fonts() -> None:
    """优先使用 Arial；系统未安装时回退到开源无衬线字体。"""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig, output_root: str | Path, stem: str, *, dpi: int = 300) -> list[Path]:
    """同时保存 PNG 预览和 PDF 正式图，并返回两个路径。"""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    outputs = [root / f"{stem}.png", root / f"{stem}.pdf"]
    fig.savefig(outputs[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    return outputs
