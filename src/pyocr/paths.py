from __future__ import annotations

from pathlib import Path


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def norm_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)

