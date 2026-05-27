from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.core.density_grid import DensityGrid
from app.core.xyz_reader import PointCloudStats


def get_file_signature(file_path: str | Path) -> dict:
    path = Path(file_path)
    stat = path.stat()

    return {
        "file_path": str(path.resolve()),
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def make_safe_name(file_path: str | Path) -> str:
    path = Path(file_path)
    return path.stem.replace(" ", "_")


def stats_cache_path(file_path: str | Path, cache_dir: str | Path = "cache") -> Path:
    cache_dir = Path(cache_dir)
    return cache_dir / f"{make_safe_name(file_path)}_stats.json"


def density_cache_path(
    file_path: str | Path,
    cell_size: float,
    cache_dir: str | Path = "cache",
) -> Path:
    cache_dir = Path(cache_dir)
    cell_text = str(cell_size).replace(".", "_")
    return cache_dir / f"{make_safe_name(file_path)}_density_cell_{cell_text}.npy"


def save_stats_cache(
    stats: PointCloudStats,
    cache_dir: str | Path = "cache",
) -> None:
    path = stats_cache_path(stats.file_path, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "signature": get_file_signature(stats.file_path),
        "stats": {
            "file_path": str(stats.file_path),
            "point_count": stats.point_count,
            "min_x": stats.min_x,
            "max_x": stats.max_x,
            "min_y": stats.min_y,
            "max_y": stats.max_y,
            "min_z": stats.min_z,
            "max_z": stats.max_z,
        },
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_stats_cache(
    file_path: str | Path,
    cache_dir: str | Path = "cache",
) -> PointCloudStats | None:
    path = stats_cache_path(file_path, cache_dir)

    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    current_signature = get_file_signature(file_path)

    if payload.get("signature") != current_signature:
        return None

    s = payload["stats"]

    return PointCloudStats(
        file_path=Path(file_path),
        point_count=int(s["point_count"]),
        min_x=float(s["min_x"]),
        max_x=float(s["max_x"]),
        min_y=float(s["min_y"]),
        max_y=float(s["max_y"]),
        min_z=float(s["min_z"]),
        max_z=float(s["max_z"]),
    )


def save_density_cache(
    grid: DensityGrid,
    file_path: str | Path,
    cache_dir: str | Path = "cache",
) -> None:
    path = density_cache_path(file_path, grid.cell_size, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, grid.density)


def load_density_cache(
    file_path: str | Path,
    stats: PointCloudStats,
    cell_size: float,
    cache_dir: str | Path = "cache",
) -> DensityGrid | None:
    path = density_cache_path(file_path, cell_size, cache_dir)

    if not path.exists():
        return None

    density = np.load(path)

    return DensityGrid(
        density=density,
        cell_size=cell_size,
        min_x=stats.min_x,
        min_y=stats.min_y,
    )