from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from app.core.density_parameters import (
    CELL_SIZE_DECIMAL_PLACES,
    normalize_cell_size,
)
from app.core.density_grid import DensityGrid
from app.core.xyz_reader import PointCloudStats


CACHE_VERSION = 2
_DENSITY_DTYPE = np.dtype(np.uint32)


def _canonical_source_path(file_path: str | Path) -> str:
    return os.path.normcase(str(Path(file_path).resolve()))


def get_file_signature(file_path: str | Path) -> dict[str, str | int]:
    path = Path(file_path)
    stat = path.stat()

    return {
        "canonical_path": _canonical_source_path(path),
        "file_size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def make_safe_name(file_path: str | Path) -> str:
    stem = Path(file_path).stem
    safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem)
    safe_stem = safe_stem.strip(" .")
    return safe_stem or "source"


def _source_identifier(file_path: str | Path) -> str:
    canonical_path = _canonical_source_path(file_path)
    return hashlib.sha256(canonical_path.encode("utf-8")).hexdigest()[:12]


def _cache_key(file_path: str | Path) -> str:
    return f"{make_safe_name(file_path)}_{_source_identifier(file_path)}"


def _cell_size_text(cell_size: float) -> str:
    canonical_cell_size = normalize_cell_size(cell_size)
    text = f"{canonical_cell_size:.{CELL_SIZE_DECIMAL_PLACES}f}"
    return text.rstrip("0").rstrip(".").replace(".", "_")


def stats_cache_path(file_path: str | Path, cache_dir: str | Path = "cache") -> Path:
    cache_dir = Path(cache_dir)
    return cache_dir / f"{_cache_key(file_path)}_stats.json"


def density_cache_path(
    file_path: str | Path,
    cell_size: float,
    cache_dir: str | Path = "cache",
) -> Path:
    cache_dir = Path(cache_dir)
    cell_text = _cell_size_text(cell_size)
    return cache_dir / f"{_cache_key(file_path)}_density_cell_{cell_text}.npy"


def density_metadata_path(
    file_path: str | Path,
    cell_size: float,
    cache_dir: str | Path = "cache",
) -> Path:
    return density_cache_path(file_path, cell_size, cache_dir).with_suffix(
        ".meta.json"
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_save_npy(path: Path, density: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(file_descriptor, "wb") as file:
            np.save(file, density, allow_pickle=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    return payload if isinstance(payload, dict) else None


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def save_stats_cache(
    stats: PointCloudStats,
    cache_dir: str | Path = "cache",
) -> None:
    path = stats_cache_path(stats.file_path, cache_dir)
    payload = {
        "cache_version": CACHE_VERSION,
        "kind": "statistics",
        "source_signature": get_file_signature(stats.file_path),
        "stats": {
            "point_count": stats.point_count,
            "min_x": stats.min_x,
            "max_x": stats.max_x,
            "min_y": stats.min_y,
            "max_y": stats.max_y,
            "min_z": stats.min_z,
            "max_z": stats.max_z,
        },
    }
    _atomic_write_json(path, payload)


def load_stats_cache(
    file_path: str | Path,
    cache_dir: str | Path = "cache",
) -> PointCloudStats | None:
    path = stats_cache_path(file_path, cache_dir)
    if not path.is_file():
        return None

    current_signature = get_file_signature(file_path)
    payload = _read_json_object(path)
    if payload is None:
        return None
    if payload.get("cache_version") != CACHE_VERSION:
        return None
    if payload.get("kind") != "statistics":
        return None
    if payload.get("source_signature") != current_signature:
        return None

    stats_payload = payload.get("stats")
    if not isinstance(stats_payload, dict):
        return None

    point_count_value = stats_payload.get("point_count")
    if not isinstance(point_count_value, int) or isinstance(point_count_value, bool):
        return None
    point_count = point_count_value

    bound_names = ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")
    if not all(_is_json_number(stats_payload.get(name)) for name in bound_names):
        return None
    bounds = {name: float(stats_payload[name]) for name in bound_names}

    if point_count < 0 or not all(math.isfinite(value) for value in bounds.values()):
        return None

    return PointCloudStats(
        file_path=Path(file_path),
        point_count=point_count,
        **bounds,
    )


def _expected_density_shape(
    stats: PointCloudStats,
    cell_size: float,
) -> tuple[int, int]:
    cell_size = normalize_cell_size(cell_size)
    width = int(np.ceil(stats.width / cell_size)) + 1
    height = int(np.ceil(stats.height / cell_size)) + 1
    return height, width


def save_density_cache(
    grid: DensityGrid,
    file_path: str | Path,
    cache_dir: str | Path = "cache",
) -> None:
    cell_size = normalize_cell_size(grid.cell_size)
    path = density_cache_path(file_path, cell_size, cache_dir)
    metadata_path = density_metadata_path(file_path, cell_size, cache_dir)
    metadata = {
        "cache_version": CACHE_VERSION,
        "kind": "density",
        "source_signature": get_file_signature(file_path),
        "cell_size": cell_size,
        "grid": {
            "min_x": grid.min_x,
            "min_y": grid.min_y,
            "shape": list(grid.density.shape),
            "dtype": str(grid.density.dtype),
        },
    }

    _atomic_save_npy(path, grid.density)
    _atomic_write_json(metadata_path, metadata)


def _load_density_array(path: Path) -> np.ndarray | None:
    try:
        density = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return None

    return density if isinstance(density, np.ndarray) else None


def load_density_cache(
    file_path: str | Path,
    stats: PointCloudStats,
    cell_size: float,
    cache_dir: str | Path = "cache",
) -> DensityGrid | None:
    cell_size = normalize_cell_size(cell_size)
    path = density_cache_path(file_path, cell_size, cache_dir)
    metadata_path = density_metadata_path(file_path, cell_size, cache_dir)
    if not path.is_file() or not metadata_path.is_file():
        return None

    current_signature = get_file_signature(file_path)
    metadata = _read_json_object(metadata_path)
    if metadata is None:
        return None
    if metadata.get("cache_version") != CACHE_VERSION:
        return None
    if metadata.get("kind") != "density":
        return None
    if metadata.get("source_signature") != current_signature:
        return None

    grid_metadata = metadata.get("grid")
    if not isinstance(grid_metadata, dict):
        return None

    cell_size_value = metadata.get("cell_size")
    min_x_value = grid_metadata.get("min_x")
    min_y_value = grid_metadata.get("min_y")
    shape_value = grid_metadata.get("shape")
    dtype_value = grid_metadata.get("dtype")

    if not _is_json_number(cell_size_value):
        return None
    if not _is_json_number(min_x_value) or not _is_json_number(min_y_value):
        return None
    if not isinstance(shape_value, list) or len(shape_value) != 2:
        return None
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in shape_value
    ):
        return None
    if not isinstance(dtype_value, str):
        return None

    expected_shape = _expected_density_shape(stats, cell_size)
    try:
        cached_cell_size = normalize_cell_size(cell_size_value)
        cached_min_x = float(min_x_value)
        cached_min_y = float(min_y_value)
        cached_dtype = np.dtype(dtype_value)
    except (TypeError, ValueError, OverflowError):
        return None
    cached_shape = tuple(shape_value)

    if cached_cell_size != cell_size:
        return None
    if cached_min_x != stats.min_x or cached_min_y != stats.min_y:
        return None
    if cached_shape != expected_shape:
        return None
    if cached_dtype != _DENSITY_DTYPE:
        return None

    density = _load_density_array(path)
    if density is None:
        return None
    if density.shape != expected_shape or density.shape != cached_shape:
        return None
    if density.dtype != _DENSITY_DTYPE or density.dtype != cached_dtype:
        return None

    return DensityGrid(
        density=density,
        cell_size=cell_size,
        min_x=cached_min_x,
        min_y=cached_min_y,
    )
