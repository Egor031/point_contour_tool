from __future__ import annotations

import math
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

from app.core.density_grid import DensityGrid
from app.core.preview_export import density_to_image


DENSITY_PREVIEW_MAX_SIZE = 3000
SUPPORTED_POINT_CLOUD_EXTENSIONS = frozenset({".asc", ".xyz", ".xyzn"})


def validate_density_request(
    source_path: str | Path,
    cell_size: object,
) -> tuple[Path, float]:
    source_text = str(source_path).strip()
    if not source_text:
        raise ValueError("Select a source point cloud.")

    source = Path(source_text).expanduser()
    if not source.is_file():
        raise ValueError(f"Source file not found: {source}")
    if source.suffix.lower() not in SUPPORTED_POINT_CLOUD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_POINT_CLOUD_EXTENSIONS))
        raise ValueError(f"Unsupported source format. Expected: {supported}.")

    try:
        validated_cell_size = float(cell_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("Cell size must be numeric.") from exc

    if not math.isfinite(validated_cell_size) or validated_cell_size <= 0:
        raise ValueError("Cell size must be finite and greater than zero.")

    return source, validated_cell_size


def density_grid_to_preview(
    grid: DensityGrid,
    max_size: int = DENSITY_PREVIEW_MAX_SIZE,
) -> np.ndarray:
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    return density_to_image(grid.density, max_size=max_size)


def preview_to_texture_rgba(preview: np.ndarray) -> np.ndarray:
    if preview.ndim != 2:
        raise ValueError("Density preview must be a two-dimensional grayscale image.")

    normalized = preview.astype(np.float32) / 255.0
    rgba = np.empty((*preview.shape, 4), dtype=np.float32)
    rgba[:, :, 0] = normalized
    rgba[:, :, 1] = normalized
    rgba[:, :, 2] = normalized
    rgba[:, :, 3] = 1.0
    return rgba


def grid_preview_params(grid: DensityGrid) -> tuple[float, float, float, int, int]:
    return grid.min_x, grid.min_y, grid.cell_size, grid.width, grid.height


def reset_source_dependent_state(target: MutableMapping[str, Any]) -> None:
    target.update(
        {
            "roi_first_world": None,
            "rectangle_roi": None,
            "polygon_finished": False,
            "selection_applied": False,
            "editing_overlay_visible": True,
            "selection_kind": None,
            "selection_polygon_points": [],
            "mode": "rectangle",
            "polygon_points": [],
            "mask_edits": [],
            "last_brush_image": None,
            "last_brush_world": None,
            "brush_cursor_image": None,
            "active_brush_stroke_id": None,
            "next_brush_stroke_id": 1,
            "holes": [],
            "hole_groups": [],
            "visible_hole_group_ids": {},
            "pick_manual_hole_center": False,
            "manual_hole_center_world": None,
            "suppress_brush_until_mouse_release": False,
            "contour_points": [],
            "contour_file": "",
            "mixed_contour_elements": [],
            "mixed_contour_file": "",
            "demo_summary_file": "",
        }
    )
