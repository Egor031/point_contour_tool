from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

from app.core.density_parameters import normalize_cell_size
from app.core.density_grid import DensityGrid
from app.core.preview_export import density_to_image
from app.core.working_area import WorkingArea


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

    try:
        validated_cell_size = normalize_cell_size(validated_cell_size)
    except ValueError as exc:
        raise ValueError("Cell size must be finite and greater than zero.") from exc

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


def apply_working_area_state(
    target: MutableMapping[str, Any],
    working_area: WorkingArea,
    density_session: object | None,
) -> None:
    target["active_working_area"] = working_area
    target["working_area_density_session"] = density_session
    target["selection_applied"] = True
    target["selection_kind"] = working_area.kind
    target["selection_polygon_points"] = list(working_area.polygon_points)
    target["editing_overlay_visible"] = False
    invalidate_preliminary_contour_state(target)


def invalidate_preliminary_contour_state(
    target: MutableMapping[str, Any],
) -> None:
    had_hole_session = target.get("hole_detection_session") is not None
    target["contour_processing_result"] = None
    target["mask_editing_session"] = None
    target["contour_points"] = []
    target["contour_file"] = ""
    target["processing_mask_preview"] = None
    target["mask_edits"] = []
    target["last_brush_image"] = None
    target["last_brush_world"] = None
    target["brush_cursor_image"] = None
    target["active_brush_stroke_id"] = None
    target["next_brush_stroke_id"] = 1
    target["undo_history"] = []
    target["coarse_mask_revision"] = int(
        target.get("coarse_mask_revision", 0)
    ) + 1
    target["hole_detection_session"] = None
    target["holes_outdated"] = bool(
        had_hole_session or target.get("holes_outdated", False)
    )


def enter_rectangle_mode_state(target: MutableMapping[str, Any]) -> None:
    target["mode"] = "rectangle"
    target["roi_first_world"] = None
    target["roi_current_world"] = None
    target["editing_overlay_visible"] = True


def enter_polygon_mode_state(target: MutableMapping[str, Any]) -> None:
    target["mode"] = "polygon"
    target["roi_first_world"] = None
    target["roi_current_world"] = None
    target["editing_overlay_visible"] = True

    if target.get("polygon_points"):
        return

    working_area = target.get("active_working_area")
    if isinstance(working_area, WorkingArea) and working_area.kind == "polygon":
        target["polygon_points"] = list(working_area.polygon_points)
        target["polygon_finished"] = True


def begin_rectangle_draft(
    target: MutableMapping[str, Any],
    anchor: tuple[float, float],
) -> None:
    target["editing_overlay_visible"] = True
    target["rectangle_roi"] = None
    target["roi_first_world"] = anchor
    target["roi_current_world"] = anchor


def update_rectangle_transient(
    target: MutableMapping[str, Any],
    current: tuple[float, float],
) -> bool:
    if target.get("mode") != "rectangle" or target.get("roi_first_world") is None:
        return False
    target["roi_current_world"] = current
    return True


def finish_rectangle_draft(
    target: MutableMapping[str, Any],
    second_point: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    anchor = target.get("roi_first_world")
    if anchor is None:
        return None

    first_x, first_y = anchor
    second_x, second_y = second_point
    bounds = (
        min(first_x, second_x),
        min(first_y, second_y),
        max(first_x, second_x),
        max(first_y, second_y),
    )
    target["rectangle_roi"] = bounds
    target["roi_first_world"] = None
    target["roi_current_world"] = None
    return bounds


def clear_polygon_draft_state(target: MutableMapping[str, Any]) -> None:
    target["polygon_points"] = []
    target["polygon_finished"] = False


def active_working_area_is_visible(target: MutableMapping[str, Any]) -> bool:
    return (
        isinstance(target.get("active_working_area"), WorkingArea)
        and not bool(target.get("editing_overlay_visible"))
    )


def working_area_draft_visibility(
    target: MutableMapping[str, Any],
) -> tuple[bool, bool]:
    if not bool(target.get("editing_overlay_visible")):
        return False, False
    mode = target.get("mode")
    return mode == "rectangle", mode == "polygon"


def clear_working_area_state(target: MutableMapping[str, Any]) -> None:
    had_hole_session = target.get("hole_detection_session") is not None
    target.update(
        {
            "roi_first_world": None,
            "roi_current_world": None,
            "rectangle_roi": None,
            "polygon_finished": False,
            "selection_applied": False,
            "editing_overlay_visible": False,
            "selection_kind": None,
            "selection_polygon_points": [],
            "mode": "rectangle",
            "polygon_points": [],
            "active_working_area": None,
            "working_area_density_session": None,
            "contour_processing_result": None,
            "mask_editing_session": None,
            "contour_points": [],
            "contour_file": "",
            "processing_mask_preview": None,
            "mask_edits": [],
            "last_brush_image": None,
            "last_brush_world": None,
            "brush_cursor_image": None,
            "active_brush_stroke_id": None,
            "next_brush_stroke_id": 1,
            "undo_history": [],
            "coarse_mask_revision": int(
                target.get("coarse_mask_revision", 0)
            ) + 1,
            "hole_detection_session": None,
            "holes_outdated": bool(
                had_hole_session or target.get("holes_outdated", False)
            ),
        }
    )


def reset_source_dependent_state(target: MutableMapping[str, Any]) -> None:
    target.update(
        {
            "roi_first_world": None,
            "roi_current_world": None,
            "rectangle_roi": None,
            "polygon_finished": False,
            "selection_applied": False,
            "editing_overlay_visible": True,
            "selection_kind": None,
            "selection_polygon_points": [],
            "mode": "rectangle",
            "polygon_points": [],
            "active_working_area": None,
            "working_area_density_session": None,
            "mask_edits": [],
            "last_brush_image": None,
            "last_brush_world": None,
            "brush_cursor_image": None,
            "active_brush_stroke_id": None,
            "next_brush_stroke_id": 1,
            "undo_history": [],
            "coarse_mask_revision": int(
                target.get("coarse_mask_revision", 0)
            ) + 1,
            "hole_detection_session": None,
            "holes_outdated": False,
            "hole_overlay_source": None,
            "holes": [],
            "hole_groups": [],
            "visible_hole_group_ids": {},
            "pick_manual_hole_center": False,
            "manual_hole_center_world": None,
            "suppress_brush_until_mouse_release": False,
            "contour_points": [],
            "contour_file": "",
            "contour_processing_result": None,
            "mask_editing_session": None,
            "processing_mask_preview": None,
            "mixed_contour_elements": [],
            "mixed_contour_file": "",
            "demo_summary_file": "",
        }
    )
