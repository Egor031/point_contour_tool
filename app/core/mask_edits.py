from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core.density_grid import DensityGrid


@dataclass
class MaskEditsStats:
    total_edits: int
    edits_inside_grid: int
    edits_outside_grid: int
    edits_that_touched_white_mask: int
    changed_cells: int
    world_min_x: float | None
    world_min_y: float | None
    world_max_x: float | None
    world_max_y: float | None
    pixel_min_ix: int | None
    pixel_min_iy: int | None
    pixel_max_ix: int | None
    pixel_max_iy: int | None


def rasterize_mask_edit_cells(
    mask_shape: tuple[int, int],
    grid: DensityGrid,
    edit: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return grid row/column indices covered by one existing mask-edit command."""
    height, width = mask_shape
    x = float(edit["x"])
    y = float(edit["y"])
    radius_mm = float(edit.get("radius_mm", 0.0))

    ix = int((x - grid.min_x) / grid.cell_size)
    iy = int((y - grid.min_y) / grid.cell_size)
    radius_cells = max(1, int(math.ceil(radius_mm / grid.cell_size)))

    min_x = max(0, ix - radius_cells)
    max_x = min(width - 1, ix + radius_cells)
    min_y = max(0, iy - radius_cells)
    max_y = min(height - 1, iy + radius_cells)
    if min_x > max_x or min_y > max_y:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty

    local = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
    cv2.circle(
        local,
        center=(ix - min_x, iy - min_y),
        radius=radius_cells,
        color=1,
        thickness=-1,
    )
    rows, columns = np.nonzero(local)
    return rows + min_y, columns + min_x


def load_mask_edits(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        edits = payload.get("edits", [])
    else:
        edits = payload

    if not isinstance(edits, list):
        raise ValueError("mask edits JSON must contain a list of edits")

    return edits


def apply_mask_edits(
    mask: np.ndarray,
    grid: DensityGrid,
    edits: list[dict[str, Any]],
) -> tuple[np.ndarray, MaskEditsStats]:
    edited = mask.astype(np.uint8).copy()
    before = edited.copy()
    height, width = edited.shape

    total_edits = len(edits)
    edits_inside_grid = 0
    edits_outside_grid = 0
    edits_that_touched_white_mask = 0

    world_xs: list[float] = []
    world_ys: list[float] = []
    pixel_ixs: list[int] = []
    pixel_iys: list[int] = []

    for edit in edits:
        mode = edit.get("mode")
        if mode not in {"remove", "add"}:
            continue

        x = float(edit["x"])
        y = float(edit["y"])
        ix = int((x - grid.min_x) / grid.cell_size)
        iy = int((y - grid.min_y) / grid.cell_size)

        world_xs.append(x)
        world_ys.append(y)
        pixel_ixs.append(ix)
        pixel_iys.append(iy)

        edit_inside_grid = 0 <= ix < width and 0 <= iy < height
        if edit_inside_grid:
            edits_inside_grid += 1
        else:
            edits_outside_grid += 1

        rows, columns = rasterize_mask_edit_cells(edited.shape, grid, edit)
        if mode == "remove" and np.any(edited[rows, columns] > 0):
            edits_that_touched_white_mask += 1

        value = 0 if mode == "remove" else 1
        edited[rows, columns] = value

    changed_cells = int((edited != before).sum())
    stats = MaskEditsStats(
        total_edits=total_edits,
        edits_inside_grid=edits_inside_grid,
        edits_outside_grid=edits_outside_grid,
        edits_that_touched_white_mask=edits_that_touched_white_mask,
        changed_cells=changed_cells,
        world_min_x=min(world_xs) if world_xs else None,
        world_min_y=min(world_ys) if world_ys else None,
        world_max_x=max(world_xs) if world_xs else None,
        world_max_y=max(world_ys) if world_ys else None,
        pixel_min_ix=min(pixel_ixs) if pixel_ixs else None,
        pixel_min_iy=min(pixel_iys) if pixel_iys else None,
        pixel_max_ix=max(pixel_ixs) if pixel_ixs else None,
        pixel_max_iy=max(pixel_iys) if pixel_iys else None,
    )

    return edited, stats
def save_mask_edits_debug_preview(
    mask: np.ndarray,
    grid: DensityGrid,
    edits: list[dict[str, Any]],
    output_path: str | Path,
    max_size: int = 3000,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = mask.shape
    image = (mask.astype(np.uint8) * 180)
    image = np.flipud(image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for edit in edits:
        if edit.get("mode") != "remove":
            continue

        x = float(edit["x"])
        y = float(edit["y"])
        radius_mm = float(edit.get("radius_mm", 0.0))

        ix = int((x - grid.min_x) / grid.cell_size)
        iy = int((y - grid.min_y) / grid.cell_size)
        radius_cells = int(math.ceil(radius_mm / grid.cell_size))
        if radius_cells <= 0:
            radius_cells = 1

        display_iy = (height - 1) - iy
        cv2.circle(
            image_bgr,
            center=(ix, display_iy),
            radius=radius_cells,
            color=(0, 0, 255),
            thickness=2,
        )

    scale = min(max_size / max(width, height), 1.0)
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    cv2.imwrite(str(output_path), image_bgr)
