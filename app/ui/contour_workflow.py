from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from app.core.contour_extractor import ContourResult
from app.core.density_grid import DensityGrid
from app.core.mask_edits import rasterize_mask_edit_cells
from app.core.mask_processing import ThresholdMode
from app.core.working_area import WorkingArea
from app.services.coarse_processing import (
    MaskProcessingResult,
    build_processing_masks,
    extract_preliminary_contour,
)


@dataclass(frozen=True, slots=True)
class PreliminaryContourParameters:
    threshold_mode: ThresholdMode = "auto"
    manual_threshold: float | None = None
    min_component_area: int = 0
    keep_largest: bool = False
    fill_holes_area: int = 0
    simplify_mm: float = 0.0


@dataclass(frozen=True, slots=True)
class PreliminaryContourSession:
    masks: MaskProcessingResult
    contour: ContourResult
    density_session: object | None
    working_area: WorkingArea | None
    parameters: PreliminaryContourParameters


@dataclass(frozen=True, slots=True)
class MaskStrokeDelta:
    flat_indices: np.ndarray
    previous_values: np.ndarray


@dataclass(slots=True)
class MaskEditingSession:
    grid: DensityGrid
    working_area: WorkingArea | None
    base_mask: np.ndarray
    edited_mask: np.ndarray
    working_area_mask: np.ndarray
    history: list[MaskStrokeDelta] = field(default_factory=list)
    semantic_edits: list[dict[str, object]] = field(default_factory=list)
    semantic_stroke_lengths: list[int] = field(default_factory=list)
    contour_stale: bool = False
    _active_previous: dict[int, int] | None = None
    _active_had_command: bool = False
    _active_semantic_start: int | None = None

    @classmethod
    def from_preliminary_contour(
        cls,
        result: PreliminaryContourSession,
    ) -> MaskEditingSession:
        base_mask = result.masks.contour_mask
        working_area_mask = working_area_mask_for_grid(
            result.masks.grid,
            result.working_area,
        )
        edited_mask = build_effective_mask(
            base_mask,
            result.masks.grid,
            result.working_area,
            [],
        )
        return cls(
            grid=result.masks.grid,
            working_area=result.working_area,
            base_mask=base_mask,
            edited_mask=edited_mask,
            working_area_mask=working_area_mask,
        )

    def begin_stroke(self) -> None:
        if self._active_previous is None:
            self._active_previous = {}
            self._active_had_command = False
            self._active_semantic_start = len(self.semantic_edits)

    def apply_edit(self, edit: dict[str, object]) -> bool:
        mode = edit.get("mode")
        if mode not in {"add", "remove"}:
            raise ValueError("Mask edit mode must be add or remove.")
        self.begin_stroke()
        assert self._active_previous is not None

        rows, columns = rasterize_clipped_mask_edit_cells(
            self.edited_mask.shape,
            self.grid,
            self.working_area_mask,
            edit,
        )
        if rows.size == 0:
            return False
        self._active_had_command = True
        self.semantic_edits.append(dict(edit))

        new_value = 1 if mode == "add" else 0
        changed = self.edited_mask[rows, columns] != new_value
        rows, columns = rows[changed], columns[changed]
        if rows.size == 0:
            return False

        flat_indices = np.ravel_multi_index(
            (rows, columns),
            self.edited_mask.shape,
        )
        flat_mask = self.edited_mask.ravel()
        for flat_index in flat_indices:
            index = int(flat_index)
            self._active_previous.setdefault(index, int(flat_mask[index]))
        flat_mask[flat_indices] = new_value
        self.edited_mask[~self.working_area_mask] = 0
        self.contour_stale = True
        return True

    def finish_stroke(self) -> bool:
        previous = self._active_previous
        self._active_previous = None
        had_command = self._active_had_command
        self._active_had_command = False
        semantic_start = self._active_semantic_start
        self._active_semantic_start = None
        if not had_command:
            return False
        previous = previous or {}
        indices = np.fromiter(previous.keys(), dtype=np.intp)
        values = np.fromiter(previous.values(), dtype=np.uint8)
        self.history.append(MaskStrokeDelta(indices, values))
        assert semantic_start is not None
        self.semantic_stroke_lengths.append(len(self.semantic_edits) - semantic_start)
        return True

    def undo_last_stroke(self) -> bool:
        self._active_previous = None
        self._active_had_command = False
        self._active_semantic_start = None
        if not self.history:
            return False
        delta = self.history.pop()
        semantic_count = self.semantic_stroke_lengths.pop()
        if semantic_count:
            del self.semantic_edits[-semantic_count:]
        self.edited_mask.ravel()[delta.flat_indices] = delta.previous_values
        self.edited_mask[~self.working_area_mask] = 0
        self.contour_stale = True
        return True

    def clear_edits(self) -> bool:
        self._active_previous = None
        self._active_had_command = False
        self._active_semantic_start = None
        changed = bool(self.semantic_edits) or not np.array_equal(
            self.edited_mask,
            self.base_mask,
        )
        self.edited_mask[...] = self.base_mask > 0
        self.edited_mask[~self.working_area_mask] = 0
        self.history.clear()
        self.semantic_edits.clear()
        self.semantic_stroke_lengths.clear()
        if changed:
            self.contour_stale = True
        return changed

    def mark_contour_rebuilt(self) -> None:
        self.contour_stale = False

    def update_contour_stale(self, contour_mask: np.ndarray) -> bool:
        self.contour_stale = not np.array_equal(self.edited_mask, contour_mask)
        return self.contour_stale


def working_area_mask_for_grid(
    grid: DensityGrid,
    working_area: WorkingArea | None,
) -> np.ndarray:
    if working_area is None:
        return np.ones(grid.density.shape, dtype=bool)
    return working_area.to_grid_mask(grid)


def rasterize_clipped_mask_edit_cells(
    mask_shape: tuple[int, int],
    grid: DensityGrid,
    working_area_mask: np.ndarray,
    edit: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    if working_area_mask.shape != mask_shape:
        raise ValueError("Working Area mask shape does not match the processing mask.")
    rows, columns = rasterize_mask_edit_cells(mask_shape, grid, edit)
    allowed = working_area_mask[rows, columns]
    return rows[allowed], columns[allowed]


def build_effective_mask(
    base_mask: np.ndarray,
    grid: DensityGrid,
    working_area: WorkingArea | None,
    semantic_edits: list[dict[str, object]],
) -> np.ndarray:
    """Replay GUI mask commands on any base mask using GUI clipping semantics."""
    if base_mask.shape != grid.density.shape:
        raise ValueError("Base mask shape does not match the density grid.")

    working_area_mask = working_area_mask_for_grid(grid, working_area)
    effective = (base_mask > 0).astype(np.uint8)
    effective[~working_area_mask] = 0
    for edit in semantic_edits:
        mode = edit.get("mode")
        if mode not in {"add", "remove"}:
            raise ValueError("Mask edit mode must be add or remove.")
        rows, columns = rasterize_clipped_mask_edit_cells(
            effective.shape,
            grid,
            working_area_mask,
            edit,
        )
        effective[rows, columns] = 1 if mode == "add" else 0
    effective[~working_area_mask] = 0
    return effective


def validate_brush_diameter(brush_diameter_mm: object) -> float:
    try:
        diameter = float(brush_diameter_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError("Brush diameter must be numeric.") from exc
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("Brush diameter must be a positive finite value.")
    return diameter


def mask_edits_for_world_segment(
    grid: DensityGrid,
    start_world: tuple[float, float],
    end_world: tuple[float, float],
    *,
    mode: str,
    brush_diameter_mm: object,
    stroke_id: int,
) -> list[dict[str, object]]:
    diameter = validate_brush_diameter(brush_diameter_mm)
    radius = diameter / 2.0
    start_x, start_y = start_world
    end_x, end_y = end_world
    distance = math.hypot(end_x - start_x, end_y - start_y)
    if distance == 0:
        return [
            {
                "stroke_id": stroke_id,
                "mode": mode,
                "x": start_x,
                "y": start_y,
                "radius_mm": radius,
            }
        ]
    max_spacing = max(grid.cell_size * 0.5, radius)
    steps = max(1, int(math.ceil(distance / max_spacing)))
    return [
        {
            "stroke_id": stroke_id,
            "mode": mode,
            "x": start_x + (end_x - start_x) * step / steps,
            "y": start_y + (end_y - start_y) * step / steps,
            "radius_mm": radius,
        }
        for step in range(steps + 1)
    ]


def _validated_manual_threshold(value: object) -> float:
    if value is None:
        raise ValueError("Manual threshold is required.")
    try:
        manual_threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Manual threshold must be numeric.") from exc
    if not math.isfinite(manual_threshold):
        raise ValueError("Manual threshold must be finite.")
    return manual_threshold


def _validated_fill_holes_area(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("Fill holes area must be an integer number of cells.")
    try:
        area = int(value)
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Fill holes area must be an integer number of cells."
        ) from exc
    if not math.isfinite(numeric_value) or numeric_value != area:
        raise ValueError("Fill holes area must be an integer number of cells.")
    if area < 0:
        raise ValueError("Fill holes area must be non-negative.")
    return area


def validate_preliminary_contour_parameters(
    parameters: PreliminaryContourParameters,
) -> PreliminaryContourParameters:
    parameters = replace(
        parameters,
        fill_holes_area=_validated_fill_holes_area(parameters.fill_holes_area),
    )
    if parameters.threshold_mode == "auto":
        return replace(parameters, manual_threshold=None)
    if parameters.threshold_mode != "manual":
        raise ValueError("Threshold mode must be Auto or Manual.")
    return replace(
        parameters,
        manual_threshold=_validated_manual_threshold(parameters.manual_threshold),
    )


def preliminary_contour_parameters_for_threshold(
    threshold_mode: object,
    manual_threshold: object = None,
    *,
    keep_largest: bool = False,
    fill_holes_area: object = 0,
) -> PreliminaryContourParameters:
    validated_fill_holes_area = _validated_fill_holes_area(fill_holes_area)
    normalized_mode = str(threshold_mode).strip().lower()
    if normalized_mode == "auto":
        return PreliminaryContourParameters(
            keep_largest=bool(keep_largest),
            fill_holes_area=validated_fill_holes_area,
        )
    if normalized_mode == "manual":
        return PreliminaryContourParameters(
            threshold_mode="manual",
            manual_threshold=_validated_manual_threshold(manual_threshold),
            keep_largest=bool(keep_largest),
            fill_holes_area=validated_fill_holes_area,
        )
    raise ValueError("Threshold mode must be Auto or Manual.")


def processing_mask_to_preview(
    mask: np.ndarray,
    *,
    preview_width: int,
    preview_height: int,
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Processing mask must be a two-dimensional array.")
    if preview_width <= 0 or preview_height <= 0:
        raise ValueError("Preview dimensions must be positive.")

    preview_mask = np.flipud((mask > 0).astype(np.uint8))
    if preview_mask.shape != (preview_height, preview_width):
        preview_mask = cv2.resize(
            preview_mask,
            (preview_width, preview_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return preview_mask


def find_preliminary_contour_for_working_area(
    grid: DensityGrid,
    working_area: WorkingArea | None,
    *,
    density_session: object | None = None,
    parameters: PreliminaryContourParameters | None = None,
) -> PreliminaryContourSession:
    selected_parameters = validate_preliminary_contour_parameters(
        parameters or PreliminaryContourParameters()
    )
    if working_area is None:
        roi, polygon_roi = None, None
    else:
        roi, polygon_roi = working_area.processing_parameters()
    masks = build_processing_masks(
        grid,
        threshold_mode=selected_parameters.threshold_mode,
        manual_threshold=selected_parameters.manual_threshold,
        min_component_area=selected_parameters.min_component_area,
        keep_largest=selected_parameters.keep_largest,
        roi=roi,
        polygon_roi=polygon_roi,
        mask_edits=None,
        fill_holes_area=selected_parameters.fill_holes_area,
    )

    if not np.any(masks.contour_mask):
        area_name = "Working Area" if working_area is not None else "full scan"
        raise ValueError(f"The processing mask is empty inside the {area_name}.")

    try:
        contour = extract_preliminary_contour(
            masks,
            simplify_mm=selected_parameters.simplify_mm,
        )
    except ValueError as exc:
        raise ValueError("Preliminary contour was not found.") from exc

    return PreliminaryContourSession(
        masks=masks,
        contour=contour,
        density_session=density_session,
        working_area=working_area,
        parameters=selected_parameters,
    )


def rebuild_preliminary_contour_from_edited_mask(
    result: PreliminaryContourSession,
    editing: MaskEditingSession,
) -> PreliminaryContourSession:
    if editing.grid is not result.masks.grid or editing.working_area != result.working_area:
        raise ValueError("Mask edit session does not belong to this contour result.")

    edited_masks = replace(
        result.masks,
        contour_mask=editing.edited_mask.copy(),
    )
    try:
        contour = extract_preliminary_contour(
            edited_masks,
            simplify_mm=result.parameters.simplify_mm,
        )
    except ValueError as exc:
        raise ValueError("Preliminary contour was not found.") from exc

    rebuilt = replace(result, masks=edited_masks, contour=contour)
    editing.mark_contour_rebuilt()
    return rebuilt


def rebase_preliminary_contour_with_edits(
    grid: DensityGrid,
    working_area: WorkingArea | None,
    edits: list[dict[str, object]],
    *,
    density_session: object | None = None,
    parameters: PreliminaryContourParameters | None = None,
) -> tuple[PreliminaryContourSession, MaskEditingSession]:
    result = find_preliminary_contour_for_working_area(
        grid,
        working_area,
        density_session=density_session,
        parameters=parameters,
    )
    editing = MaskEditingSession.from_preliminary_contour(result)

    current_stroke: object = object()
    for index, edit in enumerate(edits):
        stroke_id = edit.get("stroke_id", ("legacy", index))
        if stroke_id != current_stroke:
            editing.finish_stroke()
            editing.begin_stroke()
            current_stroke = stroke_id
        editing.apply_edit(edit)
    editing.finish_stroke()

    if edits:
        result = rebuild_preliminary_contour_from_edited_mask(result, editing)
    return result, editing
