from __future__ import annotations

import math
from collections.abc import MutableMapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from app.core.working_area import WorkingArea
from app.services.coarse_processing import (
    HoleDetectionResult,
    find_hole_candidates,
)
from app.ui.contour_workflow import (
    MaskEditingSession,
    PreliminaryContourSession,
    build_effective_mask,
)


@dataclass(frozen=True, slots=True)
class HoleDetectionParameters:
    min_diameter_mm: float = 8.0
    max_diameter_mm: float | None = None
    min_circularity: float = 0.55
    max_aspect_ratio_deviation: float = 0.35
    max_error_ratio: float = 0.18
    group_tolerance_mm: float = 1.5


@dataclass(frozen=True, slots=True)
class HoleDetectionSession:
    result: HoleDetectionResult
    density_session: object | None
    contour_session: PreliminaryContourSession
    mask_editing_session: MaskEditingSession
    working_area: WorkingArea | None
    coarse_mask_revision: int
    parameters: HoleDetectionParameters

    @property
    def rejected_count(self) -> int:
        return len(self.result.candidates) - self.result.accepted_count


def validate_hole_detection_parameters(
    min_diameter_mm: object,
    max_diameter_mm: object,
    min_circularity: object = 0.55,
    max_error_ratio: object = 0.18,
) -> HoleDetectionParameters:
    try:
        minimum = float(min_diameter_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError("Hole diameters must be numeric.") from exc

    if max_diameter_mm is None:
        maximum = None
    else:
        try:
            maximum = float(max_diameter_mm)
        except (TypeError, ValueError) as exc:
            raise ValueError("Hole diameters must be numeric.") from exc

    if not math.isfinite(minimum) or minimum < 0:
        raise ValueError("Minimum hole diameter must be finite and non-negative.")
    if maximum is not None and not math.isfinite(maximum):
        raise ValueError("Maximum hole diameter must be finite.")

    normalized_maximum = None if maximum is None or maximum <= 0 else maximum
    if normalized_maximum is not None and normalized_maximum < minimum:
        raise ValueError(
            "Maximum hole diameter must be zero (unlimited) or at least the minimum."
        )

    try:
        circularity = float(min_circularity)
        error_ratio = float(max_error_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("Hole quality limits must be numeric.") from exc
    if not math.isfinite(circularity) or circularity < 0:
        raise ValueError("Minimum circularity must be finite and non-negative.")
    if not math.isfinite(error_ratio) or error_ratio < 0:
        raise ValueError(
            "Maximum circle error ratio must be finite and non-negative."
        )

    return HoleDetectionParameters(
        min_diameter_mm=minimum,
        max_diameter_mm=normalized_maximum,
        min_circularity=circularity,
        max_error_ratio=error_ratio,
    )


def current_effective_hole_mask(
    contour_session: PreliminaryContourSession,
    editing_session: MaskEditingSession,
) -> np.ndarray:
    if editing_session.grid is not contour_session.masks.grid:
        raise ValueError("Mask edit session does not belong to this contour result.")
    if editing_session.working_area != contour_session.working_area:
        raise ValueError("Mask edit session has a different Working Area.")
    return build_effective_mask(
        contour_session.masks.mask_for_holes,
        contour_session.masks.grid,
        contour_session.working_area,
        editing_session.semantic_edits,
    )


def find_holes_for_current_mask(
    contour_session: PreliminaryContourSession,
    editing_session: MaskEditingSession,
    *,
    density_session: object | None,
    coarse_mask_revision: int,
    parameters: HoleDetectionParameters | None = None,
) -> HoleDetectionSession:
    selected = parameters or HoleDetectionParameters()
    validated_limits = validate_hole_detection_parameters(
        selected.min_diameter_mm,
        selected.max_diameter_mm,
        selected.min_circularity,
        selected.max_error_ratio,
    )
    selected = replace(
        selected,
        min_diameter_mm=validated_limits.min_diameter_mm,
        max_diameter_mm=validated_limits.max_diameter_mm,
        min_circularity=validated_limits.min_circularity,
        max_error_ratio=validated_limits.max_error_ratio,
    )
    if (
        contour_session.density_session is not None
        and contour_session.density_session is not density_session
    ):
        raise ValueError("Contour result does not belong to this density session.")
    effective_mask = current_effective_hole_mask(
        contour_session,
        editing_session,
    )
    adapted_masks = replace(
        contour_session.masks,
        mask_for_holes=effective_mask,
    )
    result = find_hole_candidates(
        adapted_masks,
        min_diameter_mm=selected.min_diameter_mm,
        max_diameter_mm=selected.max_diameter_mm,
        min_circularity=selected.min_circularity,
        max_aspect_ratio_deviation=selected.max_aspect_ratio_deviation,
        max_error_ratio=selected.max_error_ratio,
        group_tolerance_mm=selected.group_tolerance_mm,
    )
    return HoleDetectionSession(
        result=result,
        density_session=density_session,
        contour_session=contour_session,
        mask_editing_session=editing_session,
        working_area=contour_session.working_area,
        coarse_mask_revision=int(coarse_mask_revision),
        parameters=selected,
    )


def hole_detection_session_is_current(
    session: object,
    *,
    density_session: object | None,
    contour_session: object,
    mask_editing_session: object,
    coarse_mask_revision: int,
) -> bool:
    return (
        isinstance(session, HoleDetectionSession)
        and session.density_session is density_session
        and session.contour_session is contour_session
        and session.mask_editing_session is mask_editing_session
        and session.coarse_mask_revision == int(coarse_mask_revision)
    )


def invalidate_hole_detection_state(
    target: MutableMapping[str, Any],
    *,
    mark_outdated: bool = True,
) -> bool:
    had_session = isinstance(
        target.get("hole_detection_session"),
        HoleDetectionSession,
    )
    target["hole_detection_session"] = None
    target["holes_outdated"] = bool(
        mark_outdated
        and (had_session or target.get("holes_outdated", False))
    )
    return had_session
