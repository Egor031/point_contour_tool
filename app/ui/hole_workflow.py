from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from app.core.hole_detector import HoleCandidate
from app.core.working_boundary_cloud import (
    HoleDecisionSnapshot,
    HoleDetectorMetrics,
)
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
    automatic_acceptance: dict[int, bool] = field(default_factory=dict)
    manual_overrides: dict[int, bool] = field(default_factory=dict)

    @property
    def rejected_count(self) -> int:
        return len(self.result.candidates) - self.result.accepted_count


@dataclass(frozen=True, slots=True)
class HoleHitRegion:
    candidate_id: int
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float


class DuplicateReviewedHoleIdError(ValueError):
    """Raised when two semantic holes have the same public ID."""


class UnsupportedReviewedHoleSourceError(ValueError):
    """Raised for legacy holes without reliable detector/manual provenance."""


def current_hole_review_revision(target: Mapping[str, Any]) -> int:
    value = target.get("hole_review_revision", 0)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise TypeError("hole_review_revision must be an integer")
    if value < 0:
        raise ValueError("hole_review_revision must be non-negative")
    return value


def advance_hole_review_revision(target: MutableMapping[str, Any]) -> int:
    revision = current_hole_review_revision(target) + 1
    target["hole_review_revision"] = revision
    return revision


def _candidate_metrics(candidate: HoleCandidate) -> HoleDetectorMetrics:
    return HoleDetectorMetrics(
        area_cells=candidate.area_cells,
        area_mm2=candidate.area_mm2,
        bbox_width_mm=candidate.bbox_width_mm,
        bbox_height_mm=candidate.bbox_height_mm,
        aspect_ratio=candidate.aspect_ratio,
        circularity=candidate.circularity,
        mean_error_mm=candidate.mean_error_mm,
        max_error_mm=candidate.max_error_mm,
        error_ratio=candidate.error_ratio,
    )


def _detector_hole_snapshot(
    session: HoleDetectionSession,
    candidate: HoleCandidate,
) -> HoleDecisionSnapshot:
    hole_id = int(candidate.id)
    if hole_id in session.automatic_acceptance:
        automatic_accepted = bool(session.automatic_acceptance[hole_id])
    elif hole_id in session.manual_overrides:
        raise ValueError(
            f"automatic decision is missing for overridden hole {hole_id}"
        )
    else:
        automatic_accepted = bool(candidate.accepted)

    has_override = hole_id in session.manual_overrides
    final_accepted = (
        bool(session.manual_overrides[hole_id])
        if has_override
        else automatic_accepted
    )
    return HoleDecisionSnapshot(
        hole_id=hole_id,
        group_id=(
            str(candidate.group_id) if candidate.group_id is not None else None
        ),
        origin="detector",
        automatic_accepted=automatic_accepted,
        final_accepted=final_accepted,
        decision_source="user" if has_override else "automatic",
        automatic_reject_reason=candidate.reject_reason or None,
        preliminary_center_x=candidate.center_x,
        preliminary_center_y=candidate.center_y,
        preliminary_radius=candidate.radius,
        detector_metrics=_candidate_metrics(candidate),
    )


def _manual_hole_snapshot(hole: Mapping[str, Any]) -> HoleDecisionSnapshot:
    source = str(hole.get("source", "legacy"))
    if source != "manual":
        raise UnsupportedReviewedHoleSourceError(
            "loaded legacy holes have no reliable detector/manual provenance "
            "or source-session binding"
        )
    required_fields = ("id", "accepted", "center_x", "center_y", "radius")
    missing = [name for name in required_fields if name not in hole]
    if missing:
        raise ValueError(f"manual hole is missing field: {missing[0]}")
    group_id = hole.get("group_id")
    return HoleDecisionSnapshot(
        hole_id=hole["id"],
        group_id=str(group_id) if group_id is not None else None,
        origin="manual",
        automatic_accepted=None,
        final_accepted=hole["accepted"],
        decision_source="user",
        automatic_reject_reason=None,
        preliminary_center_x=hole["center_x"],
        preliminary_center_y=hole["center_y"],
        preliminary_radius=hole["radius"],
        detector_metrics=None,
    )


def build_reviewed_hole_snapshots(
    session: HoleDetectionSession | None,
    manual_holes: Iterable[Mapping[str, Any]] = (),
) -> tuple[HoleDecisionSnapshot, ...]:
    """Normalize the current live review state into immutable semantic holes."""
    snapshots: list[HoleDecisionSnapshot] = []
    seen_ids: set[int] = set()

    def append_unique(snapshot: HoleDecisionSnapshot) -> None:
        if snapshot.hole_id in seen_ids:
            raise DuplicateReviewedHoleIdError(
                f"duplicate reviewed hole ID: {snapshot.hole_id}"
            )
        seen_ids.add(snapshot.hole_id)
        snapshots.append(snapshot)

    if session is not None:
        if not isinstance(session, HoleDetectionSession):
            raise TypeError("session must be HoleDetectionSession or None")
        for candidate in session.result.candidates:
            append_unique(_detector_hole_snapshot(session, candidate))

    for hole in manual_holes:
        if not isinstance(hole, Mapping):
            raise TypeError("manual_holes must contain mappings")
        append_unique(_manual_hole_snapshot(hole))

    return tuple(snapshots)


def accepted_hole_snapshots(
    snapshots: Iterable[HoleDecisionSnapshot],
) -> tuple[HoleDecisionSnapshot, ...]:
    result = tuple(snapshots)
    if not all(isinstance(snapshot, HoleDecisionSnapshot) for snapshot in result):
        raise TypeError("snapshots must contain only HoleDecisionSnapshot objects")
    return tuple(snapshot for snapshot in result if snapshot.final_accepted)


def hit_test_hole_regions(
    regions: list[HoleHitRegion],
    cursor: tuple[float, float],
    *,
    tolerance_px: float = 5.0,
) -> int | None:
    """Return the closest candidate whose screen-space ellipse contains cursor."""
    tolerance = float(tolerance_px)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Hole hit-test tolerance must be finite and non-negative.")

    cursor_x, cursor_y = cursor
    hits: list[tuple[float, int]] = []
    for region in regions:
        radius_x = max(0.0, float(region.radius_x)) + tolerance
        radius_y = max(0.0, float(region.radius_y)) + tolerance
        if radius_x <= 0 or radius_y <= 0:
            continue
        dx = float(cursor_x) - float(region.center_x)
        dy = float(cursor_y) - float(region.center_y)
        if (dx / radius_x) ** 2 + (dy / radius_y) ** 2 <= 1.0:
            hits.append((dx * dx + dy * dy, int(region.candidate_id)))

    if not hits:
        return None
    return min(hits)[1]


def apply_manual_hole_status(
    session: HoleDetectionSession,
    candidate_id: int,
    *,
    accepted: bool,
) -> bool:
    """Apply a user decision without replacing detector geometry or diagnostics."""
    candidate = next(
        (
            item
            for item in session.result.candidates
            if int(item.id) == int(candidate_id)
        ),
        None,
    )
    if candidate is None:
        raise KeyError(candidate_id)

    requested_status = bool(accepted)
    if bool(candidate.accepted) == requested_status:
        return False

    session.automatic_acceptance.setdefault(
        int(candidate.id),
        bool(candidate.accepted),
    )
    session.manual_overrides[int(candidate.id)] = requested_status
    candidate.accepted = requested_status

    accepted_counts: dict[str, int] = {}
    for item in session.result.candidates:
        if item.accepted and item.group_id is not None:
            group_id = str(item.group_id)
            accepted_counts[group_id] = accepted_counts.get(group_id, 0) + 1
    for group in session.result.groups:
        group_id = str(group.get("id", ""))
        group["count"] = accepted_counts.get(group_id, 0)
    return True


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
        automatic_acceptance={
            int(candidate.id): bool(candidate.accepted)
            for candidate in result.candidates
        },
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
    if had_session:
        advance_hole_review_revision(target)
    return had_session
