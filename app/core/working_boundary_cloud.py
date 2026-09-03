from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal

import numpy as np


HoleOrigin = Literal["detector", "manual"]
HoleDecisionSource = Literal["automatic", "user"]


def _validate_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_positive_real(value: object, name: str) -> float:
    result = _validate_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def _validate_non_negative_real(value: object, name: str) -> float:
    result = _validate_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_point_xy(point: object, name: str) -> tuple[float, float]:
    if isinstance(point, np.ndarray):
        if point.shape != (2,):
            raise ValueError(f"{name} must have shape (2,)")
        values = point.tolist()
    else:
        try:
            values = tuple(point)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(f"{name} must be a two-coordinate point") from exc
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two coordinates")
    return (
        _validate_real(values[0], f"{name} x-coordinate"),
        _validate_real(values[1], f"{name} y-coordinate"),
    )


def _immutable_numeric_array(
    value: np.ndarray,
    *,
    columns: int,
    name: str,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.ndim != 2 or value.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns})")
    if not (
        np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
    ):
        raise TypeError(f"{name} must have a real numeric dtype")

    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} coordinates must be finite")
    result.setflags(write=False)
    return result


def _validate_contour_world(value: np.ndarray) -> np.ndarray:
    contour = _immutable_numeric_array(
        value,
        columns=2,
        name="preliminary_contour_world",
    )
    if len(contour) < 3:
        raise ValueError("preliminary_contour_world must contain at least three points")
    if len(np.unique(contour, axis=0)) < 3:
        raise ValueError(
            "preliminary_contour_world must contain at least three distinct points"
        )

    relative = contour - contour[0]
    twice_area = float(
        np.dot(relative[:, 0], np.roll(relative[:, 1], -1))
        - np.dot(relative[:, 1], np.roll(relative[:, 0], -1))
    )
    if twice_area == 0.0:
        raise ValueError("preliminary_contour_world must enclose a non-zero area")
    return contour


def validate_search_width(value: object, name: str = "search_width") -> float:
    """Validate a per-side boundary search width in world/source units."""
    return _validate_positive_real(value, name)


def _empty_xyz_points() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AxisAlignedBoundingBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        min_x = _validate_real(self.min_x, "min_x")
        min_y = _validate_real(self.min_y, "min_y")
        max_x = _validate_real(self.max_x, "max_x")
        max_y = _validate_real(self.max_y, "max_y")
        if min_x > max_x or min_y > max_y:
            raise ValueError("bounding-box minimums must not exceed maximums")
        object.__setattr__(self, "min_x", min_x)
        object.__setattr__(self, "min_y", min_y)
        object.__setattr__(self, "max_x", max_x)
        object.__setattr__(self, "max_y", max_y)

    def contains(self, point_xy: object) -> bool:
        x, y = _validate_point_xy(point_xy, "point_xy")
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y


@dataclass(frozen=True, slots=True)
class BoundaryExtractionParameters:
    outer_boundary_search_width: float
    hole_boundary_search_width: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outer_boundary_search_width",
            validate_search_width(
                self.outer_boundary_search_width,
                "outer_boundary_search_width",
            ),
        )
        object.__setattr__(
            self,
            "hole_boundary_search_width",
            validate_search_width(
                self.hole_boundary_search_width,
                "hole_boundary_search_width",
            ),
        )


@dataclass(frozen=True, slots=True)
class HoleDetectorMetrics:
    area_cells: int
    area_mm2: float
    bbox_width_mm: float
    bbox_height_mm: float
    aspect_ratio: float
    circularity: float
    mean_error_mm: float
    max_error_mm: float
    error_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "area_cells",
            _validate_integer(self.area_cells, "area_cells", minimum=0),
        )
        for name in (
            "area_mm2",
            "aspect_ratio",
            "circularity",
            "mean_error_mm",
            "max_error_mm",
            "error_ratio",
        ):
            object.__setattr__(
                self,
                name,
                _validate_non_negative_real(getattr(self, name), name),
            )
        for name in ("bbox_width_mm", "bbox_height_mm"):
            object.__setattr__(
                self,
                name,
                _validate_positive_real(getattr(self, name), name),
            )


@dataclass(frozen=True, slots=True)
class HoleDecisionSnapshot:
    hole_id: int
    group_id: str | None
    origin: HoleOrigin
    automatic_accepted: bool | None
    final_accepted: bool
    decision_source: HoleDecisionSource
    automatic_reject_reason: str | None
    preliminary_center_x: float
    preliminary_center_y: float
    preliminary_radius: float
    detector_metrics: HoleDetectorMetrics | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hole_id",
            _validate_integer(self.hole_id, "hole_id", minimum=1),
        )
        if self.group_id is not None:
            if not isinstance(self.group_id, str):
                raise TypeError("group_id must be a string or None")
            if not self.group_id:
                raise ValueError("group_id must not be empty")
        if self.origin not in ("detector", "manual"):
            raise ValueError("origin must be 'detector' or 'manual'")
        if self.decision_source not in ("automatic", "user"):
            raise ValueError("decision_source must be 'automatic' or 'user'")
        if not isinstance(self.final_accepted, (bool, np.bool_)):
            raise TypeError("final_accepted must be a boolean")
        object.__setattr__(self, "final_accepted", bool(self.final_accepted))
        if self.automatic_accepted is not None:
            if not isinstance(self.automatic_accepted, (bool, np.bool_)):
                raise TypeError("automatic_accepted must be a boolean or None")
            object.__setattr__(
                self,
                "automatic_accepted",
                bool(self.automatic_accepted),
            )
        if self.automatic_reject_reason is not None and not isinstance(
            self.automatic_reject_reason,
            str,
        ):
            raise TypeError("automatic_reject_reason must be a string or None")
        if self.detector_metrics is not None and not isinstance(
            self.detector_metrics,
            HoleDetectorMetrics,
        ):
            raise TypeError("detector_metrics must be HoleDetectorMetrics or None")

        if self.origin == "manual":
            if self.automatic_accepted is not None:
                raise ValueError("manual holes cannot have an automatic decision")
            if self.decision_source != "user":
                raise ValueError("manual holes must have a user decision source")
            if self.automatic_reject_reason is not None:
                raise ValueError("manual holes cannot have an automatic reject reason")
            if self.detector_metrics is not None:
                raise ValueError("manual holes cannot have detector metrics")
        else:
            if self.automatic_accepted is None:
                raise ValueError("detector holes must have an automatic decision")
            if self.detector_metrics is None:
                raise ValueError("detector holes must have detector metrics")

        if self.decision_source == "automatic":
            if self.origin != "detector":
                raise ValueError("only detector holes can have an automatic decision")
            if self.final_accepted != self.automatic_accepted:
                raise ValueError(
                    "an automatic final decision must match automatic_accepted"
                )

        object.__setattr__(
            self,
            "preliminary_center_x",
            _validate_real(self.preliminary_center_x, "preliminary_center_x"),
        )
        object.__setattr__(
            self,
            "preliminary_center_y",
            _validate_real(self.preliminary_center_y, "preliminary_center_y"),
        )
        object.__setattr__(
            self,
            "preliminary_radius",
            _validate_positive_real(self.preliminary_radius, "preliminary_radius"),
        )

    @property
    def preliminary_center(self) -> tuple[float, float]:
        return self.preliminary_center_x, self.preliminary_center_y


@dataclass(frozen=True, slots=True)
class OuterBoundaryCloud:
    preliminary_contour_world: np.ndarray
    search_width: float
    points_xyz: np.ndarray = field(default_factory=_empty_xyz_points)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preliminary_contour_world",
            _validate_contour_world(self.preliminary_contour_world),
        )
        object.__setattr__(
            self,
            "search_width",
            validate_search_width(self.search_width),
        )
        object.__setattr__(
            self,
            "points_xyz",
            _immutable_numeric_array(
                self.points_xyz,
                columns=3,
                name="points_xyz",
            ),
        )

    def contains_search_point(self, point_xy: object) -> bool:
        return point_is_near_closed_contour(
            point_xy,
            self.preliminary_contour_world,
            self.search_width,
        )

    def segment_search_aabbs(self) -> tuple[AxisAlignedBoundingBox, ...]:
        return closed_contour_segment_aabbs(
            self.preliminary_contour_world,
            self.search_width,
        )


@dataclass(frozen=True, slots=True)
class HoleBoundaryCloud:
    decision: HoleDecisionSnapshot
    search_width: float
    points_xyz: np.ndarray = field(default_factory=_empty_xyz_points)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, HoleDecisionSnapshot):
            raise TypeError("decision must be a HoleDecisionSnapshot")
        if not self.decision.final_accepted:
            raise ValueError("hole boundary clouds require a final accepted decision")
        object.__setattr__(
            self,
            "search_width",
            validate_search_width(self.search_width),
        )
        object.__setattr__(
            self,
            "points_xyz",
            _immutable_numeric_array(
                self.points_xyz,
                columns=3,
                name="points_xyz",
            ),
        )

    def contains_search_point(self, point_xy: object) -> bool:
        return point_is_near_hole_boundary(
            point_xy,
            self.decision.preliminary_center,
            self.decision.preliminary_radius,
            self.search_width,
        )

    def search_aabb(self) -> AxisAlignedBoundingBox:
        return hole_search_aabb(
            self.decision.preliminary_center,
            self.decision.preliminary_radius,
            self.search_width,
        )


@dataclass(frozen=True, slots=True)
class WorkingBoundaryCloud:
    parameters: BoundaryExtractionParameters
    outer_boundary: OuterBoundaryCloud
    holes: tuple[HoleBoundaryCloud, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, BoundaryExtractionParameters):
            raise TypeError("parameters must be BoundaryExtractionParameters")
        if not isinstance(self.outer_boundary, OuterBoundaryCloud):
            raise TypeError("outer_boundary must be OuterBoundaryCloud")
        try:
            holes = tuple(self.holes)
        except TypeError as exc:
            raise TypeError("holes must be an iterable of HoleBoundaryCloud") from exc
        if not all(isinstance(hole, HoleBoundaryCloud) for hole in holes):
            raise TypeError("holes must contain only HoleBoundaryCloud objects")

        hole_ids = [hole.decision.hole_id for hole in holes]
        if len(hole_ids) != len(set(hole_ids)):
            raise ValueError("hole boundary clouds must have unique hole IDs")
        if self.outer_boundary.search_width != self.parameters.outer_boundary_search_width:
            raise ValueError("outer boundary search width does not match parameters")
        if any(
            hole.search_width != self.parameters.hole_boundary_search_width
            for hole in holes
        ):
            raise ValueError("hole boundary search width does not match parameters")
        object.__setattr__(self, "holes", holes)


def segment_search_aabb(
    segment_start: object,
    segment_end: object,
    search_width: object,
) -> AxisAlignedBoundingBox:
    start_x, start_y = _validate_point_xy(segment_start, "segment_start")
    end_x, end_y = _validate_point_xy(segment_end, "segment_end")
    width = validate_search_width(search_width)
    return AxisAlignedBoundingBox(
        min(start_x, end_x) - width,
        min(start_y, end_y) - width,
        max(start_x, end_x) + width,
        max(start_y, end_y) + width,
    )


def closed_contour_segment_aabbs(
    contour_world: np.ndarray,
    search_width: object,
) -> tuple[AxisAlignedBoundingBox, ...]:
    contour = _validate_contour_world(contour_world)
    width = validate_search_width(search_width)
    return tuple(
        segment_search_aabb(contour[index], contour[(index + 1) % len(contour)], width)
        for index in range(len(contour))
    )


def hole_search_aabb(
    center_xy: object,
    radius: object,
    search_width: object,
) -> AxisAlignedBoundingBox:
    center_x, center_y = _validate_point_xy(center_xy, "center_xy")
    validated_radius = _validate_positive_real(radius, "radius")
    width = validate_search_width(search_width)
    extent = validated_radius + width
    return AxisAlignedBoundingBox(
        center_x - extent,
        center_y - extent,
        center_x + extent,
        center_y + extent,
    )


def _point_distance_to_segment(
    point_xy: tuple[float, float],
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> float:
    point = np.asarray(point_xy, dtype=np.float64)
    segment = segment_end - segment_start
    length_squared = float(np.dot(segment, segment))
    if length_squared == 0.0:
        return float(np.linalg.norm(point - segment_start))
    projection = float(np.dot(point - segment_start, segment) / length_squared)
    projection = min(1.0, max(0.0, projection))
    closest = segment_start + projection * segment
    return float(np.linalg.norm(point - closest))


def point_is_near_segment(
    point_xy: object,
    segment_start: object,
    segment_end: object,
    search_width: object,
) -> bool:
    """Return whether a world-space point is within a segment search band."""
    point = _validate_point_xy(point_xy, "point_xy")
    start = np.asarray(_validate_point_xy(segment_start, "segment_start"))
    end = np.asarray(_validate_point_xy(segment_end, "segment_end"))
    width = validate_search_width(search_width)
    return _point_distance_to_segment(point, start, end) <= width


def point_is_near_closed_contour(
    point_xy: object,
    contour_world: np.ndarray,
    search_width: object,
) -> bool:
    """Reference predicate for a per-side band around a closed world-space contour."""
    point = _validate_point_xy(point_xy, "point_xy")
    contour = _validate_contour_world(contour_world)
    width = validate_search_width(search_width)
    return any(
        _point_distance_to_segment(
            point,
            contour[index],
            contour[(index + 1) % len(contour)],
        )
        <= width
        for index in range(len(contour))
    )


def point_is_near_hole_boundary(
    point_xy: object,
    center_xy: object,
    radius: object,
    search_width: object,
) -> bool:
    """Reference predicate for a world-space annulus around a preliminary hole."""
    point_x, point_y = _validate_point_xy(point_xy, "point_xy")
    center_x, center_y = _validate_point_xy(center_xy, "center_xy")
    validated_radius = _validate_positive_real(radius, "radius")
    width = validate_search_width(search_width)
    distance_to_center = math.hypot(point_x - center_x, point_y - center_y)
    return abs(distance_to_center - validated_radius) <= width
