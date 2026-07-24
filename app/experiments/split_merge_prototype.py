"""Geometric primitives for an experimental cyclic split-merge prototype."""

from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np


@dataclass(frozen=True)
class SegmentStatistics:
    start_index: int
    end_index: int
    range_points_count: int
    internal_points_count: int
    arc_length_mm: float
    chord_length_mm: float
    mean_squared_error_mm2: float
    rms_error_mm: float
    max_error_mm: float


def _integer_index(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _validate_contour(contour: np.ndarray) -> np.ndarray:
    if not isinstance(contour, np.ndarray):
        raise TypeError("contour must be a numpy.ndarray")
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must have shape (N, 2)")
    if len(contour) < 2:
        raise ValueError("contour must contain at least two points")

    try:
        contour_float = contour.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("contour coordinates must be numeric") from exc
    if not np.isfinite(contour_float).all():
        raise ValueError("contour coordinates must be finite")
    return contour_float


def _validate_points(points: np.ndarray) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if not np.isfinite(points_array).all():
        raise ValueError("point coordinates must be finite")
    return points_array


def _validate_endpoint(point: np.ndarray, name: str) -> np.ndarray:
    point_array = np.asarray(point, dtype=np.float64)
    if point_array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    if not np.isfinite(point_array).all():
        raise ValueError(f"{name} coordinates must be finite")
    return point_array


def cyclic_indices(
    point_count: int,
    start_index: int,
    end_index: int,
) -> list[int]:
    """Return an inclusive index range moving forward around a cyclic array."""
    point_count = _integer_index(point_count, "point_count")
    start_index = _integer_index(start_index, "start_index")
    end_index = _integer_index(end_index, "end_index")

    if point_count <= 0:
        raise ValueError("point_count must be greater than zero")
    if not 0 <= start_index < point_count:
        raise IndexError("start_index is outside the contour")
    if not 0 <= end_index < point_count:
        raise IndexError("end_index is outside the contour")

    length = (end_index - start_index) % point_count + 1
    return [(start_index + offset) % point_count for offset in range(length)]


def cyclic_segment_points(
    contour: np.ndarray,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    """Return a copy of the inclusive, forward cyclic contour range."""
    contour_array = _validate_contour(contour)
    indices = cyclic_indices(len(contour_array), start_index, end_index)
    return contour_array[indices]


def point_to_segment_distances(
    points: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean distances from points to a finite line segment."""
    points_array = _validate_points(points)
    start = _validate_endpoint(segment_start, "segment_start")
    end = _validate_endpoint(segment_end, "segment_end")

    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared == 0.0:
        return np.linalg.norm(points_array - start, axis=1)

    projection = ((points_array - start) @ segment) / length_squared
    projection = np.clip(projection, 0.0, 1.0)
    closest_points = start + projection[:, None] * segment
    return np.linalg.norm(points_array - closest_points, axis=1)


def point_to_line_distances(
    points: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> np.ndarray:
    """Compute perpendicular distances from points to an infinite line."""
    points_array = _validate_points(points)
    start = _validate_endpoint(line_start, "line_start")
    end = _validate_endpoint(line_end, "line_end")

    direction = end - start
    direction_length = float(np.linalg.norm(direction))
    if direction_length == 0.0:
        raise ValueError("line_start and line_end must define a non-degenerate line")

    relative = points_array - start
    cross_magnitudes = np.abs(
        direction[0] * relative[:, 1] - direction[1] * relative[:, 0]
    )
    return cross_magnitudes / direction_length


def compute_segment_statistics(
    contour: np.ndarray,
    start_index: int,
    end_index: int,
) -> SegmentStatistics:
    """Compute geometric and chord-error statistics for a cyclic contour side."""
    contour_array = _validate_contour(contour)
    indices = cyclic_indices(len(contour_array), start_index, end_index)
    if indices[0] == indices[-1]:
        raise ValueError("a contour segment must have different endpoint indices")

    segment_points = contour_array[indices]
    segment_start = segment_points[0]
    segment_end = segment_points[-1]
    consecutive_steps = np.diff(segment_points, axis=0)

    arc_length = float(np.linalg.norm(consecutive_steps, axis=1).sum())
    chord_length = float(np.linalg.norm(segment_end - segment_start))
    internal_points = segment_points[1:-1]

    if len(internal_points) == 0:
        mean_squared_error = 0.0
        rms_error = 0.0
        max_error = 0.0
    else:
        distances = point_to_segment_distances(
            internal_points,
            segment_start,
            segment_end,
        )
        mean_squared_error = float(np.mean(np.square(distances)))
        rms_error = float(np.sqrt(mean_squared_error))
        max_error = float(np.max(distances))

    return SegmentStatistics(
        start_index=indices[0],
        end_index=indices[-1],
        range_points_count=len(segment_points),
        internal_points_count=len(internal_points),
        arc_length_mm=arc_length,
        chord_length_mm=chord_length,
        mean_squared_error_mm2=mean_squared_error,
        rms_error_mm=rms_error,
        max_error_mm=max_error,
    )
