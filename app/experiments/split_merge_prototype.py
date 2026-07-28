"""Geometric primitives for an experimental cyclic split-merge prototype."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
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


@dataclass(frozen=True)
class SplitCandidate:
    segment_start_index: int
    segment_end_index: int
    split_index: int
    split_point: tuple[float, float]
    distance_to_line_mm: float
    arc_distance_from_start_mm: float
    arc_distance_to_end_mm: float
    eligible_points_count: int


@dataclass(frozen=True)
class SplitEvaluation:
    segment_start_index: int
    segment_end_index: int
    split_index: int
    parent_statistics: SegmentStatistics
    left_statistics: SegmentStatistics
    right_statistics: SegmentStatistics
    parent_sse_mm2: float
    post_split_sse_mm2: float
    post_split_mse_mm2: float
    post_split_rms_mm: float
    sse_reduction_mm2: float
    sse_reduction_fraction: float | None
    post_split_max_error_mm: float
    max_error_reduction_mm: float


@dataclass(frozen=True)
class SplitDecisionPolicy:
    min_child_arc_length_mm: float
    parent_rms_tolerance_mm: float
    min_rms_reduction_fraction: float
    corner_penalty_rms_mm: float

    def __post_init__(self) -> None:
        parameter_bounds = (
            ("min_child_arc_length_mm", None),
            ("parent_rms_tolerance_mm", None),
            ("min_rms_reduction_fraction", 1.0),
            ("corner_penalty_rms_mm", None),
        )
        for name, maximum in parameter_bounds:
            value = _validate_policy_parameter(
                getattr(self, name),
                name,
                maximum,
            )
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SplitDecision:
    accepted: bool
    reason: str
    evaluation: SplitEvaluation
    parent_rms_mm: float
    post_split_rms_mm: float
    rms_reduction_mm: float
    rms_reduction_fraction: float | None
    corner_penalty_rms_mm: float
    net_gain_mm: float


def _integer_index(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _validate_policy_parameter(
    value: float,
    name: str,
    maximum: float | None,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")

    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite")
    if value_float < 0.0:
        raise ValueError(f"{name} must be non-negative")
    if maximum is not None and value_float > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")
    return value_float


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


def _validate_endpoint_arc_length(value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("min_endpoint_arc_length_mm must be a real number")
    if not isinstance(value, Real):
        raise TypeError("min_endpoint_arc_length_mm must be a real number")

    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError("min_endpoint_arc_length_mm must be finite")
    if value_float < 0.0:
        raise ValueError("min_endpoint_arc_length_mm must be non-negative")
    return value_float


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


def find_split_candidate(
    contour: np.ndarray,
    start_index: int,
    end_index: int,
    min_endpoint_arc_length_mm: float = 0.0,
) -> SplitCandidate | None:
    """Find the largest infinite-line deviation inside one cyclic segment."""
    contour_array = _validate_contour(contour)
    endpoint_margin = _validate_endpoint_arc_length(
        min_endpoint_arc_length_mm
    )
    indices = cyclic_indices(len(contour_array), start_index, end_index)
    if indices[0] == indices[-1]:
        raise ValueError("a contour segment must have different endpoint indices")

    segment_points = contour_array[indices]
    segment_start = segment_points[0]
    segment_end = segment_points[-1]
    if float(np.linalg.norm(segment_end - segment_start)) == 0.0:
        raise ValueError(
            "segment endpoints define a degenerate chord with identical coordinates"
        )

    if len(segment_points) <= 2:
        return None

    step_lengths = np.linalg.norm(np.diff(segment_points, axis=0), axis=1)
    cumulative_arc_lengths = np.concatenate(
        (np.array([0.0], dtype=np.float64), np.cumsum(step_lengths))
    )
    total_arc_length = float(cumulative_arc_lengths[-1])
    internal_arc_from_start = cumulative_arc_lengths[1:-1]
    internal_arc_to_end = total_arc_length - internal_arc_from_start
    eligible_mask = (
        (internal_arc_from_start >= endpoint_margin)
        & (internal_arc_to_end >= endpoint_margin)
    )

    eligible_positions = np.flatnonzero(eligible_mask)
    if len(eligible_positions) == 0:
        return None

    internal_points = segment_points[1:-1]
    eligible_points = internal_points[eligible_mask]
    distances = point_to_line_distances(
        eligible_points,
        segment_start,
        segment_end,
    )
    selected_eligible_position = int(np.argmax(distances))
    selected_internal_position = int(
        eligible_positions[selected_eligible_position]
    )
    selected_segment_position = selected_internal_position + 1
    selected_point = eligible_points[selected_eligible_position]

    return SplitCandidate(
        segment_start_index=indices[0],
        segment_end_index=indices[-1],
        split_index=indices[selected_segment_position],
        split_point=(float(selected_point[0]), float(selected_point[1])),
        distance_to_line_mm=float(distances[selected_eligible_position]),
        arc_distance_from_start_mm=float(
            internal_arc_from_start[selected_internal_position]
        ),
        arc_distance_to_end_mm=float(
            internal_arc_to_end[selected_internal_position]
        ),
        eligible_points_count=len(eligible_points),
    )


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


def evaluate_split(
    contour: np.ndarray,
    start_index: int,
    end_index: int,
    split_index: int,
) -> SplitEvaluation:
    """Measure the geometric error before and after one proposed split."""
    contour_array = _validate_contour(contour)
    segment_indices = cyclic_indices(
        len(contour_array),
        start_index,
        end_index,
    )
    if segment_indices[0] == segment_indices[-1]:
        raise ValueError("a contour segment must have different endpoint indices")

    split_index = _integer_index(split_index, "split_index")
    cyclic_indices(len(contour_array), split_index, split_index)
    if split_index not in segment_indices[1:-1]:
        raise ValueError(
            "split_index must be an internal point of the cyclic segment"
        )

    parent = compute_segment_statistics(
        contour_array,
        segment_indices[0],
        segment_indices[-1],
    )
    left = compute_segment_statistics(
        contour_array,
        segment_indices[0],
        split_index,
    )
    right = compute_segment_statistics(
        contour_array,
        split_index,
        segment_indices[-1],
    )

    parent_sse = (
        parent.mean_squared_error_mm2 * parent.internal_points_count
    )
    left_sse = left.mean_squared_error_mm2 * left.internal_points_count
    right_sse = right.mean_squared_error_mm2 * right.internal_points_count
    post_split_sse = left_sse + right_sse
    post_split_mse = post_split_sse / parent.internal_points_count
    post_split_rms = float(np.sqrt(post_split_mse))

    sse_reduction = parent_sse - post_split_sse
    sse_reduction_fraction = (
        None if parent_sse == 0.0 else sse_reduction / parent_sse
    )
    post_split_max_error = max(left.max_error_mm, right.max_error_mm)
    max_error_reduction = parent.max_error_mm - post_split_max_error

    return SplitEvaluation(
        segment_start_index=segment_indices[0],
        segment_end_index=segment_indices[-1],
        split_index=split_index,
        parent_statistics=parent,
        left_statistics=left,
        right_statistics=right,
        parent_sse_mm2=float(parent_sse),
        post_split_sse_mm2=float(post_split_sse),
        post_split_mse_mm2=float(post_split_mse),
        post_split_rms_mm=post_split_rms,
        sse_reduction_mm2=float(sse_reduction),
        sse_reduction_fraction=(
            None
            if sse_reduction_fraction is None
            else float(sse_reduction_fraction)
        ),
        post_split_max_error_mm=float(post_split_max_error),
        max_error_reduction_mm=float(max_error_reduction),
    )


def decide_split(
    evaluation: SplitEvaluation,
    policy: SplitDecisionPolicy,
) -> SplitDecision:
    """Apply an explainable local split policy to an existing evaluation."""
    if not isinstance(evaluation, SplitEvaluation):
        raise TypeError("evaluation must be a SplitEvaluation")
    if not isinstance(policy, SplitDecisionPolicy):
        raise TypeError("policy must be a SplitDecisionPolicy")

    parent_rms = evaluation.parent_statistics.rms_error_mm
    post_split_rms = evaluation.post_split_rms_mm
    rms_reduction = parent_rms - post_split_rms
    rms_reduction_fraction = (
        None if parent_rms == 0.0 else rms_reduction / parent_rms
    )
    net_gain = rms_reduction - policy.corner_penalty_rms_mm

    if (
        evaluation.left_statistics.arc_length_mm
        < policy.min_child_arc_length_mm
    ):
        accepted = False
        reason = "left_child_too_short"
    elif (
        evaluation.right_statistics.arc_length_mm
        < policy.min_child_arc_length_mm
    ):
        accepted = False
        reason = "right_child_too_short"
    elif parent_rms <= policy.parent_rms_tolerance_mm:
        accepted = False
        reason = "parent_within_tolerance"
    elif rms_reduction <= 0.0:
        accepted = False
        reason = "no_rms_improvement"
    elif (
        rms_reduction_fraction is not None
        and rms_reduction_fraction < policy.min_rms_reduction_fraction
    ):
        accepted = False
        reason = "relative_improvement_too_small"
    elif net_gain <= 0.0:
        accepted = False
        reason = "corner_penalty_not_overcome"
    else:
        accepted = True
        reason = "accepted"

    return SplitDecision(
        accepted=accepted,
        reason=reason,
        evaluation=evaluation,
        parent_rms_mm=float(parent_rms),
        post_split_rms_mm=float(post_split_rms),
        rms_reduction_mm=float(rms_reduction),
        rms_reduction_fraction=(
            None
            if rms_reduction_fraction is None
            else float(rms_reduction_fraction)
        ),
        corner_penalty_rms_mm=policy.corner_penalty_rms_mm,
        net_gain_mm=float(net_gain),
    )
