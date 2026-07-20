from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import ezdxf
import numpy as np

from app.core.xyz_reader import iter_xyz_points


@dataclass
class RefinedLine:
    id: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    length_mm: float
    points_count: int
    mean_error_mm: float
    max_error_mm: float
    edge_filter_used: bool
    points_before_edge_filter: int
    points_after_edge_filter: int
    trim_outlier_percent: float
    points_after_trim: int
    segment_angle_deg: float
    fitted_angle_deg: float
    angle_diff_deg: float
    line_end_trim_percent: float
    points_after_end_trim: int
    normal_x: float
    normal_y: float
    normal_offset_min: float | None
    normal_offset_max: float | None
    selected_offset_min: float | None
    selected_offset_max: float | None
    core_found: bool
    core_start_t: float | None
    core_end_t: float | None
    core_windows_count: int
    total_windows_count: int
    window_diagnostics: list[dict]
    mean_distance_to_contour_mm: float
    max_distance_to_contour_mm: float
    max_point_contour_distance_mm: float
    points_after_contour_distance_filter: int
    contour_start_index: int
    contour_end_index: int
    endpoint_mode: str
    endpoint_t_start: float | None
    endpoint_t_end: float | None


@dataclass
class SegmentDiagnostic:
    id: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    length_mm: float
    points_count: int
    mean_error_mm: float | None
    max_error_mm: float | None
    accepted: bool
    reject_reason: str
    edge_filter_used: bool
    points_before_edge_filter: int
    points_after_edge_filter: int
    trim_outlier_percent: float
    points_after_trim: int
    segment_angle_deg: float
    fitted_angle_deg: float | None
    angle_diff_deg: float | None
    line_end_trim_percent: float
    points_after_end_trim: int
    normal_x: float
    normal_y: float
    normal_offset_min: float | None
    normal_offset_max: float | None
    selected_offset_min: float | None
    selected_offset_max: float | None
    core_found: bool
    core_start_t: float | None
    core_end_t: float | None
    core_windows_count: int
    total_windows_count: int
    window_diagnostics: list[dict]
    mean_distance_to_contour_mm: float | None
    max_distance_to_contour_mm: float | None
    max_point_contour_distance_mm: float
    points_after_contour_distance_filter: int
    endpoint_mode: str
    endpoint_t_start: float | None
    endpoint_t_end: float | None


@dataclass
class LineApproximationResult:
    lines: list[RefinedLine]
    rejected_segments: list[SegmentDiagnostic]
    total_segments: int
    rejected_by_reason: dict[str, int]
    chained_lines: list["ChainedLine"]
    chained_successful_intersections: int
    chained_warnings_count: int
    mixed_contour_elements: list[dict]


@dataclass
class ChainedLine:
    original_line_id: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    length_mm: float
    intersection_prev_ok: bool
    intersection_next_ok: bool
    warnings: list[str]


def load_contour_csv(path: str | Path) -> np.ndarray:
    path = Path(path)
    points = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        field_names = {str(name).strip().lower() for name in reader.fieldnames or []}
        if "x" not in field_names or "y" not in field_names:
            raise ValueError("contour CSV must contain x,y columns")

        for row_number, row in enumerate(reader, start=2):
            normalized_row = {str(key).strip().lower(): value for key, value in row.items()}
            try:
                points.append((float(normalized_row["x"]), float(normalized_row["y"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid contour coordinates at row {row_number}") from exc

    if len(points) < 3:
        raise ValueError("contour CSV must contain at least 3 points")

    return np.asarray(points, dtype=np.float64)


def load_boundary_points(path: str | Path) -> np.ndarray:
    points = [(x, y) for x, y, _z in iter_xyz_points(path)]
    if len(points) == 0:
        raise ValueError("boundary points file contains no valid XYZ points")

    return np.asarray(points, dtype=np.float64)


def simplify_contour_points(
    contour_points: np.ndarray,
    tolerance_mm: float,
) -> np.ndarray:
    if contour_points.ndim != 2 or contour_points.shape[1] != 2:
        raise ValueError("contour_points must have shape (N, 2)")

    if len(contour_points) < 3 or tolerance_mm <= 0:
        return contour_points.astype(np.float64, copy=True)

    contour_cv = contour_points.reshape((-1, 1, 2)).astype(np.float32)
    simplified = cv2.approxPolyDP(
        contour_cv,
        epsilon=float(tolerance_mm),
        closed=True,
    )

    return simplified[:, 0, :].astype(np.float64)


def _points_near_segment(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    band_mm: float,
) -> np.ndarray:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 0:
        return np.empty((0, 2), dtype=np.float64)

    relative = points - start
    t = np.clip((relative @ segment) / length_sq, 0.0, 1.0)
    closest = start + t[:, None] * segment
    distances = np.linalg.norm(points - closest, axis=1)

    return points[distances <= band_mm]


def _contour_segment_indices(
    contour_points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[int, int]:
    if len(contour_points) == 0:
        return 0, 0

    start_index = int(np.argmin(np.linalg.norm(contour_points - start, axis=1)))
    end_index = int(np.argmin(np.linalg.norm(contour_points - end, axis=1)))
    return start_index, end_index


def _contour_points_between_indices(
    contour_points: np.ndarray,
    start_index: int,
    end_index: int,
) -> np.ndarray:
    if len(contour_points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    if start_index <= end_index:
        segment_points = contour_points[start_index : end_index + 1]
    else:
        segment_points = np.vstack(
            (contour_points[start_index:], contour_points[: end_index + 1])
        )

    if len(segment_points) < 2:
        return np.vstack((start, end)).astype(np.float64)

    return segment_points.astype(np.float64, copy=False)


def _contour_segment_points(
    contour_points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    start_index, end_index = _contour_segment_indices(contour_points, start, end)
    return _contour_points_between_indices(contour_points, start_index, end_index)


def _point_distances_to_polyline(
    points: np.ndarray,
    polyline_points: np.ndarray,
) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=np.float64)
    if len(polyline_points) < 2:
        return np.full((len(points),), np.inf, dtype=np.float64)

    min_distances = np.full((len(points),), np.inf, dtype=np.float64)
    for index in range(len(polyline_points) - 1):
        start = polyline_points[index]
        end = polyline_points[index + 1]
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 0:
            distances = np.linalg.norm(points - start, axis=1)
        else:
            t = np.clip(((points - start) @ segment) / length_sq, 0.0, 1.0)
            closest = start + t[:, None] * segment
            distances = np.linalg.norm(points - closest, axis=1)
        min_distances = np.minimum(min_distances, distances)

    return min_distances


def _point_distances_to_segment(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=np.float64)

    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 0:
        return np.linalg.norm(points - start, axis=1)

    t = np.clip(((points - start) @ segment) / length_sq, 0.0, 1.0)
    closest = start + t[:, None] * segment
    return np.linalg.norm(points - closest, axis=1)


def _filter_points_near_contour(
    points: np.ndarray,
    contour_segment: np.ndarray,
    max_distance_mm: float,
) -> np.ndarray:
    if max_distance_mm <= 0:
        return points

    distances = _point_distances_to_polyline(points, contour_segment)
    return points[distances <= max_distance_mm]


def _write_debug_points_csv(
    path: Path,
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    contour_segment: np.ndarray,
    segment_mid: np.ndarray,
    normal: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t_values = _segment_projection_t(points, start, end)
    segment_distances = _point_distances_to_segment(points, start, end)
    contour_distances = _point_distances_to_polyline(points, contour_segment)
    normal_offsets = (points - segment_mid) @ normal if len(points) else []

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "x",
                "y",
                "z",
                "t_along_segment",
                "distance_to_segment",
                "distance_to_contour",
                "normal_offset",
            ]
        )
        for index, point in enumerate(points):
            writer.writerow(
                [
                    float(point[0]),
                    float(point[1]),
                    0.0,
                    float(t_values[index]),
                    float(segment_distances[index]),
                    float(contour_distances[index]),
                    float(normal_offsets[index]),
                ]
            )


def _save_debug_line_files(
    output_dir: Path,
    base_name: str,
    segment_id: int,
    stages: dict[str, np.ndarray],
    summary: dict,
    start: np.ndarray,
    end: np.ndarray,
    contour_segment: np.ndarray,
    segment_mid: np.ndarray,
    normal: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffixes = {
        "all_candidate_points": "all",
        "after_contour_distance_filter": "contour",
        "after_end_trim": "end_trim",
        "after_edge_filter": "edge",
        "after_core_filter": "core",
        "after_robust_trim": "robust_trim",
    }
    for stage_name, suffix in suffixes.items():
        points = stages.get(stage_name)
        if points is None:
            points = np.empty((0, 2), dtype=np.float64)
        _write_debug_points_csv(
            output_dir / f"{base_name}_debug_line_{segment_id}_{suffix}.csv",
            points,
            start,
            end,
            contour_segment,
            segment_mid,
            normal,
        )

    json_path = output_dir / f"{base_name}_debug_line_{segment_id}.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0].astype(np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise ValueError("cannot fit line to degenerate points")

    return centroid, direction / norm


def _project_point_to_line(
    point: np.ndarray,
    line_point: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    return line_point + np.dot(point - line_point, direction) * direction


def _line_endpoint_points(
    mode: str,
    start: np.ndarray,
    end: np.ndarray,
    fit_points: np.ndarray,
    line_point: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    if mode == "contour_projection":
        t_start = float(np.dot(start - line_point, direction))
        t_end = float(np.dot(end - line_point, direction))
        return (
            line_point + t_start * direction,
            line_point + t_end * direction,
            t_start,
            t_end,
        )

    if mode == "fit_points_range":
        if len(fit_points) == 0:
            return (
                line_point.copy(),
                line_point.copy(),
                None,
                None,
            )

        t_values = (fit_points - line_point) @ direction
        t_start = float(np.percentile(t_values, 5.0))
        t_end = float(np.percentile(t_values, 95.0))
        if t_end < t_start:
            t_start, t_end = t_end, t_start

        return (
            line_point + t_start * direction,
            line_point + t_end * direction,
            t_start,
            t_end,
        )

    raise ValueError(f"unsupported endpoint_mode: {mode}")


def _line_errors(
    points: np.ndarray,
    line_point: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    relative = points - line_point
    projected = line_point + (relative @ direction)[:, None] * direction
    return np.linalg.norm(points - projected, axis=1)


def _line_angle_deg(direction: np.ndarray) -> float:
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    angle = angle % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def _angle_diff_deg(angle_a: float, angle_b: float) -> float:
    diff = abs((angle_a - angle_b) % 180.0)
    return min(diff, 180.0 - diff)


def _contour_center(contour_points: np.ndarray) -> np.ndarray:
    min_xy = contour_points.min(axis=0)
    max_xy = contour_points.max(axis=0)
    return (min_xy + max_xy) / 2.0


def _outward_segment_normal(
    start: np.ndarray,
    end: np.ndarray,
    contour_center: np.ndarray,
) -> np.ndarray:
    segment = end - start
    segment_length = float(np.linalg.norm(segment))
    if segment_length <= 0:
        raise ValueError("degenerate segment")

    direction = segment / segment_length
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    segment_mid = (start + end) / 2.0
    if float(np.dot(contour_center - segment_mid, normal)) > 0:
        normal = -normal

    return normal


def _edge_filter_points(
    points: np.ndarray,
    segment_mid: np.ndarray,
    normal: np.ndarray,
    line_edge_percentile: float,
) -> np.ndarray:
    percentile = max(0.0, min(100.0, float(line_edge_percentile)))
    if len(points) == 0 or percentile >= 100.0:
        return points

    offsets = (points - segment_mid) @ normal
    cutoff = np.percentile(offsets, 100.0 - percentile)
    return points[offsets >= cutoff]


def _end_trim_segment_points(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    trim_percent: float,
) -> np.ndarray:
    percent = max(0.0, min(49.0, float(trim_percent)))
    if len(points) == 0 or percent <= 0:
        return points

    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 0:
        return np.empty((0, 2), dtype=np.float64)

    t = ((points - start) @ segment) / length_sq
    start_t = percent / 100.0
    end_t = 1.0 - start_t
    return points[(t >= start_t) & (t <= end_t)]


def _segment_projection_t(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 0:
        return np.empty((0,), dtype=np.float64)

    return ((points - start) @ segment) / length_sq


def _find_line_core_points(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    window_mm: float,
    step_mm: float,
    max_window_mean_error_mm: float,
    max_window_angle_diff_deg: float,
    min_line_points: int,
) -> tuple[np.ndarray, bool, float | None, float | None, int, int, list[dict]]:
    segment = end - start
    segment_length = float(np.linalg.norm(segment))
    if (
        len(points) < min_line_points
        or segment_length <= 0
        or window_mm <= 0
        or step_mm <= 0
    ):
        return points, False, None, None, 0, 0, []

    t_values = _segment_projection_t(points, start, end)
    window_t = min(1.0, float(window_mm) / segment_length)
    step_t = min(1.0, float(step_mm) / segment_length)
    max_start_t = max(0.0, 1.0 - window_t)
    starts = list(np.arange(0.0, max_start_t + 1e-9, step_t))
    if not starts or starts[-1] < max_start_t:
        starts.append(max_start_t)

    windows: list[dict] = []
    for window_index, start_t in enumerate(starts, start=1):
        end_t = min(1.0, start_t + window_t)
        in_window = (t_values >= start_t) & (t_values <= end_t)
        window_points = points[in_window]
        diagnostic = {
            "id": window_index,
            "start_t": float(start_t),
            "end_t": float(end_t),
            "points_count": int(len(window_points)),
            "mean_error_mm": None,
            "angle_deg": None,
            "good": False,
            "core_selected": False,
        }

        if len(window_points) >= 2:
            try:
                line_point, direction = _fit_line(window_points)
                errors = _line_errors(window_points, line_point, direction)
                mean_error = float(errors.mean())
                angle = _line_angle_deg(direction)
                diagnostic["mean_error_mm"] = mean_error
                diagnostic["angle_deg"] = angle
                diagnostic["good"] = mean_error <= max_window_mean_error_mm
            except ValueError:
                diagnostic["good"] = False

        windows.append(diagnostic)

    best_start_index: int | None = None
    best_end_index: int | None = None
    current_start_index: int | None = None
    previous_good_angle: float | None = None

    for index, window in enumerate(windows):
        angle = window["angle_deg"]
        is_good_window = bool(window["good"]) and angle is not None
        if not is_good_window:
            if current_start_index is not None:
                if (
                    best_start_index is None
                    or index - current_start_index > best_end_index - best_start_index + 1
                ):
                    best_start_index = current_start_index
                    best_end_index = index - 1
                current_start_index = None
            previous_good_angle = None
            continue

        if (
            previous_good_angle is not None
            and _angle_diff_deg(float(previous_good_angle), float(angle))
            > max_window_angle_diff_deg
        ):
            if current_start_index is not None:
                if (
                    best_start_index is None
                    or index - current_start_index > best_end_index - best_start_index + 1
                ):
                    best_start_index = current_start_index
                    best_end_index = index - 1
            current_start_index = index

        if current_start_index is None:
            current_start_index = index
        previous_good_angle = float(angle)

    if current_start_index is not None:
        index = len(windows)
        if (
            best_start_index is None
            or index - current_start_index > best_end_index - best_start_index + 1
        ):
            best_start_index = current_start_index
            best_end_index = index - 1

    total_windows_count = len(windows)
    if best_start_index is None or best_end_index is None:
        return points, False, None, None, 0, total_windows_count, windows

    core_start_t = float(windows[best_start_index]["start_t"])
    core_end_t = float(windows[best_end_index]["end_t"])
    for index, window in enumerate(windows):
        window["core_selected"] = best_start_index <= index <= best_end_index
    core_mask = (t_values >= core_start_t) & (t_values <= core_end_t)
    core_points = points[core_mask]
    core_windows_count = best_end_index - best_start_index + 1

    if len(core_points) < min_line_points:
        return points, False, None, None, 0, total_windows_count, windows

    return (
        core_points,
        True,
        core_start_t,
        core_end_t,
        core_windows_count,
        total_windows_count,
        windows,
    )


def _offset_range(
    points: np.ndarray,
    segment_mid: np.ndarray,
    normal: np.ndarray,
) -> tuple[float | None, float | None]:
    if len(points) == 0:
        return None, None

    offsets = (points - segment_mid) @ normal
    return float(offsets.min()), float(offsets.max())


def _trim_line_outliers(
    points: np.ndarray,
    line_point: np.ndarray,
    direction: np.ndarray,
    trim_percent: float,
    min_line_points: int,
) -> np.ndarray:
    percent = max(0.0, min(100.0, float(trim_percent)))
    if len(points) <= min_line_points or percent <= 0:
        return points

    remove_count = int(np.floor(len(points) * percent / 100.0))
    max_remove_count = len(points) - min_line_points
    remove_count = min(remove_count, max_remove_count)
    if remove_count <= 0:
        return points

    errors = _line_errors(points, line_point, direction)
    keep_count = len(points) - remove_count
    keep_indices = np.argsort(errors)[:keep_count]
    return points[keep_indices]


def approximate_lines(
    contour_points: np.ndarray,
    boundary_points: np.ndarray,
    line_simplify_mm: float = 5.0,
    line_fit_band_mm: float = 3.0,
    min_line_points: int = 20,
    min_line_length_mm: float = 20.0,
    max_line_mean_error_mm: float = 1.0,
    max_line_error_mm: float = 3.0,
    line_edge_percentile: float = 20.0,
    line_trim_outlier_percent: float = 10.0,
    max_line_angle_diff_deg: float = 12.0,
    line_end_trim_percent: float = 10.0,
    line_window_mm: float = 30.0,
    line_window_step_mm: float = 10.0,
    max_window_mean_error_mm: float = 1.0,
    max_window_angle_diff_deg: float = 5.0,
    max_point_contour_distance_mm: float = 6.0,
    max_line_contour_distance_mm: float = 5.0,
    debug_line_id: int | None = None,
    debug_output_dir: str | Path | None = None,
    debug_base_name: str | None = None,
    endpoint_mode: str = "contour_projection",
) -> LineApproximationResult:
    if endpoint_mode not in {"contour_projection", "fit_points_range"}:
        raise ValueError(
            "endpoint_mode must be 'contour_projection' or 'fit_points_range'"
        )

    simplified = simplify_contour_points(contour_points, line_simplify_mm)
    if len(simplified) < 2:
        return LineApproximationResult(
            lines=[],
            rejected_segments=[],
            total_segments=0,
            rejected_by_reason={},
            chained_lines=[],
            chained_successful_intersections=0,
            chained_warnings_count=0,
            mixed_contour_elements=[],
        )

    lines: list[RefinedLine] = []
    rejected_segments: list[SegmentDiagnostic] = []
    rejected_by_reason: dict[str, int] = {}
    contour_center = _contour_center(contour_points)
    debug_output_path = Path(debug_output_dir) if debug_output_dir is not None else None
    debug_name = debug_base_name or "refined_lines"

    def save_debug_line(
        segment_id: int,
        start_point: np.ndarray,
        end_point: np.ndarray,
        length_mm: float,
        segment_angle_deg: float,
        normal: np.ndarray,
        normal_offset_min: float | None,
        normal_offset_max: float | None,
        selected_offset_min: float | None,
        selected_offset_max: float | None,
        contour_segment: np.ndarray,
        stages: dict[str, np.ndarray],
        accepted: bool,
        reject_reason: str,
        fitted_start: np.ndarray | None,
        fitted_end: np.ndarray | None,
    ) -> None:
        if (
            debug_line_id is None
            or segment_id != debug_line_id
            or debug_output_path is None
        ):
            return

        summary = {
            "segment_id": segment_id,
            "segment_start": {
                "x": float(start_point[0]),
                "y": float(start_point[1]),
            },
            "segment_end": {
                "x": float(end_point[0]),
                "y": float(end_point[1]),
            },
            "segment_length_mm": float(length_mm),
            "segment_angle_deg": segment_angle_deg,
            "normal_x": float(normal[0]),
            "normal_y": float(normal[1]),
            "normal_offset_min": normal_offset_min,
            "normal_offset_max": normal_offset_max,
            "selected_offset_min": selected_offset_min,
            "selected_offset_max": selected_offset_max,
            "points_count_by_stage": {
                name: int(len(points)) for name, points in stages.items()
            },
            "accepted": accepted,
            "reject_reason": reject_reason,
            "fitted_line_start": (
                {
                    "x": float(fitted_start[0]),
                    "y": float(fitted_start[1]),
                }
                if fitted_start is not None
                else None
            ),
            "fitted_line_end": (
                {
                    "x": float(fitted_end[0]),
                    "y": float(fitted_end[1]),
                }
                if fitted_end is not None
                else None
            ),
        }
        _save_debug_line_files(
            debug_output_path,
            debug_name,
            segment_id,
            stages,
            summary,
            start_point,
            end_point,
            contour_segment,
            (start_point + end_point) / 2.0,
            normal,
        )

    def append_rejected(
        segment_id: int,
        start_point: np.ndarray,
        end_point: np.ndarray,
        length_mm: float,
        points_count: int,
        mean_error_mm: float | None,
        max_error_mm: float | None,
        reject_reason: str,
        edge_filter_used: bool,
        points_before_edge_filter: int,
        points_after_edge_filter: int,
        points_after_trim: int,
        segment_angle_deg: float,
        fitted_angle_deg: float | None,
        angle_diff_deg: float | None,
        line_end_trim_percent_value: float,
        points_after_end_trim: int,
        normal: np.ndarray,
        normal_offset_min: float | None,
        normal_offset_max: float | None,
        selected_offset_min: float | None,
        selected_offset_max: float | None,
        core_found: bool,
        core_start_t: float | None,
        core_end_t: float | None,
        core_windows_count: int,
        total_windows_count: int,
        window_diagnostics: list[dict],
        mean_distance_to_contour_mm: float | None,
        max_distance_to_contour_mm: float | None,
        points_after_contour_distance_filter: int,
        endpoint_t_start: float | None = None,
        endpoint_t_end: float | None = None,
    ) -> None:
        rejected_by_reason[reject_reason] = rejected_by_reason.get(reject_reason, 0) + 1
        rejected_segments.append(
            SegmentDiagnostic(
                id=segment_id,
                start_x=float(start_point[0]),
                start_y=float(start_point[1]),
                end_x=float(end_point[0]),
                end_y=float(end_point[1]),
                length_mm=length_mm,
                points_count=points_count,
                mean_error_mm=mean_error_mm,
                max_error_mm=max_error_mm,
                accepted=False,
                reject_reason=reject_reason,
                edge_filter_used=edge_filter_used,
                points_before_edge_filter=points_before_edge_filter,
                points_after_edge_filter=points_after_edge_filter,
                trim_outlier_percent=float(line_trim_outlier_percent),
                points_after_trim=points_after_trim,
                segment_angle_deg=segment_angle_deg,
                fitted_angle_deg=fitted_angle_deg,
                angle_diff_deg=angle_diff_deg,
                line_end_trim_percent=float(line_end_trim_percent_value),
                points_after_end_trim=points_after_end_trim,
                normal_x=float(normal[0]),
                normal_y=float(normal[1]),
                normal_offset_min=normal_offset_min,
                normal_offset_max=normal_offset_max,
                selected_offset_min=selected_offset_min,
                selected_offset_max=selected_offset_max,
                core_found=core_found,
                core_start_t=core_start_t,
                core_end_t=core_end_t,
                core_windows_count=core_windows_count,
                total_windows_count=total_windows_count,
                window_diagnostics=window_diagnostics,
                mean_distance_to_contour_mm=mean_distance_to_contour_mm,
                max_distance_to_contour_mm=max_distance_to_contour_mm,
                max_point_contour_distance_mm=float(max_point_contour_distance_mm),
                points_after_contour_distance_filter=points_after_contour_distance_filter,
                endpoint_mode=endpoint_mode,
                endpoint_t_start=endpoint_t_start,
                endpoint_t_end=endpoint_t_end,
            )
        )

    for index in range(len(simplified)):
        segment_id = index + 1
        start = simplified[index]
        end = simplified[(index + 1) % len(simplified)]
        segment_length = float(np.linalg.norm(end - start))
        if segment_length <= 0:
            continue

        segment_direction = (end - start) / segment_length
        segment_angle = _line_angle_deg(segment_direction)
        segment_mid = (start + end) / 2.0
        normal = _outward_segment_normal(start, end, contour_center)
        contour_start_index, contour_end_index = _contour_segment_indices(
            contour_points,
            start,
            end,
        )
        contour_segment = _contour_points_between_indices(
            contour_points,
            contour_start_index,
            contour_end_index,
        )

        nearby_points = _points_near_segment(
            boundary_points,
            start,
            end,
            line_fit_band_mm,
        )
        debug_stages = {
            "all_candidate_points": nearby_points,
            "after_contour_distance_filter": np.empty((0, 2), dtype=np.float64),
            "after_end_trim": np.empty((0, 2), dtype=np.float64),
            "after_edge_filter": np.empty((0, 2), dtype=np.float64),
            "after_core_filter": np.empty((0, 2), dtype=np.float64),
            "after_robust_trim": np.empty((0, 2), dtype=np.float64),
        }
        contour_filtered_points = _filter_points_near_contour(
            nearby_points,
            contour_segment,
            max_point_contour_distance_mm,
        )
        debug_stages["after_contour_distance_filter"] = contour_filtered_points
        points_after_contour_distance_filter = int(len(contour_filtered_points))
        end_trimmed_points = _end_trim_segment_points(
            contour_filtered_points,
            start,
            end,
            line_end_trim_percent,
        )
        debug_stages["after_end_trim"] = end_trimmed_points
        points_count = int(len(end_trimmed_points))
        normal_offset_min, normal_offset_max = _offset_range(
            nearby_points,
            segment_mid,
            normal,
        )
        if points_count < min_line_points:
            save_debug_line(
                segment_id=segment_id,
                start_point=start,
                end_point=end,
                length_mm=segment_length,
                segment_angle_deg=segment_angle,
                normal=normal,
                normal_offset_min=normal_offset_min,
                normal_offset_max=normal_offset_max,
                selected_offset_min=None,
                selected_offset_max=None,
                contour_segment=contour_segment,
                stages=debug_stages,
                accepted=False,
                reject_reason="too_few_points",
                fitted_start=None,
                fitted_end=None,
            )
            append_rejected(
                segment_id=segment_id,
                start_point=start,
                end_point=end,
                length_mm=segment_length,
                points_count=points_count,
                mean_error_mm=None,
                max_error_mm=None,
                reject_reason="too_few_points",
                edge_filter_used=False,
                points_before_edge_filter=points_count,
                points_after_edge_filter=points_count,
                points_after_trim=points_count,
                segment_angle_deg=segment_angle,
                fitted_angle_deg=None,
                angle_diff_deg=None,
                line_end_trim_percent_value=line_end_trim_percent,
                points_after_end_trim=points_count,
                normal=normal,
                normal_offset_min=normal_offset_min,
                normal_offset_max=normal_offset_max,
                selected_offset_min=None,
                selected_offset_max=None,
                core_found=False,
                core_start_t=None,
                core_end_t=None,
                core_windows_count=0,
                total_windows_count=0,
                window_diagnostics=[],
                mean_distance_to_contour_mm=None,
                max_distance_to_contour_mm=None,
                points_after_contour_distance_filter=points_after_contour_distance_filter,
            )
            continue

        edge_points = _edge_filter_points(
            end_trimmed_points,
            segment_mid,
            normal,
            line_edge_percentile,
        )
        debug_stages["after_edge_filter"] = edge_points
        points_before_edge_filter = points_count
        points_after_edge_filter = int(len(edge_points))
        edge_filter_used = points_after_edge_filter >= min_line_points
        base_fit_points = edge_points if edge_filter_used else end_trimmed_points
        (
            fit_points,
            core_found,
            core_start_t,
            core_end_t,
            core_windows_count,
            total_windows_count,
            window_diagnostics,
        ) = _find_line_core_points(
            base_fit_points,
            start,
            end,
            line_window_mm,
            line_window_step_mm,
            max_window_mean_error_mm,
            max_window_angle_diff_deg,
            min_line_points,
        )
        debug_stages["after_core_filter"] = fit_points
        selected_offset_min, selected_offset_max = _offset_range(
            fit_points,
            segment_mid,
            normal,
        )

        try:
            line_point, direction = _fit_line(fit_points)
            trimmed_points = _trim_line_outliers(
                fit_points,
                line_point,
                direction,
                line_trim_outlier_percent,
                min_line_points,
            )
            if len(trimmed_points) != len(fit_points):
                line_point, direction = _fit_line(trimmed_points)
        except ValueError:
            save_debug_line(
                segment_id=segment_id,
                start_point=start,
                end_point=end,
                length_mm=segment_length,
                segment_angle_deg=segment_angle,
                normal=normal,
                normal_offset_min=normal_offset_min,
                normal_offset_max=normal_offset_max,
                selected_offset_min=selected_offset_min,
                selected_offset_max=selected_offset_max,
                contour_segment=contour_segment,
                stages=debug_stages,
                accepted=False,
                reject_reason="fit_failed",
                fitted_start=None,
                fitted_end=None,
            )
            append_rejected(
                segment_id=segment_id,
                start_point=start,
                end_point=end,
                length_mm=segment_length,
                points_count=points_count,
                mean_error_mm=None,
                max_error_mm=None,
                reject_reason="fit_failed",
                edge_filter_used=edge_filter_used,
                points_before_edge_filter=points_before_edge_filter,
                points_after_edge_filter=points_after_edge_filter,
                points_after_trim=int(len(fit_points)),
                segment_angle_deg=segment_angle,
                fitted_angle_deg=None,
                angle_diff_deg=None,
                line_end_trim_percent_value=line_end_trim_percent,
                points_after_end_trim=points_count,
                normal=normal,
                normal_offset_min=normal_offset_min,
                normal_offset_max=normal_offset_max,
                selected_offset_min=selected_offset_min,
                selected_offset_max=selected_offset_max,
                core_found=core_found,
                core_start_t=core_start_t,
                core_end_t=core_end_t,
                core_windows_count=core_windows_count,
                total_windows_count=total_windows_count,
                window_diagnostics=window_diagnostics,
                mean_distance_to_contour_mm=None,
                max_distance_to_contour_mm=None,
                points_after_contour_distance_filter=points_after_contour_distance_filter,
            )
            continue

        debug_stages["after_robust_trim"] = trimmed_points
        (
            refined_start,
            refined_end,
            endpoint_t_start,
            endpoint_t_end,
        ) = _line_endpoint_points(
            endpoint_mode,
            start,
            end,
            trimmed_points,
            line_point,
            direction,
        )
        refined_length = float(np.linalg.norm(refined_end - refined_start))

        points_after_trim = int(len(trimmed_points))
        errors = _line_errors(trimmed_points, line_point, direction)
        mean_error = float(errors.mean())
        max_error = float(errors.max())
        fitted_angle = _line_angle_deg(direction)
        angle_diff = _angle_diff_deg(segment_angle, fitted_angle)
        contour_line_errors = _line_errors(contour_segment, line_point, direction)
        mean_distance_to_contour = float(contour_line_errors.mean())
        max_distance_to_contour = float(contour_line_errors.max())

        reject_reason = ""
        if refined_length <= 1e-9:
            reject_reason = "invalid_endpoint_range"
        elif refined_length < min_line_length_mm:
            reject_reason = "too_short"
        elif angle_diff > max_line_angle_diff_deg:
            reject_reason = "angle_diff_too_high"
        elif max_distance_to_contour > max_line_contour_distance_mm:
            reject_reason = "too_far_from_mask_contour"
        elif mean_error > max_line_mean_error_mm:
            reject_reason = "mean_error_too_high"
        elif max_error > max_line_error_mm:
            reject_reason = "max_error_too_high"

        if reject_reason:
            save_debug_line(
                segment_id=segment_id,
                start_point=start,
                end_point=end,
                length_mm=segment_length,
                segment_angle_deg=segment_angle,
                normal=normal,
                normal_offset_min=normal_offset_min,
                normal_offset_max=normal_offset_max,
                selected_offset_min=selected_offset_min,
                selected_offset_max=selected_offset_max,
                contour_segment=contour_segment,
                stages=debug_stages,
                accepted=False,
                reject_reason=reject_reason,
                fitted_start=refined_start,
                fitted_end=refined_end,
            )
            append_rejected(
                segment_id=segment_id,
                start_point=refined_start,
                end_point=refined_end,
                length_mm=refined_length,
                points_count=points_count,
                mean_error_mm=mean_error,
                max_error_mm=max_error,
                reject_reason=reject_reason,
                edge_filter_used=edge_filter_used,
                points_before_edge_filter=points_before_edge_filter,
                points_after_edge_filter=points_after_edge_filter,
                points_after_trim=points_after_trim,
                segment_angle_deg=segment_angle,
                fitted_angle_deg=fitted_angle,
                angle_diff_deg=angle_diff,
                line_end_trim_percent_value=line_end_trim_percent,
                points_after_end_trim=points_count,
                normal=normal,
                normal_offset_min=normal_offset_min,
                normal_offset_max=normal_offset_max,
                selected_offset_min=selected_offset_min,
                selected_offset_max=selected_offset_max,
                core_found=core_found,
                core_start_t=core_start_t,
                core_end_t=core_end_t,
                core_windows_count=core_windows_count,
                total_windows_count=total_windows_count,
                window_diagnostics=window_diagnostics,
                mean_distance_to_contour_mm=mean_distance_to_contour,
                max_distance_to_contour_mm=max_distance_to_contour,
                points_after_contour_distance_filter=points_after_contour_distance_filter,
                endpoint_t_start=endpoint_t_start,
                endpoint_t_end=endpoint_t_end,
            )
            continue

        save_debug_line(
            segment_id=segment_id,
            start_point=start,
            end_point=end,
            length_mm=segment_length,
            segment_angle_deg=segment_angle,
            normal=normal,
            normal_offset_min=normal_offset_min,
            normal_offset_max=normal_offset_max,
            selected_offset_min=selected_offset_min,
            selected_offset_max=selected_offset_max,
            contour_segment=contour_segment,
            stages=debug_stages,
            accepted=True,
            reject_reason="",
            fitted_start=refined_start,
            fitted_end=refined_end,
        )
        lines.append(
            RefinedLine(
                id=segment_id,
                start_x=float(refined_start[0]),
                start_y=float(refined_start[1]),
                end_x=float(refined_end[0]),
                end_y=float(refined_end[1]),
                length_mm=refined_length,
                points_count=points_count,
                mean_error_mm=mean_error,
                max_error_mm=max_error,
                edge_filter_used=edge_filter_used,
                points_before_edge_filter=points_before_edge_filter,
                points_after_edge_filter=points_after_edge_filter,
                trim_outlier_percent=float(line_trim_outlier_percent),
                points_after_trim=points_after_trim,
                segment_angle_deg=segment_angle,
                fitted_angle_deg=fitted_angle,
                angle_diff_deg=angle_diff,
                line_end_trim_percent=float(line_end_trim_percent),
                points_after_end_trim=points_count,
                normal_x=float(normal[0]),
                normal_y=float(normal[1]),
                normal_offset_min=normal_offset_min,
                normal_offset_max=normal_offset_max,
                selected_offset_min=selected_offset_min,
                selected_offset_max=selected_offset_max,
                core_found=core_found,
                core_start_t=core_start_t,
                core_end_t=core_end_t,
                core_windows_count=core_windows_count,
                total_windows_count=total_windows_count,
                window_diagnostics=window_diagnostics,
                mean_distance_to_contour_mm=mean_distance_to_contour,
                max_distance_to_contour_mm=max_distance_to_contour,
                max_point_contour_distance_mm=float(max_point_contour_distance_mm),
                points_after_contour_distance_filter=points_after_contour_distance_filter,
                contour_start_index=contour_start_index,
                contour_end_index=contour_end_index,
                endpoint_mode=endpoint_mode,
                endpoint_t_start=endpoint_t_start,
                endpoint_t_end=endpoint_t_end,
            )
        )

    return LineApproximationResult(
        lines=lines,
        rejected_segments=rejected_segments,
        total_segments=len(simplified),
        rejected_by_reason=rejected_by_reason,
        chained_lines=[],
        chained_successful_intersections=0,
        chained_warnings_count=0,
        mixed_contour_elements=[],
    )


def _line_to_json(line: RefinedLine) -> dict:
    return {
        "id": line.id,
        "start_x": line.start_x,
        "start_y": line.start_y,
        "end_x": line.end_x,
        "end_y": line.end_y,
        "length_mm": line.length_mm,
        "points_count": line.points_count,
        "mean_error_mm": line.mean_error_mm,
        "max_error_mm": line.max_error_mm,
        "edge_filter_used": line.edge_filter_used,
        "points_before_edge_filter": line.points_before_edge_filter,
        "points_after_edge_filter": line.points_after_edge_filter,
        "trim_outlier_percent": line.trim_outlier_percent,
        "points_after_trim": line.points_after_trim,
        "segment_angle_deg": line.segment_angle_deg,
        "fitted_angle_deg": line.fitted_angle_deg,
        "angle_diff_deg": line.angle_diff_deg,
        "line_end_trim_percent": line.line_end_trim_percent,
        "points_after_end_trim": line.points_after_end_trim,
        "normal_x": line.normal_x,
        "normal_y": line.normal_y,
        "normal_offset_min": line.normal_offset_min,
        "normal_offset_max": line.normal_offset_max,
        "selected_offset_min": line.selected_offset_min,
        "selected_offset_max": line.selected_offset_max,
        "core_found": line.core_found,
        "core_start_t": line.core_start_t,
        "core_end_t": line.core_end_t,
        "core_windows_count": line.core_windows_count,
        "total_windows_count": line.total_windows_count,
        "window_diagnostics": line.window_diagnostics,
        "mean_distance_to_contour_mm": line.mean_distance_to_contour_mm,
        "max_distance_to_contour_mm": line.max_distance_to_contour_mm,
        "max_point_contour_distance_mm": line.max_point_contour_distance_mm,
        "points_after_contour_distance_filter": (
            line.points_after_contour_distance_filter
        ),
        "contour_start_index": line.contour_start_index,
        "contour_end_index": line.contour_end_index,
        "endpoint_mode": line.endpoint_mode,
        "endpoint_t_start": line.endpoint_t_start,
        "endpoint_t_end": line.endpoint_t_end,
    }


def _segment_to_json(segment: SegmentDiagnostic) -> dict:
    payload = {
        "id": segment.id,
        "start_x": segment.start_x,
        "start_y": segment.start_y,
        "end_x": segment.end_x,
        "end_y": segment.end_y,
        "length_mm": segment.length_mm,
        "points_count": segment.points_count,
        "accepted": segment.accepted,
        "reject_reason": segment.reject_reason,
        "edge_filter_used": segment.edge_filter_used,
        "points_before_edge_filter": segment.points_before_edge_filter,
        "points_after_edge_filter": segment.points_after_edge_filter,
        "trim_outlier_percent": segment.trim_outlier_percent,
        "points_after_trim": segment.points_after_trim,
        "segment_angle_deg": segment.segment_angle_deg,
        "line_end_trim_percent": segment.line_end_trim_percent,
        "points_after_end_trim": segment.points_after_end_trim,
        "normal_x": segment.normal_x,
        "normal_y": segment.normal_y,
        "normal_offset_min": segment.normal_offset_min,
        "normal_offset_max": segment.normal_offset_max,
        "selected_offset_min": segment.selected_offset_min,
        "selected_offset_max": segment.selected_offset_max,
        "core_found": segment.core_found,
        "core_start_t": segment.core_start_t,
        "core_end_t": segment.core_end_t,
        "core_windows_count": segment.core_windows_count,
        "total_windows_count": segment.total_windows_count,
        "window_diagnostics": segment.window_diagnostics,
        "max_point_contour_distance_mm": segment.max_point_contour_distance_mm,
        "points_after_contour_distance_filter": (
            segment.points_after_contour_distance_filter
        ),
        "endpoint_mode": segment.endpoint_mode,
        "endpoint_t_start": segment.endpoint_t_start,
        "endpoint_t_end": segment.endpoint_t_end,
    }
    if segment.mean_error_mm is not None:
        payload["mean_error_mm"] = segment.mean_error_mm
    if segment.max_error_mm is not None:
        payload["max_error_mm"] = segment.max_error_mm
    if segment.fitted_angle_deg is not None:
        payload["fitted_angle_deg"] = segment.fitted_angle_deg
    if segment.angle_diff_deg is not None:
        payload["angle_diff_deg"] = segment.angle_diff_deg
    if segment.mean_distance_to_contour_mm is not None:
        payload["mean_distance_to_contour_mm"] = segment.mean_distance_to_contour_mm
    if segment.max_distance_to_contour_mm is not None:
        payload["max_distance_to_contour_mm"] = segment.max_distance_to_contour_mm

    return payload


def save_refined_lines_json(
    result: LineApproximationResult,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "lines": [_line_to_json(line) for line in result.lines],
        "rejected_segments": [
            _segment_to_json(segment) for segment in result.rejected_segments
        ],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _refined_line_points(line: RefinedLine) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([line.start_x, line.start_y], dtype=np.float64),
        np.array([line.end_x, line.end_y], dtype=np.float64),
    )


def _line_intersection(
    line_a: RefinedLine,
    line_b: RefinedLine,
    parallel_angle_eps_deg: float,
) -> tuple[np.ndarray | None, str | None]:
    start_a, end_a = _refined_line_points(line_a)
    start_b, end_b = _refined_line_points(line_b)
    direction_a = end_a - start_a
    direction_b = end_b - start_b
    length_a = float(np.linalg.norm(direction_a))
    length_b = float(np.linalg.norm(direction_b))
    if length_a <= 0 or length_b <= 0:
        return None, "degenerate_line"

    angle_diff = _angle_diff_deg(
        _line_angle_deg(direction_a / length_a),
        _line_angle_deg(direction_b / length_b),
    )
    if angle_diff <= parallel_angle_eps_deg:
        return None, "parallel_lines"

    matrix = np.column_stack((direction_a, -direction_b))
    rhs = start_b - start_a
    try:
        parameters = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None, "intersection_failed"

    return start_a + parameters[0] * direction_a, None


def _intersection_is_close_to_lines(
    point: np.ndarray,
    line_a: RefinedLine,
    line_b: RefinedLine,
    max_distance_mm: float,
) -> bool:
    start_a, end_a = _refined_line_points(line_a)
    start_b, end_b = _refined_line_points(line_b)
    point_array = point.reshape((1, 2))
    distance_a = float(_point_distances_to_segment(point_array, start_a, end_a)[0])
    distance_b = float(_point_distances_to_segment(point_array, start_b, end_b)[0])
    return distance_a <= max_distance_mm and distance_b <= max_distance_mm


def build_chained_lines(
    lines: list[RefinedLine],
    max_chain_intersection_distance_mm: float = 150.0,
    parallel_angle_eps_deg: float = 3.0,
) -> tuple[list[ChainedLine], int, int]:
    ordered_lines = sorted(lines, key=lambda line: line.id)
    if not ordered_lines:
        return [], 0, 0

    intersections: list[np.ndarray | None] = []
    intersection_ok: list[bool] = []
    intersection_warnings: list[list[str]] = []
    total = len(ordered_lines)

    for index in range(total):
        current_line = ordered_lines[index]
        next_line = ordered_lines[(index + 1) % total]
        point, warning = _line_intersection(
            current_line,
            next_line,
            parallel_angle_eps_deg,
        )
        warnings: list[str] = []
        ok = point is not None
        if warning is not None:
            warnings.append(warning)
        if ok and not _intersection_is_close_to_lines(
            point,
            current_line,
            next_line,
            max_chain_intersection_distance_mm,
        ):
            ok = False
            warnings.append("intersection_too_far")

        intersections.append(point if ok else None)
        intersection_ok.append(ok)
        intersection_warnings.append(warnings)

    chained_lines: list[ChainedLine] = []
    for index, line in enumerate(ordered_lines):
        original_start, original_end = _refined_line_points(line)
        previous_index = (index - 1) % total
        next_index = index
        start_point = intersections[previous_index]
        end_point = intersections[next_index]
        warnings = []
        warnings.extend(
            f"prev_{warning}" for warning in intersection_warnings[previous_index]
        )
        warnings.extend(f"next_{warning}" for warning in intersection_warnings[next_index])

        intersection_prev_ok = start_point is not None
        intersection_next_ok = end_point is not None
        if start_point is None:
            start_point = original_start
        if end_point is None:
            end_point = original_end

        length_mm = float(np.linalg.norm(end_point - start_point))
        chained_lines.append(
            ChainedLine(
                original_line_id=line.id,
                start_x=float(start_point[0]),
                start_y=float(start_point[1]),
                end_x=float(end_point[0]),
                end_y=float(end_point[1]),
                length_mm=length_mm,
                intersection_prev_ok=intersection_prev_ok,
                intersection_next_ok=intersection_next_ok,
                warnings=warnings,
            )
        )

    successful_intersections = sum(1 for ok in intersection_ok if ok)
    warnings_count = sum(len(line.warnings) for line in chained_lines)
    return chained_lines, successful_intersections, warnings_count


def save_chained_lines_json(
    chained_lines: list[ChainedLine],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lines": [
            {
                "original_line_id": line.original_line_id,
                "start_x": line.start_x,
                "start_y": line.start_y,
                "end_x": line.end_x,
                "end_y": line.end_y,
                "length_mm": line.length_mm,
                "intersection_prev_ok": line.intersection_prev_ok,
                "intersection_next_ok": line.intersection_next_ok,
                "warnings": line.warnings,
            }
            for line in chained_lines
        ]
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_chained_lines_dxf(
    chained_lines: list[ChainedLine],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    doc.layers.add(name="REFINED_CHAINED_LINES", color=3)

    msp = doc.modelspace()
    for line in chained_lines:
        msp.add_line(
            (line.start_x, line.start_y),
            (line.end_x, line.end_y),
            dxfattribs={"layer": "REFINED_CHAINED_LINES"},
        )

    doc.saveas(output_path)


def _contour_gap_indices(
    contour_length: int,
    start_index: int,
    end_index: int,
) -> list[int]:
    if contour_length <= 0:
        return []

    indices = [start_index % contour_length]
    current = start_index % contour_length
    target = end_index % contour_length
    while current != target:
        current = (current + 1) % contour_length
        indices.append(current)
        if len(indices) > contour_length + 1:
            break

    return indices


def build_mixed_contour(
    lines: list[RefinedLine],
    contour_points: np.ndarray,
) -> list[dict]:
    ordered_lines = sorted(lines, key=lambda line: line.id)
    if not ordered_lines:
        return []

    contour_length = len(contour_points)
    elements: list[dict] = []
    element_id = 1
    for index, line in enumerate(ordered_lines):
        elements.append(
            {
                "id": element_id,
                "type": "LINE",
                "source": "accepted_refined_line",
                "original_line_id": line.id,
                "start": {"x": line.start_x, "y": line.start_y},
                "end": {"x": line.end_x, "y": line.end_y},
            }
        )
        element_id += 1

        next_line = ordered_lines[(index + 1) % len(ordered_lines)]
        gap_indices = _contour_gap_indices(
            contour_length,
            line.contour_end_index,
            next_line.contour_start_index,
        )
        if len(gap_indices) >= 2:
            gap_points = contour_points[gap_indices]
            elements.append(
                {
                    "id": element_id,
                    "type": "POLYLINE",
                    "source": "contour_gap",
                    "contour_start_index": int(gap_indices[0]),
                    "contour_end_index": int(gap_indices[-1]),
                    "points": [
                        {"x": float(point[0]), "y": float(point[1])}
                        for point in gap_points
                    ],
                }
            )
            element_id += 1

    return elements


def save_mixed_contour_json(
    elements: list[dict],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"elements": elements}
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_mixed_contour_dxf(
    elements: list[dict],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    doc.layers.add(name="MIXED_LINES", color=5)
    doc.layers.add(name="MIXED_POLYLINE_GAPS", color=1)

    msp = doc.modelspace()
    for element in elements:
        if element["type"] == "LINE":
            start = element["start"]
            end = element["end"]
            msp.add_line(
                (start["x"], start["y"]),
                (end["x"], end["y"]),
                dxfattribs={"layer": "MIXED_LINES"},
            )
        elif element["type"] == "POLYLINE":
            points = [(point["x"], point["y"]) for point in element["points"]]
            if len(points) >= 2:
                msp.add_lwpolyline(
                    points,
                    dxfattribs={"layer": "MIXED_POLYLINE_GAPS"},
                )

    doc.saveas(output_path)


def save_mixed_contour_with_holes_dxf(
    elements: list[dict],
    output_path: str | Path,
    holes: list[dict] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    doc.layers.add(name="MIXED_LINES", color=5)
    doc.layers.add(name="MIXED_POLYLINE_GAPS", color=1)
    doc.layers.add(name="HOLES", color=3)

    msp = doc.modelspace()
    for element in elements:
        if element["type"] == "LINE":
            start = element["start"]
            end = element["end"]
            msp.add_line(
                (start["x"], start["y"]),
                (end["x"], end["y"]),
                dxfattribs={"layer": "MIXED_LINES"},
            )
        elif element["type"] == "POLYLINE":
            points = [(point["x"], point["y"]) for point in element["points"]]
            if len(points) >= 2:
                msp.add_lwpolyline(
                    points,
                    dxfattribs={"layer": "MIXED_POLYLINE_GAPS"},
                )

    for hole in holes or []:
        msp.add_circle(
            center=(float(hole["center_x"]), float(hole["center_y"])),
            radius=float(hole["radius"]),
            dxfattribs={"layer": "HOLES"},
        )

    doc.saveas(output_path)


def save_refined_lines_dxf(
    lines: list[RefinedLine],
    output_path: str | Path,
    show_line_labels: bool = False,
    line_label_height_mm: float = 10.0,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    doc.layers.add(name="REFINED_LINES", color=5)
    if show_line_labels:
        doc.layers.add(name="REFINED_LINE_LABELS", color=2)

    msp = doc.modelspace()
    for line in lines:
        msp.add_line(
            (line.start_x, line.start_y),
            (line.end_x, line.end_y),
            dxfattribs={"layer": "REFINED_LINES"},
        )
        if show_line_labels:
            mid_x = (line.start_x + line.end_x) / 2.0
            mid_y = (line.start_y + line.end_y) / 2.0
            msp.add_text(
                f"L{line.id}",
                dxfattribs={
                    "layer": "REFINED_LINE_LABELS",
                    "height": float(line_label_height_mm),
                },
            ).set_placement((mid_x, mid_y))

    doc.saveas(output_path)


def run_line_approximation(
    contour_csv_path: str | Path,
    boundary_points_path: str | Path,
    output_json_path: str | Path,
    output_dxf_path: str | Path,
    line_simplify_mm: float = 5.0,
    line_fit_band_mm: float = 3.0,
    min_line_points: int = 20,
    min_line_length_mm: float = 20.0,
    max_line_mean_error_mm: float = 1.0,
    max_line_error_mm: float = 3.0,
    line_edge_percentile: float = 20.0,
    line_trim_outlier_percent: float = 10.0,
    max_line_angle_diff_deg: float = 12.0,
    line_end_trim_percent: float = 10.0,
    line_window_mm: float = 30.0,
    line_window_step_mm: float = 10.0,
    max_window_mean_error_mm: float = 1.0,
    max_window_angle_diff_deg: float = 5.0,
    max_point_contour_distance_mm: float = 6.0,
    max_line_contour_distance_mm: float = 5.0,
    debug_line_id: int | None = None,
    debug_output_dir: str | Path | None = None,
    debug_base_name: str | None = None,
    show_line_labels: bool = False,
    line_label_height_mm: float = 10.0,
    chain_lines: bool = False,
    chained_json_path: str | Path | None = None,
    chained_dxf_path: str | Path | None = None,
    max_chain_intersection_distance_mm: float = 150.0,
    parallel_angle_eps_deg: float = 3.0,
    mixed_contour: bool = False,
    mixed_contour_json_path: str | Path | None = None,
    mixed_contour_dxf_path: str | Path | None = None,
    mixed_with_holes_dxf_path: str | Path | None = None,
    mixed_dxf_holes: list[dict] | None = None,
    endpoint_mode: str = "contour_projection",
) -> LineApproximationResult:
    contour_points = load_contour_csv(contour_csv_path)
    boundary_points = load_boundary_points(boundary_points_path)
    result = approximate_lines(
        contour_points=contour_points,
        boundary_points=boundary_points,
        line_simplify_mm=line_simplify_mm,
        line_fit_band_mm=line_fit_band_mm,
        min_line_points=min_line_points,
        min_line_length_mm=min_line_length_mm,
        max_line_mean_error_mm=max_line_mean_error_mm,
        max_line_error_mm=max_line_error_mm,
        line_edge_percentile=line_edge_percentile,
        line_trim_outlier_percent=line_trim_outlier_percent,
        max_line_angle_diff_deg=max_line_angle_diff_deg,
        line_end_trim_percent=line_end_trim_percent,
        line_window_mm=line_window_mm,
        line_window_step_mm=line_window_step_mm,
        max_window_mean_error_mm=max_window_mean_error_mm,
        max_window_angle_diff_deg=max_window_angle_diff_deg,
        max_point_contour_distance_mm=max_point_contour_distance_mm,
        max_line_contour_distance_mm=max_line_contour_distance_mm,
        debug_line_id=debug_line_id,
        debug_output_dir=debug_output_dir,
        debug_base_name=debug_base_name,
        endpoint_mode=endpoint_mode,
    )
    save_refined_lines_json(result, output_json_path)
    save_refined_lines_dxf(
        result.lines,
        output_dxf_path,
        show_line_labels=show_line_labels,
        line_label_height_mm=line_label_height_mm,
    )
    if chain_lines:
        chained_lines, successful_intersections, warnings_count = build_chained_lines(
            result.lines,
            max_chain_intersection_distance_mm=max_chain_intersection_distance_mm,
            parallel_angle_eps_deg=parallel_angle_eps_deg,
        )
        result.chained_lines = chained_lines
        result.chained_successful_intersections = successful_intersections
        result.chained_warnings_count = warnings_count
        if chained_json_path is not None:
            save_chained_lines_json(chained_lines, chained_json_path)
        if chained_dxf_path is not None:
            save_chained_lines_dxf(chained_lines, chained_dxf_path)
    if mixed_contour:
        mixed_elements = build_mixed_contour(result.lines, contour_points)
        result.mixed_contour_elements = mixed_elements
        if mixed_contour_json_path is not None:
            save_mixed_contour_json(mixed_elements, mixed_contour_json_path)
        if mixed_contour_dxf_path is not None:
            save_mixed_contour_dxf(mixed_elements, mixed_contour_dxf_path)
        if mixed_with_holes_dxf_path is not None:
            save_mixed_contour_with_holes_dxf(
                mixed_elements,
                mixed_with_holes_dxf_path,
                holes=mixed_dxf_holes,
            )

    return result
