"""Compare mask and ContourPy extraction for synthetic rectangles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import contourpy
import numpy as np


# Running ``python -m experiments.contourpy_density_prototype`` from the app
# directory puts app itself, rather than its parent, on sys.path.
_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))

from app.core.contour_extractor import (  # noqa: E402
    contour_pixels_to_world,
    extract_external_contour,
)
from app.core.density_grid import DensityGrid  # noqa: E402


@dataclass(frozen=True)
class SyntheticRectangleField:
    density: np.ndarray
    x_coordinates_mm: np.ndarray
    y_coordinates_mm: np.ndarray

    min_x_mm: float
    min_y_mm: float
    cell_size_mm: float
    threshold: float

    rectangle_min_x_mm: float
    rectangle_max_x_mm: float
    rectangle_min_y_mm: float
    rectangle_max_y_mm: float


@dataclass(frozen=True)
class SyntheticRotatedRectangleField:
    density: np.ndarray
    x_coordinates_mm: np.ndarray
    y_coordinates_mm: np.ndarray

    min_x_mm: float
    min_y_mm: float
    cell_size_mm: float
    threshold: float

    center_x_mm: float
    center_y_mm: float
    rectangle_width_mm: float
    rectangle_height_mm: float
    angle_degrees: float


@dataclass(frozen=True)
class SyntheticPointDensityRotatedRectangleField:
    density: np.ndarray
    points_xy_mm: np.ndarray

    x_coordinates_mm: np.ndarray
    y_coordinates_mm: np.ndarray

    min_x_mm: float
    min_y_mm: float
    cell_size_mm: float
    threshold: float

    center_x_mm: float
    center_y_mm: float
    rectangle_width_mm: float
    rectangle_height_mm: float
    angle_degrees: float

    point_spacing_mm: float
    point_offset_x_mm: float
    point_offset_y_mm: float


RotatedRectangleField = (
    SyntheticRotatedRectangleField
    | SyntheticPointDensityRotatedRectangleField
)
SyntheticField = SyntheticRectangleField | RotatedRectangleField


@dataclass(frozen=True)
class RectangleContourMetrics:
    point_count: int
    signed_area_mm2: float

    min_x_mm: float
    max_x_mm: float
    min_y_mm: float
    max_y_mm: float

    bounding_box_error_mm: float
    area_error_mm2: float
    rms_boundary_error_mm: float
    max_boundary_error_mm: float


@dataclass(frozen=True)
class RectangleContourComparison:
    field: SyntheticRectangleField

    mask_contour: np.ndarray
    contourpy_contour: np.ndarray

    mask_metrics: RectangleContourMetrics
    contourpy_metrics: RectangleContourMetrics


@dataclass(frozen=True)
class RotatedRectangleContourMetrics:
    point_count: int
    signed_area_mm2: float
    area_error_mm2: float

    rms_boundary_error_mm: float
    max_boundary_error_mm: float

    mean_true_side_rms_mm: float
    max_true_side_rms_mm: float

    mean_fitted_side_rms_mm: float
    max_fitted_side_rms_mm: float

    mean_side_angle_error_deg: float
    max_side_angle_error_deg: float


@dataclass(frozen=True)
class RotatedRectangleContourComparison:
    field: SyntheticRotatedRectangleField

    mask_contour: np.ndarray
    contourpy_contour: np.ndarray

    mask_metrics: RotatedRectangleContourMetrics
    contourpy_metrics: RotatedRectangleContourMetrics


@dataclass(frozen=True)
class PointDensityRotatedRectangleComparison:
    field: SyntheticPointDensityRotatedRectangleField

    mask_contour: np.ndarray
    contourpy_contour: np.ndarray

    mask_metrics: RotatedRectangleContourMetrics
    contourpy_metrics: RotatedRectangleContourMetrics


def _finite_float(value: float, name: str) -> float:
    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be finite")
    return value_float


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value_int = int(value)
    if value_int <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value_int


def accumulate_points_to_density(
    points_xy_mm: np.ndarray,
    *,
    min_x_mm: float,
    min_y_mm: float,
    width_cells: int,
    height_cells: int,
    cell_size_mm: float,
) -> np.ndarray:
    """Accumulate finite XY points into project-compatible uint32 cells."""
    points = np.asarray(points_xy_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy_mm must have shape (N, 2)")
    if not np.isfinite(points).all():
        raise ValueError("point coordinates must be finite")

    width_cells = _positive_integer(width_cells, "width_cells")
    height_cells = _positive_integer(height_cells, "height_cells")
    min_x_mm = _finite_float(min_x_mm, "min_x_mm")
    min_y_mm = _finite_float(min_y_mm, "min_y_mm")
    cell_size_mm = _finite_float(cell_size_mm, "cell_size_mm")
    if cell_size_mm <= 0.0:
        raise ValueError("cell_size_mm must be greater than zero")

    field_max_x_mm = min_x_mm + width_cells * cell_size_mm
    field_max_y_mm = min_y_mm + height_cells * cell_size_mm
    outside_grid = (
        (points[:, 0] < min_x_mm)
        | (points[:, 0] >= field_max_x_mm)
        | (points[:, 1] < min_y_mm)
        | (points[:, 1] >= field_max_y_mm)
    )
    if np.any(outside_grid):
        raise ValueError("all points must lie inside the density grid")

    ix = np.floor(
        (points[:, 0] - min_x_mm) / cell_size_mm
    ).astype(np.int64)
    iy = np.floor(
        (points[:, 1] - min_y_mm) / cell_size_mm
    ).astype(np.int64)

    if len(points) > 0:
        linear_indices = iy * width_cells + ix
        _cells, cell_counts = np.unique(
            linear_indices, return_counts=True
        )
        if int(np.max(cell_counts)) > np.iinfo(np.uint32).max:
            raise OverflowError("a density cell would overflow uint32")

    density = np.zeros(
        (height_cells, width_cells),
        dtype=np.uint32,
    )
    np.add.at(density, (iy, ix), np.uint32(1))
    return density


def create_synthetic_rectangle_field(
    *,
    min_x_mm: float = 0.0,
    min_y_mm: float = 0.0,
    width_cells: int = 80,
    height_cells: int = 60,
    cell_size_mm: float = 1.0,
    rectangle_min_x_mm: float = 13.25,
    rectangle_max_x_mm: float = 56.75,
    rectangle_min_y_mm: float = 11.4,
    rectangle_max_y_mm: float = 43.6,
    outside_density: float = 0.0,
    inside_density: float = 10.0,
    threshold: float = 5.0,
    transition_half_width_mm: float = 2.0,
) -> SyntheticRectangleField:
    """Create a continuous cell-centred density field for one rectangle."""
    width_cells = _positive_integer(width_cells, "width_cells")
    height_cells = _positive_integer(height_cells, "height_cells")

    min_x_mm = _finite_float(min_x_mm, "min_x_mm")
    min_y_mm = _finite_float(min_y_mm, "min_y_mm")
    cell_size_mm = _finite_float(cell_size_mm, "cell_size_mm")
    rectangle_min_x_mm = _finite_float(
        rectangle_min_x_mm, "rectangle_min_x_mm"
    )
    rectangle_max_x_mm = _finite_float(
        rectangle_max_x_mm, "rectangle_max_x_mm"
    )
    rectangle_min_y_mm = _finite_float(
        rectangle_min_y_mm, "rectangle_min_y_mm"
    )
    rectangle_max_y_mm = _finite_float(
        rectangle_max_y_mm, "rectangle_max_y_mm"
    )
    outside_density = _finite_float(outside_density, "outside_density")
    inside_density = _finite_float(inside_density, "inside_density")
    threshold = _finite_float(threshold, "threshold")
    transition_half_width_mm = _finite_float(
        transition_half_width_mm, "transition_half_width_mm"
    )

    if cell_size_mm <= 0.0:
        raise ValueError("cell_size_mm must be greater than zero")
    if transition_half_width_mm <= 0.0:
        raise ValueError("transition_half_width_mm must be greater than zero")
    if rectangle_min_x_mm >= rectangle_max_x_mm:
        raise ValueError("rectangle X bounds must be ordered")
    if rectangle_min_y_mm >= rectangle_max_y_mm:
        raise ValueError("rectangle Y bounds must be ordered")
    if not outside_density < threshold < inside_density:
        raise ValueError(
            "threshold must be strictly between outside and inside density"
        )

    field_max_x_mm = min_x_mm + width_cells * cell_size_mm
    field_max_y_mm = min_y_mm + height_cells * cell_size_mm
    if not (
        min_x_mm < rectangle_min_x_mm < rectangle_max_x_mm < field_max_x_mm
        and min_y_mm
        < rectangle_min_y_mm
        < rectangle_max_y_mm
        < field_max_y_mm
    ):
        raise ValueError("rectangle must lie strictly inside the field")

    x_coordinates_mm = min_x_mm + (
        np.arange(width_cells, dtype=np.float64) + 0.5
    ) * cell_size_mm
    y_coordinates_mm = min_y_mm + (
        np.arange(height_cells, dtype=np.float64) + 0.5
    ) * cell_size_mm
    x_grid, y_grid = np.meshgrid(x_coordinates_mm, y_coordinates_mm)

    rectangle_center_x = (
        rectangle_min_x_mm + rectangle_max_x_mm
    ) / 2.0
    rectangle_center_y = (
        rectangle_min_y_mm + rectangle_max_y_mm
    ) / 2.0
    rectangle_half_width = (
        rectangle_max_x_mm - rectangle_min_x_mm
    ) / 2.0
    rectangle_half_height = (
        rectangle_max_y_mm - rectangle_min_y_mm
    ) / 2.0

    distance_from_x_slab = (
        np.abs(x_grid - rectangle_center_x) - rectangle_half_width
    )
    distance_from_y_slab = (
        np.abs(y_grid - rectangle_center_y) - rectangle_half_height
    )

    outside_distance = np.hypot(
        np.maximum(distance_from_x_slab, 0.0),
        np.maximum(distance_from_y_slab, 0.0),
    )
    inside_distance = np.minimum(
        np.maximum(distance_from_x_slab, distance_from_y_slab),
        0.0,
    )
    signed_distance = outside_distance + inside_distance

    slope = (
        (inside_density - outside_density)
        / (2.0 * transition_half_width_mm)
    )
    density = threshold - signed_distance * slope
    density = np.clip(
        density, outside_density, inside_density
    ).astype(np.float64, copy=False)

    return SyntheticRectangleField(
        density=density,
        x_coordinates_mm=x_coordinates_mm,
        y_coordinates_mm=y_coordinates_mm,
        min_x_mm=min_x_mm,
        min_y_mm=min_y_mm,
        cell_size_mm=cell_size_mm,
        threshold=threshold,
        rectangle_min_x_mm=rectangle_min_x_mm,
        rectangle_max_x_mm=rectangle_max_x_mm,
        rectangle_min_y_mm=rectangle_min_y_mm,
        rectangle_max_y_mm=rectangle_max_y_mm,
    )


def create_synthetic_rotated_rectangle_field(
    *,
    min_x_mm: float = 0.0,
    min_y_mm: float = 0.0,
    width_cells: int = 100,
    height_cells: int = 80,
    cell_size_mm: float = 1.0,
    center_x_mm: float = 50.3,
    center_y_mm: float = 40.7,
    rectangle_width_mm: float = 46.2,
    rectangle_height_mm: float = 28.6,
    angle_degrees: float = 23.0,
    outside_density: float = 0.0,
    inside_density: float = 10.0,
    threshold: float = 5.0,
    transition_half_width_mm: float = 2.0,
) -> SyntheticRotatedRectangleField:
    """Create a cell-centred density field for one rotated rectangle."""
    width_cells = _positive_integer(width_cells, "width_cells")
    height_cells = _positive_integer(height_cells, "height_cells")

    min_x_mm = _finite_float(min_x_mm, "min_x_mm")
    min_y_mm = _finite_float(min_y_mm, "min_y_mm")
    cell_size_mm = _finite_float(cell_size_mm, "cell_size_mm")
    center_x_mm = _finite_float(center_x_mm, "center_x_mm")
    center_y_mm = _finite_float(center_y_mm, "center_y_mm")
    rectangle_width_mm = _finite_float(
        rectangle_width_mm, "rectangle_width_mm"
    )
    rectangle_height_mm = _finite_float(
        rectangle_height_mm, "rectangle_height_mm"
    )
    angle_degrees = _finite_float(angle_degrees, "angle_degrees")
    outside_density = _finite_float(outside_density, "outside_density")
    inside_density = _finite_float(inside_density, "inside_density")
    threshold = _finite_float(threshold, "threshold")
    transition_half_width_mm = _finite_float(
        transition_half_width_mm, "transition_half_width_mm"
    )

    if cell_size_mm <= 0.0:
        raise ValueError("cell_size_mm must be greater than zero")
    if rectangle_width_mm <= 0.0 or rectangle_height_mm <= 0.0:
        raise ValueError("rectangle dimensions must be greater than zero")
    if transition_half_width_mm <= 0.0:
        raise ValueError("transition_half_width_mm must be greater than zero")
    if not outside_density < threshold < inside_density:
        raise ValueError(
            "threshold must be strictly between outside and inside density"
        )

    angle_remainder = float(np.mod(angle_degrees, 45.0))
    if np.isclose(angle_remainder, 0.0, atol=1e-12) or np.isclose(
        angle_remainder, 45.0, atol=1e-12
    ):
        raise ValueError("angle_degrees must not be a multiple of 45 degrees")

    angle_radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    half_width = rectangle_width_mm / 2.0
    half_height = rectangle_height_mm / 2.0

    extent_x = abs(cosine) * half_width + abs(sine) * half_height
    extent_y = abs(sine) * half_width + abs(cosine) * half_height
    field_max_x_mm = min_x_mm + width_cells * cell_size_mm
    field_max_y_mm = min_y_mm + height_cells * cell_size_mm
    margin = transition_half_width_mm
    if not (
        center_x_mm - extent_x > min_x_mm + margin
        and center_x_mm + extent_x < field_max_x_mm - margin
        and center_y_mm - extent_y > min_y_mm + margin
        and center_y_mm + extent_y < field_max_y_mm - margin
    ):
        raise ValueError(
            "rotated rectangle and its transition band must lie inside "
            "the field"
        )

    x_coordinates_mm = min_x_mm + (
        np.arange(width_cells, dtype=np.float64) + 0.5
    ) * cell_size_mm
    y_coordinates_mm = min_y_mm + (
        np.arange(height_cells, dtype=np.float64) + 0.5
    ) * cell_size_mm
    x_grid, y_grid = np.meshgrid(x_coordinates_mm, y_coordinates_mm)

    translated_x = x_grid - center_x_mm
    translated_y = y_grid - center_y_mm
    local_x = cosine * translated_x + sine * translated_y
    local_y = -sine * translated_x + cosine * translated_y

    q_x = np.abs(local_x) - half_width
    q_y = np.abs(local_y) - half_height
    outside_distance = np.hypot(
        np.maximum(q_x, 0.0),
        np.maximum(q_y, 0.0),
    )
    inside_distance = np.minimum(np.maximum(q_x, q_y), 0.0)
    signed_distance = outside_distance + inside_distance

    slope = (
        (inside_density - outside_density)
        / (2.0 * transition_half_width_mm)
    )
    density = threshold - signed_distance * slope
    density = np.clip(
        density, outside_density, inside_density
    ).astype(np.float64, copy=False)

    return SyntheticRotatedRectangleField(
        density=density,
        x_coordinates_mm=x_coordinates_mm,
        y_coordinates_mm=y_coordinates_mm,
        min_x_mm=min_x_mm,
        min_y_mm=min_y_mm,
        cell_size_mm=cell_size_mm,
        threshold=threshold,
        center_x_mm=center_x_mm,
        center_y_mm=center_y_mm,
        rectangle_width_mm=rectangle_width_mm,
        rectangle_height_mm=rectangle_height_mm,
        angle_degrees=angle_degrees,
    )


def create_synthetic_point_density_rotated_rectangle_field(
    *,
    min_x_mm: float = 0.0,
    min_y_mm: float = 0.0,
    width_cells: int = 100,
    height_cells: int = 80,
    cell_size_mm: float = 1.0,
    center_x_mm: float = 50.3,
    center_y_mm: float = 40.7,
    rectangle_width_mm: float = 46.2,
    rectangle_height_mm: float = 28.6,
    angle_degrees: float = 23.0,
    point_spacing_mm: float = 0.25,
    point_offset_x_mm: float = 0.13,
    point_offset_y_mm: float = 0.07,
    threshold: float = 8.0,
) -> SyntheticPointDensityRotatedRectangleField:
    """Create raw integer density from a deterministic world XY lattice."""
    width_cells = _positive_integer(width_cells, "width_cells")
    height_cells = _positive_integer(height_cells, "height_cells")

    min_x_mm = _finite_float(min_x_mm, "min_x_mm")
    min_y_mm = _finite_float(min_y_mm, "min_y_mm")
    cell_size_mm = _finite_float(cell_size_mm, "cell_size_mm")
    center_x_mm = _finite_float(center_x_mm, "center_x_mm")
    center_y_mm = _finite_float(center_y_mm, "center_y_mm")
    rectangle_width_mm = _finite_float(
        rectangle_width_mm, "rectangle_width_mm"
    )
    rectangle_height_mm = _finite_float(
        rectangle_height_mm, "rectangle_height_mm"
    )
    angle_degrees = _finite_float(angle_degrees, "angle_degrees")
    point_spacing_mm = _finite_float(
        point_spacing_mm, "point_spacing_mm"
    )
    point_offset_x_mm = _finite_float(
        point_offset_x_mm, "point_offset_x_mm"
    )
    point_offset_y_mm = _finite_float(
        point_offset_y_mm, "point_offset_y_mm"
    )
    threshold = _finite_float(threshold, "threshold")

    if cell_size_mm <= 0.0:
        raise ValueError("cell_size_mm must be greater than zero")
    if rectangle_width_mm <= 0.0 or rectangle_height_mm <= 0.0:
        raise ValueError("rectangle dimensions must be greater than zero")
    if point_spacing_mm <= 0.0:
        raise ValueError("point_spacing_mm must be greater than zero")
    if not 0.0 <= point_offset_x_mm < point_spacing_mm:
        raise ValueError(
            "point_offset_x_mm must be in [0, point_spacing_mm)"
        )
    if not 0.0 <= point_offset_y_mm < point_spacing_mm:
        raise ValueError(
            "point_offset_y_mm must be in [0, point_spacing_mm)"
        )

    angle_remainder = float(np.mod(angle_degrees, 45.0))
    if np.isclose(angle_remainder, 0.0, atol=1e-12) or np.isclose(
        angle_remainder, 45.0, atol=1e-12
    ):
        raise ValueError("angle_degrees must not be a multiple of 45 degrees")

    angle_radians = np.deg2rad(angle_degrees)
    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    half_width = rectangle_width_mm / 2.0
    half_height = rectangle_height_mm / 2.0
    extent_x = abs(cosine) * half_width + abs(sine) * half_height
    extent_y = abs(sine) * half_width + abs(cosine) * half_height

    field_max_x_mm = min_x_mm + width_cells * cell_size_mm
    field_max_y_mm = min_y_mm + height_cells * cell_size_mm
    margin = 2.0 * cell_size_mm
    if not (
        center_x_mm - extent_x > min_x_mm + margin
        and center_x_mm + extent_x < field_max_x_mm - margin
        and center_y_mm - extent_y > min_y_mm + margin
        and center_y_mm + extent_y < field_max_y_mm - margin
    ):
        raise ValueError(
            "rotated rectangle must lie inside the field with margin"
        )

    x_points = np.arange(
        min_x_mm + point_offset_x_mm,
        field_max_x_mm,
        point_spacing_mm,
        dtype=np.float64,
    )
    y_points = np.arange(
        min_y_mm + point_offset_y_mm,
        field_max_y_mm,
        point_spacing_mm,
        dtype=np.float64,
    )
    x_grid, y_grid = np.meshgrid(x_points, y_points)

    translated_x = x_grid - center_x_mm
    translated_y = y_grid - center_y_mm
    local_x = cosine * translated_x + sine * translated_y
    local_y = -sine * translated_x + cosine * translated_y
    coordinate_scale = max(
        1.0,
        abs(center_x_mm),
        abs(center_y_mm),
        rectangle_width_mm,
        rectangle_height_mm,
    )
    tolerance_mm = (
        16.0 * np.finfo(np.float64).eps * coordinate_scale
    )
    inside_rectangle = (
        (np.abs(local_x) <= half_width + tolerance_mm)
        & (np.abs(local_y) <= half_height + tolerance_mm)
    )
    points_xy_mm = np.column_stack(
        (x_grid[inside_rectangle], y_grid[inside_rectangle])
    ).astype(np.float64, copy=False)
    if len(points_xy_mm) == 0:
        raise ValueError("the rotated rectangle contains no lattice points")

    density = accumulate_points_to_density(
        points_xy_mm,
        min_x_mm=min_x_mm,
        min_y_mm=min_y_mm,
        width_cells=width_cells,
        height_cells=height_cells,
        cell_size_mm=cell_size_mm,
    )
    if not 0.0 < threshold < int(np.max(density)):
        raise ValueError(
            "threshold must be positive and below the maximum cell count"
        )

    x_coordinates_mm = min_x_mm + (
        np.arange(width_cells, dtype=np.float64) + 0.5
    ) * cell_size_mm
    y_coordinates_mm = min_y_mm + (
        np.arange(height_cells, dtype=np.float64) + 0.5
    ) * cell_size_mm

    return SyntheticPointDensityRotatedRectangleField(
        density=density,
        points_xy_mm=points_xy_mm,
        x_coordinates_mm=x_coordinates_mm,
        y_coordinates_mm=y_coordinates_mm,
        min_x_mm=min_x_mm,
        min_y_mm=min_y_mm,
        cell_size_mm=cell_size_mm,
        threshold=threshold,
        center_x_mm=center_x_mm,
        center_y_mm=center_y_mm,
        rectangle_width_mm=rectangle_width_mm,
        rectangle_height_mm=rectangle_height_mm,
        angle_degrees=angle_degrees,
        point_spacing_mm=point_spacing_mm,
        point_offset_x_mm=point_offset_x_mm,
        point_offset_y_mm=point_offset_y_mm,
    )


def signed_ring_area(points: np.ndarray) -> float:
    """Return the signed shoelace area of an ordered ring."""
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(points_array) < 3:
        raise ValueError("a ring must contain at least three points")
    if not np.isfinite(points_array).all():
        raise ValueError("ring coordinates must be finite")

    x_coordinates = points_array[:, 0]
    y_coordinates = points_array[:, 1]
    return float(
        0.5
        * (
            np.dot(x_coordinates, np.roll(y_coordinates, -1))
            - np.dot(y_coordinates, np.roll(x_coordinates, -1))
        )
    )


def normalize_external_ring(points: np.ndarray) -> np.ndarray:
    """Return an implicitly closed, counter-clockwise float64 ring."""
    ring = np.asarray(points, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if not np.isfinite(ring).all():
        raise ValueError("ring coordinates must be finite")

    ring = ring.copy()
    if len(ring) >= 2 and np.allclose(
        ring[0], ring[-1], rtol=0.0, atol=1e-12
    ):
        ring = ring[:-1].copy()

    if len(np.unique(ring, axis=0)) < 3:
        raise ValueError("a ring must contain at least three unique points")

    area = signed_ring_area(ring)
    if area == 0.0:
        raise ValueError("ring must have non-zero signed area")
    if area < 0.0:
        ring = ring[::-1].copy()

    return ring.astype(np.float64, copy=False)


def rotated_rectangle_vertices(
    field: RotatedRectangleField,
) -> np.ndarray:
    """Return the four exact rectangle vertices in counter-clockwise order."""
    half_width = field.rectangle_width_mm / 2.0
    half_height = field.rectangle_height_mm / 2.0
    local_vertices = np.array(
        [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
        ],
        dtype=np.float64,
    )

    angle_radians = np.deg2rad(field.angle_degrees)
    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    rotation = np.array(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )
    vertices = local_vertices @ rotation.T
    vertices += np.array(
        [field.center_x_mm, field.center_y_mm],
        dtype=np.float64,
    )
    return normalize_external_ring(vertices)


def _ring_segments(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices_array = normalize_external_ring(vertices)
    return vertices_array, np.roll(vertices_array, -1, axis=0)


def extract_mask_external_contour(
    field: SyntheticField,
) -> np.ndarray:
    """Extract the threshold-mask baseline with the project OpenCV code."""
    mask = field.density >= field.threshold
    contour_pixels = extract_external_contour(mask)
    grid = DensityGrid(
        density=field.density,
        cell_size=field.cell_size_mm,
        min_x=field.min_x_mm,
        min_y=field.min_y_mm,
    )
    contour_world = contour_pixels_to_world(contour_pixels, grid)
    return normalize_external_ring(contour_world)


def extract_contourpy_external_contour(
    field: SyntheticField,
) -> np.ndarray:
    """Extract the largest closed level line from the continuous density."""
    generator = contourpy.contour_generator(
        x=field.x_coordinates_mm,
        y=field.y_coordinates_mm,
        z=np.asarray(field.density, dtype=np.float64),
    )
    lines = generator.lines(field.threshold)

    closure_tolerance_mm = max(1e-12, field.cell_size_mm * 1e-9)
    closed_rings: list[np.ndarray] = []
    for line in lines:
        line_array = np.asarray(line, dtype=np.float64)
        if (
            line_array.ndim == 2
            and line_array.shape[1] == 2
            and len(line_array) >= 4
            and np.allclose(
                line_array[0],
                line_array[-1],
                rtol=0.0,
                atol=closure_tolerance_mm,
            )
        ):
            closed_rings.append(normalize_external_ring(line_array))

    if not closed_rings:
        raise ValueError("ContourPy returned no closed level contours")

    return max(closed_rings, key=lambda ring: abs(signed_ring_area(ring)))


def _point_to_segment_distances(
    points: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> np.ndarray:
    segment = segment_end - segment_start
    length_squared = float(np.dot(segment, segment))
    projection = ((points - segment_start) @ segment) / length_squared
    projection = np.clip(projection, 0.0, 1.0)
    closest_points = segment_start + projection[:, np.newaxis] * segment
    return np.linalg.norm(points - closest_points, axis=1)


def compute_rectangle_contour_metrics(
    contour: np.ndarray,
    field: SyntheticRectangleField,
) -> RectangleContourMetrics:
    """Measure one contour against the exact finite rectangle boundary."""
    ring = normalize_external_ring(contour)
    area = signed_ring_area(ring)

    actual_min_x = float(np.min(ring[:, 0]))
    actual_max_x = float(np.max(ring[:, 0]))
    actual_min_y = float(np.min(ring[:, 1]))
    actual_max_y = float(np.max(ring[:, 1]))

    bounding_box_error = (
        abs(actual_min_x - field.rectangle_min_x_mm)
        + abs(actual_max_x - field.rectangle_max_x_mm)
        + abs(actual_min_y - field.rectangle_min_y_mm)
        + abs(actual_max_y - field.rectangle_max_y_mm)
    )

    expected_area = (
        field.rectangle_max_x_mm - field.rectangle_min_x_mm
    ) * (field.rectangle_max_y_mm - field.rectangle_min_y_mm)
    area_error = abs(abs(area) - expected_area)

    lower_left = np.array(
        [field.rectangle_min_x_mm, field.rectangle_min_y_mm],
        dtype=np.float64,
    )
    lower_right = np.array(
        [field.rectangle_max_x_mm, field.rectangle_min_y_mm],
        dtype=np.float64,
    )
    upper_right = np.array(
        [field.rectangle_max_x_mm, field.rectangle_max_y_mm],
        dtype=np.float64,
    )
    upper_left = np.array(
        [field.rectangle_min_x_mm, field.rectangle_max_y_mm],
        dtype=np.float64,
    )
    boundary_distances = np.minimum.reduce(
        (
            _point_to_segment_distances(ring, lower_left, lower_right),
            _point_to_segment_distances(ring, lower_right, upper_right),
            _point_to_segment_distances(ring, upper_right, upper_left),
            _point_to_segment_distances(ring, upper_left, lower_left),
        )
    )

    rms_boundary_error = float(
        np.sqrt(np.mean(np.square(boundary_distances)))
    )
    max_boundary_error = float(np.max(boundary_distances))

    return RectangleContourMetrics(
        point_count=len(ring),
        signed_area_mm2=area,
        min_x_mm=actual_min_x,
        max_x_mm=actual_max_x,
        min_y_mm=actual_min_y,
        max_y_mm=actual_max_y,
        bounding_box_error_mm=float(bounding_box_error),
        area_error_mm2=float(area_error),
        rms_boundary_error_mm=rms_boundary_error,
        max_boundary_error_mm=max_boundary_error,
    )


def compute_rotated_rectangle_contour_metrics(
    contour: np.ndarray,
    field: RotatedRectangleField,
    *,
    corner_exclusion_mm: float | None = None,
) -> RotatedRectangleContourMetrics:
    """Measure boundary and four independent line fits for a rotated box."""
    ring = normalize_external_ring(contour)
    area = signed_ring_area(ring)
    vertices = rotated_rectangle_vertices(field)
    segment_starts, segment_ends = _ring_segments(vertices)

    distances_to_segments = np.column_stack(
        [
            _point_to_segment_distances(ring, start, end)
            for start, end in zip(segment_starts, segment_ends)
        ]
    )
    assigned_sides = np.argmin(distances_to_segments, axis=1)
    boundary_distances = np.min(distances_to_segments, axis=1)

    rms_boundary_error = float(
        np.sqrt(np.mean(np.square(boundary_distances)))
    )
    max_boundary_error = float(np.max(boundary_distances))
    expected_area = (
        field.rectangle_width_mm * field.rectangle_height_mm
    )
    area_error = abs(abs(area) - expected_area)

    if corner_exclusion_mm is None:
        corner_exclusion_mm = 2.0 * field.cell_size_mm
    corner_exclusion_mm = _finite_float(
        corner_exclusion_mm, "corner_exclusion_mm"
    )
    if corner_exclusion_mm < 0.0:
        raise ValueError("corner_exclusion_mm must be non-negative")

    distances_to_vertices = np.linalg.norm(
        ring[:, np.newaxis, :] - vertices[np.newaxis, :, :],
        axis=2,
    )
    near_any_corner = (
        np.min(distances_to_vertices, axis=1) < corner_exclusion_mm
    )

    true_side_rms_values: list[float] = []
    fitted_side_rms_values: list[float] = []
    angle_error_values: list[float] = []

    for side_index, (start, end) in enumerate(
        zip(segment_starts, segment_ends)
    ):
        side_points = ring[
            (assigned_sides == side_index) & ~near_any_corner
        ]
        if len(side_points) < 3:
            raise ValueError(
                f"side {side_index} has fewer than three points "
                "after corner exclusion"
            )

        true_direction = end - start
        true_direction /= np.linalg.norm(true_direction)
        relative_to_true_line = side_points - start
        true_line_distances = np.abs(
            true_direction[0] * relative_to_true_line[:, 1]
            - true_direction[1] * relative_to_true_line[:, 0]
        )
        true_side_rms_values.append(
            float(np.sqrt(np.mean(np.square(true_line_distances))))
        )

        centroid = np.mean(side_points, axis=0)
        centered_points = side_points - centroid
        _u, _singular_values, right_vectors = np.linalg.svd(
            centered_points,
            full_matrices=False,
        )
        fitted_direction = right_vectors[0].astype(
            np.float64, copy=False
        )
        fitted_direction /= np.linalg.norm(fitted_direction)

        fitted_projection = (
            centered_points @ fitted_direction
        )[:, np.newaxis] * fitted_direction
        fitted_line_distances = np.linalg.norm(
            centered_points - fitted_projection,
            axis=1,
        )
        fitted_side_rms_values.append(
            float(np.sqrt(np.mean(np.square(fitted_line_distances))))
        )

        direction_dot = float(
            np.clip(
                abs(np.dot(fitted_direction, true_direction)),
                0.0,
                1.0,
            )
        )
        angle_error_values.append(
            float(np.degrees(np.arccos(direction_dot)))
        )

    return RotatedRectangleContourMetrics(
        point_count=len(ring),
        signed_area_mm2=area,
        area_error_mm2=float(area_error),
        rms_boundary_error_mm=rms_boundary_error,
        max_boundary_error_mm=max_boundary_error,
        mean_true_side_rms_mm=float(np.mean(true_side_rms_values)),
        max_true_side_rms_mm=float(np.max(true_side_rms_values)),
        mean_fitted_side_rms_mm=float(np.mean(fitted_side_rms_values)),
        max_fitted_side_rms_mm=float(np.max(fitted_side_rms_values)),
        mean_side_angle_error_deg=float(np.mean(angle_error_values)),
        max_side_angle_error_deg=float(np.max(angle_error_values)),
    )


def run_rectangle_comparison() -> RectangleContourComparison:
    """Run the deterministic default synthetic comparison."""
    field = create_synthetic_rectangle_field()
    mask_contour = extract_mask_external_contour(field)
    contourpy_contour = extract_contourpy_external_contour(field)
    mask_metrics = compute_rectangle_contour_metrics(mask_contour, field)
    contourpy_metrics = compute_rectangle_contour_metrics(
        contourpy_contour, field
    )

    return RectangleContourComparison(
        field=field,
        mask_contour=mask_contour,
        contourpy_contour=contourpy_contour,
        mask_metrics=mask_metrics,
        contourpy_metrics=contourpy_metrics,
    )


def run_rotated_rectangle_comparison(
) -> RotatedRectangleContourComparison:
    """Run the deterministic rotated-rectangle comparison."""
    field = create_synthetic_rotated_rectangle_field()
    mask_contour = extract_mask_external_contour(field)
    contourpy_contour = extract_contourpy_external_contour(field)
    mask_metrics = compute_rotated_rectangle_contour_metrics(
        mask_contour, field
    )
    contourpy_metrics = compute_rotated_rectangle_contour_metrics(
        contourpy_contour, field
    )

    return RotatedRectangleContourComparison(
        field=field,
        mask_contour=mask_contour,
        contourpy_contour=contourpy_contour,
        mask_metrics=mask_metrics,
        contourpy_metrics=contourpy_metrics,
    )


def run_point_density_rotated_rectangle_comparison(
) -> PointDensityRotatedRectangleComparison:
    """Compare both extractors on raw counts from deterministic XY points."""
    field = create_synthetic_point_density_rotated_rectangle_field()
    mask_contour = extract_mask_external_contour(field)
    contourpy_contour = extract_contourpy_external_contour(field)
    mask_metrics = compute_rotated_rectangle_contour_metrics(
        mask_contour, field
    )
    contourpy_metrics = compute_rotated_rectangle_contour_metrics(
        contourpy_contour, field
    )

    return PointDensityRotatedRectangleComparison(
        field=field,
        mask_contour=mask_contour,
        contourpy_contour=contourpy_contour,
        mask_metrics=mask_metrics,
        contourpy_metrics=contourpy_metrics,
    )


def _print_metrics(name: str, metrics: RectangleContourMetrics) -> None:
    print(f"{name}:")
    print(f"  points: {metrics.point_count}")
    print(f"  bbox error: {metrics.bounding_box_error_mm:.6f} mm")
    print(f"  area error: {metrics.area_error_mm2:.6f} mm^2")
    print(
        "  RMS boundary error: "
        f"{metrics.rms_boundary_error_mm:.6f} mm"
    )
    print(
        "  max boundary error: "
        f"{metrics.max_boundary_error_mm:.6f} mm"
    )


def _print_rotated_metrics(
    name: str,
    metrics: RotatedRectangleContourMetrics,
) -> None:
    print(f"{name}:")
    print(f"  points: {metrics.point_count}")
    print(f"  area error: {metrics.area_error_mm2:.6f} mm^2")
    print(
        "  RMS boundary error: "
        f"{metrics.rms_boundary_error_mm:.6f} mm"
    )
    print(
        "  max boundary error: "
        f"{metrics.max_boundary_error_mm:.6f} mm"
    )
    print(
        "  mean true-side RMS: "
        f"{metrics.mean_true_side_rms_mm:.6f} mm"
    )
    print(
        "  max true-side RMS: "
        f"{metrics.max_true_side_rms_mm:.6f} mm"
    )
    print(
        "  mean fitted-side RMS: "
        f"{metrics.mean_fitted_side_rms_mm:.6f} mm"
    )
    print(
        "  max fitted-side RMS: "
        f"{metrics.max_fitted_side_rms_mm:.6f} mm"
    )
    print(
        "  mean side angle error: "
        f"{metrics.mean_side_angle_error_deg:.6f} deg"
    )
    print(
        "  max side angle error: "
        f"{metrics.max_side_angle_error_deg:.6f} deg"
    )


if __name__ == "__main__":
    comparison = run_rectangle_comparison()
    print("Synthetic rectangle contour comparison")
    print()
    _print_metrics("Mask contour", comparison.mask_metrics)
    print()
    _print_metrics("ContourPy contour", comparison.contourpy_metrics)
    print()
    improvement = (
        comparison.mask_metrics.rms_boundary_error_mm
        - comparison.contourpy_metrics.rms_boundary_error_mm
    )
    print(f"ContourPy RMS improvement: {improvement:.6f} mm")

    rotated_comparison = run_rotated_rectangle_comparison()
    print()
    print("Rotated rectangle contour comparison")
    print(f"  angle: {rotated_comparison.field.angle_degrees:.6f} deg")
    print()
    _print_rotated_metrics(
        "Mask contour", rotated_comparison.mask_metrics
    )
    print()
    _print_rotated_metrics(
        "ContourPy contour", rotated_comparison.contourpy_metrics
    )
    print()
    rotated_mask_metrics = rotated_comparison.mask_metrics
    rotated_contourpy_metrics = rotated_comparison.contourpy_metrics
    boundary_rms_improvement = (
        rotated_mask_metrics.rms_boundary_error_mm
        - rotated_contourpy_metrics.rms_boundary_error_mm
    )
    fitted_rms_improvement = (
        rotated_mask_metrics.mean_fitted_side_rms_mm
        - rotated_contourpy_metrics.mean_fitted_side_rms_mm
    )
    angle_error_improvement = (
        rotated_mask_metrics.mean_side_angle_error_deg
        - rotated_contourpy_metrics.mean_side_angle_error_deg
    )
    print("ContourPy improvements:")
    print(
        "  boundary RMS: "
        f"{boundary_rms_improvement:.6f} mm"
    )
    print(
        "  fitted-side RMS: "
        f"{fitted_rms_improvement:.6f} mm"
    )
    print(
        "  mean angle error: "
        f"{angle_error_improvement:.6f} deg"
    )

    point_comparison = run_point_density_rotated_rectangle_comparison()
    point_field = point_comparison.field
    positive_counts = point_field.density[point_field.density > 0]
    nonzero_cells = int(np.count_nonzero(point_field.density))
    maximum_cell_count = int(np.max(point_field.density))
    median_positive_count = float(np.median(positive_counts))

    print()
    print("Point-generated density rotated rectangle comparison")
    print(f"  angle: {point_field.angle_degrees:.6f} deg")
    print(f"  generated points: {len(point_field.points_xy_mm)}")
    print(f"  density dtype: {point_field.density.dtype}")
    print(f"  nonzero cells: {nonzero_cells}")
    print(f"  maximum cell count: {maximum_cell_count}")
    print(f"  median positive cell count: {median_positive_count:.6f}")
    print(f"  threshold: {point_field.threshold:.6f}")
    print()
    _print_rotated_metrics(
        "Mask contour", point_comparison.mask_metrics
    )
    print()
    _print_rotated_metrics(
        "ContourPy contour", point_comparison.contourpy_metrics
    )

    point_mask_metrics = point_comparison.mask_metrics
    point_contourpy_metrics = point_comparison.contourpy_metrics
    point_boundary_improvement = (
        point_mask_metrics.rms_boundary_error_mm
        - point_contourpy_metrics.rms_boundary_error_mm
    )
    point_fitted_improvement = (
        point_mask_metrics.mean_fitted_side_rms_mm
        - point_contourpy_metrics.mean_fitted_side_rms_mm
    )
    point_angle_improvement = (
        point_mask_metrics.mean_side_angle_error_deg
        - point_contourpy_metrics.mean_side_angle_error_deg
    )
    point_area_improvement = (
        point_mask_metrics.area_error_mm2
        - point_contourpy_metrics.area_error_mm2
    )

    print()
    print("ContourPy improvements:")
    print(f"  boundary RMS: {point_boundary_improvement:.6f} mm")
    print(f"  fitted-side RMS: {point_fitted_improvement:.6f} mm")
    print(f"  mean angle error: {point_angle_improvement:.6f} deg")
    print(f"  area error: {point_area_improvement:.6f} mm^2")

    floating_tolerance = 1e-12
    print()
    print("Experimental verdict:")
    print(
        "  boundary RMS improved: "
        + (
            "yes"
            if point_contourpy_metrics.rms_boundary_error_mm
            < point_mask_metrics.rms_boundary_error_mm
            else "no"
        )
    )
    print(
        "  fitted-side RMS improved: "
        + (
            "yes"
            if point_contourpy_metrics.mean_fitted_side_rms_mm
            < point_mask_metrics.mean_fitted_side_rms_mm
            else "no"
        )
    )
    print(
        "  mean angle error not worse: "
        + (
            "yes"
            if point_contourpy_metrics.mean_side_angle_error_deg
            <= (
                point_mask_metrics.mean_side_angle_error_deg
                + floating_tolerance
            )
            else "no"
        )
    )
    print(
        "  area error not worse: "
        + (
            "yes"
            if point_contourpy_metrics.area_error_mm2
            <= point_mask_metrics.area_error_mm2 + floating_tolerance
            else "no"
        )
    )
