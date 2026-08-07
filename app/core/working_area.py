from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.core.density_grid import DensityGrid
from app.core.mask_processing import apply_polygon_roi_to_mask, apply_roi_to_mask


Point = tuple[float, float]
RectangleBounds = tuple[float, float, float, float]
WorkingAreaKind = Literal["rectangle", "polygon"]


def _finite_point(point: tuple[float, float]) -> Point:
    x, y = float(point[0]), float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Working area coordinates must be finite")
    return x, y


@dataclass(frozen=True, slots=True)
class WorkingArea:
    kind: WorkingAreaKind
    rectangle_bounds: RectangleBounds | None = None
    polygon_points: tuple[Point, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "rectangle":
            if self.rectangle_bounds is None or self.polygon_points:
                raise ValueError("Rectangle working area requires only bounds")
            min_x, min_y, max_x, max_y = (
                float(value) for value in self.rectangle_bounds
            )
            if not all(math.isfinite(value) for value in (min_x, min_y, max_x, max_y)):
                raise ValueError("Working area coordinates must be finite")
            normalized = (
                min(min_x, max_x),
                min(min_y, max_y),
                max(min_x, max_x),
                max(min_y, max_y),
            )
            if normalized[0] == normalized[2] or normalized[1] == normalized[3]:
                raise ValueError("Rectangle working area must have positive size")
            object.__setattr__(self, "rectangle_bounds", normalized)
            return

        if self.kind != "polygon" or self.rectangle_bounds is not None:
            raise ValueError("Unsupported working area kind")

        points = tuple(_finite_point(point) for point in self.polygon_points)
        if len(set(points)) < 3:
            raise ValueError("Polygon working area requires at least 3 unique points")

        twice_area = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
        if math.isclose(twice_area, 0.0, abs_tol=1e-12):
            raise ValueError("Polygon working area must have positive area")
        object.__setattr__(self, "polygon_points", points)

    @classmethod
    def from_rectangle_points(cls, point_a: Point, point_b: Point) -> WorkingArea:
        ax, ay = _finite_point(point_a)
        bx, by = _finite_point(point_b)
        return cls(kind="rectangle", rectangle_bounds=(ax, ay, bx, by))

    @classmethod
    def from_rectangle_bounds(cls, bounds: RectangleBounds) -> WorkingArea:
        return cls(kind="rectangle", rectangle_bounds=bounds)

    @classmethod
    def from_polygon(cls, points: list[Point] | tuple[Point, ...]) -> WorkingArea:
        return cls(kind="polygon", polygon_points=tuple(points))

    def processing_parameters(
        self,
    ) -> tuple[RectangleBounds | None, list[Point] | None]:
        if self.kind == "rectangle":
            return self.rectangle_bounds, None
        return None, list(self.polygon_points)

    def to_grid_mask(self, grid: DensityGrid) -> np.ndarray:
        full_mask = np.ones(grid.density.shape, dtype=np.uint8)
        if self.kind == "rectangle":
            assert self.rectangle_bounds is not None
            selected = apply_roi_to_mask(full_mask, grid, self.rectangle_bounds)
        else:
            selected = apply_polygon_roi_to_mask(
                full_mask,
                grid,
                list(self.polygon_points),
            )
        return selected.astype(bool)
