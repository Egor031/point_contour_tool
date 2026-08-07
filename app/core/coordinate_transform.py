from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CoordinateTransform:
    """Transform between world, grid-cell-center, and preview-pixel spaces."""

    grid_min_x: float
    grid_min_y: float
    cell_size: float
    grid_width: int
    grid_height: int
    preview_width: int
    preview_height: int
    preview_flips_y: bool = True

    def __post_init__(self) -> None:
        numeric_values = (self.grid_min_x, self.grid_min_y, self.cell_size)
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Grid origin and cell size must be finite")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be greater than zero")

        dimensions = (
            self.grid_width,
            self.grid_height,
            self.preview_width,
            self.preview_height,
        )
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in dimensions
        ):
            raise TypeError("Grid and preview dimensions must be integers")
        if not all(value > 0 for value in dimensions):
            raise ValueError("Grid and preview dimensions must be greater than zero")

    @property
    def scale_x(self) -> float:
        return self.preview_width / self.grid_width

    @property
    def scale_y(self) -> float:
        return self.preview_height / self.grid_height

    def grid_to_world(self, grid_x: float, grid_y: float) -> tuple[float, float]:
        """Map continuous grid coordinates whose integer values are cell centers."""
        world_x = self.grid_min_x + (grid_x + 0.5) * self.cell_size
        world_y = self.grid_min_y + (grid_y + 0.5) * self.cell_size
        return world_x, world_y

    def world_to_grid(self, world_x: float, world_y: float) -> tuple[float, float]:
        """Map world coordinates to continuous cell-center grid coordinates."""
        grid_x = (world_x - self.grid_min_x) / self.cell_size - 0.5
        grid_y = (world_y - self.grid_min_y) / self.cell_size - 0.5
        return grid_x, grid_y

    def grid_cell_center_to_world(
        self,
        cell_x: int,
        cell_y: int,
    ) -> tuple[float, float]:
        return self.grid_to_world(float(cell_x), float(cell_y))

    def grid_to_preview(
        self,
        grid_x: float,
        grid_y: float,
    ) -> tuple[float, float]:
        """Map cell-center grid coordinates to OpenCV pixel-center coordinates."""
        preview_x = (grid_x + 0.5) * self.scale_x - 0.5
        if self.preview_flips_y:
            displayed_grid_y = self.grid_height - 1.0 - grid_y
        else:
            displayed_grid_y = grid_y
        preview_y = (displayed_grid_y + 0.5) * self.scale_y - 0.5
        return preview_x, preview_y

    def preview_to_grid(
        self,
        preview_x: float,
        preview_y: float,
    ) -> tuple[float, float]:
        """Map OpenCV pixel-center coordinates to continuous grid coordinates."""
        grid_x = (preview_x + 0.5) / self.scale_x - 0.5
        displayed_grid_y = (preview_y + 0.5) / self.scale_y - 0.5
        if self.preview_flips_y:
            grid_y = self.grid_height - 1.0 - displayed_grid_y
        else:
            grid_y = displayed_grid_y
        return grid_x, grid_y

    def world_to_preview(
        self,
        world_x: float,
        world_y: float,
    ) -> tuple[float, float]:
        return self.grid_to_preview(*self.world_to_grid(world_x, world_y))

    def preview_to_world(
        self,
        preview_x: float,
        preview_y: float,
    ) -> tuple[float, float]:
        return self.grid_to_world(*self.preview_to_grid(preview_x, preview_y))

    def world_radius_to_preview(self, radius: float) -> tuple[float, float]:
        radius_cells = abs(radius) / self.cell_size
        return radius_cells * self.scale_x, radius_cells * self.scale_y
