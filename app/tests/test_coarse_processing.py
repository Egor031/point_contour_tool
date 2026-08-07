from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.density_grid import DensityGrid  # noqa: E402
from app.services.coarse_processing import (  # noqa: E402
    build_processing_masks,
    extract_preliminary_contour,
    find_hole_candidates,
    prepare_density,
)


def _write_xyz(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.write_text(
        "".join(f"{x} {y} {z}\n" for x, y, z in points),
        encoding="utf-8",
    )


def _grid_with_hole(
    *,
    size: int = 31,
    cell_size: float = 1.0,
    min_x: float = 0.0,
    min_y: float = 0.0,
) -> DensityGrid:
    density = np.zeros((size, size), dtype=np.uint8)
    density[2 : size - 2, 2 : size - 2] = 2
    cv2.circle(
        density,
        center=(size // 2, size // 2),
        radius=2,
        color=0,
        thickness=-1,
    )
    return DensityGrid(
        density=density.astype(np.uint32),
        cell_size=cell_size,
        min_x=min_x,
        min_y=min_y,
    )


class TestCoarseProcessingFacade(unittest.TestCase):
    def test_prepare_density_returns_stats_grid_and_cache_information(self):
        points = [
            (10.0, 20.0, -1.0),
            (10.2, 20.2, 0.0),
            (11.0, 21.0, 2.0),
            (12.0, 22.0, 3.0),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "density.xyz"
            cache_dir = temp_path / "cache"
            _write_xyz(source_path, points)

            first = prepare_density(
                source_path,
                cell_size=1.0,
                cache_dir=cache_dir,
            )
            second = prepare_density(
                source_path,
                cell_size=1.0,
                cache_dir=cache_dir,
            )

        self.assertEqual(first.source_path, source_path)
        self.assertEqual(first.stats.point_count, 4)
        self.assertEqual(first.grid.width, 3)
        self.assertEqual(first.grid.height, 3)
        self.assertEqual(int(first.grid.density.sum()), 4)
        self.assertFalse(first.stats_from_cache)
        self.assertFalse(first.density_from_cache)
        self.assertTrue(second.stats_from_cache)
        self.assertTrue(second.density_from_cache)
        np.testing.assert_array_equal(first.grid.density, second.grid.density)

    def test_build_processing_masks_exposes_pre_and_post_fill_masks(self):
        grid = _grid_with_hole()
        edits = [
            {"mode": "remove", "x": 6.2, "y": 6.2, "radius_mm": 0.1},
        ]

        result = build_processing_masks(
            grid,
            threshold_mode="manual",
            manual_threshold=1.0,
            roi=(1.5, 1.5, 29.5, 29.5),
            mask_edits=edits,
            fill_holes_area=20,
        )

        self.assertIs(result.grid, grid)
        self.assertEqual(result.threshold_result.threshold, 1.0)
        self.assertIsNotNone(result.mask_before_edits)
        self.assertIsNotNone(result.mask_edits_stats)
        self.assertEqual(result.mask_before_edits[6, 6], 1)
        self.assertEqual(result.mask_for_holes[6, 6], 0)
        self.assertEqual(result.contour_mask[6, 6], 1)
        self.assertEqual(result.mask_for_holes[15, 15], 0)
        self.assertEqual(result.contour_mask[15, 15], 1)
        self.assertIsNot(result.mask_for_holes, result.contour_mask)

    def test_extract_preliminary_contour_uses_contour_mask(self):
        grid = _grid_with_hole(
            cell_size=0.5,
            min_x=100.0,
            min_y=-50.0,
        )
        masks = build_processing_masks(
            grid,
            threshold_mode="manual",
            manual_threshold=1.0,
            fill_holes_area=20,
        )

        contour = extract_preliminary_contour(masks)

        self.assertGreater(contour.point_count, 0)
        self.assertAlmostEqual(float(contour.contour_world[:, 0].min()), 101.25)
        self.assertAlmostEqual(float(contour.contour_world[:, 0].max()), 114.25)
        self.assertAlmostEqual(float(contour.contour_world[:, 1].min()), -48.75)
        self.assertAlmostEqual(float(contour.contour_world[:, 1].max()), -35.75)

    def test_find_hole_candidates_returns_candidates_groups_and_count(self):
        grid = _grid_with_hole()
        masks = build_processing_masks(
            grid,
            threshold_mode="manual",
            manual_threshold=1.0,
        )

        holes = find_hole_candidates(
            masks,
            min_diameter_mm=0.0,
            min_circularity=0.0,
            max_aspect_ratio_deviation=1.0,
            max_error_ratio=1.0,
            group_tolerance_mm=1.0,
        )

        self.assertEqual(len(holes.candidates), 1)
        self.assertEqual(holes.accepted_count, 1)
        self.assertEqual(len(holes.groups), 1)
        self.assertEqual(holes.groups[0]["id"], "G1")
        self.assertEqual(holes.groups[0]["count"], 1)

    def test_downstream_facade_operations_do_not_reopen_source(self):
        size = 21
        center = (size // 2, size // 2)
        points = []
        for y in range(size):
            for x in range(size):
                if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= 3**2:
                    continue
                points.append((float(x), float(y), 0.0))

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.xyz"
            _write_xyz(source_path, points)
            density = prepare_density(
                source_path,
                cell_size=1.0,
                use_cache=False,
            )

            source_path.unlink()

            masks = build_processing_masks(
                density.grid,
                threshold_mode="manual",
                manual_threshold=1.0,
            )
            contour = extract_preliminary_contour(masks)
            holes = find_hole_candidates(
                masks,
                min_diameter_mm=0.0,
                min_circularity=0.0,
                max_aspect_ratio_deviation=1.0,
                max_error_ratio=1.0,
            )

        self.assertGreater(contour.point_count, 0)
        self.assertEqual(len(holes.candidates), 1)


if __name__ == "__main__":
    unittest.main()
