from __future__ import annotations

import math
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


from app.core.contour_extractor import (  # noqa: E402
    build_external_contour,
    contour_pixels_to_world,
    extract_external_contour,
)
from app.core.density_grid import DensityGrid, build_density_grid  # noqa: E402
from app.core.hole_detector import (  # noqa: E402
    HoleCandidate,
    cluster_holes_by_diameter,
    detect_circular_holes,
)
from app.core.mask_edits import apply_mask_edits  # noqa: E402
from app.core.mask_processing import (  # noqa: E402
    apply_polygon_roi_to_mask,
    apply_roi_to_mask,
    build_mask_from_density,
    fill_small_holes,
    keep_largest_component,
    remove_small_components,
)
from app.core.xyz_reader import compute_stats  # noqa: E402


def _write_xyz(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.write_text(
        "".join(f"{x} {y} {z}\n" for x, y, z in points),
        encoding="utf-8",
    )


def _grid(
    shape: tuple[int, int],
    *,
    cell_size: float = 1.0,
    min_x: float = 0.0,
    min_y: float = 0.0,
) -> DensityGrid:
    return DensityGrid(
        density=np.zeros(shape, dtype=np.uint32),
        cell_size=cell_size,
        min_x=min_x,
        min_y=min_y,
    )


def _mask_with_circular_hole(
    size: int = 61,
    center: tuple[int, int] = (30, 30),
    radius: int = 6,
) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[2 : size - 2, 2 : size - 2] = 1
    cv2.circle(mask, center=center, radius=radius, color=0, thickness=-1)
    return mask


def _detect_holes(mask: np.ndarray, grid: DensityGrid, **overrides):
    options = {
        "min_diameter_mm": 0.0,
        "max_diameter_mm": None,
        "min_circularity": 0.0,
        "max_aspect_ratio_deviation": 1.0,
        "max_error_ratio": 1.0,
    }
    options.update(overrides)
    return detect_circular_holes(mask=mask, grid=grid, **options)


def _hole_candidate(
    candidate_id: int,
    diameter: float,
    *,
    accepted: bool = True,
) -> HoleCandidate:
    radius = diameter / 2.0
    return HoleCandidate(
        id=candidate_id,
        center_x=float(candidate_id),
        center_y=0.0,
        radius=radius,
        diameter=diameter,
        center_px=float(candidate_id),
        center_py=0.0,
        radius_px=radius,
        area_cells=1,
        area_mm2=math.pi * radius * radius,
        bbox_width_mm=diameter,
        bbox_height_mm=diameter,
        aspect_ratio=1.0,
        circularity=1.0,
        mean_error_mm=0.0,
        max_error_mm=0.0,
        error_ratio=0.0,
        accepted=accepted,
    )


class TestStatisticsAndDensityCharacterization(unittest.TestCase):
    def test_compute_stats_returns_current_point_count_and_bounds(self):
        points = [
            (-2.5, 4.0, 10.0),
            (3.0, -1.0, -2.0),
            (0.5, 2.0, 7.5),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "stats.xyz"
            _write_xyz(source_path, points)

            stats = compute_stats(source_path)

        self.assertEqual(stats.file_path, source_path)
        self.assertEqual(stats.point_count, 3)
        self.assertEqual(stats.min_x, -2.5)
        self.assertEqual(stats.max_x, 3.0)
        self.assertEqual(stats.min_y, -1.0)
        self.assertEqual(stats.max_y, 4.0)
        self.assertEqual(stats.min_z, -2.0)
        self.assertEqual(stats.max_z, 10.0)
        self.assertEqual(stats.width, 5.5)
        self.assertEqual(stats.height, 5.0)

    def test_density_grid_shape_and_known_cell_counts(self):
        points = [
            (10.0, -1.0, 0.0),
            (10.9, -0.1, 1.0),
            (11.0, 0.0, 2.0),
            (11.9, 0.9, 3.0),
            (12.0, 1.0, 4.0),
        ]
        expected = np.array(
            [
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 1],
            ],
            dtype=np.uint32,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "density.xyz"
            _write_xyz(source_path, points)
            stats = compute_stats(source_path)

            grid = build_density_grid(source_path, stats, cell_size=1.0)

        self.assertEqual(grid.cell_size, 1.0)
        self.assertEqual(grid.min_x, 10.0)
        self.assertEqual(grid.min_y, -1.0)
        self.assertEqual(grid.width, 3)
        self.assertEqual(grid.height, 3)
        np.testing.assert_array_equal(grid.density, expected)
        self.assertEqual(int(grid.density.sum()), len(points))

    def test_current_pixel_to_world_transform_uses_cell_centers(self):
        grid = _grid((2, 3), cell_size=2.0, min_x=10.0, min_y=-4.0)
        pixels = np.array([[0.0, 0.0], [2.0, 1.0]], dtype=np.float32)

        world = contour_pixels_to_world(pixels, grid)

        np.testing.assert_allclose(
            world,
            np.array([[11.0, -3.0], [15.0, -1.0]], dtype=np.float64),
        )


class TestMaskAndRoiCharacterization(unittest.TestCase):
    def test_threshold_component_cleanup_and_keep_largest(self):
        density = np.array(
            [
                [0, 0, 0, 0, 0, 0, 5],
                [0, 3, 5, 0, 0, 0, 0],
                [0, 5, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 4, 4, 0],
                [0, 0, 0, 0, 0, 0, 2],
            ],
            dtype=np.uint32,
        )
        expected_largest = np.zeros_like(density, dtype=np.uint8)
        expected_largest[1:3, 1:3] = 1

        result = build_mask_from_density(
            density,
            mode="manual",
            manual_threshold=3.0,
        )
        cleaned = remove_small_components(result.mask, min_area_cells=3)
        largest = keep_largest_component(result.mask)

        self.assertEqual(result.threshold, 3.0)
        self.assertEqual(result.mode, "manual")
        self.assertEqual(result.nonzero_density_cells, 8)
        self.assertEqual(result.mask_cells, 7)
        self.assertEqual(result.mask[1, 1], 1)
        self.assertEqual(result.mask[4, 6], 0)
        np.testing.assert_array_equal(cleaned, expected_largest)
        np.testing.assert_array_equal(largest, expected_largest)

    def test_rectangle_roi_uses_world_cell_centers(self):
        mask = np.ones((3, 4), dtype=np.uint8)
        grid = _grid(mask.shape, cell_size=2.0, min_x=10.0, min_y=20.0)
        expected = np.zeros_like(mask)
        expected[0:2, 1:3] = 1

        filtered = apply_roi_to_mask(
            mask,
            grid,
            roi=(12.5, 20.5, 15.5, 23.5),
        )

        np.testing.assert_array_equal(filtered, expected)

    def test_polygon_roi_uses_current_world_to_pixel_rounding(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        grid = _grid(mask.shape)
        expected = np.zeros_like(mask)
        expected[1:4, 1:4] = 1

        filtered = apply_polygon_roi_to_mask(
            mask,
            grid,
            polygon_points=[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        )

        np.testing.assert_array_equal(filtered, expected)

    def test_multiple_add_and_remove_mask_edits_are_applied_in_order(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        grid = _grid(mask.shape)
        edits = [
            {"mode": "add", "x": 2.2, "y": 2.2, "radius_mm": 0.25},
            {"mode": "remove", "x": 2.2, "y": 2.2, "radius_mm": 0.25},
            {"mode": "add", "x": 4.2, "y": 4.2, "radius_mm": 0.25},
        ]
        expected = np.zeros_like(mask)
        expected[3, 4] = 1
        expected[4, 3:6] = 1
        expected[5, 4] = 1

        edited, stats = apply_mask_edits(mask, grid, edits)

        np.testing.assert_array_equal(edited, expected)
        self.assertEqual(stats.total_edits, 3)
        self.assertEqual(stats.edits_inside_grid, 3)
        self.assertEqual(stats.edits_outside_grid, 0)
        self.assertEqual(stats.edits_that_touched_white_mask, 1)
        self.assertEqual(stats.changed_cells, 5)


class TestContourAndHolesCharacterization(unittest.TestCase):
    def test_holes_use_pre_fill_mask_while_contour_uses_filled_mask(self):
        mask = _mask_with_circular_hole(size=41, center=(20, 20), radius=4)
        grid = _grid(mask.shape)

        # This mirrors the production ordering without patching app.main:
        # holes keep the pre-fill copy, while contour receives the filled mask.
        mask_for_holes = mask.copy()
        contour_mask = fill_small_holes(mask, max_hole_area_cells=60)

        holes_before_fill = _detect_holes(mask_for_holes, grid)
        holes_after_fill = _detect_holes(contour_mask, grid)
        contour = build_external_contour(contour_mask, grid)

        self.assertEqual(mask_for_holes[20, 20], 0)
        self.assertEqual(contour_mask[20, 20], 1)
        self.assertEqual(len(holes_before_fill), 1)
        self.assertEqual(holes_after_fill, [])
        self.assertGreater(contour.point_count, 0)

    def test_external_contour_ignores_hole_and_selects_largest_component(self):
        mask = np.zeros((16, 22), dtype=np.uint8)
        mask[1:4, 1:4] = 1
        mask[5:14, 9:20] = 1
        mask[8:10, 13:15] = 0
        mask_without_hole = mask.copy()
        mask_without_hole[8:10, 13:15] = 1
        grid = _grid(mask.shape, cell_size=2.0, min_x=100.0, min_y=-50.0)

        pixels_with_hole = extract_external_contour(mask)
        pixels_without_hole = extract_external_contour(mask_without_hole)
        result = build_external_contour(mask, grid)

        np.testing.assert_array_equal(pixels_with_hole, pixels_without_hole)
        self.assertGreater(result.point_count, 0)
        self.assertEqual(float(result.contour_world[:, 0].min()), 119.0)
        self.assertEqual(float(result.contour_world[:, 0].max()), 139.0)
        self.assertEqual(float(result.contour_world[:, 1].min()), -39.0)
        self.assertEqual(float(result.contour_world[:, 1].max()), -23.0)

    def test_hole_detection_geometry_and_fitted_diameter_search_range(self):
        mask = _mask_with_circular_hole()
        grid = _grid(mask.shape, min_x=100.0, min_y=200.0)

        baseline = _detect_holes(mask, grid)

        self.assertEqual(len(baseline), 1)
        candidate = baseline[0]
        self.assertTrue(candidate.accepted)
        self.assertAlmostEqual(candidate.center_x, 130.5, delta=0.25)
        self.assertAlmostEqual(candidate.center_y, 230.5, delta=0.25)
        self.assertGreater(candidate.radius, 5.0)
        self.assertLess(candidate.radius, 6.5)
        self.assertAlmostEqual(candidate.diameter, candidate.radius * 2.0)

        equivalent_diameter = 2.0 * math.sqrt(candidate.area_mm2 / math.pi)
        filter_boundary = (candidate.diameter + equivalent_diameter) / 2.0
        self.assertLess(candidate.diameter, filter_boundary)
        self.assertLess(filter_boundary, equivalent_diameter)

        below_min_at_boundary = _detect_holes(
            mask,
            grid,
            min_diameter_mm=filter_boundary,
        )
        below_larger_min = _detect_holes(
            mask,
            grid,
            min_diameter_mm=equivalent_diameter + 0.1,
        )
        max_at_boundary = _detect_holes(
            mask,
            grid,
            max_diameter_mm=filter_boundary,
        )[0]

        # Product requirement: the same fitted diameter shown to the user is
        # now used by Min/Max instead of the legacy equivalent-area diameter.
        self.assertEqual(below_min_at_boundary, [])
        self.assertEqual(below_larger_min, [])
        self.assertTrue(max_at_boundary.accepted)

    def test_fitted_diameter_filter_can_accept_when_equivalent_is_smaller(self):
        mask = np.zeros((81, 81), dtype=np.uint8)
        mask[2:-2, 2:-2] = 1
        cv2.ellipse(mask, (40, 40), (10, 4), 0, 0, 360, 0, -1)
        grid = _grid(mask.shape)

        candidate = _detect_holes(
            mask,
            grid,
            max_aspect_ratio_deviation=10.0,
        )[0]
        equivalent_diameter = 2.0 * math.sqrt(candidate.area_mm2 / math.pi)
        filter_boundary = (candidate.diameter + equivalent_diameter) / 2.0
        self.assertLess(equivalent_diameter, filter_boundary)
        self.assertLess(filter_boundary, candidate.diameter)

        min_filtered = _detect_holes(
            mask,
            grid,
            min_diameter_mm=filter_boundary,
            max_aspect_ratio_deviation=10.0,
        )[0]
        above_max = _detect_holes(
            mask,
            grid,
            max_diameter_mm=filter_boundary,
            max_aspect_ratio_deviation=10.0,
        )

        self.assertTrue(min_filtered.accepted)
        self.assertEqual(above_max, [])

    def test_hole_border_and_quality_rejections_are_unchanged(self):
        grid = _grid((81, 81))
        solid = np.zeros((81, 81), dtype=np.uint8)
        solid[2:-2, 2:-2] = 1
        self.assertEqual(_detect_holes(solid, grid), [])

        ellipse = solid.copy()
        cv2.ellipse(ellipse, (40, 40), (10, 4), 0, 0, 360, 0, -1)
        bad_aspect = _detect_holes(
            ellipse,
            grid,
            max_aspect_ratio_deviation=0.1,
            max_error_ratio=1.0,
        )[0]
        bad_fit = _detect_holes(
            ellipse,
            grid,
            max_aspect_ratio_deviation=10.0,
            max_error_ratio=0.1,
        )[0]

        rectangle = solid.copy()
        cv2.rectangle(rectangle, (28, 38), (52, 42), 0, -1)
        low_circularity = _detect_holes(
            rectangle,
            grid,
            min_circularity=0.8,
            max_aspect_ratio_deviation=10.0,
            max_error_ratio=1.0,
        )[0]

        self.assertEqual(bad_aspect.reject_reason, "bad_aspect_ratio")
        self.assertEqual(bad_fit.reject_reason, "bad_circle_fit")
        self.assertEqual(low_circularity.reject_reason, "low_circularity")
        quality_reasons = {
            item.reject_reason
            for item in (bad_aspect, bad_fit, low_circularity)
        }
        self.assertNotIn("too_small", quality_reasons)
        self.assertNotIn("too_large", quality_reasons)

    def test_hole_grouping_uses_diameter_tolerance_for_accepted_holes(self):
        holes = [
            _hole_candidate(1, 10.0),
            _hole_candidate(2, 10.6),
            _hole_candidate(3, 13.0),
            _hole_candidate(4, 10.2, accepted=False),
        ]

        groups = cluster_holes_by_diameter(holes, tolerance_mm=1.0)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["id"], "G1")
        self.assertEqual(groups[0]["count"], 2)
        self.assertAlmostEqual(groups[0]["diameter"], 10.3)
        self.assertEqual(groups[1]["id"], "G2")
        self.assertEqual(groups[1]["count"], 1)
        self.assertEqual(holes[0].group_id, "G1")
        self.assertEqual(holes[1].group_id, "G1")
        self.assertEqual(holes[2].group_id, "G2")
        self.assertIsNone(holes[3].group_id)


class TestSourceDependencyCharacterization(unittest.TestCase):
    def test_downstream_grid_and_mask_operations_do_not_reopen_source(self):
        points = [
            (float(x), float(y), 0.0)
            for y in range(5)
            for x in range(5)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.xyz"
            _write_xyz(source_path, points)
            stats = compute_stats(source_path)
            grid = build_density_grid(source_path, stats, cell_size=1.0)

            source_path.unlink()

            mask = build_mask_from_density(
                grid.density,
                mode="manual",
                manual_threshold=1.0,
            ).mask
            mask = apply_roi_to_mask(mask, grid, (0.0, 0.0, 4.5, 4.5))
            mask = apply_polygon_roi_to_mask(
                mask,
                grid,
                [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
            )
            mask, _stats = apply_mask_edits(
                mask,
                grid,
                [{"mode": "remove", "x": 2.2, "y": 2.2, "radius_mm": 0.1}],
            )
            holes = _detect_holes(mask, grid)
            contour_mask = fill_small_holes(mask, max_hole_area_cells=5)
            contour = build_external_contour(contour_mask, grid)

            self.assertIsInstance(holes, list)
            self.assertGreater(contour.point_count, 0)
            with self.assertRaises(FileNotFoundError):
                compute_stats(source_path)
            with self.assertRaises(FileNotFoundError):
                build_density_grid(source_path, stats, cell_size=1.0)


if __name__ == "__main__":
    unittest.main()
