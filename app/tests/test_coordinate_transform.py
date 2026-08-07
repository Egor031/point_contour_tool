from __future__ import annotations

import sys
import unittest
from pathlib import Path


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.coordinate_transform import CoordinateTransform  # noqa: E402


class TestCoordinateTransform(unittest.TestCase):
    def test_preview_without_resize_matches_grid_cell_centers(self):
        transform = CoordinateTransform(
            grid_min_x=0.0,
            grid_min_y=0.0,
            cell_size=1.0,
            grid_width=100,
            grid_height=50,
            preview_width=100,
            preview_height=50,
        )

        self.assertEqual(transform.grid_to_preview(0.0, 0.0), (0.0, 49.0))
        self.assertEqual(transform.grid_to_preview(99.0, 49.0), (99.0, 0.0))
        self.assertEqual(transform.preview_to_grid(0.0, 49.0), (0.0, 0.0))

    def test_preview_downsampled_by_two_uses_opencv_pixel_centers(self):
        transform = CoordinateTransform(
            grid_min_x=0.0,
            grid_min_y=0.0,
            cell_size=1.0,
            grid_width=200,
            grid_height=100,
            preview_width=100,
            preview_height=50,
        )

        self.assertEqual(transform.grid_to_preview(0.5, 98.5), (0.0, 0.0))
        self.assertEqual(transform.preview_to_grid(0.0, 0.0), (0.5, 98.5))
        self.assertEqual(transform.world_to_preview(1.0, 99.0), (0.0, 0.0))
        self.assertEqual(transform.preview_to_world(0.0, 0.0), (1.0, 99.0))

    def test_unequal_resize_round_trip_has_no_systematic_offset(self):
        transform = CoordinateTransform(
            grid_min_x=0.0,
            grid_min_y=0.0,
            cell_size=1.0,
            grid_width=101,
            grid_height=77,
            preview_width=67,
            preview_height=51,
        )

        for grid_point in ((0.0, 0.0), (25.25, 13.75), (50.0, 38.0), (100.0, 76.0)):
            with self.subTest(grid_point=grid_point):
                restored = transform.preview_to_grid(
                    *transform.grid_to_preview(*grid_point)
                )
                self.assertAlmostEqual(restored[0], grid_point[0], places=12)
                self.assertAlmostEqual(restored[1], grid_point[1], places=12)

        radius_x, radius_y = transform.world_radius_to_preview(10.0)
        self.assertAlmostEqual(radius_x, 10.0 * 67.0 / 101.0)
        self.assertAlmostEqual(radius_y, 10.0 * 51.0 / 77.0)
        self.assertNotAlmostEqual(radius_x, radius_y)

    def test_nonzero_world_origin_and_cell_size(self):
        transform = CoordinateTransform(
            grid_min_x=-125.5,
            grid_min_y=48.25,
            cell_size=0.8,
            grid_width=20,
            grid_height=10,
            preview_width=10,
            preview_height=5,
        )

        self.assertEqual(
            transform.grid_cell_center_to_world(0, 0),
            (-125.1, 48.65),
        )
        grid = transform.world_to_grid(-121.1, 52.65)
        self.assertAlmostEqual(grid[0], 5.0)
        self.assertAlmostEqual(grid[1], 5.0)

    def test_first_cell_center_preserves_production_semantics(self):
        transform = CoordinateTransform(
            grid_min_x=10.0,
            grid_min_y=-3.0,
            cell_size=0.5,
            grid_width=3,
            grid_height=4,
            preview_width=3,
            preview_height=4,
        )

        self.assertEqual(transform.grid_cell_center_to_world(0, 0), (10.25, -2.75))

    def test_world_preview_world_round_trip_is_continuous(self):
        transform = CoordinateTransform(
            grid_min_x=-125.5,
            grid_min_y=48.25,
            cell_size=0.8,
            grid_width=101,
            grid_height=77,
            preview_width=67,
            preview_height=51,
        )

        for world_point in ((-125.5, 48.25), (-100.125, 63.75), (-44.7, 109.85)):
            with self.subTest(world_point=world_point):
                restored = transform.preview_to_world(
                    *transform.world_to_preview(*world_point)
                )
                self.assertAlmostEqual(restored[0], world_point[0], places=11)
                self.assertAlmostEqual(restored[1], world_point[1], places=11)

    def test_same_world_point_tracks_resized_preview_position(self):
        full = CoordinateTransform(
            grid_min_x=10.0,
            grid_min_y=20.0,
            cell_size=0.5,
            grid_width=400,
            grid_height=200,
            preview_width=400,
            preview_height=200,
        )
        downsampled = CoordinateTransform(
            grid_min_x=10.0,
            grid_min_y=20.0,
            cell_size=0.5,
            grid_width=400,
            grid_height=200,
            preview_width=100,
            preview_height=50,
        )
        world_point = (60.25, 45.25)

        full_position = full.world_to_preview(*world_point)
        small_position = downsampled.world_to_preview(*world_point)

        self.assertAlmostEqual(
            (full_position[0] + 0.5) / full.preview_width,
            (small_position[0] + 0.5) / downsampled.preview_width,
        )
        self.assertAlmostEqual(
            (full_position[1] + 0.5) / full.preview_height,
            (small_position[1] + 0.5) / downsampled.preview_height,
        )
        self.assertNotAlmostEqual(small_position[0], full.world_to_grid(*world_point)[0])

    def test_preview_pixel_center_click_maps_to_expected_world_position(self):
        transform = CoordinateTransform(
            grid_min_x=100.0,
            grid_min_y=-50.0,
            cell_size=1.0,
            grid_width=200,
            grid_height=100,
            preview_width=100,
            preview_height=50,
        )

        self.assertEqual(transform.preview_to_world(10.0, 20.0), (121.0, 9.0))


if __name__ == "__main__":
    unittest.main()
