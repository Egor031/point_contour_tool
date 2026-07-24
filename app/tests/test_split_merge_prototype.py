import math
import unittest

import numpy as np

from experiments.split_merge_prototype import (
    compute_segment_statistics,
    cyclic_indices,
    cyclic_segment_points,
    point_to_line_distances,
    point_to_segment_distances,
)


class TestCyclicIndices(unittest.TestCase):
    def test_regular_range(self):
        self.assertEqual([2, 3, 4, 5], cyclic_indices(10, 2, 5))

    def test_wrapped_range(self):
        self.assertEqual([8, 9, 0, 1, 2], cyclic_indices(10, 8, 2))

    def test_equal_endpoints(self):
        self.assertEqual([4], cyclic_indices(10, 4, 4))

    def test_invalid_indices(self):
        for start, end in [(-1, 2), (10, 2), (2, -1), (2, 10)]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(IndexError):
                    cyclic_indices(10, start, end)

    def test_invalid_point_count(self):
        for point_count in (0, -1):
            with self.subTest(point_count=point_count):
                with self.assertRaises(ValueError):
                    cyclic_indices(point_count, 0, 0)

    def test_cyclic_segment_points_returns_wrapped_copy(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        found = cyclic_segment_points(contour, 3, 1)
        expected = np.array(
            [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
            dtype=np.float64,
        )
        np.testing.assert_allclose(found, expected)
        found[0] = 100.0
        np.testing.assert_allclose(contour[3], [0.0, 1.0])


class TestPointDistances(unittest.TestCase):
    def test_point_to_segment_distances(self):
        points = np.array(
            [[1.0, 1.0], [-1.0, 0.0], [3.0, 0.0], [1.0, 0.0]],
            dtype=np.float64,
        )
        distances = point_to_segment_distances(
            points,
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([2.0, 0.0], dtype=np.float64),
        )
        np.testing.assert_allclose(distances, [1.0, 1.0, 1.0, 0.0])

    def test_degenerate_segment(self):
        points = np.array([[4.0, 5.0], [1.0, 1.0]], dtype=np.float64)
        endpoint = np.array([1.0, 1.0], dtype=np.float64)
        distances = point_to_segment_distances(points, endpoint, endpoint)
        np.testing.assert_allclose(distances, [5.0, 0.0])

    def test_point_to_horizontal_line(self):
        points = np.array(
            [[1.0, 2.0], [-3.0, -4.0], [5.0, 0.0]],
            dtype=np.float64,
        )
        distances = point_to_line_distances(
            points,
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([2.0, 0.0], dtype=np.float64),
        )
        np.testing.assert_allclose(distances, [2.0, 4.0, 0.0])

    def test_point_to_vertical_line(self):
        points = np.array(
            [[2.0, 1.0], [-1.0, 4.0], [0.0, -2.0]],
            dtype=np.float64,
        )
        distances = point_to_line_distances(
            points,
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([0.0, 3.0], dtype=np.float64),
        )
        np.testing.assert_allclose(distances, [2.0, 1.0, 0.0])

    def test_degenerate_line_is_rejected(self):
        points = np.array([[1.0, 2.0]], dtype=np.float64)
        endpoint = np.array([0.0, 0.0], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "non-degenerate"):
            point_to_line_distances(points, endpoint, endpoint)


class TestSegmentStatistics(unittest.TestCase):
    def test_perfect_line(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float64,
        )
        stats = compute_segment_statistics(contour, 0, 3)

        self.assertEqual(4, stats.range_points_count)
        self.assertEqual(2, stats.internal_points_count)
        self.assertAlmostEqual(3.0, stats.arc_length_mm)
        self.assertAlmostEqual(3.0, stats.chord_length_mm)
        self.assertAlmostEqual(0.0, stats.mean_squared_error_mm2)
        self.assertAlmostEqual(0.0, stats.rms_error_mm)
        self.assertAlmostEqual(0.0, stats.max_error_mm)

    def test_offset_internal_point(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
            dtype=np.float64,
        )
        stats = compute_segment_statistics(contour, 0, 2)

        self.assertAlmostEqual(2.0 * math.sqrt(2.0), stats.arc_length_mm)
        self.assertAlmostEqual(2.0, stats.chord_length_mm)
        self.assertAlmostEqual(1.0, stats.mean_squared_error_mm2)
        self.assertAlmostEqual(1.0, stats.rms_error_mm)
        self.assertAlmostEqual(1.0, stats.max_error_mm)

    def test_wrapped_segment(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        stats = compute_segment_statistics(contour, 3, 1)

        self.assertEqual(3, stats.range_points_count)
        self.assertEqual(1, stats.internal_points_count)
        self.assertAlmostEqual(2.0, stats.arc_length_mm)
        self.assertAlmostEqual(math.sqrt(2.0), stats.chord_length_mm)
        self.assertAlmostEqual(0.5, stats.mean_squared_error_mm2)
        self.assertAlmostEqual(math.sqrt(0.5), stats.rms_error_mm)
        self.assertAlmostEqual(math.sqrt(0.5), stats.max_error_mm)

    def test_no_internal_points(self):
        contour = np.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]],
            dtype=np.float64,
        )
        stats = compute_segment_statistics(contour, 0, 1)

        self.assertEqual(2, stats.range_points_count)
        self.assertEqual(0, stats.internal_points_count)
        self.assertAlmostEqual(2.0, stats.arc_length_mm)
        self.assertAlmostEqual(2.0, stats.chord_length_mm)
        self.assertAlmostEqual(0.0, stats.mean_squared_error_mm2)
        self.assertAlmostEqual(0.0, stats.rms_error_mm)
        self.assertAlmostEqual(0.0, stats.max_error_mm)

    def test_same_geometry_in_reverse_order(self):
        contour = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float64,
        )
        forward = compute_segment_statistics(contour, 0, 2)
        reverse = compute_segment_statistics(contour[::-1].copy(), 2, 4)

        for field in (
            "arc_length_mm",
            "chord_length_mm",
            "mean_squared_error_mm2",
            "rms_error_mm",
            "max_error_mm",
        ):
            with self.subTest(field=field):
                self.assertAlmostEqual(
                    getattr(forward, field),
                    getattr(reverse, field),
                )

    def test_degenerate_chord_uses_endpoint_distance(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 2.0]],
            dtype=np.float64,
        )
        stats = compute_segment_statistics(contour, 0, 2)

        self.assertAlmostEqual(0.0, stats.chord_length_mm)
        self.assertAlmostEqual(1.0, stats.mean_squared_error_mm2)
        self.assertAlmostEqual(1.0, stats.rms_error_mm)
        self.assertAlmostEqual(1.0, stats.max_error_mm)

    def test_equal_endpoint_indices_are_rejected(self):
        contour = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "different endpoint indices"):
            compute_segment_statistics(contour, 0, 0)

    def test_invalid_contour_shape_is_rejected(self):
        invalid_contours = (
            np.array([0.0, 1.0], dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            np.zeros((1, 2), dtype=np.float64),
        )
        for contour in invalid_contours:
            with self.subTest(shape=contour.shape):
                with self.assertRaises(ValueError):
                    compute_segment_statistics(contour, 0, 1)

    def test_non_finite_contour_is_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            contour = np.array([[0.0, 0.0], [1.0, value]], dtype=np.float64)
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    compute_segment_statistics(contour, 0, 1)


if __name__ == "__main__":
    unittest.main()
