import math
import unittest

import numpy as np

from experiments.split_merge_prototype import (
    compute_segment_statistics,
    cyclic_indices,
    cyclic_segment_points,
    find_split_candidate,
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


class TestFindSplitCandidate(unittest.TestCase):
    def test_obvious_corner(self):
        contour = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 2.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float64,
        )
        candidate = find_split_candidate(contour, 0, 4)

        self.assertIsNotNone(candidate)
        self.assertEqual(2, candidate.split_index)
        self.assertEqual((2.0, 2.0), candidate.split_point)
        self.assertAlmostEqual(2.0, candidate.distance_to_line_mm)
        self.assertAlmostEqual(
            1.0 + math.sqrt(5.0),
            candidate.arc_distance_from_start_mm,
        )
        self.assertAlmostEqual(
            1.0 + math.sqrt(5.0),
            candidate.arc_distance_to_end_mm,
        )
        self.assertEqual(3, candidate.eligible_points_count)

    def test_perfect_line_returns_first_internal_point(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float64,
        )
        candidate = find_split_candidate(contour, 0, 3)

        self.assertIsNotNone(candidate)
        self.assertEqual(1, candidate.split_index)
        self.assertAlmostEqual(0.0, candidate.distance_to_line_mm)
        self.assertEqual(2, candidate.eligible_points_count)

    def test_endpoint_filter_uses_arc_length_mm(self):
        contour = np.array(
            [[0.0, 0.0], [0.5, 1.0], [2.0, 0.5], [4.0, 0.0]],
            dtype=np.float64,
        )

        unfiltered = find_split_candidate(
            contour,
            0,
            3,
            min_endpoint_arc_length_mm=0.0,
        )
        filtered = find_split_candidate(
            contour,
            0,
            3,
            min_endpoint_arc_length_mm=1.5,
        )

        self.assertIsNotNone(unfiltered)
        self.assertIsNotNone(filtered)
        self.assertEqual(1, unfiltered.split_index)
        self.assertEqual(2, filtered.split_index)
        self.assertEqual(2, unfiltered.eligible_points_count)
        self.assertEqual(1, filtered.eligible_points_count)

    def test_endpoint_filter_can_exclude_all_points(self):
        contour = np.array(
            [[0.0, 0.0], [0.5, 1.0], [2.0, 0.5], [4.0, 0.0]],
            dtype=np.float64,
        )
        self.assertIsNone(
            find_split_candidate(
                contour,
                0,
                3,
                min_endpoint_arc_length_mm=3.0,
            )
        )

    def test_segment_without_internal_points(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        self.assertIsNone(find_split_candidate(contour, 0, 1))

    def test_wrapped_segment_preserves_source_index_and_arc_lengths(self):
        contour = np.array(
            [
                [1.0, 2.0],
                [2.0, 0.0],
                [100.0, 100.0],
                [-1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )
        candidate = find_split_candidate(contour, 3, 1)

        self.assertIsNotNone(candidate)
        self.assertEqual(0, candidate.split_index)
        self.assertEqual((1.0, 2.0), candidate.split_point)
        self.assertAlmostEqual(2.0, candidate.distance_to_line_mm)
        self.assertAlmostEqual(
            1.0 + math.sqrt(5.0),
            candidate.arc_distance_from_start_mm,
        )
        self.assertAlmostEqual(
            math.sqrt(5.0),
            candidate.arc_distance_to_end_mm,
        )
        self.assertEqual(2, candidate.eligible_points_count)

    def test_equal_maxima_select_first_in_traversal_order(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, -1.0], [3.0, 0.0]],
            dtype=np.float64,
        )
        candidate = find_split_candidate(contour, 0, 3)

        self.assertIsNotNone(candidate)
        self.assertEqual(1, candidate.split_index)
        self.assertEqual((1.0, 1.0), candidate.split_point)

    def test_distance_is_to_infinite_line(self):
        contour = np.array(
            [[0.0, 0.0], [2.0, 1.0], [1.0, 0.0]],
            dtype=np.float64,
        )
        candidate = find_split_candidate(contour, 0, 2)

        self.assertIsNotNone(candidate)
        self.assertAlmostEqual(1.0, candidate.distance_to_line_mm)
        self.assertNotAlmostEqual(
            math.sqrt(2.0),
            candidate.distance_to_line_mm,
        )

    def test_reversed_geometry_selects_same_point(self):
        contour = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 2.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float64,
        )
        forward = find_split_candidate(contour, 0, 4)
        reverse = find_split_candidate(contour[::-1].copy(), 0, 4)

        self.assertIsNotNone(forward)
        self.assertIsNotNone(reverse)
        self.assertEqual(forward.split_point, reverse.split_point)
        self.assertAlmostEqual(
            forward.distance_to_line_mm,
            reverse.distance_to_line_mm,
        )
        self.assertAlmostEqual(
            forward.arc_distance_from_start_mm,
            reverse.arc_distance_to_end_mm,
        )
        self.assertAlmostEqual(
            forward.arc_distance_to_end_mm,
            reverse.arc_distance_from_start_mm,
        )

    def test_degenerate_chord_is_rejected(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(ValueError, "degenerate chord"):
            find_split_candidate(contour, 0, 2)

    def test_equal_endpoint_indices_are_rejected(self):
        contour = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "different endpoint indices"):
            find_split_candidate(contour, 0, 0)

    def test_invalid_endpoint_margin_is_rejected(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
            dtype=np.float64,
        )
        for value in (-1.0, np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    find_split_candidate(
                        contour,
                        0,
                        2,
                        min_endpoint_arc_length_mm=value,
                    )

        with self.assertRaises(TypeError):
            find_split_candidate(
                contour,
                0,
                2,
                min_endpoint_arc_length_mm=True,
            )


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
