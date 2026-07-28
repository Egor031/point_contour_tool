import math
import unittest

import numpy as np

from experiments.split_merge_prototype import (
    SegmentStatistics,
    SplitDecisionPolicy,
    SplitEvaluation,
    compute_segment_statistics,
    cyclic_indices,
    cyclic_segment_points,
    decide_split,
    evaluate_split,
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


class TestEvaluateSplit(unittest.TestCase):
    def test_right_angle_reduces_error_to_zero(self):
        contour = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=np.float64,
        )
        evaluation = evaluate_split(contour, 0, 6, 3)

        self.assertGreater(evaluation.parent_sse_mm2, 0.0)
        self.assertAlmostEqual(
            0.0,
            evaluation.left_statistics.mean_squared_error_mm2,
        )
        self.assertAlmostEqual(
            0.0,
            evaluation.right_statistics.mean_squared_error_mm2,
        )
        self.assertAlmostEqual(0.0, evaluation.post_split_sse_mm2)
        self.assertAlmostEqual(0.0, evaluation.post_split_mse_mm2)
        self.assertAlmostEqual(0.0, evaluation.post_split_rms_mm)
        self.assertAlmostEqual(
            evaluation.parent_sse_mm2,
            evaluation.sse_reduction_mm2,
        )
        self.assertAlmostEqual(1.0, evaluation.sse_reduction_fraction)
        self.assertAlmostEqual(0.0, evaluation.post_split_max_error_mm)
        self.assertGreater(evaluation.max_error_reduction_mm, 0.0)

    def test_perfect_line_has_undefined_reduction_fraction(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float64,
        )
        evaluation = evaluate_split(contour, 0, 3, 1)

        self.assertAlmostEqual(0.0, evaluation.parent_sse_mm2)
        self.assertAlmostEqual(0.0, evaluation.post_split_sse_mm2)
        self.assertAlmostEqual(0.0, evaluation.sse_reduction_mm2)
        self.assertIsNone(evaluation.sse_reduction_fraction)
        self.assertAlmostEqual(
            0.0,
            evaluation.parent_statistics.max_error_mm,
        )
        self.assertAlmostEqual(0.0, evaluation.post_split_max_error_mm)
        self.assertAlmostEqual(0.0, evaluation.max_error_reduction_mm)

    def test_post_split_mse_is_weighted_by_observation_count(self):
        contour = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 2.0],
                [5.0, 0.0],
                [6.0, 0.0],
            ],
            dtype=np.float64,
        )
        evaluation = evaluate_split(contour, 0, 6, 2)
        left = evaluation.left_statistics
        right = evaluation.right_statistics

        self.assertNotEqual(
            left.internal_points_count,
            right.internal_points_count,
        )
        self.assertNotAlmostEqual(
            left.mean_squared_error_mm2,
            right.mean_squared_error_mm2,
        )
        expected_post_sse = (
            left.mean_squared_error_mm2 * left.internal_points_count
            + right.mean_squared_error_mm2 * right.internal_points_count
        )
        expected_post_mse = (
            expected_post_sse
            / evaluation.parent_statistics.internal_points_count
        )
        simple_mean = (
            left.mean_squared_error_mm2
            + right.mean_squared_error_mm2
        ) / 2.0

        self.assertAlmostEqual(
            expected_post_sse,
            evaluation.post_split_sse_mm2,
        )
        self.assertAlmostEqual(
            expected_post_mse,
            evaluation.post_split_mse_mm2,
        )
        self.assertNotAlmostEqual(
            simple_mean,
            evaluation.post_split_mse_mm2,
        )

    def test_wrapped_segment_uses_positive_cyclic_ranges(self):
        contour = np.array(
            [
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
                [9.0, 9.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float64,
        )
        evaluation = evaluate_split(contour, 4, 2, 7)

        self.assertEqual(4, evaluation.segment_start_index)
        self.assertEqual(2, evaluation.segment_end_index)
        self.assertEqual(7, evaluation.split_index)
        self.assertEqual(7, evaluation.parent_statistics.range_points_count)
        self.assertEqual(4, evaluation.left_statistics.range_points_count)
        self.assertEqual(4, evaluation.right_statistics.range_points_count)
        self.assertGreater(evaluation.parent_sse_mm2, 0.0)
        self.assertAlmostEqual(0.0, evaluation.post_split_sse_mm2)
        self.assertAlmostEqual(
            evaluation.parent_sse_mm2,
            evaluation.sse_reduction_mm2,
        )

    def test_reversed_geometry_has_identical_metrics(self):
        contour = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=np.float64,
        )
        forward = evaluate_split(contour, 0, 6, 3)
        reverse = evaluate_split(contour[::-1].copy(), 0, 6, 3)

        for field in (
            "parent_sse_mm2",
            "post_split_sse_mm2",
            "post_split_mse_mm2",
            "post_split_rms_mm",
            "sse_reduction_mm2",
            "post_split_max_error_mm",
            "max_error_reduction_mm",
        ):
            with self.subTest(field=field):
                self.assertAlmostEqual(
                    getattr(forward, field),
                    getattr(reverse, field),
                )

    def test_split_next_to_endpoint_allows_empty_child_interior(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float64,
        )

        for split_index, empty_child in ((1, "left"), (2, "right")):
            with self.subTest(split_index=split_index):
                evaluation = evaluate_split(contour, 0, 3, split_index)
                child = getattr(evaluation, f"{empty_child}_statistics")
                self.assertEqual(0, child.internal_points_count)
                self.assertAlmostEqual(
                    0.0,
                    child.mean_squared_error_mm2,
                )
                self.assertAlmostEqual(0.0, child.rms_error_mm)
                self.assertAlmostEqual(0.0, child.max_error_mm)
                self.assertTrue(np.isfinite(evaluation.post_split_mse_mm2))

    def test_split_equal_to_endpoint_is_rejected(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
            dtype=np.float64,
        )
        for split_index in (0, 2):
            with self.subTest(split_index=split_index):
                with self.assertRaisesRegex(ValueError, "internal point"):
                    evaluate_split(contour, 0, 2, split_index)

    def test_split_outside_wrapped_segment_is_rejected(self):
        contour = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(ValueError, "internal point"):
            evaluate_split(contour, 4, 1, 2)

    def test_equal_segment_endpoints_are_rejected(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
            dtype=np.float64,
        )
        with self.assertRaisesRegex(ValueError, "different endpoint indices"):
            evaluate_split(contour, 1, 1, 2)

    def test_find_candidate_then_evaluate_split(self):
        contour = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=np.float64,
        )
        candidate = find_split_candidate(contour, 0, 6)

        self.assertIsNotNone(candidate)
        evaluation = evaluate_split(
            contour,
            candidate.segment_start_index,
            candidate.segment_end_index,
            candidate.split_index,
        )
        self.assertEqual(3, candidate.split_index)
        self.assertGreater(evaluation.sse_reduction_mm2, 0.0)
        self.assertGreater(evaluation.max_error_reduction_mm, 0.0)


class TestSplitDecisionPolicy(unittest.TestCase):
    def test_valid_configuration(self):
        policy = SplitDecisionPolicy(
            min_child_arc_length_mm=2.0,
            parent_rms_tolerance_mm=0.1,
            min_rms_reduction_fraction=0.25,
            corner_penalty_rms_mm=0.05,
        )
        self.assertEqual(2.0, policy.min_child_arc_length_mm)
        self.assertEqual(0.1, policy.parent_rms_tolerance_mm)
        self.assertEqual(0.25, policy.min_rms_reduction_fraction)
        self.assertEqual(0.05, policy.corner_penalty_rms_mm)

    def test_negative_values_are_rejected(self):
        valid = {
            "min_child_arc_length_mm": 1.0,
            "parent_rms_tolerance_mm": 0.1,
            "min_rms_reduction_fraction": 0.25,
            "corner_penalty_rms_mm": 0.05,
        }
        for name in valid:
            values = valid.copy()
            values[name] = -0.01
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    SplitDecisionPolicy(**values)

    def test_non_finite_values_are_rejected(self):
        valid = {
            "min_child_arc_length_mm": 1.0,
            "parent_rms_tolerance_mm": 0.1,
            "min_rms_reduction_fraction": 0.25,
            "corner_penalty_rms_mm": 0.05,
        }
        for name in valid:
            for value in (np.nan, np.inf, -np.inf):
                values = valid.copy()
                values[name] = value
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        SplitDecisionPolicy(**values)

    def test_bool_values_are_rejected(self):
        valid = {
            "min_child_arc_length_mm": 1.0,
            "parent_rms_tolerance_mm": 0.1,
            "min_rms_reduction_fraction": 0.25,
            "corner_penalty_rms_mm": 0.05,
        }
        for name in valid:
            values = valid.copy()
            values[name] = True
            with self.subTest(name=name):
                with self.assertRaises(TypeError):
                    SplitDecisionPolicy(**values)

    def test_non_numeric_values_are_rejected(self):
        with self.assertRaises(TypeError):
            SplitDecisionPolicy(
                min_child_arc_length_mm="1",
                parent_rms_tolerance_mm=0.1,
                min_rms_reduction_fraction=0.25,
                corner_penalty_rms_mm=0.05,
            )

    def test_reduction_fraction_above_one_is_rejected(self):
        with self.assertRaises(ValueError):
            SplitDecisionPolicy(
                min_child_arc_length_mm=1.0,
                parent_rms_tolerance_mm=0.1,
                min_rms_reduction_fraction=1.01,
                corner_penalty_rms_mm=0.05,
            )

    def test_reduction_fraction_boundaries_are_valid(self):
        for fraction in (0.0, 1.0):
            with self.subTest(fraction=fraction):
                policy = SplitDecisionPolicy(
                    min_child_arc_length_mm=0.0,
                    parent_rms_tolerance_mm=0.0,
                    min_rms_reduction_fraction=fraction,
                    corner_penalty_rms_mm=0.0,
                )
                self.assertEqual(fraction, policy.min_rms_reduction_fraction)


class TestDecideSplit(unittest.TestCase):
    @staticmethod
    def _policy(**overrides):
        values = {
            "min_child_arc_length_mm": 0.0,
            "parent_rms_tolerance_mm": 0.0,
            "min_rms_reduction_fraction": 0.0,
            "corner_penalty_rms_mm": 0.0,
        }
        values.update(overrides)
        return SplitDecisionPolicy(**values)

    @staticmethod
    def _statistics(
        start_index,
        end_index,
        arc_length_mm,
        rms_error_mm,
    ):
        return SegmentStatistics(
            start_index=start_index,
            end_index=end_index,
            range_points_count=3,
            internal_points_count=1,
            arc_length_mm=arc_length_mm,
            chord_length_mm=arc_length_mm,
            mean_squared_error_mm2=rms_error_mm**2,
            rms_error_mm=rms_error_mm,
            max_error_mm=rms_error_mm,
        )

    @classmethod
    def _evaluation(
        cls,
        parent_rms,
        post_split_rms,
        left_arc=10.0,
        right_arc=10.0,
    ):
        parent = cls._statistics(0, 4, left_arc + right_arc, parent_rms)
        left = cls._statistics(0, 2, left_arc, post_split_rms)
        right = cls._statistics(2, 4, right_arc, post_split_rms)
        return SplitEvaluation(
            segment_start_index=0,
            segment_end_index=4,
            split_index=2,
            parent_statistics=parent,
            left_statistics=left,
            right_statistics=right,
            parent_sse_mm2=parent_rms**2,
            post_split_sse_mm2=post_split_rms**2,
            post_split_mse_mm2=post_split_rms**2,
            post_split_rms_mm=post_split_rms,
            sse_reduction_mm2=parent_rms**2 - post_split_rms**2,
            sse_reduction_fraction=None,
            post_split_max_error_mm=post_split_rms,
            max_error_reduction_mm=parent_rms - post_split_rms,
        )

    @staticmethod
    def _right_angle_contour():
        return np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
            ],
            dtype=np.float64,
        )

    def test_right_angle_is_accepted_and_fields_are_consistent(self):
        contour = self._right_angle_contour()
        candidate = find_split_candidate(contour, 0, 6)
        self.assertIsNotNone(candidate)
        evaluation = evaluate_split(
            contour,
            candidate.segment_start_index,
            candidate.segment_end_index,
            candidate.split_index,
        )
        policy = self._policy(
            min_child_arc_length_mm=2.0,
            parent_rms_tolerance_mm=0.1,
            min_rms_reduction_fraction=0.5,
            corner_penalty_rms_mm=0.1,
        )
        decision = decide_split(evaluation, policy)

        self.assertTrue(decision.accepted)
        self.assertEqual("accepted", decision.reason)
        self.assertGreater(decision.rms_reduction_mm, 0.0)
        self.assertGreater(decision.rms_reduction_fraction, 0.0)
        self.assertGreater(decision.net_gain_mm, 0.0)
        self.assertAlmostEqual(
            decision.parent_rms_mm - decision.post_split_rms_mm,
            decision.rms_reduction_mm,
        )
        self.assertAlmostEqual(
            decision.rms_reduction_mm / decision.parent_rms_mm,
            decision.rms_reduction_fraction,
        )
        self.assertAlmostEqual(
            decision.rms_reduction_mm - decision.corner_penalty_rms_mm,
            decision.net_gain_mm,
        )
        self.assertIs(evaluation, decision.evaluation)

    def test_perfect_line_is_within_tolerance(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float64,
        )
        evaluation = evaluate_split(contour, 0, 3, 1)
        decision = decide_split(evaluation, self._policy())

        self.assertFalse(decision.accepted)
        self.assertEqual("parent_within_tolerance", decision.reason)
        self.assertIsNone(decision.rms_reduction_fraction)

    def test_left_child_too_short(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [4.0, 0.0]],
            dtype=np.float64,
        )
        evaluation = evaluate_split(contour, 0, 3, 1)
        policy = self._policy(min_child_arc_length_mm=1.5)
        decision = decide_split(evaluation, policy)

        self.assertFalse(decision.accepted)
        self.assertEqual("left_child_too_short", decision.reason)

    def test_right_child_too_short(self):
        contour = np.array(
            [[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0]],
            dtype=np.float64,
        )
        evaluation = evaluate_split(contour, 0, 3, 2)
        policy = self._policy(min_child_arc_length_mm=1.5)
        decision = decide_split(evaluation, policy)

        self.assertFalse(decision.accepted)
        self.assertEqual("right_child_too_short", decision.reason)

    def test_short_child_precedes_parent_tolerance_reason(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float64,
        )
        evaluation = evaluate_split(contour, 0, 3, 1)
        policy = self._policy(
            min_child_arc_length_mm=1.5,
            parent_rms_tolerance_mm=1.0,
        )
        decision = decide_split(evaluation, policy)

        self.assertEqual("left_child_too_short", decision.reason)

    def test_no_rms_improvement(self):
        for post_split_rms in (1.0, 1.2):
            with self.subTest(post_split_rms=post_split_rms):
                evaluation = self._evaluation(1.0, post_split_rms)
                decision = decide_split(evaluation, self._policy())
                self.assertFalse(decision.accepted)
                self.assertEqual("no_rms_improvement", decision.reason)

    def test_relative_improvement_too_small(self):
        evaluation = self._evaluation(1.0, 0.8)
        policy = self._policy(min_rms_reduction_fraction=0.3)
        decision = decide_split(evaluation, policy)

        self.assertFalse(decision.accepted)
        self.assertEqual(
            "relative_improvement_too_small",
            decision.reason,
        )

    def test_corner_penalty_must_be_strictly_overcome(self):
        evaluation = self._evaluation(1.0, 0.75)
        for penalty in (0.25, 0.3):
            with self.subTest(penalty=penalty):
                policy = self._policy(
                    min_rms_reduction_fraction=0.1,
                    corner_penalty_rms_mm=penalty,
                )
                decision = decide_split(evaluation, policy)
                self.assertFalse(decision.accepted)
                self.assertEqual(
                    "corner_penalty_not_overcome",
                    decision.reason,
                )

    def test_relative_improvement_equality_is_accepted(self):
        evaluation = self._evaluation(1.0, 0.75)
        policy = self._policy(
            min_rms_reduction_fraction=0.25,
            corner_penalty_rms_mm=0.1,
        )
        decision = decide_split(evaluation, policy)

        self.assertTrue(decision.accepted)
        self.assertEqual("accepted", decision.reason)

    def test_child_arc_length_equality_is_accepted(self):
        evaluation = self._evaluation(
            1.0,
            0.5,
            left_arc=2.0,
            right_arc=2.0,
        )
        policy = self._policy(
            min_child_arc_length_mm=2.0,
            corner_penalty_rms_mm=0.1,
        )
        decision = decide_split(evaluation, policy)

        self.assertTrue(decision.accepted)
        self.assertEqual("accepted", decision.reason)

    def test_weak_noise_is_rejected_by_corner_penalty(self):
        contour = np.array(
            [[0.0, 0.0], [1.0, 0.05], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float64,
        )
        evaluation = evaluate_split(contour, 0, 3, 1)
        policy = self._policy(
            parent_rms_tolerance_mm=0.0,
            min_rms_reduction_fraction=0.0,
            corner_penalty_rms_mm=0.02,
        )
        decision = decide_split(evaluation, policy)

        self.assertGreater(decision.rms_reduction_mm, 0.0)
        self.assertEqual("corner_penalty_not_overcome", decision.reason)

    def test_reversed_geometry_has_identical_decision(self):
        contour = self._right_angle_contour()
        forward = evaluate_split(contour, 0, 6, 3)
        reverse = evaluate_split(contour[::-1].copy(), 0, 6, 3)
        policy = self._policy(
            min_child_arc_length_mm=2.0,
            parent_rms_tolerance_mm=0.1,
            min_rms_reduction_fraction=0.5,
            corner_penalty_rms_mm=0.1,
        )
        forward_decision = decide_split(forward, policy)
        reverse_decision = decide_split(reverse, policy)

        self.assertEqual(
            forward_decision.accepted,
            reverse_decision.accepted,
        )
        self.assertEqual(forward_decision.reason, reverse_decision.reason)
        for field in (
            "parent_rms_mm",
            "post_split_rms_mm",
            "rms_reduction_mm",
            "rms_reduction_fraction",
            "net_gain_mm",
        ):
            with self.subTest(field=field):
                self.assertAlmostEqual(
                    getattr(forward_decision, field),
                    getattr(reverse_decision, field),
                )

    def test_invalid_argument_types(self):
        evaluation = self._evaluation(1.0, 0.5)
        policy = self._policy()

        with self.assertRaises(TypeError):
            decide_split(object(), policy)
        with self.assertRaises(TypeError):
            decide_split(evaluation, object())


if __name__ == "__main__":
    unittest.main()
