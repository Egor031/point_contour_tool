from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.working_boundary_cloud import point_is_near_closed_contour  # noqa: E402
from app.core.xyz_reader import iter_xyz_points  # noqa: E402
from app.services.boundary_extraction import (  # noqa: E402
    OuterBoundarySegmentIndex,
    SourceChangedDuringExtractionError,
    _XYZBlockAccumulator,
    _point_matches_candidate_segments,
    extract_outer_boundary_points,
)


def _square(size: float = 10.0) -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]],
        dtype=np.float64,
    )


def _write_xyz(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.write_text(
        "".join(f"{x:.17g} {y:.17g} {z:.17g}\n" for x, y, z in points),
        encoding="utf-8",
    )


def _brute_force_points(
    points: list[tuple[float, float, float]],
    contour: np.ndarray,
    search_width: float,
) -> np.ndarray:
    selected = [
        point
        for point in points
        if point_is_near_closed_contour(point[:2], contour, search_width)
    ]
    if not selected:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(selected, dtype=np.float64)


def _subdivided_rectangle(size: int = 50) -> np.ndarray:
    points: list[tuple[float, float]] = []
    points.extend((float(x), 0.0) for x in range(size))
    points.extend((float(size), float(y)) for y in range(size))
    points.extend((float(x), float(size)) for x in range(size, 0, -1))
    points.extend((0.0, float(y)) for y in range(size, 0, -1))
    return np.asarray(points, dtype=np.float64)


class OuterExtractionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.source_path = Path(self._temporary_directory.name) / "source.xyz"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def extract(
        self,
        points: list[tuple[float, float, float]],
        contour: np.ndarray | None = None,
        search_width: float = 0.5,
        **kwargs: object,
    ):
        _write_xyz(self.source_path, points)
        return extract_outer_boundary_points(
            self.source_path,
            _square() if contour is None else contour,
            search_width,
            **kwargs,
        )


class TestOuterBoundaryExtractionGeometry(OuterExtractionTestCase):
    def test_selects_near_segment_and_closing_segment_but_rejects_far_point(self):
        points = [
            (5.0, 0.25, 101.125),
            (5.0, 5.0, 202.25),
            (-0.25, 5.0, -303.5),
        ]
        result = self.extract(points)

        np.testing.assert_array_equal(
            result.cloud.points_xyz,
            np.array([points[0], points[2]], dtype=np.float64),
        )
        self.assertEqual(result.statistics.source_points_seen, 3)
        self.assertEqual(result.statistics.accepted_points, 2)

    def test_point_near_two_segments_is_stored_once(self):
        point = (0.2, 0.2, 77.0)
        result = self.extract([point])
        np.testing.assert_array_equal(result.cloud.points_xyz, [point])
        self.assertEqual(result.statistics.accepted_points, 1)

    def test_preserves_source_order_and_original_z(self):
        points = [
            (2.0, 0.1, 1.2345678901234567),
            (5.0, 5.0, 999.0),
            (10.1, 8.0, -12345.125),
            (7.0, 9.9, 0.00000000000000025),
        ]
        result = self.extract(points, search_width=0.2)
        expected = np.asarray([points[0], points[2], points[3]], dtype=np.float64)
        np.testing.assert_array_equal(result.cloud.points_xyz, expected)
        np.testing.assert_array_equal(result.cloud.points_xyz[:, 2], expected[:, 2])

    def test_search_width_boundary_is_inclusive_in_world_units(self):
        points = [
            (5.0, 0.125, 1.0),
            (5.0, 0.1250000001, 2.0),
        ]
        result = self.extract(points, search_width=0.125)
        np.testing.assert_array_equal(result.cloud.points_xyz, [points[0]])

    def test_multiple_segments_in_one_bucket_are_checked(self):
        index = OuterBoundarySegmentIndex(_square(), 0.5, tile_size=100.0)
        candidates = index.candidate_segment_ids((0.2, 0.2))
        self.assertGreaterEqual(len(candidates), 2)

        result = self.extract([(0.2, 0.2, 1.0)], tile_size=100.0)
        self.assertEqual(result.statistics.accepted_points, 1)

    def test_empty_bucket_skips_exact_distance_checks(self):
        result = self.extract([(1000.0, 1000.0, 3.0)])
        self.assertEqual(result.statistics.broad_phase_candidate_checks, 0)
        self.assertEqual(result.statistics.exact_segment_checks, 0)
        self.assertEqual(result.cloud.points_xyz.shape, (0, 3))


class TestOuterBoundaryExtractionReferenceEquivalence(OuterExtractionTestCase):
    def test_fast_result_matches_brute_force_for_synthetic_contours(self):
        contours = (
            _square(),
            np.array([[0.0, 0.0], [12.0, 3.0], [4.0, 11.0]], dtype=np.float64),
            np.array(
                [[-8.0, -3.0], [9.0, 7.0], [12.0, 13.0], [-5.0, 5.0]],
                dtype=np.float64,
            ),
        )
        points = [
            (float(x), float(y), float(x * 100 + y))
            for x in np.linspace(-10.0, 15.0, 18)
            for y in np.linspace(-8.0, 15.0, 17)
        ]

        for contour in contours:
            with self.subTest(contour=contour):
                result = self.extract(points, contour, 0.75)
                expected = _brute_force_points(points, contour, 0.75)
                np.testing.assert_array_equal(result.cloud.points_xyz, expected)

    def test_diagonal_and_long_segments_have_no_false_negatives(self):
        contour = np.array(
            [[-1000.0, -500.0], [1000.0, 500.0], [1000.0, 510.0], [-1000.0, -490.0]],
            dtype=np.float64,
        )
        points = [
            (-750.0, -374.7, 1.0),
            (0.0, 0.4, 2.0),
            (750.0, 375.3, 3.0),
            (0.0, 5.0, 4.0),
            (2000.0, 2000.0, 5.0),
        ]
        result = self.extract(points, contour, 0.5)
        expected = _brute_force_points(points, contour, 0.5)
        np.testing.assert_array_equal(result.cloud.points_xyz, expected)

    def test_candidate_segment_order_does_not_change_membership(self):
        index = OuterBoundarySegmentIndex(_square(), 0.5, tile_size=100.0)
        candidates = index.candidate_segment_ids((0.2, 0.2))
        forward, _ = _point_matches_candidate_segments((0.2, 0.2), index, candidates)
        reverse, _ = _point_matches_candidate_segments(
            (0.2, 0.2),
            index,
            tuple(reversed(candidates)),
        )
        self.assertTrue(forward)
        self.assertEqual(forward, reverse)

    def test_repeated_extraction_is_deterministic(self):
        points = [(x / 10.0, 0.1, float(x)) for x in range(100)]
        _write_xyz(self.source_path, points)
        first = extract_outer_boundary_points(self.source_path, _square(), 0.2)
        second = extract_outer_boundary_points(self.source_path, _square(), 0.2)
        np.testing.assert_array_equal(first.cloud.points_xyz, second.cloud.points_xyz)
        self.assertEqual(first.statistics, second.statistics)


class TestOuterBoundaryExtractionSource(OuterExtractionTestCase):
    def test_source_reader_is_invoked_once(self):
        _write_xyz(self.source_path, [(1.0, 0.0, 2.0), (5.0, 5.0, 3.0)])
        with patch(
            "app.services.boundary_extraction.iter_xyz_points",
            wraps=iter_xyz_points,
        ) as reader:
            result = extract_outer_boundary_points(self.source_path, _square(), 0.5)

        reader.assert_called_once_with(self.source_path)
        self.assertEqual(result.statistics.source_points_seen, 2)

    def test_empty_source_produces_canonical_empty_array(self):
        result = self.extract([])
        self.assertEqual(result.cloud.points_xyz.shape, (0, 3))
        self.assertEqual(result.cloud.points_xyz.dtype, np.float64)

    def test_all_and_no_matching_points(self):
        all_points = [(float(x), 0.1, float(-x)) for x in range(1, 10)]
        all_result = self.extract(all_points, search_width=0.2)
        np.testing.assert_array_equal(all_result.cloud.points_xyz, all_points)

        none_result = self.extract([(3.0, 3.0, 1.0), (7.0, 7.0, 2.0)], search_width=0.2)
        self.assertEqual(none_result.cloud.points_xyz.shape, (0, 3))

    def test_source_signature_change_rejects_result(self):
        _write_xyz(self.source_path, [(1.0, 0.1, 2.0)])
        before = {
            "canonical_path": str(self.source_path.resolve()),
            "file_size": 12,
            "mtime_ns": 100,
        }
        after = {**before, "mtime_ns": 101}
        with patch(
            "app.services.boundary_extraction.get_file_signature",
            side_effect=(before, after),
        ):
            with self.assertRaisesRegex(
                SourceChangedDuringExtractionError,
                "changed during outer boundary extraction",
            ):
                extract_outer_boundary_points(self.source_path, _square(), 0.5)

    def test_numpy_block_accumulator_spans_blocks_without_tuple_objects(self):
        accumulator = _XYZBlockAccumulator(block_size=2)
        points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
        for point in points:
            accumulator.append(*point)
        result = accumulator.to_array()
        np.testing.assert_array_equal(result, points)
        self.assertEqual(result.dtype, np.float64)


class TestOuterBoundaryExtractionScale(OuterExtractionTestCase):
    def test_spatial_index_avoids_most_full_scan_checks(self):
        contour = _subdivided_rectangle(50)
        points = [
            *((x + 0.5, 0.1, float(x)) for x in range(50)),
            *((x + 0.5, 25.0, float(100 + x)) for x in range(50)),
        ]
        result = self.extract(points, contour, 0.2)
        expected = _brute_force_points(points, contour, 0.2)
        np.testing.assert_array_equal(result.cloud.points_xyz, expected)

        statistics = result.statistics
        self.assertEqual(statistics.source_points_seen, 100)
        self.assertGreater(statistics.brute_force_segment_checks, 10_000)
        self.assertLess(
            statistics.exact_segment_checks,
            statistics.brute_force_segment_checks // 20,
        )
        self.assertEqual(
            statistics.exact_segment_checks_avoided,
            statistics.brute_force_segment_checks - statistics.exact_segment_checks,
        )


if __name__ == "__main__":
    unittest.main()
