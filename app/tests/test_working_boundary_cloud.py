from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.working_boundary_cloud import (  # noqa: E402
    BoundaryExtractionParameters,
    HoleBoundaryCloud,
    HoleDecisionSnapshot,
    HoleDetectorMetrics,
    OuterBoundaryCloud,
    WorkingBoundaryCloud,
    closed_contour_segment_aabbs,
    hole_search_aabb,
    point_is_near_closed_contour,
    point_is_near_hole_boundary,
    point_is_near_segment,
    segment_search_aabb,
)


def _square() -> np.ndarray:
    return np.array(
        [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]],
        dtype=np.float64,
    )


def _metrics() -> HoleDetectorMetrics:
    return HoleDetectorMetrics(
        area_cells=80,
        area_mm2=51.2,
        bbox_width_mm=8.0,
        bbox_height_mm=8.4,
        aspect_ratio=1.05,
        circularity=0.91,
        mean_error_mm=0.18,
        max_error_mm=0.61,
        error_ratio=0.045,
    )


def _detector_decision(
    hole_id: int,
    *,
    group_id: str | None = "G1",
    automatic_accepted: bool = True,
    final_accepted: bool = True,
    decision_source: str = "automatic",
    reject_reason: str | None = None,
    center: tuple[float, float] = (50.0, -25.0),
    radius: float = 4.0,
) -> HoleDecisionSnapshot:
    return HoleDecisionSnapshot(
        hole_id=hole_id,
        group_id=group_id,
        origin="detector",
        automatic_accepted=automatic_accepted,
        final_accepted=final_accepted,
        decision_source=decision_source,  # type: ignore[arg-type]
        automatic_reject_reason=reject_reason,
        preliminary_center_x=center[0],
        preliminary_center_y=center[1],
        preliminary_radius=radius,
        detector_metrics=_metrics(),
    )


def _manual_decision(
    hole_id: int,
    *,
    group_id: str | None = None,
    center: tuple[float, float] = (70.0, 12.0),
    radius: float = 3.5,
) -> HoleDecisionSnapshot:
    return HoleDecisionSnapshot(
        hole_id=hole_id,
        group_id=group_id,
        origin="manual",
        automatic_accepted=None,
        final_accepted=True,
        decision_source="user",
        automatic_reject_reason=None,
        preliminary_center_x=center[0],
        preliminary_center_y=center[1],
        preliminary_radius=radius,
        detector_metrics=None,
    )


class TestOuterBoundaryReferenceGeometry(unittest.TestCase):
    def test_single_segment_predicate_matches_closed_contour_geometry(self):
        self.assertTrue(
            point_is_near_segment((4.0, 0.5), (0.0, 0.0), (8.0, 0.0), 0.5)
        )
        self.assertFalse(
            point_is_near_segment((4.0, 0.51), (0.0, 0.0), (8.0, 0.0), 0.5)
        )

    def test_point_on_segment_is_inside_search_band(self):
        self.assertTrue(point_is_near_closed_contour((20.0, 20.0), _square(), 0.5))

    def test_points_inside_and_outside_search_width(self):
        contour = _square()
        self.assertTrue(point_is_near_closed_contour((20.0, 21.25), contour, 1.25))
        self.assertFalse(point_is_near_closed_contour((20.0, 21.26), contour, 1.25))

    def test_closing_segment_is_checked_without_duplicated_first_point(self):
        contour = np.array(
            [[0.0, 0.0], [8.0, 0.0], [8.0, 5.0], [0.0, 5.0]],
            dtype=np.float64,
        )
        self.assertFalse(np.array_equal(contour[0], contour[-1]))
        self.assertTrue(point_is_near_closed_contour((-0.2, 2.5), contour, 0.2))

    def test_search_width_uses_world_coordinates_without_grid_or_raster(self):
        contour = _square() + np.array([10_000.0, -20_000.0])
        self.assertTrue(
            point_is_near_closed_contour((10_020.0, -19_978.0), contour, 2.0)
        )
        self.assertFalse(
            point_is_near_closed_contour((10_020.0, -19_977.9), contour, 2.0)
        )

    def test_segment_aabb_is_conservative_for_exact_search(self):
        contour = _square()
        boxes = closed_contour_segment_aabbs(contour, 1.5)
        self.assertEqual(len(boxes), len(contour))

        for x in np.linspace(8.0, 32.0, 25):
            for y in np.linspace(18.0, 42.0, 25):
                point = (float(x), float(y))
                if point_is_near_closed_contour(point, contour, 1.5):
                    self.assertTrue(any(box.contains(point) for box in boxes))

        closing_box = boxes[-1]
        self.assertTrue(closing_box.contains((9.0, 30.0)))

    def test_degenerate_segment_aabb_is_safe(self):
        box = segment_search_aabb((3.0, 4.0), (3.0, 4.0), 2.0)
        self.assertTrue(box.contains((5.0, 4.0)))
        self.assertEqual((box.min_x, box.min_y, box.max_x, box.max_y), (1.0, 2.0, 5.0, 6.0))

    def test_duplicate_neighbors_are_allowed_but_degenerate_contours_are_rejected(self):
        with_duplicate = np.array(
            [[0.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
            dtype=np.float64,
        )
        cloud = OuterBoundaryCloud(with_duplicate, 1.0)
        self.assertTrue(point_is_near_closed_contour((4.5, 1.5), cloud.preliminary_contour_world, 0.5))

        invalid_contours = (
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64),
        )
        for contour in invalid_contours:
            with self.subTest(contour=contour):
                with self.assertRaises(ValueError):
                    OuterBoundaryCloud(contour, 1.0)

    def test_invalid_contour_shape_dtype_and_coordinates_are_rejected(self):
        invalid_arrays = (
            np.zeros((3,), dtype=np.float64),
            np.zeros((3, 3), dtype=np.float64),
            np.array([[0, 0], [1, 0], [0, 1]], dtype=object),
            np.array([[0.0, 0.0], [1.0, np.nan], [0.0, 1.0]]),
            np.array([[0.0, 0.0], [1.0, np.inf], [0.0, 1.0]]),
        )
        for contour in invalid_arrays:
            with self.subTest(contour=contour):
                with self.assertRaises((TypeError, ValueError)):
                    OuterBoundaryCloud(contour, 1.0)


class TestHoleBoundaryReferenceGeometry(unittest.TestCase):
    def test_annulus_accepts_circle_and_nearby_inner_and_outer_points(self):
        center = (100.0, -50.0)
        self.assertTrue(point_is_near_hole_boundary((105.0, -50.0), center, 5.0, 0.5))
        self.assertTrue(point_is_near_hole_boundary((104.6, -50.0), center, 5.0, 0.5))
        self.assertTrue(point_is_near_hole_boundary((105.4, -50.0), center, 5.0, 0.5))

    def test_annulus_rejects_points_too_far_inside_and_outside(self):
        center = (100.0, -50.0)
        self.assertFalse(point_is_near_hole_boundary((100.0, -50.0), center, 5.0, 0.5))
        self.assertFalse(point_is_near_hole_boundary((103.0, -50.0), center, 5.0, 0.5))
        self.assertFalse(point_is_near_hole_boundary((106.0, -50.0), center, 5.0, 0.5))

    def test_hole_aabb_is_conservative_for_exact_annulus(self):
        center = (7.0, -3.0)
        box = hole_search_aabb(center, 4.0, 1.25)
        for x in np.linspace(1.0, 13.0, 49):
            for y in np.linspace(-9.0, 3.0, 49):
                point = (float(x), float(y))
                if point_is_near_hole_boundary(point, center, 4.0, 1.25):
                    self.assertTrue(box.contains(point))

        self.assertEqual(
            (box.min_x, box.min_y, box.max_x, box.max_y),
            (1.75, -8.25, 12.25, 2.25),
        )

    def test_hole_search_depends_only_on_semantic_circle(self):
        decision = _manual_decision(7, center=(12.0, 9.0), radius=3.0)
        self.assertTrue(
            point_is_near_hole_boundary(
                (15.25, 9.0),
                decision.preliminary_center,
                decision.preliminary_radius,
                0.25,
            )
        )

    def test_invalid_hole_geometry_is_rejected(self):
        invalid_calls = (
            ((np.nan, 0.0), 2.0, 1.0),
            ((0.0, np.inf), 2.0, 1.0),
            ((0.0, 0.0), 0.0, 1.0),
            ((0.0, 0.0), -1.0, 1.0),
            ((0.0, 0.0), np.nan, 1.0),
            ((0.0, 0.0), 1.0, 0.0),
        )
        for center, radius, width in invalid_calls:
            with self.subTest(center=center, radius=radius, width=width):
                with self.assertRaises((TypeError, ValueError)):
                    point_is_near_hole_boundary((1.0, 0.0), center, radius, width)


class TestWorkingBoundaryCloudModel(unittest.TestCase):
    def test_search_parameters_reject_invalid_values_and_have_no_defaults(self):
        parameters = BoundaryExtractionParameters(2.0, 1.5)
        self.assertEqual(parameters.outer_boundary_search_width, 2.0)
        self.assertEqual(parameters.hole_boundary_search_width, 1.5)
        with self.assertRaises(TypeError):
            BoundaryExtractionParameters(True, 1.0)
        for invalid in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    BoundaryExtractionParameters(invalid, 1.0)

    def test_accepted_detector_snapshot_preserves_identity_geometry_and_provenance(self):
        decision = _detector_decision(17, group_id="G3")
        self.assertEqual(decision.hole_id, 17)
        self.assertEqual(decision.group_id, "G3")
        self.assertEqual(decision.origin, "detector")
        self.assertTrue(decision.automatic_accepted)
        self.assertTrue(decision.final_accepted)
        self.assertEqual(decision.decision_source, "automatic")
        self.assertEqual(decision.preliminary_center, (50.0, -25.0))
        self.assertEqual(decision.preliminary_radius, 4.0)
        self.assertEqual(decision.detector_metrics, _metrics())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            decision.group_id = "G4"  # type: ignore[misc]

    def test_manually_accepted_automatic_reject_preserves_reason_and_metrics(self):
        decision = _detector_decision(
            8,
            group_id=None,
            automatic_accepted=False,
            final_accepted=True,
            decision_source="user",
            reject_reason="bad_circle_fit",
        )
        self.assertIsNone(decision.group_id)
        self.assertFalse(decision.automatic_accepted)
        self.assertTrue(decision.final_accepted)
        self.assertEqual(decision.decision_source, "user")
        self.assertEqual(decision.automatic_reject_reason, "bad_circle_fit")
        self.assertEqual(decision.detector_metrics.mean_error_mm, 0.18)

    def test_manual_hole_has_user_provenance_and_no_automatic_data(self):
        decision = _manual_decision(21)
        self.assertEqual(decision.origin, "manual")
        self.assertIsNone(decision.automatic_accepted)
        self.assertEqual(decision.decision_source, "user")
        self.assertIsNone(decision.detector_metrics)
        self.assertIsNone(decision.group_id)

        hole = HoleBoundaryCloud(decision, 0.5)
        self.assertTrue(hole.contains_search_point((73.5, 12.0)))
        self.assertTrue(hole.search_aabb().contains((74.0, 12.0)))

    def test_invalid_cross_field_decision_semantics_are_rejected(self):
        with self.assertRaises(ValueError):
            HoleDecisionSnapshot(
                hole_id=1,
                group_id=None,
                origin="manual",
                automatic_accepted=True,
                final_accepted=True,
                decision_source="user",
                automatic_reject_reason=None,
                preliminary_center_x=0.0,
                preliminary_center_y=0.0,
                preliminary_radius=1.0,
                detector_metrics=None,
            )
        with self.assertRaises(ValueError):
            _detector_decision(
                1,
                automatic_accepted=False,
                final_accepted=True,
                decision_source="automatic",
            )

    def test_cloud_arrays_are_float64_read_only_copies_and_preserve_z(self):
        contour = _square().astype(np.int32)
        points = np.array([[1, 2, 17], [3, 4, -9]], dtype=np.int16)
        outer = OuterBoundaryCloud(contour, 2.0, points)
        self.assertEqual(outer.preliminary_contour_world.dtype, np.float64)
        self.assertEqual(outer.points_xyz.dtype, np.float64)
        np.testing.assert_array_equal(outer.points_xyz[:, 2], [17.0, -9.0])
        self.assertFalse(outer.preliminary_contour_world.flags.writeable)
        self.assertFalse(outer.points_xyz.flags.writeable)

        contour[0] = 999
        points[0] = 999
        np.testing.assert_array_equal(outer.preliminary_contour_world[0], [10.0, 20.0])
        np.testing.assert_array_equal(outer.points_xyz[0], [1.0, 2.0, 17.0])
        self.assertTrue(outer.contains_search_point((20.0, 22.0)))
        self.assertTrue(any(box.contains((20.0, 22.0)) for box in outer.segment_search_aabbs()))

    def test_empty_clouds_have_canonical_shape(self):
        outer = OuterBoundaryCloud(_square(), 2.0)
        hole = HoleBoundaryCloud(_manual_decision(1), 1.0)
        self.assertEqual(outer.points_xyz.shape, (0, 3))
        self.assertEqual(hole.points_xyz.shape, (0, 3))
        self.assertEqual(outer.points_xyz.dtype, np.float64)
        self.assertEqual(hole.points_xyz.dtype, np.float64)

    def test_invalid_cloud_array_shape_dtype_and_values_are_rejected(self):
        invalid_points = (
            np.empty((0, 2), dtype=np.float64),
            np.array([[1.0, 2.0, np.nan]]),
            np.array([[1, 2, 3]], dtype=object),
            np.array([[True, False, True]], dtype=bool),
        )
        for points in invalid_points:
            with self.subTest(points=points):
                with self.assertRaises((TypeError, ValueError)):
                    OuterBoundaryCloud(_square(), 1.0, points)

    def test_same_group_holes_remain_distinct_and_different_groups_do_not_mix(self):
        parameters = BoundaryExtractionParameters(2.0, 1.0)
        holes = (
            HoleBoundaryCloud(_detector_decision(1, group_id="G1"), 1.0),
            HoleBoundaryCloud(_detector_decision(2, group_id="G1"), 1.0),
            HoleBoundaryCloud(_detector_decision(3, group_id="G2"), 1.0),
        )
        cloud = WorkingBoundaryCloud(
            parameters,
            OuterBoundaryCloud(_square(), 2.0),
            holes,
        )
        self.assertEqual([hole.decision.hole_id for hole in cloud.holes], [1, 2, 3])
        self.assertEqual([hole.decision.group_id for hole in cloud.holes], ["G1", "G1", "G2"])
        self.assertIsNot(cloud.holes[0], cloud.holes[1])

    def test_working_cloud_rejects_duplicate_ids_rejected_holes_and_width_mismatch(self):
        parameters = BoundaryExtractionParameters(2.0, 1.0)
        outer = OuterBoundaryCloud(_square(), 2.0)
        duplicate = HoleBoundaryCloud(_manual_decision(1), 1.0)
        with self.assertRaises(ValueError):
            WorkingBoundaryCloud(parameters, outer, (duplicate, duplicate))

        rejected = _detector_decision(
            2,
            automatic_accepted=True,
            final_accepted=False,
            decision_source="user",
        )
        with self.assertRaises(ValueError):
            HoleBoundaryCloud(rejected, 1.0)
        with self.assertRaises(ValueError):
            WorkingBoundaryCloud(
                parameters,
                OuterBoundaryCloud(_square(), 2.5),
            )


if __name__ == "__main__":
    unittest.main()
