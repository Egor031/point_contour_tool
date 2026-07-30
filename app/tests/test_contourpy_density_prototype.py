from dataclasses import fields
import unittest

import numpy as np

from experiments.contourpy_density_prototype import (
    RectangleContourMetrics,
    RotatedRectangleContourMetrics,
    create_synthetic_rectangle_field,
    create_synthetic_rotated_rectangle_field,
    compute_rotated_rectangle_contour_metrics,
    extract_contourpy_external_contour,
    extract_mask_external_contour,
    normalize_external_ring,
    run_rectangle_comparison,
    run_rotated_rectangle_comparison,
    rotated_rectangle_vertices,
    signed_ring_area,
)


class TestContourPyDensityPrototype(unittest.TestCase):
    def _assert_contour_contract(self, contour):
        self.assertEqual(2, contour.ndim)
        self.assertEqual(2, contour.shape[1])
        self.assertEqual(np.dtype(np.float64), contour.dtype)
        self.assertTrue(np.isfinite(contour).all())
        self.assertFalse(np.allclose(contour[0], contour[-1]))
        self.assertGreaterEqual(len(np.unique(contour, axis=0)), 3)
        self.assertGreater(signed_ring_area(contour), 0.0)

    def test_cell_center_coordinates(self):
        field = create_synthetic_rectangle_field(
            min_x_mm=2.0,
            min_y_mm=-3.0,
            height_cells=80,
            cell_size_mm=0.75,
        )

        self.assertAlmostEqual(
            2.0 + 0.5 * 0.75, field.x_coordinates_mm[0]
        )
        self.assertAlmostEqual(
            -3.0 + 0.5 * 0.75, field.y_coordinates_mm[0]
        )
        np.testing.assert_allclose(np.diff(field.x_coordinates_mm), 0.75)
        np.testing.assert_allclose(np.diff(field.y_coordinates_mm), 0.75)

    def test_density_shape_dtype_and_range(self):
        field = create_synthetic_rectangle_field()

        self.assertEqual(
            (
                len(field.y_coordinates_mm),
                len(field.x_coordinates_mm),
            ),
            field.density.shape,
        )
        self.assertEqual(np.dtype(np.float64), field.density.dtype)
        self.assertTrue(np.isfinite(field.density).all())
        self.assertGreaterEqual(float(field.density.min()), 0.0)
        self.assertLessEqual(float(field.density.max()), 10.0)

    def test_rectangle_boundaries_do_not_coincide_with_grid(self):
        field = create_synthetic_rectangle_field()
        x_boundaries = (
            field.rectangle_min_x_mm,
            field.rectangle_max_x_mm,
        )
        y_boundaries = (
            field.rectangle_min_y_mm,
            field.rectangle_max_y_mm,
        )
        x_edges = field.min_x_mm + np.arange(
            len(field.x_coordinates_mm) + 1
        ) * field.cell_size_mm
        y_edges = field.min_y_mm + np.arange(
            len(field.y_coordinates_mm) + 1
        ) * field.cell_size_mm

        for boundary in x_boundaries:
            self.assertFalse(
                np.isclose(field.x_coordinates_mm, boundary).any()
            )
            self.assertFalse(np.isclose(x_edges, boundary).any())
        for boundary in y_boundaries:
            self.assertFalse(
                np.isclose(field.y_coordinates_mm, boundary).any()
            )
            self.assertFalse(np.isclose(y_edges, boundary).any())

    def test_density_changes_in_the_correct_direction(self):
        field = create_synthetic_rectangle_field()
        rectangle_center = np.array(
            [
                (
                    field.rectangle_min_x_mm
                    + field.rectangle_max_x_mm
                )
                / 2.0,
                (
                    field.rectangle_min_y_mm
                    + field.rectangle_max_y_mm
                )
                / 2.0,
            ]
        )
        center_ix = int(
            np.argmin(
                np.abs(field.x_coordinates_mm - rectangle_center[0])
            )
        )
        center_iy = int(
            np.argmin(
                np.abs(field.y_coordinates_mm - rectangle_center[1])
            )
        )

        self.assertGreater(
            field.density[center_iy, center_ix], field.threshold
        )
        self.assertLess(field.density[0, 0], field.threshold)

    def test_normalization_removes_duplicate_closure(self):
        ring = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )

        normalized = normalize_external_ring(ring)

        self.assertEqual(4, len(normalized))
        np.testing.assert_allclose(normalized, ring[:-1])

    def test_normalization_makes_ring_counter_clockwise(self):
        clockwise_ring = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [2.0, 1.0],
                [2.0, 0.0],
            ]
        )
        self.assertLess(signed_ring_area(clockwise_ring), 0.0)

        normalized = normalize_external_ring(clockwise_ring)

        self.assertGreater(signed_ring_area(normalized), 0.0)

    def test_mask_contour_contract(self):
        contour = extract_mask_external_contour(
            create_synthetic_rectangle_field()
        )

        self._assert_contour_contract(contour)

    def test_contourpy_contour_contract(self):
        contour = extract_contourpy_external_contour(
            create_synthetic_rectangle_field()
        )

        self._assert_contour_contract(contour)

    def test_both_methods_return_one_external_object(self):
        field = create_synthetic_rectangle_field()

        mask_contour = extract_mask_external_contour(field)
        contourpy_contour = extract_contourpy_external_contour(field)

        self.assertIsInstance(mask_contour, np.ndarray)
        self.assertIsInstance(contourpy_contour, np.ndarray)
        self.assertGreater(len(mask_contour), 0)
        self.assertGreater(len(contourpy_contour), 0)

    def test_contourpy_bounding_box_is_more_accurate(self):
        comparison = run_rectangle_comparison()

        self.assertLess(
            comparison.contourpy_metrics.bounding_box_error_mm,
            comparison.mask_metrics.bounding_box_error_mm,
        )

    def test_contourpy_rms_boundary_error_is_lower(self):
        comparison = run_rectangle_comparison()

        self.assertLess(
            comparison.contourpy_metrics.rms_boundary_error_mm,
            comparison.mask_metrics.rms_boundary_error_mm,
        )

    def test_contourpy_area_error_is_not_worse(self):
        comparison = run_rectangle_comparison()

        self.assertLessEqual(
            comparison.contourpy_metrics.area_error_mm2,
            comparison.mask_metrics.area_error_mm2 + 1e-12,
        )

    def test_comparison_is_deterministic(self):
        first = run_rectangle_comparison()
        second = run_rectangle_comparison()

        for metrics_name in ("mask_metrics", "contourpy_metrics"):
            first_metrics = getattr(first, metrics_name)
            second_metrics = getattr(second, metrics_name)
            for metric_field in fields(RectangleContourMetrics):
                first_value = getattr(first_metrics, metric_field.name)
                second_value = getattr(second_metrics, metric_field.name)
                if isinstance(first_value, int):
                    self.assertEqual(first_value, second_value)
                else:
                    self.assertAlmostEqual(first_value, second_value)


class TestRotatedRectangleDensityPrototype(unittest.TestCase):
    def _assert_contour_contract(self, contour):
        self.assertEqual(2, contour.ndim)
        self.assertEqual(2, contour.shape[1])
        self.assertEqual(np.dtype(np.float64), contour.dtype)
        self.assertTrue(np.isfinite(contour).all())
        self.assertFalse(np.allclose(contour[0], contour[-1]))
        self.assertGreaterEqual(len(np.unique(contour, axis=0)), 3)
        self.assertGreater(signed_ring_area(contour), 0.0)

    def test_rotated_field_shape_dtype_and_finiteness(self):
        field = create_synthetic_rotated_rectangle_field()

        self.assertEqual((80, 100), field.density.shape)
        self.assertEqual(
            (
                len(field.y_coordinates_mm),
                len(field.x_coordinates_mm),
            ),
            field.density.shape,
        )
        self.assertEqual(np.dtype(np.float64), field.density.dtype)
        self.assertEqual(
            np.dtype(np.float64), field.x_coordinates_mm.dtype
        )
        self.assertEqual(
            np.dtype(np.float64), field.y_coordinates_mm.dtype
        )
        self.assertTrue(np.isfinite(field.density).all())
        self.assertTrue(np.isfinite(field.x_coordinates_mm).all())
        self.assertTrue(np.isfinite(field.y_coordinates_mm).all())
        self.assertGreaterEqual(float(field.density.min()), 0.0)
        self.assertLessEqual(float(field.density.max()), 10.0)

    def test_rotated_density_changes_in_the_correct_direction(self):
        field = create_synthetic_rotated_rectangle_field()
        center_ix = int(
            np.argmin(
                np.abs(field.x_coordinates_mm - field.center_x_mm)
            )
        )
        center_iy = int(
            np.argmin(
                np.abs(field.y_coordinates_mm - field.center_y_mm)
            )
        )

        self.assertGreater(
            field.density[center_iy, center_ix], field.threshold
        )
        self.assertLess(field.density[0, 0], field.threshold)

    def test_rotated_angle_is_not_axis_or_diagonal_aligned(self):
        field = create_synthetic_rotated_rectangle_field()

        remainder_45 = float(np.mod(field.angle_degrees, 45.0))
        remainder_90 = float(np.mod(field.angle_degrees, 90.0))
        self.assertFalse(np.isclose(remainder_45, 0.0, atol=1e-12))
        self.assertFalse(np.isclose(remainder_45, 45.0, atol=1e-12))
        self.assertFalse(np.isclose(remainder_90, 0.0, atol=1e-12))
        self.assertFalse(np.isclose(remainder_90, 90.0, atol=1e-12))

    def test_rotated_vertices_contract(self):
        vertices = rotated_rectangle_vertices(
            create_synthetic_rotated_rectangle_field()
        )

        self.assertEqual((4, 2), vertices.shape)
        self.assertEqual(np.dtype(np.float64), vertices.dtype)
        self.assertTrue(np.isfinite(vertices).all())
        self.assertEqual(4, len(np.unique(vertices, axis=0)))
        self.assertGreater(signed_ring_area(vertices), 0.0)
        self.assertFalse(np.allclose(vertices[0], vertices[-1]))

    def test_rotated_vertex_side_lengths_match_dimensions(self):
        field = create_synthetic_rotated_rectangle_field()
        vertices = rotated_rectangle_vertices(field)
        side_lengths = np.linalg.norm(
            np.roll(vertices, -1, axis=0) - vertices,
            axis=1,
        )
        expected_lengths = np.array(
            [
                field.rectangle_height_mm,
                field.rectangle_height_mm,
                field.rectangle_width_mm,
                field.rectangle_width_mm,
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            np.sort(side_lengths),
            np.sort(expected_lengths),
            rtol=0.0,
            atol=1e-12,
        )

    def test_rotated_vertex_center_matches_field_center(self):
        field = create_synthetic_rotated_rectangle_field()
        vertices = rotated_rectangle_vertices(field)

        np.testing.assert_allclose(
            np.mean(vertices, axis=0),
            [field.center_x_mm, field.center_y_mm],
            rtol=0.0,
            atol=1e-12,
        )

    def test_rotated_mask_contour_contract(self):
        contour = extract_mask_external_contour(
            create_synthetic_rotated_rectangle_field()
        )

        self._assert_contour_contract(contour)

    def test_rotated_contourpy_contour_contract(self):
        contour = extract_contourpy_external_contour(
            create_synthetic_rotated_rectangle_field()
        )

        self._assert_contour_contract(contour)

    def test_rotated_metrics_are_finite_and_non_negative(self):
        comparison = run_rotated_rectangle_comparison()

        for metrics in (
            comparison.mask_metrics,
            comparison.contourpy_metrics,
        ):
            self.assertGreater(metrics.point_count, 0)
            self.assertTrue(np.isfinite(metrics.signed_area_mm2))
            self.assertGreater(metrics.signed_area_mm2, 0.0)
            for metric_field in fields(RotatedRectangleContourMetrics):
                if metric_field.name in ("point_count", "signed_area_mm2"):
                    continue
                value = getattr(metrics, metric_field.name)
                self.assertTrue(
                    np.isfinite(value),
                    msg=f"{metric_field.name} is not finite",
                )
                self.assertGreaterEqual(
                    value,
                    0.0,
                    msg=f"{metric_field.name} is negative",
                )

    def test_contourpy_reduces_rotated_boundary_rms(self):
        comparison = run_rotated_rectangle_comparison()

        self.assertLess(
            comparison.contourpy_metrics.rms_boundary_error_mm,
            comparison.mask_metrics.rms_boundary_error_mm,
        )

    def test_contourpy_reduces_mean_true_side_rms(self):
        comparison = run_rotated_rectangle_comparison()

        self.assertLess(
            comparison.contourpy_metrics.mean_true_side_rms_mm,
            comparison.mask_metrics.mean_true_side_rms_mm,
        )

    def test_contourpy_reduces_mean_fitted_side_rms(self):
        comparison = run_rotated_rectangle_comparison()

        self.assertLess(
            comparison.contourpy_metrics.mean_fitted_side_rms_mm,
            comparison.mask_metrics.mean_fitted_side_rms_mm,
        )

    def test_contourpy_does_not_worsen_mean_angle_error(self):
        comparison = run_rotated_rectangle_comparison()

        self.assertLessEqual(
            comparison.contourpy_metrics.mean_side_angle_error_deg,
            comparison.mask_metrics.mean_side_angle_error_deg + 1e-12,
        )

    def test_contourpy_does_not_worsen_rotated_area_error(self):
        comparison = run_rotated_rectangle_comparison()

        self.assertLessEqual(
            comparison.contourpy_metrics.area_error_mm2,
            comparison.mask_metrics.area_error_mm2 + 1e-12,
        )

    def test_rotated_comparison_is_deterministic(self):
        first = run_rotated_rectangle_comparison()
        second = run_rotated_rectangle_comparison()

        for metrics_name in ("mask_metrics", "contourpy_metrics"):
            first_metrics = getattr(first, metrics_name)
            second_metrics = getattr(second, metrics_name)
            for metric_field in fields(RotatedRectangleContourMetrics):
                first_value = getattr(first_metrics, metric_field.name)
                second_value = getattr(second_metrics, metric_field.name)
                if isinstance(first_value, int):
                    self.assertEqual(first_value, second_value)
                else:
                    self.assertAlmostEqual(first_value, second_value)


if __name__ == "__main__":
    unittest.main()
