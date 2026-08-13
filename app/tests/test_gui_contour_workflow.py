from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.density_grid import DensityGrid  # noqa: E402
from app.core.working_area import WorkingArea  # noqa: E402
from app.ui import contour_workflow  # noqa: E402
from app.ui.contour_workflow import (  # noqa: E402
    MaskEditingSession,
    PreliminaryContourParameters,
    find_preliminary_contour_for_working_area,
    mask_edits_for_world_segment,
    preliminary_contour_parameters_for_threshold,
    processing_mask_to_preview,
    rebuild_preliminary_contour_from_edited_mask,
)
from app.ui.density_workflow import (  # noqa: E402
    apply_working_area_state,
    invalidate_preliminary_contour_state,
    reset_source_dependent_state,
)


def _two_part_grid() -> DensityGrid:
    density = np.zeros((30, 50), dtype=np.uint32)
    density[5:15, 5:15] = 10
    density[5:20, 30:48] = 10
    return DensityGrid(
        density=density,
        cell_size=1.0,
        min_x=0.0,
        min_y=0.0,
    )


def _threshold_comparison_grid() -> DensityGrid:
    density = np.zeros((20, 20), dtype=np.uint32)
    density[4:16, 4:16] = 1
    density[10:16, 4:16] = 5
    return DensityGrid(
        density=density,
        cell_size=1.0,
        min_x=0.0,
        min_y=0.0,
    )


def _threshold_comparison_area() -> WorkingArea:
    return WorkingArea.from_rectangle_bounds((3.5, 3.5, 16.5, 16.5))


class TestGuiContourWorkflow(unittest.TestCase):
    def _mask_editing_fixture(self):
        grid = _threshold_comparison_grid()
        result = find_preliminary_contour_for_working_area(
            grid,
            _threshold_comparison_area(),
        )
        return result, MaskEditingSession.from_preliminary_contour(result)

    def test_mask_editing_starts_with_immutable_equivalent_base(self):
        result, editing = self._mask_editing_fixture()

        np.testing.assert_array_equal(editing.base_mask, editing.edited_mask)
        self.assertIs(editing.base_mask, result.masks.contour_mask)

        editing.begin_stroke()
        editing.apply_edit(
            {"mode": "remove", "x": 5.5, "y": 5.5, "radius_mm": 0.5}
        )
        editing.finish_stroke()
        self.assertEqual(editing.base_mask[5, 5], 1)
        self.assertEqual(editing.edited_mask[5, 5], 0)

        editing.clear_edits()
        np.testing.assert_array_equal(editing.edited_mask, editing.base_mask)
        self.assertEqual(editing.history, [])

    def test_add_remove_and_working_area_clipping(self):
        result, editing = self._mask_editing_fixture()
        editing.edited_mask.fill(0)

        editing.begin_stroke()
        self.assertTrue(editing.apply_edit(
            {"mode": "add", "x": 5.5, "y": 5.5, "radius_mm": 0.5}
        ))
        editing.apply_edit(
            {"mode": "add", "x": 0.5, "y": 0.5, "radius_mm": 4.0}
        )
        editing.finish_stroke()

        self.assertEqual(editing.edited_mask[5, 5], 1)
        self.assertFalse(np.any(editing.edited_mask[~editing.working_area_mask]))

        editing.begin_stroke()
        self.assertTrue(editing.apply_edit(
            {"mode": "remove", "x": 5.5, "y": 5.5, "radius_mm": 0.5}
        ))
        editing.finish_stroke()
        self.assertEqual(editing.edited_mask[5, 5], 0)

    def test_drag_is_one_undo_step_and_is_interpolated_in_world_space(self):
        result, editing = self._mask_editing_fixture()
        editing.edited_mask.fill(0)
        edits = mask_edits_for_world_segment(
            editing.grid,
            (5.5, 5.5),
            (14.5, 5.5),
            mode="add",
            brush_diameter_mm=1.0,
            stroke_id=7,
        )

        editing.begin_stroke()
        for edit in edits:
            editing.apply_edit(edit)
        editing.finish_stroke()

        self.assertEqual(len(editing.history), 1)
        self.assertTrue(np.all(editing.edited_mask[5, 5:15] == 1))
        self.assertTrue(editing.undo_last_stroke())
        self.assertFalse(np.any(editing.edited_mask))
        self.assertFalse(editing.undo_last_stroke())

    def test_physical_brush_is_independent_of_preview_scale_and_zoom(self):
        grid = DensityGrid(
            density=np.zeros((40, 40), dtype=np.uint32),
            cell_size=0.5,
            min_x=-10.0,
            min_y=20.0,
        )
        edit = mask_edits_for_world_segment(
            grid,
            (-4.75, 25.25),
            (-4.75, 25.25),
            mode="add",
            brush_diameter_mm=3.0,
            stroke_id=1,
        )[0]
        from app.core.mask_edits import rasterize_mask_edit_cells

        rows, columns = rasterize_mask_edit_cells(grid.density.shape, grid, edit)
        preview_positions = []
        from app.core.coordinate_transform import CoordinateTransform
        for preview_size in ((40, 40), (10, 10)):
            transform = CoordinateTransform(
                grid_min_x=grid.min_x,
                grid_min_y=grid.min_y,
                cell_size=grid.cell_size,
                grid_width=grid.width,
                grid_height=grid.height,
                preview_width=preview_size[0],
                preview_height=preview_size[1],
            )
            world = transform.preview_to_world(*transform.world_to_preview(-4.75, 25.25))
            preview_positions.append(world)

        self.assertEqual((int(rows.min()), int(rows.max())), (7, 13))
        self.assertEqual((int(columns.min()), int(columns.max())), (7, 13))
        self.assertAlmostEqual(preview_positions[0][0], preview_positions[1][0])
        self.assertAlmostEqual(preview_positions[0][1], preview_positions[1][1])

    def test_preview_uses_current_edited_mask_without_processing(self):
        _result, editing = self._mask_editing_fixture()
        before = processing_mask_to_preview(
            editing.edited_mask,
            preview_width=10,
            preview_height=10,
        )
        editing.begin_stroke()
        editing.apply_edit(
            {"mode": "remove", "x": 5.5, "y": 5.5, "radius_mm": 2.0}
        )
        editing.finish_stroke()
        after = processing_mask_to_preview(
            editing.edited_mask,
            preview_width=10,
            preview_height=10,
        )

        self.assertFalse(np.array_equal(before, after))
        self.assertTrue(editing.contour_stale)

    def test_rebuild_uses_edited_mask_without_reprocessing_or_source(self):
        result, editing = self._mask_editing_fixture()
        old_contour = result.contour.contour_world.copy()
        editing.begin_stroke()
        editing.apply_edit(
            {"mode": "remove", "x": 5.5, "y": 5.5, "radius_mm": 3.0}
        )
        editing.finish_stroke()

        with (
            patch.object(contour_workflow, "build_processing_masks") as build_masks,
            patch("app.core.xyz_reader.iter_xyz_points") as read_source,
            patch("app.core.xyz_reader.compute_stats") as compute_stats,
            patch("app.core.density_grid.build_density_grid") as build_density,
        ):
            rebuilt = rebuild_preliminary_contour_from_edited_mask(result, editing)

        build_masks.assert_not_called()
        read_source.assert_not_called()
        compute_stats.assert_not_called()
        build_density.assert_not_called()
        np.testing.assert_array_equal(rebuilt.masks.contour_mask, editing.edited_mask)
        self.assertFalse(np.array_equal(old_contour, rebuilt.contour.contour_world))
        self.assertFalse(editing.contour_stale)

    def test_failed_rebuild_keeps_previous_result_and_stale_state(self):
        result, editing = self._mask_editing_fixture()
        editing.edited_mask.fill(0)
        editing.contour_stale = True

        with self.assertRaisesRegex(ValueError, "not found"):
            rebuild_preliminary_contour_from_edited_mask(result, editing)

        self.assertGreater(result.contour.point_count, 0)
        self.assertTrue(editing.contour_stale)

    def test_new_find_creates_fresh_edit_session(self):
        first, editing = self._mask_editing_fixture()
        editing.begin_stroke()
        editing.apply_edit(
            {"mode": "remove", "x": 5.5, "y": 5.5, "radius_mm": 1.0}
        )
        editing.finish_stroke()
        second = find_preliminary_contour_for_working_area(
            first.masks.grid,
            first.working_area,
            parameters=preliminary_contour_parameters_for_threshold("Manual", 5.0),
        )
        replacement = MaskEditingSession.from_preliminary_contour(second)

        self.assertEqual(replacement.history, [])
        np.testing.assert_array_equal(replacement.base_mask, replacement.edited_mask)

    def test_auto_threshold_keeps_existing_default_behavior(self):
        grid = _threshold_comparison_grid()
        area = _threshold_comparison_area()

        existing_default = find_preliminary_contour_for_working_area(grid, area)
        explicit_auto = find_preliminary_contour_for_working_area(
            grid,
            area,
            parameters=preliminary_contour_parameters_for_threshold(
                "Auto",
                999.0,
            ),
        )

        self.assertEqual(explicit_auto.parameters.threshold_mode, "auto")
        self.assertIsNone(explicit_auto.parameters.manual_threshold)
        self.assertEqual(
            explicit_auto.masks.threshold_result.threshold,
            existing_default.masks.threshold_result.threshold,
        )
        np.testing.assert_array_equal(
            explicit_auto.masks.contour_mask,
            existing_default.masks.contour_mask,
        )

    def test_manual_threshold_reaches_production_mask_pipeline(self):
        grid = _threshold_comparison_grid()
        parameters = preliminary_contour_parameters_for_threshold("Manual", 5.0)

        result = find_preliminary_contour_for_working_area(
            grid,
            _threshold_comparison_area(),
            parameters=parameters,
        )

        self.assertEqual(result.parameters.threshold_mode, "manual")
        self.assertEqual(result.parameters.manual_threshold, 5.0)
        self.assertEqual(result.masks.threshold_result.mode, "manual")
        self.assertEqual(result.masks.threshold_result.threshold, 5.0)

    def test_manual_threshold_changes_mask_and_preview_without_source(self):
        grid = _threshold_comparison_grid()
        area = _threshold_comparison_area()
        auto_result = find_preliminary_contour_for_working_area(grid, area)
        manual_result = find_preliminary_contour_for_working_area(
            grid,
            area,
            parameters=preliminary_contour_parameters_for_threshold("Manual", 5.0),
        )

        self.assertIs(auto_result.masks.grid, grid)
        self.assertIs(manual_result.masks.grid, grid)
        self.assertGreater(
            int(auto_result.masks.contour_mask.sum()),
            int(manual_result.masks.contour_mask.sum()),
        )
        auto_preview = processing_mask_to_preview(
            auto_result.masks.contour_mask,
            preview_width=10,
            preview_height=10,
        )
        manual_preview = processing_mask_to_preview(
            manual_result.masks.contour_mask,
            preview_width=10,
            preview_height=10,
        )
        self.assertFalse(np.array_equal(auto_preview, manual_preview))
        np.testing.assert_array_equal(
            manual_preview,
            processing_mask_to_preview(
                manual_result.masks.contour_mask,
                preview_width=10,
                preview_height=10,
            ),
        )

    def test_invalid_manual_threshold_is_rejected_before_processing(self):
        grid = _threshold_comparison_grid()
        invalid = PreliminaryContourParameters(
            threshold_mode="manual",
            manual_threshold=float("nan"),
        )

        with patch.object(contour_workflow, "build_processing_masks") as build_masks:
            with self.assertRaisesRegex(ValueError, "must be finite"):
                find_preliminary_contour_for_working_area(
                    grid,
                    _threshold_comparison_area(),
                    parameters=invalid,
                )

        build_masks.assert_not_called()

    def test_invalid_manual_threshold_does_not_replace_previous_gui_result(self):
        from app.ui import viewer_app

        class FakeDensityResult:
            def __init__(self, grid: DensityGrid):
                self.grid = grid

        grid = _threshold_comparison_grid()
        area = _threshold_comparison_area()
        density_result = FakeDensityResult(grid)
        previous_result = find_preliminary_contour_for_working_area(grid, area)
        previous_points = list(map(tuple, previous_result.contour.contour_world))
        replacement_state = {
            "density_result": density_result,
            "active_working_area": area,
            "working_area_density_session": density_result,
            "contour_processing_result": previous_result,
            "contour_points": previous_points,
            "contour_file": "",
        }

        def get_value(tag):
            if tag == viewer_app.CONTOUR_THRESHOLD_MODE_TAG:
                return "Manual"
            if tag == viewer_app.CONTOUR_MANUAL_THRESHOLD_TAG:
                return float("nan")
            raise AssertionError(f"Unexpected DPG value request: {tag}")

        with (
            patch.object(viewer_app, "DensityProcessingResult", FakeDensityResult),
            patch.dict(viewer_app.state, replacement_state, clear=False),
            patch.object(viewer_app.dpg, "get_value", side_effect=get_value),
            patch.object(viewer_app, "_set_status") as set_status,
            patch.object(
                viewer_app,
                "find_preliminary_contour_for_working_area",
            ) as find_contour,
        ):
            viewer_app._find_contour_callback()

            find_contour.assert_not_called()
            self.assertIs(
                viewer_app.state["contour_processing_result"],
                previous_result,
            )
            self.assertEqual(viewer_app.state["contour_points"], previous_points)
            self.assertIn("must be finite", set_status.call_args.args[0])

    def test_processing_mask_preview_flips_y_without_resize(self):
        mask = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.uint8,
        )

        preview = processing_mask_to_preview(
            mask,
            preview_width=3,
            preview_height=3,
        )

        np.testing.assert_array_equal(preview, np.flipud(mask))
        self.assertEqual(preview.dtype, np.uint8)

    def test_processing_mask_preview_uses_nearest_neighbor_resize(self):
        mask = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        )

        preview = processing_mask_to_preview(
            mask,
            preview_width=2,
            preview_height=2,
        )

        np.testing.assert_array_equal(
            preview,
            np.array([[0, 1], [0, 0]], dtype=np.uint8),
        )

    def test_invalidating_contour_clears_processing_mask_preview_state(self):
        state = {
            "contour_processing_result": object(),
            "mask_editing_session": object(),
            "contour_points": [(1.0, 2.0)],
            "contour_file": "",
            "processing_mask_preview": np.ones((2, 2), dtype=np.uint8),
            "mask_edits": [{"mode": "add"}],
        }

        invalidate_preliminary_contour_state(state)

        self.assertIsNone(state["contour_processing_result"])
        self.assertIsNone(state["mask_editing_session"])
        self.assertIsNone(state["processing_mask_preview"])
        self.assertEqual(state["mask_edits"], [])
        self.assertEqual(state["contour_points"], [])

    def test_production_defaults_are_explicit(self):
        parameters = PreliminaryContourParameters()

        self.assertEqual(parameters.threshold_mode, "auto")
        self.assertIsNone(parameters.manual_threshold)
        self.assertEqual(parameters.min_component_area, 0)
        self.assertFalse(parameters.keep_largest)
        self.assertEqual(parameters.fill_holes_area, 0)
        self.assertEqual(parameters.simplify_mm, 0.0)

    def test_rectangle_working_area_builds_preliminary_contour(self):
        grid = _two_part_grid()
        area = WorkingArea.from_rectangle_bounds((4.5, 4.5, 15.5, 15.5))
        density_session = object()

        result = find_preliminary_contour_for_working_area(
            grid,
            area,
            density_session=density_session,
        )

        self.assertGreater(result.contour.point_count, 0)
        self.assertIs(result.density_session, density_session)
        self.assertIs(result.working_area, area)
        self.assertEqual(result.masks.threshold_result.mode, "auto")
        self.assertEqual(result.masks.threshold_result.threshold, 10.0)

    def test_polygon_working_area_builds_preliminary_contour(self):
        grid = _two_part_grid()
        area = WorkingArea.from_polygon(
            [(4.0, 4.0), (16.0, 4.0), (16.0, 16.0), (4.0, 16.0)]
        )

        result = find_preliminary_contour_for_working_area(grid, area)

        self.assertGreater(result.contour.point_count, 0)
        self.assertEqual(result.masks.contour_mask[10, 10], 1)
        self.assertEqual(result.masks.contour_mask[10, 35], 0)

    def test_working_area_excludes_larger_part_outside_workspace(self):
        grid = _two_part_grid()
        area = WorkingArea.from_rectangle_bounds((4.5, 4.5, 15.5, 15.5))

        result = find_preliminary_contour_for_working_area(grid, area)

        self.assertLess(float(result.contour.contour_world[:, 0].max()), 20.0)
        self.assertEqual(int(result.masks.contour_mask[:, 30:].sum()), 0)

    def test_missing_working_area_is_a_controlled_error(self):
        with self.assertRaisesRegex(
            ValueError,
            "Select and apply a Working Area first",
        ):
            find_preliminary_contour_for_working_area(_two_part_grid(), None)

    def test_empty_workspace_mask_is_a_controlled_error(self):
        area = WorkingArea.from_rectangle_bounds((20.5, 20.5, 25.5, 25.5))

        with self.assertRaisesRegex(ValueError, "processing mask is empty"):
            find_preliminary_contour_for_working_area(_two_part_grid(), area)

    def test_contour_stage_needs_only_in_memory_grid_and_working_area(self):
        grid = _two_part_grid()
        area = WorkingArea.from_rectangle_bounds((4.5, 4.5, 15.5, 15.5))

        result = find_preliminary_contour_for_working_area(grid, area)

        self.assertGreater(result.contour.point_count, 0)
        self.assertIs(result.masks.grid, grid)

    def test_new_working_area_invalidates_previous_contour(self):
        grid = _two_part_grid()
        area_a = WorkingArea.from_rectangle_bounds((4.5, 4.5, 15.5, 15.5))
        area_b = WorkingArea.from_rectangle_bounds((29.5, 4.5, 48.5, 20.5))
        session = object()
        contour = find_preliminary_contour_for_working_area(
            grid,
            area_a,
            density_session=session,
        )
        state = {
            "contour_processing_result": contour,
            "mask_editing_session": MaskEditingSession.from_preliminary_contour(
                contour
            ),
            "contour_points": list(map(tuple, contour.contour.contour_world)),
            "contour_file": "",
            "processing_mask_preview": np.ones((2, 2), dtype=np.uint8),
        }

        apply_working_area_state(state, area_b, session)

        self.assertIsNone(state["contour_processing_result"])
        self.assertIsNone(state["mask_editing_session"])
        self.assertIsNone(state["processing_mask_preview"])
        self.assertEqual(state["contour_points"], [])
        self.assertIs(state["active_working_area"], area_b)

    def test_new_density_state_invalidates_previous_contour(self):
        grid = _two_part_grid()
        area = WorkingArea.from_rectangle_bounds((4.5, 4.5, 15.5, 15.5))
        contour = find_preliminary_contour_for_working_area(grid, area)
        state = {
            "contour_processing_result": contour,
            "mask_editing_session": MaskEditingSession.from_preliminary_contour(
                contour
            ),
            "contour_points": list(map(tuple, contour.contour.contour_world)),
            "contour_file": "",
            "processing_mask_preview": np.ones((2, 2), dtype=np.uint8),
        }

        reset_source_dependent_state(state)

        self.assertIsNone(state["contour_processing_result"])
        self.assertIsNone(state["mask_editing_session"])
        self.assertIsNone(state["processing_mask_preview"])
        self.assertEqual(state["contour_points"], [])


if __name__ == "__main__":
    unittest.main()
