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
from app.core.xyz_reader import PointCloudStats  # noqa: E402
from app.core.working_area import WorkingArea  # noqa: E402
from app.services.coarse_processing import DensityProcessingResult  # noqa: E402
from app.ui import contour_workflow  # noqa: E402
from app.ui.contour_workflow import (  # noqa: E402
    MaskEditingSession,
    PreliminaryContourParameters,
    find_preliminary_contour_for_working_area,
    mask_edits_for_world_segment,
    preliminary_contour_parameters_for_threshold,
    processing_mask_to_preview,
    rebuild_preliminary_contour_from_edited_mask,
    rebase_preliminary_contour_with_edits,
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


def _fill_holes_grid() -> DensityGrid:
    density = np.zeros((32, 32), dtype=np.uint32)
    density[2:30, 2:30] = 5
    density[7:9, 7:9] = 0
    density[14:20, 14:20] = 0
    return DensityGrid(
        density=density,
        cell_size=1.0,
        min_x=0.0,
        min_y=0.0,
    )


class TestGuiContourWorkflow(unittest.TestCase):
    def _density_session(self, grid: DensityGrid) -> DensityProcessingResult:
        return DensityProcessingResult(
            source_path=Path("unused.xyz"),
            stats=PointCloudStats(
                file_path=Path("unused.xyz"),
                point_count=int(grid.density.sum()),
                min_x=grid.min_x,
                max_x=grid.min_x + grid.width * grid.cell_size,
                min_y=grid.min_y,
                max_y=grid.min_y + grid.height * grid.cell_size,
                min_z=0.0,
                max_z=0.0,
            ),
            grid=grid,
            stats_from_cache=False,
            density_from_cache=False,
        )

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

    def test_threshold_rebase_replays_add_remove_and_preserves_undo(self):
        grid = _threshold_comparison_grid()
        area = _threshold_comparison_area()
        edits = [
            {
                "stroke_id": 1,
                "mode": "remove",
                "x": 5.5,
                "y": 12.5,
                "radius_mm": 1.0,
            },
            {
                "stroke_id": 2,
                "mode": "add",
                "x": 5.5,
                "y": 5.5,
                "radius_mm": 1.0,
            },
        ]

        with (
            patch("app.core.xyz_reader.iter_xyz_points") as source_reader,
            patch("app.core.xyz_reader.compute_stats") as compute_stats,
            patch("app.core.density_grid.build_density_grid") as build_density,
        ):
            result, editing = rebase_preliminary_contour_with_edits(
                grid,
                area,
                edits,
                parameters=preliminary_contour_parameters_for_threshold(
                    "Manual",
                    5.0,
                ),
            )

        source_reader.assert_not_called()
        compute_stats.assert_not_called()
        build_density.assert_not_called()
        self.assertEqual(editing.edited_mask[12, 5], 0)
        self.assertEqual(editing.edited_mask[5, 5], 1)
        self.assertEqual(len(editing.history), 2)
        np.testing.assert_array_equal(result.masks.contour_mask, editing.edited_mask)
        self.assertTrue(editing.undo_last_stroke())
        self.assertEqual(editing.edited_mask[5, 5], editing.base_mask[5, 5])

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

    def test_fill_holes_area_preserves_prefill_mask_and_fills_only_in_range(self):
        grid = _fill_holes_grid()
        disabled = find_preliminary_contour_for_working_area(
            grid,
            None,
            parameters=PreliminaryContourParameters(
                threshold_mode="manual",
                manual_threshold=1.0,
                fill_holes_area=0,
            ),
        )
        below_small_hole = find_preliminary_contour_for_working_area(
            grid,
            None,
            parameters=PreliminaryContourParameters(
                threshold_mode="manual",
                manual_threshold=1.0,
                fill_holes_area=3,
            ),
        )
        filled = find_preliminary_contour_for_working_area(
            grid,
            None,
            parameters=PreliminaryContourParameters(
                threshold_mode="manual",
                manual_threshold=1.0,
                fill_holes_area=4,
            ),
        )

        np.testing.assert_array_equal(
            disabled.masks.mask_for_holes,
            disabled.masks.contour_mask,
        )
        self.assertEqual(disabled.masks.contour_mask[7, 7], 0)
        self.assertEqual(below_small_hole.masks.contour_mask[7, 7], 0)
        self.assertEqual(filled.masks.mask_for_holes[7, 7], 0)
        self.assertEqual(filled.masks.contour_mask[7, 7], 1)
        self.assertEqual(filled.masks.mask_for_holes[15, 15], 0)
        self.assertEqual(filled.masks.contour_mask[15, 15], 0)
        self.assertEqual(filled.parameters.fill_holes_area, 4)
        self.assertIs(filled.masks.grid, grid)

        disabled_preview = processing_mask_to_preview(
            disabled.masks.contour_mask,
            preview_width=32,
            preview_height=32,
        )
        filled_preview = processing_mask_to_preview(
            filled.masks.contour_mask,
            preview_width=32,
            preview_height=32,
        )
        self.assertFalse(np.array_equal(disabled_preview, filled_preview))
        self.assertEqual(filled_preview[32 - 1 - 7, 7], 1)

    def test_working_area_is_applied_before_fill_and_keeps_cut_hole_open(self):
        grid = _fill_holes_grid()
        area = WorkingArea.from_rectangle_bounds((8.5, 2.5, 29.5, 29.5))
        result = find_preliminary_contour_for_working_area(
            grid,
            area,
            parameters=PreliminaryContourParameters(
                threshold_mode="manual",
                manual_threshold=1.0,
                fill_holes_area=100,
            ),
        )

        self.assertEqual(result.masks.mask_for_holes[7, 8], 0)
        self.assertEqual(result.masks.contour_mask[7, 8], 0)
        self.assertEqual(result.masks.mask_for_holes[15, 15], 0)
        self.assertEqual(result.masks.contour_mask[15, 15], 1)

    def test_fill_holes_refind_reuses_grid_without_source_or_density(self):
        grid = _fill_holes_grid()

        with (
            patch("app.core.xyz_reader.iter_xyz_points") as source_reader,
            patch("app.core.xyz_reader.compute_stats") as compute_stats,
            patch("app.core.density_grid.build_density_grid") as build_density,
        ):
            disabled = find_preliminary_contour_for_working_area(
                grid,
                None,
                parameters=PreliminaryContourParameters(
                    threshold_mode="manual",
                    manual_threshold=1.0,
                    fill_holes_area=0,
                ),
            )
            filled = find_preliminary_contour_for_working_area(
                grid,
                None,
                parameters=PreliminaryContourParameters(
                    threshold_mode="manual",
                    manual_threshold=1.0,
                    fill_holes_area=4,
                ),
            )

        self.assertIs(disabled.masks.grid, grid)
        self.assertIs(filled.masks.grid, grid)
        self.assertFalse(
            np.array_equal(
                disabled.masks.contour_mask,
                filled.masks.contour_mask,
            )
        )
        source_reader.assert_not_called()
        compute_stats.assert_not_called()
        build_density.assert_not_called()

    def test_invalid_fill_holes_area_is_rejected_before_processing(self):
        grid = _fill_holes_grid()

        with patch.object(contour_workflow, "build_processing_masks") as build_masks:
            for invalid_area in (-1, 1.5, float("inf"), True):
                with self.subTest(invalid_area=invalid_area):
                    invalid = PreliminaryContourParameters(
                        threshold_mode="manual",
                        manual_threshold=1.0,
                        fill_holes_area=invalid_area,
                    )
                    with self.assertRaises(ValueError):
                        find_preliminary_contour_for_working_area(
                            grid,
                            None,
                            parameters=invalid,
                        )

        build_masks.assert_not_called()

    def test_fill_holes_control_is_pending_until_find_and_invalid_preserves_result(self):
        from app.ui import viewer_app

        grid = _fill_holes_grid()
        density_session = self._density_session(grid)
        previous = find_preliminary_contour_for_working_area(
            grid,
            None,
            density_session=density_session,
            parameters=PreliminaryContourParameters(
                threshold_mode="manual",
                manual_threshold=1.0,
                fill_holes_area=0,
            ),
        )
        target = {
            **viewer_app.state,
            "density_result": density_session,
            "active_working_area": None,
            "working_area_density_session": None,
            "contour_processing_result": previous,
            "contour_points": list(map(tuple, previous.contour.contour_world)),
        }

        def selected_value(tag):
            return {
                viewer_app.CONTOUR_THRESHOLD_MODE_TAG: "Manual",
                viewer_app.CONTOUR_MANUAL_THRESHOLD_TAG: 1.0,
                viewer_app.HOLE_KEEP_LARGEST_TAG: False,
                viewer_app.CONTOUR_FILL_HOLES_AREA_TAG: 4,
            }[tag]

        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(viewer_app.dpg, "get_value", side_effect=selected_value),
            patch.object(viewer_app, "_set_status"),
            patch.object(viewer_app, "rebase_preliminary_contour_with_edits") as rebase,
        ):
            selected = viewer_app._selected_contour_parameters()
            viewer_app._contour_fill_holes_settings_callback()

            self.assertEqual(selected.fill_holes_area, 4)
            self.assertIs(viewer_app.state["contour_processing_result"], previous)

        rebase.assert_not_called()

        def invalid_value(tag):
            if tag == viewer_app.CONTOUR_FILL_HOLES_AREA_TAG:
                return -1
            return selected_value(tag)

        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(viewer_app.dpg, "get_value", side_effect=invalid_value),
            patch.object(viewer_app, "_set_status"),
            patch.object(viewer_app, "rebase_preliminary_contour_with_edits") as rebase,
        ):
            viewer_app._find_contour_callback()
            self.assertIs(viewer_app.state["contour_processing_result"], previous)

        rebase.assert_not_called()

    def test_contour_info_reports_applied_fill_holes_area(self):
        from app.ui import viewer_app

        result = find_preliminary_contour_for_working_area(
            _fill_holes_grid(),
            None,
            parameters=PreliminaryContourParameters(
                threshold_mode="manual",
                manual_threshold=1.0,
                fill_holes_area=4,
            ),
        )
        target = {
            **viewer_app.state,
            "contour_processing_result": result,
            "mask_editing_session": MaskEditingSession.from_preliminary_contour(
                result
            ),
        }
        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(viewer_app.dpg, "set_value") as set_value,
        ):
            viewer_app._update_contour_info()

        displayed = set_value.call_args.args[1]
        self.assertIn("Fill holes max area: 4 cells", displayed)

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
            if tag == viewer_app.HOLE_KEEP_LARGEST_TAG:
                return False
            if tag == viewer_app.CONTOUR_FILL_HOLES_AREA_TAG:
                return 0
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

    def test_keep_largest_parameter_uses_production_mask_processing_path(self):
        grid = _two_part_grid()
        parameters = PreliminaryContourParameters(
            threshold_mode="manual",
            manual_threshold=1.0,
            keep_largest=True,
        )

        with patch.object(
            contour_workflow,
            "build_processing_masks",
            wraps=contour_workflow.build_processing_masks,
        ) as build_masks:
            result = find_preliminary_contour_for_working_area(
                grid,
                None,
                parameters=parameters,
            )

        self.assertTrue(build_masks.call_args.kwargs["keep_largest"])
        self.assertFalse(np.any(result.masks.contour_mask[5:15, 5:15]))
        self.assertTrue(np.any(result.masks.contour_mask[5:20, 30:48]))

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

    def test_missing_working_area_uses_full_scan(self):
        grid = _two_part_grid()

        result = find_preliminary_contour_for_working_area(grid, None)
        editing = MaskEditingSession.from_preliminary_contour(result)

        self.assertIsNone(result.working_area)
        self.assertGreater(result.contour.point_count, 0)
        self.assertGreater(int(result.masks.contour_mask[:, 30:].sum()), 0)
        self.assertTrue(np.all(editing.working_area_mask))

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

    def test_unified_undo_uses_chronological_actions_not_current_tool(self):
        from app.ui import viewer_app

        density_session = object()
        target = {
            "density_result": density_session,
            "polygon_points": [(1.0, 1.0), (2.0, 2.0)],
            "polygon_finished": False,
            "selection_applied": False,
            "editing_overlay_visible": True,
            "mode": "mask_brush",
            "undo_history": [
                {
                    "kind": "polygon_point",
                    "point_count": 1,
                    "density_session": density_session,
                },
                {
                    "kind": "polygon_point",
                    "point_count": 2,
                    "density_session": density_session,
                },
            ],
        }
        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(viewer_app, "_update_polygon_points_text"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app.dpg, "set_value"),
        ):
            self.assertTrue(viewer_app._undo_user_action())
            self.assertEqual(viewer_app.state["polygon_points"], [(1.0, 1.0)])

    def test_unified_undo_prefers_latest_brush_then_hidden_polygon_draft(self):
        from app.ui import viewer_app

        result, editing = self._mask_editing_fixture()
        density_session = type("DensitySession", (), {"grid": result.masks.grid})()
        brush_edit = {
            "stroke_id": 2,
            "mode": "remove",
            "x": 5.5,
            "y": 5.5,
            "radius_mm": 0.5,
        }
        editing.begin_stroke()
        editing.apply_edit(brush_edit)
        editing.finish_stroke()
        target = {
            "density_result": density_session,
            "active_working_area": result.working_area,
            "working_area_density_session": density_session,
            "contour_processing_result": result,
            "mask_editing_session": editing,
            "mask_edits": [brush_edit],
            "polygon_points": [(1.0, 1.0)],
            "polygon_finished": False,
            "selection_applied": True,
            "editing_overlay_visible": False,
            "mode": "mask_brush",
            "active_brush_stroke_id": None,
            "last_brush_image": None,
            "last_brush_world": None,
            "undo_history": [
                {
                    "kind": "polygon_point",
                    "point_count": 1,
                    "density_session": density_session,
                },
                {
                    "kind": "mask_stroke",
                    "stroke_id": 2,
                    "density_session": density_session,
                    "mask_editing_session": editing,
                },
            ],
        }
        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(viewer_app, "_refresh_contour_from_current_edits", return_value=True),
            patch.object(viewer_app, "_update_mask_edits_count"),
            patch.object(viewer_app, "_update_last_brush_debug"),
            patch.object(viewer_app, "_update_polygon_points_text"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
            patch.object(viewer_app.dpg, "set_value"),
        ):
            self.assertTrue(viewer_app._undo_user_action())
            self.assertEqual(viewer_app.state["mask_edits"], [])
            self.assertEqual(viewer_app.state["polygon_points"], [(1.0, 1.0)])

            self.assertTrue(viewer_app._undo_user_action())
            self.assertEqual(viewer_app.state["polygon_points"], [])

    def test_completed_brush_stroke_triggers_one_contour_refresh(self):
        from app.ui import viewer_app

        result, editing = self._mask_editing_fixture()
        density_session = type("DensitySession", (), {"grid": result.masks.grid})()
        edit = {
            "stroke_id": 1,
            "mode": "remove",
            "x": 5.5,
            "y": 5.5,
            "radius_mm": 0.5,
        }
        editing.begin_stroke()
        editing.apply_edit(edit)
        target = {
            "density_result": density_session,
            "active_working_area": result.working_area,
            "working_area_density_session": density_session,
            "contour_processing_result": result,
            "mask_editing_session": editing,
            "mask_edits": [edit],
            "active_brush_stroke_id": 1,
            "undo_history": [],
        }
        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(
                viewer_app,
                "_refresh_contour_from_current_edits",
                return_value=True,
            ) as refresh,
            patch.object(viewer_app, "_set_status"),
        ):
            self.assertTrue(viewer_app._finish_active_mask_stroke())
            self.assertEqual(len(viewer_app.state["undo_history"]), 1)

        refresh.assert_called_once_with()

    def test_threshold_refresh_is_inactive_before_first_find(self):
        from app.ui import viewer_app

        target = {
            "density_result": object(),
            "contour_processing_result": None,
        }
        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(
                viewer_app,
                "rebase_preliminary_contour_with_edits",
            ) as rebase,
        ):
            self.assertFalse(
                viewer_app._refresh_contour_for_settings(preserve_edits=True)
            )

        rebase.assert_not_called()

    def test_keep_largest_commit_rebases_in_memory_and_invalidates_holes(self):
        from app.ui import viewer_app

        grid = _two_part_grid()
        density_session = self._density_session(grid)
        contour = find_preliminary_contour_for_working_area(
            grid,
            None,
            density_session=density_session,
            parameters=PreliminaryContourParameters(
                threshold_mode="manual",
                manual_threshold=1.0,
            ),
        )
        editing = MaskEditingSession.from_preliminary_contour(contour)
        target = {
            **viewer_app.state,
            "density_result": density_session,
            "active_working_area": None,
            "working_area_density_session": None,
            "contour_processing_result": contour,
            "mask_editing_session": editing,
            "mask_edits": [],
            "undo_history": [],
            "contour_points": list(map(tuple, contour.contour.contour_world)),
            "coarse_mask_revision": 6,
            "hole_detection_session": object(),
            "holes_outdated": True,
        }

        def selected_value(tag):
            if tag == viewer_app.CONTOUR_THRESHOLD_MODE_TAG:
                return "Manual"
            if tag == viewer_app.CONTOUR_MANUAL_THRESHOLD_TAG:
                return 1.0
            if tag == viewer_app.HOLE_KEEP_LARGEST_TAG:
                return True
            if tag == viewer_app.CONTOUR_FILL_HOLES_AREA_TAG:
                return 0
            raise AssertionError(tag)

        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(viewer_app.dpg, "get_value", side_effect=selected_value),
            patch.object(
                contour_workflow,
                "build_processing_masks",
                wraps=contour_workflow.build_processing_masks,
            ) as build_masks,
            patch.object(
                viewer_app,
                "_refresh_contour_for_settings",
                wraps=viewer_app._refresh_contour_for_settings,
            ) as refresh,
            patch.object(viewer_app, "_update_processing_mask_texture"),
            patch.object(viewer_app, "_set_mask_edit_controls_enabled"),
            patch.object(viewer_app, "_update_mask_edits_count"),
            patch.object(viewer_app, "_update_contour_info"),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
            patch("app.core.xyz_reader.iter_xyz_points") as source_reader,
            patch("app.core.xyz_reader.compute_stats") as compute_stats,
            patch("app.core.density_grid.build_density_grid") as build_density,
        ):
            viewer_app._keep_largest_component_callback()

            updated = viewer_app.state["contour_processing_result"]
            self.assertTrue(updated.parameters.keep_largest)
            self.assertIs(updated.masks.grid, grid)
            self.assertFalse(np.any(updated.masks.contour_mask[5:15, 5:15]))
            self.assertTrue(np.any(updated.masks.contour_mask[5:20, 30:48]))
            self.assertIsNone(viewer_app.state["hole_detection_session"])
            self.assertTrue(viewer_app.state["holes_outdated"])
            self.assertEqual(viewer_app.state["coarse_mask_revision"], 7)

        refresh.assert_called_once_with(preserve_edits=True)
        self.assertTrue(build_masks.call_args.kwargs["keep_largest"])
        source_reader.assert_not_called()
        compute_stats.assert_not_called()
        build_density.assert_not_called()

    def test_ctrl_z_dispatches_the_same_unified_undo(self):
        from app.ui import viewer_app

        def is_key_down(key):
            return key == viewer_app.dpg.mvKey_LControl

        with (
            patch.object(viewer_app.dpg, "is_key_down", side_effect=is_key_down),
            patch.object(viewer_app, "_undo_user_action") as undo,
        ):
            viewer_app._key_press_callback(app_data=viewer_app.dpg.mvKey_Z)

        undo.assert_called_once_with()

    def test_unified_undo_discards_actions_from_old_density(self):
        from app.ui import viewer_app

        target = {
            "density_result": object(),
            "polygon_points": [(1.0, 1.0)],
            "undo_history": [
                {
                    "kind": "polygon_point",
                    "point_count": 1,
                    "density_session": object(),
                }
            ],
        }
        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(viewer_app, "_set_status"),
        ):
            self.assertFalse(viewer_app._undo_user_action())
            self.assertEqual(viewer_app.state["polygon_points"], [(1.0, 1.0)])
            self.assertEqual(viewer_app.state["undo_history"], [])

    def test_undo_apply_restores_polygon_draft_then_undoes_last_point(self):
        from app.ui import viewer_app

        density_session = self._density_session(_two_part_grid())
        points = [(4.5, 4.5), (15.5, 4.5), (15.5, 15.5), (4.5, 15.5)]
        point_actions = [
            {
                "kind": "polygon_point",
                "point_count": count,
                "density_session": density_session,
            }
            for count in range(1, len(points) + 1)
        ]
        target = {
            **viewer_app.state,
            "density_result": density_session,
            "active_working_area": None,
            "working_area_density_session": None,
            "mode": "polygon",
            "polygon_points": list(points),
            "polygon_finished": True,
            "selection_applied": False,
            "editing_overlay_visible": True,
            "undo_history": point_actions,
            "contour_processing_result": None,
        }
        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(viewer_app, "_build_selection_inside_mask", return_value=np.ones((2, 2), dtype=np.uint8)),
            patch.object(viewer_app, "_update_selection_texture"),
            patch.object(viewer_app, "_delete_processing_mask_texture"),
            patch.object(viewer_app, "_set_mask_edit_controls_enabled"),
            patch.object(viewer_app, "_update_mask_edits_count"),
            patch.object(viewer_app, "_redraw_brush_cursor_overlay"),
            patch.object(viewer_app, "_update_working_area_info"),
            patch.object(viewer_app, "_update_contour_info"),
            patch.object(viewer_app, "_update_polygon_points_text"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app.dpg, "set_value"),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=False),
        ):
            viewer_app._apply_selection_callback()
            self.assertIsInstance(viewer_app.get_active_working_area(), WorkingArea)

            self.assertTrue(viewer_app._undo_user_action())
            self.assertIsNone(viewer_app.get_active_working_area())
            self.assertEqual(viewer_app.state["polygon_points"], points)
            self.assertEqual(viewer_app.state["mode"], "polygon")
            self.assertTrue(viewer_app.state["editing_overlay_visible"])

            self.assertTrue(viewer_app._undo_user_action())
            self.assertEqual(viewer_app.state["polygon_points"], points[:-1])

    def test_undo_apply_restores_previous_area_and_rebuilds_its_contour(self):
        from app.ui import viewer_app

        grid = _two_part_grid()
        density_session = self._density_session(grid)
        area_a = WorkingArea.from_rectangle_bounds((4.5, 4.5, 15.5, 15.5))
        points_b = [(29.5, 4.5), (48.5, 4.5), (48.5, 20.5), (29.5, 20.5)]
        contour_a = find_preliminary_contour_for_working_area(
            grid,
            area_a,
            density_session=density_session,
        )
        target = {
            **viewer_app.state,
            "density_result": density_session,
            "active_working_area": area_a,
            "working_area_density_session": density_session,
            "mode": "polygon",
            "polygon_points": list(points_b),
            "polygon_finished": True,
            "selection_applied": True,
            "editing_overlay_visible": True,
            "undo_history": [
                {
                    "kind": "polygon_point",
                    "point_count": count,
                    "density_session": density_session,
                }
                for count in range(1, len(points_b) + 1)
            ],
            "contour_processing_result": contour_a,
        }
        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(viewer_app, "_build_selection_inside_mask", return_value=np.ones((2, 2), dtype=np.uint8)),
            patch.object(viewer_app, "_update_selection_texture"),
            patch.object(viewer_app, "_delete_processing_mask_texture"),
            patch.object(viewer_app, "_set_mask_edit_controls_enabled"),
            patch.object(viewer_app, "_update_mask_edits_count"),
            patch.object(viewer_app, "_redraw_brush_cursor_overlay"),
            patch.object(viewer_app, "_update_working_area_info"),
            patch.object(viewer_app, "_update_contour_info"),
            patch.object(viewer_app, "_update_polygon_points_text"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_update_processing_mask_texture"),
            patch.object(
                viewer_app,
                "_selected_contour_parameters",
                return_value=PreliminaryContourParameters(),
            ),
            patch.object(viewer_app, "_set_status"),
            patch.object(viewer_app.dpg, "set_value"),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=False),
            patch("app.core.xyz_reader.iter_xyz_points") as source_reader,
            patch("app.core.xyz_reader.compute_stats") as compute_stats,
            patch("app.core.density_grid.build_density_grid") as build_density,
        ):
            viewer_app._apply_selection_callback()
            self.assertNotEqual(viewer_app.get_active_working_area(), area_a)

            self.assertTrue(viewer_app._undo_user_action())
            self.assertEqual(viewer_app.get_active_working_area(), area_a)
            self.assertEqual(viewer_app.state["polygon_points"], points_b)
            self.assertEqual(
                viewer_app.state["contour_processing_result"].working_area,
                area_a,
            )
        source_reader.assert_not_called()
        compute_stats.assert_not_called()
        build_density.assert_not_called()

    def test_undo_clear_then_undo_apply_preserves_history_order(self):
        from app.ui import viewer_app

        density_session = self._density_session(_two_part_grid())
        points = [(4.5, 4.5), (15.5, 4.5), (15.5, 15.5), (4.5, 15.5)]
        point_actions = [
            {
                "kind": "polygon_point",
                "point_count": count,
                "density_session": density_session,
            }
            for count in range(1, len(points) + 1)
        ]
        target = {
            **viewer_app.state,
            "density_result": density_session,
            "active_working_area": None,
            "working_area_density_session": None,
            "mode": "polygon",
            "polygon_points": list(points),
            "polygon_finished": True,
            "selection_applied": False,
            "editing_overlay_visible": True,
            "undo_history": point_actions,
            "contour_processing_result": None,
        }
        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(viewer_app, "_build_selection_inside_mask", return_value=np.ones((2, 2), dtype=np.uint8)),
            patch.object(viewer_app, "_update_selection_texture"),
            patch.object(viewer_app, "_delete_processing_mask_texture"),
            patch.object(viewer_app, "_set_mask_edit_controls_enabled"),
            patch.object(viewer_app, "_update_mask_edits_count"),
            patch.object(viewer_app, "_redraw_brush_cursor_overlay"),
            patch.object(viewer_app, "_update_working_area_info"),
            patch.object(viewer_app, "_update_contour_info"),
            patch.object(viewer_app, "_update_polygon_points_text"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app.dpg, "set_value"),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=False),
        ):
            viewer_app._apply_selection_callback()
            applied = viewer_app.get_active_working_area()
            viewer_app._clear_selection_callback()
            self.assertIsNone(viewer_app.get_active_working_area())

            self.assertTrue(viewer_app._undo_user_action())
            self.assertEqual(viewer_app.get_active_working_area(), applied)

            self.assertTrue(viewer_app._undo_user_action())
            self.assertIsNone(viewer_app.get_active_working_area())
            self.assertEqual(viewer_app.state["polygon_points"], points)

    def test_undo_rectangle_apply_restores_rectangle_draft(self):
        from app.ui import viewer_app

        density_session = self._density_session(_two_part_grid())
        bounds = (4.5, 4.5, 15.5, 15.5)
        target = {
            **viewer_app.state,
            "density_result": density_session,
            "active_working_area": None,
            "working_area_density_session": None,
            "mode": "rectangle",
            "rectangle_roi": bounds,
            "roi_first_world": None,
            "roi_current_world": None,
            "selection_applied": False,
            "editing_overlay_visible": True,
            "undo_history": [],
            "contour_processing_result": None,
        }
        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(viewer_app, "_build_selection_inside_mask", return_value=np.ones((2, 2), dtype=np.uint8)),
            patch.object(viewer_app, "_update_selection_texture"),
            patch.object(viewer_app, "_delete_processing_mask_texture"),
            patch.object(viewer_app, "_set_mask_edit_controls_enabled"),
            patch.object(viewer_app, "_update_mask_edits_count"),
            patch.object(viewer_app, "_redraw_brush_cursor_overlay"),
            patch.object(viewer_app, "_update_working_area_info"),
            patch.object(viewer_app, "_update_contour_info"),
            patch.object(viewer_app, "_update_polygon_points_text"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app.dpg, "set_value"),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=False),
        ):
            viewer_app._apply_selection_callback()
            self.assertTrue(viewer_app._undo_user_action())

            self.assertIsNone(viewer_app.get_active_working_area())
            self.assertEqual(viewer_app.state["rectangle_roi"], bounds)
            self.assertEqual(viewer_app.state["mode"], "rectangle")
            self.assertTrue(viewer_app.state["editing_overlay_visible"])

    def test_brush_undo_precedes_working_area_apply_undo(self):
        from app.ui import viewer_app

        density_session = object()
        editing_session = object()
        target = {
            "density_result": density_session,
            "mask_editing_session": editing_session,
            "undo_history": [
                {
                    "kind": "working_area_apply",
                    "density_session": density_session,
                },
                {
                    "kind": "mask_stroke",
                    "density_session": density_session,
                    "mask_editing_session": editing_session,
                },
            ],
        }
        with (
            patch.dict(viewer_app.state, target, clear=True),
            patch.object(
                viewer_app,
                "_undo_last_brush_stroke",
                return_value=True,
            ) as undo_brush,
            patch.object(
                viewer_app,
                "_restore_working_area_undo_action",
                return_value=True,
            ) as undo_area,
        ):
            self.assertTrue(viewer_app._undo_user_action())
            undo_brush.assert_called_once_with(finish_active=False)
            undo_area.assert_not_called()

            self.assertTrue(viewer_app._undo_user_action())
            undo_area.assert_called_once()


if __name__ == "__main__":
    unittest.main()
