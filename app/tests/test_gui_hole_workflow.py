import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.density_grid import DensityGrid  # noqa: E402
from app.core.working_area import WorkingArea  # noqa: E402
from app.ui import hole_workflow  # noqa: E402
from app.ui.contour_workflow import (  # noqa: E402
    MaskEditingSession,
    PreliminaryContourParameters,
    find_preliminary_contour_for_working_area,
    mask_edits_for_world_segment,
)
from app.ui.density_workflow import (  # noqa: E402
    apply_working_area_state,
    reset_source_dependent_state,
)
from app.ui.hole_workflow import (  # noqa: E402
    HoleDetectionParameters,
    current_effective_hole_mask,
    find_holes_for_current_mask,
    hole_detection_session_is_current,
    invalidate_hole_detection_state,
    validate_hole_detection_parameters,
)


def _coarse_mask(*, with_hole: bool = True) -> np.ndarray:
    mask = np.zeros((61, 61), dtype=np.uint8)
    mask[2:-2, 2:-2] = 1
    if with_hole:
        cv2.circle(mask, (30, 30), 6, 0, -1)
    return mask


def _many_tiny_holes_mask() -> tuple[np.ndarray, int, int]:
    mask = np.ones((721, 721), dtype=np.uint8)
    tiny_count = 0
    for row in range(30):
        for column in range(34):
            cv2.circle(
                mask,
                (10 + column * 20, 10 + row * 20),
                1,
                0,
                -1,
            )
            tiny_count += 1

    proper_centers = ((100, 680), (220, 680), (340, 680), (460, 680), (580, 680))
    for center in proper_centers:
        cv2.circle(mask, center, 3, 0, -1)
    return mask, tiny_count, len(proper_centers)


def _coarse_session(
    mask: np.ndarray,
    working_area: WorkingArea | None = None,
    *,
    fill_holes_area: int = 0,
):
    grid = DensityGrid(
        density=mask.astype(np.uint32),
        cell_size=1.0,
        min_x=0.0,
        min_y=0.0,
    )
    density_session = object()
    contour = find_preliminary_contour_for_working_area(
        grid,
        working_area,
        density_session=density_session,
        parameters=PreliminaryContourParameters(
            threshold_mode="manual",
            manual_threshold=1.0,
            fill_holes_area=fill_holes_area,
        ),
    )
    editing = MaskEditingSession.from_preliminary_contour(contour)
    return density_session, contour, editing


def _find(
    density_session: object,
    contour,
    editing: MaskEditingSession,
    *,
    revision: int = 1,
):
    return find_holes_for_current_mask(
        contour,
        editing,
        density_session=density_session,
        coarse_mask_revision=revision,
        parameters=HoleDetectionParameters(min_diameter_mm=0.0),
    )


def _stroke(
    editing: MaskEditingSession,
    *,
    mode: str,
    start: tuple[float, float],
    end: tuple[float, float] | None = None,
    diameter: float = 15.0,
    stroke_id: int = 1,
) -> None:
    editing.begin_stroke()
    for edit in mask_edits_for_world_segment(
        editing.grid,
        start,
        end or start,
        mode=mode,
        brush_diameter_mm=diameter,
        stroke_id=stroke_id,
    ):
        editing.apply_edit(edit)
    editing.finish_stroke()


class TestGuiHoleWorkflow(unittest.TestCase):
    def test_parameter_validation_and_production_defaults(self):
        default = HoleDetectionParameters()
        self.assertEqual(default.min_diameter_mm, 8.0)
        self.assertIsNone(default.max_diameter_mm)
        self.assertEqual(default.min_circularity, 0.55)
        self.assertEqual(default.max_aspect_ratio_deviation, 0.35)
        self.assertEqual(default.max_error_ratio, 0.18)
        self.assertEqual(default.group_tolerance_mm, 1.5)

        self.assertIsNone(
            validate_hole_detection_parameters(0, 0).max_diameter_mm
        )
        self.assertIsNone(
            validate_hole_detection_parameters(0, None).max_diameter_mm
        )
        configured = validate_hole_detection_parameters(4, 10, 0.35, 0.8)
        self.assertEqual(configured.min_circularity, 0.35)
        self.assertEqual(configured.max_error_ratio, 0.8)
        for minimum, maximum in ((-1, 0), (float("nan"), 0), (5, 4)):
            with self.subTest(minimum=minimum, maximum=maximum):
                with self.assertRaises(ValueError):
                    validate_hole_detection_parameters(minimum, maximum)
        for circularity, error_ratio in (
            (-0.1, 0.18),
            (float("nan"), 0.18),
            (0.55, -0.1),
            (0.55, float("inf")),
        ):
            with self.subTest(
                circularity=circularity,
                error_ratio=error_ratio,
            ):
                with self.assertRaises(ValueError):
                    validate_hole_detection_parameters(
                        0,
                        0,
                        circularity,
                        error_ratio,
                    )

    def test_find_holes_uses_current_edited_mask_not_stale_prefill_mask(self):
        density, contour, editing = _coarse_session(_coarse_mask())
        self.assertEqual(len(_find(density, contour, editing).result.candidates), 1)

        _stroke(editing, mode="add", start=(30.5, 30.5))
        self.assertEqual(contour.masks.mask_for_holes[30, 30], 0)
        self.assertEqual(current_effective_hole_mask(contour, editing)[30, 30], 1)
        self.assertEqual(_find(density, contour, editing).result.candidates, [])

    def test_remove_can_create_closed_hole_but_channel_to_border_cannot(self):
        density, contour, editing = _coarse_session(
            _coarse_mask(with_hole=False)
        )
        _stroke(editing, mode="remove", start=(30.5, 30.5))
        self.assertEqual(len(_find(density, contour, editing).result.candidates), 1)

        _stroke(
            editing,
            mode="remove",
            start=(30.5, 30.5),
            end=(1.5, 30.5),
            diameter=3.0,
            stroke_id=2,
        )
        self.assertEqual(_find(density, contour, editing).result.candidates, [])

    def test_undo_and_clear_edits_restore_hole_detection_from_effective_mask(self):
        density, contour, editing = _coarse_session(_coarse_mask())
        _stroke(editing, mode="add", start=(30.5, 30.5))
        self.assertEqual(_find(density, contour, editing).result.candidates, [])

        self.assertTrue(editing.undo_last_stroke())
        self.assertEqual(len(_find(density, contour, editing).result.candidates), 1)

        _stroke(editing, mode="add", start=(30.5, 30.5), stroke_id=2)
        self.assertTrue(editing.clear_edits())
        self.assertEqual(len(_find(density, contour, editing).result.candidates), 1)

    def test_full_scan_rectangle_and_polygon_working_areas(self):
        mask = _coarse_mask()
        full_density, full_contour, full_editing = _coarse_session(mask)
        self.assertEqual(
            len(_find(full_density, full_contour, full_editing).result.candidates),
            1,
        )

        areas = (
            WorkingArea.from_rectangle_bounds((15.5, 15.5, 45.5, 45.5)),
            WorkingArea.from_polygon(
                [(15.5, 15.5), (45.5, 15.5), (45.5, 45.5), (15.5, 45.5)]
            ),
        )
        for area in areas:
            with self.subTest(kind=area.kind):
                density, contour, editing = _coarse_session(mask, area)
                self.assertEqual(
                    len(_find(density, contour, editing).result.candidates),
                    1,
                )

    def test_working_area_cutting_hole_opens_it_to_outer_background(self):
        area = WorkingArea.from_rectangle_bounds((30.5, 15.5, 45.5, 45.5))
        density, contour, editing = _coarse_session(_coarse_mask(), area)
        self.assertEqual(_find(density, contour, editing).result.candidates, [])

    def test_session_binds_density_contour_editing_area_revision_and_parameters(self):
        area = WorkingArea.from_rectangle_bounds((15.5, 15.5, 45.5, 45.5))
        density, contour, editing = _coarse_session(_coarse_mask(), area)
        session = _find(density, contour, editing, revision=7)

        self.assertIs(session.density_session, density)
        self.assertIs(session.contour_session, contour)
        self.assertIs(session.mask_editing_session, editing)
        self.assertIs(session.working_area, area)
        self.assertEqual(session.coarse_mask_revision, 7)
        self.assertTrue(
            hole_detection_session_is_current(
                session,
                density_session=density,
                contour_session=contour,
                mask_editing_session=editing,
                coarse_mask_revision=7,
            )
        )
        self.assertFalse(
            hole_detection_session_is_current(
                session,
                density_session=density,
                contour_session=contour,
                mask_editing_session=editing,
                coarse_mask_revision=8,
            )
        )

    def test_find_holes_calls_only_existing_facade_on_in_memory_data(self):
        density, contour, editing = _coarse_session(_coarse_mask())
        real_find = hole_workflow.find_hole_candidates

        with (
            patch.object(
                hole_workflow,
                "find_hole_candidates",
                wraps=real_find,
            ) as find_candidates,
            patch(
                "app.services.coarse_processing.prepare_density"
            ) as prepare_density,
            patch("app.core.xyz_reader.compute_stats") as compute_stats,
            patch("app.core.xyz_reader.iter_xyz_points") as iter_points,
            patch("app.core.density_grid.build_density_grid") as build_density,
        ):
            session = _find(density, contour, editing)

        self.assertEqual(len(session.result.candidates), 1)
        find_candidates.assert_called_once()
        prepare_density.assert_not_called()
        compute_stats.assert_not_called()
        iter_points.assert_not_called()
        build_density.assert_not_called()

    def test_quality_parameters_are_forwarded_to_production_facade(self):
        density, contour, editing = _coarse_session(_coarse_mask())

        with patch.object(
            hole_workflow,
            "find_hole_candidates",
            wraps=hole_workflow.find_hole_candidates,
        ) as find_candidates:
            session = find_holes_for_current_mask(
                contour,
                editing,
                density_session=density,
                coarse_mask_revision=1,
                parameters=HoleDetectionParameters(
                    min_diameter_mm=0.0,
                    min_circularity=0.35,
                    max_error_ratio=0.8,
                ),
            )

        self.assertEqual(session.parameters.min_circularity, 0.35)
        self.assertEqual(session.parameters.max_error_ratio, 0.8)
        self.assertEqual(find_candidates.call_args.kwargs["min_circularity"], 0.35)
        self.assertEqual(find_candidates.call_args.kwargs["max_error_ratio"], 0.8)

    def test_invalidation_marks_result_outdated_without_automatic_detection(self):
        density, contour, editing = _coarse_session(_coarse_mask())
        session = _find(density, contour, editing)
        state = {
            "hole_detection_session": session,
            "holes_outdated": False,
        }

        self.assertTrue(invalidate_hole_detection_state(state))
        self.assertIsNone(state["hole_detection_session"])
        self.assertTrue(state["holes_outdated"])

    def test_density_and_working_area_changes_invalidate_holes(self):
        density, contour, editing = _coarse_session(_coarse_mask())
        session = _find(density, contour, editing)
        state = {
            "density_result": density,
            "hole_detection_session": session,
            "holes_outdated": False,
            "coarse_mask_revision": 1,
        }
        reset_source_dependent_state(state)
        self.assertIsNone(state["hole_detection_session"])
        self.assertFalse(state["holes_outdated"])

        state["hole_detection_session"] = session
        state["holes_outdated"] = False
        apply_working_area_state(
            state,
            WorkingArea.from_rectangle_bounds((10.5, 10.5, 50.5, 50.5)),
            density,
        )
        self.assertIsNone(state["hole_detection_session"])
        self.assertTrue(state["holes_outdated"])

    def test_nonzero_fill_calls_detector_with_prefill_mask(self):
        mask = _coarse_mask()
        mask[10, 10] = 0
        density, contour, editing = _coarse_session(
            mask,
            fill_holes_area=2,
        )

        self.assertEqual(contour.masks.mask_for_holes[10, 10], 0)
        self.assertEqual(contour.masks.contour_mask[10, 10], 1)
        with patch.object(
            hole_workflow,
            "find_hole_candidates",
            wraps=hole_workflow.find_hole_candidates,
        ) as find_candidates:
            session = _find(density, contour, editing)

        find_candidates.assert_called_once()
        detector_masks = find_candidates.call_args.args[0]
        np.testing.assert_array_equal(
            detector_masks.mask_for_holes,
            contour.masks.mask_for_holes,
        )
        self.assertEqual(len(session.result.candidates), 1)

    def test_fill_area_alone_does_not_change_effective_hole_mask(self):
        mask = _coarse_mask()
        mask[10, 10] = 0
        density_0, contour_0, editing_0 = _coarse_session(
            mask,
            fill_holes_area=0,
        )
        density_2, contour_2, editing_2 = _coarse_session(
            mask,
            fill_holes_area=2,
        )

        np.testing.assert_array_equal(
            current_effective_hole_mask(contour_0, editing_0),
            current_effective_hole_mask(contour_2, editing_2),
        )
        self.assertEqual(contour_0.masks.contour_mask[10, 10], 0)
        self.assertEqual(contour_2.masks.contour_mask[10, 10], 1)
        self.assertEqual(len(_find(density_0, contour_0, editing_0).result.candidates), 1)
        self.assertEqual(len(_find(density_2, contour_2, editing_2).result.candidates), 1)

    def test_postfill_noop_edit_is_replayed_for_holes_and_can_be_undone(self):
        mask = _coarse_mask()
        mask[10, 10] = 0
        density, contour, editing = _coarse_session(
            mask,
            fill_holes_area=2,
        )
        self.assertEqual(editing.edited_mask[10, 10], 1)

        _stroke(
            editing,
            mode="add",
            start=(10.5, 10.5),
            diameter=1.0,
        )
        self.assertEqual(editing.edited_mask[10, 10], 1)
        self.assertEqual(current_effective_hole_mask(contour, editing)[10, 10], 1)

        self.assertTrue(editing.undo_last_stroke())
        self.assertEqual(current_effective_hole_mask(contour, editing)[10, 10], 0)
        _stroke(
            editing,
            mode="add",
            start=(10.5, 10.5),
            diameter=1.0,
            stroke_id=2,
        )
        self.assertTrue(editing.clear_edits())
        np.testing.assert_array_equal(
            current_effective_hole_mask(contour, editing),
            contour.masks.mask_for_holes,
        )

    def test_rectangle_and_polygon_clip_edge_edits_to_working_area(self):
        mask = _coarse_mask(with_hole=False)
        areas = (
            WorkingArea.from_rectangle_bounds((10.5, 10.5, 45.5, 45.5)),
            WorkingArea.from_polygon(
                [(10.5, 10.5), (45.5, 10.5), (42.5, 45.5), (13.5, 45.5)]
            ),
        )
        for area in areas:
            with self.subTest(kind=area.kind):
                _density, contour, editing = _coarse_session(mask, area)
                _stroke(
                    editing,
                    mode="remove",
                    start=(11.5, 25.5),
                    diameter=8.0,
                )
                effective = current_effective_hole_mask(contour, editing)
                np.testing.assert_array_equal(effective, editing.edited_mask)
                self.assertFalse(np.any(effective[~editing.working_area_mask]))
                changed = effective != contour.masks.mask_for_holes
                self.assertTrue(np.any(changed))
                self.assertFalse(np.any(changed & ~editing.working_area_mask))

    def test_synthetic_prefill_postfill_add_and_undo_smoke(self):
        mask = _coarse_mask()
        mask[10, 10] = 0
        density, contour, editing = _coarse_session(
            mask,
            fill_holes_area=2,
        )

        self.assertEqual(contour.masks.contour_mask[10, 10], 1)
        self.assertEqual(contour.masks.mask_for_holes[10, 10], 0)
        self.assertEqual(contour.masks.contour_mask[30, 30], 0)
        self.assertEqual(contour.masks.mask_for_holes[30, 30], 0)
        self.assertEqual(len(_find(density, contour, editing).result.candidates), 1)

        _stroke(editing, mode="add", start=(30.5, 30.5), stroke_id=10)
        self.assertEqual(_find(density, contour, editing).result.candidates, [])
        self.assertTrue(editing.undo_last_stroke())
        self.assertEqual(len(_find(density, contour, editing).result.candidates), 1)

    def test_many_tiny_voids_are_excluded_from_gui_candidate_result(self):
        mask, tiny_count, expected_in_range = _many_tiny_holes_mask()
        density, contour, editing = _coarse_session(mask)
        component_count = cv2.connectedComponents(1 - mask, connectivity=8)[0] - 1

        session = find_holes_for_current_mask(
            contour,
            editing,
            density_session=density,
            coarse_mask_revision=1,
            parameters=HoleDetectionParameters(
                min_diameter_mm=4.0,
                max_diameter_mm=10.0,
            ),
        )

        self.assertGreaterEqual(tiny_count, 1000)
        self.assertEqual(component_count, tiny_count + expected_in_range)
        self.assertEqual(len(session.result.candidates), expected_in_range)
        self.assertEqual(
            len(session.result.candidates),
            session.result.accepted_count + session.rejected_count,
        )
        self.assertTrue(
            all(
                4.0 <= candidate.diameter <= 10.0
                for candidate in session.result.candidates
            )
        )
        self.assertTrue(
            all(
                candidate.reject_reason not in {"too_small", "too_large"}
                for candidate in session.result.candidates
            )
        )
        self.assertEqual(
            sum(int(group["count"]) for group in session.result.groups),
            session.result.accepted_count,
        )

    def test_in_memory_overlay_draws_only_filtered_candidates(self):
        from app.ui import viewer_app

        mask, _tiny_count, expected_in_range = _many_tiny_holes_mask()
        density, contour, editing = _coarse_session(mask)
        session = find_holes_for_current_mask(
            contour,
            editing,
            density_session=density,
            coarse_mask_revision=1,
            parameters=HoleDetectionParameters(
                min_diameter_mm=4.0,
                max_diameter_mm=10.0,
            ),
        )

        with (
            patch.dict(
                viewer_app.state,
                {"hole_overlay_source": "in_memory"},
                clear=False,
            ),
            patch.object(
                viewer_app,
                "_active_hole_detection_session",
                return_value=session,
            ),
            patch.object(
                viewer_app.dpg,
                "does_item_exist",
                side_effect=lambda tag: tag == viewer_app.IMAGE_TAG,
            ),
            patch.object(viewer_app.dpg, "add_draw_layer"),
            patch.object(viewer_app.dpg, "draw_text") as draw_text,
            patch.object(viewer_app, "_display_layer_enabled", return_value=True),
            patch.object(viewer_app, "_world_to_drawlist", return_value=(1.0, 1.0)),
            patch.object(viewer_app, "_world_radius_to_draw_radii", return_value=(2.0, 2.0)),
            patch.object(viewer_app, "_draw_world_radius_ellipse") as draw_circle,
        ):
            viewer_app._redraw_holes_overlay()

        self.assertEqual(draw_circle.call_count, expected_in_range)
        self.assertEqual(draw_text.call_count, expected_in_range)

    def test_gui_statistics_use_only_filtered_candidates(self):
        from app.ui import viewer_app

        mask, _tiny_count, expected_in_range = _many_tiny_holes_mask()
        density, contour, editing = _coarse_session(mask)
        session = find_holes_for_current_mask(
            contour,
            editing,
            density_session=density,
            coarse_mask_revision=1,
            parameters=HoleDetectionParameters(
                min_diameter_mm=4.0,
                max_diameter_mm=10.0,
            ),
        )

        with (
            patch.object(
                viewer_app,
                "_active_hole_detection_session",
                return_value=session,
            ),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(viewer_app.dpg, "set_value") as set_value,
        ):
            viewer_app._update_hole_detection_info()

        displayed = set_value.call_args.args[1]
        self.assertIn(f"Hole candidates: {expected_in_range}", displayed)
        self.assertIn(
            f"Accepted: {session.result.accepted_count} | "
            f"Rejected: {session.rejected_count}",
            displayed,
        )

    def test_gui_find_holes_success_and_invalid_parameters_preserve_result(self):
        from app.ui import viewer_app

        density, contour, editing = _coarse_session(_coarse_mask())
        density_result = SimpleNamespace(grid=editing.grid)
        contour = replace(contour, density_session=density_result)
        replacement = {
            "density_result": density_result,
            "active_working_area": None,
            "working_area_density_session": None,
            "contour_processing_result": contour,
            "mask_editing_session": editing,
            "coarse_mask_revision": 3,
            "hole_detection_session": None,
            "holes_outdated": False,
            "hole_overlay_source": None,
        }

        def valid_value(tag):
            if tag == viewer_app.HOLE_MIN_DIAMETER_TAG:
                return 0.0
            if tag == viewer_app.HOLE_MAX_DIAMETER_TAG:
                return 0.0
            if tag == viewer_app.HOLE_MIN_CIRCULARITY_TAG:
                return 0.35
            if tag == viewer_app.HOLE_MAX_ERROR_RATIO_TAG:
                return 0.8
            raise AssertionError(tag)

        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "get_value", side_effect=valid_value),
            patch.object(
                viewer_app,
                "find_holes_for_current_mask",
                wraps=viewer_app.find_holes_for_current_mask,
            ) as find_holes_once,
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_set_status"),
        ):
            viewer_app._find_holes_callback()
            previous = viewer_app.state["hole_detection_session"]
            self.assertIsInstance(previous, hole_workflow.HoleDetectionSession)
            self.assertEqual(len(previous.result.candidates), 1)
            self.assertEqual(previous.result.accepted_count, 1)
            self.assertEqual(previous.rejected_count, 0)
            self.assertEqual(previous.parameters.min_circularity, 0.35)
            self.assertEqual(previous.parameters.max_error_ratio, 0.8)
            self.assertEqual(viewer_app.state["hole_overlay_source"], "in_memory")
            find_holes_once.assert_called_once()

            def invalid_value(tag):
                if tag == viewer_app.HOLE_MIN_DIAMETER_TAG:
                    return 10.0
                if tag == viewer_app.HOLE_MAX_DIAMETER_TAG:
                    return 5.0
                if tag == viewer_app.HOLE_MIN_CIRCULARITY_TAG:
                    return 0.55
                if tag == viewer_app.HOLE_MAX_ERROR_RATIO_TAG:
                    return 0.18
                raise AssertionError(tag)

            with (
                patch.object(
                    viewer_app.dpg,
                    "get_value",
                    side_effect=invalid_value,
                ),
                patch.object(
                    viewer_app,
                    "find_holes_for_current_mask",
                ) as find_holes,
            ):
                viewer_app._find_holes_callback()

            find_holes.assert_not_called()
            self.assertIs(viewer_app.state["hole_detection_session"], previous)

    def test_gui_successful_zero_candidates_is_stored_as_current_result(self):
        from app.ui import viewer_app

        _density, contour, editing = _coarse_session(
            _coarse_mask(with_hole=False)
        )
        density_result = SimpleNamespace(grid=editing.grid)
        contour = replace(contour, density_session=density_result)
        replacement = {
            "density_result": density_result,
            "active_working_area": None,
            "working_area_density_session": None,
            "contour_processing_result": contour,
            "mask_editing_session": editing,
            "coarse_mask_revision": 8,
            "hole_detection_session": None,
            "holes_outdated": True,
            "hole_overlay_source": None,
        }

        def selected_value(tag):
            return {
                viewer_app.HOLE_MIN_DIAMETER_TAG: 0.0,
                viewer_app.HOLE_MAX_DIAMETER_TAG: 0.0,
                viewer_app.HOLE_MIN_CIRCULARITY_TAG: 0.35,
                viewer_app.HOLE_MAX_ERROR_RATIO_TAG: 0.8,
            }[tag]

        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "get_value", side_effect=selected_value),
            patch.object(
                viewer_app,
                "find_holes_for_current_mask",
                wraps=viewer_app.find_holes_for_current_mask,
            ) as find_holes,
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_set_status") as set_status,
        ):
            viewer_app._find_holes_callback()
            find_holes.assert_called_once()
            session = viewer_app.state["hole_detection_session"]
            self.assertIsInstance(session, hole_workflow.HoleDetectionSession)
            self.assertEqual(session.result.candidates, [])
            self.assertFalse(viewer_app.state["holes_outdated"])
            self.assertEqual(viewer_app.state["hole_overlay_source"], "in_memory")
            self.assertIn("total=0", set_status.call_args.args[0])

    def test_effective_mask_changes_invalidate_without_running_detector(self):
        from app.ui import viewer_app

        density, contour, editing = _coarse_session(_coarse_mask())
        session = _find(density, contour, editing, revision=4)
        replacement = {
            "coarse_mask_revision": 4,
            "hole_detection_session": session,
            "holes_outdated": False,
            "hole_overlay_source": "in_memory",
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(
                viewer_app,
                "find_holes_for_current_mask",
            ) as find_holes,
        ):
            viewer_app._mark_effective_mask_changed()

            self.assertEqual(viewer_app.state["coarse_mask_revision"], 5)
            self.assertIsNone(viewer_app.state["hole_detection_session"])
            self.assertTrue(viewer_app.state["holes_outdated"])
            find_holes.assert_not_called()

    def test_quality_setting_commit_only_invalidates_hole_result(self):
        from app.ui import viewer_app

        density, contour, editing = _coarse_session(_coarse_mask())
        session = _find(density, contour, editing, revision=4)
        replacement = {
            "density_result": density,
            "contour_processing_result": contour,
            "mask_editing_session": editing,
            "coarse_mask_revision": 4,
            "hole_detection_session": session,
            "holes_outdated": False,
        }

        def valid_value(tag):
            return {
                viewer_app.HOLE_MIN_DIAMETER_TAG: 4.0,
                viewer_app.HOLE_MAX_DIAMETER_TAG: 10.0,
                viewer_app.HOLE_MIN_CIRCULARITY_TAG: 0.35,
                viewer_app.HOLE_MAX_ERROR_RATIO_TAG: 0.8,
            }[tag]

        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "get_value", side_effect=valid_value),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
            patch.object(viewer_app, "rebase_preliminary_contour_with_edits") as rebase,
            patch.object(viewer_app, "find_holes_for_current_mask") as find_holes,
            patch("app.services.coarse_processing.prepare_density") as prepare_density,
            patch("app.core.xyz_reader.compute_stats") as compute_stats,
            patch("app.core.density_grid.build_density_grid") as build_density,
        ):
            viewer_app._hole_detection_quality_settings_callback()

            self.assertIsNone(viewer_app.state["hole_detection_session"])
            self.assertTrue(viewer_app.state["holes_outdated"])
            self.assertIs(viewer_app.state["contour_processing_result"], contour)
            self.assertIs(viewer_app.state["mask_editing_session"], editing)
            self.assertEqual(viewer_app.state["coarse_mask_revision"], 4)

        rebase.assert_not_called()
        find_holes.assert_not_called()
        prepare_density.assert_not_called()
        compute_stats.assert_not_called()
        build_density.assert_not_called()

        replacement["hole_detection_session"] = session
        replacement["holes_outdated"] = False

        def invalid_value(tag):
            if tag == viewer_app.HOLE_MIN_CIRCULARITY_TAG:
                return float("nan")
            return valid_value(tag)

        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "get_value", side_effect=invalid_value),
            patch.object(viewer_app, "_set_status"),
        ):
            viewer_app._hole_detection_quality_settings_callback()
            self.assertIs(viewer_app.state["hole_detection_session"], session)
            self.assertFalse(viewer_app.state["holes_outdated"])


if __name__ == "__main__":
    unittest.main()
