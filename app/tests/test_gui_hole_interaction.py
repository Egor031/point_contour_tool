import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.coordinate_transform import CoordinateTransform  # noqa: E402
from app.core.density_grid import DensityGrid  # noqa: E402
from app.core.hole_detector import HoleCandidate  # noqa: E402
from app.services.coarse_processing import HoleDetectionResult  # noqa: E402
from app.ui import viewer_app  # noqa: E402
from app.ui.contour_workflow import MaskEditingSession  # noqa: E402
from app.ui.hole_workflow import (  # noqa: E402
    HoleDetectionParameters,
    HoleDetectionSession,
    HoleHitRegion,
    apply_manual_hole_status,
    hit_test_hole_regions,
)


def _candidate(
    candidate_id: int,
    center_x: float,
    center_y: float,
    radius: float,
    *,
    accepted: bool,
    reject_reason: str = "",
    group_id: str | None = None,
) -> HoleCandidate:
    return HoleCandidate(
        id=candidate_id,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        diameter=radius * 2.0,
        center_px=center_x,
        center_py=center_y,
        radius_px=radius,
        area_cells=17 + candidate_id,
        area_mm2=18.5 + candidate_id,
        bbox_width_mm=radius * 2.0,
        bbox_height_mm=radius * 2.0,
        aspect_ratio=1.0,
        circularity=0.9,
        mean_error_mm=0.1,
        max_error_mm=0.2,
        error_ratio=0.04,
        accepted=accepted,
        reject_reason=reject_reason,
        group_id=group_id,
    )


def _session() -> HoleDetectionSession:
    candidates = [
        _candidate(1, 20.5, 30.5, 4.0, accepted=True, group_id="G1"),
        _candidate(
            2,
            60.5,
            15.5,
            2.0,
            accepted=False,
            reject_reason="circle_fit_error",
        ),
        _candidate(3, 64.5, 16.5, 1.0, accepted=True, group_id="G1"),
    ]
    return HoleDetectionSession(
        result=HoleDetectionResult(
            candidates=candidates,
            groups=[
                {
                    "id": "G1",
                    "name": "group",
                    "diameter": 5.0,
                    "radius": 2.5,
                    "count": 2,
                    "enabled": True,
                }
            ],
        ),
        density_session=object(),
        contour_session=object(),
        mask_editing_session=object(),
        working_area=None,
        coarse_mask_revision=1,
        parameters=HoleDetectionParameters(min_diameter_mm=0.0),
        automatic_acceptance={1: True, 2: False, 3: True},
    )


def _gesture(
    button: str,
    *,
    start: tuple[float, float],
    current: tuple[float, float],
    dragged: bool = False,
    start_world: tuple[float, float] | None = None,
) -> dict:
    return {
        "start_screen": start,
        "last_screen": current,
        "start_world": start_world,
        "dragged": dragged,
        "rectangle_drag_started": False,
        "brush_last_screen": None,
        "pan_last_screen": start,
        "button": button,
    }


def _geometry_and_metrics(candidate: HoleCandidate) -> tuple:
    return (
        candidate.id,
        candidate.center_x,
        candidate.center_y,
        candidate.radius,
        candidate.diameter,
        candidate.center_px,
        candidate.center_py,
        candidate.radius_px,
        candidate.area_cells,
        candidate.area_mm2,
        candidate.bbox_width_mm,
        candidate.bbox_height_mm,
        candidate.aspect_ratio,
        candidate.circularity,
        candidate.mean_error_mm,
        candidate.max_error_mm,
        candidate.error_ratio,
        candidate.reject_reason,
        candidate.group_id,
    )


class TestHoleInteractionDomain(unittest.TestCase):
    def test_review_holes_mode_and_state_are_removed(self):
        self.assertFalse(hasattr(viewer_app, "REVIEW_HOLES_MODE_TAG"))
        self.assertNotIn("mode_before_hole_review", viewer_app.state)
        self.assertNotEqual(viewer_app.state["mode"], "review_holes")

    def test_drag_threshold_is_screen_space_and_strict(self):
        threshold = viewer_app.MOUSE_DRAG_THRESHOLD_PX
        self.assertFalse(
            viewer_app._gesture_is_drag((10.0, 10.0), (10.0 + threshold, 10.0))
        )
        self.assertTrue(
            viewer_app._gesture_is_drag(
                (10.0, 10.0),
                (10.0 + threshold + 0.01, 10.0),
            )
        )

    def test_hit_test_has_tolerance_and_deterministic_overlap(self):
        regions = [HoleHitRegion(9, 10.0, 10.0, 1.0, 1.0)]
        self.assertEqual(
            hit_test_hole_regions(regions, (15.0, 10.0), tolerance_px=5.0),
            9,
        )
        overlapping = [
            HoleHitRegion(8, 10.0, 10.0, 20.0, 20.0),
            HoleHitRegion(3, 16.0, 10.0, 20.0, 20.0),
        ]
        self.assertEqual(hit_test_hole_regions(overlapping, (13.0, 10.0)), 3)
        self.assertIsNone(hit_test_hole_regions(regions, (30.0, 30.0)))

    def test_manual_status_is_directional_and_preserves_candidate_data(self):
        session = _session()
        accepted = session.result.candidates[0]
        rejected = session.result.candidates[1]
        accepted_snapshot = _geometry_and_metrics(accepted)
        rejected_snapshot = _geometry_and_metrics(rejected)

        self.assertTrue(apply_manual_hole_status(session, 2, accepted=True))
        self.assertFalse(apply_manual_hole_status(session, 2, accepted=True))
        self.assertTrue(apply_manual_hole_status(session, 1, accepted=False))
        self.assertFalse(apply_manual_hole_status(session, 1, accepted=False))

        self.assertEqual(_geometry_and_metrics(accepted), accepted_snapshot)
        self.assertEqual(_geometry_and_metrics(rejected), rejected_snapshot)
        self.assertEqual(rejected.reject_reason, "circle_fit_error")
        self.assertEqual(session.automatic_acceptance, {1: True, 2: False, 3: True})
        self.assertEqual(session.manual_overrides, {1: False, 2: True})
        self.assertEqual(session.result.groups[0]["count"], 1)


class TestHoleInteractionViewer(unittest.TestCase):
    def tearDown(self):
        viewer_app.state["hovered_hole_id"] = None
        viewer_app.state["mouse_gestures"] = {}
        viewer_app.state["last_event_mouse_position"] = None
        viewer_app.state["pan_redraw_pending"] = False
        viewer_app.state["pick_manual_hole_center"] = False
        viewer_app.state["mode"] = "rectangle"

    @staticmethod
    def _preview_click_data(button: str):
        button_id = {
            "left": viewer_app.dpg.mvMouseButton_Left,
            "right": viewer_app.dpg.mvMouseButton_Right,
            "middle": viewer_app.dpg.mvMouseButton_Middle,
        }[button]
        return button_id, viewer_app.IMAGE_TAG

    def test_sidebar_click_never_starts_preview_gesture_or_rectangle(self):
        replacement = {
            "mode": "rectangle",
            "mouse_gestures": {},
            "last_event_mouse_position": (20.0, 30.0),
            "roi_first_world": None,
            "roi_current_world": None,
            "rectangle_roi": None,
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(viewer_app, "_route_short_left_click") as route,
        ):
            for index in range(10):
                viewer_app._mouse_down_callback(
                    _app_data=(viewer_app.dpg.mvMouseButton_Left, "show_contour"),
                    _user_data="left",
                )
                viewer_app.state["last_event_mouse_position"] = (
                    500.0 + index,
                    400.0 + index,
                )
                viewer_app._mouse_release_callback(_user_data="left")
            self.assertEqual(viewer_app.state["mouse_gestures"], {})
            self.assertIsNone(viewer_app.state["roi_first_world"])
            self.assertIsNone(viewer_app.state["roi_current_world"])
            self.assertIsNone(viewer_app.state["rectangle_roi"])
        route.assert_not_called()

    def test_sidebar_click_does_not_reach_polygon_brush_or_manual_hole(self):
        for mode, manual_pick in (
            ("polygon", False),
            ("mask_brush", False),
            ("rectangle", True),
        ):
            with self.subTest(mode=mode, manual_pick=manual_pick):
                replacement = {
                    "mode": mode,
                    "mouse_gestures": {},
                    "last_event_mouse_position": (600.0, 500.0),
                    "polygon_points": [],
                    "pick_manual_hole_center": manual_pick,
                    "manual_hole_center_world": None,
                }
                with (
                    patch.dict(viewer_app.state, replacement, clear=False),
                    patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
                    patch.object(viewer_app, "_apply_mask_brush_at_screen") as brush,
                    patch.object(viewer_app, "_finish_active_mask_stroke") as finish_brush,
                    patch.object(viewer_app, "_finish_manual_hole_center_pick") as pick,
                ):
                    viewer_app._mouse_down_callback(
                        _app_data=(
                            viewer_app.dpg.mvMouseButton_Left,
                            "sidebar_control",
                        ),
                        _user_data="left",
                    )
                    viewer_app._mouse_release_callback(_user_data="left")
                    self.assertEqual(viewer_app.state["polygon_points"], [])
                    self.assertIsNone(
                        viewer_app.state["manual_hole_center_world"]
                    )
                brush.assert_not_called()
                finish_brush.assert_not_called()
                pick.assert_not_called()

    def test_sidebar_lmb_and_rmb_do_not_change_hole_status(self):
        replacement = {
            "mouse_gestures": {},
            "last_event_mouse_position": (700.0, 600.0),
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(viewer_app, "_hole_at_screen", return_value=2),
            patch.object(viewer_app, "_set_manual_hole_status_by_id") as status,
        ):
            for button, button_id in (
                ("left", viewer_app.dpg.mvMouseButton_Left),
                ("right", viewer_app.dpg.mvMouseButton_Right),
            ):
                viewer_app._mouse_down_callback(
                    _app_data=(button_id, "sidebar_control"),
                    _user_data=button,
                )
                viewer_app._mouse_release_callback(_user_data=button)

        status.assert_not_called()

    def test_delayed_preview_click_uses_event_position_not_current_cursor(self):
        event_position = (120.0, 130.0)
        current_cursor = (900.0, 800.0)
        replacement = {
            "mode": "rectangle",
            "mouse_gestures": {},
            "last_event_mouse_position": event_position,
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(
                viewer_app.dpg,
                "get_mouse_pos",
                return_value=current_cursor,
            ) as get_mouse_pos,
            patch.object(
                viewer_app,
                "screen_to_world",
                return_value=(0.0, 0.0, 0.0, 0.0, 12.0, 13.0),
            ),
            patch.object(viewer_app, "_route_short_left_click") as route,
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
        ):
            viewer_app._mouse_down_callback(
                _app_data=self._preview_click_data("left"),
                _user_data="left",
            )
            self.assertEqual(
                viewer_app.state["mouse_gestures"]["left"]["start_screen"],
                event_position,
            )
            viewer_app._mouse_release_callback(_user_data="left")

        route.assert_called_once_with(event_position)
        get_mouse_pos.assert_not_called()

    def test_move_app_data_drives_drag_threshold_and_rectangle_position(self):
        start = (40.0, 50.0)
        event_move = (40.0 + viewer_app.MOUSE_DRAG_THRESHOLD_PX + 1.0, 50.0)
        replacement = {
            "mode": "rectangle",
            "mouse_gestures": {},
            "last_event_mouse_position": start,
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(
                viewer_app,
                "screen_to_world",
                return_value=(0.0, 0.0, 0.0, 0.0, 4.0, 5.0),
            ),
            patch.object(viewer_app, "_update_pan_from_mouse"),
            patch.object(viewer_app, "_update_rectangle_drag_from_mouse") as rectangle,
            patch.object(viewer_app, "_update_mask_brush_from_mouse"),
            patch.object(viewer_app, "_update_brush_cursor_from_mouse"),
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
            patch.object(viewer_app, "_update_debug_coords"),
            patch.object(viewer_app.dpg, "set_value"),
        ):
            viewer_app._mouse_down_callback(
                _app_data=self._preview_click_data("left"),
                _user_data="left",
            )
            viewer_app._mouse_move_callback(_app_data=event_move)
            gesture = viewer_app.state["mouse_gestures"]["left"]
            self.assertEqual(gesture["last_screen"], event_move)
            self.assertTrue(gesture["dragged"])
        rectangle.assert_called_once_with(event_move)

    def test_event_time_right_move_pans_and_global_release_works_outside(self):
        start = (100.0, 100.0)
        outside = (2505.0, 1705.0)
        replacement = {
            "mode": "rectangle",
            "mouse_gestures": {},
            "last_event_mouse_position": start,
            "pan_x": 0.0,
            "pan_y": 0.0,
            "last_pan_mouse": None,
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(
                viewer_app,
                "screen_to_world",
                return_value=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
            ),
            patch.object(
                viewer_app.dpg,
                "is_mouse_button_down",
                side_effect=lambda button: button
                == viewer_app.dpg.mvMouseButton_Right,
            ),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_update_rectangle_drag_from_mouse"),
            patch.object(viewer_app, "_update_mask_brush_from_mouse"),
            patch.object(viewer_app, "_update_brush_cursor_from_mouse"),
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
            patch.object(viewer_app, "_update_debug_coords"),
            patch.object(viewer_app.dpg, "set_value"),
        ):
            viewer_app._mouse_down_callback(
                _app_data=self._preview_click_data("right"),
                _user_data="right",
            )
            viewer_app._mouse_move_callback(_app_data=outside)
            self.assertEqual(
                (viewer_app.state["pan_x"], viewer_app.state["pan_y"]),
                (outside[0] - start[0], outside[1] - start[1]),
            )
            viewer_app._mouse_release_callback(_user_data="right")
            self.assertNotIn("right", viewer_app.state["mouse_gestures"])

    def test_visual_hole_hover_uses_current_cursor_not_queued_move_position(self):
        event_position = (110.0, 120.0)
        current_cursor = (510.0, 520.0)
        replacement = {
            "mouse_gestures": {},
            "last_event_mouse_position": None,
            "pick_manual_hole_center": False,
            "hovered_hole_id": None,
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "get_mouse_pos", return_value=current_cursor),
            patch.object(
                viewer_app.dpg,
                "does_item_exist",
                side_effect=lambda tag: tag == viewer_app.IMAGE_TAG,
            ),
            patch.object(viewer_app.dpg, "is_item_hovered", return_value=True),
            patch.object(viewer_app, "_hole_at_screen", return_value=2) as hit_test,
            patch.object(viewer_app, "_redraw_holes_overlay"),
            patch.object(viewer_app, "_update_pan_from_mouse"),
            patch.object(viewer_app, "_update_rectangle_drag_from_mouse"),
            patch.object(viewer_app, "_update_mask_brush_from_mouse"),
            patch.object(viewer_app, "_update_brush_cursor_from_mouse"),
            patch.object(viewer_app, "_update_debug_coords"),
            patch.object(viewer_app, "screen_to_world", return_value=None),
            patch.object(viewer_app.dpg, "set_value"),
        ):
            viewer_app._mouse_move_callback(_app_data=event_position)
            self.assertEqual(
                viewer_app.state["last_event_mouse_position"],
                event_position,
            )
            self.assertEqual(viewer_app.state["hovered_hole_id"], 2)
        hit_test.assert_called_once_with(*current_cursor)

    def _screen_hit(
        self,
        session: HoleDetectionSession,
        *,
        zoom: float,
        pan: tuple[float, float],
        candidate_id: int,
    ) -> int | None:
        transform = CoordinateTransform(
            grid_min_x=0.0,
            grid_min_y=0.0,
            cell_size=1.0,
            grid_width=100,
            grid_height=80,
            preview_width=50,
            preview_height=40,
        )
        replacement = {
            "image_width": 50,
            "image_height": 40,
            "zoom": zoom,
            "pan_x": pan[0],
            "pan_y": pan[1],
            "pick_manual_hole_center": False,
        }
        candidate = next(
            item for item in session.result.candidates if item.id == candidate_id
        )
        canvas_origin = (100.0, 200.0)
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app, "_get_coordinate_transform", return_value=transform),
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(viewer_app, "_display_layer_enabled", return_value=True),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(viewer_app.dpg, "is_item_hovered", return_value=True),
            patch.object(
                viewer_app.dpg,
                "get_item_rect_min",
                return_value=canvas_origin,
            ),
        ):
            draw_x, draw_y = viewer_app._world_to_drawlist(
                candidate.center_x,
                candidate.center_y,
            )
            return viewer_app._hole_at_screen(
                canvas_origin[0] + draw_x,
                canvas_origin[1] + draw_y,
            )

    def test_hit_test_uses_preview_transform_zoom_and_pan(self):
        session = _session()
        for zoom, pan, candidate_id in (
            (1.0, (0.0, 0.0), 1),
            (3.0, (0.0, 0.0), 2),
            (2.0, (37.0, -19.0), 3),
        ):
            with self.subTest(zoom=zoom, pan=pan, candidate_id=candidate_id):
                self.assertEqual(
                    self._screen_hit(
                        session,
                        zoom=zoom,
                        pan=pan,
                        candidate_id=candidate_id,
                    ),
                    candidate_id,
                )

    def test_hover_is_active_in_rectangle_polygon_and_brush_modes(self):
        for mode in ("rectangle", "polygon", "mask_brush"):
            with self.subTest(mode=mode):
                with (
                    patch.dict(
                        viewer_app.state,
                        {
                            "mode": mode,
                            "hovered_hole_id": None,
                            "mouse_gestures": {},
                            "pick_manual_hole_center": False,
                        },
                        clear=False,
                    ),
                    patch.object(
                        viewer_app.dpg,
                        "get_mouse_pos",
                        return_value=(5.0, 5.0),
                    ),
                    patch.object(
                        viewer_app.dpg,
                        "does_item_exist",
                        side_effect=lambda tag: tag == viewer_app.IMAGE_TAG,
                    ),
                    patch.object(
                        viewer_app.dpg,
                        "is_item_hovered",
                        return_value=True,
                    ),
                    patch.object(viewer_app, "_hole_at_screen", return_value=2),
                    patch.object(viewer_app, "_redraw_holes_overlay") as redraw,
                ):
                    self.assertEqual(
                        viewer_app._update_hole_hover_from_mouse(),
                        2,
                    )
                    self.assertEqual(viewer_app.state["hovered_hole_id"], 2)
                    redraw.assert_called_once()

    def test_short_left_accepts_rejected_and_consumes_rectangle_click(self):
        session = _session()
        replacement = {
            "mode": "rectangle",
            "roi_first_world": None,
            "rectangle_roi": None,
            "mouse_gestures": {
                "left": _gesture("left", start=(10.0, 10.0), current=(11.0, 11.0))
            },
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "get_mouse_pos", return_value=(11.0, 11.0)),
            patch.object(viewer_app, "_hole_at_screen", return_value=2),
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
            patch.object(viewer_app, "begin_rectangle_draft") as begin_roi,
            patch.object(viewer_app, "finish_rectangle_draft") as finish_roi,
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
        ):
            viewer_app._mouse_release_callback(_user_data="left")

        self.assertTrue(session.result.candidates[1].accepted)
        begin_roi.assert_not_called()
        finish_roi.assert_not_called()

    def test_down_release_router_classifies_short_click_once(self):
        replacement = {
            "mouse_gestures": {},
            "mode": "rectangle",
            "last_event_mouse_position": (10.0, 10.0),
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(
                viewer_app,
                "screen_to_world",
                return_value=(0.0, 0.0, 0.0, 0.0, 4.0, 5.0),
            ),
            patch.object(viewer_app, "_route_short_left_click") as route,
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
        ):
            viewer_app._mouse_down_callback(
                _app_data=(viewer_app.dpg.mvMouseButton_Left, viewer_app.IMAGE_TAG),
                _user_data="left",
            )
            self.assertIn("left", viewer_app.state["mouse_gestures"])
            viewer_app.state["last_event_mouse_position"] = (12.0, 11.0)
            viewer_app._mouse_release_callback(_user_data="left")
            self.assertNotIn("left", viewer_app.state["mouse_gestures"])
        route.assert_called_once_with((12.0, 11.0))

    def test_directional_short_clicks_do_not_toggle_matching_status(self):
        session = _session()
        with (
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(viewer_app, "_update_hole_detection_info") as update_info,
            patch.object(viewer_app, "_redraw_preview") as redraw,
            patch.object(viewer_app, "_set_status"),
        ):
            self.assertFalse(
                viewer_app._set_manual_hole_status_by_id(1, accepted=True)
            )
            self.assertFalse(
                viewer_app._set_manual_hole_status_by_id(2, accepted=False)
            )
        update_info.assert_not_called()
        redraw.assert_not_called()

    def test_short_right_rejects_without_pan(self):
        session = _session()
        replacement = {
            "pan_x": 12.0,
            "pan_y": 9.0,
            "mouse_gestures": {
                "right": _gesture("right", start=(10.0, 10.0), current=(12.0, 11.0))
            },
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "get_mouse_pos", return_value=(12.0, 11.0)),
            patch.object(viewer_app, "_hole_at_screen", return_value=1),
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
        ):
            viewer_app._mouse_release_callback(_user_data="right")
            self.assertEqual((viewer_app.state["pan_x"], viewer_app.state["pan_y"]), (12.0, 9.0))
        self.assertFalse(session.result.candidates[0].accepted)

    def test_short_click_outside_holes_routes_to_rectangle_and_polygon(self):
        with (
            patch.dict(
                viewer_app.state,
                {"mode": "rectangle", "roi_first_world": None},
                clear=False,
            ),
            patch.object(viewer_app, "_hole_at_screen", return_value=None),
            patch.object(viewer_app, "_warn_preview_grid_params_not_loaded", return_value=False),
            patch.object(
                viewer_app,
                "screen_to_world",
                return_value=(0.0, 0.0, 0.0, 0.0, 4.0, 5.0),
            ),
            patch.object(viewer_app, "begin_rectangle_draft") as begin_roi,
            patch.object(viewer_app.dpg, "set_value"),
            patch.object(viewer_app, "_redraw_polygon_overlay"),
        ):
            self.assertTrue(viewer_app._route_short_left_click((10.0, 10.0)))
        begin_roi.assert_called_once_with(viewer_app.state, (4.0, 5.0))

        points: list[tuple[float, float]] = []
        with (
            patch.dict(
                viewer_app.state,
                {
                    "mode": "polygon",
                    "polygon_points": points,
                    "undo_history": [],
                    "polygon_finished": False,
                },
                clear=False,
            ),
            patch.object(viewer_app, "_hole_at_screen", return_value=None),
            patch.object(viewer_app, "_warn_preview_grid_params_not_loaded", return_value=False),
            patch.object(
                viewer_app,
                "screen_to_world",
                return_value=(0.0, 0.0, 0.0, 0.0, 7.0, 8.0),
            ),
            patch.object(viewer_app.dpg, "set_value"),
            patch.object(viewer_app, "_update_polygon_points_text"),
            patch.object(viewer_app, "_redraw_preview"),
        ):
            self.assertTrue(viewer_app._route_short_left_click((10.0, 10.0)))
            self.assertEqual(viewer_app.state["polygon_points"], [(7.0, 8.0)])

    def test_polygon_click_on_hole_does_not_add_vertex(self):
        session = _session()
        points: list[tuple[float, float]] = []
        with (
            patch.dict(
                viewer_app.state,
                {"mode": "polygon", "polygon_points": points},
                clear=False,
            ),
            patch.object(viewer_app, "_hole_at_screen", return_value=2),
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
        ):
            self.assertTrue(viewer_app._route_short_left_click((10.0, 10.0)))
        self.assertEqual(points, [])
        self.assertTrue(session.result.candidates[1].accepted)

    def test_rectangle_drag_builds_roi_even_when_started_over_hole(self):
        gesture = _gesture(
            "left",
            start=(10.0, 10.0),
            current=(30.0, 40.0),
            dragged=True,
            start_world=(1.0, 2.0),
        )
        replacement = {
            "mode": "rectangle",
            "roi_first_world": None,
            "roi_current_world": None,
            "rectangle_roi": None,
            "mouse_gestures": {"left": gesture},
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(
                viewer_app,
                "screen_to_world",
                return_value=(0.0, 0.0, 0.0, 0.0, 5.0, 7.0),
            ),
            patch.object(viewer_app, "_redraw_polygon_overlay"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app.dpg, "set_value"),
            patch.object(viewer_app, "_set_manual_hole_status_by_id") as status,
        ):
            viewer_app._finish_drag_gesture("left", gesture, (30.0, 40.0))
            self.assertEqual(viewer_app.state["rectangle_roi"], (1.0, 2.0, 5.0, 7.0))
        status.assert_not_called()

    def test_pan_drag_started_over_hole_moves_view_without_status_change(self):
        session = _session()
        gesture = _gesture(
            "right",
            start=(10.0, 10.0),
            current=(35.0, 28.0),
            dragged=True,
        )
        replacement = {
            "pan_x": 2.0,
            "pan_y": 3.0,
            "mouse_gestures": {"right": gesture},
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "is_mouse_button_down", return_value=True),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_manual_hole_status_by_id") as status,
        ):
            viewer_app._update_pan_from_mouse()
            self.assertEqual((viewer_app.state["pan_x"], viewer_app.state["pan_y"]), (27.0, 21.0))
        self.assertTrue(session.result.candidates[0].accepted)
        status.assert_not_called()

    def test_right_pan_moves_coalesce_until_frame_flush(self):
        gesture = _gesture(
            "right",
            start=(0.0, 0.0),
            current=(0.0, 0.0),
        )
        replacement = {
            "pan_x": 0.0,
            "pan_y": 0.0,
            "pan_redraw_pending": False,
            "mouse_gestures": {"right": gesture},
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "is_mouse_button_down", return_value=True),
            patch.object(viewer_app.dpg, "set_value"),
            patch.object(viewer_app, "_redraw_preview") as redraw,
            patch.object(viewer_app, "_update_rectangle_drag_from_mouse"),
            patch.object(viewer_app, "_update_mask_brush_from_mouse"),
            patch.object(viewer_app, "_update_brush_cursor_from_mouse"),
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
            patch.object(viewer_app, "_update_debug_coords"),
            patch.object(viewer_app, "screen_to_world", return_value=None),
        ):
            for position in ((10.0, 0.0), (20.0, 0.0), (30.0, 0.0)):
                viewer_app._mouse_move_callback(_app_data=position)

            self.assertEqual(viewer_app.state["pan_x"], 30.0)
            self.assertEqual(gesture["pan_last_screen"], (30.0, 0.0))
            self.assertTrue(viewer_app.state["pan_redraw_pending"])
            redraw.assert_not_called()

            self.assertTrue(viewer_app._flush_pending_pan_redraw())
            redraw.assert_called_once()
            self.assertFalse(viewer_app.state["pan_redraw_pending"])
            self.assertFalse(viewer_app._flush_pending_pan_redraw())
            redraw.assert_called_once()

    def test_large_pan_queue_and_direction_change_use_one_redraw(self):
        gesture = _gesture(
            "right",
            start=(0.0, 0.0),
            current=(0.0, 0.0),
        )
        replacement = {
            "pan_x": 0.0,
            "pan_y": 0.0,
            "pan_redraw_pending": False,
            "mouse_gestures": {"right": gesture},
        }
        move_positions = [
            (float(x), 0.0) for x in range(10, 501, 10)
        ] + [
            (float(x), 0.0) for x in range(490, -1, -10)
        ]
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "is_mouse_button_down", return_value=True),
            patch.object(viewer_app.dpg, "set_value"),
            patch.object(viewer_app, "_redraw_preview") as redraw,
            patch.object(viewer_app, "_update_rectangle_drag_from_mouse"),
            patch.object(viewer_app, "_update_mask_brush_from_mouse"),
            patch.object(viewer_app, "_update_brush_cursor_from_mouse"),
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
            patch.object(viewer_app, "_update_debug_coords"),
            patch.object(viewer_app, "screen_to_world", return_value=None),
        ):
            for position in move_positions[:50]:
                viewer_app._mouse_move_callback(_app_data=position)
            self.assertEqual(viewer_app.state["pan_x"], 500.0)

            for position in move_positions[50:]:
                viewer_app._mouse_move_callback(_app_data=position)
            self.assertEqual(viewer_app.state["pan_x"], 0.0)
            self.assertEqual(gesture["pan_last_screen"], (0.0, 0.0))
            self.assertTrue(viewer_app.state["pan_redraw_pending"])
            redraw.assert_not_called()

            viewer_app._flush_pending_pan_redraw()
            redraw.assert_called_once()
            self.assertFalse(viewer_app.state["pan_redraw_pending"])

    def test_middle_pan_moves_coalesce_until_frame_flush(self):
        gesture = _gesture(
            "middle",
            start=(5.0, 7.0),
            current=(5.0, 7.0),
            dragged=True,
        )
        replacement = {
            "pan_x": 2.0,
            "pan_y": 3.0,
            "pan_redraw_pending": False,
            "mouse_gestures": {"middle": gesture},
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "is_mouse_button_down", return_value=True),
            patch.object(viewer_app, "_redraw_preview") as redraw,
        ):
            for position in ((15.0, 17.0), (25.0, 27.0), (35.0, 37.0)):
                gesture["last_screen"] = position
                viewer_app._update_pan_from_mouse()

            self.assertEqual(
                (viewer_app.state["pan_x"], viewer_app.state["pan_y"]),
                (32.0, 33.0),
            )
            self.assertTrue(viewer_app.state["pan_redraw_pending"])
            redraw.assert_not_called()
            viewer_app._flush_pending_pan_redraw()
            redraw.assert_called_once()
            self.assertFalse(viewer_app.state["pan_redraw_pending"])

    def test_pan_move_arriving_during_redraw_stays_pending(self):
        redraw_count = 0

        def redraw_with_concurrent_pan():
            nonlocal redraw_count
            redraw_count += 1
            if redraw_count == 1:
                viewer_app.state["pan_redraw_pending"] = True

        with (
            patch.dict(
                viewer_app.state,
                {"pan_redraw_pending": True},
                clear=False,
            ),
            patch.object(
                viewer_app,
                "_redraw_preview",
                side_effect=redraw_with_concurrent_pan,
            ),
        ):
            self.assertTrue(viewer_app._flush_pending_pan_redraw())
            self.assertTrue(viewer_app.state["pan_redraw_pending"])
            self.assertTrue(viewer_app._flush_pending_pan_redraw())
            self.assertFalse(viewer_app.state["pan_redraw_pending"])

        self.assertEqual(redraw_count, 2)

    def test_pan_coalescing_keeps_all_brush_move_callbacks(self):
        positions = ((8.0, 5.0), (11.0, 6.0), (14.0, 8.0))
        gesture = _gesture(
            "left",
            start=(2.0, 2.0),
            current=(2.0, 2.0),
        )
        replacement = {
            "mode": "mask_brush",
            "pan_redraw_pending": False,
            "mouse_gestures": {"left": gesture},
        }
        brush_positions = []

        def record_brush_position():
            brush_positions.append(tuple(gesture["last_screen"]))

        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "set_value"),
            patch.object(viewer_app, "_update_rectangle_drag_from_mouse"),
            patch.object(
                viewer_app,
                "_update_mask_brush_from_mouse",
                side_effect=record_brush_position,
            ),
            patch.object(viewer_app, "_update_brush_cursor_from_mouse"),
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
            patch.object(viewer_app, "_update_debug_coords"),
            patch.object(viewer_app, "screen_to_world", return_value=None),
        ):
            for position in positions:
                viewer_app._mouse_move_callback(_app_data=position)

        self.assertEqual(brush_positions, list(positions))
        self.assertFalse(viewer_app.state["pan_redraw_pending"])

    def test_queued_right_moves_do_not_pan_after_physical_release(self):
        gesture = _gesture(
            "right",
            start=(100.0, 100.0),
            current=(100.0, 100.0),
        )
        replacement = {
            "pan_x": 0.0,
            "pan_y": 0.0,
            "last_pan_mouse": None,
            "mouse_gestures": {"right": gesture},
        }
        physical_down = True

        def is_mouse_button_down(button):
            return physical_down and button == viewer_app.dpg.mvMouseButton_Right

        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(
                viewer_app.dpg,
                "is_mouse_button_down",
                side_effect=is_mouse_button_down,
            ),
            patch.object(viewer_app.dpg, "set_value"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_update_rectangle_drag_from_mouse"),
            patch.object(viewer_app, "_update_mask_brush_from_mouse"),
            patch.object(viewer_app, "_update_brush_cursor_from_mouse"),
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
            patch.object(viewer_app, "_update_debug_coords"),
            patch.object(viewer_app, "screen_to_world", return_value=None),
        ):
            viewer_app._mouse_move_callback(_app_data=(112.0, 100.0))
            self.assertEqual(
                (viewer_app.state["pan_x"], viewer_app.state["pan_y"]),
                (12.0, 0.0),
            )

            physical_down = False
            for current in ((124.0, 100.0), (136.0, 100.0), (148.0, 100.0)):
                viewer_app._mouse_move_callback(_app_data=current)
                self.assertEqual(
                    (viewer_app.state["pan_x"], viewer_app.state["pan_y"]),
                    (12.0, 0.0),
                )
                self.assertIn("right", viewer_app.state["mouse_gestures"])

            viewer_app._mouse_release_callback(_user_data="right")
            self.assertNotIn("right", viewer_app.state["mouse_gestures"])
            viewer_app._mouse_move_callback(_app_data=(170.0, 100.0))
            self.assertEqual(
                (viewer_app.state["pan_x"], viewer_app.state["pan_y"]),
                (12.0, 0.0),
            )

    def test_queued_middle_moves_do_not_pan_after_physical_release(self):
        gesture = _gesture(
            "middle",
            start=(10.0, 10.0),
            current=(25.0, 18.0),
            dragged=True,
        )
        replacement = {
            "pan_x": 2.0,
            "pan_y": 3.0,
            "mouse_gestures": {"middle": gesture},
        }
        physical_down = True

        def is_mouse_button_down(button):
            return physical_down and button == viewer_app.dpg.mvMouseButton_Middle

        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(
                viewer_app.dpg,
                "is_mouse_button_down",
                side_effect=is_mouse_button_down,
            ),
            patch.object(viewer_app, "_redraw_preview"),
        ):
            viewer_app._update_pan_from_mouse()
            self.assertEqual(
                (viewer_app.state["pan_x"], viewer_app.state["pan_y"]),
                (17.0, 11.0),
            )

            physical_down = False
            for current in ((35.0, 24.0), (45.0, 30.0), (55.0, 36.0)):
                gesture["last_screen"] = current
                viewer_app._update_pan_from_mouse()
                self.assertEqual(
                    (viewer_app.state["pan_x"], viewer_app.state["pan_y"]),
                    (17.0, 11.0),
                )
            self.assertIn("middle", viewer_app.state["mouse_gestures"])

    def test_right_pan_and_release_do_not_depend_on_preview_hover(self):
        gesture = _gesture(
            "right",
            start=(10.0, 10.0),
            current=(30.0, 20.0),
            dragged=True,
        )
        replacement = {
            "pan_x": 0.0,
            "pan_y": 0.0,
            "pan_redraw_pending": False,
            "mouse_gestures": {"right": gesture},
        }
        physical_down = True
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(
                viewer_app.dpg,
                "is_mouse_button_down",
                side_effect=lambda button: physical_down
                and button == viewer_app.dpg.mvMouseButton_Right,
            ),
            patch.object(viewer_app.dpg, "get_mouse_pos", return_value=(40.0, 30.0)),
            patch.object(viewer_app.dpg, "is_item_hovered") as is_hovered,
            patch.object(viewer_app, "_redraw_preview") as redraw,
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
        ):
            viewer_app._update_pan_from_mouse()
            self.assertEqual(
                (viewer_app.state["pan_x"], viewer_app.state["pan_y"]),
                (20.0, 10.0),
            )
            self.assertTrue(viewer_app.state["pan_redraw_pending"])
            redraw.assert_not_called()

            physical_down = False
            gesture["last_screen"] = (40.0, 30.0)
            viewer_app._update_pan_from_mouse()
            viewer_app._mouse_release_callback(_user_data="right")
            self.assertEqual(
                (viewer_app.state["pan_x"], viewer_app.state["pan_y"]),
                (20.0, 10.0),
            )
            self.assertNotIn("right", viewer_app.state["mouse_gestures"])
            self.assertTrue(viewer_app.state["pan_redraw_pending"])
            viewer_app._flush_pending_pan_redraw()
            redraw.assert_called_once()
            self.assertFalse(viewer_app.state["pan_redraw_pending"])
        is_hovered.assert_not_called()

    def test_right_drag_over_accepted_hole_stops_on_physical_release(self):
        session = _session()
        gesture = _gesture(
            "right",
            start=(10.0, 10.0),
            current=(30.0, 20.0),
            dragged=True,
        )
        replacement = {
            "pan_x": 0.0,
            "pan_y": 0.0,
            "mouse_gestures": {"right": gesture},
        }
        physical_down = True

        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(
                viewer_app.dpg,
                "is_mouse_button_down",
                side_effect=lambda button: physical_down
                and button == viewer_app.dpg.mvMouseButton_Right,
            ),
            patch.object(viewer_app.dpg, "get_mouse_pos", return_value=(45.0, 30.0)),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_route_short_right_click") as short_click,
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
        ):
            viewer_app._update_pan_from_mouse()
            self.assertEqual(
                (viewer_app.state["pan_x"], viewer_app.state["pan_y"]),
                (20.0, 10.0),
            )

            physical_down = False
            gesture["last_screen"] = (45.0, 30.0)
            viewer_app._update_pan_from_mouse()
            viewer_app._mouse_release_callback(_user_data="right")
            self.assertEqual(
                (viewer_app.state["pan_x"], viewer_app.state["pan_y"]),
                (20.0, 10.0),
            )
            self.assertNotIn("right", viewer_app.state["mouse_gestures"])

        self.assertTrue(session.result.candidates[0].accepted)
        short_click.assert_not_called()

    def test_drag_release_never_routes_hole_action(self):
        gesture = _gesture(
            "right",
            start=(10.0, 10.0),
            current=(30.0, 10.0),
            dragged=True,
        )
        with (
            patch.dict(
                viewer_app.state,
                {"mouse_gestures": {"right": gesture}},
                clear=False,
            ),
            patch.object(viewer_app.dpg, "get_mouse_pos", return_value=(30.0, 10.0)),
            patch.object(viewer_app, "_finish_drag_gesture") as finish_drag,
            patch.object(viewer_app, "_route_short_right_click") as short_click,
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
        ):
            viewer_app._mouse_release_callback(_user_data="right")
        finish_drag.assert_called_once()
        short_click.assert_not_called()

    def test_short_right_click_on_rejected_hole_does_not_change_status(self):
        session = _session()
        gesture = _gesture(
            "right",
            start=(10.0, 10.0),
            current=(12.0, 11.0),
        )
        with (
            patch.dict(
                viewer_app.state,
                {"mouse_gestures": {"right": gesture}},
                clear=False,
            ),
            patch.object(viewer_app.dpg, "get_mouse_pos", return_value=(12.0, 11.0)),
            patch.object(viewer_app, "_hole_at_screen", return_value=2),
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
            patch.object(viewer_app, "_update_hole_hover_from_mouse"),
        ):
            viewer_app._mouse_release_callback(_user_data="right")

        self.assertFalse(session.result.candidates[1].accepted)

    def test_wheel_zoom_and_hit_test_still_work(self):
        replacement = {
            "zoom": 1.0,
            "pan_x": 0.0,
            "pan_y": 0.0,
            "image_width": 100,
            "image_height": 80,
            "last_event_mouse_position": (50.0, 40.0),
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(
                viewer_app.dpg,
                "get_item_state",
                return_value={"pos": (0.0, 0.0), "rect_size": (400.0, 300.0)},
            ),
            patch.object(viewer_app.dpg, "get_item_rect_min", return_value=(0.0, 0.0)),
            patch.object(viewer_app.dpg, "set_value"),
            patch.object(viewer_app, "_redraw_preview"),
        ):
            viewer_app._mouse_wheel_callback(app_data=1.0)
            self.assertEqual(viewer_app.state["zoom"], viewer_app.ZOOM_STEP)
        self.assertEqual(
            self._screen_hit(
                _session(),
                zoom=viewer_app.ZOOM_STEP,
                pan=(13.0, -8.0),
                candidate_id=2,
            ),
            2,
        )

    def test_visible_preview_wheel_bounds_are_half_open(self):
        with (
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(
                viewer_app.dpg,
                "get_item_state",
                return_value={
                    "pos": (100.0, 200.0),
                    "rect_size": (300.0, 150.0),
                },
            ),
        ):
            for position in (
                (100.0, 200.0),
                (399.999, 349.999),
                (250.0, 275.0),
            ):
                with self.subTest(position=position):
                    self.assertTrue(
                        viewer_app._screen_position_is_in_visible_preview(position)
                    )

            for position in (
                (99.999, 200.0),
                (100.0, 199.999),
                (400.0, 250.0),
                (250.0, 350.0),
            ):
                with self.subTest(position=position):
                    self.assertFalse(
                        viewer_app._screen_position_is_in_visible_preview(position)
                    )

    def test_wheel_over_sidebar_ui_does_not_zoom_hidden_drawlist(self):
        replacement = {
            "zoom": 1.0,
            "last_event_mouse_position": None,
        }
        ui_positions = {
            "sidebar": (450.0, 20.0),
            "input": (470.0, 55.0),
            "combo": (470.0, 90.0),
            "collapsing_header": (470.0, 125.0),
            "status_log": (470.0, 210.0),
            "nested_scrollable_child": (470.0, 260.0),
            "sidebar_scroll_boundary": (470.0, 299.0),
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(
                viewer_app.dpg,
                "get_item_state",
                return_value={"pos": (0.0, 0.0), "rect_size": (400.0, 300.0)},
            ),
            patch.object(
                viewer_app.dpg,
                "get_item_rect_min",
                return_value=(0.0, 0.0),
            ),
            patch.object(viewer_app, "_set_zoom") as set_zoom,
        ):
            for area, position in ui_positions.items():
                with self.subTest(area=area):
                    self.assertTrue(
                        viewer_app._screen_position_is_on_preview(position)
                    )
                    viewer_app.state["last_event_mouse_position"] = position
                    viewer_app._mouse_wheel_callback(app_data=1.0)
                    viewer_app._mouse_wheel_callback(app_data=-1.0)

        set_zoom.assert_not_called()

    def test_delayed_sidebar_wheel_does_not_use_current_preview_cursor(self):
        replacement = {
            "zoom": 1.0,
            "last_event_mouse_position": (470.0, 120.0),
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(
                viewer_app.dpg,
                "get_item_state",
                return_value={"pos": (0.0, 0.0), "rect_size": (400.0, 300.0)},
            ),
            patch.object(
                viewer_app.dpg,
                "get_mouse_pos",
                return_value=(100.0, 100.0),
            ) as get_mouse_pos,
            patch.object(viewer_app, "_set_zoom") as set_zoom,
        ):
            viewer_app._mouse_wheel_callback(app_data=1.0)

        set_zoom.assert_not_called()
        get_mouse_pos.assert_not_called()

    def test_delayed_preview_wheel_uses_event_position_and_zoom_direction(self):
        event_position = (160.0, 140.0)
        replacement = {
            "zoom": 2.0,
            "last_event_mouse_position": event_position,
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=True),
            patch.object(
                viewer_app.dpg,
                "get_item_state",
                return_value={"pos": (10.0, 20.0), "rect_size": (400.0, 300.0)},
            ),
            patch.object(
                viewer_app.dpg,
                "get_item_rect_min",
                return_value=(10.0, 20.0),
            ),
            patch.object(
                viewer_app.dpg,
                "get_mouse_pos",
                return_value=(700.0, 200.0),
            ) as get_mouse_pos,
            patch.object(viewer_app, "_set_zoom") as set_zoom,
        ):
            viewer_app._mouse_wheel_callback(app_data=-1.0)

        set_zoom.assert_called_once_with(
            2.0 / viewer_app.ZOOM_STEP,
            anchor_canvas_pos=(150.0, 120.0),
        )
        get_mouse_pos.assert_not_called()

    def test_wheel_without_event_position_is_noop(self):
        with (
            patch.dict(
                viewer_app.state,
                {"zoom": 1.0, "last_event_mouse_position": None},
                clear=False,
            ),
            patch.object(viewer_app.dpg, "get_item_state") as get_item_state,
            patch.object(viewer_app, "_set_zoom") as set_zoom,
        ):
            viewer_app._mouse_wheel_callback(app_data=1.0)

        get_item_state.assert_not_called()
        set_zoom.assert_not_called()

    def test_hover_and_short_click_use_same_hit_test(self):
        session = _session()
        with (
            patch.dict(
                viewer_app.state,
                {
                    "mode": "rectangle",
                    "hovered_hole_id": None,
                    "mouse_gestures": {},
                },
                clear=False,
            ),
            patch.object(viewer_app.dpg, "get_mouse_pos", return_value=(10.0, 10.0)),
            patch.object(
                viewer_app.dpg,
                "does_item_exist",
                side_effect=lambda tag: tag == viewer_app.IMAGE_TAG,
            ),
            patch.object(viewer_app.dpg, "is_item_hovered", return_value=True),
            patch.object(viewer_app, "_hole_at_screen", return_value=2) as hit_test,
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(viewer_app, "_redraw_holes_overlay"),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
        ):
            self.assertEqual(viewer_app._update_hole_hover_from_mouse(), 2)
            self.assertTrue(viewer_app._route_short_left_click((10.0, 10.0)))
        self.assertEqual(hit_test.call_count, 2)
        self.assertTrue(session.result.candidates[1].accepted)

    def test_hover_highlight_adds_one_screen_space_ring(self):
        session = _session()
        with (
            patch.dict(
                viewer_app.state,
                {"hole_overlay_source": "in_memory", "hovered_hole_id": 2},
                clear=False,
            ),
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(
                viewer_app.dpg,
                "does_item_exist",
                side_effect=lambda tag: tag == viewer_app.IMAGE_TAG,
            ),
            patch.object(viewer_app.dpg, "add_draw_layer"),
            patch.object(viewer_app.dpg, "draw_text"),
            patch.object(viewer_app, "_display_layer_enabled", return_value=True),
            patch.object(
                viewer_app,
                "_hole_candidate_hit_region",
                side_effect=lambda candidate: HoleHitRegion(
                    candidate.id,
                    10.0,
                    10.0,
                    3.0,
                    2.0,
                ),
            ),
            patch.object(viewer_app, "_draw_world_radius_ellipse") as draw_ellipse,
        ):
            viewer_app._redraw_holes_overlay()

        highlight = next(
            call
            for call in draw_ellipse.call_args_list
            if call.kwargs["thickness"] == 4
        )
        self.assertEqual(highlight.args[1], (7.0, 6.0))

    def test_brush_drag_creates_stroke_but_short_hole_click_does_not(self):
        grid = DensityGrid(
            density=np.ones((20, 20), dtype=np.uint32),
            cell_size=1.0,
            min_x=0.0,
            min_y=0.0,
        )
        editing = MaskEditingSession(
            grid=grid,
            working_area=None,
            base_mask=np.ones((20, 20), dtype=np.uint8),
            edited_mask=np.ones((20, 20), dtype=np.uint8),
            working_area_mask=np.ones((20, 20), dtype=bool),
        )
        gesture = _gesture(
            "left",
            start=(2.0, 2.0),
            current=(8.0, 2.0),
            dragged=True,
            start_world=(2.5, 2.5),
        )
        replacement = {
            "mode": "mask_brush",
            "mouse_gestures": {"left": gesture},
            "mask_edits": [],
            "last_brush_world": None,
            "last_brush_image": None,
            "active_brush_stroke_id": None,
            "next_brush_stroke_id": 1,
            "contour_processing_result": None,
        }
        with (
            patch.dict(viewer_app.state, replacement, clear=False),
            patch.object(viewer_app, "_warn_preview_grid_params_not_loaded", return_value=False),
            patch.object(
                viewer_app,
                "screen_to_world",
                return_value=(8.0, 2.0, 8.0, 2.0, 8.5, 2.5),
            ),
            patch.object(viewer_app, "_active_mask_editing_session", return_value=editing),
            patch.object(viewer_app.dpg, "get_value", return_value=2.0),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=False),
            patch.object(viewer_app, "_update_processing_mask_texture"),
            patch.object(viewer_app, "_update_contour_info"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
        ):
            viewer_app._finish_drag_gesture("left", gesture, (8.0, 2.0))
        self.assertEqual(len(editing.history), 1)
        self.assertTrue(replacement["mask_edits"])
        self.assertEqual(editing.edited_mask[2, 5], 0)

        session = _session()
        with (
            patch.dict(viewer_app.state, {"mode": "mask_brush"}, clear=False),
            patch.object(viewer_app, "_hole_at_screen", return_value=2),
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
            patch.object(viewer_app, "_apply_mask_brush_at_screen") as brush,
        ):
            self.assertTrue(viewer_app._route_short_left_click((5.0, 5.0)))
        brush.assert_not_called()

    def test_manual_hole_pick_has_priority_over_existing_candidate(self):
        with (
            patch.dict(
                viewer_app.state,
                {"pick_manual_hole_center": True, "mode": "mask_brush"},
                clear=False,
            ),
            patch.object(viewer_app, "_warn_preview_grid_params_not_loaded", return_value=False),
            patch.object(
                viewer_app,
                "screen_to_world",
                return_value=(0.0, 0.0, 0.0, 0.0, 4.0, 5.0),
            ),
            patch.object(viewer_app, "_finish_manual_hole_center_pick") as pick,
            patch.object(viewer_app, "_hole_at_screen") as hit_test,
        ):
            self.assertTrue(viewer_app._route_short_left_click((5.0, 5.0)))
        pick.assert_called_once_with(4.0, 5.0)
        hit_test.assert_not_called()

    def test_mouse_and_id_controls_share_helper_without_processing(self):
        with (
            patch.object(viewer_app, "_hole_at_screen", return_value=2),
            patch.object(viewer_app, "_set_manual_hole_status_by_id") as status,
        ):
            viewer_app._route_short_left_click((1.0, 1.0))
        status.assert_called_once_with(2, accepted=True)

        with (
            patch.object(viewer_app.dpg, "get_value", return_value=2),
            patch.object(viewer_app, "_set_manual_hole_status_by_id") as status,
        ):
            viewer_app._accept_selected_hole_callback()
            viewer_app._reject_selected_hole_callback()
        self.assertEqual(
            status.call_args_list[0].kwargs,
            {"accepted": True},
        )
        self.assertEqual(
            status.call_args_list[1].kwargs,
            {"accepted": False},
        )

        session = _session()
        with (
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
            patch.object(viewer_app, "find_holes_for_current_mask") as detector,
            patch.object(viewer_app, "find_preliminary_contour_for_working_area") as contour,
            patch("app.services.coarse_processing.prepare_density") as density,
            patch("app.core.xyz_reader.compute_stats") as source,
        ):
            viewer_app._set_manual_hole_status_by_id(2, accepted=True)
        detector.assert_not_called()
        contour.assert_not_called()
        density.assert_not_called()
        source.assert_not_called()

    def test_legacy_id_controls_still_work(self):
        legacy = {
            "id": 17,
            "accepted": False,
            "enabled": False,
            "reject_reason": "automatic_reason",
            "group_id": None,
        }

        def control_value(tag):
            if tag == viewer_app.MOVE_HOLE_ID_TAG:
                return 17
            if tag == viewer_app.MOVE_HOLE_TARGET_GROUP_TAG:
                return ""
            raise AssertionError(tag)

        with (
            patch.dict(
                viewer_app.state,
                {"holes": [legacy], "hole_groups": []},
                clear=False,
            ),
            patch.object(viewer_app, "_active_hole_detection_session", return_value=None),
            patch.object(viewer_app.dpg, "get_value", side_effect=control_value),
            patch.object(viewer_app, "_refresh_hole_views"),
            patch.object(viewer_app, "_set_status"),
        ):
            viewer_app._accept_selected_hole_callback()
            self.assertTrue(legacy["accepted"])
            viewer_app._reject_selected_hole_callback()
            self.assertFalse(legacy["accepted"])
            self.assertEqual(legacy["reject_reason"], "manual_reject")


if __name__ == "__main__":
    unittest.main()
