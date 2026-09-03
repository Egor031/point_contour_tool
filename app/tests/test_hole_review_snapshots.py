from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.hole_detector import HoleCandidate  # noqa: E402
from app.services.coarse_processing import HoleDetectionResult  # noqa: E402
from app.ui.hole_workflow import (  # noqa: E402
    DuplicateReviewedHoleIdError,
    HoleDetectionParameters,
    HoleDetectionSession,
    UnsupportedReviewedHoleSourceError,
    accepted_hole_snapshots,
    advance_hole_review_revision,
    apply_manual_hole_status,
    build_reviewed_hole_snapshots,
    current_hole_review_revision,
    invalidate_hole_detection_state,
)


def _candidate(
    hole_id: int,
    *,
    accepted: bool,
    group_id: str | None,
    center: tuple[float, float] = (10.5, -20.25),
    radius: float = 3.75,
    reject_reason: str = "",
) -> HoleCandidate:
    return HoleCandidate(
        id=hole_id,
        center_x=center[0],
        center_y=center[1],
        radius=radius,
        diameter=radius * 2.0,
        center_px=11.0,
        center_py=12.0,
        radius_px=4.0,
        area_cells=91,
        area_mm2=56.875,
        bbox_width_mm=7.4,
        bbox_height_mm=7.7,
        aspect_ratio=1.04,
        circularity=0.89,
        mean_error_mm=0.16,
        max_error_mm=0.52,
        error_ratio=0.043,
        accepted=accepted,
        reject_reason=reject_reason,
        group_id=group_id,
    )


def _session(*candidates: HoleCandidate) -> HoleDetectionSession:
    groups = [
        {
            "id": group_id,
            "name": group_id,
            "diameter": 8.0,
            "radius": 4.0,
            "count": sum(
                1
                for candidate in candidates
                if candidate.accepted and candidate.group_id == group_id
            ),
            "enabled": True,
        }
        for group_id in sorted(
            {
                candidate.group_id
                for candidate in candidates
                if candidate.group_id is not None
            }
        )
    ]
    return HoleDetectionSession(
        result=HoleDetectionResult(candidates=list(candidates), groups=groups),
        density_session=None,
        contour_session=object(),  # type: ignore[arg-type]
        mask_editing_session=object(),  # type: ignore[arg-type]
        working_area=None,
        coarse_mask_revision=1,
        parameters=HoleDetectionParameters(),
        automatic_acceptance={
            int(candidate.id): bool(candidate.accepted) for candidate in candidates
        },
    )


def _manual_hole(
    hole_id: int,
    *,
    group_id: str | None = None,
    accepted: bool = True,
) -> dict:
    return {
        "id": hole_id,
        "accepted": accepted,
        "enabled": accepted,
        "reject_reason": "" if accepted else "manual_reject",
        "center_x": 70.25,
        "center_y": -4.5,
        "radius": 2.75,
        "diameter": 5.5,
        "group_id": group_id,
        "source": "manual",
    }


class TestReviewedHoleSnapshotBuilder(unittest.TestCase):
    def test_automatic_decisions_are_preserved(self):
        accepted = _candidate(1, accepted=True, group_id="G1")
        rejected = _candidate(
            2,
            accepted=False,
            group_id=None,
            reject_reason="bad_circle_fit",
        )
        snapshots = build_reviewed_hole_snapshots(_session(accepted, rejected))

        self.assertTrue(snapshots[0].automatic_accepted)
        self.assertTrue(snapshots[0].final_accepted)
        self.assertEqual(snapshots[0].decision_source, "automatic")
        self.assertEqual(snapshots[0].group_id, "G1")
        self.assertFalse(snapshots[1].automatic_accepted)
        self.assertFalse(snapshots[1].final_accepted)
        self.assertEqual(snapshots[1].decision_source, "automatic")

    def test_manual_accept_preserves_reject_reason_metrics_and_geometry(self):
        candidate = _candidate(
            7,
            accepted=False,
            group_id=None,
            center=(31.5, 42.25),
            radius=5.125,
            reject_reason="low_circularity",
        )
        session = _session(candidate)
        self.assertTrue(apply_manual_hole_status(session, 7, accepted=True))

        snapshot = build_reviewed_hole_snapshots(session)[0]
        self.assertFalse(snapshot.automatic_accepted)
        self.assertTrue(snapshot.final_accepted)
        self.assertEqual(snapshot.decision_source, "user")
        self.assertEqual(snapshot.automatic_reject_reason, "low_circularity")
        self.assertEqual(snapshot.preliminary_center, (31.5, 42.25))
        self.assertEqual(snapshot.preliminary_radius, 5.125)
        self.assertEqual(snapshot.detector_metrics.area_cells, 91)
        self.assertEqual(snapshot.detector_metrics.mean_error_mm, 0.16)
        self.assertIsNone(snapshot.group_id)

    def test_manual_reject_preserves_automatic_acceptance(self):
        candidate = _candidate(3, accepted=True, group_id="G1")
        session = _session(candidate)
        self.assertTrue(apply_manual_hole_status(session, 3, accepted=False))

        snapshot = build_reviewed_hole_snapshots(session)[0]
        self.assertTrue(snapshot.automatic_accepted)
        self.assertFalse(snapshot.final_accepted)
        self.assertEqual(snapshot.decision_source, "user")
        self.assertEqual(snapshot.group_id, "G1")

    def test_manual_hole_uses_same_domain_snapshot(self):
        snapshot = build_reviewed_hole_snapshots(
            None,
            [_manual_hole(15, group_id="G4")],
        )[0]
        self.assertEqual(snapshot.hole_id, 15)
        self.assertEqual(snapshot.origin, "manual")
        self.assertIsNone(snapshot.automatic_accepted)
        self.assertTrue(snapshot.final_accepted)
        self.assertEqual(snapshot.decision_source, "user")
        self.assertEqual(snapshot.preliminary_center, (70.25, -4.5))
        self.assertEqual(snapshot.preliminary_radius, 2.75)
        self.assertEqual(snapshot.group_id, "G4")
        self.assertIsNone(snapshot.detector_metrics)
        self.assertIsNone(snapshot.automatic_reject_reason)

        rejected = build_reviewed_hole_snapshots(
            None,
            [_manual_hole(16, accepted=False)],
        )[0]
        self.assertFalse(rejected.final_accepted)

    def test_current_candidate_group_has_priority_and_holes_remain_distinct(self):
        first = _candidate(1, accepted=True, group_id="G1")
        second = _candidate(2, accepted=True, group_id="G1")
        third = _candidate(3, accepted=True, group_id="G3")
        session = _session(first, second, third)
        first.group_id = "G2"

        snapshots = build_reviewed_hole_snapshots(session)
        self.assertEqual([snapshot.hole_id for snapshot in snapshots], [1, 2, 3])
        self.assertEqual(
            [snapshot.group_id for snapshot in snapshots],
            ["G2", "G1", "G3"],
        )

    def test_accepted_view_excludes_final_rejected(self):
        automatic = _candidate(1, accepted=True, group_id="G1")
        rejected = _candidate(2, accepted=True, group_id="G1")
        session = _session(automatic, rejected)
        apply_manual_hole_status(session, 2, accepted=False)
        snapshots = build_reviewed_hole_snapshots(
            session,
            [_manual_hole(3)],
        )

        self.assertEqual(
            [snapshot.hole_id for snapshot in accepted_hole_snapshots(snapshots)],
            [1, 3],
        )

    def test_duplicate_ids_raise_instead_of_overwriting(self):
        session = _session(_candidate(1, accepted=True, group_id="G1"))
        with self.assertRaisesRegex(
            DuplicateReviewedHoleIdError,
            "duplicate reviewed hole ID: 1",
        ):
            build_reviewed_hole_snapshots(session, [_manual_hole(1)])

    def test_unbound_legacy_holes_are_not_wbc_input(self):
        from app.ui import viewer_app

        legacy = viewer_app._normalize_hole_json_item(
            {
                "id": 9,
                "accepted": True,
                "center_x": 1.0,
                "center_y": 2.0,
                "radius": 3.0,
            }
        )
        self.assertNotIn("source", legacy)
        with self.assertRaises(UnsupportedReviewedHoleSourceError):
            build_reviewed_hole_snapshots(None, [legacy])

    def test_builder_does_not_require_raster_state(self):
        session = _session(_candidate(1, accepted=True, group_id=None))
        snapshots = build_reviewed_hole_snapshots(session)
        self.assertEqual(len(snapshots), 1)

    def test_viewer_public_adapter_combines_live_detector_and_manual_holes(self):
        from app.ui import viewer_app

        session = _session(_candidate(1, accepted=True, group_id="G1"))
        with (
            patch.dict(
                viewer_app.state,
                {"holes": [_manual_hole(2)]},
                clear=False,
            ),
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
        ):
            snapshots = viewer_app.current_reviewed_hole_snapshots()
        self.assertEqual([snapshot.hole_id for snapshot in snapshots], [1, 2])
        self.assertEqual([snapshot.origin for snapshot in snapshots], ["detector", "manual"])


class TestHoleReviewRevision(unittest.TestCase):
    def test_revision_helpers_are_monotonic(self):
        state = {"hole_review_revision": 4}
        self.assertEqual(current_hole_review_revision(state), 4)
        self.assertEqual(advance_hole_review_revision(state), 5)
        self.assertEqual(advance_hole_review_revision(state), 6)

    def test_status_change_increments_but_noop_does_not(self):
        from app.ui import viewer_app

        session = _session(_candidate(1, accepted=False, group_id=None))
        with (
            patch.dict(
                viewer_app.state,
                {"hole_review_revision": 10, "holes": []},
                clear=False,
            ),
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
        ):
            self.assertTrue(viewer_app._set_manual_hole_status_by_id(1, accepted=True))
            self.assertEqual(viewer_app.state["hole_review_revision"], 11)
            self.assertFalse(viewer_app._set_manual_hole_status_by_id(1, accepted=True))
            self.assertEqual(viewer_app.state["hole_review_revision"], 11)
            self.assertTrue(viewer_app._set_manual_hole_status_by_id(1, accepted=False))
            self.assertEqual(viewer_app.state["hole_review_revision"], 12)
            self.assertFalse(viewer_app._set_manual_hole_status_by_id(1, accepted=False))
            self.assertEqual(viewer_app.state["hole_review_revision"], 12)

    def test_manual_add_and_clear_increment_revision_and_avoid_detector_ids(self):
        from app.ui import viewer_app

        session = _session(
            _candidate(1, accepted=True, group_id="G1"),
            _candidate(2, accepted=True, group_id="G1"),
        )

        def value(tag):
            return {
                viewer_app.MANUAL_HOLE_X_TAG: 20.0,
                viewer_app.MANUAL_HOLE_Y_TAG: 30.0,
                viewer_app.MANUAL_HOLE_DIAMETER_TAG: 6.0,
                viewer_app.MOVE_HOLE_TARGET_GROUP_TAG: "G1",
            }[tag]

        with (
            patch.dict(
                viewer_app.state,
                {
                    "hole_review_revision": 20,
                    "holes": [],
                    "hole_groups": session.result.groups,
                    "hole_detection_session": session,
                },
                clear=False,
            ),
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(viewer_app.dpg, "get_value", side_effect=value),
            patch.object(viewer_app.dpg, "does_item_exist", return_value=False),
            patch.object(viewer_app, "_refresh_hole_views"),
            patch.object(viewer_app, "_set_status"),
        ):
            viewer_app._add_manual_hole_callback()
            self.assertEqual(viewer_app.state["holes"][0]["id"], 3)
            self.assertEqual(viewer_app.state["hole_review_revision"], 21)

            with (
                patch.object(viewer_app, "_update_holes_stats"),
                patch.object(viewer_app, "_update_hole_groups_display"),
                patch.object(viewer_app, "_update_hole_group_target_combo"),
                patch.object(viewer_app, "_redraw_preview"),
            ):
                viewer_app._clear_holes_callback()
            self.assertEqual(viewer_app.state["hole_review_revision"], 22)
            self.assertEqual(viewer_app.state["holes"], [])
            self.assertIsNone(viewer_app.state["hole_detection_session"])

    def test_group_reassignment_increments_only_when_group_changes(self):
        from app.ui import viewer_app

        candidate = _candidate(5, accepted=True, group_id="G1")
        session = _session(candidate)
        groups = [
            *session.result.groups,
            {"id": "G2", "name": "G2", "diameter": 8.0, "radius": 4.0},
        ]

        def value(tag):
            return {
                viewer_app.MOVE_HOLE_ID_TAG: 5,
                viewer_app.MOVE_HOLE_TARGET_GROUP_TAG: "G2",
            }[tag]

        with (
            patch.dict(
                viewer_app.state,
                {
                    "hole_review_revision": 30,
                    "holes": [],
                    "hole_groups": groups,
                },
                clear=False,
            ),
            patch.object(viewer_app, "_active_hole_detection_session", return_value=session),
            patch.object(viewer_app.dpg, "get_value", side_effect=value),
            patch.object(viewer_app, "_refresh_hole_views"),
            patch.object(viewer_app, "_set_status"),
        ):
            viewer_app._move_hole_to_group_callback()
            self.assertEqual(candidate.group_id, "G2")
            self.assertEqual(viewer_app.state["hole_review_revision"], 31)
            viewer_app._move_hole_to_group_callback()
            self.assertEqual(viewer_app.state["hole_review_revision"], 31)

    def test_new_find_holes_replaces_review_and_increments_revision(self):
        from app.ui import viewer_app

        session = _session(_candidate(1, accepted=True, group_id="G1"))
        contour = object()
        editing = object()
        with (
            patch.dict(
                viewer_app.state,
                {
                    "hole_review_revision": 40,
                    "holes": [_manual_hole(9)],
                    "hole_groups": [{"id": "OLD"}],
                    "contour_processing_result": contour,
                },
                clear=False,
            ),
            patch.object(viewer_app, "PreliminaryContourSession", object),
            patch.object(viewer_app, "_active_mask_editing_session", return_value=editing),
            patch.object(
                viewer_app,
                "_selected_hole_detection_parameters",
                return_value=HoleDetectionParameters(),
            ),
            patch.object(viewer_app, "find_holes_for_current_mask", return_value=session),
            patch.object(viewer_app, "_update_hole_detection_info"),
            patch.object(viewer_app, "_update_hole_groups_display"),
            patch.object(viewer_app, "_update_hole_group_target_combo"),
            patch.object(viewer_app, "_redraw_preview"),
            patch.object(viewer_app, "_set_status"),
        ):
            viewer_app._find_holes_callback()
            self.assertIs(viewer_app.state["hole_detection_session"], session)
            self.assertEqual(viewer_app.state["holes"], [])
            self.assertIs(viewer_app.state["hole_groups"], session.result.groups)
            self.assertEqual(viewer_app.state["hole_review_revision"], 41)

    def test_invalidation_changes_revision_but_visibility_does_not(self):
        from app.ui import viewer_app

        session = _session(_candidate(1, accepted=True, group_id="G1"))
        state = {
            "hole_detection_session": session,
            "holes_outdated": False,
            "hole_review_revision": 50,
        }
        self.assertTrue(invalidate_hole_detection_state(state))
        self.assertEqual(state["hole_review_revision"], 51)

        with (
            patch.dict(
                viewer_app.state,
                {
                    "hole_review_revision": 60,
                    "visible_hole_group_ids": {},
                },
                clear=False,
            ),
            patch.object(viewer_app.dpg, "get_value", return_value=True),
            patch.object(viewer_app, "_redraw_preview"),
        ):
            viewer_app._hole_group_visibility_callback("show", user_data="G1")
            self.assertEqual(viewer_app.state["hole_review_revision"], 60)

            with (
                patch.object(viewer_app.dpg, "set_value"),
                patch.object(viewer_app, "_redraw_preview"),
            ):
                viewer_app.state.update(
                    {
                        "zoom": 1.0,
                        "pan_x": 0.0,
                        "pan_y": 0.0,
                    }
                )
                viewer_app._set_zoom(2.0, anchor_canvas_pos=(10.0, 10.0))
            self.assertEqual(viewer_app.state["hole_review_revision"], 60)

            viewer_app.state["mouse_gestures"] = {
                "right": {
                    "dragged": True,
                    "last_screen": (10.0, 0.0),
                    "pan_last_screen": (0.0, 0.0),
                }
            }
            with patch.object(viewer_app.dpg, "is_mouse_button_down", return_value=True):
                viewer_app._update_pan_from_mouse()
            self.assertEqual(viewer_app.state["hole_review_revision"], 60)

            with (
                patch.object(viewer_app.dpg, "does_item_exist", return_value=False),
                patch.object(viewer_app, "_redraw_holes_overlay"),
            ):
                viewer_app._update_hole_hover_from_mouse(force=True)
            self.assertEqual(viewer_app.state["hole_review_revision"], 60)


if __name__ == "__main__":
    unittest.main()
