from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.density_grid import DensityGrid  # noqa: E402
from app.core.working_area import WorkingArea  # noqa: E402
from app.ui.density_workflow import (  # noqa: E402
    active_working_area_is_visible,
    apply_working_area_state,
    begin_rectangle_draft,
    clear_polygon_draft_state,
    clear_working_area_state,
    enter_polygon_mode_state,
    enter_rectangle_mode_state,
    finish_rectangle_draft,
    reset_source_dependent_state,
    update_rectangle_transient,
    working_area_draft_visibility,
)


class TestWorkingArea(unittest.TestCase):
    def test_rectangle_points_are_normalized_in_world_coordinates(self):
        area = WorkingArea.from_rectangle_points((8.0, -2.0), (3.0, 5.0))

        self.assertEqual(area.rectangle_bounds, (3.0, -2.0, 8.0, 5.0))

    def test_polygon_preserves_vertices_and_rejects_invalid_geometry(self):
        points = [(1.0, 2.0), (5.0, 2.0), (4.0, 6.0), (1.5, 5.0)]
        area = WorkingArea.from_polygon(points)

        self.assertEqual(area.polygon_points, tuple(points))
        with self.assertRaises(ValueError):
            WorkingArea.from_polygon([(0.0, 0.0), (1.0, 0.0)])
        with self.assertRaises(ValueError):
            WorkingArea.from_polygon([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])

    def test_processing_parameters_are_mutually_exclusive(self):
        rectangle = WorkingArea.from_rectangle_bounds((1.0, 2.0, 5.0, 6.0))
        polygon = WorkingArea.from_polygon(
            [(1.0, 1.0), (4.0, 1.0), (3.0, 5.0)]
        )

        self.assertEqual(
            rectangle.processing_parameters(),
            ((1.0, 2.0, 5.0, 6.0), None),
        )
        self.assertEqual(
            polygon.processing_parameters(),
            (None, [(1.0, 1.0), (4.0, 1.0), (3.0, 5.0)]),
        )

    def test_rectangle_grid_mask_uses_cell_centers_and_nonzero_origin(self):
        grid = DensityGrid(
            density=np.zeros((3, 4), dtype=np.uint32),
            cell_size=2.0,
            min_x=10.0,
            min_y=20.0,
        )
        area = WorkingArea.from_rectangle_points((16.0, 24.0), (12.0, 20.0))

        expected = np.zeros((3, 4), dtype=bool)
        expected[0:2, 1:3] = True
        np.testing.assert_array_equal(area.to_grid_mask(grid), expected)

    def test_polygon_grid_mask_uses_existing_production_roi_mapping(self):
        grid = DensityGrid(
            density=np.zeros((3, 4), dtype=np.uint32),
            cell_size=2.0,
            min_x=10.0,
            min_y=20.0,
        )
        area = WorkingArea.from_polygon(
            [(10.0, 20.0), (14.0, 20.0), (14.0, 24.0), (10.0, 24.0)]
        )

        expected = np.zeros((3, 4), dtype=bool)
        expected[:, 0:3] = True
        np.testing.assert_array_equal(area.to_grid_mask(grid), expected)

    def test_rectangle_transient_lifecycle(self):
        target: dict[str, object] = {
            "mode": "rectangle",
            "rectangle_roi": (0.0, 0.0, 1.0, 1.0),
            "roi_first_world": None,
            "roi_current_world": None,
        }

        self.assertIsNone(target["roi_first_world"])
        self.assertFalse(update_rectangle_transient(target, (3.0, 9.0)))
        begin_rectangle_draft(target, (8.0, 5.0))
        self.assertEqual(target["roi_first_world"], (8.0, 5.0))
        self.assertIsNone(target["rectangle_roi"])
        self.assertTrue(update_rectangle_transient(target, (3.0, 9.0)))
        self.assertEqual(target["roi_current_world"], (3.0, 9.0))

        bounds = finish_rectangle_draft(target, (3.0, 9.0))

        self.assertEqual(bounds, (3.0, 5.0, 8.0, 9.0))
        self.assertEqual(target["rectangle_roi"], bounds)
        self.assertIsNone(target["roi_first_world"])
        self.assertIsNone(target["roi_current_world"])

    def test_polygon_draft_survives_rectangle_mode_and_return(self):
        points = [(float(index), float(index % 7)) for index in range(40)]
        target: dict[str, object] = {
            "polygon_points": list(points),
            "polygon_finished": False,
            "rectangle_roi": None,
            "roi_first_world": (1.0, 2.0),
            "roi_current_world": (3.0, 4.0),
        }

        enter_rectangle_mode_state(target)
        self.assertEqual(target["polygon_points"], points)
        self.assertIsNone(target["roi_first_world"])
        self.assertIsNone(target["roi_current_world"])
        self.assertEqual(working_area_draft_visibility(target), (True, False))

        enter_polygon_mode_state(target)
        self.assertEqual(target["polygon_points"], points)
        self.assertEqual(working_area_draft_visibility(target), (False, True))

    def test_rectangle_draft_survives_polygon_mode_and_return(self):
        rectangle = (1.0, 2.0, 8.0, 9.0)
        target: dict[str, object] = {
            "rectangle_roi": rectangle,
            "polygon_points": [],
            "active_working_area": None,
        }

        enter_polygon_mode_state(target)
        self.assertEqual(target["rectangle_roi"], rectangle)
        enter_rectangle_mode_state(target)
        self.assertEqual(target["rectangle_roi"], rectangle)

    def test_polygon_mode_copies_active_polygon_only_when_draft_is_empty(self):
        active = WorkingArea.from_polygon(
            [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)]
        )
        target: dict[str, object] = {
            "polygon_points": [],
            "polygon_finished": False,
            "active_working_area": active,
        }

        enter_polygon_mode_state(target)

        self.assertEqual(target["polygon_points"], list(active.polygon_points))
        self.assertTrue(target["polygon_finished"])
        self.assertIs(target["active_working_area"], active)

    def test_active_area_survives_new_draft_until_next_apply(self):
        session = object()
        active_a = WorkingArea.from_rectangle_bounds((0.0, 0.0, 4.0, 4.0))
        draft_b = WorkingArea.from_rectangle_bounds((1.0, 1.0, 6.0, 5.0))
        target: dict[str, object] = {"polygon_points": []}

        apply_working_area_state(target, active_a, session)
        self.assertTrue(active_working_area_is_visible(target))
        enter_rectangle_mode_state(target)
        begin_rectangle_draft(target, (1.0, 1.0))
        finish_rectangle_draft(target, (6.0, 5.0))

        self.assertIs(target["active_working_area"], active_a)
        self.assertFalse(active_working_area_is_visible(target))
        apply_working_area_state(target, draft_b, session)
        self.assertIs(target["active_working_area"], draft_b)
        self.assertTrue(active_working_area_is_visible(target))

    def test_clear_polygon_keeps_rectangle_and_active_area(self):
        active = WorkingArea.from_rectangle_bounds((0.0, 0.0, 8.0, 8.0))
        rectangle = (1.0, 1.0, 4.0, 4.0)
        target: dict[str, object] = {
            "active_working_area": active,
            "rectangle_roi": rectangle,
            "polygon_points": [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)],
            "polygon_finished": True,
        }

        clear_polygon_draft_state(target)

        self.assertEqual(target["polygon_points"], [])
        self.assertFalse(target["polygon_finished"])
        self.assertEqual(target["rectangle_roi"], rectangle)
        self.assertIs(target["active_working_area"], active)

    def test_clear_selection_removes_active_drafts_and_transient_rectangle(self):
        target: dict[str, object] = {
            "active_working_area": WorkingArea.from_rectangle_bounds(
                (0.0, 0.0, 8.0, 8.0)
            ),
            "working_area_density_session": object(),
            "rectangle_roi": (1.0, 1.0, 4.0, 4.0),
            "polygon_points": [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)],
            "polygon_finished": True,
            "roi_first_world": (1.0, 1.0),
            "roi_current_world": (3.0, 4.0),
        }

        clear_working_area_state(target)

        self.assertIsNone(target["active_working_area"])
        self.assertIsNone(target["rectangle_roi"])
        self.assertEqual(target["polygon_points"], [])
        self.assertIsNone(target["roi_first_world"])
        self.assertIsNone(target["roi_current_world"])
        self.assertEqual(target["mode"], "rectangle")
        self.assertFalse(target["editing_overlay_visible"])

    def test_new_density_state_reset_invalidates_active_working_area(self):
        target: dict[str, object] = {}
        area = WorkingArea.from_rectangle_bounds((0.0, 0.0, 4.0, 4.0))
        apply_working_area_state(target, area, object())

        reset_source_dependent_state(target)

        self.assertIsNone(target["active_working_area"])
        self.assertIsNone(target["working_area_density_session"])
        self.assertFalse(target["selection_applied"])


if __name__ == "__main__":
    unittest.main()
