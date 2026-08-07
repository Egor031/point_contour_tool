from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.density_grid import DensityGrid  # noqa: E402
from app.ui import viewer_app  # noqa: E402
from app.ui.density_workflow import (  # noqa: E402
    density_grid_to_preview,
    grid_preview_params,
    preview_to_texture_rgba,
    reset_source_dependent_state,
    validate_density_request,
)


def _grid(width: int, height: int) -> DensityGrid:
    density = np.arange(width * height, dtype=np.uint32).reshape(height, width)
    return DensityGrid(
        density=density,
        cell_size=0.8,
        min_x=-125.5,
        min_y=48.25,
    )


class TestGuiDensityWorkflow(unittest.TestCase):
    def test_density_grid_becomes_uint8_preview_and_rgba_texture(self):
        preview = density_grid_to_preview(_grid(width=4, height=3), max_size=100)
        texture = preview_to_texture_rgba(preview)

        self.assertEqual(preview.shape, (3, 4))
        self.assertEqual(preview.dtype, np.uint8)
        self.assertEqual(texture.shape, (3, 4, 4))
        self.assertEqual(texture.dtype, np.float32)
        np.testing.assert_array_equal(texture[:, :, 3], np.ones((3, 4)))

    def test_density_preview_downsamples_to_actual_dimensions(self):
        preview = density_grid_to_preview(_grid(width=200, height=100), max_size=50)

        self.assertEqual(preview.shape, (25, 50))
        self.assertEqual(preview.dtype, np.uint8)

    def test_in_memory_grid_supplies_preview_metadata_without_report(self):
        grid = _grid(width=17, height=9)
        previous_result = viewer_app.state["density_result"]
        viewer_app.state["density_result"] = SimpleNamespace(grid=grid)
        try:
            params = viewer_app._get_preview_params()
        finally:
            viewer_app.state["density_result"] = previous_result

        self.assertEqual(params, (-125.5, 48.25, 0.8, 17, 9))
        self.assertEqual(grid_preview_params(grid), params)

    def test_density_request_validation_rejects_invalid_source_and_cell(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "source.xyz"
            source.write_text("0 0 0\n", encoding="utf-8")
            unsupported = temp_path / "source.csv"
            unsupported.write_text("0,0,0\n", encoding="utf-8")

            validated_source, validated_cell = validate_density_request(source, 0.8)
            self.assertEqual(validated_source, source)
            self.assertEqual(validated_cell, 0.8)

            invalid_requests = [
                ("", 0.8),
                (temp_path / "missing.xyz", 0.8),
                (unsupported, 0.8),
                (source, 0.0),
                (source, -1.0),
                (source, float("nan")),
                (source, float("inf")),
            ]
            for source_path, cell_size in invalid_requests:
                with self.subTest(source=source_path, cell_size=cell_size):
                    with self.assertRaises(ValueError):
                        validate_density_request(source_path, cell_size)

    def test_state_reset_removes_source_dependent_geometry_only(self):
        session = object()
        target = {
            "density_result": session,
            "rectangle_roi": (1.0, 2.0, 3.0, 4.0),
            "polygon_points": [(1.0, 2.0)],
            "mask_edits": [{"mode": "add"}],
            "contour_points": [(1.0, 2.0)],
            "mixed_contour_elements": [{"type": "LINE"}],
            "holes": [{"id": 1}],
            "hole_groups": [{"id": "G1"}],
            "manual_hole_center_world": (3.0, 4.0),
            "selection_applied": True,
            "unrelated": "preserved",
        }

        reset_source_dependent_state(target)

        self.assertIs(target["density_result"], session)
        self.assertEqual(target["unrelated"], "preserved")
        self.assertIsNone(target["rectangle_roi"])
        self.assertEqual(target["polygon_points"], [])
        self.assertEqual(target["mask_edits"], [])
        self.assertEqual(target["contour_points"], [])
        self.assertEqual(target["mixed_contour_elements"], [])
        self.assertEqual(target["holes"], [])
        self.assertEqual(target["hole_groups"], [])
        self.assertIsNone(target["manual_hole_center_world"])
        self.assertFalse(target["selection_applied"])


if __name__ == "__main__":
    unittest.main()
