from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.cache import density_cache_path, density_metadata_path  # noqa: E402
from app.core.density_parameters import normalize_cell_size  # noqa: E402
from app.services.coarse_processing import prepare_density  # noqa: E402
from app.ui.density_workflow import validate_density_request  # noqa: E402


NOISY_CELL_ABOVE = 0.80000001192092896
NOISY_CELL_BELOW = 0.79999995231628418


def _write_xyz(path: Path) -> None:
    path.write_text(
        "0 0 0\n0.4 0.4 1\n0.8 0.8 2\n1.6 1.6 3\n",
        encoding="utf-8",
    )


class TestCellSizeNormalization(unittest.TestCase):
    def test_gui_float_noise_has_one_canonical_value(self):
        self.assertEqual(normalize_cell_size(NOISY_CELL_ABOVE), 0.8)
        self.assertEqual(normalize_cell_size(NOISY_CELL_BELOW), 0.8)
        self.assertEqual(normalize_cell_size(0.8), 0.8)
        self.assertEqual(normalize_cell_size(1.5000001192092896), 1.5)
        self.assertNotEqual(normalize_cell_size(0.8), normalize_cell_size(0.81))

    def test_gui_validation_returns_canonical_cell_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.xyz"
            _write_xyz(source)

            _, cell_above = validate_density_request(source, NOISY_CELL_ABOVE)
            _, cell_below = validate_density_request(source, NOISY_CELL_BELOW)

            self.assertEqual(cell_above, 0.8)
            self.assertEqual(cell_below, 0.8)

    def test_noisy_values_share_readable_cache_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.xyz"
            cache_dir = root / "cache"
            _write_xyz(source)

            canonical_path = density_cache_path(source, 0.8, cache_dir)

            self.assertEqual(
                density_cache_path(source, NOISY_CELL_ABOVE, cache_dir),
                canonical_path,
            )
            self.assertEqual(
                density_cache_path(source, NOISY_CELL_BELOW, cache_dir),
                canonical_path,
            )
            self.assertTrue(canonical_path.name.endswith("density_cell_0_8.npy"))

    def test_second_noisy_value_reuses_first_density_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.xyz"
            cache_dir = root / "cache"
            _write_xyz(source)

            first = prepare_density(source, NOISY_CELL_ABOVE, cache_dir)
            second = prepare_density(source, NOISY_CELL_BELOW, cache_dir)
            distinct = prepare_density(source, 0.81, cache_dir)
            metadata = json.loads(
                density_metadata_path(source, 0.8, cache_dir).read_text(
                    encoding="utf-8"
                )
            )

            self.assertFalse(first.density_from_cache)
            self.assertTrue(second.density_from_cache)
            self.assertEqual(first.grid.cell_size, 0.8)
            self.assertEqual(second.grid.cell_size, 0.8)
            self.assertEqual(metadata["cell_size"], 0.8)
            self.assertFalse(distinct.density_from_cache)
            self.assertEqual(distinct.grid.cell_size, 0.81)
            self.assertNotEqual(
                density_cache_path(source, 0.8, cache_dir),
                density_cache_path(source, 0.81, cache_dir),
            )


if __name__ == "__main__":
    unittest.main()
