from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.cache import (  # noqa: E402
    CACHE_VERSION,
    density_cache_path,
    density_metadata_path,
    get_file_signature,
    stats_cache_path,
)
from app.services.coarse_processing import prepare_density  # noqa: E402


def _write_xyz(path: Path, points: list[tuple[float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{x} {y} {z}\n" for x, y, z in points),
        encoding="utf-8",
    )


def _points() -> list[tuple[float, float, float]]:
    return [
        (0.0, 0.0, 0.0),
        (0.2, 0.2, 1.0),
        (1.0, 1.0, 2.0),
        (2.0, 2.0, 3.0),
    ]


class TestCacheRegression(unittest.TestCase):
    def test_unchanged_source_and_cell_use_stats_and_density_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.xyz"
            cache_dir = root / "cache"
            _write_xyz(source, _points())

            first = prepare_density(source, 1.0, cache_dir)
            second = prepare_density(source, 1.0, cache_dir)

            self.assertFalse(first.stats_from_cache)
            self.assertFalse(first.density_from_cache)
            self.assertTrue(second.stats_from_cache)
            self.assertTrue(second.density_from_cache)
            self.assertTrue(stats_cache_path(source, cache_dir).is_file())
            self.assertTrue(density_cache_path(source, 1.0, cache_dir).is_file())
            self.assertTrue(density_metadata_path(source, 1.0, cache_dir).is_file())

    def test_different_cell_size_does_not_reuse_density_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.xyz"
            cache_dir = root / "cache"
            _write_xyz(source, _points())

            prepare_density(source, 1.0, cache_dir)
            different_cell = prepare_density(source, 0.5, cache_dir)
            repeated_cell = prepare_density(source, 0.5, cache_dir)

            self.assertTrue(different_cell.stats_from_cache)
            self.assertFalse(different_cell.density_from_cache)
            self.assertTrue(repeated_cell.stats_from_cache)
            self.assertTrue(repeated_cell.density_from_cache)
            self.assertNotEqual(
                density_cache_path(source, 1.0, cache_dir),
                density_cache_path(source, 0.5, cache_dir),
            )

    def test_modified_source_invalidates_statistics_and_density(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.xyz"
            cache_dir = root / "cache"
            _write_xyz(source, _points())

            first_signature = get_file_signature(source)
            prepare_density(source, 1.0, cache_dir)

            changed_points = _points() + [(3.0, 3.0, 4.0)]
            _write_xyz(source, changed_points)
            second_signature = get_file_signature(source)
            rebuilt = prepare_density(source, 1.0, cache_dir)

            self.assertNotEqual(first_signature, second_signature)
            self.assertFalse(rebuilt.stats_from_cache)
            self.assertFalse(rebuilt.density_from_cache)
            self.assertEqual(rebuilt.stats.point_count, len(changed_points))
            self.assertEqual(int(rebuilt.grid.density.sum()), len(changed_points))

    def test_same_stem_in_different_directories_uses_distinct_cache_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_a = root / "scan_a" / "detail.xyz"
            source_b = root / "scan_b" / "detail.xyz"
            cache_dir = root / "cache"
            _write_xyz(source_a, _points())
            _write_xyz(source_b, _points() + [(4.0, 4.0, 4.0)])

            first_a = prepare_density(source_a, 1.0, cache_dir)
            first_b = prepare_density(source_b, 1.0, cache_dir)
            second_a = prepare_density(source_a, 1.0, cache_dir)
            second_b = prepare_density(source_b, 1.0, cache_dir)

            self.assertFalse(first_a.density_from_cache)
            self.assertFalse(first_b.density_from_cache)
            self.assertTrue(second_a.density_from_cache)
            self.assertTrue(second_b.density_from_cache)
            self.assertNotEqual(
                stats_cache_path(source_a, cache_dir),
                stats_cache_path(source_b, cache_dir),
            )
            self.assertNotEqual(
                density_cache_path(source_a, 1.0, cache_dir),
                density_cache_path(source_b, 1.0, cache_dir),
            )

    def test_malformed_stats_and_density_json_are_controlled_cache_misses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.xyz"
            cache_dir = root / "cache"
            _write_xyz(source, _points())
            prepare_density(source, 1.0, cache_dir)

            stats_cache_path(source, cache_dir).write_text("{broken", encoding="utf-8")
            rebuilt_stats = prepare_density(source, 1.0, cache_dir)

            self.assertFalse(rebuilt_stats.stats_from_cache)
            self.assertTrue(rebuilt_stats.density_from_cache)

            density_metadata_path(source, 1.0, cache_dir).write_text(
                "{broken",
                encoding="utf-8",
            )
            rebuilt_density = prepare_density(source, 1.0, cache_dir)

            self.assertTrue(rebuilt_density.stats_from_cache)
            self.assertFalse(rebuilt_density.density_from_cache)

    def test_density_array_with_wrong_shape_is_not_used(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.xyz"
            cache_dir = root / "cache"
            _write_xyz(source, _points())
            expected = prepare_density(source, 1.0, cache_dir)

            np.save(
                density_cache_path(source, 1.0, cache_dir),
                np.zeros((1, 1), dtype=np.uint32),
            )
            rebuilt = prepare_density(source, 1.0, cache_dir)

            self.assertTrue(rebuilt.stats_from_cache)
            self.assertFalse(rebuilt.density_from_cache)
            self.assertEqual(rebuilt.grid.density.shape, expected.grid.density.shape)

    def test_density_array_with_wrong_dtype_is_not_used(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.xyz"
            cache_dir = root / "cache"
            _write_xyz(source, _points())
            expected = prepare_density(source, 1.0, cache_dir)

            np.save(
                density_cache_path(source, 1.0, cache_dir),
                np.zeros(expected.grid.density.shape, dtype=np.float32),
            )
            rebuilt = prepare_density(source, 1.0, cache_dir)

            self.assertTrue(rebuilt.stats_from_cache)
            self.assertFalse(rebuilt.density_from_cache)
            self.assertEqual(rebuilt.grid.density.dtype, np.dtype(np.uint32))

    def test_legacy_and_unsupported_cache_versions_are_not_used(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.xyz"
            cache_dir = root / "cache"
            _write_xyz(source, _points())
            prepare_density(source, 1.0, cache_dir)

            stats_path = stats_cache_path(source, cache_dir)
            stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
            stats_payload.pop("cache_version")
            stats_path.write_text(json.dumps(stats_payload), encoding="utf-8")

            metadata_path = density_metadata_path(source, 1.0, cache_dir)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["cache_version"] = CACHE_VERSION + 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            rebuilt = prepare_density(source, 1.0, cache_dir)

            self.assertFalse(rebuilt.stats_from_cache)
            self.assertFalse(rebuilt.density_from_cache)


if __name__ == "__main__":
    unittest.main()
