from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.density_grid import build_density_grid  # noqa: E402
from app.core.progress import ProcessingProgress  # noqa: E402
from app.core.xyz_reader import compute_stats  # noqa: E402
from app.services.coarse_processing import prepare_density  # noqa: E402


def _write_points(path: Path, newline: bytes = b"\n", count: int = 4) -> bytes:
    lines = [
        f"{index} {index * 2} {index * 3}".encode("ascii")
        for index in range(count)
    ]
    payload = newline.join(lines) + newline
    path.write_bytes(payload)
    return payload


class TestProcessingProgress(unittest.TestCase):
    def assert_stage_progress(
        self,
        events: list[ProcessingProgress],
        stage: str,
        total: int,
    ) -> None:
        self.assertGreaterEqual(len(events), 2)
        self.assertTrue(all(event.stage == stage for event in events))
        self.assertEqual(events[0].completed, 0)
        self.assertEqual(events[0].fraction, 0.0)
        self.assertEqual(events[-1].completed, total)
        self.assertEqual(events[-1].total, total)
        self.assertEqual(events[-1].fraction, 1.0)
        self.assertTrue(
            all(
                left.completed <= right.completed
                for left, right in zip(events, events[1:])
            )
        )
        self.assertTrue(
            all(
                left.fraction <= right.fraction
                for left, right in zip(events, events[1:])
            )
        )
        self.assertTrue(all(0.0 <= event.fraction <= 1.0 for event in events))

    def test_statistics_reports_monotonic_byte_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "statistics.xyz"
            payload = _write_points(source)
            events: list[ProcessingProgress] = []

            stats = compute_stats(source, progress_callback=events.append)

        self.assertEqual(stats.point_count, 4)
        self.assert_stage_progress(events, "statistics", len(payload))

    def test_density_reports_monotonic_byte_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "density.xyz"
            payload = _write_points(source)
            stats = compute_stats(source, progress_callback=lambda _event: None)
            events: list[ProcessingProgress] = []

            grid = build_density_grid(
                source,
                stats,
                cell_size=1.0,
                progress_callback=events.append,
            )

        self.assertEqual(int(grid.density.sum()), 4)
        self.assert_stage_progress(events, "density", len(payload))

    def test_lf_and_crlf_sources_finish_at_exact_file_size(self):
        for newline in (b"\n", b"\r\n"):
            with self.subTest(newline=newline), tempfile.TemporaryDirectory() as temp_dir:
                source = Path(temp_dir) / "line_endings.xyz"
                payload = _write_points(source, newline=newline)
                events: list[ProcessingProgress] = []

                compute_stats(source, progress_callback=events.append)

                self.assert_stage_progress(events, "statistics", len(payload))
                self.assertEqual(events[-1].completed, source.stat().st_size)

    def test_progress_callback_is_optional(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "optional.xyz"
            _write_points(source)

            result = prepare_density(source, cell_size=1.0, use_cache=False)

        self.assertEqual(result.stats.point_count, 4)
        self.assertEqual(int(result.grid.density.sum()), 4)

    def test_facade_cache_hit_emits_no_source_read_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "cached.xyz"
            cache_dir = temp_path / "cache"
            _write_points(source)
            first_events: list[ProcessingProgress] = []
            second_events: list[ProcessingProgress] = []

            first = prepare_density(
                source,
                cell_size=1.0,
                cache_dir=cache_dir,
                progress_callback=first_events.append,
            )
            second = prepare_density(
                source,
                cell_size=1.0,
                cache_dir=cache_dir,
                progress_callback=second_events.append,
            )

        self.assertFalse(first.stats_from_cache)
        self.assertFalse(first.density_from_cache)
        self.assertEqual(first_events[0].stage, "statistics")
        self.assertEqual(first_events[-1].stage, "density")
        self.assertTrue(second.stats_from_cache)
        self.assertTrue(second.density_from_cache)
        self.assertEqual(second_events, [])

    def test_callback_is_throttled_for_many_points(self):
        point_count = 50_000
        line = b"123456789.0 123456789.0 0.0\r\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "many_points.xyz"
            source.write_bytes(line * point_count)
            events: list[ProcessingProgress] = []

            stats = compute_stats(source, progress_callback=events.append)

        self.assertEqual(stats.point_count, point_count)
        self.assertLess(len(events), point_count // 100)
        self.assertEqual(events[-1].fraction, 1.0)

    def test_progress_event_is_immutable(self):
        event = ProcessingProgress(
            stage="statistics",
            completed=0,
            total=10,
            fraction=0.0,
        )

        with self.assertRaises(FrozenInstanceError):
            event.fraction = 0.5  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
