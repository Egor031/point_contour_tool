from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path


_APP_DIRECTORY = Path(__file__).resolve().parents[1]
_PACKAGE_PARENT = _APP_DIRECTORY.parent
if str(_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_PARENT))


from app.core.progress import ProcessingProgress  # noqa: E402
from app.services.coarse_processing import prepare_density  # noqa: E402
from app.ui.density_worker import (  # noqa: E402
    DensityWorker,
    DensityWorkerError,
    DensityWorkerProgress,
    DensityWorkerResult,
    format_byte_count,
    format_progress_bytes,
    progress_stage_label,
)


def _write_xyz(path: Path) -> None:
    path.write_text("0 0 0\n1 1 1\n2 2 2\n", encoding="utf-8")


class TestDensityWorker(unittest.TestCase):
    def _wait_for_messages(self, worker: DensityWorker):
        self.assertTrue(worker.wait(timeout=5.0))
        messages = worker.drain_messages()
        self.assertTrue(messages)
        return messages

    def test_worker_success_emits_progress_and_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "source.xyz"
            cache_dir = temp_path / "cache"
            _write_xyz(source)

            def prepare(source_path, *, cell_size, progress_callback):
                return prepare_density(
                    source_path,
                    cell_size=cell_size,
                    cache_dir=cache_dir,
                    use_cache=False,
                    progress_callback=progress_callback,
                )

            worker = DensityWorker(prepare_density_func=prepare)
            self.assertTrue(worker.start(source, 1.0))
            messages = self._wait_for_messages(worker)

        progress_messages = [
            message for message in messages if isinstance(message, DensityWorkerProgress)
        ]
        result_messages = [
            message for message in messages if isinstance(message, DensityWorkerResult)
        ]
        self.assertTrue(progress_messages)
        self.assertEqual(len(result_messages), 1)
        self.assertEqual(result_messages[0].result.stats.point_count, 3)
        self.assertEqual(int(result_messages[0].result.grid.density.sum()), 3)
        self.assertEqual(result_messages[0].preview.ndim, 2)
        self.assertEqual(result_messages[0].texture_rgba.shape[-1], 4)

    def test_progress_event_is_forwarded_without_modification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.xyz"
            _write_xyz(source)
            prepared = prepare_density(
                source,
                cell_size=1.0,
                use_cache=False,
                progress_callback=lambda _event: None,
            )
            expected_progress = ProcessingProgress(
                stage="density",
                completed=5,
                total=10,
                fraction=0.5,
            )

            def prepare(_source_path, *, cell_size, progress_callback):
                self.assertEqual(cell_size, 1.0)
                progress_callback(expected_progress)
                return prepared

            worker = DensityWorker(prepare_density_func=prepare)
            self.assertTrue(worker.start(source, 1.0))
            messages = self._wait_for_messages(worker)

        forwarded = next(
            message for message in messages if isinstance(message, DensityWorkerProgress)
        )
        self.assertIs(forwarded.progress, expected_progress)

    def test_worker_error_is_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            missing_source = temp_path / "missing.xyz"
            cache_dir = temp_path / "cache"

            def prepare(source_path, *, cell_size, progress_callback):
                return prepare_density(
                    source_path,
                    cell_size=cell_size,
                    cache_dir=cache_dir,
                    progress_callback=progress_callback,
                )

            worker = DensityWorker(prepare_density_func=prepare)
            self.assertTrue(worker.start(missing_source, 1.0))
            messages = self._wait_for_messages(worker)

        errors = [message for message in messages if isinstance(message, DensityWorkerError)]
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0].error, FileNotFoundError)

    def test_worker_rejects_second_active_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.xyz"
            _write_xyz(source)
            prepared = prepare_density(
                source,
                cell_size=1.0,
                use_cache=False,
                progress_callback=lambda _event: None,
            )
            entered = threading.Event()
            release = threading.Event()

            def prepare(_source_path, *, cell_size, progress_callback):
                entered.set()
                release.wait(timeout=5.0)
                return prepared

            worker = DensityWorker(prepare_density_func=prepare)
            self.assertTrue(worker.start(source, 1.0))
            self.assertTrue(entered.wait(timeout=2.0))
            self.assertFalse(worker.start(source, 0.5))
            release.set()
            self._wait_for_messages(worker)
            self.assertTrue(worker.is_active)
            worker.mark_finished()
            self.assertFalse(worker.is_active)

    def test_cache_hit_finishes_without_fake_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "source.xyz"
            cache_dir = temp_path / "cache"
            _write_xyz(source)

            def prepare(source_path, *, cell_size, progress_callback):
                return prepare_density(
                    source_path,
                    cell_size=cell_size,
                    cache_dir=cache_dir,
                    progress_callback=progress_callback,
                )

            worker = DensityWorker(prepare_density_func=prepare)
            self.assertTrue(worker.start(source, 1.0))
            self._wait_for_messages(worker)
            worker.mark_finished()

            self.assertTrue(worker.start(source, 1.0))
            messages = self._wait_for_messages(worker)

        self.assertFalse(
            any(isinstance(message, DensityWorkerProgress) for message in messages)
        )
        result = next(
            message for message in messages if isinstance(message, DensityWorkerResult)
        ).result
        self.assertTrue(result.stats_from_cache)
        self.assertTrue(result.density_from_cache)

    def test_progress_formatting_is_pure_and_human_readable(self):
        progress = ProcessingProgress(
            stage="statistics",
            completed=1024 * 1024,
            total=2 * 1024 * 1024,
            fraction=0.5,
        )

        self.assertEqual(format_byte_count(0), "0 B")
        self.assertEqual(format_byte_count(1024), "1.0 KB")
        self.assertEqual(format_progress_bytes(progress), "1.0 MB / 2.0 MB")
        self.assertEqual(
            progress_stage_label("statistics"),
            "Reading point cloud / calculating statistics",
        )
        self.assertEqual(progress_stage_label("density"), "Building density map")


if __name__ == "__main__":
    unittest.main()
