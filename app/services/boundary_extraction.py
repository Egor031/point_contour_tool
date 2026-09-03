from __future__ import annotations

import math
import operator
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.cache import get_file_signature
from app.core.working_boundary_cloud import (
    AxisAlignedBoundingBox,
    OuterBoundaryCloud,
    point_is_near_segment,
    validate_search_width,
)
from app.core.xyz_reader import iter_xyz_points


_XYZ_ACCUMULATOR_BLOCK_SIZE = 16_384


class SourceChangedDuringExtractionError(RuntimeError):
    """Raised when the source identity changes during a streaming extraction."""


@dataclass(frozen=True, slots=True)
class SourceFileSignature:
    canonical_path: str
    file_size: int
    mtime_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_path, str):
            raise TypeError("canonical_path must be a string")
        if not self.canonical_path:
            raise ValueError("canonical_path must not be empty")
        for name in ("file_size", "mtime_ns"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be an integer")
            try:
                normalized = operator.index(value)
            except TypeError as exc:
                raise TypeError(f"{name} must be an integer") from exc
            if normalized < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, normalized)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, str | int],
    ) -> SourceFileSignature:
        try:
            canonical_path = value["canonical_path"]
            file_size = value["file_size"]
            mtime_ns = value["mtime_ns"]
        except KeyError as exc:
            raise ValueError("source signature is missing a required field") from exc
        return cls(
            canonical_path=canonical_path,  # type: ignore[arg-type]
            file_size=file_size,  # type: ignore[arg-type]
            mtime_ns=mtime_ns,  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class OuterBoundaryExtractionStatistics:
    source_points_seen: int
    broad_phase_candidate_checks: int
    exact_segment_checks: int
    accepted_points: int
    segment_count: int
    tile_size: float
    occupied_buckets: int
    segment_bucket_entries: int

    @property
    def brute_force_segment_checks(self) -> int:
        return self.source_points_seen * self.segment_count

    @property
    def exact_segment_checks_avoided(self) -> int:
        return self.brute_force_segment_checks - self.exact_segment_checks


@dataclass(frozen=True, slots=True)
class OuterBoundaryExtractionResult:
    cloud: OuterBoundaryCloud
    source_signature: SourceFileSignature
    statistics: OuterBoundaryExtractionStatistics


class OuterBoundarySegmentIndex:
    """Uniform world-space buckets containing conservative segment candidates."""

    def __init__(
        self,
        preliminary_contour_world: np.ndarray,
        search_width: object,
        *,
        tile_size: object | None = None,
    ) -> None:
        empty_cloud = OuterBoundaryCloud(
            preliminary_contour_world=preliminary_contour_world,
            search_width=search_width,
        )
        self.contour_world = empty_cloud.preliminary_contour_world
        self.search_width = empty_cloud.search_width
        self.segment_aabbs = empty_cloud.segment_search_aabbs()
        self.tile_size = (
            _choose_tile_size(self.contour_world, self.search_width)
            if tile_size is None
            else validate_search_width(tile_size, "tile_size")
        )

        mutable_buckets: dict[tuple[int, int], list[int]] = {}
        for segment_id, aabb in enumerate(self.segment_aabbs):
            for bucket_key in self._bucket_keys_for_aabb(aabb):
                mutable_buckets.setdefault(bucket_key, []).append(segment_id)
        self._buckets = {
            bucket_key: tuple(segment_ids)
            for bucket_key, segment_ids in mutable_buckets.items()
        }
        self.segment_bucket_entries = sum(map(len, self._buckets.values()))

    @property
    def segment_count(self) -> int:
        return len(self.contour_world)

    @property
    def occupied_bucket_count(self) -> int:
        return len(self._buckets)

    def candidate_segment_ids(self, point_xy: tuple[float, float]) -> tuple[int, ...]:
        return self._buckets.get(self._bucket_key(point_xy), ())

    def _bucket_key(self, point_xy: tuple[float, float]) -> tuple[int, int]:
        x, y = point_xy
        return math.floor(x / self.tile_size), math.floor(y / self.tile_size)

    def _bucket_keys_for_aabb(
        self,
        aabb: AxisAlignedBoundingBox,
    ) -> Iterator[tuple[int, int]]:
        min_column, min_row = self._bucket_key((aabb.min_x, aabb.min_y))
        max_column, max_row = self._bucket_key((aabb.max_x, aabb.max_y))
        for column in range(min_column, max_column + 1):
            for row in range(min_row, max_row + 1):
                yield column, row


class _XYZBlockAccumulator:
    def __init__(self, block_size: int = _XYZ_ACCUMULATOR_BLOCK_SIZE) -> None:
        self._block_size = block_size
        self._blocks: list[np.ndarray] = []
        self._current = np.empty((block_size, 3), dtype=np.float64)
        self._current_size = 0
        self._size = 0

    def append(self, x: float, y: float, z: float) -> None:
        if self._current_size == self._block_size:
            self._blocks.append(self._current)
            self._current = np.empty((self._block_size, 3), dtype=np.float64)
            self._current_size = 0
        self._current[self._current_size] = (x, y, z)
        self._current_size += 1
        self._size += 1

    def to_array(self) -> np.ndarray:
        if self._size == 0:
            return np.empty((0, 3), dtype=np.float64)
        arrays = [*self._blocks, self._current[: self._current_size]]
        if len(arrays) == 1:
            return np.array(arrays[0], dtype=np.float64, copy=True, order="C")
        return np.concatenate(arrays, axis=0)


def _choose_tile_size(contour_world: np.ndarray, search_width: float) -> float:
    following = np.roll(contour_world, -1, axis=0)
    lengths = np.hypot(
        following[:, 0] - contour_world[:, 0],
        following[:, 1] - contour_world[:, 1],
    )
    non_zero_lengths = lengths[lengths > 0.0]
    median_segment_length = float(np.median(non_zero_lengths))
    full_search_band_width = search_width * 2.0
    if not math.isfinite(full_search_band_width):
        full_search_band_width = search_width
    return max(full_search_band_width, median_segment_length)


def _read_source_signature(source_path: str | Path) -> SourceFileSignature:
    return SourceFileSignature.from_mapping(get_file_signature(source_path))


def _validate_source_point(
    point: tuple[float, float, float],
    point_number: int,
) -> tuple[float, float, float]:
    try:
        x, y, z = point
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"source point {point_number} must contain exactly three coordinates"
        ) from exc
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError(f"source point {point_number} coordinates must be finite")
    return x, y, z


def _point_matches_candidate_segments(
    point_xy: tuple[float, float],
    index: OuterBoundarySegmentIndex,
    candidate_segment_ids: tuple[int, ...],
) -> tuple[bool, int]:
    exact_checks = 0
    contour = index.contour_world
    for segment_id in candidate_segment_ids:
        exact_checks += 1
        if point_is_near_segment(
            point_xy,
            contour[segment_id],
            contour[(segment_id + 1) % len(contour)],
            index.search_width,
        ):
            return True, exact_checks
    return False, exact_checks


def extract_outer_boundary_points(
    source_path: str | Path,
    preliminary_contour_world: np.ndarray,
    search_width: object,
    *,
    tile_size: object | None = None,
) -> OuterBoundaryExtractionResult:
    """Extract original XYZ near a preliminary closed outer contour in one pass."""
    source_path = Path(source_path)
    source_signature_before = _read_source_signature(source_path)
    index = OuterBoundarySegmentIndex(
        preliminary_contour_world,
        search_width,
        tile_size=tile_size,
    )
    points = _XYZBlockAccumulator()
    source_points_seen = 0
    broad_phase_candidate_checks = 0
    exact_segment_checks = 0

    for source_points_seen, source_point in enumerate(
        iter_xyz_points(source_path),
        start=1,
    ):
        x, y, z = _validate_source_point(source_point, source_points_seen)
        candidate_segment_ids = index.candidate_segment_ids((x, y))
        broad_phase_candidate_checks += len(candidate_segment_ids)
        matches, checks = _point_matches_candidate_segments(
            (x, y),
            index,
            candidate_segment_ids,
        )
        exact_segment_checks += checks
        if matches:
            points.append(x, y, z)

    source_signature_after = _read_source_signature(source_path)
    if source_signature_after != source_signature_before:
        raise SourceChangedDuringExtractionError(
            "source file changed during outer boundary extraction"
        )

    points_xyz = points.to_array()
    cloud = OuterBoundaryCloud(
        preliminary_contour_world=index.contour_world,
        search_width=index.search_width,
        points_xyz=points_xyz,
    )
    statistics = OuterBoundaryExtractionStatistics(
        source_points_seen=source_points_seen,
        broad_phase_candidate_checks=broad_phase_candidate_checks,
        exact_segment_checks=exact_segment_checks,
        accepted_points=len(points_xyz),
        segment_count=index.segment_count,
        tile_size=index.tile_size,
        occupied_buckets=index.occupied_bucket_count,
        segment_bucket_entries=index.segment_bucket_entries,
    )
    return OuterBoundaryExtractionResult(
        cloud=cloud,
        source_signature=source_signature_before,
        statistics=statistics,
    )
