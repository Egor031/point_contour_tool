from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.cache import (
    load_density_cache,
    load_stats_cache,
    save_density_cache,
    save_stats_cache,
)
from app.core.contour_extractor import ContourResult, build_external_contour
from app.core.density_grid import DensityGrid, build_density_grid
from app.core.hole_detector import (
    HoleCandidate,
    cluster_holes_by_diameter,
    detect_circular_holes,
)
from app.core.mask_edits import MaskEditsStats, apply_mask_edits
from app.core.mask_processing import (
    MaskResult,
    ThresholdMode,
    apply_polygon_roi_to_mask,
    apply_roi_to_mask,
    build_mask_from_density,
    fill_small_holes,
    keep_largest_component,
    remove_small_components,
)
from app.core.xyz_reader import PointCloudStats, compute_stats


@dataclass
class StatisticsProcessingResult:
    stats: PointCloudStats
    from_cache: bool


@dataclass
class DensityProcessingResult:
    source_path: Path
    stats: PointCloudStats
    grid: DensityGrid
    stats_from_cache: bool
    density_from_cache: bool


@dataclass
class MaskProcessingResult:
    grid: DensityGrid
    threshold_result: MaskResult
    mask_for_holes: np.ndarray
    contour_mask: np.ndarray
    mask_before_edits: np.ndarray | None
    mask_edits_stats: MaskEditsStats | None


@dataclass
class HoleDetectionResult:
    candidates: list[HoleCandidate]
    groups: list[dict]

    @property
    def accepted_count(self) -> int:
        return sum(1 for candidate in self.candidates if candidate.accepted)


def prepare_statistics(
    source_path: str | Path,
    cache_dir: str | Path = "cache",
    use_cache: bool = True,
) -> StatisticsProcessingResult:
    source_path = Path(source_path)

    stats = load_stats_cache(source_path, cache_dir) if use_cache else None
    from_cache = stats is not None

    if stats is None:
        stats = compute_stats(source_path)
        if use_cache:
            save_stats_cache(stats, cache_dir)

    return StatisticsProcessingResult(stats=stats, from_cache=from_cache)


def prepare_density(
    source_path: str | Path,
    cell_size: float,
    cache_dir: str | Path = "cache",
    use_cache: bool = True,
    statistics: StatisticsProcessingResult | None = None,
) -> DensityProcessingResult:
    source_path = Path(source_path)
    statistics_result = statistics or prepare_statistics(
        source_path=source_path,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    stats = statistics_result.stats

    grid = (
        load_density_cache(source_path, stats, cell_size, cache_dir)
        if use_cache
        else None
    )
    density_from_cache = grid is not None

    if grid is None:
        grid = build_density_grid(source_path, stats, cell_size)
        if use_cache:
            save_density_cache(grid, source_path, cache_dir)

    return DensityProcessingResult(
        source_path=source_path,
        stats=stats,
        grid=grid,
        stats_from_cache=statistics_result.from_cache,
        density_from_cache=density_from_cache,
    )


def build_processing_masks(
    grid: DensityGrid,
    *,
    threshold_mode: ThresholdMode = "auto",
    manual_threshold: float | None = None,
    min_component_area: int = 0,
    keep_largest: bool = False,
    roi: tuple[float, float, float, float] | None = None,
    polygon_roi: list[tuple[float, float]] | None = None,
    mask_edits: list[dict[str, Any]] | None = None,
    fill_holes_area: int = 0,
) -> MaskProcessingResult:
    threshold_result = build_mask_from_density(
        grid.density,
        mode=threshold_mode,
        manual_threshold=manual_threshold,
    )
    mask = threshold_result.mask

    if min_component_area > 0:
        mask = remove_small_components(mask, min_component_area)

    if keep_largest:
        mask = keep_largest_component(mask)

    if roi is not None:
        mask = apply_roi_to_mask(mask, grid, roi)

    if polygon_roi is not None:
        mask = apply_polygon_roi_to_mask(mask, grid, polygon_roi)

    mask_before_edits = None
    mask_edits_stats = None
    if mask_edits is not None:
        mask_before_edits = mask.copy()
        mask, mask_edits_stats = apply_mask_edits(mask, grid, mask_edits)

    mask_for_holes = mask.copy()
    contour_mask = mask_for_holes.copy()

    if fill_holes_area > 0:
        contour_mask = fill_small_holes(contour_mask, fill_holes_area)
        # Preserve the two fill passes performed by the existing CLI pipeline.
        contour_mask = fill_small_holes(contour_mask, fill_holes_area)

    return MaskProcessingResult(
        grid=grid,
        threshold_result=threshold_result,
        mask_for_holes=mask_for_holes,
        contour_mask=contour_mask,
        mask_before_edits=mask_before_edits,
        mask_edits_stats=mask_edits_stats,
    )


def extract_preliminary_contour(
    masks: MaskProcessingResult,
    simplify_mm: float = 0.0,
) -> ContourResult:
    return build_external_contour(
        mask=masks.contour_mask,
        grid=masks.grid,
        simplify_mm=simplify_mm,
    )


def find_hole_candidates(
    masks: MaskProcessingResult,
    *,
    min_diameter_mm: float = 8.0,
    max_diameter_mm: float | None = None,
    min_circularity: float = 0.55,
    max_aspect_ratio_deviation: float = 0.35,
    max_error_ratio: float = 0.18,
    group_tolerance_mm: float = 1.5,
) -> HoleDetectionResult:
    candidates = detect_circular_holes(
        mask=masks.mask_for_holes,
        grid=masks.grid,
        min_diameter_mm=min_diameter_mm,
        max_diameter_mm=max_diameter_mm,
        min_circularity=min_circularity,
        max_aspect_ratio_deviation=max_aspect_ratio_deviation,
        max_error_ratio=max_error_ratio,
    )
    groups = cluster_holes_by_diameter(
        holes=candidates,
        tolerance_mm=group_tolerance_mm,
    )
    return HoleDetectionResult(candidates=candidates, groups=groups)
