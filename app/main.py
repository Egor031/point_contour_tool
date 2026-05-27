from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.core.cache import (
    load_density_cache,
    load_stats_cache,
    save_density_cache,
    save_stats_cache,
)
from app.core.density_grid import build_density_grid
from app.core.preview_export import (
    save_density_preview,
    save_report,
    save_smoothed_density_preview,
    save_mask_preview,
    save_contour_preview,
)
from app.core.mask_processing import (
    build_mask_from_density,
    fill_small_holes,
    keep_largest_component,
    remove_small_components,
)
from app.core.contour_extractor import (
    build_external_contour,
    save_contour_csv,
)

from app.core.xyz_reader import compute_stats

from app.exporters.dxf_exporter import save_contour_dxf

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlatScanContour: build density preview from .asc/.xyz point cloud"
    )

    parser.add_argument(
        "input_file",
        help="Path to .asc/.xyz file",
    )

    parser.add_argument(
        "--cell",
        type=float,
        default=0.5,
        help="Cell size in source units, probably millimeters. Default: 0.5",
    )

    parser.add_argument(
        "--out",
        default="data/output",
        help="Output directory. Default: data/output",
    )

    parser.add_argument(
        "--cache",
        default="cache",
        help="Cache directory. Default: cache",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache usage",
    )

    parser.add_argument(
        "--preview-size",
        type=int,
        default=3000,
        help="Max preview image size in pixels. Default: 3000",
    )

    parser.add_argument(
    "--smooth-mm",
    type=float,
    default=0.0,
    help="Smoothing sigma in source units, probably millimeters. Default: 2.0",
    )

    parser.add_argument(
    "--threshold",
    default="auto",
    help="Density threshold. Use 'auto' or number. Default: auto",
    )

    parser.add_argument(
        "--min-component-area",
        type=int,
        default=0,
        help="Remove mask components smaller than this area in cells. Default: 0",
    )

    parser.add_argument(
        "--keep-largest",
        action="store_true",
        help="Keep only largest connected component in mask",
    )
    parser.add_argument(
        "--fill-holes-area",
        type=int,
        default=0,
        help="Fill internal holes smaller than this area in cells. Default: 0",
    )

    parser.add_argument(
        "--contour",
        action="store_true",
        help="Extract external contour from final mask",
    )

    parser.add_argument(
        "--dxf",
        action="store_true",
        help="Export extracted contour to DXF. Requires --contour.",
    )

    parser.add_argument(
        "--simplify-mm",
        type=float,
        default=0.0,
        help="Simplify contour tolerance in source units, probably millimeters. Default: 0.0",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.out)
    cache_dir = Path(args.cache)

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    base_name = input_path.stem

    print(f"Файл: {input_path}")
    print(f"Размер ячейки: {args.cell}")

    t0 = time.perf_counter()

    print("\n[1/3] Считаем или загружаем статистику файла...")

    stats = None
    if not args.no_cache:
        stats = load_stats_cache(input_path, cache_dir)

    if stats is not None:
        print("Статистика загружена из кэша.")
    else:
        stats = compute_stats(input_path)
        if not args.no_cache:
            save_stats_cache(stats, cache_dir)
            print("Статистика сохранена в кэш.")

    t1 = time.perf_counter()

    print(f"Точек: {stats.point_count:,}")
    print(f"X: {stats.min_x:.3f} .. {stats.max_x:.3f}")
    print(f"Y: {stats.min_y:.3f} .. {stats.max_y:.3f}")
    print(f"Z: {stats.min_z:.3f} .. {stats.max_z:.3f}")
    print(f"Время статистики: {t1 - t0:.2f} сек")

    print("\n[2/3] Строим или загружаем карту плотности...")

    grid = None
    if not args.no_cache:
        grid = load_density_cache(input_path, stats, args.cell, cache_dir)

    if grid is not None:
        print("Density grid загружен из кэша.")
    else:
        grid = build_density_grid(input_path, stats, args.cell)
        if not args.no_cache:
            save_density_cache(grid, input_path, cache_dir)
            print("Density grid сохранён в кэш.")

    t2 = time.perf_counter()
    print(f"Время подготовки density grid: {t2 - t1:.2f} сек")

    print("\n[3/3] Сохраняем preview, mask и report...")

    cell_text = str(args.cell).replace(".", "_")
    smooth_text = str(args.smooth_mm).replace(".", "_")

    preview_path = output_dir / f"{base_name}_density_cell_{cell_text}.png"
    smooth_preview_path = (
        output_dir / f"{base_name}_density_cell_{cell_text}_smooth_{smooth_text}mm.png"
    )
    mask_path = output_dir / f"{base_name}_mask_cell_{cell_text}_threshold_{args.threshold}.png"
    report_path = output_dir / f"{base_name}_report_cell_{cell_text}.txt"

    contour_preview_path = (
        output_dir / f"{base_name}_contour_cell_{cell_text}_threshold_{args.threshold}.png"
    )
    contour_csv_path = (
        output_dir / f"{base_name}_contour_cell_{cell_text}_threshold_{args.threshold}.csv"
    )

    contour_dxf_path = (
        output_dir / f"{base_name}_contour_cell_{cell_text}_threshold_{args.threshold}.dxf"
    )

    save_density_preview(grid, preview_path, max_size=args.preview_size)

    if args.smooth_mm > 0:
        save_smoothed_density_preview(
            grid,
            smooth_preview_path,
            sigma_mm=args.smooth_mm,
            max_size=args.preview_size,
        )

    if args.threshold == "auto":
        mask_result = build_mask_from_density(
            grid.density,
            mode="auto",
        )
    else:
        manual_threshold = float(args.threshold)
        mask_result = build_mask_from_density(
            grid.density,
            mode="manual",
            manual_threshold=manual_threshold,
        )

    mask = mask_result.mask

    if args.min_component_area > 0:
        mask = remove_small_components(mask, args.min_component_area)

    if args.keep_largest:
        mask = keep_largest_component(mask)
    
    if args.fill_holes_area > 0:
        mask = fill_small_holes(mask, args.fill_holes_area)

    save_mask_preview(mask, mask_path, max_size=args.preview_size)
    save_report(stats, grid, report_path)

    contour_result = None

    if args.contour:
        contour_result = build_external_contour(
            mask=mask,
            grid=grid,
            simplify_mm=args.simplify_mm,
        )

        save_contour_preview(
            mask=mask,
            contour_pixels=contour_result.contour_pixels,
            output_path=contour_preview_path,
            max_size=args.preview_size,
        )

        save_contour_csv(
            contour_world=contour_result.contour_world,
            output_path=contour_csv_path,
        )

        if args.dxf:
            save_contour_dxf(
                contour_world=contour_result.contour_world,
                output_path=contour_dxf_path,
                close=True,
            )

    t3 = time.perf_counter()

    print(f"Preview:        {preview_path}")
    if args.smooth_mm > 0:
        print(f"Smooth preview: {smooth_preview_path}")
    print(f"Mask preview:   {mask_path}")
    print(f"Report:         {report_path}")
    print(f"Mask threshold: {mask_result.threshold:.3f}")
    print(f"Mask cells:     {int(mask.sum()):,}")
    if contour_result is not None:
        print(f"Contour preview: {contour_preview_path}")
        print(f"Contour CSV:     {contour_csv_path}")

        if args.dxf:
            print(f"Contour DXF:     {contour_dxf_path}")

        print(f"Contour points:  {contour_result.point_count:,}")
    print(f"Время сохранения: {t3 - t2:.2f} сек")

    print(f"\nГотово. Общее время: {t3 - t0:.2f} сек")

    
if __name__ == "__main__":
    main()