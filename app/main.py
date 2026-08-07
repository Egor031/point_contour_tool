from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from app.core.preview_export import (
    save_density_preview,
    save_report,
    save_smoothed_density_preview,
    save_mask_preview,
    save_contour_preview,
    save_holes_preview,
)
from app.core.mask_edits import (
    load_mask_edits,
    save_mask_edits_debug_preview,
)
from app.core.contour_extractor import (
    save_contour_csv,
)

from app.core.hole_detector import (
    save_holes_csv,
    save_holes_json,
)
from app.core.line_approximation import run_line_approximation

from app.core.xyz_reader import export_decimated_points

from app.exporters.boundary_xyz_exporter import (
    build_boundary_band_mask,
    export_boundary_points,
)
from app.exporters.clean_xyz_exporter import export_clean_points
from app.exporters.dxf_exporter import load_holes_json_circles, save_contour_dxf
from app.services.coarse_processing import (
    build_processing_masks,
    extract_preliminary_contour,
    find_hole_candidates,
    prepare_density,
    prepare_statistics,
)


def parse_roi_poly(value: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []

    for raw_point in value.split(";"):
        raw_point = raw_point.strip()
        if not raw_point:
            continue

        parts = [part.strip() for part in raw_point.split(",")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "ROI polygon points must use format x1,y1;x2,y2;x3,y3;..."
            )

        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "ROI polygon coordinates must be numbers"
            ) from exc

        points.append((x, y))

    if len(points) < 3:
        raise argparse.ArgumentTypeError("ROI polygon must contain at least 3 points")

    return points


def write_demo_summary(output_path: str | Path, lines: list[str]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_value(value) -> str:
    if value is None:
        return "not available"
    return str(value)


def _path_if(condition: bool, path: Path) -> str:
    return str(path) if condition else "not available"


def _file_size_text(path: Path | None) -> str:
    if path is None:
        return "not available"
    try:
        return str(path.stat().st_size)
    except OSError:
        return "not available"


def _demo_summary_notes() -> list[str]:
    return [
        "Notes:",
        "- Contour is generated from density/mask representation.",
        "- Mixed contour is experimental.",
        "- Holes are currently detected from density/mask; point-based hole refinement is not implemented yet.",
        "- Polyline gaps preserve complex, curved or short geometry.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlatScanContour: build density preview from .asc/.xyz point cloud"
    )

    parser.add_argument(
        "input_file",
        nargs="?",
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
        "--roi",
        type=float,
        nargs=4,
        metavar=("min_x", "min_y", "max_x", "max_y"),
        help="Limit mask processing to ROI in world coordinates: min_x min_y max_x max_y",
    )
    parser.add_argument(
        "--roi-poly",
        type=parse_roi_poly,
        help="Limit mask processing to polygon ROI: x1,y1;x2,y2;x3,y3;...",
    )
    parser.add_argument(
        "--mask-edits",
        help="Path to mask edits JSON from preview viewer",
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
        "--holes-json",
        help="Path to edited holes JSON for optional DXF circle export",
    )

    parser.add_argument(
        "--export-clean",
        action="store_true",
        help="Export cleaned point cloud using final mask",
    )

    parser.add_argument(
        "--export-boundary",
        action="store_true",
        help="Export point cloud near final mask boundary",
    )

    parser.add_argument(
        "--export-decimated",
        action="store_true",
        help="Export streaming grid-decimated point cloud for CAD preview",
    )

    parser.add_argument(
        "--decimate-cell-mm",
        type=float,
        default=2.0,
        help="Decimation grid cell size. Default: 2.0",
    )

    parser.add_argument(
        "--demo-summary",
        action="store_true",
        help="Write a demonstration summary report at the end of processing",
    )

    parser.add_argument(
        "--boundary-width-mm",
        type=float,
        default=5.0,
        help="Boundary band width in source units. Default: 5.0",
    )

    parser.add_argument(
        "--simplify-mm",
        type=float,
        default=0.0,
        help="Simplify contour tolerance in source units, probably millimeters. Default: 0.0",
    )

    parser.add_argument(
        "--holes",
        action="store_true",
        help="Detect circular holes inside final mask",
    )

    parser.add_argument(
        "--min-hole-diameter-mm",
        type=float,
        default=8.0,
        help="Minimum hole diameter to keep, in source units. Default: 8.0",
    )

    parser.add_argument(
        "--max-hole-diameter-mm",
        type=float,
        default=0.0,
        help="Maximum hole diameter to keep. 0 means no limit. Default: 0",
    )

    parser.add_argument(
        "--min-circularity",
        type=float,
        default=0.55,
        help="Minimum circularity for hole candidate. Default: 0.55",
    )

    parser.add_argument(
        "--max-circle-error-ratio",
        type=float,
        default=0.18,
        help="Maximum mean fitting error divided by radius. Default: 0.18",
    )

    parser.add_argument(
        "--hole-group-tolerance-mm",
        type=float,
        default=1.5,
        help="Maximum diameter difference within a hole group. Default: 1.5",
    )

    parser.add_argument(
        "--approx-lines",
        action="store_true",
        help="Experimental: approximate straight contour segments from boundary points",
    )
    parser.add_argument(
        "--approx-contour-csv",
        help="Contour CSV with x,y columns for --approx-lines",
    )
    parser.add_argument(
        "--approx-boundary-points",
        help="Boundary point cloud file with x y z points for --approx-lines",
    )
    parser.add_argument(
        "--line-simplify-mm",
        type=float,
        default=5.0,
        help="Douglas-Peucker tolerance for line approximation. Default: 5.0",
    )
    parser.add_argument(
        "--line-fit-band-mm",
        type=float,
        default=3.0,
        help="Boundary point band around simplified contour segment. Default: 3.0",
    )
    parser.add_argument(
        "--min-line-points",
        type=int,
        default=20,
        help="Minimum boundary points for a fitted line. Default: 20",
    )
    parser.add_argument(
        "--min-line-length-mm",
        type=float,
        default=20.0,
        help="Minimum accepted line length. Default: 20.0",
    )
    parser.add_argument(
        "--max-line-mean-error-mm",
        type=float,
        default=1.0,
        help="Maximum mean line fitting error. Default: 1.0",
    )
    parser.add_argument(
        "--max-line-error-mm",
        type=float,
        default=3.0,
        help="Maximum line fitting error. Default: 3.0",
    )
    parser.add_argument(
        "--line-edge-percentile",
        type=float,
        default=20.0,
        help="Outer edge percentile used for line fitting. Default: 20.0",
    )
    parser.add_argument(
        "--line-trim-outlier-percent",
        type=float,
        default=10.0,
        help="Largest-error points percent trimmed before final line fit. Default: 10.0",
    )
    parser.add_argument(
        "--max-line-angle-diff-deg",
        type=float,
        default=12.0,
        help="Maximum angle difference between segment and fitted line. Default: 12.0",
    )
    parser.add_argument(
        "--line-end-trim-percent",
        type=float,
        default=10.0,
        help="Percent trimmed from each segment end before line fitting. Default: 10.0",
    )
    parser.add_argument(
        "--line-endpoint-mode",
        choices=("contour_projection", "fit_points_range"),
        default="contour_projection",
        help=(
            "How refined line endpoints are computed. "
            "Default: contour_projection"
        ),
    )
    parser.add_argument(
        "--line-window-mm",
        type=float,
        default=30.0,
        help="Sliding window length for straight core detection. Default: 30.0",
    )
    parser.add_argument(
        "--line-window-step-mm",
        type=float,
        default=10.0,
        help="Sliding window step for straight core detection. Default: 10.0",
    )
    parser.add_argument(
        "--max-window-mean-error-mm",
        type=float,
        default=1.0,
        help="Maximum mean fit error for a good line core window. Default: 1.0",
    )
    parser.add_argument(
        "--max-window-angle-diff-deg",
        type=float,
        default=5.0,
        help="Maximum angle difference between neighboring good windows. Default: 5.0",
    )
    parser.add_argument(
        "--max-point-contour-distance-mm",
        type=float,
        default=6.0,
        help="Maximum boundary point distance to source contour segment. Default: 6.0",
    )
    parser.add_argument(
        "--max-line-contour-distance-mm",
        type=float,
        default=5.0,
        help="Maximum fitted line distance to source mask contour. Default: 5.0",
    )
    parser.add_argument(
        "--debug-line-id",
        type=int,
        help="Save debug CSV/JSON files for one experimental line segment id",
    )
    parser.add_argument(
        "--show-line-labels",
        action="store_true",
        help="Add accepted line id labels to experimental refined_lines.dxf",
    )
    parser.add_argument(
        "--line-label-height-mm",
        type=float,
        default=10.0,
        help="Text height for experimental refined line labels. Default: 10.0",
    )
    parser.add_argument(
        "--chain-lines",
        action="store_true",
        help="Build experimental chained contour from accepted refined lines",
    )
    parser.add_argument(
        "--max-chain-intersection-distance-mm",
        type=float,
        default=150.0,
        help="Maximum accepted distance from chained intersection to source segments. Default: 150.0",
    )
    parser.add_argument(
        "--parallel-angle-eps-deg",
        type=float,
        default=3.0,
        help="Angle threshold for treating neighboring refined lines as parallel. Default: 3.0",
    )
    parser.add_argument(
        "--mixed-contour",
        action="store_true",
        help="Build experimental mixed contour from accepted lines and source contour gaps",
    )
    parser.add_argument(
        "--mixed-dxf",
        action="store_true",
        help="Export experimental mixed contour and holes into one DXF. Requires --mixed-contour.",
    )

    args = parser.parse_args()

    if args.mixed_dxf and not args.mixed_contour:
        raise ValueError("--mixed-dxf requires --mixed-contour")

    if args.approx_lines:
        if not args.approx_contour_csv:
            raise ValueError("--approx-contour-csv is required with --approx-lines")
        if not args.approx_boundary_points:
            raise ValueError("--approx-boundary-points is required with --approx-lines")

        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(args.approx_contour_csv).stem
        refined_lines_json_path = output_dir / f"{base_name}_refined_lines.json"
        refined_lines_dxf_path = output_dir / f"{base_name}_refined_lines.dxf"
        chained_lines_json_path = output_dir / f"{base_name}_refined_chained_lines.json"
        chained_lines_dxf_path = output_dir / f"{base_name}_refined_chained_lines.dxf"
        mixed_contour_json_path = output_dir / f"{base_name}_mixed_contour.json"
        mixed_contour_dxf_path = output_dir / f"{base_name}_mixed_contour.dxf"
        mixed_with_holes_dxf_path = output_dir / f"{base_name}_mixed_with_holes.dxf"
        mixed_dxf_holes = []
        if args.mixed_dxf and args.holes_json:
            mixed_dxf_holes = load_holes_json_circles(args.holes_json)

        line_result = run_line_approximation(
            contour_csv_path=args.approx_contour_csv,
            boundary_points_path=args.approx_boundary_points,
            output_json_path=refined_lines_json_path,
            output_dxf_path=refined_lines_dxf_path,
            line_simplify_mm=args.line_simplify_mm,
            line_fit_band_mm=args.line_fit_band_mm,
            min_line_points=args.min_line_points,
            min_line_length_mm=args.min_line_length_mm,
            max_line_mean_error_mm=args.max_line_mean_error_mm,
            max_line_error_mm=args.max_line_error_mm,
            line_edge_percentile=args.line_edge_percentile,
            line_trim_outlier_percent=args.line_trim_outlier_percent,
            max_line_angle_diff_deg=args.max_line_angle_diff_deg,
            line_end_trim_percent=args.line_end_trim_percent,
            line_window_mm=args.line_window_mm,
            line_window_step_mm=args.line_window_step_mm,
            max_window_mean_error_mm=args.max_window_mean_error_mm,
            max_window_angle_diff_deg=args.max_window_angle_diff_deg,
            max_point_contour_distance_mm=args.max_point_contour_distance_mm,
            max_line_contour_distance_mm=args.max_line_contour_distance_mm,
            debug_line_id=args.debug_line_id,
            debug_output_dir=output_dir,
            debug_base_name=base_name,
            show_line_labels=args.show_line_labels,
            line_label_height_mm=args.line_label_height_mm,
            chain_lines=args.chain_lines,
            chained_json_path=chained_lines_json_path,
            chained_dxf_path=chained_lines_dxf_path,
            max_chain_intersection_distance_mm=(
                args.max_chain_intersection_distance_mm
            ),
            parallel_angle_eps_deg=args.parallel_angle_eps_deg,
            mixed_contour=args.mixed_contour,
            mixed_contour_json_path=mixed_contour_json_path,
            mixed_contour_dxf_path=mixed_contour_dxf_path,
            mixed_with_holes_dxf_path=(
                mixed_with_holes_dxf_path if args.mixed_dxf else None
            ),
            mixed_dxf_holes=mixed_dxf_holes,
            endpoint_mode=args.line_endpoint_mode,
        )

        print(f"Refined lines JSON: {refined_lines_json_path}")
        print(f"Refined lines DXF:  {refined_lines_dxf_path}")
        if args.chain_lines:
            print(f"Chained lines JSON: {chained_lines_json_path}")
            print(f"Chained lines DXF:  {chained_lines_dxf_path}")
            print(f"Chained lines count: {len(line_result.chained_lines):,}")
            print(
                "Successful intersections: "
                f"{line_result.chained_successful_intersections:,}"
            )
            print(f"Chained warnings count: {line_result.chained_warnings_count:,}")
        if args.mixed_contour:
            print(f"Mixed contour JSON: {mixed_contour_json_path}")
            print(f"Mixed contour DXF:  {mixed_contour_dxf_path}")
            print(f"Mixed contour elements: {len(line_result.mixed_contour_elements):,}")
        if args.mixed_dxf:
            if not mixed_dxf_holes:
                print("Warning: no holes available for mixed_with_holes.dxf")
            print(f"Mixed with holes DXF: {mixed_with_holes_dxf_path}")
        if args.demo_summary:
            demo_summary_path = output_dir / f"{base_name}_demo_summary.txt"
            mixed_line_count = sum(
                1
                for item in line_result.mixed_contour_elements
                if str(item.get("type", "")).upper() == "LINE"
            )
            mixed_polyline_count = sum(
                1
                for item in line_result.mixed_contour_elements
                if str(item.get("type", "")).upper() == "POLYLINE"
            )
            summary_lines = [
                "Demo summary",
                "",
                f"Input file path: {_summary_value(args.approx_boundary_points)}",
                "Input file size: not available",
                f"Cell size: {_summary_value(args.cell)}",
                f"Threshold: {_summary_value(args.threshold)}",
                "Grid size: not available",
                "",
                "Contour:",
                "  enabled: disabled",
                "  contour points count: not available",
                "  contour output path: not available",
                "",
                "Holes:",
                f"  enabled: {'enabled' if args.holes_json else 'disabled'}",
                "  total holes: not available",
                "  accepted holes: not available",
                "  rejected holes: not available",
                "  groups count: not available",
                f"  holes json path: {_summary_value(args.holes_json)}",
                "",
                "DXF:",
                "  ordinary dxf path: not available",
                f"  mixed_with_holes.dxf path: {_path_if(args.mixed_dxf, mixed_with_holes_dxf_path)}",
                "",
                "Refined lines:",
                f"  refined lines count: {len(line_result.lines)}",
                f"  refined_lines.json path: {refined_lines_json_path}",
                f"  refined_lines.dxf path: {refined_lines_dxf_path}",
                "",
                "Mixed contour:",
                f"  total elements: {len(line_result.mixed_contour_elements) if args.mixed_contour else 'not available'}",
                f"  LINE count: {mixed_line_count if args.mixed_contour else 'not available'}",
                f"  POLYLINE count: {mixed_polyline_count if args.mixed_contour else 'not available'}",
                f"  mixed_contour.json path: {_path_if(args.mixed_contour, mixed_contour_json_path)}",
                f"  mixed_contour.dxf path: {_path_if(args.mixed_contour, mixed_contour_dxf_path)}",
                "",
                *_demo_summary_notes(),
            ]
            write_demo_summary(demo_summary_path, summary_lines)
            print(f"Demo summary: {demo_summary_path}")
        print(f"Total segments:     {line_result.total_segments:,}")
        print(f"Accepted lines:     {len(line_result.lines):,}")
        print(f"Rejected segments:  {len(line_result.rejected_segments):,}")
        print("Rejected by reason:")
        if line_result.rejected_by_reason:
            for reason, count in sorted(line_result.rejected_by_reason.items()):
                print(f"  {reason}: {count:,}")
        else:
            print("  none: 0")
        return

    if not args.input_file:
        raise ValueError("input_file is required unless --approx-lines is used")

    input_path = Path(args.input_file)
    output_dir = Path(args.out)
    cache_dir = Path(args.cache)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    input_file_size = input_path.stat().st_size
    if input_file_size > 10 * 1024 * 1024 * 1024:
        print(
            "Warning: large input file, decimated export is recommended for CAD preview."
        )

    base_name = input_path.stem

    

    print(f"Файл: {input_path}")
    print(f"Размер ячейки: {args.cell}")

    t0 = time.perf_counter()

    print("\n[1/3] Считаем или загружаем статистику файла...")

    statistics_result = prepare_statistics(
        source_path=input_path,
        cache_dir=cache_dir,
        use_cache=not args.no_cache,
    )
    stats = statistics_result.stats

    if statistics_result.from_cache:
        print("Статистика загружена из кэша.")
    elif not args.no_cache:
        print("Статистика сохранена в кэш.")

    t1 = time.perf_counter()

    print(f"Точек: {stats.point_count:,}")
    print(f"X: {stats.min_x:.3f} .. {stats.max_x:.3f}")
    print(f"Y: {stats.min_y:.3f} .. {stats.max_y:.3f}")
    print(f"Z: {stats.min_z:.3f} .. {stats.max_z:.3f}")
    print(f"Время статистики: {t1 - t0:.2f} сек")

    print("\n[2/3] Строим или загружаем карту плотности...")

    density_result = prepare_density(
        source_path=input_path,
        cell_size=args.cell,
        cache_dir=cache_dir,
        use_cache=not args.no_cache,
        statistics=statistics_result,
    )
    grid = density_result.grid

    if density_result.density_from_cache:
        print("Density grid загружен из кэша.")
    elif not args.no_cache:
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
    mask_after_edits_path = (
        output_dir / f"{base_name}_mask_after_edits_cell_{cell_text}_threshold_{args.threshold}.png"
    )
    mask_edits_debug_path = (
        output_dir / f"{base_name}_mask_edits_debug_cell_{cell_text}_threshold_{args.threshold}.png"
    )
    report_path = output_dir / f"{base_name}_report_cell_{cell_text}.txt"
    demo_summary_path = output_dir / f"{base_name}_demo_summary.txt"
    clean_path = (
        output_dir / f"{base_name}_clean_cell_{cell_text}_threshold_{args.threshold}.asc"
    )
    boundary_width_text = str(args.boundary_width_mm).replace(".", "_")
    boundary_path = (
        output_dir
        / (
            f"{base_name}_boundary_cell_{cell_text}_threshold_{args.threshold}"
            f"_width_{boundary_width_text}mm.asc"
        )
    )
    decimate_cell_text = str(args.decimate_cell_mm).replace(".", "_")
    decimated_path = output_dir / f"{base_name}_decimated_cell_{decimate_cell_text}.asc"

    holes_csv_path = (
        output_dir / f"{base_name}_holes_cell_{cell_text}_threshold_{args.threshold}.csv"
    )
    holes_json_path = (
        output_dir / f"{base_name}_holes_cell_{cell_text}_threshold_{args.threshold}.json"
    )

    contour_preview_path = (
        output_dir / f"{base_name}_contour_cell_{cell_text}_threshold_{args.threshold}.png"
    )
    contour_csv_path = (
        output_dir / f"{base_name}_contour_cell_{cell_text}_threshold_{args.threshold}.csv"
    )

    contour_dxf_path = (
        output_dir / f"{base_name}_contour_cell_{cell_text}_threshold_{args.threshold}.dxf"
    )

    holes_preview_path = (
        output_dir / f"{base_name}_holes_cell_{cell_text}_threshold_{args.threshold}.png"
        )

    save_density_preview(grid, preview_path, max_size=args.preview_size)

    if args.smooth_mm > 0:
        save_smoothed_density_preview(
            grid,
            smooth_preview_path,
            sigma_mm=args.smooth_mm,
            max_size=args.preview_size,
        )

    threshold_mode = "auto"
    manual_threshold = None
    if args.threshold != "auto":
        threshold_mode = "manual"
        manual_threshold = float(args.threshold)

    roi = tuple(args.roi) if args.roi is not None else None
    mask_edits = load_mask_edits(args.mask_edits) if args.mask_edits else None
    masks = build_processing_masks(
        grid=grid,
        threshold_mode=threshold_mode,
        manual_threshold=manual_threshold,
        min_component_area=args.min_component_area,
        keep_largest=args.keep_largest,
        roi=roi,
        polygon_roi=args.roi_poly,
        mask_edits=mask_edits,
        fill_holes_area=args.fill_holes_area,
    )
    mask_result = masks.threshold_result
    mask_for_holes = masks.mask_for_holes
    mask = masks.contour_mask

    if roi is not None:
        print(
            "Applied ROI: "
            f"min_x={roi[0]:.3f}, min_y={roi[1]:.3f}, "
            f"max_x={roi[2]:.3f}, max_y={roi[3]:.3f}"
        )

    if args.roi_poly is not None:
        mask_after_polygon = masks.mask_before_edits
        if mask_after_polygon is None:
            mask_after_polygon = mask_for_holes
        print(
            "Applied polygon ROI: "
            f"points={len(args.roi_poly)}, "
            f"mask_cells={int(mask_after_polygon.sum()):,}"
        )

    if args.mask_edits:
        mask_before_edits = masks.mask_before_edits
        mask_edits_stats = masks.mask_edits_stats
        if (
            mask_before_edits is None
            or mask_edits_stats is None
            or mask_edits is None
        ):
            raise RuntimeError("Mask edits result is incomplete")
        mask_cells_before_edits = int(mask_before_edits.sum())
        mask_cells_after_edits = int(mask_for_holes.sum())
        mask_cells_changed_by_edits = int(
            (mask_for_holes != mask_before_edits).sum()
        )
        print(f"Applied mask edits: {len(mask_edits):,}")
        print(f"Mask cells before edits: {mask_cells_before_edits:,}")
        print(f"Mask cells after edits:  {mask_cells_after_edits:,}")
        print(f"Mask cells changed by edits: {mask_cells_changed_by_edits:,}")
        print(f"Mask edits total: {mask_edits_stats.total_edits:,}")
        print(f"Mask edits inside grid: {mask_edits_stats.edits_inside_grid:,}")
        print(f"Mask edits outside grid: {mask_edits_stats.edits_outside_grid:,}")
        print(
            "Mask edits touched white mask: "
            f"{mask_edits_stats.edits_that_touched_white_mask:,}"
        )
        print(f"Mask edits changed cells: {mask_edits_stats.changed_cells:,}")
        print(
            "Mask edits world bbox: "
            f"{mask_edits_stats.world_min_x}, {mask_edits_stats.world_min_y} .. "
            f"{mask_edits_stats.world_max_x}, {mask_edits_stats.world_max_y}"
        )
        print(
            "Mask edits pixel bbox: "
            f"{mask_edits_stats.pixel_min_ix}, {mask_edits_stats.pixel_min_iy} .. "
            f"{mask_edits_stats.pixel_max_ix}, {mask_edits_stats.pixel_max_iy}"
        )
        save_mask_preview(
            mask_for_holes,
            mask_after_edits_path,
            max_size=args.preview_size,
        )
        save_mask_edits_debug_preview(
            mask=mask_before_edits,
            grid=grid,
            edits=mask_edits,
            output_path=mask_edits_debug_path,
            max_size=args.preview_size,
        )
    holes = []
    hole_groups = []

    if args.holes:
        max_hole_diameter = (
            None if args.max_hole_diameter_mm <= 0 else args.max_hole_diameter_mm
        )

        hole_detection_result = find_hole_candidates(
            masks=masks,
            min_diameter_mm=args.min_hole_diameter_mm,
            max_diameter_mm=max_hole_diameter,
            min_circularity=args.min_circularity,
            max_error_ratio=args.max_circle_error_ratio,
            group_tolerance_mm=args.hole_group_tolerance_mm,
        )
        holes = hole_detection_result.candidates
        hole_groups = hole_detection_result.groups

        save_holes_csv(
            holes=holes,
            output_path=holes_csv_path,
            only_accepted=False,
        )
        save_holes_json(
            holes=holes,
            output_path=holes_json_path,
            groups=hole_groups,
            only_accepted=False,
        )

        save_holes_preview(
            mask=mask_for_holes,
            holes=holes,
            output_path=holes_preview_path,
            max_size=args.preview_size,
        )

    save_mask_preview(mask, mask_path, max_size=args.preview_size)
    save_report(stats, grid, report_path)

    decimated_point_count = None

    if args.export_decimated:
        decimated_point_count = export_decimated_points(
            input_path=input_path,
            output_path=decimated_path,
            stats=stats,
            decimate_cell_mm=args.decimate_cell_mm,
        )

    clean_point_count = None

    if args.export_clean:
        clean_point_count = export_clean_points(
            input_path=input_path,
            output_path=clean_path,
            grid=grid,
            mask=mask,
        )

    boundary_point_count = None

    if args.export_boundary:
        boundary_width_cells = int(math.ceil(args.boundary_width_mm / grid.cell_size))
        boundary_mask = build_boundary_band_mask(
            mask=mask,
            width_cells=boundary_width_cells,
        )
        boundary_point_count = export_boundary_points(
            input_path=input_path,
            output_path=boundary_path,
            grid=grid,
            boundary_mask=boundary_mask,
        )

    dxf_holes = []
    if args.holes_json:
        dxf_holes = load_holes_json_circles(args.holes_json)

    contour_result = None

    if args.contour:
        contour_result = extract_preliminary_contour(
            masks=masks,
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
                holes=dxf_holes,
            )

    t3 = time.perf_counter()

    print(f"Preview:        {preview_path}")
    if args.smooth_mm > 0:
        print(f"Smooth preview: {smooth_preview_path}")
    print(f"Mask preview:   {mask_path}")
    print(f"Report:         {report_path}")
    print(f"Mask threshold: {mask_result.threshold:.3f}")
    print(f"Mask cells:     {int(mask.sum()):,}")
    if decimated_point_count is not None:
        reduction_ratio = (
            stats.point_count / decimated_point_count
            if decimated_point_count > 0
            else 0.0
        )
        print(f"Decimated points: {decimated_path}")
        print(f"Input points count:    {stats.point_count:,}")
        print(f"Exported points count: {decimated_point_count:,}")
        print(f"Reduction ratio:       {reduction_ratio:.3f}x")
    if clean_point_count is not None:
        print(f"Clean points:   {clean_path}")
        print(f"Clean count:    {clean_point_count:,}")
    if boundary_point_count is not None:
        print(f"Boundary points:{boundary_path}")
        print(f"Boundary count: {boundary_point_count:,}")
    if args.holes_json:
        print(f"Holes JSON input:{args.holes_json}")
        print(f"DXF holes count: {len(dxf_holes):,}")
    if contour_result is not None:
        print(f"Contour preview: {contour_preview_path}")
        print(f"Contour CSV:     {contour_csv_path}")

        if args.dxf:
            print(f"Contour DXF:     {contour_dxf_path}")

        print(f"Contour points:  {contour_result.point_count:,}")

    if args.holes:
        accepted_holes = [hole for hole in holes if hole.accepted]

        print(f"Holes CSV:      {holes_csv_path}")
        print(f"Holes JSON:     {holes_json_path}")
        print(f"Hole candidates:{len(holes):,}")
        print(f"Accepted holes: {len(accepted_holes):,}")

        for hole in accepted_holes[:20]:
            print(
                f"  Hole {hole.id}: "
                f"center=({hole.center_x:.3f}, {hole.center_y:.3f}), "
                f"r={hole.radius:.3f}, "
                f"d={hole.diameter:.3f}, "
                f"err={hole.error_ratio:.3f}"
            )

        if len(accepted_holes) > 20:
            print("  ...")
    print(f"Holes preview:  {holes_preview_path}")
    if args.demo_summary:
        total_holes = len(holes) if args.holes else "not available"
        accepted_holes_count = (
            sum(1 for hole in holes if hole.accepted) if args.holes else "not available"
        )
        rejected_holes_count = (
            len(holes) - sum(1 for hole in holes if hole.accepted)
            if args.holes
            else "not available"
        )
        holes_json_summary_path = (
            str(args.holes_json)
            if args.holes_json
            else str(holes_json_path)
            if args.holes
            else "not available"
        )
        if grid is not None and hasattr(grid, "width_cells") and hasattr(grid, "height_cells"):
            grid_size = f"{grid.width_cells} x {grid.height_cells}"
        elif grid is not None and hasattr(grid, "density"):
            grid_size = f"{grid.density.shape[1]} x {grid.density.shape[0]}"
        else:
            grid_size = "not available"

        summary_lines = [
            "Demo summary",
            "",
            f"Input file path: {input_path}",
            f"Input file size: {_file_size_text(input_path)}",
            f"Cell size: {args.cell}",
            f"Threshold: {mask_result.threshold:.3f}",
            f"Grid size: {grid_size}",
            "",
            "Contour:",
            f"  enabled: {'enabled' if args.contour else 'disabled'}",
            "  contour points count: "
            f"{contour_result.point_count if contour_result is not None else 'not available'}",
            f"  contour output path: {_path_if(contour_result is not None, contour_csv_path)}",
            "",
            "Holes:",
            f"  enabled: {'enabled' if args.holes else 'disabled'}",
            f"  total holes: {total_holes}",
            f"  accepted holes: {accepted_holes_count}",
            f"  rejected holes: {rejected_holes_count}",
            f"  groups count: {len(hole_groups) if args.holes else 'not available'}",
            f"  holes json path: {holes_json_summary_path}",
            "",
            "DXF:",
            f"  ordinary dxf path: {_path_if(bool(args.dxf and contour_result is not None), contour_dxf_path)}",
            "  mixed_with_holes.dxf path: not available",
            "",
            "Refined lines:",
            "  refined lines count: not available",
            "  refined_lines.json path: not available",
            "  refined_lines.dxf path: not available",
            "",
            "Mixed contour:",
            "  total elements: not available",
            "  LINE count: not available",
            "  POLYLINE count: not available",
            "  mixed_contour.json path: not available",
            "  mixed_contour.dxf path: not available",
            "",
            *_demo_summary_notes(),
        ]
        write_demo_summary(demo_summary_path, summary_lines)
        print(f"Demo summary:   {demo_summary_path}")
    print(f"Время сохранения: {t3 - t2:.2f} сек")

    print(f"\nГотово. Общее время: {t3 - t0:.2f} сек")

    
if __name__ == "__main__":
    main()
