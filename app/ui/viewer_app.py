from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from app.core.coordinate_transform import CoordinateTransform
from app.core.density_grid import DensityGrid
from app.core.working_area import WorkingArea
from app.services.coarse_processing import DensityProcessingResult
from app.ui.density_workflow import (
    SUPPORTED_POINT_CLOUD_EXTENSIONS,
    active_working_area_is_visible,
    apply_working_area_state,
    begin_rectangle_draft,
    clear_polygon_draft_state,
    clear_working_area_state,
    enter_polygon_mode_state,
    enter_rectangle_mode_state,
    finish_rectangle_draft,
    grid_preview_params,
    reset_source_dependent_state,
    update_rectangle_transient,
    validate_density_request,
    working_area_draft_visibility,
)
from app.ui.density_worker import (
    DensityWorker,
    DensityWorkerError,
    DensityWorkerProgress,
    DensityWorkerResult,
    format_progress_bytes,
    progress_stage_label,
)
from app.ui.contour_workflow import (
    MaskEditingSession,
    PreliminaryContourParameters,
    PreliminaryContourSession,
    find_preliminary_contour_for_working_area,
    mask_edits_for_world_segment,
    preliminary_contour_parameters_for_threshold,
    processing_mask_to_preview,
    rebase_preliminary_contour_with_edits,
    rebuild_preliminary_contour_from_edited_mask,
    validate_brush_diameter,
)
from app.ui.hole_workflow import (
    HoleHitRegion,
    HoleDetectionParameters,
    HoleDetectionSession,
    apply_manual_hole_status,
    find_holes_for_current_mask,
    hit_test_hole_regions,
    hole_detection_session_is_current,
    invalidate_hole_detection_state,
    validate_hole_detection_parameters,
)


TEXTURE_TAG = "preview_texture"
SELECTION_TEXTURE_TAG = "selection_dim_texture"
PROCESSING_MASK_TEXTURE_TAG = "processing_mask_texture"
IMAGE_TAG = "preview_drawlist"
POLYGON_LAYER_TAG = "polygon_overlay_layer"
SELECTION_LAYER_TAG = "selection_dim_layer"
PROCESSING_MASK_LAYER_TAG = "processing_mask_layer"
MASK_EDITS_LAYER_TAG = "mask_edits_layer"
HOLES_LAYER_TAG = "holes_overlay_layer"
MANUAL_HOLE_CENTER_LAYER_TAG = "manual_hole_center_overlay_layer"
BRUSH_CURSOR_LAYER_TAG = "brush_cursor_overlay_layer"
CONTOUR_LAYER_TAG = "contour_overlay_layer"
MIXED_CONTOUR_LAYER_TAG = "mixed_contour_overlay_layer"
SHOW_ROI_OVERLAY_TAG = "show_roi_overlay"
SHOW_MASK_REMOVE_EDITS_TAG = "show_mask_remove_edits"
SHOW_MASK_ADD_EDITS_TAG = "show_mask_add_edits"
SHOW_ACCEPTED_HOLES_TAG = "show_accepted_holes"
SHOW_REJECTED_HOLES_TAG = "show_rejected_holes"
SHOW_UNGROUPED_HOLES_TAG = "show_ungrouped_holes"
SHOW_OVERSIZED_HOLES_TAG = "show_oversized_holes"
MAX_DISPLAYED_HOLE_DIAMETER_TAG = "max_displayed_hole_diameter_mm"
SHOW_CONTOUR_TAG = "show_contour"
SHOW_PROCESSING_MASK_TAG = "show_processing_mask"
SHOW_MIXED_CONTOUR_LINES_TAG = "show_mixed_contour_lines"
SHOW_MIXED_CONTOUR_GAPS_TAG = "show_mixed_contour_gaps"
STATUS_TAG = "status_text"
COORDS_TAG = "coords_text"
ROI_STATUS_TAG = "roi_status_text"
WORKING_AREA_INFO_TAG = "working_area_info_text"
ROI_OUTPUT_TAG = "roi_output_text"
POLYGON_COUNT_TAG = "polygon_count_text"
POLYGON_LAST_TAG = "polygon_last_text"
POLYGON_POINTS_TAG = "polygon_points_text"
POLYGON_OUTPUT_TAG = "polygon_output_text"
ZOOM_TEXT_TAG = "zoom_text"
COMMAND_OUTPUT_TAG = "command_output_text"
DEBUG_COORDS_TAG = "debug_coords_text"
BRUSH_SIZE_TAG = "brush_size_mm"
BRUSH_MODE_TAG = "brush_mode"
BRUSH_EDITS_COUNT_TAG = "brush_edits_count_text"
MASK_BRUSH_BUTTON_TAG = "mask_brush_button"
MASK_CLEAR_BUTTON_TAG = "mask_clear_button"
LAST_BRUSH_DEBUG_TAG = "last_brush_debug_text"
HOLES_STATS_TAG = "holes_stats_text"
HOLE_DETECTION_INFO_TAG = "hole_detection_info_text"
HOLE_MIN_DIAMETER_TAG = "hole_min_diameter_mm"
HOLE_MAX_DIAMETER_TAG = "hole_max_diameter_mm"
HOLE_ADVANCED_HEADER_TAG = "advanced_hole_detection_header"
HOLE_KEEP_LARGEST_TAG = "hole_keep_largest_component"
HOLE_MIN_CIRCULARITY_TAG = "hole_min_circularity"
HOLE_MAX_ERROR_RATIO_TAG = "hole_max_circle_error_ratio"
FIND_HOLES_BUTTON_TAG = "find_holes_button"
HOLE_GROUPS_CONTAINER_TAG = "hole_groups_container"
HOLE_HOVER_INFO_TAG = "hole_hover_info_text"
MOVE_HOLE_ID_TAG = "move_hole_id"
MOVE_HOLE_TARGET_GROUP_TAG = "move_hole_target_group"
EDIT_GROUP_TARGET_TAG = "edit_group_target"
EDIT_GROUP_DIAMETER_TAG = "edit_group_diameter"
MANUAL_HOLE_X_TAG = "manual_hole_x"
MANUAL_HOLE_Y_TAG = "manual_hole_y"
MANUAL_HOLE_DIAMETER_TAG = "manual_hole_diameter"
MANUAL_HOLE_PICK_TAG = "manual_hole_pick_center"
CONTOUR_INFO_TAG = "contour_info_text"
CONTOUR_THRESHOLD_MODE_TAG = "contour_threshold_mode"
CONTOUR_MANUAL_THRESHOLD_TAG = "contour_manual_threshold"
CONTOUR_FILL_HOLES_AREA_TAG = "contour_fill_holes_area_cells"
MIXED_CONTOUR_INFO_TAG = "mixed_contour_info_text"
DEMO_SUMMARY_INFO_TAG = "demo_summary_info_text"
DENSITY_SOURCE_FILE_TAG = "density_source_file"
DENSITY_CELL_SIZE_TAG = "density_cell_size"
DENSITY_SESSION_INFO_TAG = "density_session_info"
DENSITY_SOURCE_DIALOG_TAG = "open_density_source_dialog"
DENSITY_SELECT_BUTTON_TAG = "density_select_button"
DENSITY_BUILD_BUTTON_TAG = "density_build_button"
DENSITY_PROGRESS_STAGE_TAG = "density_progress_stage"
DENSITY_PROGRESS_BAR_TAG = "density_progress_bar"
DENSITY_PROGRESS_BYTES_TAG = "density_progress_bytes"
DENSITY_SETUP_WINDOW_TAG = "density_setup_window"
DENSITY_SUMMARY_TAG = "density_summary_text"
DENSITY_STATUS_TAG = "density_status_text"
UNDO_BUTTON_TAG = "undo_button"

CMD_INPUT_FILE_TAG = "cmd_input_file_path"
CMD_CELL_TAG = "cmd_cell"
CMD_THRESHOLD_TAG = "cmd_threshold"
CMD_FILL_HOLES_TAG = "cmd_fill_holes_area"
CMD_BOUNDARY_WIDTH_TAG = "cmd_boundary_width_mm"
CMD_SIMPLIFY_TAG = "cmd_simplify_mm"
CMD_MASK_EDITS_TAG = "cmd_mask_edits"
CMD_KEEP_LARGEST_TAG = "cmd_keep_largest"
CMD_CONTOUR_TAG = "cmd_contour"
CMD_DXF_TAG = "cmd_dxf"
CMD_EXPORT_CLEAN_TAG = "cmd_export_clean"
CMD_EXPORT_BOUNDARY_TAG = "cmd_export_boundary"
CMD_HOLES_TAG = "cmd_holes"

PARAM_GRID_MIN_X = "param_grid_min_x"
PARAM_GRID_MIN_Y = "param_grid_min_y"
PARAM_CELL_SIZE = "param_cell_size"
PARAM_GRID_WIDTH = "param_original_grid_width"
PARAM_GRID_HEIGHT = "param_original_grid_height"

MIN_ZOOM = 0.1
MAX_ZOOM = 16.0
ZOOM_STEP = 1.25
MOUSE_DRAG_THRESHOLD_PX = 5.0
CANVAS_WIDTH = 2400
CANVAS_HEIGHT = 1600

state = {
    "density_result": None,
    "density_preview": None,
    "texture_tag": TEXTURE_TAG,
    "image_width": 0,
    "image_height": 0,
    "zoom": 1.0,
    "pan_x": 0.0,
    "pan_y": 0.0,
    "last_pan_mouse": None,
    "roi_first_world": None,
    "roi_current_world": None,
    "rectangle_roi": None,
    "polygon_finished": False,
    "selection_applied": False,
    "editing_overlay_visible": True,
    "selection_kind": None,
    "selection_polygon_points": [],
    "mode": "rectangle",
    "polygon_points": [],
    "active_working_area": None,
    "working_area_density_session": None,
    "mask_edits": [],
    "last_brush_image": None,
    "last_brush_world": None,
    "brush_cursor_image": None,
    "active_brush_stroke_id": None,
    "next_brush_stroke_id": 1,
    "coarse_mask_revision": 0,
    "hole_detection_session": None,
    "holes_outdated": False,
    "hole_overlay_source": None,
    "hovered_hole_id": None,
    "mouse_gestures": {},
    "holes": [],
    "hole_groups": [],
    "visible_hole_group_ids": {},
    "pick_manual_hole_center": False,
    "manual_hole_center_world": None,
    "contour_points": [],
    "contour_file": "",
    "contour_processing_result": None,
    "mask_editing_session": None,
    "undo_history": [],
    "processing_mask_preview": None,
    "mixed_contour_elements": [],
    "mixed_contour_file": "",
    "demo_summary_file": "",
    "report_loaded": False,
}

_density_worker = DensityWorker()


def _set_status(message: str) -> None:
    dpg.set_value(STATUS_TAG, message)


def _set_density_status(message: str) -> None:
    _set_status(message)
    if dpg.does_item_exist(DENSITY_STATUS_TAG):
        dpg.set_value(DENSITY_STATUS_TAG, message)


def _set_density_controls_enabled(enabled: bool) -> None:
    for tag in (
        DENSITY_SOURCE_FILE_TAG,
        DENSITY_SELECT_BUTTON_TAG,
        DENSITY_CELL_SIZE_TAG,
        DENSITY_BUILD_BUTTON_TAG,
    ):
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, enabled=enabled)


def _set_density_progress_waiting() -> None:
    dpg.set_value(DENSITY_PROGRESS_STAGE_TAG, "Processing stage: checking cache...")
    dpg.set_value(DENSITY_PROGRESS_BAR_TAG, 0.0)
    dpg.configure_item(DENSITY_PROGRESS_BAR_TAG, overlay="Waiting")
    dpg.set_value(DENSITY_PROGRESS_BYTES_TAG, "Waiting for source-read progress...")


def _apply_density_progress(message: DensityWorkerProgress) -> None:
    progress = message.progress
    dpg.set_value(
        DENSITY_PROGRESS_STAGE_TAG,
        f"Processing stage: {progress_stage_label(progress.stage)}",
    )
    dpg.set_value(DENSITY_PROGRESS_BAR_TAG, progress.fraction)
    dpg.configure_item(
        DENSITY_PROGRESS_BAR_TAG,
        overlay=f"{progress.fraction * 100.0:.0f}%",
    )
    dpg.set_value(DENSITY_PROGRESS_BYTES_TAG, format_progress_bytes(progress))


def _finish_density_worker_result(message: DensityWorkerResult) -> None:
    result = message.result
    try:
        _show_density_result(result, message.preview, message.texture_rgba)
    except Exception as exc:
        dpg.set_value(DENSITY_PROGRESS_STAGE_TAG, "Processing stage: display failed")
        dpg.configure_item(DENSITY_PROGRESS_BAR_TAG, overlay="Error")
        _set_density_status(f"Could not display density map: {exc}")
    else:
        if result.stats_from_cache and result.density_from_cache:
            dpg.set_value(
                DENSITY_PROGRESS_STAGE_TAG,
                "Processing stage: loaded cached density",
            )
            dpg.set_value(DENSITY_PROGRESS_BAR_TAG, 0.0)
            dpg.configure_item(DENSITY_PROGRESS_BAR_TAG, overlay="Cache hit")
            dpg.set_value(DENSITY_PROGRESS_BYTES_TAG, "No source read required")
        elif result.density_from_cache:
            dpg.set_value(
                DENSITY_PROGRESS_STAGE_TAG,
                "Processing stage: statistics complete; density loaded from cache",
            )
            dpg.configure_item(DENSITY_PROGRESS_BAR_TAG, overlay="Done")
        else:
            dpg.set_value(
                DENSITY_PROGRESS_STAGE_TAG,
                "Processing stage: density map ready",
            )
            dpg.set_value(DENSITY_PROGRESS_BAR_TAG, 1.0)
            dpg.configure_item(DENSITY_PROGRESS_BAR_TAG, overlay="100%")

        if result.density_from_cache:
            _set_density_status("Density map loaded from cache.")
        else:
            _set_density_status("Density map is ready.")
    finally:
        _density_worker.mark_finished()
        _set_density_controls_enabled(True)


def _finish_density_worker_error(message: DensityWorkerError) -> None:
    dpg.set_value(DENSITY_PROGRESS_STAGE_TAG, "Processing stage: failed")
    dpg.configure_item(DENSITY_PROGRESS_BAR_TAG, overlay="Error")
    _set_density_status(f"Could not build density map: {message.error}")
    _density_worker.mark_finished()
    _set_density_controls_enabled(True)


def _process_density_worker_messages() -> None:
    pending_progress: DensityWorkerProgress | None = None
    for message in _density_worker.drain_messages():
        if isinstance(message, DensityWorkerProgress):
            pending_progress = message
            continue

        if pending_progress is not None:
            _apply_density_progress(pending_progress)
            pending_progress = None

        if isinstance(message, DensityWorkerResult):
            _finish_density_worker_result(message)
        else:
            _finish_density_worker_error(message)

    if pending_progress is not None:
        _apply_density_progress(pending_progress)


def _update_density_session_info() -> None:
    result = state["density_result"]
    if result is None:
        text = "Density map has not been built."
        summary = "No density map"
        if dpg.does_item_exist(DENSITY_SESSION_INFO_TAG):
            dpg.set_value(DENSITY_SESSION_INFO_TAG, text)
        if dpg.does_item_exist(DENSITY_SUMMARY_TAG):
            dpg.set_value(DENSITY_SUMMARY_TAG, summary)
        return

    cache_status = "cache" if result.density_from_cache else "calculated"
    text = (
        f"{result.source_path.name} | {result.stats.point_count:,} pts | "
        f"cell {result.grid.cell_size:g} mm | "
        f"grid {result.grid.width} x {result.grid.height} | {cache_status}"
    )
    if dpg.does_item_exist(DENSITY_SESSION_INFO_TAG):
        dpg.set_value(DENSITY_SESSION_INFO_TAG, text)
    if dpg.does_item_exist(DENSITY_SUMMARY_TAG):
        dpg.set_value(DENSITY_SUMMARY_TAG, text)


def get_active_working_area() -> WorkingArea | None:
    working_area = state.get("active_working_area")
    density_result = state.get("density_result")
    if not isinstance(working_area, WorkingArea) or density_result is None:
        return None
    if state.get("working_area_density_session") is not density_result:
        return None
    return working_area


def _update_working_area_info() -> None:
    if not dpg.does_item_exist(WORKING_AREA_INFO_TAG):
        return

    working_area = get_active_working_area()
    if working_area is None:
        dpg.set_value(WORKING_AREA_INFO_TAG, "Working area: Full scan")
        return

    if working_area.kind == "rectangle":
        assert working_area.rectangle_bounds is not None
        min_x, min_y, max_x, max_y = working_area.rectangle_bounds
        dpg.set_value(
            WORKING_AREA_INFO_TAG,
            "Working area: Rectangle | "
            f"{max_x - min_x:.3f} x {max_y - min_y:.3f} mm",
        )
    else:
        dpg.set_value(
            WORKING_AREA_INFO_TAG,
            f"Working area: Polygon | vertices: {len(working_area.polygon_points)}",
        )


def _set_grid_parameter_values(grid: DensityGrid) -> None:
    grid_min_x, grid_min_y, cell_size, grid_width, grid_height = (
        grid_preview_params(grid)
    )
    dpg.set_value(PARAM_GRID_MIN_X, grid_min_x)
    dpg.set_value(PARAM_GRID_MIN_Y, grid_min_y)
    dpg.set_value(PARAM_CELL_SIZE, cell_size)
    dpg.set_value(PARAM_GRID_WIDTH, grid_width)
    dpg.set_value(PARAM_GRID_HEIGHT, grid_height)
    state["report_loaded"] = False


def _clear_source_dependent_ui_state() -> None:
    reset_source_dependent_state(state)
    _delete_processing_mask_texture()
    _set_mask_edit_controls_enabled(False)
    _update_mask_edits_count()
    _redraw_brush_cursor_overlay()

    if dpg.does_item_exist(SELECTION_TEXTURE_TAG):
        dpg.delete_item(SELECTION_TEXTURE_TAG)
    if dpg.does_item_exist(MANUAL_HOLE_PICK_TAG):
        dpg.set_value(MANUAL_HOLE_PICK_TAG, False)
    if dpg.does_item_exist(HOLE_HOVER_INFO_TAG):
        dpg.set_value(HOLE_HOVER_INFO_TAG, "Hover a hole candidate for details.")
    if dpg.does_item_exist(MANUAL_HOLE_X_TAG):
        dpg.set_value(MANUAL_HOLE_X_TAG, 0.0)
    if dpg.does_item_exist(MANUAL_HOLE_Y_TAG):
        dpg.set_value(MANUAL_HOLE_Y_TAG, 0.0)
    if dpg.does_item_exist(MOVE_HOLE_ID_TAG):
        dpg.set_value(MOVE_HOLE_ID_TAG, 0)
    if dpg.does_item_exist(ROI_STATUS_TAG):
        dpg.set_value(ROI_STATUS_TAG, "ROI mode: click two image corners.")
    if dpg.does_item_exist(ROI_OUTPUT_TAG):
        dpg.set_value(ROI_OUTPUT_TAG, "")
    if dpg.does_item_exist(POLYGON_OUTPUT_TAG):
        dpg.set_value(POLYGON_OUTPUT_TAG, "")
    if dpg.does_item_exist(COORDS_TAG):
        dpg.set_value(
            COORDS_TAG,
            "World X: - | World Y: -",
        )
    if dpg.does_item_exist(DEBUG_COORDS_TAG):
        dpg.set_value(
            DEBUG_COORDS_TAG,
            "mouse_screen_x=- mouse_screen_y=-\n"
            "draw_min_x=- draw_min_y=-\n"
            "local_mouse_x=- local_mouse_y=-\n"
            "image_x=- image_y=-\n"
            "zoom=1\n"
            "pan_x=0 pan_y=0\n"
            "image_origin_x=0 image_origin_y=0",
        )

    _update_polygon_points_text()
    _update_contour_info()
    _update_holes_stats()
    _update_hole_detection_info()
    _update_hole_groups_display()
    _update_hole_group_target_combo()
    _update_mixed_contour_info()
    _update_mask_edits_count()
    _update_last_brush_debug()
    _update_demo_summary_info()
    _update_working_area_info()


def _show_density_result(
    result: DensityProcessingResult,
    preview: np.ndarray,
    texture_rgba: np.ndarray,
) -> None:
    preview_height, preview_width = preview.shape
    new_texture_tag = f"{TEXTURE_TAG}_{id(result)}"

    with dpg.texture_registry():
        dpg.add_static_texture(
            width=preview_width,
            height=preview_height,
            default_value=texture_rgba.ravel(),
            tag=new_texture_tag,
        )

    if dpg.does_item_exist(IMAGE_TAG):
        dpg.delete_item(IMAGE_TAG)
    old_texture_tag = state["texture_tag"]
    if dpg.does_item_exist(old_texture_tag):
        dpg.delete_item(old_texture_tag)

    if dpg.does_item_exist("image_hint"):
        dpg.delete_item("image_hint")

    state["density_result"] = result
    state["density_preview"] = preview
    state["texture_tag"] = new_texture_tag
    state["image_width"] = preview_width
    state["image_height"] = preview_height
    state["zoom"] = 1.0
    state["pan_x"] = 0.0
    state["pan_y"] = 0.0
    state["last_pan_mouse"] = None

    _set_grid_parameter_values(result.grid)
    _clear_source_dependent_ui_state()
    if dpg.does_item_exist(CMD_INPUT_FILE_TAG):
        dpg.set_value(CMD_INPUT_FILE_TAG, str(result.source_path))
    if dpg.does_item_exist(CMD_CELL_TAG):
        dpg.set_value(CMD_CELL_TAG, result.grid.cell_size)
    dpg.set_value(ZOOM_TEXT_TAG, "Zoom: 100%")
    _update_density_session_info()
    _redraw_preview()


def _select_density_source_callback(_sender, app_data) -> None:
    if _density_worker.is_active:
        _set_density_status("Density processing is already active.")
        return

    selections = app_data.get("selections", {})
    if not selections:
        return
    dpg.set_value(DENSITY_SOURCE_FILE_TAG, next(iter(selections.values())))
    _set_density_status("Source selected. Review cell size and build density.")


def _build_density_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    if _density_worker.is_active:
        _set_density_status("Density processing is already active.")
        return

    try:
        source_path, cell_size = validate_density_request(
            dpg.get_value(DENSITY_SOURCE_FILE_TAG),
            dpg.get_value(DENSITY_CELL_SIZE_TAG),
        )
    except ValueError as exc:
        _set_density_status(str(exc))
        return

    try:
        started = _density_worker.start(source_path, cell_size)
    except Exception as exc:
        _set_density_status(f"Could not start density worker: {exc}")
        return

    if not started:
        _set_density_status("Density processing is already active.")
        return

    _set_density_controls_enabled(False)
    _set_density_progress_waiting()
    _set_density_status("Building density map in background...")


def _format_cli_float(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _get_preview_params() -> tuple[float, float, float, int, int] | None:
    density_result = state["density_result"]
    if density_result is not None:
        return grid_preview_params(density_result.grid)

    grid_min_x = float(dpg.get_value(PARAM_GRID_MIN_X))
    grid_min_y = float(dpg.get_value(PARAM_GRID_MIN_Y))
    cell_size = float(dpg.get_value(PARAM_CELL_SIZE))
    grid_width = int(dpg.get_value(PARAM_GRID_WIDTH))
    grid_height = int(dpg.get_value(PARAM_GRID_HEIGHT))

    if cell_size <= 0 or grid_width <= 0 or grid_height <= 0:
        return None

    return grid_min_x, grid_min_y, cell_size, grid_width, grid_height


def _get_coordinate_transform() -> CoordinateTransform | None:
    params = _get_preview_params()
    preview_width = int(state["image_width"])
    preview_height = int(state["image_height"])
    if params is None or preview_width <= 0 or preview_height <= 0:
        return None

    grid_min_x, grid_min_y, cell_size, grid_width, grid_height = params
    return CoordinateTransform(
        grid_min_x=grid_min_x,
        grid_min_y=grid_min_y,
        cell_size=cell_size,
        grid_width=grid_width,
        grid_height=grid_height,
        preview_width=preview_width,
        preview_height=preview_height,
    )


def _parse_report_float_value(value: str) -> float:
    value = value.strip().replace(" ", "")
    if "," in value and "." in value:
        value = value.replace(",", "")
    elif "," in value:
        parts = value.split(",")
        if len(parts) == 2 and 1 <= len(parts[1]) <= 6:
            value = ".".join(parts)
        else:
            value = value.replace(",", "")

    return float(value)


def _parse_report_int_value(value: str) -> int:
    value = value.strip().replace(" ", "").replace(",", "")
    return int(round(float(value)))


def _find_report_float(text: str, patterns: list[str], label: str) -> float:
    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if match:
            return _parse_report_float_value(match.group(1))

    raise ValueError(f"Field not found: {label}.")


def _find_report_int(text: str, patterns: list[str], label: str) -> int:
    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if match:
            return _parse_report_int_value(match.group(1))

    raise ValueError(f"Field not found: {label}.")


def _parse_report_text(text: str) -> tuple[float, float, float, int, int]:
    number = r"([-+]?\d[\d,]*(?:\.\d+)?)"
    normalized = text

    grid_min_x = _find_report_float(
        normalized,
        [
            rf"Bounding\s+box\s*:\s*.*?^\s*X\s*:\s*{number}\s*\.\.",
            rf"^\s*X\s*:\s*{number}\s*\.\.",
            rf"Bounding\s+box\s+X\s+min\s*[:=]\s*{number}",
        ],
        "Bounding box X min",
    )
    grid_min_y = _find_report_float(
        normalized,
        [
            rf"Bounding\s+box\s*:\s*.*?^\s*Y\s*:\s*{number}\s*\.\.",
            rf"^\s*Y\s*:\s*{number}\s*\.\.",
            rf"Bounding\s+box\s+Y\s+min\s*[:=]\s*{number}",
        ],
        "Bounding box Y min",
    )
    cell_size = _find_report_float(
        normalized,
        [
            rf"Density\s+grid\s*:\s*.*?^\s*Cell\s+size\s*[:=]\s*{number}",
            rf"Cell\s+size\s*[:=]\s*{number}",
        ],
        "Density grid Cell size",
    )
    width_cells = _find_report_int(
        normalized,
        [
            rf"Density\s+grid\s*:\s*.*?^\s*Width\s+cells\s*[:=]\s*{number}",
            rf"Width\s+cells\s*[:=]\s*{number}",
        ],
        "Width cells",
    )
    height_cells = _find_report_int(
        normalized,
        [
            rf"Density\s+grid\s*:\s*.*?^\s*Height\s+cells\s*[:=]\s*{number}",
            rf"Height\s+cells\s*[:=]\s*{number}",
        ],
        "Height cells",
    )

    if cell_size <= 0 or width_cells <= 0 or height_cells <= 0:
        raise ValueError("Report contains non-positive grid values.")

    return grid_min_x, grid_min_y, cell_size, width_cells, height_cells


def _preview_grid_params_are_default() -> bool:
    if state["density_result"] is not None:
        return False

    params = _get_preview_params()
    if params is None:
        return True

    grid_min_x, grid_min_y, _cell_size, _grid_width, _grid_height = params
    return (
        not bool(state["report_loaded"])
        and abs(grid_min_x) < 0.000001
        and abs(grid_min_y) < 0.000001
    )


def _warn_preview_grid_params_not_loaded() -> bool:
    if not _preview_grid_params_are_default():
        return False

    message = "Preview grid parameters are not loaded. Load report.txt first."
    _set_status(message)
    if dpg.does_item_exist(ROI_STATUS_TAG):
        dpg.set_value(ROI_STATUS_TAG, message)
    return True


def _display_layer_enabled(tag: str, default: bool = True) -> bool:
    if not dpg.does_item_exist(tag):
        return default

    return bool(dpg.get_value(tag))


def _scaled_image_size() -> tuple[int, int]:
    zoom = float(state["zoom"])
    width = max(1, int(round(int(state["image_width"]) * zoom)))
    height = max(1, int(round(int(state["image_height"]) * zoom)))

    return width, height


def _preview_to_draw_scales() -> tuple[float, float]:
    image_width = int(state["image_width"])
    image_height = int(state["image_height"])
    if image_width <= 0 or image_height <= 0:
        return 1.0, 1.0

    scaled_width, scaled_height = _scaled_image_size()
    return scaled_width / image_width, scaled_height / image_height


def _image_origin() -> tuple[float, float]:
    return float(state["pan_x"]), float(state["pan_y"])


def _screen_to_canvas(mouse_x: float, mouse_y: float) -> tuple[float, float] | None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return None

    canvas_left, canvas_top = dpg.get_item_rect_min(IMAGE_TAG)
    return mouse_x - canvas_left, mouse_y - canvas_top


def screen_to_image(mouse_x: float, mouse_y: float) -> tuple[float, float] | None:
    canvas_pos = _screen_to_canvas(mouse_x, mouse_y)
    if canvas_pos is None:
        return None

    canvas_x, canvas_y = canvas_pos
    origin_x, origin_y = _image_origin()
    zoom = float(state["zoom"])
    if zoom <= 0:
        return None

    image_width = int(state["image_width"])
    image_height = int(state["image_height"])
    if image_width <= 0 or image_height <= 0:
        return None

    draw_scale_x, draw_scale_y = _preview_to_draw_scales()
    image_edge_x = (canvas_x - origin_x) / draw_scale_x
    image_edge_y = (canvas_y - origin_y) / draw_scale_y
    if (
        image_edge_x < 0
        or image_edge_y < 0
        or image_edge_x >= image_width
        or image_edge_y >= image_height
    ):
        return None

    return image_edge_x - 0.5, image_edge_y - 0.5


def image_to_drawlist(image_x: float, image_y: float) -> tuple[float, float]:
    origin_x, origin_y = _image_origin()
    draw_scale_x, draw_scale_y = _preview_to_draw_scales()
    return (
        origin_x + (image_x + 0.5) * draw_scale_x,
        origin_y + (image_y + 0.5) * draw_scale_y,
    )


def image_to_world(
    image_x: float,
    image_y: float,
) -> tuple[float, float, float, float] | None:
    transform = _get_coordinate_transform()
    if transform is None:
        return None

    grid_x, grid_y = transform.preview_to_grid(image_x, image_y)
    world_x, world_y = transform.grid_to_world(grid_x, grid_y)
    return grid_x, grid_y, world_x, world_y


def screen_to_world(
    mouse_x: float,
    mouse_y: float,
) -> tuple[float, float, float, float, float, float] | None:
    image_pos = screen_to_image(mouse_x, mouse_y)
    if image_pos is None:
        return None

    image_x, image_y = image_pos
    world_pos = image_to_world(image_x, image_y)
    if world_pos is None:
        return None

    grid_x, grid_y, world_x, world_y = world_pos
    return image_x, image_y, grid_x, grid_y, world_x, world_y


def _set_zoom(
    zoom: float,
    anchor_canvas_pos: tuple[float, float] | None = None,
) -> None:
    old_zoom = float(state["zoom"])
    new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))

    if abs(new_zoom - old_zoom) < 0.0001:
        return

    if anchor_canvas_pos is None:
        anchor_canvas_pos = (CANVAS_WIDTH / 2.0, CANVAS_HEIGHT / 2.0)

    anchor_x, anchor_y = anchor_canvas_pos
    image_anchor_x = (anchor_x - float(state["pan_x"])) / old_zoom
    image_anchor_y = (anchor_y - float(state["pan_y"])) / old_zoom

    state["zoom"] = new_zoom
    state["pan_x"] = anchor_x - image_anchor_x * new_zoom
    state["pan_y"] = anchor_y - image_anchor_y * new_zoom
    dpg.set_value(ZOOM_TEXT_TAG, f"Zoom: {new_zoom * 100:.0f}%")
    _redraw_preview()


def _redraw_preview() -> None:
    texture_tag = state["texture_tag"]
    if not dpg.does_item_exist(texture_tag):
        return

    image_width = int(state["image_width"])
    image_height = int(state["image_height"])
    if image_width <= 0 or image_height <= 0:
        return

    if dpg.does_item_exist(IMAGE_TAG):
        dpg.delete_item(IMAGE_TAG)

    with dpg.drawlist(
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        parent="image_area",
        tag=IMAGE_TAG,
    ):
        scaled_width, scaled_height = _scaled_image_size()
        pan_x = float(state["pan_x"])
        pan_y = float(state["pan_y"])
        dpg.draw_image(
            texture_tag,
            (pan_x, pan_y),
            (pan_x + scaled_width, pan_y + scaled_height),
        )

    _redraw_selection_overlay()
    _redraw_processing_mask_overlay()
    _redraw_polygon_overlay()
    _redraw_mask_edits_overlay()
    _redraw_holes_overlay()
    _redraw_manual_hole_center_overlay()
    _redraw_contour_overlay()
    _redraw_mixed_contour_overlay()
    _redraw_brush_cursor_overlay()


def _redraw_selection_overlay() -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if dpg.does_item_exist(SELECTION_LAYER_TAG):
        dpg.delete_item(SELECTION_LAYER_TAG)

    if not active_working_area_is_visible(state) or not dpg.does_item_exist(
        SELECTION_TEXTURE_TAG
    ):
        return

    dpg.add_draw_layer(parent=IMAGE_TAG, tag=SELECTION_LAYER_TAG)

    scaled_width, scaled_height = _scaled_image_size()
    pan_x = float(state["pan_x"])
    pan_y = float(state["pan_y"])
    dpg.draw_image(
        SELECTION_TEXTURE_TAG,
        (pan_x, pan_y),
        (pan_x + scaled_width, pan_y + scaled_height),
        parent=SELECTION_LAYER_TAG,
    )


def _delete_processing_mask_texture() -> None:
    if dpg.does_item_exist(PROCESSING_MASK_LAYER_TAG):
        dpg.delete_item(PROCESSING_MASK_LAYER_TAG)
    if dpg.does_item_exist(PROCESSING_MASK_TEXTURE_TAG):
        dpg.delete_item(PROCESSING_MASK_TEXTURE_TAG)
    state["processing_mask_preview"] = None


def _update_processing_mask_texture(
    mask: np.ndarray,
) -> None:
    preview_width = int(state["image_width"])
    preview_height = int(state["image_height"])
    preview_mask = processing_mask_to_preview(
        mask,
        preview_width=preview_width,
        preview_height=preview_height,
    )

    overlay = np.zeros((preview_height, preview_width, 4), dtype=np.float32)
    overlay[preview_mask > 0] = (0.08, 0.72, 1.0, 0.34)

    texture_value = overlay.ravel().tolist()
    if dpg.does_item_exist(PROCESSING_MASK_TEXTURE_TAG):
        dpg.set_value(PROCESSING_MASK_TEXTURE_TAG, texture_value)
    else:
        with dpg.texture_registry():
            dpg.add_static_texture(
                width=preview_width,
                height=preview_height,
                default_value=texture_value,
                tag=PROCESSING_MASK_TEXTURE_TAG,
            )
    state["processing_mask_preview"] = preview_mask


def _redraw_processing_mask_overlay() -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if dpg.does_item_exist(PROCESSING_MASK_LAYER_TAG):
        dpg.delete_item(PROCESSING_MASK_LAYER_TAG)

    if not isinstance(state.get("mask_editing_session"), MaskEditingSession):
        return
    if state.get("processing_mask_preview") is None:
        return
    if not _display_layer_enabled(SHOW_PROCESSING_MASK_TAG, default=False):
        return
    if not dpg.does_item_exist(PROCESSING_MASK_TEXTURE_TAG):
        return

    dpg.add_draw_layer(parent=IMAGE_TAG, tag=PROCESSING_MASK_LAYER_TAG)
    scaled_width, scaled_height = _scaled_image_size()
    pan_x = float(state["pan_x"])
    pan_y = float(state["pan_y"])
    dpg.draw_image(
        PROCESSING_MASK_TEXTURE_TAG,
        (pan_x, pan_y),
        (pan_x + scaled_width, pan_y + scaled_height),
        parent=PROCESSING_MASK_LAYER_TAG,
    )


def _mouse_to_world() -> tuple[float, float, float, float, float, float] | None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return None

    if not dpg.is_item_hovered(IMAGE_TAG):
        return None

    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
    return screen_to_world(mouse_x, mouse_y)


def _world_to_drawlist(world_x: float, world_y: float) -> tuple[float, float] | None:
    transform = _get_coordinate_transform()
    if transform is None:
        return None

    return image_to_drawlist(*transform.world_to_preview(world_x, world_y))


def _update_polygon_points_text() -> None:
    points = state["polygon_points"]
    dpg.set_value(POLYGON_COUNT_TAG, f"Polygon points count: {len(points)}")

    if not points:
        dpg.set_value(POLYGON_LAST_TAG, "Last point: -")
        dpg.set_value(POLYGON_POINTS_TAG, "Polygon points: none")
        return

    last_x, last_y = points[-1]
    dpg.set_value(POLYGON_LAST_TAG, f"Last point: {last_x:.6f}, {last_y:.6f}")

    lines = ["Polygon points:"]
    for index, (x, y) in enumerate(points, start=1):
        lines.append(f"{index}: {x:.6f}, {y:.6f}")

    dpg.set_value(POLYGON_POINTS_TAG, "\n".join(lines))


def _undo_last_polygon_point(*, allow_hidden_draft: bool = False) -> bool:
    if (
        not allow_hidden_draft
        and state["selection_applied"]
        and not state["editing_overlay_visible"]
    ):
        dpg.set_value(ROI_STATUS_TAG, "Select Polygon ROI mode first.")
        return False

    points = state["polygon_points"]
    if not points:
        dpg.set_value(ROI_STATUS_TAG, "No polygon points to undo.")
        _update_polygon_points_text()
        return False

    points.pop()
    state["polygon_finished"] = False
    state["editing_overlay_visible"] = True
    dpg.set_value(POLYGON_OUTPUT_TAG, "")
    _update_polygon_points_text()
    _redraw_preview()

    if points:
        dpg.set_value(ROI_STATUS_TAG, "Last polygon point removed.")
    else:
        dpg.set_value(ROI_STATUS_TAG, "Polygon points cleared.")

    return True


def _redraw_polygon_overlay() -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if dpg.does_item_exist(POLYGON_LAYER_TAG):
        dpg.delete_item(POLYGON_LAYER_TAG)

    if not _display_layer_enabled(SHOW_ROI_OVERLAY_TAG):
        return

    dpg.add_draw_layer(parent=IMAGE_TAG, tag=POLYGON_LAYER_TAG)

    active_working_area = state.get("active_working_area")
    if active_working_area_is_visible(state) and isinstance(
        active_working_area, WorkingArea
    ):
        _draw_working_area_boundary(
            active_working_area,
            POLYGON_LAYER_TAG,
            color=(90, 255, 150, 245),
            thickness=3,
        )

    if not state["editing_overlay_visible"]:
        return

    show_rectangle_draft, show_polygon_draft = working_area_draft_visibility(state)
    rectangle_roi = state["rectangle_roi"]
    if show_rectangle_draft and rectangle_roi is not None:
        min_x, min_y, max_x, max_y = rectangle_roi
        pixel_a = _world_to_drawlist(min_x, min_y)
        pixel_b = _world_to_drawlist(max_x, max_y)
        if pixel_a is not None and pixel_b is not None:
            ax, ay = pixel_a
            bx, by = pixel_b
            dpg.draw_rectangle(
                (min(ax, bx), min(ay, by)),
                (max(ax, bx), max(ay, by)),
                color=(80, 220, 255, 230),
                thickness=2,
                parent=POLYGON_LAYER_TAG,
            )

    transient_anchor = state.get("roi_first_world")
    transient_current = state.get("roi_current_world")
    if show_rectangle_draft and transient_anchor is not None:
        anchor_point = _world_to_drawlist(*transient_anchor)
        current_point = _world_to_drawlist(*(transient_current or transient_anchor))
        if anchor_point is not None and current_point is not None:
            ax, ay = anchor_point
            cx, cy = current_point
            dpg.draw_rectangle(
                (min(ax, cx), min(ay, cy)),
                (max(ax, cx), max(ay, cy)),
                color=(80, 220, 255, 230),
                thickness=2,
                parent=POLYGON_LAYER_TAG,
            )
            dpg.draw_circle(
                anchor_point,
                4,
                color=(255, 255, 255, 255),
                fill=(80, 220, 255, 230),
                parent=POLYGON_LAYER_TAG,
            )

    pixel_points = []
    visible_polygon_points = state["polygon_points"] if show_polygon_draft else []
    for world_x, world_y in visible_polygon_points:
        pixel_point = _world_to_drawlist(world_x, world_y)
        if pixel_point is not None:
            pixel_points.append(pixel_point)

    if len(pixel_points) >= 2:
        for point_a, point_b in zip(pixel_points, pixel_points[1:]):
            dpg.draw_line(
                point_a,
                point_b,
                color=(255, 210, 70, 255),
                thickness=2,
                parent=POLYGON_LAYER_TAG,
            )

    if len(pixel_points) >= 3:
        dpg.draw_line(
            pixel_points[-1],
            pixel_points[0],
            color=(255, 210, 70, 180),
            thickness=2,
            parent=POLYGON_LAYER_TAG,
        )

    for index, (pixel_x, pixel_y) in enumerate(pixel_points, start=1):
        dpg.draw_circle(
            (pixel_x, pixel_y),
            4,
            color=(255, 80, 80, 255),
            fill=(255, 80, 80, 220),
            parent=POLYGON_LAYER_TAG,
        )
        dpg.draw_text(
            (pixel_x + 6, pixel_y + 6),
            str(index),
            color=(255, 255, 255, 255),
            size=14,
            parent=POLYGON_LAYER_TAG,
        )


def _draw_working_area_boundary(
    working_area: WorkingArea,
    layer_tag: str,
    *,
    color: tuple[int, int, int, int],
    thickness: int,
) -> None:
    if working_area.kind == "rectangle":
        assert working_area.rectangle_bounds is not None
        min_x, min_y, max_x, max_y = working_area.rectangle_bounds
        point_a = _world_to_drawlist(min_x, min_y)
        point_b = _world_to_drawlist(max_x, max_y)
        if point_a is None or point_b is None:
            return
        ax, ay = point_a
        bx, by = point_b
        dpg.draw_rectangle(
            (min(ax, bx), min(ay, by)),
            (max(ax, bx), max(ay, by)),
            color=color,
            thickness=thickness,
            parent=layer_tag,
        )
        return

    draw_points = [
        point
        for world_point in working_area.polygon_points
        if (point := _world_to_drawlist(*world_point)) is not None
    ]
    if len(draw_points) != len(working_area.polygon_points):
        return
    for point_a, point_b in zip(draw_points, draw_points[1:] + draw_points[:1]):
        dpg.draw_line(
            point_a,
            point_b,
            color=color,
            thickness=thickness,
            parent=layer_tag,
        )


def _world_radius_to_draw_radii(radius_mm: float) -> tuple[float, float]:
    transform = _get_coordinate_transform()
    if transform is None:
        return 1.0, 1.0

    radius_preview_x, radius_preview_y = transform.world_radius_to_preview(
        radius_mm
    )
    draw_scale_x, draw_scale_y = _preview_to_draw_scales()
    return (
        max(1.0, radius_preview_x * draw_scale_x),
        max(1.0, radius_preview_y * draw_scale_y),
    )


def _draw_world_radius_ellipse(
    center: tuple[float, float],
    radii: tuple[float, float],
    *,
    color: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
    thickness: float,
    parent: str,
) -> None:
    center_x, center_y = center
    radius_x, radius_y = radii
    dpg.draw_ellipse(
        (center_x - radius_x, center_y - radius_y),
        (center_x + radius_x, center_y + radius_y),
        color=color,
        fill=fill,
        thickness=thickness,
        parent=parent,
    )


def _get_brush_edit_mode() -> str:
    if not dpg.does_item_exist(BRUSH_MODE_TAG):
        return "remove"

    mode_label = str(dpg.get_value(BRUSH_MODE_TAG))
    if mode_label == "Add to mask":
        return "add"

    return "remove"


def _set_mask_edit_controls_enabled(enabled: bool) -> None:
    for tag in (
        MASK_BRUSH_BUTTON_TAG,
        MASK_CLEAR_BUTTON_TAG,
        BRUSH_MODE_TAG,
        BRUSH_SIZE_TAG,
        FIND_HOLES_BUTTON_TAG,
    ):
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, enabled=enabled)


def _active_mask_editing_session() -> MaskEditingSession | None:
    editing = state.get("mask_editing_session")
    result = state.get("contour_processing_result")
    density_result = state.get("density_result")
    working_area = get_active_working_area()
    if not isinstance(editing, MaskEditingSession):
        return None
    if not isinstance(result, PreliminaryContourSession):
        return None
    if editing.grid is not getattr(density_result, "grid", None):
        return None
    if editing.working_area != working_area:
        return None
    return editing


def _active_hole_detection_session() -> HoleDetectionSession | None:
    session = state.get("hole_detection_session")
    if hole_detection_session_is_current(
        session,
        density_session=state.get("density_result"),
        contour_session=state.get("contour_processing_result"),
        mask_editing_session=state.get("mask_editing_session"),
        coarse_mask_revision=int(state.get("coarse_mask_revision", 0)),
    ):
        assert isinstance(session, HoleDetectionSession)
        return session
    return None


def _invalidate_hole_detection(*, mark_outdated: bool = True) -> bool:
    state["hovered_hole_id"] = None
    invalidated = invalidate_hole_detection_state(
        state,
        mark_outdated=mark_outdated,
    )
    _update_hole_detection_info()
    return invalidated


def _mark_effective_mask_changed() -> None:
    state["coarse_mask_revision"] = int(
        state.get("coarse_mask_revision", 0)
    ) + 1
    _invalidate_hole_detection()


def _clear_mask_editing_state() -> None:
    _invalidate_hole_detection()
    state["mask_editing_session"] = None
    state["mask_edits"] = []
    state["last_brush_image"] = None
    state["last_brush_world"] = None
    state["brush_cursor_image"] = None
    state["active_brush_stroke_id"] = None
    state["next_brush_stroke_id"] = 1
    state["undo_history"] = [
        action
        for action in state["undo_history"]
        if action.get("kind") != "mask_stroke"
    ]
    if state.get("mode") == "mask_brush":
        state["mode"] = "rectangle"
    _set_mask_edit_controls_enabled(False)
    _update_mask_edits_count()
    _redraw_brush_cursor_overlay()


def _brush_edit_colors(mode: str) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if mode == "add":
        return (60, 170, 255, 220), (60, 150, 255, 90)

    return (255, 80, 80, 210), (255, 60, 60, 85)


def _redraw_mask_edits_overlay() -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if dpg.does_item_exist(MASK_EDITS_LAYER_TAG):
        dpg.delete_item(MASK_EDITS_LAYER_TAG)

    if not state["mask_edits"]:
        return

    dpg.add_draw_layer(parent=IMAGE_TAG, tag=MASK_EDITS_LAYER_TAG)

    for edit in state["mask_edits"]:
        edit_mode = str(edit.get("mode", "remove"))
        if edit_mode == "add":
            if not _display_layer_enabled(SHOW_MASK_ADD_EDITS_TAG):
                continue
        elif not _display_layer_enabled(SHOW_MASK_REMOVE_EDITS_TAG):
            continue

        pixel_point = _world_to_drawlist(edit["x"], edit["y"])
        if pixel_point is None:
            continue

        radii = _world_radius_to_draw_radii(edit["radius_mm"])
        color, fill = _brush_edit_colors(edit_mode)
        _draw_world_radius_ellipse(
            pixel_point,
            radii,
            color=color,
            fill=fill,
            thickness=2,
            parent=MASK_EDITS_LAYER_TAG,
        )


def _redraw_brush_cursor_overlay() -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if dpg.does_item_exist(BRUSH_CURSOR_LAYER_TAG):
        dpg.delete_item(BRUSH_CURSOR_LAYER_TAG)

    if state["mode"] != "mask_brush":
        return

    image_pos = state["brush_cursor_image"]
    if image_pos is None:
        return

    try:
        radius_mm = validate_brush_diameter(dpg.get_value(BRUSH_SIZE_TAG)) / 2.0
    except ValueError:
        return
    radii = _world_radius_to_draw_radii(radius_mm)
    center = image_to_drawlist(*image_pos)
    color, _fill = _brush_edit_colors(_get_brush_edit_mode())

    dpg.add_draw_layer(parent=IMAGE_TAG, tag=BRUSH_CURSOR_LAYER_TAG)
    _draw_world_radius_ellipse(
        center,
        radii,
        color=color,
        fill=(0, 0, 0, 0),
        thickness=2,
        parent=BRUSH_CURSOR_LAYER_TAG,
    )


def _update_brush_cursor_from_mouse() -> None:
    image_pos = None
    if (
        state["mode"] == "mask_brush"
        and dpg.does_item_exist(IMAGE_TAG)
        and dpg.is_item_hovered(IMAGE_TAG)
    ):
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        image_pos = screen_to_image(mouse_x, mouse_y)

    state["brush_cursor_image"] = image_pos
    _redraw_brush_cursor_overlay()


def _hole_overlay_colors(
    accepted: bool,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if accepted:
        return (80, 220, 120, 230), (80, 220, 120, 45)

    return (255, 140, 50, 230), (255, 110, 40, 45)


def _visible_hole_candidates(session: HoleDetectionSession) -> list:
    return [
        candidate
        for candidate in session.result.candidates
        if (
            candidate.accepted
            and _display_layer_enabled(SHOW_ACCEPTED_HOLES_TAG)
        )
        or (
            not candidate.accepted
            and _display_layer_enabled(SHOW_REJECTED_HOLES_TAG)
        )
    ]


def _hole_candidate_hit_region(candidate) -> HoleHitRegion | None:
    center = _world_to_drawlist(candidate.center_x, candidate.center_y)
    if center is None:
        return None
    radius_x, radius_y = _world_radius_to_draw_radii(candidate.radius)
    return HoleHitRegion(
        candidate_id=int(candidate.id),
        center_x=center[0],
        center_y=center[1],
        radius_x=radius_x,
        radius_y=radius_y,
    )


def _hole_at_screen(mouse_x: float, mouse_y: float) -> int | None:
    if state.get("pick_manual_hole_center"):
        return None
    if not dpg.does_item_exist(IMAGE_TAG) or not dpg.is_item_hovered(IMAGE_TAG):
        return None
    canvas_pos = _screen_to_canvas(mouse_x, mouse_y)
    if canvas_pos is None:
        return None
    session = _active_hole_detection_session()
    if session is None:
        return None

    regions = [
        region
        for candidate in _visible_hole_candidates(session)
        if (region := _hole_candidate_hit_region(candidate)) is not None
    ]
    return hit_test_hole_regions(regions, canvas_pos, tolerance_px=5.0)


def _redraw_holes_overlay() -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if dpg.does_item_exist(HOLES_LAYER_TAG):
        dpg.delete_item(HOLES_LAYER_TAG)

    if state.get("hole_overlay_source") == "in_memory":
        session = _active_hole_detection_session()
        if session is None or not session.result.candidates:
            return

        dpg.add_draw_layer(parent=IMAGE_TAG, tag=HOLES_LAYER_TAG)
        for candidate in session.result.candidates:
            if candidate.accepted:
                if not _display_layer_enabled(SHOW_ACCEPTED_HOLES_TAG):
                    continue
            elif not _display_layer_enabled(SHOW_REJECTED_HOLES_TAG):
                continue

            region = _hole_candidate_hit_region(candidate)
            if region is None:
                continue
            center = (region.center_x, region.center_y)
            radii = (region.radius_x, region.radius_y)
            color, fill = _hole_overlay_colors(candidate.accepted)
            _draw_world_radius_ellipse(
                center,
                radii,
                color=color,
                fill=fill,
                thickness=2,
                parent=HOLES_LAYER_TAG,
            )
            dpg.draw_text(
                (center[0] + radii[0] + 4, center[1] - 7),
                str(candidate.id),
                color=color,
                size=14,
                parent=HOLES_LAYER_TAG,
            )
            if state.get("hovered_hole_id") == int(candidate.id):
                _draw_world_radius_ellipse(
                    center,
                    (radii[0] + 4.0, radii[1] + 4.0),
                    color=(255, 245, 60, 255),
                    fill=(0, 0, 0, 0),
                    thickness=4,
                    parent=HOLES_LAYER_TAG,
                )
        return

    holes = state["holes"]
    if not holes:
        return

    dpg.add_draw_layer(parent=IMAGE_TAG, tag=HOLES_LAYER_TAG)

    max_diameter = float(dpg.get_value(MAX_DISPLAYED_HOLE_DIAMETER_TAG))
    show_oversized = _display_layer_enabled(
        SHOW_OVERSIZED_HOLES_TAG,
        default=False,
    )
    visible_group_ids = state["visible_hole_group_ids"]
    groups_by_id = {
        str(group.get("id", "")): group for group in state["hole_groups"]
    }

    for hole in holes:
        accepted = bool(hole.get("accepted", False))
        group_id = hole.get("group_id")
        group = groups_by_id.get(str(group_id)) if accepted and group_id else None
        if group is not None:
            displayed_diameter = float(group.get("diameter", 0.0))
            displayed_radius = float(group.get("radius", displayed_diameter / 2.0))
        else:
            displayed_diameter = float(
                hole.get("diameter", float(hole["radius"]) * 2.0)
            )
            displayed_radius = float(hole["radius"])

        if (
            not show_oversized
            and max_diameter > 0
            and displayed_diameter > max_diameter
        ):
            continue

        if accepted:
            if not _display_layer_enabled(SHOW_ACCEPTED_HOLES_TAG):
                continue
            if group is not None:
                if not visible_group_ids.get(str(group_id), False):
                    continue
            elif not _display_layer_enabled(SHOW_UNGROUPED_HOLES_TAG):
                continue
        elif not _display_layer_enabled(SHOW_REJECTED_HOLES_TAG):
            continue

        center = _world_to_drawlist(hole["center_x"], hole["center_y"])
        if center is None:
            continue

        radii = _world_radius_to_draw_radii(displayed_radius)
        color, fill = _hole_overlay_colors(accepted)
        _draw_world_radius_ellipse(
            center,
            radii,
            color=color,
            fill=fill,
            thickness=2,
            parent=HOLES_LAYER_TAG,
        )
        dpg.draw_text(
            (center[0] + radii[0] + 4, center[1] - 7),
            str(hole.get("id", "")),
            color=color,
            size=14,
            parent=HOLES_LAYER_TAG,
        )


def _redraw_manual_hole_center_overlay() -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if dpg.does_item_exist(MANUAL_HOLE_CENTER_LAYER_TAG):
        dpg.delete_item(MANUAL_HOLE_CENTER_LAYER_TAG)

    center_world = state["manual_hole_center_world"]
    if center_world is None:
        return

    pixel_point = _world_to_drawlist(center_world[0], center_world[1])
    if pixel_point is None:
        return

    x, y = pixel_point
    dpg.add_draw_layer(parent=IMAGE_TAG, tag=MANUAL_HOLE_CENTER_LAYER_TAG)
    dpg.draw_circle(
        (x, y),
        6,
        color=(80, 180, 255, 255),
        fill=(80, 180, 255, 80),
        thickness=2,
        parent=MANUAL_HOLE_CENTER_LAYER_TAG,
    )
    dpg.draw_line(
        (x - 10, y),
        (x + 10, y),
        color=(80, 180, 255, 255),
        thickness=2,
        parent=MANUAL_HOLE_CENTER_LAYER_TAG,
    )
    dpg.draw_line(
        (x, y - 10),
        (x, y + 10),
        color=(80, 180, 255, 255),
        thickness=2,
        parent=MANUAL_HOLE_CENTER_LAYER_TAG,
    )


def _redraw_contour_overlay() -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if dpg.does_item_exist(CONTOUR_LAYER_TAG):
        dpg.delete_item(CONTOUR_LAYER_TAG)

    contour_points = state["contour_points"]
    if not contour_points or not _display_layer_enabled(SHOW_CONTOUR_TAG):
        return

    draw_points = []
    for world_x, world_y in contour_points:
        point = _world_to_drawlist(world_x, world_y)
        if point is not None:
            draw_points.append(point)

    if len(draw_points) < 2:
        return

    dpg.add_draw_layer(parent=IMAGE_TAG, tag=CONTOUR_LAYER_TAG)
    for point_a, point_b in zip(draw_points, draw_points[1:]):
        dpg.draw_line(
            point_a,
            point_b,
            color=(80, 240, 255, 245),
            thickness=2,
            parent=CONTOUR_LAYER_TAG,
        )

    if len(draw_points) >= 3:
        dpg.draw_line(
            draw_points[-1],
            draw_points[0],
            color=(80, 240, 255, 245),
            thickness=2,
            parent=CONTOUR_LAYER_TAG,
        )


def _redraw_mixed_contour_overlay() -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if dpg.does_item_exist(MIXED_CONTOUR_LAYER_TAG):
        dpg.delete_item(MIXED_CONTOUR_LAYER_TAG)

    elements = state["mixed_contour_elements"]
    if not elements:
        return

    show_lines = _display_layer_enabled(SHOW_MIXED_CONTOUR_LINES_TAG)
    show_gaps = _display_layer_enabled(SHOW_MIXED_CONTOUR_GAPS_TAG)
    if not show_lines and not show_gaps:
        return

    dpg.add_draw_layer(parent=IMAGE_TAG, tag=MIXED_CONTOUR_LAYER_TAG)
    for element in elements:
        element_type = str(element.get("type", "")).upper()
        if element_type == "LINE":
            if not show_lines:
                continue
            start = element.get("start", {})
            end = element.get("end", {})
            point_a = _world_to_drawlist(float(start["x"]), float(start["y"]))
            point_b = _world_to_drawlist(float(end["x"]), float(end["y"]))
            if point_a is None or point_b is None:
                continue

            dpg.draw_line(
                point_a,
                point_b,
                color=(255, 245, 90, 255),
                thickness=3,
                parent=MIXED_CONTOUR_LAYER_TAG,
            )
        elif element_type == "POLYLINE":
            if not show_gaps:
                continue
            draw_points = []
            for point in element.get("points", []):
                draw_point = _world_to_drawlist(float(point["x"]), float(point["y"]))
                if draw_point is not None:
                    draw_points.append(draw_point)

            for point_a, point_b in zip(draw_points, draw_points[1:]):
                dpg.draw_line(
                    point_a,
                    point_b,
                    color=(255, 120, 220, 235),
                    thickness=2,
                    parent=MIXED_CONTOUR_LAYER_TAG,
                )


def _update_hole_hover_from_mouse(*, force: bool = False) -> int | None:
    candidate_id = None
    drag_in_progress = any(
        bool(gesture.get("dragged"))
        for gesture in state.get("mouse_gestures", {}).values()
    )
    if not drag_in_progress and not state.get("pick_manual_hole_center"):
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        candidate_id = _hole_at_screen(mouse_x, mouse_y)

    if not force and state.get("hovered_hole_id") == candidate_id:
        return candidate_id

    state["hovered_hole_id"] = candidate_id
    if dpg.does_item_exist(HOLE_HOVER_INFO_TAG):
        session = _active_hole_detection_session()
        candidate = None
        if session is not None and candidate_id is not None:
            candidate = next(
                (
                    item
                    for item in session.result.candidates
                    if int(item.id) == int(candidate_id)
                ),
                None,
            )
        if candidate is None:
            text = "Hover a hole candidate for details."
        else:
            status = "Accepted" if candidate.accepted else "Rejected"
            text = (
                f"Hole #{candidate.id} | Diameter: {candidate.diameter:.3f} mm\n"
                f"Status: {status}"
            )
            if candidate.reject_reason:
                text += f" | Reason: {candidate.reject_reason}"
        dpg.set_value(HOLE_HOVER_INFO_TAG, text)
    _redraw_holes_overlay()
    return candidate_id


def _gesture_distance_squared(
    start: tuple[float, float],
    current: tuple[float, float],
) -> float:
    dx = float(current[0]) - float(start[0])
    dy = float(current[1]) - float(start[1])
    return dx * dx + dy * dy


def _gesture_is_drag(
    start: tuple[float, float],
    current: tuple[float, float],
) -> bool:
    return _gesture_distance_squared(start, current) > MOUSE_DRAG_THRESHOLD_PX**2


def _mouse_down_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    button = str(_user_data or "")
    if button not in {"left", "right", "middle"}:
        return
    if not dpg.does_item_exist(IMAGE_TAG) or not dpg.is_item_hovered(IMAGE_TAG):
        return

    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
    coords = screen_to_world(mouse_x, mouse_y)
    start_world = None if coords is None else (coords[-2], coords[-1])
    state.setdefault("mouse_gestures", {})[button] = {
        "start_screen": (mouse_x, mouse_y),
        "last_screen": (mouse_x, mouse_y),
        "start_world": start_world,
        "dragged": False,
        "rectangle_drag_started": False,
        "brush_last_screen": None,
        "pan_last_screen": (mouse_x, mouse_y),
    }
    if button == "left":
        state["last_brush_image"] = None
        state["last_brush_world"] = None
        state["active_brush_stroke_id"] = None


def _update_mouse_gestures(current: tuple[float, float]) -> None:
    for gesture in state.get("mouse_gestures", {}).values():
        gesture["last_screen"] = current
        if not gesture["dragged"] and _gesture_is_drag(
            gesture["start_screen"],
            current,
        ):
            gesture["dragged"] = True


def _update_rectangle_drag_from_mouse(
    current: tuple[float, float],
) -> None:
    gesture = state.get("mouse_gestures", {}).get("left")
    if (
        state.get("mode") != "rectangle"
        or gesture is None
        or not gesture.get("dragged")
    ):
        return

    world = gesture.get("start_world")
    if world is None:
        return
    if not gesture.get("rectangle_drag_started"):
        if state.get("roi_first_world") is None:
            begin_rectangle_draft(state, world)
        gesture["rectangle_drag_started"] = True

    coords = screen_to_world(*current)
    if coords is not None and update_rectangle_transient(
        state,
        (coords[-2], coords[-1]),
    ):
        _redraw_polygon_overlay()


def _apply_mask_brush_at_screen(
    current: tuple[float, float],
    *,
    initial_world: tuple[float, float] | None = None,
) -> bool:
    if state.get("mode") != "mask_brush":
        return False
    if _warn_preview_grid_params_not_loaded():
        return False

    coords = screen_to_world(*current)
    if coords is None:
        return False
    editing = _active_mask_editing_session()
    if editing is None:
        _set_status("Find a preliminary contour before editing its mask.")
        return False

    image_x, image_y, _grid_x, _grid_y, world_x, world_y = coords
    try:
        brush_diameter_mm = validate_brush_diameter(
            dpg.get_value(BRUSH_SIZE_TAG)
        )
    except ValueError as exc:
        _set_status(str(exc))
        return False

    stroke_id = state["active_brush_stroke_id"]
    if stroke_id is None:
        stroke_id = int(state["next_brush_stroke_id"])
        state["active_brush_stroke_id"] = stroke_id
        state["next_brush_stroke_id"] = stroke_id + 1
    start_world = state["last_brush_world"] or initial_world or (world_x, world_y)
    edits = mask_edits_for_world_segment(
        editing.grid,
        start_world,
        (world_x, world_y),
        mode=_get_brush_edit_mode(),
        brush_diameter_mm=brush_diameter_mm,
        stroke_id=stroke_id,
    )
    changed = False
    for edit in edits:
        if editing.apply_edit(edit):
            changed = True
    state["mask_edits"].extend(edits)
    state["last_brush_image"] = (image_x, image_y)
    state["last_brush_world"] = (world_x, world_y)
    _update_mask_edits_count()
    _update_last_brush_debug()
    if changed:
        _update_processing_mask_texture(editing.edited_mask)
        _update_contour_info()
        _redraw_preview()
        _set_status("Mask modified. Release the mouse to update contour.")
    return True


def _update_mask_brush_from_mouse() -> None:
    gesture = state.get("mouse_gestures", {}).get("left")
    if (
        state.get("mode") != "mask_brush"
        or gesture is None
        or not gesture.get("dragged")
    ):
        return
    current = tuple(gesture["last_screen"])
    if gesture.get("brush_last_screen") == current:
        return
    if _apply_mask_brush_at_screen(
        current,
        initial_world=gesture.get("start_world"),
    ):
        gesture["brush_last_screen"] = current


def _mouse_move_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    mouse_pos = tuple(dpg.get_mouse_pos(local=False))
    _update_mouse_gestures(mouse_pos)
    _update_pan_from_mouse()
    _update_rectangle_drag_from_mouse(mouse_pos)
    _update_mask_brush_from_mouse()
    _update_brush_cursor_from_mouse()
    _update_hole_hover_from_mouse()
    _update_debug_coords()

    coords = _mouse_to_world()
    if coords is None:
        dpg.set_value(COORDS_TAG, "World X: - | World Y: -")
        return

    _pixel_x, _pixel_y, _grid_x, _grid_y, world_x, world_y = coords
    if update_rectangle_transient(state, (world_x, world_y)):
        _redraw_polygon_overlay()
    dpg.set_value(
        COORDS_TAG,
        f"World X: {world_x:.6f} | World Y: {world_y:.6f}",
    )


def _finish_active_mask_stroke() -> bool:
    editing = _active_mask_editing_session()
    if editing is None:
        return False
    stroke_id = state.get("active_brush_stroke_id")
    finished = editing.finish_stroke()
    processing_result = state.get("contour_processing_result")
    if finished and isinstance(processing_result, PreliminaryContourSession):
        state["undo_history"].append(
            {
                "kind": "mask_stroke",
                "stroke_id": stroke_id,
                "density_session": state.get("density_result"),
                "mask_editing_session": editing,
            }
        )
        editing.update_contour_stale(processing_result.masks.contour_mask)
        if _refresh_contour_from_current_edits():
            _set_status("Mask edit applied; preliminary contour updated.")
    return finished


def _update_mask_edits_count() -> None:
    if dpg.does_item_exist(BRUSH_EDITS_COUNT_TAG):
        stroke_ids = {
            edit.get("stroke_id", index)
            for index, edit in enumerate(state["mask_edits"])
        }
        dpg.set_value(
            BRUSH_EDITS_COUNT_TAG,
            f"Mask edit strokes: {len(stroke_ids)}",
        )


def _update_holes_stats() -> None:
    if not dpg.does_item_exist(HOLES_STATS_TAG):
        return

    holes = state["holes"]
    groups = state["hole_groups"]
    accepted_count = sum(1 for hole in holes if bool(hole.get("accepted", False)))
    rejected_count = len(holes) - accepted_count
    dpg.set_value(
        HOLES_STATS_TAG,
        "holes total: {}\naccepted: {}\nrejected: {}\ngroups count: {}".format(
            len(holes),
            accepted_count,
            rejected_count,
            len(groups),
        ),
    )


def _update_hole_detection_info() -> None:
    session = _active_hole_detection_session()
    if not dpg.does_item_exist(HOLE_DETECTION_INFO_TAG):
        return

    if session is not None:
        dpg.set_value(
            HOLE_DETECTION_INFO_TAG,
            "Hole candidates: {}\nAccepted: {} | Rejected: {}\nGroups: {}".format(
                len(session.result.candidates),
                session.result.accepted_count,
                session.rejected_count,
                len(session.result.groups),
            ),
        )
    elif state.get("holes_outdated"):
        dpg.set_value(
            HOLE_DETECTION_INFO_TAG,
            "Holes outdated. Press Find holes.",
        )
    else:
        dpg.set_value(HOLE_DETECTION_INFO_TAG, "Holes: not searched")


def _selected_hole_detection_parameters() -> HoleDetectionParameters:
    return validate_hole_detection_parameters(
        dpg.get_value(HOLE_MIN_DIAMETER_TAG),
        dpg.get_value(HOLE_MAX_DIAMETER_TAG),
        dpg.get_value(HOLE_MIN_CIRCULARITY_TAG),
        dpg.get_value(HOLE_MAX_ERROR_RATIO_TAG),
    )


def _hole_detection_quality_settings_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    try:
        _selected_hole_detection_parameters()
    except ValueError as exc:
        _set_status(f"Invalid hole detection settings: {exc}")
        return

    _invalidate_hole_detection()
    _redraw_preview()
    _set_status("Hole detection settings changed. Press Find holes to apply.")


def _find_holes_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    contour_session = state.get("contour_processing_result")
    editing_session = _active_mask_editing_session()
    if (
        not isinstance(contour_session, PreliminaryContourSession)
        or editing_session is None
    ):
        _set_status("Find a preliminary contour before finding holes.")
        return

    try:
        parameters = _selected_hole_detection_parameters()
    except ValueError as exc:
        _set_status(f"Invalid hole detection settings: {exc}")
        return

    _set_status("Finding hole candidates...")
    try:
        session = find_holes_for_current_mask(
            contour_session,
            editing_session,
            density_session=state.get("density_result"),
            coarse_mask_revision=int(state.get("coarse_mask_revision", 0)),
            parameters=parameters,
        )
    except Exception as exc:
        _set_status(f"Could not find holes: {exc}")
        return

    state["hole_detection_session"] = session
    state["holes_outdated"] = False
    state["hole_overlay_source"] = "in_memory"
    _update_hole_detection_info()
    _redraw_preview()
    _set_status(
        "Hole candidates ready: "
        f"total={len(session.result.candidates)}, "
        f"accepted={session.result.accepted_count}, "
        f"rejected={session.rejected_count}."
    )


def _hole_group_visibility_callback(sender, _app_data=None, user_data=None) -> None:
    group_id = str(user_data)
    state["visible_hole_group_ids"][group_id] = bool(dpg.get_value(sender))
    _redraw_preview()


def _update_hole_groups_display() -> None:
    if not dpg.does_item_exist(HOLE_GROUPS_CONTAINER_TAG):
        return

    dpg.delete_item(HOLE_GROUPS_CONTAINER_TAG, children_only=True)
    groups = state["hole_groups"]
    if not groups:
        dpg.add_text(
            "No hole groups loaded.",
            parent=HOLE_GROUPS_CONTAINER_TAG,
            wrap=285,
        )
        return

    for group in groups:
        group_id = str(group.get("id", ""))
        visible = state["visible_hole_group_ids"].get(group_id, True)
        with dpg.group(parent=HOLE_GROUPS_CONTAINER_TAG):
            dpg.add_checkbox(
                label=f"Show group {group_id}",
                default_value=visible,
                callback=_hole_group_visibility_callback,
                user_data=group_id,
            )
            dpg.add_text(
                "{} | diameter={:.3f} | count={} | enabled={}".format(
                    str(group.get("name", "")),
                    float(group.get("diameter", 0.0)),
                    int(group.get("count", 0)),
                    bool(group.get("enabled", True)),
                ),
                wrap=285,
            )
            dpg.add_separator()


def _recount_hole_group_counts() -> None:
    accepted_counts: dict[str, int] = {}
    for hole in state["holes"]:
        if not bool(hole.get("accepted", False)):
            continue

        group_id = hole.get("group_id")
        if group_id is None:
            continue

        group_id = str(group_id)
        accepted_counts[group_id] = accepted_counts.get(group_id, 0) + 1

    for group in state["hole_groups"]:
        group_id = str(group.get("id", ""))
        group["count"] = accepted_counts.get(group_id, 0)


def _update_hole_group_target_combo() -> None:
    group_ids = [str(group.get("id", "")) for group in state["hole_groups"]]
    for tag in (MOVE_HOLE_TARGET_GROUP_TAG, EDIT_GROUP_TARGET_TAG):
        if not dpg.does_item_exist(tag):
            continue

        current_value = str(dpg.get_value(tag) or "")
        if current_value not in group_ids:
            current_value = group_ids[0] if group_ids else ""

        dpg.configure_item(tag, items=group_ids)
        dpg.set_value(tag, current_value)


def _get_selected_hole() -> tuple[int | None, dict | None]:
    hole_id = int(dpg.get_value(MOVE_HOLE_ID_TAG))
    hole = next(
        (item for item in state["holes"] if int(item.get("id", -1)) == hole_id),
        None,
    )
    if hole is None:
        _set_status(f"Hole not found: {hole_id}")
        return hole_id, None

    return hole_id, hole


def _refresh_hole_views() -> None:
    state["hole_overlay_source"] = "legacy"
    _recount_hole_group_counts()
    _update_hole_groups_display()
    _update_holes_stats()
    _redraw_preview()


def _set_manual_hole_status_by_id(hole_id: int, *, accepted: bool) -> bool:
    session = _active_hole_detection_session()
    if session is not None:
        try:
            changed = apply_manual_hole_status(
                session,
                hole_id,
                accepted=accepted,
            )
        except KeyError:
            _set_status(f"Hole not found: {hole_id}")
            return False

        action = "accepted" if accepted else "rejected"
        if changed:
            _update_hole_detection_info()
            _redraw_preview()
            _set_status(f"Hole {hole_id} {action} manually.")
        else:
            _set_status(f"Hole {hole_id} is already {action}.")
        return changed

    hole = next(
        (item for item in state["holes"] if int(item.get("id", -1)) == hole_id),
        None,
    )
    if hole is None:
        _set_status(f"Hole not found: {hole_id}")
        return False

    if accepted:
        target_group_id = str(
            dpg.get_value(MOVE_HOLE_TARGET_GROUP_TAG) or ""
        ).strip()
        if not hole.get("group_id") and target_group_id:
            group = next(
                (
                    item
                    for item in state["hole_groups"]
                    if str(item.get("id", "")) == target_group_id
                ),
                None,
            )
            if group is None:
                _set_status(f"Hole group not found: {target_group_id}")
                return False
            hole["group_id"] = target_group_id

        hole["accepted"] = True
        hole["enabled"] = True
        hole["reject_reason"] = ""
    else:
        hole["accepted"] = False
        hole["enabled"] = False
        if not hole.get("reject_reason"):
            hole["reject_reason"] = "manual_reject"

    _refresh_hole_views()
    action = "accepted" if accepted else "rejected"
    _set_status(f"Hole {hole_id} {action}.")
    return True


def _move_hole_to_group_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    hole_id, hole = _get_selected_hole()
    if hole is None:
        return

    target_group_id = str(dpg.get_value(MOVE_HOLE_TARGET_GROUP_TAG) or "").strip()

    group = next(
        (
            item
            for item in state["hole_groups"]
            if str(item.get("id", "")) == target_group_id
        ),
        None,
    )
    if group is None:
        _set_status(f"Hole group not found: {target_group_id}")
        return

    hole["group_id"] = target_group_id
    _refresh_hole_views()
    _set_status(f"Hole {hole_id} moved to group {target_group_id}.")


def _accept_selected_hole_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    _set_manual_hole_status_by_id(
        int(dpg.get_value(MOVE_HOLE_ID_TAG)),
        accepted=True,
    )


def _reject_selected_hole_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    _set_manual_hole_status_by_id(
        int(dpg.get_value(MOVE_HOLE_ID_TAG)),
        accepted=False,
    )


def _manual_hole_pick_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    enabled = bool(dpg.get_value(MANUAL_HOLE_PICK_TAG))
    state["pick_manual_hole_center"] = enabled
    if enabled:
        _set_status("Pick hole center: click preview.")
    else:
        _set_status("Pick hole center disabled.")


def _finish_manual_hole_center_pick(world_x: float, world_y: float) -> None:
    state["manual_hole_center_world"] = (world_x, world_y)
    state["pick_manual_hole_center"] = False
    if dpg.does_item_exist(MANUAL_HOLE_PICK_TAG):
        dpg.set_value(MANUAL_HOLE_PICK_TAG, False)
    dpg.set_value(MANUAL_HOLE_X_TAG, world_x)
    dpg.set_value(MANUAL_HOLE_Y_TAG, world_y)
    _redraw_preview()
    _set_status(
        f"Manual hole center picked: x={world_x:.6f}, y={world_y:.6f}"
    )


def _add_manual_hole_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    try:
        x = float(dpg.get_value(MANUAL_HOLE_X_TAG))
        y = float(dpg.get_value(MANUAL_HOLE_Y_TAG))
        diameter = float(dpg.get_value(MANUAL_HOLE_DIAMETER_TAG))
    except (TypeError, ValueError):
        _set_status("Manual hole X, Y and Diameter mm must be numbers.")
        return

    if diameter <= 0:
        _set_status("Manual hole Diameter mm must be greater than 0.")
        return

    target_group_id = str(dpg.get_value(MOVE_HOLE_TARGET_GROUP_TAG) or "").strip()
    group_id = None
    if target_group_id:
        group = next(
            (
                item
                for item in state["hole_groups"]
                if str(item.get("id", "")) == target_group_id
            ),
            None,
        )
        if group is None:
            _set_status(f"Hole group not found: {target_group_id}")
            return

        group_id = target_group_id

    max_id = 0
    for hole in state["holes"]:
        try:
            max_id = max(max_id, int(hole.get("id", 0)))
        except (TypeError, ValueError):
            continue

    new_hole_id = max_id + 1
    state["holes"].append(
        {
            "id": new_hole_id,
            "accepted": True,
            "enabled": True,
            "reject_reason": "",
            "center_x": x,
            "center_y": y,
            "radius": diameter / 2.0,
            "diameter": diameter,
            "group_id": group_id,
            "source": "manual",
        }
    )

    if dpg.does_item_exist(MOVE_HOLE_ID_TAG):
        dpg.set_value(MOVE_HOLE_ID_TAG, new_hole_id)

    state["manual_hole_center_world"] = None
    state["pick_manual_hole_center"] = False
    if dpg.does_item_exist(MANUAL_HOLE_PICK_TAG):
        dpg.set_value(MANUAL_HOLE_PICK_TAG, False)

    _refresh_hole_views()
    _set_status(f"Manual hole {new_hole_id} added.")


def _apply_group_diameter_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    group_id = str(dpg.get_value(EDIT_GROUP_TARGET_TAG) or "").strip()
    diameter = float(dpg.get_value(EDIT_GROUP_DIAMETER_TAG))

    group = next(
        (
            item
            for item in state["hole_groups"]
            if str(item.get("id", "")) == group_id
        ),
        None,
    )
    if group is None:
        _set_status(f"Hole group not found: {group_id}")
        return

    if diameter <= 0:
        _set_status("New group diameter must be greater than 0.")
        return

    diameter_text = _format_cli_float(diameter)
    group["diameter"] = diameter
    group["radius"] = diameter / 2.0
    group["name"] = f"Ø{diameter_text}"
    _update_hole_groups_display()
    _redraw_preview()
    _set_status(f"Group {group_id} diameter updated to {diameter_text} mm.")


def _update_contour_info() -> None:
    if not dpg.does_item_exist(CONTOUR_INFO_TAG):
        return

    processing_result = state.get("contour_processing_result")
    if isinstance(processing_result, PreliminaryContourSession):
        editing = state.get("mask_editing_session")
        state_text = "editing in progress" if (
            isinstance(editing, MaskEditingSession) and editing.contour_stale
        ) else "ready"
        dpg.set_value(
            CONTOUR_INFO_TAG,
            f"Preliminary contour: {state_text}\n"
            f"Points: {processing_result.contour.point_count}\n"
            f"Threshold: {processing_result.masks.threshold_result.threshold:.3f}\n"
            "Fill holes max area: "
            f"{processing_result.parameters.fill_holes_area} cells",
        )
        return

    contour_file = str(state["contour_file"])
    if contour_file:
        text = (
            "Preliminary contour: not built\n"
            f"Legacy contour file: {contour_file}\n"
            f"Points: {len(state['contour_points'])}"
        )
    elif state["contour_points"]:
        text = (
            "Preliminary contour: not active\n"
            f"Unbound contour points: {len(state['contour_points'])}"
        )
    else:
        text = "Preliminary contour: not built\nPoints: 0"
    dpg.set_value(
        CONTOUR_INFO_TAG,
        text,
    )


def _selected_contour_parameters() -> PreliminaryContourParameters:
    threshold_mode = dpg.get_value(CONTOUR_THRESHOLD_MODE_TAG)
    manual_threshold = None
    if str(threshold_mode).strip().lower() == "manual":
        manual_threshold = dpg.get_value(CONTOUR_MANUAL_THRESHOLD_TAG)
    return preliminary_contour_parameters_for_threshold(
        threshold_mode,
        manual_threshold,
        keep_largest=bool(dpg.get_value(HOLE_KEEP_LARGEST_TAG)),
        fill_holes_area=dpg.get_value(CONTOUR_FILL_HOLES_AREA_TAG),
    )


def _contour_fill_holes_settings_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    try:
        _selected_contour_parameters()
    except ValueError as exc:
        _set_status(f"Invalid contour settings: {exc}")
        return
    _set_status("Contour settings changed. Press Find contour to apply.")


def _contour_threshold_settings_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    threshold_mode = str(dpg.get_value(CONTOUR_THRESHOLD_MODE_TAG)).strip().lower()
    if dpg.does_item_exist(CONTOUR_MANUAL_THRESHOLD_TAG):
        dpg.configure_item(
            CONTOUR_MANUAL_THRESHOLD_TAG,
            enabled=threshold_mode == "manual",
        )
    if isinstance(state.get("contour_processing_result"), PreliminaryContourSession):
        if _refresh_contour_for_settings(preserve_edits=True):
            label = "Auto" if threshold_mode == "auto" else "Manual"
            _set_status(f"Preliminary contour updated for {label} threshold.")
        return
    _set_status("Contour settings changed. Press Find contour to apply.")


def _manual_threshold_commit_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    if str(dpg.get_value(CONTOUR_THRESHOLD_MODE_TAG)).strip().lower() != "manual":
        return
    if _refresh_contour_for_settings(preserve_edits=True):
        _set_status("Preliminary contour updated for Manual threshold.")


def _keep_largest_component_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    if isinstance(state.get("contour_processing_result"), PreliminaryContourSession):
        if _refresh_contour_for_settings(preserve_edits=True):
            enabled = bool(dpg.get_value(HOLE_KEEP_LARGEST_TAG))
            label = "enabled" if enabled else "disabled"
            _set_status(f"Keep largest component {label}; contour updated.")
        return
    _set_status("Contour settings changed. Press Find contour to apply.")


def _find_contour_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    density_result = state.get("density_result")
    if not isinstance(density_result, DensityProcessingResult):
        _set_status("Build a density map before finding a preliminary contour.")
        return

    working_area = get_active_working_area()
    try:
        parameters = _selected_contour_parameters()
    except ValueError as exc:
        _set_status(f"Invalid contour threshold: {exc}")
        return

    _set_status("Building preliminary contour...")
    try:
        existing_edits = list(state["mask_edits"])
        processing_result, editing_session = rebase_preliminary_contour_with_edits(
            density_result.grid,
            working_area,
            existing_edits,
            density_session=density_result,
            parameters=parameters,
        )
        _update_processing_mask_texture(editing_session.edited_mask)
    except Exception as exc:
        _set_status(f"Could not build preliminary contour: {exc}")
        return

    state["contour_processing_result"] = processing_result
    state["mask_editing_session"] = editing_session
    state["mask_edits"] = existing_edits
    state["contour_points"] = [
        (float(x), float(y)) for x, y in processing_result.contour.contour_world
    ]
    state["contour_file"] = ""
    _mark_effective_mask_changed()
    for action in state["undo_history"]:
        if action.get("kind") == "mask_stroke":
            action["mask_editing_session"] = editing_session
    _set_mask_edit_controls_enabled(True)
    _update_mask_edits_count()
    _update_contour_info()
    _redraw_preview()
    _set_status(
        "Preliminary contour ready: "
        f"{processing_result.contour.point_count} points."
    )


def _refresh_contour_from_current_edits() -> bool:
    processing_result = state.get("contour_processing_result")
    editing_session = _active_mask_editing_session()
    if not isinstance(processing_result, PreliminaryContourSession) or editing_session is None:
        return False
    _mark_effective_mask_changed()
    try:
        rebuilt = rebuild_preliminary_contour_from_edited_mask(
            processing_result,
            editing_session,
        )
    except Exception as exc:
        _set_status(f"Could not update preliminary contour: {exc}")
        return False

    state["contour_processing_result"] = rebuilt
    state["contour_points"] = [
        (float(x), float(y)) for x, y in rebuilt.contour.contour_world
    ]
    state["contour_file"] = ""
    _update_processing_mask_texture(editing_session.edited_mask)
    _update_contour_info()
    _redraw_preview()
    return True


def _refresh_contour_for_settings(*, preserve_edits: bool) -> bool:
    density_result = state.get("density_result")
    current_result = state.get("contour_processing_result")
    if not isinstance(density_result, DensityProcessingResult) or not isinstance(
        current_result,
        PreliminaryContourSession,
    ):
        return False
    try:
        parameters = _selected_contour_parameters()
        edits = list(state["mask_edits"]) if preserve_edits else []
        result, editing = rebase_preliminary_contour_with_edits(
            density_result.grid,
            get_active_working_area(),
            edits,
            density_session=density_result,
            parameters=parameters,
        )
    except Exception as exc:
        _set_status(f"Could not update preliminary contour: {exc}")
        return False

    state["contour_processing_result"] = result
    state["mask_editing_session"] = editing
    state["mask_edits"] = edits
    state["contour_points"] = [
        (float(x), float(y)) for x, y in result.contour.contour_world
    ]
    state["contour_file"] = ""
    _mark_effective_mask_changed()
    for action in state["undo_history"]:
        if action.get("kind") == "mask_stroke":
            action["mask_editing_session"] = editing
    _update_processing_mask_texture(editing.edited_mask)
    _set_mask_edit_controls_enabled(True)
    _update_mask_edits_count()
    _update_contour_info()
    _redraw_preview()
    return True


def _mixed_contour_counts() -> tuple[int, int, int]:
    elements = state["mixed_contour_elements"]
    lines_count = sum(1 for item in elements if str(item.get("type", "")).upper() == "LINE")
    gaps_count = sum(
        1 for item in elements if str(item.get("type", "")).upper() == "POLYLINE"
    )
    return len(elements), lines_count, gaps_count


def _update_mixed_contour_info() -> None:
    if not dpg.does_item_exist(MIXED_CONTOUR_INFO_TAG):
        return

    total_count, lines_count, gaps_count = _mixed_contour_counts()
    mixed_file = str(state["mixed_contour_file"]) or "-"
    dpg.set_value(
        MIXED_CONTOUR_INFO_TAG,
        f"Mixed contour file: {mixed_file}\n"
        f"Elements: {total_count}\n"
        f"Lines: {lines_count}\n"
        f"Polyline gaps: {gaps_count}",
    )


def _update_demo_summary_info() -> None:
    if not dpg.does_item_exist(DEMO_SUMMARY_INFO_TAG):
        return

    summary_file = str(state["demo_summary_file"]) or "-"
    dpg.set_value(DEMO_SUMMARY_INFO_TAG, f"Demo summary file: {summary_file}")


def _reset_report_params() -> None:
    state["report_loaded"] = False
    if dpg.does_item_exist(PARAM_GRID_MIN_X):
        dpg.set_value(PARAM_GRID_MIN_X, 0.0)
    if dpg.does_item_exist(PARAM_GRID_MIN_Y):
        dpg.set_value(PARAM_GRID_MIN_Y, 0.0)
    if dpg.does_item_exist(PARAM_CELL_SIZE):
        dpg.set_value(PARAM_CELL_SIZE, 1.0)
    if dpg.does_item_exist(PARAM_GRID_WIDTH):
        dpg.set_value(PARAM_GRID_WIDTH, 0)
    if dpg.does_item_exist(PARAM_GRID_HEIGHT):
        dpg.set_value(PARAM_GRID_HEIGHT, 0)


def _clear_loaded_result_state() -> None:
    state["density_result"] = None
    state["density_preview"] = None
    _reset_report_params()
    _clear_source_dependent_ui_state()
    _update_density_session_info()


def _undo_last_brush_stroke(*, finish_active: bool = True) -> bool:
    editing = _active_mask_editing_session()
    if editing is None:
        _set_status("Find a preliminary contour before editing its mask.")
        return False
    if finish_active:
        _finish_active_mask_stroke()
    if not editing.undo_last_stroke():
        _set_status("No brush strokes to undo.")
        return False

    edits = state["mask_edits"]
    stroke_id = edits[-1].get("stroke_id") if edits else None
    if stroke_id is None:
        state["mask_edits"] = edits[:-1]
    else:
        state["mask_edits"] = [
            edit for edit in edits if edit.get("stroke_id") != stroke_id
        ]

    state["active_brush_stroke_id"] = None
    state["last_brush_image"] = None
    state["last_brush_world"] = None
    _update_mask_edits_count()
    _update_last_brush_debug()
    if _refresh_contour_from_current_edits():
        _set_status("Undid last mask edit stroke; contour updated.")
    return True


def _undo_history_for_working_area_change() -> list[dict[str, object]]:
    return [
        action
        for action in state["undo_history"]
        if action.get("kind") != "mask_stroke"
    ]


def _working_area_draft_snapshot() -> dict[str, object]:
    return {
        "mode": state["mode"],
        "rectangle_roi": state["rectangle_roi"],
        "roi_first_world": state["roi_first_world"],
        "roi_current_world": state["roi_current_world"],
        "polygon_points": list(state["polygon_points"]),
        "polygon_finished": bool(state["polygon_finished"]),
    }


def _restore_working_area_undo_action(action: dict[str, object]) -> bool:
    density_result = state.get("density_result")
    if not isinstance(density_result, DensityProcessingResult):
        return False

    history = list(state["undo_history"])
    restored_area = action.get("restored_working_area")
    if isinstance(restored_area, WorkingArea):
        apply_working_area_state(state, restored_area, density_result)
    else:
        clear_working_area_state(state)
        state["density_result"] = density_result
    state["undo_history"] = history

    draft = action.get("draft")
    if isinstance(draft, dict):
        state["mode"] = draft["mode"]
        state["rectangle_roi"] = draft["rectangle_roi"]
        state["roi_first_world"] = draft["roi_first_world"]
        state["roi_current_world"] = draft["roi_current_world"]
        state["polygon_points"] = list(draft["polygon_points"])
        state["polygon_finished"] = bool(draft["polygon_finished"])
        state["editing_overlay_visible"] = True
    else:
        state["editing_overlay_visible"] = False

    if isinstance(restored_area, WorkingArea):
        inside_mask = _build_selection_inside_mask(restored_area)
        if inside_mask is not None:
            _update_selection_texture(inside_mask)
    elif dpg.does_item_exist(SELECTION_TEXTURE_TAG):
        dpg.delete_item(SELECTION_TEXTURE_TAG)

    _delete_processing_mask_texture()
    _set_mask_edit_controls_enabled(False)
    _update_mask_edits_count()
    _redraw_brush_cursor_overlay()
    dpg.set_value(ROI_OUTPUT_TAG, "")
    dpg.set_value(POLYGON_OUTPUT_TAG, "")
    _update_polygon_points_text()
    _update_working_area_info()
    _update_contour_info()
    _update_hole_detection_info()
    _redraw_preview()

    if bool(action.get("had_contour")):
        _find_contour_callback()

    if action.get("kind") == "working_area_apply":
        dpg.set_value(
            ROI_STATUS_TAG,
            "Working Area application undone; applied shape restored as draft.",
        )
    else:
        dpg.set_value(ROI_STATUS_TAG, "Working Area clear undone.")
    return True


def _undo_user_action() -> bool:
    history = state["undo_history"]
    while history:
        action = history.pop()
        if action.get("density_session") is not state.get("density_result"):
            continue
        kind = action.get("kind")
        if kind == "polygon_point":
            expected_count = int(action["point_count"])
            if len(state["polygon_points"]) < expected_count:
                continue
            return _undo_last_polygon_point(allow_hidden_draft=True)
        if kind == "mask_stroke":
            if action.get("mask_editing_session") is not state.get(
                "mask_editing_session"
            ):
                continue
            return _undo_last_brush_stroke(finish_active=False)
        if kind in {"working_area_apply", "working_area_clear"}:
            return _restore_working_area_undo_action(action)
    _set_status("Nothing to undo.")
    return False


def _undo_user_action_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    _undo_user_action()


def _update_last_brush_debug() -> None:
    if not dpg.does_item_exist(LAST_BRUSH_DEBUG_TAG):
        return

    params = _get_preview_params()
    if params is None:
        grid_min_text = "grid_min_x=-\ngrid_min_y=-"
    else:
        grid_min_x, grid_min_y, _cell_size, _grid_width, _grid_height = params
        grid_min_text = (
            f"grid_min_x={grid_min_x:.6f}\n"
            f"grid_min_y={grid_min_y:.6f}"
        )

    last_brush_image = state["last_brush_image"]
    last_brush_world = state["last_brush_world"]

    if last_brush_image is None or last_brush_world is None:
        dpg.set_value(
            LAST_BRUSH_DEBUG_TAG,
            f"{grid_min_text}\nlast brush image x/y: -\nlast brush world x/y: -",
        )
        return

    image_x, image_y = last_brush_image
    world_x, world_y = last_brush_world
    dpg.set_value(
        LAST_BRUSH_DEBUG_TAG,
        f"{grid_min_text}\n"
        "last brush image x/y: "
        f"{image_x:.2f}, {image_y:.2f}\n"
        "last brush world x/y: "
        f"{world_x:.6f}, {world_y:.6f}",
    )


def _update_debug_coords() -> None:
    if not dpg.does_item_exist(DEBUG_COORDS_TAG):
        return

    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
    draw_min_x = 0.0
    draw_min_y = 0.0
    local_x = None
    local_y = None

    if dpg.does_item_exist(IMAGE_TAG):
        draw_min_x, draw_min_y = dpg.get_item_rect_min(IMAGE_TAG)
        local_pos = _screen_to_canvas(mouse_x, mouse_y)
        if local_pos is not None:
            local_x, local_y = local_pos

    image_pos = screen_to_image(mouse_x, mouse_y)
    origin_x, origin_y = _image_origin()

    if image_pos is None:
        image_text = "image_x=- image_y=-"
    else:
        image_x, image_y = image_pos
        image_text = f"image_x={image_x:.2f} image_y={image_y:.2f}"

    if local_x is None or local_y is None:
        local_text = "local_mouse_x=- local_mouse_y=-"
    else:
        local_text = f"local_mouse_x={local_x:.2f} local_mouse_y={local_y:.2f}"

    dpg.set_value(
        DEBUG_COORDS_TAG,
        "mouse_screen_x={:.2f} mouse_screen_y={:.2f}\n"
        "draw_min_x={:.2f} draw_min_y={:.2f}\n"
        "{}\n"
        "{}\n"
        "zoom={:.4f}\n"
        "pan_x={:.2f} pan_y={:.2f}\n"
        "image_origin_x={:.2f} image_origin_y={:.2f}".format(
            mouse_x,
            mouse_y,
            draw_min_x,
            draw_min_y,
            local_text,
            image_text,
            float(state["zoom"]),
            float(state["pan_x"]),
            float(state["pan_y"]),
            origin_x,
            origin_y,
        ),
    )


def _update_pan_from_mouse() -> None:
    moved = False
    for button in ("right", "middle"):
        gesture = state.get("mouse_gestures", {}).get(button)
        if gesture is None or not gesture.get("dragged"):
            continue
        current = tuple(gesture["last_screen"])
        last = tuple(gesture["pan_last_screen"])
        dx = current[0] - last[0]
        dy = current[1] - last[1]
        if dx == 0 and dy == 0:
            continue
        state["pan_x"] = float(state["pan_x"]) + dx
        state["pan_y"] = float(state["pan_y"]) + dy
        state["last_pan_mouse"] = current
        gesture["pan_last_screen"] = current
        moved = True
    if moved:
        _redraw_preview()


def _finish_rectangle_at_world(world_x: float, world_y: float) -> bool:
    rectangle_roi = finish_rectangle_draft(state, (world_x, world_y))
    if rectangle_roi is None:
        return False
    assert rectangle_roi is not None
    min_x, min_y, max_x, max_y = rectangle_roi
    state["editing_overlay_visible"] = True
    dpg.set_value(ROI_STATUS_TAG, "ROI rectangle ready.")
    dpg.set_value(
        ROI_OUTPUT_TAG,
        "--roi "
        f"{_format_cli_float(min_x)} "
        f"{_format_cli_float(min_y)} "
        f"{_format_cli_float(max_x)} "
        f"{_format_cli_float(max_y)}",
    )
    _redraw_preview()
    return True


def _route_short_left_click(current: tuple[float, float]) -> bool:
    if state.get("pick_manual_hole_center"):
        if _warn_preview_grid_params_not_loaded():
            return True
        coords = screen_to_world(*current)
        if coords is None:
            _set_status("Pick hole center: click inside preview image.")
        else:
            _finish_manual_hole_center_pick(coords[-2], coords[-1])
        return True

    hole_id = _hole_at_screen(*current)
    if hole_id is not None:
        state["hovered_hole_id"] = hole_id
        _set_manual_hole_status_by_id(hole_id, accepted=True)
        return True

    if _warn_preview_grid_params_not_loaded():
        return False
    coords = screen_to_world(*current)
    if coords is None:
        return False
    world_x, world_y = coords[-2], coords[-1]

    if state.get("mode") == "mask_brush":
        _apply_mask_brush_at_screen(current, initial_world=(world_x, world_y))
        _finish_active_mask_stroke()
        return True
    if state.get("mode") == "polygon":
        state["polygon_points"].append((world_x, world_y))
        state["undo_history"].append(
            {
                "kind": "polygon_point",
                "point_count": len(state["polygon_points"]),
                "density_session": state.get("density_result"),
            }
        )
        state["polygon_finished"] = False
        state["editing_overlay_visible"] = True
        dpg.set_value(
            ROI_STATUS_TAG,
            f"Polygon ROI: added point {len(state['polygon_points'])}.",
        )
        dpg.set_value(POLYGON_OUTPUT_TAG, "")
        _update_polygon_points_text()
        _redraw_preview()
        return True
    if state.get("mode") == "rectangle":
        if state.get("roi_first_world") is None:
            begin_rectangle_draft(state, (world_x, world_y))
            dpg.set_value(
                ROI_STATUS_TAG,
                f"ROI first corner: x={world_x:.6f}, y={world_y:.6f}",
            )
            dpg.set_value(ROI_OUTPUT_TAG, "")
            _redraw_polygon_overlay()
            return True
        return _finish_rectangle_at_world(world_x, world_y)
    return False


def _route_short_right_click(current: tuple[float, float]) -> bool:
    if state.get("pick_manual_hole_center"):
        return False
    hole_id = _hole_at_screen(*current)
    if hole_id is None:
        return False
    state["hovered_hole_id"] = hole_id
    _set_manual_hole_status_by_id(hole_id, accepted=False)
    return True


def _finish_drag_gesture(
    button: str,
    gesture: dict,
    current: tuple[float, float],
) -> None:
    if button in {"right", "middle"}:
        _update_pan_from_mouse()
        return
    if button != "left":
        return
    if state.get("mode") == "rectangle":
        _update_rectangle_drag_from_mouse(current)
        if gesture.get("rectangle_drag_started"):
            coords = screen_to_world(*current)
            if coords is not None:
                _finish_rectangle_at_world(coords[-2], coords[-1])
    elif state.get("mode") == "mask_brush":
        _update_mask_brush_from_mouse()
        _finish_active_mask_stroke()


def _mouse_release_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    button = str(_user_data or "")
    gesture = state.get("mouse_gestures", {}).get(button)
    if gesture is None:
        return

    current = tuple(dpg.get_mouse_pos(local=False))
    gesture["last_screen"] = current
    if not gesture.get("dragged") and _gesture_is_drag(
        gesture["start_screen"],
        current,
    ):
        gesture["dragged"] = True

    if gesture.get("dragged"):
        _finish_drag_gesture(button, gesture, current)
    elif button == "left":
        _route_short_left_click(current)
    elif button == "right":
        _route_short_right_click(current)

    state["mouse_gestures"].pop(button, None)
    if button in {"right", "middle"}:
        state["last_pan_mouse"] = None
    if button == "left":
        state["active_brush_stroke_id"] = None
        state["last_brush_image"] = None
        state["last_brush_world"] = None
    _update_hole_hover_from_mouse(force=True)


def _mouse_wheel_callback(_sender=None, app_data=None, _user_data=None) -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if not dpg.is_item_hovered(IMAGE_TAG):
        return

    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
    anchor = _screen_to_canvas(mouse_x, mouse_y)
    if anchor is None:
        return

    wheel_delta = float(app_data or 0)
    if wheel_delta > 0:
        _set_zoom(float(state["zoom"]) * ZOOM_STEP, anchor_canvas_pos=anchor)
    elif wheel_delta < 0:
        _set_zoom(float(state["zoom"]) / ZOOM_STEP, anchor_canvas_pos=anchor)


def _zoom_in_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    _set_zoom(float(state["zoom"]) * ZOOM_STEP)


def _zoom_out_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    _set_zoom(float(state["zoom"]) / ZOOM_STEP)


def _reset_view_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    state["zoom"] = 1.0
    state["pan_x"] = 0.0
    state["pan_y"] = 0.0
    state["last_pan_mouse"] = None
    dpg.set_value(ZOOM_TEXT_TAG, "Zoom: 100%")
    _redraw_preview()


def _reset_roi_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    enter_rectangle_mode_state(state)
    state["brush_cursor_image"] = None
    _redraw_brush_cursor_overlay()
    _redraw_preview()
    dpg.set_value(ROI_STATUS_TAG, "ROI mode: click two image corners.")


def _polygon_mode_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    enter_polygon_mode_state(state)
    state["brush_cursor_image"] = None
    _redraw_brush_cursor_overlay()
    _update_polygon_points_text()
    _redraw_preview()
    dpg.set_value(ROI_STATUS_TAG, "Polygon ROI mode: click image points.")


def _mask_brush_mode_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    if _active_mask_editing_session() is None:
        _set_status("Find a preliminary contour before editing its mask.")
        return
    try:
        validate_brush_diameter(dpg.get_value(BRUSH_SIZE_TAG))
    except ValueError as exc:
        _set_status(str(exc))
        return
    state["mode"] = "mask_brush"
    state["roi_first_world"] = None
    state["roi_current_world"] = None
    state["editing_overlay_visible"] = False
    state["last_brush_image"] = None
    state["last_brush_world"] = None
    state["active_brush_stroke_id"] = None
    if dpg.does_item_exist(SHOW_PROCESSING_MASK_TAG):
        dpg.set_value(SHOW_PROCESSING_MASK_TAG, True)
    _update_brush_cursor_from_mouse()
    _redraw_preview()
    dpg.set_value(ROI_STATUS_TAG, "Mask brush mode: hold left mouse button.")


def _brush_settings_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    try:
        validate_brush_diameter(dpg.get_value(BRUSH_SIZE_TAG))
    except ValueError as exc:
        _set_status(str(exc))
        return
    _redraw_brush_cursor_overlay()


def _finish_polygon_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    _finish_polygon()


def _finish_polygon() -> bool:
    points = state["polygon_points"]
    if len(points) < 3:
        dpg.set_value(ROI_STATUS_TAG, "Polygon ROI needs at least 3 points.")
        dpg.set_value(POLYGON_OUTPUT_TAG, "")
        return False

    points_text = ";".join(
        f"{_format_cli_float(x)},{_format_cli_float(y)}" for x, y in points
    )
    state["polygon_finished"] = True
    state["editing_overlay_visible"] = True
    dpg.set_value(ROI_STATUS_TAG, "Polygon ROI ready.")
    dpg.set_value(POLYGON_OUTPUT_TAG, f'--roi-poly "{points_text}"')
    return True


def _clear_polygon_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    clear_polygon_draft_state(state)
    state["undo_history"] = [
        action
        for action in state["undo_history"]
        if action.get("kind") != "polygon_point"
    ]
    dpg.set_value(POLYGON_OUTPUT_TAG, "")
    _update_polygon_points_text()
    _redraw_preview()
    dpg.set_value(ROI_STATUS_TAG, "Polygon ROI cleared.")


def _key_press_callback(_sender=None, app_data=None, _user_data=None) -> None:
    key = app_data
    undo_shortcut = (
        key == dpg.mvKey_Z
        and (
            dpg.is_key_down(dpg.mvKey_LControl)
            or dpg.is_key_down(dpg.mvKey_RControl)
        )
    )
    backspace = key == dpg.mvKey_Back

    if undo_shortcut:
        _undo_user_action()
        return

    if state["mode"] == "polygon" and backspace:
        _undo_last_polygon_point()


def _working_area_from_draft() -> WorkingArea | None:
    if state["mode"] == "polygon":
        if state["polygon_finished"]:
            return WorkingArea.from_polygon(state["polygon_points"])
        return None

    if state["mode"] == "rectangle":
        rectangle_roi = state["rectangle_roi"]
        if rectangle_roi is not None:
            return WorkingArea.from_rectangle_bounds(rectangle_roi)

    return None


def _build_selection_inside_mask(working_area: WorkingArea) -> np.ndarray | None:
    image_width = int(state["image_width"])
    image_height = int(state["image_height"])
    if image_width <= 0 or image_height <= 0:
        return None

    density_result = state.get("density_result")
    if not isinstance(density_result, DensityProcessingResult):
        return None

    grid_mask = working_area.to_grid_mask(density_result.grid).astype(np.uint8)
    preview_mask = np.flipud(grid_mask)
    if preview_mask.shape != (image_height, image_width):
        preview_mask = cv2.resize(
            preview_mask,
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return preview_mask


def _update_selection_texture(inside_mask: np.ndarray) -> None:
    image_height, image_width = inside_mask.shape
    overlay = np.zeros((image_height, image_width, 4), dtype=np.float32)
    overlay[..., 3] = 0.62
    overlay[inside_mask > 0, 3] = 0.0

    if dpg.does_item_exist(SELECTION_LAYER_TAG):
        dpg.delete_item(SELECTION_LAYER_TAG)
    if dpg.does_item_exist(SELECTION_TEXTURE_TAG):
        dpg.delete_item(SELECTION_TEXTURE_TAG)

    with dpg.texture_registry():
        dpg.add_static_texture(
            width=image_width,
            height=image_height,
            default_value=overlay.ravel().tolist(),
            tag=SELECTION_TEXTURE_TAG,
        )


def _report_working_area_apply_error(message: str) -> None:
    if isinstance(state.get("active_working_area"), WorkingArea):
        state["editing_overlay_visible"] = False
        message += " Previous active Working Area restored; select a mode to continue."
        _redraw_preview()
    dpg.set_value(ROI_STATUS_TAG, message)


def _apply_selection_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    density_result = state.get("density_result")
    if not isinstance(density_result, DensityProcessingResult):
        _report_working_area_apply_error(
            "Build a density map before applying Working Area."
        )
        return

    if state["mode"] == "polygon" and not state["polygon_finished"]:
        if not state["polygon_points"] or not _finish_polygon():
            _report_working_area_apply_error(
                "Polygon Working Area needs at least 3 valid points."
            )
            return

    if state["mode"] == "rectangle" and state["roi_first_world"] is not None:
        _report_working_area_apply_error(
            "Rectangle Working Area needs a second corner."
        )
        return

    try:
        working_area = _working_area_from_draft()
    except ValueError as exc:
        _report_working_area_apply_error(f"Invalid Working Area: {exc}")
        return

    if working_area is None:
        _report_working_area_apply_error(
            f"No finished {state['mode']} draft to apply."
        )
        return

    inside_mask = _build_selection_inside_mask(working_area)
    if inside_mask is None:
        _report_working_area_apply_error(
            "Could not map Working Area to the current grid."
        )
        return

    previous_working_area = get_active_working_area()
    draft_snapshot = _working_area_draft_snapshot()
    previous_undo_history = _undo_history_for_working_area_change()
    _update_selection_texture(inside_mask)
    had_contour = isinstance(
        state.get("contour_processing_result"),
        PreliminaryContourSession,
    )
    apply_working_area_state(state, working_area, density_result)
    state["undo_history"] = previous_undo_history
    state["undo_history"].append(
        {
            "kind": "working_area_apply",
            "density_session": density_result,
            "restored_working_area": previous_working_area,
            "draft": draft_snapshot,
            "had_contour": had_contour,
        }
    )
    _delete_processing_mask_texture()
    _set_mask_edit_controls_enabled(False)
    _update_mask_edits_count()
    _redraw_brush_cursor_overlay()
    _update_working_area_info()
    _update_contour_info()
    _update_hole_detection_info()
    _redraw_preview()
    if had_contour:
        _find_contour_callback()
    dpg.set_value(ROI_STATUS_TAG, "Working Area applied; mask edits cleared.")


def _clear_selection_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    density_result = state.get("density_result")
    previous_working_area = get_active_working_area()
    had_contour = isinstance(
        state.get("contour_processing_result"),
        PreliminaryContourSession,
    )
    previous_undo_history = _undo_history_for_working_area_change()
    clear_working_area_state(state)
    state["density_result"] = density_result
    state["undo_history"] = previous_undo_history
    if isinstance(previous_working_area, WorkingArea):
        state["undo_history"].append(
            {
                "kind": "working_area_clear",
                "density_session": density_result,
                "restored_working_area": previous_working_area,
                "draft": None,
                "had_contour": had_contour,
            }
        )
    _delete_processing_mask_texture()
    _set_mask_edit_controls_enabled(False)
    _update_mask_edits_count()
    _redraw_brush_cursor_overlay()

    dpg.set_value(ROI_OUTPUT_TAG, "")
    dpg.set_value(POLYGON_OUTPUT_TAG, "")
    _update_polygon_points_text()
    _redraw_preview()

    if dpg.does_item_exist(SELECTION_TEXTURE_TAG):
        dpg.delete_item(SELECTION_TEXTURE_TAG)

    _update_working_area_info()
    _update_contour_info()
    _update_hole_detection_info()
    if had_contour:
        _find_contour_callback()
    dpg.set_value(ROI_STATUS_TAG, "Working area: Full scan; mask edits cleared.")


def _clear_mask_edits_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    editing = _active_mask_editing_session()
    if editing is None:
        _set_status("Find a preliminary contour before editing its mask.")
        return
    editing.clear_edits()
    state["mask_edits"] = []
    state["last_brush_image"] = None
    state["last_brush_world"] = None
    state["brush_cursor_image"] = None
    state["active_brush_stroke_id"] = None
    state["next_brush_stroke_id"] = 1
    state["undo_history"] = [
        action
        for action in state["undo_history"]
        if action.get("kind") != "mask_stroke"
    ]
    _update_mask_edits_count()
    _update_last_brush_debug()
    if _refresh_contour_from_current_edits():
        _set_status("Mask edits cleared; preliminary contour updated.")


def _write_mask_edits_json(output_path: str | Path) -> None:
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".json":
        output_path = output_path.with_suffix(output_path.suffix + ".json")

    payload = {
        "edits": state["mask_edits"],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _set_status(f"Mask edits saved: {output_path.resolve()}")


def _save_mask_edits_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    dpg.show_item("save_mask_edits_dialog")


def _save_mask_edits_dialog_callback(_sender, app_data) -> None:
    file_path_name = app_data.get("file_path_name")
    if not file_path_name:
        return

    _write_mask_edits_json(file_path_name)


def _write_edited_holes_json(output_path: str | Path) -> None:
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".json":
        output_path = output_path.with_suffix(output_path.suffix + ".json")

    payload = {
        "groups": state["hole_groups"],
        "holes": state["holes"],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _set_status(f"Edited holes JSON saved: {output_path.resolve()}")


def _save_edited_holes_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    dpg.show_item("save_edited_holes_dialog")


def _save_edited_holes_dialog_callback(_sender, app_data) -> None:
    file_path_name = app_data.get("file_path_name")
    if not file_path_name:
        return

    _write_edited_holes_json(file_path_name)


def _quote_command_arg(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def _append_option(parts: list[str], name: str, value) -> None:
    parts.append(name)
    parts.append(str(value))


def _append_float_option(parts: list[str], name: str, value: float) -> None:
    _append_option(parts, name, _format_cli_float(value))


def _generate_command_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    input_file_path = str(dpg.get_value(CMD_INPUT_FILE_TAG)).strip()
    cell = float(dpg.get_value(CMD_CELL_TAG))
    threshold = str(dpg.get_value(CMD_THRESHOLD_TAG)).strip() or "auto"
    fill_holes_area = int(dpg.get_value(CMD_FILL_HOLES_TAG))
    boundary_width_mm = float(dpg.get_value(CMD_BOUNDARY_WIDTH_TAG))
    simplify_mm = float(dpg.get_value(CMD_SIMPLIFY_TAG))
    mask_edits_path = str(dpg.get_value(CMD_MASK_EDITS_TAG)).strip()

    contour_enabled = bool(dpg.get_value(CMD_CONTOUR_TAG))
    dxf_enabled = bool(dpg.get_value(CMD_DXF_TAG))

    parts = [
        "python",
        "-m",
        "app.main",
        _quote_command_arg(input_file_path),
    ]

    _append_float_option(parts, "--cell", cell)
    _append_option(parts, "--threshold", threshold)

    rectangle_roi = str(dpg.get_value(ROI_OUTPUT_TAG)).strip()
    if rectangle_roi:
        parts.append(rectangle_roi)

    polygon_roi = str(dpg.get_value(POLYGON_OUTPUT_TAG)).strip()
    if polygon_roi:
        parts.append(polygon_roi)

    if bool(dpg.get_value(CMD_KEEP_LARGEST_TAG)):
        parts.append("--keep-largest")

    if mask_edits_path:
        _append_option(parts, "--mask-edits", _quote_command_arg(mask_edits_path))

    if fill_holes_area > 0:
        _append_option(parts, "--fill-holes-area", _format_cli_float(fill_holes_area))

    if contour_enabled or dxf_enabled:
        parts.append("--contour")

    if dxf_enabled:
        parts.append("--dxf")

    if bool(dpg.get_value(CMD_EXPORT_CLEAN_TAG)):
        parts.append("--export-clean")

    if bool(dpg.get_value(CMD_EXPORT_BOUNDARY_TAG)):
        parts.append("--export-boundary")
        _append_float_option(parts, "--boundary-width-mm", boundary_width_mm)

    if simplify_mm > 0:
        _append_float_option(parts, "--simplify-mm", simplify_mm)

    if bool(dpg.get_value(CMD_HOLES_TAG)):
        parts.append("--holes")

    dpg.set_value(COMMAND_OUTPUT_TAG, " ".join(parts))


def _copy_command_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    command = str(dpg.get_value(COMMAND_OUTPUT_TAG)).strip()
    if command:
        dpg.set_clipboard_text(command)
        _set_status("Command copied to clipboard.")


def _show_png(path: str | Path, clear_mask_edits: bool = True) -> None:
    image_path = Path(path)

    try:
        width, height, _channels, data = dpg.load_image(str(image_path))
    except Exception as exc:
        _set_status(f"Could not load PNG: {exc}")
        return

    if dpg.does_item_exist(IMAGE_TAG):
        dpg.delete_item(IMAGE_TAG)

    if dpg.does_item_exist(SELECTION_TEXTURE_TAG):
        dpg.delete_item(SELECTION_TEXTURE_TAG)

    old_texture_tag = state["texture_tag"]
    if dpg.does_item_exist(old_texture_tag):
        dpg.delete_item(old_texture_tag)

    with dpg.texture_registry():
        dpg.add_static_texture(
            width=width,
            height=height,
            default_value=data,
            tag=TEXTURE_TAG,
        )

    if dpg.does_item_exist("image_hint"):
        dpg.delete_item("image_hint")

    state["density_result"] = None
    state["density_preview"] = None
    state["texture_tag"] = TEXTURE_TAG
    _update_density_session_info()
    state["image_width"] = width
    state["image_height"] = height
    state["zoom"] = 1.0
    state["pan_x"] = 0.0
    state["pan_y"] = 0.0
    state["last_pan_mouse"] = None
    state["roi_first_world"] = None
    state["roi_current_world"] = None
    state["rectangle_roi"] = None
    state["polygon_finished"] = False
    state["selection_applied"] = False
    state["editing_overlay_visible"] = True
    state["selection_kind"] = None
    state["selection_polygon_points"] = []
    state["polygon_points"] = []
    state["active_working_area"] = None
    state["working_area_density_session"] = None
    state["contour_processing_result"] = None
    _clear_mask_editing_state()
    _delete_processing_mask_texture()
    state["last_brush_image"] = None
    state["last_brush_world"] = None
    state["brush_cursor_image"] = None
    state["active_brush_stroke_id"] = None
    state["next_brush_stroke_id"] = 1
    if clear_mask_edits:
        state["mask_edits"] = []

    if int(dpg.get_value(PARAM_GRID_WIDTH)) <= 0:
        dpg.set_value(PARAM_GRID_WIDTH, width)
    if int(dpg.get_value(PARAM_GRID_HEIGHT)) <= 0:
        dpg.set_value(PARAM_GRID_HEIGHT, height)

    dpg.set_value(ROI_STATUS_TAG, "ROI mode: click two image corners.")
    dpg.set_value(ROI_OUTPUT_TAG, "")
    dpg.set_value(POLYGON_OUTPUT_TAG, "")
    dpg.set_value(ZOOM_TEXT_TAG, "Zoom: 100%")
    _update_polygon_points_text()
    _update_mask_edits_count()
    _update_last_brush_debug()
    _update_working_area_info()
    _update_contour_info()
    _redraw_preview()
    if state["contour_points"]:
        _set_status("Preview changed. Loaded contour may belong to another result.")
    else:
        _set_status(f"Loaded: {image_path}")


def _open_file_callback(_sender, app_data) -> None:
    selections = app_data.get("selections", {})
    if not selections:
        return

    selected_path = next(iter(selections.values()))
    _show_png(selected_path)


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _newest_matching_file(folder: Path, pattern: str) -> tuple[Path | None, int]:
    matches = [path for path in folder.glob(pattern) if path.is_file()]
    if not matches:
        return None, 0

    newest = max(matches, key=lambda path: path.stat().st_mtime)
    return newest, len(matches)


def _find_source_preview_file(folder: Path, stem: str) -> tuple[Path | None, int]:
    forbidden_words = (
        "contour",
        "holes",
        "mixed",
        "refined",
        "debug",
        "mask_edits",
        "overlay",
        "line",
    )
    stem_lower = stem.lower()
    png_candidates = [
        path
        for path in folder.glob(f"{stem}*.png")
        if path.is_file()
        and path.name.lower().startswith(stem_lower)
        and not any(word in path.name.lower() for word in forbidden_words)
    ]

    for marker in ("density_preview", "density", "preview"):
        matches = [
            path for path in png_candidates if marker in path.name.lower()
        ]
        if matches:
            newest = max(matches, key=lambda path: path.stat().st_mtime)
            return newest, len(matches)

    return None, 0


def _find_source_result_files(
    source_path: str | Path,
) -> tuple[dict[str, Path | None], bool, bool]:
    source_path = Path(source_path)
    stem = source_path.stem
    output_folder = Path.cwd() / "data" / "output"
    result: dict[str, Path | None] = {
        "preview": None,
        "report": None,
        "contour": None,
        "holes": None,
        "mixed_contour": None,
        "demo_summary": None,
    }

    if not output_folder.exists():
        return result, False, False

    patterns = {
        "report": f"{stem}*report*.txt",
        "contour": f"{stem}*contour*.csv",
        "holes": f"{stem}*holes*.json",
        "mixed_contour": f"{stem}*mixed_contour*.json",
        "demo_summary": f"{stem}*demo_summary*.txt",
    }

    preview_path, preview_matches_count = _find_source_preview_file(
        output_folder,
        stem,
    )
    result["preview"] = preview_path
    multiple_raw_previews_found = preview_matches_count > 1

    multiple_found = False
    for key, pattern in patterns.items():
        path, matches_count = _newest_matching_file(output_folder, pattern)
        result[key] = path
        if matches_count > 1:
            multiple_found = True

    return result, multiple_found, multiple_raw_previews_found


def _find_processing_result_files(png_path: str | Path) -> dict[str, Path | None]:
    png_path = Path(png_path)
    folder = png_path.parent
    stem = png_path.stem

    report_candidates = [folder / "report.txt"]
    contour_candidates = [folder / "contour.csv"]
    holes_candidates = [folder / "holes.json"]
    mixed_contour_candidates = sorted(folder.glob("*_mixed_contour.json"))

    marker = "_density_cell_"
    if marker in stem:
        base_name, cell_text = stem.split(marker, 1)
        if "_smooth_" in cell_text:
            cell_text = cell_text.split("_smooth_", 1)[0]

        report_candidates.insert(0, folder / f"{base_name}_report_cell_{cell_text}.txt")
        contour_candidates = sorted(
            folder.glob(f"{base_name}_contour_cell_{cell_text}_threshold_*.csv")
        ) + contour_candidates
        holes_candidates = sorted(
            folder.glob(f"{base_name}_holes_cell_{cell_text}_threshold_*.json")
        ) + holes_candidates
        mixed_contour_candidates = [
            folder / f"{base_name}_mixed_contour.json",
        ] + mixed_contour_candidates

    return {
        "report": _first_existing_path(report_candidates),
        "contour": _first_existing_path(contour_candidates),
        "holes": _first_existing_path(holes_candidates),
        "mixed_contour": _first_existing_path(mixed_contour_candidates),
    }


def _load_processing_result(path: str | Path) -> None:
    png_path = Path(path)
    _show_png(png_path, clear_mask_edits=False)

    related_files = _find_processing_result_files(png_path)
    if related_files["report"] is not None:
        _load_report(related_files["report"])
    if related_files["contour"] is not None:
        _load_contour_csv(related_files["contour"])
    if related_files["holes"] is not None:
        _load_holes_json(related_files["holes"])
    if related_files["mixed_contour"] is not None:
        _load_mixed_contour_json(related_files["mixed_contour"])

    status_lines = [
        f"Report: {'found' if related_files['report'] is not None else 'not found'}",
        f"Contour: {'found' if related_files['contour'] is not None else 'not found'}",
        f"Holes: {'found' if related_files['holes'] is not None else 'not found'}",
        "Mixed contour: "
        f"{'found' if related_files['mixed_contour'] is not None else 'not found'}",
    ]
    _set_status("\n".join(status_lines))


def _open_processing_result_callback(_sender, app_data) -> None:
    selections = app_data.get("selections", {})
    if not selections:
        return

    selected_path = next(iter(selections.values()))
    _load_processing_result(selected_path)


def _load_result_by_source_file(source_path: str | Path) -> None:
    related_files, multiple_found, multiple_raw_previews_found = (
        _find_source_result_files(source_path)
    )
    found_any = any(path is not None for path in related_files.values())
    if not found_any:
        _set_status("No matching processed result found for source file.")
        return

    _clear_loaded_result_state()

    if related_files["preview"] is not None:
        _show_png(related_files["preview"], clear_mask_edits=False)
    if related_files["report"] is not None:
        _load_report(related_files["report"])
    if related_files["contour"] is not None:
        _load_contour_csv(related_files["contour"])
    if related_files["holes"] is not None:
        _load_holes_json(related_files["holes"])
    if related_files["mixed_contour"] is not None:
        _load_mixed_contour_json(related_files["mixed_contour"])
    if related_files["demo_summary"] is not None:
        state["demo_summary_file"] = str(related_files["demo_summary"])
        _update_demo_summary_info()

    status_lines = [
        f"Preview: {'found' if related_files['preview'] is not None else 'not found'}",
        f"Report: {'found' if related_files['report'] is not None else 'not found'}",
        f"Contour: {'found' if related_files['contour'] is not None else 'not found'}",
        f"Holes: {'found' if related_files['holes'] is not None else 'not found'}",
        "Mixed contour: "
        f"{'found' if related_files['mixed_contour'] is not None else 'not found'}",
        "Demo summary: "
        f"{'found' if related_files['demo_summary'] is not None else 'not found'}",
    ]
    if related_files["preview"] is None:
        status_lines.append("Raw density preview not found for source file.")
    if multiple_raw_previews_found:
        status_lines.append("Multiple matching raw previews found, loaded newest.")
    if multiple_found:
        status_lines.append("Multiple matching results found, loaded newest.")

    _set_status("\n".join(status_lines))


def _open_source_result_callback(_sender, app_data) -> None:
    selections = app_data.get("selections", {})
    if not selections:
        return

    selected_path = next(iter(selections.values()))
    _load_result_by_source_file(selected_path)


def _load_report(path: str | Path) -> None:
    report_path = Path(path)

    try:
        text = report_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = report_path.read_text(encoding="cp1251")
    except Exception as exc:
        _set_status(f"Could not load report.txt: {exc}")
        return

    try:
        grid_min_x, grid_min_y, cell_size, width_cells, height_cells = (
            _parse_report_text(text)
        )
    except Exception as exc:
        _set_status(f"Could not parse report.txt: {exc}")
        return

    dpg.set_value(PARAM_GRID_MIN_X, grid_min_x)
    dpg.set_value(PARAM_GRID_MIN_Y, grid_min_y)
    dpg.set_value(PARAM_CELL_SIZE, cell_size)
    dpg.set_value(PARAM_GRID_WIDTH, width_cells)
    dpg.set_value(PARAM_GRID_HEIGHT, height_cells)
    if dpg.does_item_exist(CMD_CELL_TAG):
        dpg.set_value(CMD_CELL_TAG, cell_size)

    state["report_loaded"] = True
    _update_last_brush_debug()
    if dpg.does_item_exist(state["texture_tag"]):
        _redraw_preview()

    _set_status(
        "Report loaded: "
        f"grid_min_x={grid_min_x:.6f}, "
        f"grid_min_y={grid_min_y:.6f}, "
        f"cell={cell_size:.6f}, "
        f"size={width_cells}x{height_cells}"
    )


def _open_report_callback(_sender, app_data) -> None:
    selections = app_data.get("selections", {})
    if not selections:
        return

    selected_path = next(iter(selections.values()))
    _load_report(selected_path)


def _normalize_hole_json_item(item: dict) -> dict:
    required_fields = ["id", "accepted", "center_x", "center_y", "radius"]
    for field in required_fields:
        if field not in item:
            raise ValueError(f"hole field not found: {field}")

    return {
        "id": int(item["id"]),
        "accepted": bool(item["accepted"]),
        "enabled": bool(item.get("enabled", True)),
        "reject_reason": str(item.get("reject_reason", "")),
        "center_x": float(item["center_x"]),
        "center_y": float(item["center_y"]),
        "radius": float(item["radius"]),
        "diameter": float(item.get("diameter", float(item["radius"]) * 2.0)),
        "group_id": (
            str(item["group_id"]) if item.get("group_id") is not None else None
        ),
    }


def _load_holes_json(path: str | Path) -> None:
    holes_path = Path(path)

    try:
        data = json.loads(holes_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _set_status(f"Could not load holes.json: {exc}")
        return

    try:
        raw_holes = data["holes"] if isinstance(data, dict) else data
        raw_groups = data.get("groups", []) if isinstance(data, dict) else []
        if not isinstance(raw_holes, list):
            raise ValueError("holes must be a list")
        if not isinstance(raw_groups, list):
            raise ValueError("groups must be a list")

        holes = []
        for item in raw_holes:
            if not isinstance(item, dict):
                raise ValueError("hole item must be an object")
            holes.append(_normalize_hole_json_item(item))

        hole_groups = []
        for item in raw_groups:
            if not isinstance(item, dict):
                raise ValueError("group item must be an object")
            hole_groups.append(
                {
                    "id": str(item.get("id", "")),
                    "name": str(item.get("name", "")),
                    "diameter": float(item.get("diameter", 0.0)),
                    "radius": float(item.get("radius", 0.0)),
                    "count": int(item.get("count", 0)),
                    "enabled": bool(item.get("enabled", True)),
                }
            )
    except Exception as exc:
        _set_status(f"Could not parse holes.json: {exc}")
        return

    state["holes"] = holes
    state["hole_groups"] = hole_groups
    state["hole_detection_session"] = None
    state["holes_outdated"] = False
    state["hole_overlay_source"] = "legacy"
    state["visible_hole_group_ids"] = {
        str(group["id"]): bool(group.get("enabled", True)) for group in hole_groups
    }
    _recount_hole_group_counts()
    _update_holes_stats()
    _update_hole_groups_display()
    _update_hole_group_target_combo()
    _redraw_preview()
    accepted_count = sum(1 for hole in holes if bool(hole.get("accepted", False)))
    rejected_count = len(holes) - accepted_count
    _set_status(
        "Holes loaded: "
        f"total={len(holes)}, accepted={accepted_count}, rejected={rejected_count}"
    )


def _open_holes_callback(_sender, app_data) -> None:
    selections = app_data.get("selections", {})
    if not selections:
        return

    selected_path = next(iter(selections.values()))
    _load_holes_json(selected_path)


def _clear_holes_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    state["holes"] = []
    state["hole_groups"] = []
    state["visible_hole_group_ids"] = {}
    state["pick_manual_hole_center"] = False
    state["manual_hole_center_world"] = None
    if state.get("hole_overlay_source") == "legacy":
        state["hole_overlay_source"] = None
    if dpg.does_item_exist(MANUAL_HOLE_PICK_TAG):
        dpg.set_value(MANUAL_HOLE_PICK_TAG, False)
    _update_holes_stats()
    _update_hole_groups_display()
    _update_hole_group_target_combo()
    _redraw_preview()
    _set_status("Holes cleared.")


def _load_contour_csv(path: str | Path) -> None:
    contour_path = Path(path)

    try:
        with contour_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            field_names = {str(name).strip().lower() for name in reader.fieldnames or []}
            if "x" not in field_names or "y" not in field_names:
                raise ValueError("CSV must contain x,y columns")

            contour_points = []
            for row_number, row in enumerate(reader, start=2):
                normalized_row = {
                    str(key).strip().lower(): value for key, value in row.items()
                }
                try:
                    contour_points.append(
                        (float(normalized_row["x"]), float(normalized_row["y"]))
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid contour coordinates at row {row_number}"
                    ) from exc

        if not contour_points:
            raise ValueError("contour CSV contains no points")
    except Exception as exc:
        _set_status(f"Could not load contour CSV: {exc}")
        return

    state["contour_points"] = contour_points
    state["contour_file"] = str(contour_path)
    state["contour_processing_result"] = None
    _clear_mask_editing_state()
    _delete_processing_mask_texture()
    _update_contour_info()
    _redraw_preview()
    _set_status(f"Contour loaded: {contour_path}, points={len(contour_points)}")


def _open_contour_callback(_sender, app_data) -> None:
    selections = app_data.get("selections", {})
    if not selections:
        return

    selected_path = next(iter(selections.values()))
    _load_contour_csv(selected_path)


def _clear_contour_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    _mark_effective_mask_changed()
    state["contour_points"] = []
    state["contour_file"] = ""
    state["contour_processing_result"] = None
    _clear_mask_editing_state()
    _delete_processing_mask_texture()
    _update_contour_info()
    _redraw_preview()
    _set_status("Contour cleared.")


def _normalize_mixed_contour_element(item: dict) -> dict:
    element_type = str(item.get("type", "")).upper()
    if element_type == "LINE":
        start = item.get("start")
        end = item.get("end")
        if not isinstance(start, dict) or not isinstance(end, dict):
            raise ValueError("LINE element must contain start/end objects")

        return {
            "id": int(item.get("id", 0)),
            "type": "LINE",
            "source": str(item.get("source", "")),
            "original_line_id": item.get("original_line_id"),
            "start": {"x": float(start["x"]), "y": float(start["y"])},
            "end": {"x": float(end["x"]), "y": float(end["y"])},
        }

    if element_type == "POLYLINE":
        raw_points = item.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("POLYLINE element must contain points list")

        points = []
        for point in raw_points:
            if not isinstance(point, dict):
                raise ValueError("POLYLINE point must be an object")
            points.append({"x": float(point["x"]), "y": float(point["y"])})

        return {
            "id": int(item.get("id", 0)),
            "type": "POLYLINE",
            "source": str(item.get("source", "")),
            "contour_start_index": item.get("contour_start_index"),
            "contour_end_index": item.get("contour_end_index"),
            "points": points,
        }

    raise ValueError(f"Unsupported mixed contour element type: {element_type}")


def _load_mixed_contour_json(path: str | Path) -> None:
    mixed_path = Path(path)

    try:
        data = json.loads(mixed_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _set_status(f"Could not load mixed contour JSON: {exc}")
        return

    try:
        raw_elements = data.get("elements", data) if isinstance(data, dict) else data
        if not isinstance(raw_elements, list):
            raise ValueError("mixed contour elements must be a list")

        elements = []
        for item in raw_elements:
            if not isinstance(item, dict):
                raise ValueError("mixed contour element must be an object")
            elements.append(_normalize_mixed_contour_element(item))
    except Exception as exc:
        _set_status(f"Could not parse mixed contour JSON: {exc}")
        return

    state["mixed_contour_elements"] = elements
    state["mixed_contour_file"] = str(mixed_path)
    _update_mixed_contour_info()
    _redraw_preview()
    total_count, lines_count, gaps_count = _mixed_contour_counts()
    _set_status(
        "Mixed contour loaded: "
        f"elements={total_count}, lines={lines_count}, polyline gaps={gaps_count}"
    )


def _open_mixed_contour_callback(_sender, app_data) -> None:
    selections = app_data.get("selections", {})
    if not selections:
        return

    selected_path = next(iter(selections.values()))
    _load_mixed_contour_json(selected_path)


def _clear_mixed_contour_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    state["mixed_contour_elements"] = []
    state["mixed_contour_file"] = ""
    _update_mixed_contour_info()
    _redraw_preview()
    _set_status("Mixed contour cleared.")


def run() -> None:
    dpg.create_context()

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_select_density_source_callback,
        tag=DENSITY_SOURCE_DIALOG_TAG,
        width=700,
        height=400,
    ):
        for extension in sorted(SUPPORTED_POINT_CLOUD_EXTENSIONS):
            dpg.add_file_extension(extension, color=(255, 200, 120, 255))

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_open_file_callback,
        tag="open_png_dialog",
        width=700,
        height=400,
    ):
        dpg.add_file_extension(".png", color=(80, 180, 255, 255))

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_open_processing_result_callback,
        tag="open_processing_result_dialog",
        width=700,
        height=400,
    ):
        dpg.add_file_extension(".png", color=(120, 220, 180, 255))

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_open_source_result_callback,
        tag="open_source_result_dialog",
        width=700,
        height=400,
    ):
        dpg.add_file_extension(".asc", color=(255, 200, 120, 255))
        dpg.add_file_extension(".xyz", color=(255, 200, 120, 255))
        dpg.add_file_extension(".xyzn", color=(255, 200, 120, 255))

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_open_report_callback,
        tag="open_report_dialog",
        width=700,
        height=400,
    ):
        dpg.add_file_extension(".txt", color=(120, 220, 140, 255))

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_save_mask_edits_dialog_callback,
        tag="save_mask_edits_dialog",
        width=700,
        height=400,
    ):
        dpg.add_file_extension(".json", color=(120, 220, 140, 255))

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_open_holes_callback,
        tag="open_holes_dialog",
        width=700,
        height=400,
    ):
        dpg.add_file_extension(".json", color=(255, 160, 80, 255))

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_save_edited_holes_dialog_callback,
        tag="save_edited_holes_dialog",
        width=700,
        height=400,
    ):
        dpg.add_file_extension(".json", color=(80, 220, 160, 255))

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_open_contour_callback,
        tag="open_contour_dialog",
        width=700,
        height=400,
    ):
        dpg.add_file_extension(".csv", color=(80, 240, 255, 255))

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_open_mixed_contour_callback,
        tag="open_mixed_contour_dialog",
        width=700,
        height=400,
    ):
        dpg.add_file_extension(".json", color=(255, 220, 90, 255))

    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=_mouse_move_callback)
        dpg.add_mouse_click_handler(
            button=dpg.mvMouseButton_Left,
            callback=_mouse_down_callback,
            user_data="left",
        )
        dpg.add_mouse_click_handler(
            button=dpg.mvMouseButton_Right,
            callback=_mouse_down_callback,
            user_data="right",
        )
        dpg.add_mouse_click_handler(
            button=dpg.mvMouseButton_Middle,
            callback=_mouse_down_callback,
            user_data="middle",
        )
        dpg.add_mouse_release_handler(
            button=dpg.mvMouseButton_Left,
            callback=_mouse_release_callback,
            user_data="left",
        )
        dpg.add_mouse_release_handler(
            button=dpg.mvMouseButton_Right,
            callback=_mouse_release_callback,
            user_data="right",
        )
        dpg.add_mouse_release_handler(
            button=dpg.mvMouseButton_Middle,
            callback=_mouse_release_callback,
            user_data="middle",
        )
        dpg.add_mouse_wheel_handler(callback=_mouse_wheel_callback)
        dpg.add_key_press_handler(callback=_key_press_callback)

    with dpg.item_handler_registry(tag="manual_threshold_commit_handlers"):
        dpg.add_item_deactivated_after_edit_handler(
            callback=_manual_threshold_commit_callback
        )

    with dpg.item_handler_registry(tag="hole_detection_commit_handlers"):
        dpg.add_item_deactivated_after_edit_handler(
            callback=_hole_detection_quality_settings_callback
        )

    with dpg.window(
        label="Source / Density",
        tag=DENSITY_SETUP_WINDOW_TAG,
        show=False,
        width=760,
        height=260,
    ):
        dpg.add_text("Point cloud file")
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag=DENSITY_SOURCE_FILE_TAG, width=-105)
            dpg.add_button(
                label="Select",
                tag=DENSITY_SELECT_BUTTON_TAG,
                callback=lambda: dpg.show_item(DENSITY_SOURCE_DIALOG_TAG),
            )
        with dpg.group(horizontal=True):
            dpg.add_text("Cell size, mm")
            dpg.add_input_float(
                tag=DENSITY_CELL_SIZE_TAG,
                default_value=0.8,
                min_value=0.0,
                width=130,
            )
            dpg.add_button(
                label="Build density map",
                tag=DENSITY_BUILD_BUTTON_TAG,
                callback=_build_density_callback,
            )
        dpg.add_progress_bar(
            default_value=0.0,
            overlay="Idle",
            width=-1,
            tag=DENSITY_PROGRESS_BAR_TAG,
        )
        dpg.add_text("Processing stage: idle", tag=DENSITY_PROGRESS_STAGE_TAG)
        dpg.add_text("0 B / 0 B", tag=DENSITY_PROGRESS_BYTES_TAG)
        dpg.add_text("Density map has not been built.", tag=DENSITY_SESSION_INFO_TAG)
        dpg.add_text("Ready.", tag=DENSITY_STATUS_TAG, wrap=720)

    with dpg.window(label="Point Contour Preview Viewer", tag="main_window"):
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Source / Density...",
                callback=lambda: dpg.show_item(DENSITY_SETUP_WINDOW_TAG),
            )
            dpg.add_text("No density map", tag=DENSITY_SUMMARY_TAG)
            dpg.add_button(
                label="Undo",
                tag=UNDO_BUTTON_TAG,
                callback=_undo_user_action_callback,
            )

        with dpg.group(horizontal=True):
            with dpg.child_window(
                tag="image_area",
                border=False,
                horizontal_scrollbar=False,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
                width=-370,
                height=-55,
            ):
                dpg.add_text(
                    "Select a point cloud and build the density map.",
                    tag="image_hint",
                )

            with dpg.child_window(
                tag="roi_side_panel",
                width=350,
                height=-55,
                border=True,
            ):
                workspace_header = dpg.add_collapsing_header(
                    label="Workspace",
                    default_open=True,
                )
                dpg.push_container_stack(workspace_header)
                dpg.add_text("ROI mode: click two image corners.", tag=ROI_STATUS_TAG)
                dpg.add_text(
                    "Working area: Full scan",
                    tag=WORKING_AREA_INFO_TAG,
                    wrap=320,
                )
                dpg.add_button(label="Rectangle ROI mode", callback=_reset_roi_callback)
                dpg.add_button(label="Polygon ROI mode", callback=_polygon_mode_callback)
                dpg.add_button(label="Finish polygon", callback=_finish_polygon_callback)
                dpg.add_button(label="Clear polygon", callback=_clear_polygon_callback)
                dpg.add_button(label="Apply selection", callback=_apply_selection_callback)
                dpg.add_button(label="Clear selection", callback=_clear_selection_callback)
                dpg.pop_container_stack()

                display_header = dpg.add_collapsing_header(
                    label="Display",
                    default_open=True,
                )
                dpg.push_container_stack(display_header)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Reset view", callback=_reset_view_callback)
                    dpg.add_button(label="Zoom in", callback=_zoom_in_callback)
                    dpg.add_button(label="Zoom out", callback=_zoom_out_callback)
                dpg.add_checkbox(
                    label="Show ROI overlay",
                    tag=SHOW_ROI_OVERLAY_TAG,
                    default_value=True,
                    callback=lambda: _redraw_preview(),
                )
                dpg.add_checkbox(
                    label="Show mask remove edits",
                    tag=SHOW_MASK_REMOVE_EDITS_TAG,
                    default_value=True,
                    callback=lambda: _redraw_preview(),
                )
                dpg.add_checkbox(
                    label="Show mask add edits",
                    tag=SHOW_MASK_ADD_EDITS_TAG,
                    default_value=True,
                    callback=lambda: _redraw_preview(),
                )
                dpg.add_checkbox(
                    label="Show accepted holes",
                    tag=SHOW_ACCEPTED_HOLES_TAG,
                    default_value=True,
                    callback=lambda: _redraw_preview(),
                )
                dpg.add_checkbox(
                    label="Show rejected holes",
                    tag=SHOW_REJECTED_HOLES_TAG,
                    default_value=True,
                    callback=lambda: _redraw_preview(),
                )
                dpg.add_checkbox(
                    label="Show ungrouped holes",
                    tag=SHOW_UNGROUPED_HOLES_TAG,
                    default_value=True,
                    callback=lambda: _redraw_preview(),
                )
                dpg.add_text("Max displayed hole diameter mm")
                dpg.add_input_float(
                    tag=MAX_DISPLAYED_HOLE_DIAMETER_TAG,
                    default_value=200.0,
                    min_value=0.0,
                    width=-1,
                    callback=lambda: _redraw_preview(),
                )
                dpg.add_checkbox(
                    label="Show oversized holes",
                    tag=SHOW_OVERSIZED_HOLES_TAG,
                    default_value=False,
                    callback=lambda: _redraw_preview(),
                )
                dpg.add_checkbox(
                    label="Show contour",
                    tag=SHOW_CONTOUR_TAG,
                    default_value=True,
                    callback=lambda: _redraw_preview(),
                )
                dpg.add_checkbox(
                    label="Show processing mask",
                    tag=SHOW_PROCESSING_MASK_TAG,
                    default_value=False,
                    callback=lambda: _redraw_preview(),
                )
                dpg.add_checkbox(
                    label="Show mixed contour lines",
                    tag=SHOW_MIXED_CONTOUR_LINES_TAG,
                    default_value=True,
                    callback=lambda: _redraw_preview(),
                )
                dpg.add_checkbox(
                    label="Show mixed contour polyline gaps",
                    tag=SHOW_MIXED_CONTOUR_GAPS_TAG,
                    default_value=True,
                    callback=lambda: _redraw_preview(),
                )
                dpg.pop_container_stack()

                holes_header = dpg.add_collapsing_header(
                    label="Holes",
                    default_open=False,
                )
                dpg.push_container_stack(holes_header)
                dpg.add_text("Holes: not searched", tag=HOLE_DETECTION_INFO_TAG)
                dpg.add_text("Min diameter, mm")
                dpg.add_input_float(
                    tag=HOLE_MIN_DIAMETER_TAG,
                    default_value=8.0,
                    min_value=0.0,
                    width=-1,
                )
                dpg.add_text("Max diameter, mm (<= 0: unlimited)")
                dpg.add_input_float(
                    tag=HOLE_MAX_DIAMETER_TAG,
                    default_value=0.0,
                    width=-1,
                )
                advanced_holes_header = dpg.add_collapsing_header(
                    label="Advanced hole detection",
                    tag=HOLE_ADVANCED_HEADER_TAG,
                    default_open=False,
                )
                dpg.push_container_stack(advanced_holes_header)
                dpg.add_checkbox(
                    label="Keep largest component",
                    tag=HOLE_KEEP_LARGEST_TAG,
                    default_value=False,
                    callback=_keep_largest_component_callback,
                )
                dpg.add_text("Min circularity")
                dpg.add_input_float(
                    tag=HOLE_MIN_CIRCULARITY_TAG,
                    default_value=0.55,
                    min_value=0.0,
                    width=-1,
                )
                dpg.bind_item_handler_registry(
                    HOLE_MIN_CIRCULARITY_TAG,
                    "hole_detection_commit_handlers",
                )
                dpg.add_text("Max circle error ratio")
                dpg.add_input_float(
                    tag=HOLE_MAX_ERROR_RATIO_TAG,
                    default_value=0.18,
                    min_value=0.0,
                    width=-1,
                )
                dpg.bind_item_handler_registry(
                    HOLE_MAX_ERROR_RATIO_TAG,
                    "hole_detection_commit_handlers",
                )
                dpg.pop_container_stack()
                dpg.add_button(
                    label="Find holes",
                    tag=FIND_HOLES_BUTTON_TAG,
                    callback=_find_holes_callback,
                    width=-1,
                    enabled=False,
                )
                dpg.add_text(
                    "Hover a hole candidate for details.",
                    tag=HOLE_HOVER_INFO_TAG,
                    wrap=320,
                )
                dpg.pop_container_stack()

                legacy_holes_header = dpg.add_collapsing_header(
                    label="Legacy holes editor",
                    default_open=False,
                )
                dpg.push_container_stack(legacy_holes_header)
                dpg.add_text(
                    "holes total: 0\naccepted: 0\nrejected: 0\ngroups count: 0",
                    tag=HOLES_STATS_TAG,
                )
                dpg.add_button(label="Clear holes", callback=_clear_holes_callback)
                dpg.add_button(
                    label="Save edited holes JSON",
                    callback=_save_edited_holes_callback,
                )
                dpg.add_text("Hole groups")
                with dpg.child_window(
                    tag=HOLE_GROUPS_CONTAINER_TAG,
                    height=220,
                    border=True,
                    horizontal_scrollbar=False,
                ):
                    dpg.add_text("No hole groups loaded.", wrap=285)
                dpg.add_text("Move hole between groups")
                dpg.add_text("Selected hole ID")
                dpg.add_input_int(
                    tag=MOVE_HOLE_ID_TAG,
                    default_value=0,
                    min_value=0,
                    width=-1,
                )
                dpg.add_text("Target group ID")
                dpg.add_combo(
                    [],
                    tag=MOVE_HOLE_TARGET_GROUP_TAG,
                    width=-1,
                )
                dpg.add_button(
                    label="Move hole to group",
                    callback=_move_hole_to_group_callback,
                )
                dpg.add_button(
                    label="Accept selected hole",
                    callback=_accept_selected_hole_callback,
                    width=-1,
                )
                dpg.add_button(
                    label="Reject selected hole",
                    callback=_reject_selected_hole_callback,
                    width=-1,
                )
                dpg.add_text("Manual add hole")
                dpg.add_checkbox(
                    label="Pick hole center from preview",
                    tag=MANUAL_HOLE_PICK_TAG,
                    default_value=False,
                    callback=_manual_hole_pick_callback,
                )
                dpg.add_text("X")
                dpg.add_input_float(
                    tag=MANUAL_HOLE_X_TAG,
                    default_value=0.0,
                    width=-1,
                )
                dpg.add_text("Y")
                dpg.add_input_float(
                    tag=MANUAL_HOLE_Y_TAG,
                    default_value=0.0,
                    width=-1,
                )
                dpg.add_text("Diameter mm")
                dpg.add_input_float(
                    tag=MANUAL_HOLE_DIAMETER_TAG,
                    default_value=6.0,
                    min_value=0.001,
                    width=-1,
                )
                dpg.add_button(
                    label="Add manual hole",
                    callback=_add_manual_hole_callback,
                    width=-1,
                )
                dpg.add_text("Edit hole group")
                dpg.add_text("Target group ID")
                dpg.add_combo(
                    [],
                    tag=EDIT_GROUP_TARGET_TAG,
                    width=-1,
                )
                dpg.add_text("New group diameter mm")
                dpg.add_input_float(
                    tag=EDIT_GROUP_DIAMETER_TAG,
                    default_value=1.0,
                    min_value=0.001,
                    width=-1,
                )
                dpg.add_button(
                    label="Apply group diameter",
                    callback=_apply_group_diameter_callback,
                )
                dpg.pop_container_stack()

                contour_header = dpg.add_collapsing_header(
                    label="Contour",
                    default_open=False,
                )
                dpg.push_container_stack(contour_header)
                dpg.add_text(
                    "Preliminary contour: not built\nPoints: 0",
                    tag=CONTOUR_INFO_TAG,
                )
                dpg.add_text("Threshold mode")
                dpg.add_combo(
                    ["Auto", "Manual"],
                    tag=CONTOUR_THRESHOLD_MODE_TAG,
                    default_value="Auto",
                    width=-1,
                    callback=_contour_threshold_settings_callback,
                )
                dpg.add_text("Manual threshold")
                dpg.add_input_float(
                    tag=CONTOUR_MANUAL_THRESHOLD_TAG,
                    default_value=1.0,
                    width=-1,
                    enabled=False,
                )
                dpg.bind_item_handler_registry(
                    CONTOUR_MANUAL_THRESHOLD_TAG,
                    "manual_threshold_commit_handlers",
                )
                dpg.add_text("Fill holes max area (cells; 0: disabled)")
                dpg.add_input_int(
                    tag=CONTOUR_FILL_HOLES_AREA_TAG,
                    default_value=0,
                    min_value=0,
                    min_clamped=True,
                    width=-1,
                    callback=_contour_fill_holes_settings_callback,
                )
                dpg.add_button(
                    label="Find contour",
                    callback=_find_contour_callback,
                    width=-1,
                )
                dpg.add_button(label="Clear contour", callback=_clear_contour_callback)
                dpg.add_text(
                    "Mixed contour file: -\nElements: 0\nLines: 0\nPolyline gaps: 0",
                    tag=MIXED_CONTOUR_INFO_TAG,
                )
                dpg.add_button(
                    label="Clear mixed contour",
                    callback=_clear_mixed_contour_callback,
                )
                dpg.pop_container_stack()

                mask_header = dpg.add_collapsing_header(
                    label="Mask editing",
                    default_open=False,
                )
                dpg.push_container_stack(mask_header)
                dpg.add_button(
                    label="Edit processing mask",
                    tag=MASK_BRUSH_BUTTON_TAG,
                    callback=_mask_brush_mode_callback,
                    width=-1,
                    enabled=False,
                )
                dpg.add_text("Brush mode")
                dpg.add_combo(
                    ["Remove from mask", "Add to mask"],
                    tag=BRUSH_MODE_TAG,
                    default_value="Remove from mask",
                    width=-1,
                    callback=_brush_settings_callback,
                    enabled=False,
                )
                dpg.add_text("Brush diameter, mm")
                dpg.add_input_float(
                    tag=BRUSH_SIZE_TAG,
                    default_value=5.0,
                    min_value=0.001,
                    width=-1,
                    callback=_brush_settings_callback,
                    enabled=False,
                )
                dpg.add_text("Mask edit strokes: 0", tag=BRUSH_EDITS_COUNT_TAG)
                dpg.add_button(
                    label="Clear edits",
                    tag=MASK_CLEAR_BUTTON_TAG,
                    callback=_clear_mask_edits_callback,
                    enabled=False,
                )
                dpg.pop_container_stack()

                legacy_header = dpg.add_collapsing_header(
                    label="Legacy / Debug",
                    default_open=False,
                )
                dpg.push_container_stack(legacy_header)
                dpg.add_text("Prepared artifacts")
                dpg.add_button(
                    label="Open PNG",
                    callback=lambda: dpg.show_item("open_png_dialog"),
                )
                dpg.add_button(
                    label="Load processing result",
                    callback=lambda: dpg.show_item("open_processing_result_dialog"),
                )
                dpg.add_button(
                    label="Load result by source file",
                    callback=lambda: dpg.show_item("open_source_result_dialog"),
                )
                dpg.add_button(
                    label="Load report.txt",
                    callback=lambda: dpg.show_item("open_report_dialog"),
                )
                dpg.add_button(
                    label="Load holes.json",
                    callback=lambda: dpg.show_item("open_holes_dialog"),
                )
                dpg.add_button(
                    label="Load contour CSV",
                    callback=lambda: dpg.show_item("open_contour_dialog"),
                )
                dpg.add_button(
                    label="Load mixed contour JSON",
                    callback=lambda: dpg.show_item("open_mixed_contour_dialog"),
                )
                dpg.add_button(
                    label="Save mask edits JSON",
                    callback=_save_mask_edits_callback,
                )
                dpg.add_text(
                    "grid_min_x=-\n"
                    "grid_min_y=-\n"
                    "last brush image x/y: -\n"
                    "last brush world x/y: -",
                    tag=LAST_BRUSH_DEBUG_TAG,
                )
                dpg.add_separator()
                dpg.add_text("Legacy grid metadata")
                dpg.add_input_float(
                    label="grid_min_x",
                    tag=PARAM_GRID_MIN_X,
                    default_value=0.0,
                    width=-1,
                )
                dpg.add_input_float(
                    label="grid_min_y",
                    tag=PARAM_GRID_MIN_Y,
                    default_value=0.0,
                    width=-1,
                )
                dpg.add_input_float(
                    label="cell_size",
                    tag=PARAM_CELL_SIZE,
                    default_value=1.0,
                    width=-1,
                )
                dpg.add_input_int(
                    label="original_grid_width",
                    tag=PARAM_GRID_WIDTH,
                    default_value=0,
                    min_value=0,
                    width=-1,
                )
                dpg.add_input_int(
                    label="original_grid_height",
                    tag=PARAM_GRID_HEIGHT,
                    default_value=0,
                    min_value=0,
                    width=-1,
                )
                dpg.add_text("Demo summary file: -", tag=DEMO_SUMMARY_INFO_TAG)
                dpg.add_text("Coordinate debug")
                dpg.add_text(
                    "mouse_screen_x=- mouse_screen_y=-\n"
                    "draw_min_x=- draw_min_y=-\n"
                    "local_mouse_x=- local_mouse_y=-\n"
                    "image_x=- image_y=-\n"
                    "zoom=1\n"
                    "pan_x=0 pan_y=0\n"
                    "image_origin_x=0 image_origin_y=0",
                    tag=DEBUG_COORDS_TAG,
                )
                dpg.add_separator()
                dpg.add_text("Rectangle ROI")
                dpg.add_input_text(
                    tag=ROI_OUTPUT_TAG,
                    readonly=True,
                    multiline=True,
                    width=-1,
                    height=55,
                )
                dpg.add_separator()
                dpg.add_text("Polygon ROI")
                dpg.add_text("Polygon points count: 0", tag=POLYGON_COUNT_TAG)
                dpg.add_text("Last point: -", tag=POLYGON_LAST_TAG)
                dpg.add_input_text(
                    tag=POLYGON_OUTPUT_TAG,
                    readonly=True,
                    multiline=True,
                    width=-1,
                    height=90,
                )
                with dpg.tree_node(label="Full polygon points", default_open=False):
                    with dpg.child_window(height=220, border=True):
                        dpg.add_text("Polygon points: none", tag=POLYGON_POINTS_TAG)
                dpg.add_separator()
                with dpg.tree_node(label="Command generator", default_open=False):
                    dpg.add_text("input_file_path")
                    dpg.add_input_text(
                        tag=CMD_INPUT_FILE_TAG,
                        width=-1,
                    )
                    dpg.add_text("cell")
                    dpg.add_input_float(
                        tag=CMD_CELL_TAG,
                        default_value=0.5,
                        width=-1,
                    )
                    dpg.add_text("threshold")
                    dpg.add_input_text(
                        tag=CMD_THRESHOLD_TAG,
                        default_value="auto",
                        width=-1,
                    )
                    dpg.add_text("fill_holes_area")
                    dpg.add_input_int(
                        tag=CMD_FILL_HOLES_TAG,
                        default_value=0,
                        min_value=0,
                        width=-1,
                    )
                    dpg.add_text("boundary_width_mm")
                    dpg.add_input_float(
                        tag=CMD_BOUNDARY_WIDTH_TAG,
                        default_value=5.0,
                        width=-1,
                    )
                    dpg.add_text("simplify_mm")
                    dpg.add_input_float(
                        tag=CMD_SIMPLIFY_TAG,
                        default_value=0.0,
                        width=-1,
                    )
                    dpg.add_text("mask_edits")
                    dpg.add_input_text(
                        tag=CMD_MASK_EDITS_TAG,
                        width=-1,
                    )
                    dpg.add_checkbox(label="keep_largest", tag=CMD_KEEP_LARGEST_TAG)
                    dpg.add_checkbox(label="contour", tag=CMD_CONTOUR_TAG)
                    dpg.add_checkbox(label="dxf", tag=CMD_DXF_TAG)
                    dpg.add_checkbox(label="export_clean", tag=CMD_EXPORT_CLEAN_TAG)
                    dpg.add_checkbox(
                        label="export_boundary",
                        tag=CMD_EXPORT_BOUNDARY_TAG,
                    )
                    dpg.add_checkbox(label="holes", tag=CMD_HOLES_TAG)
                    dpg.add_button(
                        label="Generate command",
                        callback=_generate_command_callback,
                    )
                    dpg.add_input_text(
                        tag=COMMAND_OUTPUT_TAG,
                        readonly=True,
                        multiline=True,
                        width=-1,
                        height=120,
                    )
                    dpg.add_button(
                        label="Copy command to clipboard",
                        callback=_copy_command_callback,
                    )
                dpg.pop_container_stack()

        with dpg.group(horizontal=True):
            dpg.add_text(
                "Select a point cloud and build the density map.",
                tag=STATUS_TAG,
                wrap=430,
            )
            dpg.add_text("Zoom: 100%", tag=ZOOM_TEXT_TAG)
            dpg.add_text("World X: - | World Y: -", tag=COORDS_TAG)

    dpg.create_viewport(title="Point Contour Preview Viewer", width=1200, height=850)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    while dpg.is_dearpygui_running():
        _process_density_worker_messages()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    run()
