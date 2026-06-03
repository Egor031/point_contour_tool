from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import dearpygui.dearpygui as dpg
import numpy as np


TEXTURE_TAG = "preview_texture"
SELECTION_TEXTURE_TAG = "selection_dim_texture"
IMAGE_TAG = "preview_drawlist"
POLYGON_LAYER_TAG = "polygon_overlay_layer"
SELECTION_LAYER_TAG = "selection_dim_layer"
MASK_EDITS_LAYER_TAG = "mask_edits_layer"
HOLES_LAYER_TAG = "holes_overlay_layer"
SHOW_ROI_OVERLAY_TAG = "show_roi_overlay"
SHOW_MASK_REMOVE_EDITS_TAG = "show_mask_remove_edits"
SHOW_MASK_ADD_EDITS_TAG = "show_mask_add_edits"
SHOW_ACCEPTED_HOLES_TAG = "show_accepted_holes"
SHOW_REJECTED_HOLES_TAG = "show_rejected_holes"
STATUS_TAG = "status_text"
COORDS_TAG = "coords_text"
ROI_STATUS_TAG = "roi_status_text"
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
LAST_BRUSH_DEBUG_TAG = "last_brush_debug_text"
HOLES_STATS_TAG = "holes_stats_text"

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
CANVAS_WIDTH = 2400
CANVAS_HEIGHT = 1600

state = {
    "image_width": 0,
    "image_height": 0,
    "zoom": 1.0,
    "pan_x": 0.0,
    "pan_y": 0.0,
    "last_pan_mouse": None,
    "roi_first_world": None,
    "rectangle_roi": None,
    "polygon_finished": False,
    "selection_applied": False,
    "editing_overlay_visible": True,
    "selection_kind": None,
    "selection_polygon_points": [],
    "mode": "rectangle",
    "polygon_points": [],
    "mask_edits": [],
    "last_brush_image": None,
    "last_brush_world": None,
    "active_brush_stroke_id": None,
    "next_brush_stroke_id": 1,
    "holes": [],
    "report_loaded": False,
}


def _set_status(message: str) -> None:
    dpg.set_value(STATUS_TAG, message)


def _format_cli_float(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _get_preview_params() -> tuple[float, float, float, int, int] | None:
    grid_min_x = float(dpg.get_value(PARAM_GRID_MIN_X))
    grid_min_y = float(dpg.get_value(PARAM_GRID_MIN_Y))
    cell_size = float(dpg.get_value(PARAM_CELL_SIZE))
    grid_width = int(dpg.get_value(PARAM_GRID_WIDTH))
    grid_height = int(dpg.get_value(PARAM_GRID_HEIGHT))

    if cell_size <= 0 or grid_width <= 0 or grid_height <= 0:
        return None

    return grid_min_x, grid_min_y, cell_size, grid_width, grid_height


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

    image_x = (canvas_x - origin_x) / zoom
    image_y = (canvas_y - origin_y) / zoom

    image_width = int(state["image_width"])
    image_height = int(state["image_height"])
    if image_x < 0 or image_y < 0 or image_x >= image_width or image_y >= image_height:
        return None

    return image_x, image_y


def image_to_drawlist(image_x: float, image_y: float) -> tuple[float, float]:
    origin_x, origin_y = _image_origin()
    zoom = float(state["zoom"])
    return origin_x + image_x * zoom, origin_y + image_y * zoom


def image_to_world(
    image_x: float,
    image_y: float,
) -> tuple[float, float, float, float] | None:
    image_width = int(state["image_width"])
    image_height = int(state["image_height"])
    if image_width <= 0 or image_height <= 0:
        return None

    params = _get_preview_params()
    if params is None:
        return None

    grid_min_x, grid_min_y, cell_size, _grid_width, grid_height = params
    displayed_grid_x = image_x
    original_grid_y = (grid_height - 1) - image_y

    world_x = grid_min_x + (displayed_grid_x + 0.5) * cell_size
    world_y = grid_min_y + (original_grid_y + 0.5) * cell_size

    return displayed_grid_x, original_grid_y, world_x, world_y


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
    if not dpg.does_item_exist(TEXTURE_TAG):
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
            TEXTURE_TAG,
            (pan_x, pan_y),
            (pan_x + scaled_width, pan_y + scaled_height),
        )

    _redraw_selection_overlay()
    _redraw_polygon_overlay()
    _redraw_mask_edits_overlay()
    _redraw_holes_overlay()


def _redraw_selection_overlay() -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if dpg.does_item_exist(SELECTION_LAYER_TAG):
        dpg.delete_item(SELECTION_LAYER_TAG)

    if not state["selection_applied"] or not dpg.does_item_exist(SELECTION_TEXTURE_TAG):
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


def _mouse_to_world() -> tuple[float, float, float, float, float, float] | None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return None

    if not dpg.is_item_hovered(IMAGE_TAG):
        return None

    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
    return screen_to_world(mouse_x, mouse_y)


def _world_to_image_pixel(world_x: float, world_y: float) -> tuple[float, float] | None:
    image_width = int(state["image_width"])
    image_height = int(state["image_height"])
    if image_width <= 0 or image_height <= 0:
        return None

    params = _get_preview_params()
    if params is None:
        return None

    grid_min_x, grid_min_y, cell_size, _grid_width, grid_height = params

    grid_x = (world_x - grid_min_x) / cell_size - 0.5
    original_grid_y = (world_y - grid_min_y) / cell_size - 0.5
    displayed_grid_y = (grid_height - 1) - original_grid_y

    pixel_x = grid_x
    pixel_y = displayed_grid_y

    return image_to_drawlist(pixel_x, pixel_y)


def _world_to_original_image_pixel(
    world_x: float,
    world_y: float,
) -> tuple[float, float] | None:
    image_width = int(state["image_width"])
    image_height = int(state["image_height"])
    if image_width <= 0 or image_height <= 0:
        return None

    params = _get_preview_params()
    if params is None:
        return None

    grid_min_x, grid_min_y, cell_size, _grid_width, grid_height = params

    grid_x = (world_x - grid_min_x) / cell_size - 0.5
    original_grid_y = (world_y - grid_min_y) / cell_size - 0.5
    displayed_grid_y = (grid_height - 1) - original_grid_y

    pixel_x = grid_x
    pixel_y = displayed_grid_y

    return pixel_x, pixel_y


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


def _undo_last_polygon_point() -> bool:
    if state["selection_applied"]:
        dpg.set_value(ROI_STATUS_TAG, "Edit selection first.")
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

    if not state["editing_overlay_visible"]:
        return

    if not _display_layer_enabled(SHOW_ROI_OVERLAY_TAG):
        return

    dpg.add_draw_layer(parent=IMAGE_TAG, tag=POLYGON_LAYER_TAG)

    rectangle_roi = state["rectangle_roi"]
    if rectangle_roi is not None:
        min_x, min_y, max_x, max_y = rectangle_roi
        pixel_a = _world_to_image_pixel(min_x, min_y)
        pixel_b = _world_to_image_pixel(max_x, max_y)
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

    pixel_points = []
    for world_x, world_y in state["polygon_points"]:
        pixel_point = _world_to_image_pixel(world_x, world_y)
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


def _brush_radius_to_draw_radius(radius_mm: float) -> float:
    params = _get_preview_params()
    if params is None:
        return 1.0

    _grid_min_x, _grid_min_y, cell_size, grid_width, _grid_height = params
    image_width = int(state["image_width"])
    if cell_size <= 0 or grid_width <= 0 or image_width <= 0:
        return 1.0

    radius_cells = radius_mm / cell_size
    radius_image_px = radius_cells * image_width / grid_width
    return max(1.0, radius_image_px * float(state["zoom"]))


def _get_brush_edit_mode() -> str:
    if not dpg.does_item_exist(BRUSH_MODE_TAG):
        return "remove"

    mode_label = str(dpg.get_value(BRUSH_MODE_TAG))
    if mode_label == "Add to mask":
        return "add"

    return "remove"


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

        pixel_point = _world_to_image_pixel(edit["x"], edit["y"])
        if pixel_point is None:
            continue

        radius = _brush_radius_to_draw_radius(edit["radius_mm"])
        color, fill = _brush_edit_colors(edit_mode)
        dpg.draw_circle(
            pixel_point,
            radius,
            color=color,
            fill=fill,
            thickness=2,
            parent=MASK_EDITS_LAYER_TAG,
        )


def _hole_draw_radius(radius_mm: float) -> float:
    params = _get_preview_params()
    if params is None:
        return 1.0

    _grid_min_x, _grid_min_y, cell_size, _grid_width, _grid_height = params
    if cell_size <= 0:
        return 1.0

    return max(1.0, radius_mm / cell_size * float(state["zoom"]))


def _hole_overlay_colors(
    accepted: bool,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if accepted:
        return (80, 220, 120, 230), (80, 220, 120, 45)

    return (255, 140, 50, 230), (255, 110, 40, 45)


def _redraw_holes_overlay() -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if dpg.does_item_exist(HOLES_LAYER_TAG):
        dpg.delete_item(HOLES_LAYER_TAG)

    holes = state["holes"]
    if not holes:
        return

    dpg.add_draw_layer(parent=IMAGE_TAG, tag=HOLES_LAYER_TAG)

    for hole in holes:
        accepted = bool(hole.get("accepted", False))
        if accepted:
            if not _display_layer_enabled(SHOW_ACCEPTED_HOLES_TAG):
                continue
        elif not _display_layer_enabled(SHOW_REJECTED_HOLES_TAG):
            continue

        center = _world_to_image_pixel(hole["center_x"], hole["center_y"])
        if center is None:
            continue

        radius = _hole_draw_radius(float(hole["radius"]))
        color, fill = _hole_overlay_colors(accepted)
        dpg.draw_circle(
            center,
            radius,
            color=color,
            fill=fill,
            thickness=2,
            parent=HOLES_LAYER_TAG,
        )
        dpg.draw_text(
            (center[0] + radius + 4, center[1] - 7),
            str(hole.get("id", "")),
            color=color,
            size=14,
            parent=HOLES_LAYER_TAG,
        )


def _mouse_move_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    _update_pan_from_mouse()
    _update_mask_brush_from_mouse()
    _update_debug_coords()

    coords = _mouse_to_world()
    if coords is None:
        dpg.set_value(
            COORDS_TAG,
            "pixel_x=- pixel_y=- | grid_x=- grid_y=- | world_x=- world_y=-",
        )
        return

    pixel_x, pixel_y, grid_x, grid_y, world_x, world_y = coords
    dpg.set_value(
        COORDS_TAG,
        "pixel_x={:.1f} pixel_y={:.1f} | "
        "grid_x={:.2f} grid_y={:.2f} | "
        "world_x={:.6f} world_y={:.6f}".format(
            pixel_x,
            pixel_y,
            grid_x,
            grid_y,
            world_x,
            world_y,
        ),
    )


def _update_mask_brush_from_mouse() -> None:
    if state["mode"] != "mask_brush":
        state["last_brush_image"] = None
        state["last_brush_world"] = None
        state["active_brush_stroke_id"] = None
        return

    if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
        state["last_brush_image"] = None
        state["last_brush_world"] = None
        state["active_brush_stroke_id"] = None
        return

    if not dpg.does_item_exist(IMAGE_TAG) or not dpg.is_item_hovered(IMAGE_TAG):
        return

    if _warn_preview_grid_params_not_loaded():
        return

    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
    coords = screen_to_world(mouse_x, mouse_y)
    if coords is None:
        return

    image_x, image_y, _grid_x, _grid_y, world_x, world_y = coords
    radius_mm = float(dpg.get_value(BRUSH_SIZE_TAG))
    radius_mm = max(0.001, radius_mm)
    brush_mode = _get_brush_edit_mode()
    stroke_id = state["active_brush_stroke_id"]
    if stroke_id is None:
        stroke_id = int(state["next_brush_stroke_id"])
        state["active_brush_stroke_id"] = stroke_id
        state["next_brush_stroke_id"] = stroke_id + 1

    last_brush_image = state["last_brush_image"]
    if last_brush_image is not None:
        last_x, last_y = last_brush_image
        dx = image_x - last_x
        dy = image_y - last_y
        if (dx * dx + dy * dy) ** 0.5 < 2.0:
            return

    state["mask_edits"].append(
        {
            "stroke_id": stroke_id,
            "mode": brush_mode,
            "x": world_x,
            "y": world_y,
            "radius_mm": radius_mm,
        }
    )
    state["last_brush_image"] = (image_x, image_y)
    state["last_brush_world"] = (world_x, world_y)
    _update_mask_edits_count()
    _update_last_brush_debug()
    _redraw_preview()


def _update_mask_edits_count() -> None:
    if dpg.does_item_exist(BRUSH_EDITS_COUNT_TAG):
        dpg.set_value(
            BRUSH_EDITS_COUNT_TAG,
            f"Brush edits count: {len(state['mask_edits'])}",
        )


def _update_holes_stats() -> None:
    if not dpg.does_item_exist(HOLES_STATS_TAG):
        return

    holes = state["holes"]
    accepted_count = sum(1 for hole in holes if bool(hole.get("accepted", False)))
    rejected_count = len(holes) - accepted_count
    dpg.set_value(
        HOLES_STATS_TAG,
        "holes total: {}\naccepted: {}\nrejected: {}".format(
            len(holes),
            accepted_count,
            rejected_count,
        ),
    )


def _undo_last_brush_stroke() -> bool:
    edits = state["mask_edits"]
    if not edits:
        _set_status("No brush strokes to undo.")
        return False

    last_edit = edits[-1]
    stroke_id = last_edit.get("stroke_id")
    if stroke_id is None:
        edits.pop()
        removed_count = 1
    else:
        kept_edits = [edit for edit in edits if edit.get("stroke_id") != stroke_id]
        removed_count = len(edits) - len(kept_edits)
        state["mask_edits"] = kept_edits

    state["active_brush_stroke_id"] = None
    state["last_brush_image"] = None
    state["last_brush_world"] = None
    _update_mask_edits_count()
    _update_last_brush_debug()
    _redraw_preview()
    _set_status(f"Undid brush stroke: removed {removed_count} edits.")
    return True


def _undo_last_brush_stroke_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    _undo_last_brush_stroke()


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
    if not dpg.does_item_exist(IMAGE_TAG):
        state["last_pan_mouse"] = None
        return

    pan_button_down = (
        dpg.is_mouse_button_down(dpg.mvMouseButton_Middle)
        or dpg.is_mouse_button_down(dpg.mvMouseButton_Right)
    )

    if not pan_button_down:
        state["last_pan_mouse"] = None
        return

    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)

    if state["last_pan_mouse"] is None:
        if dpg.is_item_hovered(IMAGE_TAG):
            state["last_pan_mouse"] = (mouse_x, mouse_y)
        return

    last_x, last_y = state["last_pan_mouse"]
    dx = mouse_x - last_x
    dy = mouse_y - last_y

    state["pan_x"] = float(state["pan_x"]) + dx
    state["pan_y"] = float(state["pan_y"]) + dy
    state["last_pan_mouse"] = (mouse_x, mouse_y)
    _redraw_preview()


def _mouse_click_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    if state["mode"] == "mask_brush":
        return

    if _warn_preview_grid_params_not_loaded():
        return

    coords = _mouse_to_world()
    if coords is None:
        return

    *_unused, world_x, world_y = coords

    if state["mode"] == "polygon":
        state["polygon_points"].append((world_x, world_y))
        state["polygon_finished"] = False
        state["selection_applied"] = False
        state["editing_overlay_visible"] = True
        dpg.set_value(
            ROI_STATUS_TAG,
            f"Polygon ROI: added point {len(state['polygon_points'])}.",
        )
        dpg.set_value(POLYGON_OUTPUT_TAG, "")
        _update_polygon_points_text()
        _redraw_preview()
        return

    first_world = state["roi_first_world"]

    if first_world is None:
        state["roi_first_world"] = (world_x, world_y)
        dpg.set_value(
            ROI_STATUS_TAG,
            f"ROI first corner: x={world_x:.6f}, y={world_y:.6f}",
        )
        dpg.set_value(ROI_OUTPUT_TAG, "")
        return

    first_x, first_y = first_world
    min_x = min(first_x, world_x)
    min_y = min(first_y, world_y)
    max_x = max(first_x, world_x)
    max_y = max(first_y, world_y)

    state["roi_first_world"] = None
    state["rectangle_roi"] = (min_x, min_y, max_x, max_y)
    state["selection_applied"] = False
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
    state["roi_first_world"] = None
    state["mode"] = "rectangle"
    state["editing_overlay_visible"] = True
    dpg.set_value(ROI_STATUS_TAG, "ROI mode: click two image corners.")


def _polygon_mode_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    state["mode"] = "polygon"
    state["roi_first_world"] = None
    state["editing_overlay_visible"] = True
    dpg.set_value(ROI_STATUS_TAG, "Polygon ROI mode: click image points.")


def _mask_brush_mode_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    state["mode"] = "mask_brush"
    state["roi_first_world"] = None
    state["last_brush_image"] = None
    dpg.set_value(ROI_STATUS_TAG, "Mask brush mode: hold left mouse button.")


def _finish_polygon_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    _finish_polygon()


def _undo_last_polygon_point_callback(
    _sender=None,
    _app_data=None,
    _user_data=None,
) -> None:
    _undo_last_polygon_point()


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
    state["selection_applied"] = False
    state["editing_overlay_visible"] = True
    dpg.set_value(ROI_STATUS_TAG, "Polygon ROI ready.")
    dpg.set_value(POLYGON_OUTPUT_TAG, f'--roi-poly "{points_text}"')
    return True


def _clear_polygon_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    state["polygon_points"] = []
    state["polygon_finished"] = False
    state["selection_applied"] = False
    state["editing_overlay_visible"] = True
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

    if state["mode"] == "polygon" and (undo_shortcut or backspace):
        _undo_last_polygon_point()
        return

    if state["mode"] == "mask_brush" and undo_shortcut:
        _undo_last_brush_stroke()


def _clamp_pixel(value: float, upper_limit: int) -> int:
    return max(0, min(upper_limit - 1, int(round(value))))


def _build_selection_inside_mask() -> np.ndarray | None:
    image_width = int(state["image_width"])
    image_height = int(state["image_height"])
    if image_width <= 0 or image_height <= 0:
        return None

    inside = np.zeros((image_height, image_width), dtype=np.uint8)

    if state["polygon_finished"] and len(state["polygon_points"]) >= 3:
        points_pixels = []
        for world_x, world_y in state["polygon_points"]:
            pixel = _world_to_original_image_pixel(world_x, world_y)
            if pixel is None:
                return None
            pixel_x, pixel_y = pixel
            points_pixels.append(
                [
                    _clamp_pixel(pixel_x, image_width),
                    _clamp_pixel(pixel_y, image_height),
                ]
            )

        polygon = np.array(points_pixels, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(inside, [polygon], 1)
        state["selection_kind"] = "polygon"
        state["selection_polygon_points"] = list(state["polygon_points"])
        return inside

    rectangle_roi = state["rectangle_roi"]
    if rectangle_roi is not None:
        min_x, min_y, max_x, max_y = rectangle_roi
        pixel_a = _world_to_original_image_pixel(min_x, min_y)
        pixel_b = _world_to_original_image_pixel(max_x, max_y)
        if pixel_a is None or pixel_b is None:
            return None

        ax, ay = pixel_a
        bx, by = pixel_b
        left = _clamp_pixel(min(ax, bx), image_width)
        right = _clamp_pixel(max(ax, bx), image_width)
        top = _clamp_pixel(min(ay, by), image_height)
        bottom = _clamp_pixel(max(ay, by), image_height)
        inside[top : bottom + 1, left : right + 1] = 1
        state["selection_kind"] = "rectangle"
        state["selection_polygon_points"] = []
        return inside

    return None


def _update_selection_texture(inside_mask: np.ndarray) -> None:
    image_height, image_width = inside_mask.shape
    overlay = np.zeros((image_height, image_width, 4), dtype=np.float32)
    overlay[..., 3] = 0.62
    overlay[inside_mask > 0, 3] = 0.0

    if dpg.does_item_exist(SELECTION_TEXTURE_TAG):
        state["selection_applied"] = False
        _redraw_preview()
        dpg.delete_item(SELECTION_TEXTURE_TAG)

    with dpg.texture_registry():
        dpg.add_static_texture(
            width=image_width,
            height=image_height,
            default_value=overlay.ravel().tolist(),
            tag=SELECTION_TEXTURE_TAG,
        )


def _apply_selection_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    if (
        state["rectangle_roi"] is None
        and not state["polygon_finished"]
        and len(state["polygon_points"]) > 0
    ):
        if not _finish_polygon():
            return

    inside_mask = _build_selection_inside_mask()
    if inside_mask is None:
        dpg.set_value(ROI_STATUS_TAG, "No rectangle or finished polygon ROI to apply.")
        return

    _update_selection_texture(inside_mask)
    state["selection_applied"] = True
    state["editing_overlay_visible"] = False
    _redraw_preview()
    dpg.set_value(ROI_STATUS_TAG, "Selection applied.")


def _edit_selection_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    state["selection_applied"] = False
    state["editing_overlay_visible"] = True
    _redraw_preview()
    dpg.set_value(ROI_STATUS_TAG, "Selection edit mode.")


def _clear_selection_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    state["roi_first_world"] = None
    state["rectangle_roi"] = None
    state["polygon_points"] = []
    state["polygon_finished"] = False
    state["selection_applied"] = False
    state["editing_overlay_visible"] = True
    state["selection_kind"] = None
    state["selection_polygon_points"] = []

    dpg.set_value(ROI_OUTPUT_TAG, "")
    dpg.set_value(POLYGON_OUTPUT_TAG, "")
    _update_polygon_points_text()
    _redraw_preview()

    if dpg.does_item_exist(SELECTION_TEXTURE_TAG):
        dpg.delete_item(SELECTION_TEXTURE_TAG)

    dpg.set_value(ROI_STATUS_TAG, "Selection cleared.")


def _clear_mask_edits_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    state["mask_edits"] = []
    state["last_brush_image"] = None
    state["last_brush_world"] = None
    state["active_brush_stroke_id"] = None
    state["next_brush_stroke_id"] = 1
    _update_mask_edits_count()
    _update_last_brush_debug()
    _redraw_preview()
    _set_status("Mask edits cleared.")


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


def _show_png(path: str | Path) -> None:
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

    if dpg.does_item_exist(TEXTURE_TAG):
        dpg.delete_item(TEXTURE_TAG)

    with dpg.texture_registry():
        dpg.add_static_texture(
            width=width,
            height=height,
            default_value=data,
            tag=TEXTURE_TAG,
        )

    if dpg.does_item_exist("image_hint"):
        dpg.delete_item("image_hint")

    state["image_width"] = width
    state["image_height"] = height
    state["zoom"] = 1.0
    state["pan_x"] = 0.0
    state["pan_y"] = 0.0
    state["last_pan_mouse"] = None
    state["roi_first_world"] = None
    state["rectangle_roi"] = None
    state["polygon_finished"] = False
    state["selection_applied"] = False
    state["editing_overlay_visible"] = True
    state["selection_kind"] = None
    state["selection_polygon_points"] = []
    state["polygon_points"] = []
    state["mask_edits"] = []
    state["last_brush_image"] = None
    state["last_brush_world"] = None
    state["active_brush_stroke_id"] = None
    state["next_brush_stroke_id"] = 1

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
    _redraw_preview()
    _set_status(f"Loaded: {image_path}")


def _open_file_callback(_sender, app_data) -> None:
    selections = app_data.get("selections", {})
    if not selections:
        return

    selected_path = next(iter(selections.values()))
    _show_png(selected_path)


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
        "reject_reason": str(item.get("reject_reason", "")),
        "center_x": float(item["center_x"]),
        "center_y": float(item["center_y"]),
        "radius": float(item["radius"]),
        "diameter": float(item.get("diameter", float(item["radius"]) * 2.0)),
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
        if not isinstance(raw_holes, list):
            raise ValueError("holes must be a list")

        holes = []
        for item in raw_holes:
            if not isinstance(item, dict):
                raise ValueError("hole item must be an object")
            holes.append(_normalize_hole_json_item(item))
    except Exception as exc:
        _set_status(f"Could not parse holes.json: {exc}")
        return

    state["holes"] = holes
    _update_holes_stats()
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


def run() -> None:
    dpg.create_context()

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

    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=_mouse_move_callback)
        dpg.add_mouse_click_handler(
            button=dpg.mvMouseButton_Left,
            callback=_mouse_click_callback,
        )
        dpg.add_mouse_wheel_handler(callback=_mouse_wheel_callback)
        dpg.add_key_press_handler(callback=_key_press_callback)

    with dpg.window(label="Point Contour Preview Viewer", tag="main_window"):
        dpg.add_text("Preview viewer for density/mask PNG images.")
        dpg.add_button(
            label="Open PNG",
            callback=lambda: dpg.show_item("open_png_dialog"),
        )
        dpg.add_button(
            label="Load report.txt",
            callback=lambda: dpg.show_item("open_report_dialog"),
        )
        dpg.add_button(
            label="Load holes.json",
            callback=lambda: dpg.show_item("open_holes_dialog"),
        )
        dpg.add_text("No .asc/.xyz processing is performed here.", tag=STATUS_TAG)
        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_input_float(label="grid_min_x", tag=PARAM_GRID_MIN_X, default_value=0.0)
            dpg.add_input_float(label="grid_min_y", tag=PARAM_GRID_MIN_Y, default_value=0.0)
            dpg.add_input_float(label="cell_size", tag=PARAM_CELL_SIZE, default_value=1.0)

        with dpg.group(horizontal=True):
            dpg.add_input_int(
                label="original_grid_width",
                tag=PARAM_GRID_WIDTH,
                default_value=0,
                min_value=0,
            )
            dpg.add_input_int(
                label="original_grid_height",
                tag=PARAM_GRID_HEIGHT,
                default_value=0,
                min_value=0,
            )

        dpg.add_text(
            "pixel_x=- pixel_y=- | grid_x=- grid_y=- | world_x=- world_y=-",
            tag=COORDS_TAG,
        )
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset view", callback=_reset_view_callback)
            dpg.add_button(label="Zoom in", callback=_zoom_in_callback)
            dpg.add_button(label="Zoom out", callback=_zoom_out_callback)
            dpg.add_text("Zoom: 100%", tag=ZOOM_TEXT_TAG)
        dpg.add_separator()

        with dpg.group(horizontal=True):
            with dpg.child_window(
                tag="image_area",
                border=False,
                horizontal_scrollbar=False,
                no_scrollbar=True,
                no_scroll_with_mouse=True,
                width=-370,
            ):
                dpg.add_text(
                    "Open a density or mask preview PNG to view it here.",
                    tag="image_hint",
                )

            with dpg.child_window(tag="roi_side_panel", width=350, border=True):
                dpg.add_text("ROI tools")
                dpg.add_text("ROI mode: click two image corners.", tag=ROI_STATUS_TAG)
                dpg.add_button(label="Rectangle ROI mode", callback=_reset_roi_callback)
                dpg.add_button(label="Polygon ROI mode", callback=_polygon_mode_callback)
                dpg.add_button(label="Mask brush", callback=_mask_brush_mode_callback)
                dpg.add_button(label="Finish polygon", callback=_finish_polygon_callback)
                dpg.add_button(
                    label="Undo last polygon point",
                    callback=_undo_last_polygon_point_callback,
                )
                dpg.add_button(label="Clear polygon", callback=_clear_polygon_callback)
                dpg.add_button(label="Apply selection", callback=_apply_selection_callback)
                dpg.add_button(label="Edit selection", callback=_edit_selection_callback)
                dpg.add_button(label="Clear selection", callback=_clear_selection_callback)
                dpg.add_separator()
                dpg.add_text("Display layers")
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
                dpg.add_text(
                    "holes total: 0\naccepted: 0\nrejected: 0",
                    tag=HOLES_STATS_TAG,
                )
                dpg.add_separator()
                dpg.add_text("Mask edit brush")
                dpg.add_text("Brush mode")
                dpg.add_combo(
                    ["Remove from mask", "Add to mask"],
                    tag=BRUSH_MODE_TAG,
                    default_value="Remove from mask",
                    width=-1,
                )
                dpg.add_text("Brush size mm")
                dpg.add_input_float(
                    tag=BRUSH_SIZE_TAG,
                    default_value=5.0,
                    min_value=0.001,
                    width=-1,
                )
                dpg.add_text("Brush edits count: 0", tag=BRUSH_EDITS_COUNT_TAG)
                dpg.add_text(
                    "grid_min_x=-\n"
                    "grid_min_y=-\n"
                    "last brush image x/y: -\n"
                    "last brush world x/y: -",
                    tag=LAST_BRUSH_DEBUG_TAG,
                )
                dpg.add_button(
                    label="Undo last brush stroke",
                    callback=_undo_last_brush_stroke_callback,
                )
                dpg.add_button(label="Clear mask edits", callback=_clear_mask_edits_callback)
                dpg.add_button(
                    label="Save mask edits JSON",
                    callback=_save_mask_edits_callback,
                )
                dpg.add_separator()
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
                with dpg.tree_node(label="Command generator", default_open=True):
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

    dpg.create_viewport(title="Point Contour Preview Viewer", width=1200, height=850)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    run()
