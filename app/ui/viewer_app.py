from __future__ import annotations

from pathlib import Path

import dearpygui.dearpygui as dpg


TEXTURE_TAG = "preview_texture"
IMAGE_TAG = "preview_drawlist"
POLYGON_LAYER_TAG = "polygon_overlay_layer"
STATUS_TAG = "status_text"
COORDS_TAG = "coords_text"
ROI_STATUS_TAG = "roi_status_text"
ROI_OUTPUT_TAG = "roi_output_text"
POLYGON_COUNT_TAG = "polygon_count_text"
POLYGON_LAST_TAG = "polygon_last_text"
POLYGON_POINTS_TAG = "polygon_points_text"
POLYGON_OUTPUT_TAG = "polygon_output_text"
ZOOM_TEXT_TAG = "zoom_text"

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
    "mode": "rectangle",
    "polygon_points": [],
}


def _set_status(message: str) -> None:
    dpg.set_value(STATUS_TAG, message)


def _get_preview_params() -> tuple[float, float, float, int, int] | None:
    grid_min_x = float(dpg.get_value(PARAM_GRID_MIN_X))
    grid_min_y = float(dpg.get_value(PARAM_GRID_MIN_Y))
    cell_size = float(dpg.get_value(PARAM_CELL_SIZE))
    grid_width = int(dpg.get_value(PARAM_GRID_WIDTH))
    grid_height = int(dpg.get_value(PARAM_GRID_HEIGHT))

    if cell_size <= 0 or grid_width <= 0 or grid_height <= 0:
        return None

    return grid_min_x, grid_min_y, cell_size, grid_width, grid_height


def _scaled_image_size() -> tuple[int, int]:
    zoom = float(state["zoom"])
    width = max(1, int(round(int(state["image_width"]) * zoom)))
    height = max(1, int(round(int(state["image_height"]) * zoom)))

    return width, height


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

    _redraw_polygon_overlay()


def _mouse_to_world() -> tuple[float, float, float, float, float, float] | None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return None

    if not dpg.is_item_hovered(IMAGE_TAG):
        return None

    image_width = int(state["image_width"])
    image_height = int(state["image_height"])
    if image_width <= 0 or image_height <= 0:
        return None

    params = _get_preview_params()
    if params is None:
        return None

    grid_min_x, grid_min_y, cell_size, grid_width, grid_height = params

    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
    canvas_left, canvas_top = dpg.get_item_rect_min(IMAGE_TAG)

    canvas_x = mouse_x - canvas_left
    canvas_y = mouse_y - canvas_top

    scaled_width, scaled_height = _scaled_image_size()
    displayed_pixel_x = canvas_x - float(state["pan_x"])
    displayed_pixel_y = canvas_y - float(state["pan_y"])

    if (
        displayed_pixel_x < 0
        or displayed_pixel_y < 0
        or displayed_pixel_x >= scaled_width
        or displayed_pixel_y >= scaled_height
    ):
        return None

    zoom = float(state["zoom"])
    original_pixel_x = displayed_pixel_x / zoom
    original_pixel_y = displayed_pixel_y / zoom

    displayed_grid_x = original_pixel_x * grid_width / image_width
    displayed_grid_y = original_pixel_y * grid_height / image_height
    original_grid_y = (grid_height - 1) - displayed_grid_y

    world_x = grid_min_x + (displayed_grid_x + 0.5) * cell_size
    world_y = grid_min_y + (original_grid_y + 0.5) * cell_size

    return (
        displayed_pixel_x,
        displayed_pixel_y,
        displayed_grid_x,
        original_grid_y,
        world_x,
        world_y,
    )


def _world_to_image_pixel(world_x: float, world_y: float) -> tuple[float, float] | None:
    image_width = int(state["image_width"])
    image_height = int(state["image_height"])
    if image_width <= 0 or image_height <= 0:
        return None

    params = _get_preview_params()
    if params is None:
        return None

    grid_min_x, grid_min_y, cell_size, grid_width, grid_height = params

    grid_x = (world_x - grid_min_x) / cell_size
    original_grid_y = (world_y - grid_min_y) / cell_size
    displayed_grid_y = (grid_height - 1) - original_grid_y

    zoom = float(state["zoom"])
    pixel_x = grid_x * image_width / grid_width * zoom
    pixel_y = displayed_grid_y * image_height / grid_height * zoom

    return pixel_x + float(state["pan_x"]), pixel_y + float(state["pan_y"])


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


def _redraw_polygon_overlay() -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if dpg.does_item_exist(POLYGON_LAYER_TAG):
        dpg.delete_item(POLYGON_LAYER_TAG)

    dpg.add_draw_layer(parent=IMAGE_TAG, tag=POLYGON_LAYER_TAG)

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


def _mouse_move_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    _update_pan_from_mouse()

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
    coords = _mouse_to_world()
    if coords is None:
        return

    *_unused, world_x, world_y = coords

    if state["mode"] == "polygon":
        state["polygon_points"].append((world_x, world_y))
        dpg.set_value(
            ROI_STATUS_TAG,
            f"Polygon ROI: added point {len(state['polygon_points'])}.",
        )
        dpg.set_value(POLYGON_OUTPUT_TAG, "")
        _update_polygon_points_text()
        _redraw_polygon_overlay()
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
    dpg.set_value(ROI_STATUS_TAG, "ROI rectangle ready.")
    dpg.set_value(
        ROI_OUTPUT_TAG,
        f"--roi {min_x:.6f} {min_y:.6f} {max_x:.6f} {max_y:.6f}",
    )


def _mouse_wheel_callback(_sender=None, app_data=None, _user_data=None) -> None:
    if not dpg.does_item_exist(IMAGE_TAG):
        return

    if not dpg.is_item_hovered(IMAGE_TAG):
        return

    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
    canvas_left, canvas_top = dpg.get_item_rect_min(IMAGE_TAG)
    anchor = (mouse_x - canvas_left, mouse_y - canvas_top)

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
    dpg.set_value(ROI_STATUS_TAG, "ROI mode: click two image corners.")
    dpg.set_value(ROI_OUTPUT_TAG, "")


def _polygon_mode_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    state["mode"] = "polygon"
    state["roi_first_world"] = None
    dpg.set_value(ROI_STATUS_TAG, "Polygon ROI mode: click image points.")


def _finish_polygon_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    points = state["polygon_points"]
    if len(points) < 3:
        dpg.set_value(ROI_STATUS_TAG, "Polygon ROI needs at least 3 points.")
        dpg.set_value(POLYGON_OUTPUT_TAG, "")
        return

    points_text = ";".join(f"{x:.6f},{y:.6f}" for x, y in points)
    dpg.set_value(ROI_STATUS_TAG, "Polygon ROI ready.")
    dpg.set_value(POLYGON_OUTPUT_TAG, f'--roi-poly "{points_text}"')


def _clear_polygon_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    state["polygon_points"] = []
    dpg.set_value(POLYGON_OUTPUT_TAG, "")
    _update_polygon_points_text()
    _redraw_polygon_overlay()
    dpg.set_value(ROI_STATUS_TAG, "Polygon ROI cleared.")


def _show_png(path: str | Path) -> None:
    image_path = Path(path)

    try:
        width, height, _channels, data = dpg.load_image(str(image_path))
    except Exception as exc:
        _set_status(f"Could not load PNG: {exc}")
        return

    if dpg.does_item_exist(IMAGE_TAG):
        dpg.delete_item(IMAGE_TAG)

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
    state["polygon_points"] = []

    if int(dpg.get_value(PARAM_GRID_WIDTH)) <= 0:
        dpg.set_value(PARAM_GRID_WIDTH, width)
    if int(dpg.get_value(PARAM_GRID_HEIGHT)) <= 0:
        dpg.set_value(PARAM_GRID_HEIGHT, height)

    dpg.set_value(ROI_STATUS_TAG, "ROI mode: click two image corners.")
    dpg.set_value(ROI_OUTPUT_TAG, "")
    dpg.set_value(POLYGON_OUTPUT_TAG, "")
    dpg.set_value(ZOOM_TEXT_TAG, "Zoom: 100%")
    _update_polygon_points_text()
    _redraw_preview()
    _set_status(f"Loaded: {image_path}")


def _open_file_callback(_sender, app_data) -> None:
    selections = app_data.get("selections", {})
    if not selections:
        return

    selected_path = next(iter(selections.values()))
    _show_png(selected_path)


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

    with dpg.handler_registry():
        dpg.add_mouse_move_handler(callback=_mouse_move_callback)
        dpg.add_mouse_click_handler(
            button=dpg.mvMouseButton_Left,
            callback=_mouse_click_callback,
        )
        dpg.add_mouse_wheel_handler(callback=_mouse_wheel_callback)

    with dpg.window(label="Point Contour Preview Viewer", tag="main_window"):
        dpg.add_text("Preview viewer for density/mask PNG images.")
        dpg.add_button(
            label="Open PNG",
            callback=lambda: dpg.show_item("open_png_dialog"),
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
                dpg.add_button(label="Finish polygon", callback=_finish_polygon_callback)
                dpg.add_button(label="Clear polygon", callback=_clear_polygon_callback)
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

    dpg.create_viewport(title="Point Contour Preview Viewer", width=1200, height=850)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    run()
