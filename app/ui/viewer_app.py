from __future__ import annotations

from pathlib import Path

import dearpygui.dearpygui as dpg


TEXTURE_TAG = "preview_texture"
IMAGE_TAG = "preview_image"
STATUS_TAG = "status_text"
COORDS_TAG = "coords_text"
ROI_STATUS_TAG = "roi_status_text"
ROI_OUTPUT_TAG = "roi_output_text"

PARAM_GRID_MIN_X = "param_grid_min_x"
PARAM_GRID_MIN_Y = "param_grid_min_y"
PARAM_CELL_SIZE = "param_cell_size"
PARAM_GRID_WIDTH = "param_original_grid_width"
PARAM_GRID_HEIGHT = "param_original_grid_height"

state = {
    "image_width": 0,
    "image_height": 0,
    "roi_first_world": None,
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
    image_left, image_top = dpg.get_item_rect_min(IMAGE_TAG)

    pixel_x = mouse_x - image_left
    pixel_y = mouse_y - image_top

    if pixel_x < 0 or pixel_y < 0 or pixel_x >= image_width or pixel_y >= image_height:
        return None

    displayed_grid_x = pixel_x * grid_width / image_width
    displayed_grid_y = pixel_y * grid_height / image_height
    original_grid_y = (grid_height - 1) - displayed_grid_y

    world_x = grid_min_x + (displayed_grid_x + 0.5) * cell_size
    world_y = grid_min_y + (original_grid_y + 0.5) * cell_size

    return pixel_x, pixel_y, displayed_grid_x, original_grid_y, world_x, world_y


def _mouse_move_callback(_sender=None, _app_data=None, _user_data=None) -> None:
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


def _mouse_click_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    coords = _mouse_to_world()
    if coords is None:
        return

    *_unused, world_x, world_y = coords
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


def _reset_roi_callback(_sender=None, _app_data=None, _user_data=None) -> None:
    state["roi_first_world"] = None
    dpg.set_value(ROI_STATUS_TAG, "ROI mode: click two image corners.")
    dpg.set_value(ROI_OUTPUT_TAG, "")


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

    dpg.add_image(TEXTURE_TAG, parent="image_area", tag=IMAGE_TAG)

    state["image_width"] = width
    state["image_height"] = height
    state["roi_first_world"] = None

    if int(dpg.get_value(PARAM_GRID_WIDTH)) <= 0:
        dpg.set_value(PARAM_GRID_WIDTH, width)
    if int(dpg.get_value(PARAM_GRID_HEIGHT)) <= 0:
        dpg.set_value(PARAM_GRID_HEIGHT, height)

    dpg.set_value(ROI_STATUS_TAG, "ROI mode: click two image corners.")
    dpg.set_value(ROI_OUTPUT_TAG, "")
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
        dpg.add_text("ROI mode: click two image corners.", tag=ROI_STATUS_TAG)
        dpg.add_button(label="Reset ROI", callback=_reset_roi_callback)
        dpg.add_text("", tag=ROI_OUTPUT_TAG)
        dpg.add_separator()

        with dpg.child_window(tag="image_area", border=False, horizontal_scrollbar=True):
            dpg.add_text(
                "Open a density or mask preview PNG to view it here.",
                tag="image_hint",
            )

    dpg.create_viewport(title="Point Contour Preview Viewer", width=1200, height=850)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    run()
