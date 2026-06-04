from __future__ import annotations

import json
from pathlib import Path

import ezdxf
import numpy as np


def load_holes_json_circles(path: str | Path) -> list[dict]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        holes = data.get("holes", [])
        groups = data.get("groups", [])
    elif isinstance(data, list):
        holes = data
        groups = []
    else:
        raise ValueError("holes JSON must contain an object or a list")

    if not isinstance(holes, list) or not isinstance(groups, list):
        raise ValueError("holes and groups must be lists")

    groups_by_id = {
        str(group.get("id")): group
        for group in groups
        if isinstance(group, dict) and group.get("id") is not None
    }

    circles = []
    for hole in holes:
        if not isinstance(hole, dict):
            continue
        if hole.get("accepted", True) is False:
            continue
        if hole.get("enabled", True) is False:
            continue

        group = None
        group_id = hole.get("group_id")
        if group_id is not None:
            group = groups_by_id.get(str(group_id))
            if group is not None and group.get("enabled", True) is False:
                continue

        if group is not None and group.get("radius") is not None:
            radius = float(group["radius"])
        elif hole.get("radius") is not None:
            radius = float(hole["radius"])
        elif hole.get("diameter") is not None:
            radius = float(hole["diameter"]) / 2.0
        else:
            continue

        if radius <= 0 or hole.get("center_x") is None or hole.get("center_y") is None:
            continue

        circles.append(
            {
                "center_x": float(hole["center_x"]),
                "center_y": float(hole["center_y"]),
                "radius": radius,
            }
        )

    return circles


def save_contour_dxf(
    contour_world: np.ndarray,
    output_path: str | Path,
    close: bool = True,
    holes: list[dict] | None = None,
) -> None:
    """
    Сохраняет внешний контур в DXF как LWPOLYLINE.

    contour_world:
      numpy array shape = (N, 2)
      columns = X, Y

    Пока это простой экспорт полилинии.
    Дуги, окружности отверстий и сплайны добавим позже.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if contour_world.ndim != 2 or contour_world.shape[1] != 2:
        raise ValueError("contour_world должен иметь форму (N, 2)")

    if len(contour_world) < 3:
        raise ValueError("Для DXF-контура нужно минимум 3 точки")

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM

    msp = doc.modelspace()

    points = [(float(x), float(y)) for x, y in contour_world]

    doc.layers.add(
        name="OUTER_CONTOUR",
        color=1,
    )
    doc.layers.add(
        name="HOLES",
        color=3,
    )

    msp.add_lwpolyline(
        points,
        close=close,
        dxfattribs={
            "layer": "OUTER_CONTOUR",
        },
    )

    for hole in holes or []:
        msp.add_circle(
            center=(float(hole["center_x"]), float(hole["center_y"])),
            radius=float(hole["radius"]),
            dxfattribs={"layer": "HOLES"},
        )

    doc.saveas(output_path)
