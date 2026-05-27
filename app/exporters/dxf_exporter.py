from __future__ import annotations

from pathlib import Path

import ezdxf
import numpy as np


def save_contour_dxf(
    contour_world: np.ndarray,
    output_path: str | Path,
    close: bool = True,
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

    polyline = msp.add_lwpolyline(
        points,
        close=close,
        dxfattribs={
            "layer": "OUTER_CONTOUR",
        },
    )

    doc.layers.add(
        name="OUTER_CONTOUR",
        color=1,
    )

    doc.saveas(output_path)