from __future__ import annotations

import math


CELL_SIZE_DECIMAL_PLACES = 6


def normalize_cell_size(cell_size: object) -> float:
    """Return the canonical density cell size in millimetres."""
    try:
        numeric_value = float(cell_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("cell_size must be numeric") from exc

    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError("cell_size must be finite and greater than zero")

    canonical_value = round(numeric_value, CELL_SIZE_DECIMAL_PLACES)
    if canonical_value <= 0:
        raise ValueError(
            f"cell_size must be at least {10 ** -CELL_SIZE_DECIMAL_PLACES:g}"
        )

    return canonical_value
