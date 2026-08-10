"""Summarizes the spatial coordinates of an acquisition, for `depiction-tools imzml coords`."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from depiction_io.pixel_size import PixelSize

AXIS_NAMES = ("x", "y", "z")

# From empty to full; the first one is never used for a block that holds at least one spectrum.
DENSITY_CHARS = " ░▒▓█"


def summarize_coordinates(coordinates: NDArray[np.int64], pixel_size: PixelSize | None = None) -> dict[str, Any]:
    """Returns structured information about a coordinate list, shape (n_spectra, n_dim).

    `n_duplicates` is the interesting one: it counts spectra that share a pixel with an earlier
    spectrum, which is what happens when acquisitions are merged without shifting their
    coordinates apart.
    """
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[0] == 0:
        raise ValueError(f"Expected a non-empty (n_spectra, n_dim) coordinate array, got {coordinates.shape}.")
    n_spectra, n_dim = coordinates.shape
    if n_dim > len(AXIS_NAMES):
        raise ValueError(f"Expected at most {len(AXIS_NAMES)} coordinate dimensions, got {n_dim}.")

    axes = {}
    for i_axis, name in enumerate(AXIS_NAMES[:n_dim]):
        values = coordinates[:, i_axis]
        axes[name] = {
            "min": int(values.min()),
            "max": int(values.max()),
            "extent": int(values.max() - values.min() + 1),
            "n_distinct": int(np.unique(values).size),
        }

    grid_size = math.prod(axis["extent"] for axis in axes.values())
    n_filled = int(np.unique(coordinates, axis=0).shape[0])
    return {
        "n_spectra": n_spectra,
        "n_dim": n_dim,
        **axes,
        "grid_size": grid_size,
        "n_filled": n_filled,
        "fill_fraction": round(n_filled / grid_size, 4),
        "n_missing": grid_size - n_filled,
        "n_duplicates": n_spectra - n_filled,
        "order": describe_order(coordinates[:, :2]),
        "pixel_size": (
            None if pixel_size is None else {"x": pixel_size.size_x, "y": pixel_size.size_y, "unit": pixel_size.unit}
        ),
    }


def format_coordinate_summary(summary: dict[str, Any]) -> str:
    """Formats the result of `summarize_coordinates` as a text block."""
    lines = [_line("n_spectra", f"{summary['n_spectra']} ({summary['n_dim']}D coordinates)")]
    for name in AXIS_NAMES[: summary["n_dim"]]:
        axis = summary[name]
        lines.append(
            _line(name, f"{axis['min']} - {axis['max']} (extent {axis['extent']}, {axis['n_distinct']} distinct)")
        )
    extents = " x ".join(str(summary[name]["extent"]) for name in AXIS_NAMES[: summary["n_dim"]])
    lines.append(
        _line(
            "grid",
            f"{extents} = {summary['grid_size']} px, {summary['n_filled']} filled "
            f"({summary['fill_fraction']:.1%}), {summary['n_missing']} missing",
        )
    )
    lines.append(_line("duplicates", str(summary["n_duplicates"])))
    pixel_size = summary["pixel_size"]
    lines.append(
        _line(
            "pixel size",
            "not declared" if pixel_size is None else f"{pixel_size['x']} x {pixel_size['y']} {pixel_size['unit']}",
        )
    )
    lines.append(_line("order", summary["order"]))
    return "\n".join(lines)


def format_occupancy_map(coordinates: NDArray[np.int64], *, max_width: int = 78, max_height: int = 36) -> str:
    """Returns a text map of which pixels of the bounding box are occupied.

    Each character covers a block of pixels and is shaded by how much of that block is filled, so
    holes, off-grid acquisitions and transposed axes are visible at a glance. Blocks are twice as
    tall as they are wide, because terminal characters are.
    """
    unique = np.unique(np.asarray(coordinates)[:, :2], axis=0)
    if unique.shape[0] == 0:
        raise ValueError("Expected a non-empty coordinate array.")
    origin = unique.min(axis=0)
    extent = unique.max(axis=0) - origin + 1

    block_x = math.ceil(extent[0] / max_width)
    block_y = max(2 * block_x, math.ceil(extent[1] / max_height))
    n_cols = math.ceil(extent[0] / block_x)
    n_rows = math.ceil(extent[1] / block_y)

    blocks = (unique - origin) // (block_x, block_y)
    counts = np.zeros((n_rows, n_cols), dtype=int)
    np.add.at(counts, (blocks[:, 1], blocks[:, 0]), 1)
    # A block on the right or bottom edge can be partial, but shading it against the full block
    # size is the lesser evil: the alternative makes the edge look denser than the interior.
    shades = np.minimum((counts / (block_x * block_y) * 4).astype(int), len(DENSITY_CHARS) - 2)
    chars = np.where(counts == 0, DENSITY_CHARS[0], np.array(list(DENSITY_CHARS))[shades + 1])

    lines = ["".join(row).rstrip() for row in chars]
    shading = "  ".join(f"{char} {label}" for char, label in zip(DENSITY_CHARS[1:], ["<25%", "<50%", "<75%", ">=75%"]))
    lines.append(f"1 char = {block_x} x {block_y} px;  {shading}")
    return "\n".join(lines)


def describe_order(coordinates_2d: NDArray[np.int64]) -> str:
    """Describes the order in which the spectra were acquired, as far as it can be recognized."""
    if coordinates_2d.shape[0] < 2:
        return "single spectrum"
    for i_slow, i_fast, name in ((1, 0, "row-major"), (0, 1, "column-major")):
        pattern = _scan_pattern(coordinates_2d, i_slow=i_slow, i_fast=i_fast)
        if pattern is not None:
            slow_direction, fast_direction = pattern
            serpentine = " serpentine" if fast_direction == "alternating" else ""
            slow_name, fast_name = AXIS_NAMES[i_slow], AXIS_NAMES[i_fast]
            return f"{name}{serpentine} ({slow_name} {slow_direction}, {fast_name} {fast_direction})"
    return "irregular (no row/column scan order recognized)"


def _scan_pattern(coordinates_2d: NDArray[np.int64], *, i_slow: int, i_fast: int) -> tuple[str, str] | None:
    """Classifies the acquisition order as (slow axis direction, fast axis direction).

    Returns None unless the spectra come in one uninterrupted line per slow-axis value, with the
    lines in monotone order and the fast axis monotone within every line.
    """
    slow = coordinates_2d[:, i_slow]
    fast = coordinates_2d[:, i_fast]

    lines = np.split(np.arange(len(slow)), np.flatnonzero(np.diff(slow)) + 1)
    line_values = [int(slow[indices[0]]) for indices in lines]
    slow_direction = _monotone_direction(np.diff(line_values))
    if slow_direction is None:
        return None

    directions = []
    for indices in lines:
        direction = _monotone_direction(np.diff(fast[indices]))
        if direction is None:
            return None
        directions.append(direction)
    directions = [direction for direction in directions if direction != "constant"]

    if not directions:
        # Every line holds a single spectrum, e.g. an acquisition that is one pixel wide.
        return slow_direction, "constant"
    if all(direction == "ascending" for direction in directions):
        return slow_direction, "ascending"
    if all(direction == "descending" for direction in directions):
        return slow_direction, "descending"
    if all(a != b for a, b in zip(directions, directions[1:])):
        return slow_direction, "alternating"
    return None


def _monotone_direction(diffs: NDArray[np.int64]) -> str | None:
    """Returns "ascending", "descending", "constant" for a strictly monotone step sequence, else None."""
    if len(diffs) == 0:
        return "constant"
    if np.all(diffs > 0):
        return "ascending"
    if np.all(diffs < 0):
        return "descending"
    return None


def _line(label: str, value: str) -> str:
    return f"{label + ':':<13}{value}"
