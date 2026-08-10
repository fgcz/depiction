import numpy as np
import pytest

from depiction.tools.coordinate_summary import (
    describe_order,
    format_coordinate_summary,
    format_occupancy_map,
    summarize_coordinates,
)
from depiction_io.pixel_size import PixelSize


def grid_coordinates(n_x: int, n_y: int, x_min: int = 1, y_min: int = 1) -> np.ndarray:
    """Returns the coordinates of a dense grid, in row-major order."""
    return np.array([(x, y) for y in range(y_min, y_min + n_y) for x in range(x_min, x_min + n_x)])


def test_summarize_coordinates_when_dense_grid() -> None:
    summary = summarize_coordinates(grid_coordinates(3, 2))
    assert summary == {
        "n_spectra": 6,
        "n_dim": 2,
        "x": {"min": 1, "max": 3, "extent": 3, "n_distinct": 3},
        "y": {"min": 1, "max": 2, "extent": 2, "n_distinct": 2},
        "grid_size": 6,
        "n_filled": 6,
        "fill_fraction": 1.0,
        "n_missing": 0,
        "n_duplicates": 0,
        "order": "row-major (y ascending, x ascending)",
        "pixel_size": None,
    }


def test_summarize_coordinates_when_sparse() -> None:
    summary = summarize_coordinates(np.array([[1, 1], [3, 1], [2, 5]]))
    assert summary["grid_size"] == 3 * 5
    assert summary["n_filled"] == 3
    assert summary["n_missing"] == 12
    assert summary["fill_fraction"] == 0.2


def test_summarize_coordinates_when_duplicates() -> None:
    summary = summarize_coordinates(np.array([[1, 1], [2, 1], [2, 1], [1, 2], [2, 2]]))
    assert summary["n_spectra"] == 5
    assert summary["n_filled"] == 4
    assert summary["n_duplicates"] == 1


def test_summarize_coordinates_when_3d() -> None:
    coordinates = np.array([[1, 1, 1], [2, 1, 1], [1, 1, 2], [2, 1, 2]])
    summary = summarize_coordinates(coordinates)
    assert summary["n_dim"] == 3
    assert summary["z"] == {"min": 1, "max": 2, "extent": 2, "n_distinct": 2}
    assert summary["grid_size"] == 2 * 1 * 2
    assert summary["n_duplicates"] == 0


def test_summarize_coordinates_when_pixel_size() -> None:
    summary = summarize_coordinates(grid_coordinates(2, 2), pixel_size=PixelSize(size_x=50.0, size_y=25.0, unit="um"))
    assert summary["pixel_size"] == {"x": 50.0, "y": 25.0, "unit": "um"}


@pytest.mark.parametrize("coordinates", [np.zeros((0, 2), dtype=int), np.array([1, 2])])
def test_summarize_coordinates_when_invalid_shape(coordinates: np.ndarray) -> None:
    with pytest.raises(ValueError, match="coordinate array"):
        summarize_coordinates(coordinates)


def test_summarize_coordinates_when_too_many_dimensions() -> None:
    with pytest.raises(ValueError, match="at most 3 coordinate dimensions"):
        summarize_coordinates(np.ones((2, 4), dtype=int))


@pytest.mark.parametrize(
    "coordinates,expected",
    [
        ([(1, 1)], "single spectrum"),
        ([(1, 1), (2, 1), (1, 2), (2, 2)], "row-major (y ascending, x ascending)"),
        ([(2, 1), (1, 1), (2, 2), (1, 2)], "row-major (y ascending, x descending)"),
        ([(1, 1), (2, 1), (2, 2), (1, 2)], "row-major serpentine (y ascending, x alternating)"),
        ([(1, 2), (2, 2), (1, 1), (2, 1)], "row-major (y descending, x ascending)"),
        ([(1, 1), (1, 2), (2, 1), (2, 2)], "column-major (x ascending, y ascending)"),
        ([(1, 1), (1, 2), (2, 2), (2, 1)], "column-major serpentine (x ascending, y alternating)"),
        ([(1, 1), (1, 2), (1, 3)], "row-major (y ascending, x constant)"),
        ([(1, 1), (5, 3), (2, 2)], "irregular (no row/column scan order recognized)"),
    ],
)
def test_describe_order(coordinates: list[tuple[int, int]], expected: str) -> None:
    assert describe_order(np.array(coordinates)) == expected


def test_format_coordinate_summary() -> None:
    summary = summarize_coordinates(
        np.array([[1, 1], [2, 1], [1, 2]]), pixel_size=PixelSize(size_x=50.0, size_y=50.0, unit="um")
    )
    assert format_coordinate_summary(summary) == (
        "n_spectra:   3 (2D coordinates)\n"
        "x:           1 - 2 (extent 2, 2 distinct)\n"
        "y:           1 - 2 (extent 2, 2 distinct)\n"
        "grid:        2 x 2 = 4 px, 3 filled (75.0%), 1 missing\n"
        "duplicates:  0\n"
        "pixel size:  50.0 x 50.0 um\n"
        "order:       row-major (y ascending, x ascending)"
    )


def test_format_coordinate_summary_when_no_pixel_size() -> None:
    assert "pixel size:  not declared" in format_coordinate_summary(summarize_coordinates(grid_coordinates(2, 2)))


LEGEND = "1 char = {} px;  ░ <25%  ▒ <50%  ▓ <75%  █ >=75%"


def test_format_occupancy_map() -> None:
    # A 3 x 4 bounding box, so the smallest possible block (1 x 2 px) leaves 3 columns and 2 rows.
    coordinates = np.array([(x, y) for y in (1, 2) for x in (1, 2, 3)] + [(1, 3), (1, 4), (2, 4)])
    assert format_occupancy_map(coordinates) == "███\n█▓\n" + LEGEND.format("1 x 2")


def test_format_occupancy_map_when_partially_filled_blocks() -> None:
    # A 4 x 8 bounding box in 2 x 4 px blocks, filled with 1, 3, 5 and 8 of the 8 pixels each.
    coordinates = np.array(
        [(1, 1)]
        + [(3, 1), (4, 1), (3, 2)]
        + [(1, 5), (2, 5), (1, 6), (2, 6), (1, 7)]
        + [(x, y) for y in (5, 6, 7, 8) for x in (3, 4)]
    )
    assert format_occupancy_map(coordinates, max_width=2) == "░▒\n▓█\n" + LEGEND.format("2 x 4")


def test_format_occupancy_map_when_downsampled() -> None:
    lines = format_occupancy_map(grid_coordinates(20, 40), max_width=10, max_height=10).splitlines()
    assert lines[-1] == LEGEND.format("2 x 4")
    assert lines[:-1] == ["█" * 10] * 10


if __name__ == "__main__":
    pytest.main()
