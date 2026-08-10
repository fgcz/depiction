import numpy as np
import pytest
from logot import Logot, logged
from logot.loguru import LoguruCapturer
from pytest_mock import MockerFixture

from depiction.tools.concat_spatial_imzml import (
    SpatialPlacement,
    build_concat_info,
    compute_placements,
    concat_spatial,
    resolve_output_mode,
)
from depiction_io import ImzmlModeEnum, RamReadFile

# 2 x 2 bounding box at the origin, and a 3 x 2 one somewhere else entirely.
COORDINATES_A = np.array([[1, 1], [2, 1], [1, 2]])
COORDINATES_B = np.array([[5, 10], [6, 10], [7, 11]])

PLACEMENT_A = SpatialPlacement(source_min=(1, 1), extent=(2, 2), target_min=(1, 1), n_spectra=3)


def ram_read_file(mz_arr_list: list[list[float]], coordinates: np.ndarray) -> RamReadFile:
    return RamReadFile(
        mz_arr_list=[np.array(mz_arr) for mz_arr in mz_arr_list],
        int_arr_list=[np.arange(len(mz_arr), dtype=float) + 1 for mz_arr in mz_arr_list],
        coordinates=coordinates,
    )


def test_compute_placements_when_axis_x() -> None:
    assert compute_placements([COORDINATES_A, COORDINATES_B]) == [
        PLACEMENT_A,
        SpatialPlacement(source_min=(5, 10), extent=(3, 2), target_min=(3, 1), n_spectra=3),
    ]


def test_compute_placements_when_axis_y() -> None:
    assert compute_placements([COORDINATES_A, COORDINATES_B], axis="y") == [
        PLACEMENT_A,
        SpatialPlacement(source_min=(5, 10), extent=(3, 2), target_min=(1, 3), n_spectra=3),
    ]


def test_compute_placements_when_spacing() -> None:
    placements = compute_placements([COORDINATES_A, COORDINATES_B, COORDINATES_A], spacing=2)
    assert [placement.target_min for placement in placements] == [(1, 1), (5, 1), (10, 1)]


def test_compute_placements_when_3d() -> None:
    coordinates = np.array([[1, 1, 1], [2, 1, 5]])
    assert compute_placements([coordinates]) == [
        SpatialPlacement(source_min=(1, 1), extent=(2, 1), target_min=(1, 1), n_spectra=2)
    ]


def test_compute_placements_when_invalid_axis() -> None:
    with pytest.raises(ValueError, match="Expected axis to be one of"):
        compute_placements([COORDINATES_A], axis="z")


def test_compute_placements_when_negative_spacing() -> None:
    with pytest.raises(ValueError, match="non-negative spacing"):
        compute_placements([COORDINATES_A], spacing=-1)


def test_spatial_placement_shift() -> None:
    placement = SpatialPlacement(source_min=(5, 10), extent=(3, 2), target_min=(3, 1), n_spectra=3)
    np.testing.assert_array_equal(placement.shift(COORDINATES_B), [[3, 1], [4, 1], [5, 2]])
    # A z coordinate is kept as it is.
    np.testing.assert_array_equal(placement.shift(np.array([5, 10, 7])), [3, 1, 7])
    np.testing.assert_array_equal(COORDINATES_B, [[5, 10], [6, 10], [7, 11]])


def test_resolve_output_mode_when_explicit() -> None:
    assert resolve_output_mode([], mode="processed") == ImzmlModeEnum.PROCESSED
    assert resolve_output_mode([], mode="continuous") == ImzmlModeEnum.CONTINUOUS


def test_resolve_output_mode_when_continuous_on_one_axis() -> None:
    input_files = [
        ram_read_file([[100.0, 200.0]] * 3, COORDINATES_A),
        ram_read_file([[100.0, 200.0]] * 3, COORDINATES_B),
    ]
    assert resolve_output_mode(input_files) == ImzmlModeEnum.CONTINUOUS


def test_resolve_output_mode_when_continuous_on_different_axes() -> None:
    input_files = [
        ram_read_file([[100.0, 200.0]] * 3, COORDINATES_A),
        ram_read_file([[100.0, 300.0]] * 3, COORDINATES_B),
    ]
    assert resolve_output_mode(input_files) == ImzmlModeEnum.PROCESSED


def test_resolve_output_mode_when_one_input_is_processed() -> None:
    input_files = [
        ram_read_file([[100.0, 200.0]] * 3, COORDINATES_A),
        ram_read_file([[100.0, 200.0], [100.0], [200.0]], COORDINATES_B),
    ]
    assert resolve_output_mode(input_files) == ImzmlModeEnum.PROCESSED


def test_concat_spatial_warns_when_input_has_duplicate_coordinates(mocker: MockerFixture) -> None:
    input_file = ram_read_file([[100.0, 200.0]] * 3, np.array([[1, 1], [1, 1], [2, 1]]))
    mock_write_file = mocker.MagicMock(name="mock_write_file")
    with Logot().capturing(capturer=LoguruCapturer) as logot:
        concat_spatial([input_file], mock_write_file)
    logot.assert_logged(logged.warning("1 of the 3 spectra of %s share a pixel with another%s"))


def test_build_concat_info() -> None:
    info = build_concat_info(
        ["a.imzML", "b.imzML"],
        compute_placements([COORDINATES_A, COORDINATES_B]),
        axis="x",
        spacing=0,
        imzml_mode=ImzmlModeEnum.CONTINUOUS,
    )
    assert info == {
        "axis": "x",
        "spacing": 0,
        "imzml_mode": "CONTINUOUS",
        "inputs": [
            {
                "path": "a.imzML",
                "n_spectra": 3,
                "source_min": [1, 1],
                "extent": [2, 2],
                "target_min": [1, 1],
            },
            {
                "path": "b.imzML",
                "n_spectra": 3,
                "source_min": [5, 10],
                "extent": [3, 2],
                "target_min": [3, 1],
            },
        ],
    }


if __name__ == "__main__":
    pytest.main()
