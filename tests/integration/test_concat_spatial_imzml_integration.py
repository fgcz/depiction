import json
from pathlib import Path

import numpy as np
import pytest
import xarray

from depiction.misc.integration_test_utils import IntegrationTestUtils
from depiction.tools.cli.cli_imzml import cmd_imzml_concat, cmd_imzml_coords
from depiction_io import ImzmlModeEnum, get_read_file

MZ_ARR = [100.0, 200.0, 300.0]


def write_input(path: Path, coordinates: list[tuple[int, int]], mz_arr: list[float] = MZ_ARR) -> Path:
    IntegrationTestUtils.populate_test_file(
        path=str(path),
        mz_arr_list=[mz_arr] * len(coordinates),
        int_arr_list=[[i_spectrum + 1] * len(mz_arr) for i_spectrum in range(len(coordinates))],
        imzml_mode=ImzmlModeEnum.CONTINUOUS,
        coordinates_list=coordinates,
    )
    return path


@pytest.fixture
def input_a(tmp_path: Path) -> Path:
    return write_input(tmp_path / "part_a.imzML", [(1, 1), (2, 1), (1, 2)])


@pytest.fixture
def input_b(tmp_path: Path) -> Path:
    # Deliberately overlapping input_a's coordinates: a plain merge would stack the two.
    return write_input(tmp_path / "part_b.imzML", [(1, 1), (2, 1)])


def expected_coordinates(coordinates: list[tuple[int, int]]) -> xarray.DataArray:
    return xarray.DataArray(coordinates, dims=("i", "d"), coords={"d": ["x", "y"]})


def test_concat_along_x(tmp_path: Path, input_a: Path, input_b: Path) -> None:
    output = tmp_path / "combined.imzML"

    cmd_imzml_concat([input_a, input_b], output_imzml=output)

    read_file = get_read_file(output)
    assert read_file.n_spectra == 5
    assert read_file.imzml_mode == ImzmlModeEnum.CONTINUOUS
    xarray.testing.assert_equal(
        expected_coordinates([[1, 1], [2, 1], [1, 2], [3, 1], [4, 1]]), read_file.coordinates_array_2d
    )
    with read_file.reader() as reader:
        mz_arr_list, int_arr_list = reader.get_spectra(list(range(5)))
    np.testing.assert_array_equal(np.array([MZ_ARR] * 5), mz_arr_list)
    np.testing.assert_array_equal(np.array([[1] * 3, [2] * 3, [3] * 3, [1] * 3, [2] * 3]), int_arr_list)


def test_concat_along_y_with_spacing(tmp_path: Path, input_a: Path, input_b: Path) -> None:
    output = tmp_path / "combined.imzML"

    cmd_imzml_concat([input_a, input_b], output_imzml=output, axis="y", spacing=1)

    xarray.testing.assert_equal(
        expected_coordinates([[1, 1], [2, 1], [1, 2], [1, 4], [2, 4]]),
        get_read_file(output).coordinates_array_2d,
    )


def test_concat_writes_placement_info(tmp_path: Path, input_a: Path, input_b: Path) -> None:
    output = tmp_path / "combined.imzML"

    cmd_imzml_concat([input_a, input_b], output_imzml=output)

    assert json.loads((tmp_path / "combined.concat.json").read_text()) == {
        "axis": "x",
        "spacing": 0,
        "imzml_mode": "CONTINUOUS",
        "inputs": [
            {
                "path": str(input_a),
                "n_spectra": 3,
                "source_min": [1, 1],
                "extent": [2, 2],
                "target_min": [1, 1],
            },
            {
                "path": str(input_b),
                "n_spectra": 2,
                "source_min": [1, 1],
                "extent": [2, 1],
                "target_min": [3, 1],
            },
        ],
    }


def test_concat_when_inputs_do_not_share_an_mz_axis(tmp_path: Path, input_a: Path) -> None:
    input_c = write_input(tmp_path / "part_c.imzML", [(1, 1), (2, 1)], mz_arr=[150.0, 250.0])
    output = tmp_path / "combined.imzML"

    cmd_imzml_concat([input_a, input_c], output_imzml=output)

    read_file = get_read_file(output)
    assert read_file.imzml_mode == ImzmlModeEnum.PROCESSED
    with read_file.reader() as reader:
        np.testing.assert_array_equal(MZ_ARR, reader.get_spectrum_mz(0))
        np.testing.assert_array_equal([150.0, 250.0], reader.get_spectrum_mz(3))


def test_concat_when_output_exists(tmp_path: Path, input_a: Path, input_b: Path) -> None:
    output = tmp_path / "combined.imzML"
    cmd_imzml_concat([input_a, input_b], output_imzml=output)

    with pytest.raises(ValueError, match="already exists"):
        cmd_imzml_concat([input_a, input_b], output_imzml=output)

    cmd_imzml_concat([input_a], output_imzml=output, overwrite=True)
    assert get_read_file(output).n_spectra == 3


def test_coords_reports_no_duplicates_after_concat(
    tmp_path: Path, input_a: Path, input_b: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "combined.imzML"
    cmd_imzml_concat([input_a, input_b], output_imzml=output)
    capsys.readouterr()

    cmd_imzml_coords(output, as_json=True)
    summary = json.loads(capsys.readouterr().out)

    assert summary["file"] == str(output)
    assert summary["n_spectra"] == 5
    assert summary["n_duplicates"] == 0
    assert summary["x"] == {"min": 1, "max": 4, "extent": 4, "n_distinct": 4}
    assert summary["y"] == {"min": 1, "max": 2, "extent": 2, "n_distinct": 2}
    assert summary["n_missing"] == 3


def test_coords_prints_text_and_map(tmp_path: Path, input_a: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cmd_imzml_coords(input_a, show_map=True)

    out = capsys.readouterr().out
    assert f"file: {input_a}" in out
    assert "n_spectra:   3 (2D coordinates)" in out
    assert "duplicates:  0" in out
    # Two 1 x 2 px blocks: the left one holds both of its pixels, the right one only (2, 1).
    assert "█▓\n1 char = 1 x 2 px" in out


if __name__ == "__main__":
    pytest.main()
