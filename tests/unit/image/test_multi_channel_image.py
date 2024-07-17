from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray
from pytest_mock import MockerFixture
from xarray import DataArray

from depiction.image.multi_channel_image import MultiChannelImage


@pytest.fixture
def mock_coords() -> dict[str, list[str]]:
    return {"c": ["Channel A"]}


@pytest.fixture
def mock_data(mock_coords) -> DataArray:
    """Dense mock data without any missing values."""
    return DataArray(
        [[[2.0], [4]], [[6], [8]], [[10], [12]]],
        dims=("y", "x", "c"),
        coords=mock_coords,
        attrs={"bg_value": 0},
    )


@pytest.fixture
def mock_image(mock_data) -> MultiChannelImage:
    return MultiChannelImage(data=mock_data)


def test_from_numpy_sparse() -> None:
    values = np.array([[1, 2, 3], [4, 5, 6]])
    coordinates = np.array([[0, 0], [1, 1]])
    image = MultiChannelImage.from_numpy_sparse(values=values, coordinates=coordinates, channel_names=["A", "B", "C"])
    assert image.channel_names == ["A", "B", "C"]
    values = image.data_spatial.sel(c="B")
    xarray.testing.assert_equal(DataArray([[2, 0], [0, 5]], dims=("y", "x"), coords={"c": "B"}, name="values"), values)


def test_n_channels(mock_image: MultiChannelImage) -> None:
    assert mock_image.n_channels == 1


def test_n_nonzero(mock_image: MultiChannelImage) -> None:
    assert mock_image.n_nonzero == 6


def test_n_nonzero_when_sparse(mock_image: MultiChannelImage) -> None:
    mock_image.data_spatial[1, 0, 0] = 0
    assert mock_image.n_nonzero == 5


def test_dtype(mock_image: MultiChannelImage) -> None:
    assert mock_image.dtype == float


def test_bg_value(mock_image: MultiChannelImage) -> None:
    assert mock_image.bg_value == 0.0


def test_bg_mask_when_0(mock_data: DataArray, mock_image: MultiChannelImage) -> None:
    mock_data[1, :, :] = 0
    bg_mask = mock_image.bg_mask
    expected_bg_mask = DataArray([[False, False], [True, True], [False, False]], dims=("y", "x"))
    xarray.testing.assert_equal(expected_bg_mask, bg_mask)


def test_bg_mask_when_nan(mock_image: MultiChannelImage) -> None:
    mock_image.data_spatial[1, :, :] = np.nan
    mock_image.data_spatial.attrs["bg_value"] = np.nan
    bg_mask = mock_image.bg_mask
    expected_bg_mask = DataArray([[False, False], [True, True], [False, False]], dims=("y", "x"))
    xarray.testing.assert_equal(expected_bg_mask, bg_mask)


def test_dimensions(mock_image: MultiChannelImage) -> None:
    assert mock_image.dimensions == (2, 3)


def test_channel_names_when_set(mock_image: MultiChannelImage) -> None:
    assert mock_image.channel_names == ["Channel A"]


@pytest.mark.parametrize("mock_coords", [{}])
def test_channel_names_when_not_set(mock_image: MultiChannelImage) -> None:
    assert mock_image.channel_names == ["0"]


def test_data_spatial(mock_data: DataArray, mock_image: MultiChannelImage) -> None:
    xarray.testing.assert_identical(mock_data, mock_image.data_spatial)


def test_data_flat(mock_data: DataArray, mock_image: MultiChannelImage) -> None:
    mock_data[0, 0, 0] = 0
    mock_data[1, 0, 0] = np.nan
    expected = DataArray(
        [[4.0, 8, 10, 12]],
        dims=("c", "i"),
        coords={
            "c": ["Channel A"],
            "i": pd.MultiIndex.from_tuples([(0, 1), (1, 1), (2, 0), (2, 1)], names=("y", "x")),
        },
        attrs={"bg_value": 0},
    )
    xarray.testing.assert_identical(expected, mock_image.data_flat)


def test_write_hdf5(mocker: MockerFixture, mock_image: MultiChannelImage) -> None:
    mocker.patch("xarray.DataArray.to_netcdf")
    mock_image.write_hdf5(Path("test.h5"))
    mock_image.data_spatial.to_netcdf.assert_called_once_with(Path("test.h5"), format="NETCDF4")


def test_read_hdf5(mocker: MockerFixture, mock_data: DataArray) -> None:
    mocker.patch("xarray.open_dataarray").return_value = mock_data
    image = MultiChannelImage.read_hdf5(Path("test.h5"))
    xarray.open_dataarray.assert_called_once_with(Path("test.h5"))
    xarray.testing.assert_equal(image.data_spatial, mock_data)


def test_with_channel_names(mock_image: MultiChannelImage) -> None:
    image = mock_image.with_channel_names(channel_names=["New Channel Name"])
    assert image.channel_names == ["New Channel Name"]
    assert image.dimensions == mock_image.dimensions
    assert image.n_channels == mock_image.n_channels == 1
    # TODO check values


def test_str(mock_image: MultiChannelImage) -> None:
    assert str(mock_image) == "MultiChannelImage(size_y=3, size_x=2, n_channels=1)"


def test_repr(mocker: MockerFixture, mock_image: MultiChannelImage) -> None:
    mocker.patch("xarray.DataArray.__repr__", return_value="DataArray")
    assert repr(mock_image) == "MultiChannelImage(data=DataArray)"


if __name__ == "__main__":
    pytest.main()
