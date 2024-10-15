from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray
from pytest_mock import MockerFixture
from xarray import DataArray

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.persistence.format_ome_tiff import OmeTiff


@pytest.fixture
def mock_coords() -> dict[str, list[str]]:
    return {"c": ["Channel A", "Channel B"]}


@pytest.fixture
def mock_data(mock_coords) -> DataArray:
    """Dense mock data without any missing values."""
    return DataArray(
        [[[2.0, 5], [4, 5]], [[6, 5], [8, 5]], [[10, 5], [12, 5]]],
        dims=("y", "x", "c"),
        coords=mock_coords,
    )


@pytest.fixture
def mock_data_sparse(mock_coords) -> DataArray:
    return DataArray(
        [[[0, 0], [0, 0]], [[6, 5], [8, 5]], [[0, 0], [0, 0]]],
        dims=("y", "x", "c"),
        coords=mock_coords,
    )


@pytest.fixture
def mock_image(mock_data) -> MultiChannelImage:
    """Dense mock image without any missing values."""
    return MultiChannelImage(
        data=mock_data, is_foreground=DataArray([[True, True], [True, True], [True, True]], dims=("y", "x"))
    )


@pytest.fixture
def mock_image_sparse(mock_data_sparse) -> MultiChannelImage:
    """Sparse mock image."""
    return MultiChannelImage(
        data=mock_data_sparse, is_foreground=DataArray([[False, False], [True, True], [False, False]], dims=("y", "x"))
    )


@pytest.mark.parametrize(
    ["values", "channel_names", "expected_channel_names"],
    [
        (DataArray([[1, 2, 3], [4, 5, 6]], dims=("i", "c"), coords={"c": ["A", "B", "C"]}), False, ["A", "B", "C"]),
        (DataArray([[1, 2, 3], [4, 5, 6]], dims=("i", "c")), ["A", "B", "C"], ["A", "B", "C"]),
        (DataArray([[1, 2, 3], [4, 5, 6]], dims=("i", "c")), True, ["0", "1", "2"]),
    ],
)
def test_from_flat(values, channel_names, expected_channel_names) -> None:
    coords = DataArray([[0, 0], [1, 2]], dims=("i", "d"), coords={"d": ["x", "y"]})
    image = MultiChannelImage.from_flat(values, coords, channel_names)
    assert image.channel_names == expected_channel_names
    np.testing.assert_array_equal(image.data_spatial[0, 0, :], [1, 2, 3])
    np.testing.assert_array_equal(image.data_spatial[2, 1, :], [4, 5, 6])
    expected_fg_mask = DataArray(
        [[True, False], [False, False], [False, True]], dims=("y", "x"), coords={"y": [0, 1, 2], "x": [0, 1]}
    )
    xarray.testing.assert_equal(image.fg_mask, expected_fg_mask)


def test_n_channels(mock_image: MultiChannelImage) -> None:
    assert mock_image.n_channels == 2


def test_n_nonzero(mock_image: MultiChannelImage) -> None:
    assert mock_image.n_nonzero == 6


def test_n_nonzero_when_sparse(mock_image: MultiChannelImage) -> None:
    mock_image._is_foreground[1, 0] = False
    assert mock_image.n_nonzero == 5


def test_dtype(mock_image: MultiChannelImage) -> None:
    assert mock_image.dtype == float


def test_is_foreground_label(mock_image: MultiChannelImage) -> None:
    assert mock_image.is_foreground_label == "is_foreground"


def test_bg_mask(mock_image: MultiChannelImage) -> None:
    # TODO more interesting example
    expected_bg_mask = DataArray([[False, False], [False, False], [False, False]], dims=("y", "x"))
    xarray.testing.assert_equal(expected_bg_mask, mock_image.bg_mask)


def test_bg_mask_flat(mock_image: MultiChannelImage) -> None:
    # TODO more interesting example
    expected_bg_mask_flat = DataArray(
        [False, False, False, False, False, False],
        dims="i",
        coords={"i": pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)], names=("y", "x"))},
    )
    xarray.testing.assert_equal(expected_bg_mask_flat, mock_image.bg_mask_flat)


def test_fg_mask(mock_image_sparse) -> None:
    expected_fg_mask = DataArray([[False, False], [True, True], [False, False]], dims=("y", "x"))
    xarray.testing.assert_equal(expected_fg_mask, mock_image_sparse.fg_mask)


def test_fg_mask_flat(mock_image_sparse) -> None:
    expected_fg_mask_flat = DataArray(
        [False, False, True, True, False, False],
        dims="i",
        coords={"i": pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)], names=("y", "x"))},
    )
    xarray.testing.assert_equal(expected_fg_mask_flat, mock_image_sparse.fg_mask_flat)


def test_dimensions(mock_image: MultiChannelImage) -> None:
    assert mock_image.dimensions == (2, 3)


def test_channel_names_when_set(mock_image: MultiChannelImage) -> None:
    # TODO there should be some functionality to make it work for on-the-fly generated channel names
    assert mock_image.channel_names == ["Channel A", "Channel B"]


def test_data_spatial(mock_data: DataArray, mock_image: MultiChannelImage) -> None:
    xarray.testing.assert_identical(mock_data, mock_image.data_spatial)


def test_data_flat(mock_image: MultiChannelImage) -> None:
    mock_image._is_foreground[0, 0] = False
    mock_image._is_foreground[1, 0] = False
    expected = DataArray(
        [[4.0, 8, 10, 12], [5, 5, 5, 5]],
        dims=("c", "i"),
        coords={
            "c": ["Channel A", "Channel B"],
            "i": pd.MultiIndex.from_tuples([(0, 1), (1, 1), (2, 0), (2, 1)], names=("y", "x")),
        },
    )
    xarray.testing.assert_identical(mock_image.data_flat, expected)


def test_data_flat_preserves_fg_nan(mock_image: MultiChannelImage) -> None:
    mock_image._is_foreground[0, 0] = False
    mock_image.data_spatial[1, 0, 0] = np.nan
    expected = DataArray(
        [[4.0, np.nan, 8, 10, 12], [5, 5, 5, 5, 5]],
        dims=("c", "i"),
        coords={
            "c": ["Channel A", "Channel B"],
            "i": pd.MultiIndex.from_tuples([(0, 1), (1, 0), (1, 1), (2, 0), (2, 1)], names=("y", "x")),
        },
    )
    xarray.testing.assert_identical(mock_image.data_flat, expected)


def test_coordinates_flat(mock_data: DataArray, mock_image: MultiChannelImage) -> None:
    mock_image._is_foreground[0, 0] = False
    mock_image._is_foreground[1, 0] = False
    expected = DataArray(
        [[0, 1, 2, 2], [1, 1, 0, 1]],
        dims=("d", "i"),
        coords={
            "d": ["y", "x"],
            "i": pd.MultiIndex.from_tuples([(0, 1), (1, 1), (2, 0), (2, 1)], names=("y", "x")),
        },
    )
    xarray.testing.assert_identical(mock_image.coordinates_flat, expected)


def test_recompute_is_foreground(mocker: MockerFixture, mock_image: MultiChannelImage) -> None:
    mock_compute = mocker.patch.object(
        MultiChannelImage, "_compute_is_foreground", return_value=xarray.ones_like(mock_image.fg_mask)
    )
    new_image = mock_image.recompute_is_foreground()
    xarray.testing.assert_equal(new_image.fg_mask, mock_compute.return_value)
    xarray.testing.assert_equal(new_image.data_spatial, mock_image.data_spatial)


def test_retain_channels_when_both_none(mock_image: MultiChannelImage) -> None:
    with pytest.raises(ValueError):
        mock_image.retain_channels(None, None)


def test_retain_channels_by_indices(mock_image: MultiChannelImage) -> None:
    indices = [1]
    result = mock_image.retain_channels(indices=indices)
    assert result.channel_names == ["Channel B"]
    np.testing.assert_array_equal(result.data_spatial.values, mock_image.data_spatial.values[:, :, [1]])


def test_retain_channels_by_coords(mock_image: MultiChannelImage) -> None:
    coords = ["Channel B"]
    result = mock_image.retain_channels(coords=coords)
    assert result.channel_names == coords
    np.testing.assert_array_equal(result.data_spatial.values, mock_image.data_spatial.values[:, :, [1]])


def test_retain_channels_when_both_provided(mock_image: MultiChannelImage) -> None:
    with pytest.raises(ValueError):
        mock_image.retain_channels(indices=[0, 1], coords=["red", "blue"])


def test_drop_channels_when_coords_and_allow_missing(mock_image: MultiChannelImage) -> None:
    image = mock_image.drop_channels(coords=["Channel A", "Channel NeverExisted"], allow_missing=True)
    assert image.channel_names == ["Channel B"]


def test_drop_channels_when_coords_and_not_allow_missing(mock_image: MultiChannelImage) -> None:
    assert mock_image.drop_channels(coords=["Channel A"], allow_missing=False).channel_names == ["Channel B"]
    with pytest.raises(KeyError) as error:
        mock_image.drop_channels(coords=["Channel A", "Channel NeverExisted"], allow_missing=False)
    assert "Channel NeverExisted" in str(error.value)


def test_write_hdf5(mocker: MockerFixture, mock_image: MultiChannelImage) -> None:
    mocker.patch("xarray.DataArray.to_netcdf")
    mock_image.write_hdf5(Path("test.h5"))
    mock_image.data_spatial.to_netcdf.assert_called_once_with(
        Path("test.h5"), engine="netcdf4", format="NETCDF4", group=None, mode="w"
    )


def test_read_hdf5(mocker: MockerFixture, mock_data: DataArray) -> None:
    mock_is_foreground = xarray.ones_like(mock_data.isel(c=[0])).assign_coords(c=["is_foreground"])
    persisted_data = xarray.concat([mock_data, mock_is_foreground], dim="c")
    mocker.patch("xarray.open_dataarray").return_value = persisted_data
    image = MultiChannelImage.read_hdf5(Path("test.h5"))
    xarray.open_dataarray.assert_called_once_with(Path("test.h5"), group=None)
    xarray.testing.assert_equal(image.data_spatial, mock_data)
    xarray.testing.assert_equal(image.fg_mask, mock_is_foreground.isel(c=0).drop_vars("c"))


def test_read_ome_tiff(mocker: MockerFixture, mock_data: DataArray) -> None:
    mock_read = mocker.patch.object(OmeTiff, "read", return_value=mock_data)
    mock_foreground = xarray.ones_like(mock_data.isel(c=0), dtype=bool).drop_vars("c")
    mocker.patch.object(MultiChannelImage, "_compute_is_foreground", return_value=mock_foreground)
    image = MultiChannelImage.read_ome_tiff(Path("test.ome.tiff"))
    xarray.testing.assert_equal(image.data_spatial, mock_data)
    mock_read.assert_called_once_with(Path("test.ome.tiff"))
    xarray.testing.assert_equal(image.fg_mask, mock_foreground)


def test_with_channel_names(mock_image: MultiChannelImage) -> None:
    image = mock_image.with_channel_names(channel_names=["New Channel Name", "B"])
    assert image.channel_names == ["New Channel Name", "B"]
    assert image.dimensions == mock_image.dimensions
    assert image.n_channels == mock_image.n_channels == 2
    np.testing.assert_array_equal(image.data_spatial.values, mock_image.data_spatial.values)


def test_channel_stats(mocker: MockerFixture, mock_image: MultiChannelImage) -> None:
    mock_image_channel_stats = mocker.patch("depiction.image.multi_channel_image.ImageChannelStats")
    assert mock_image.channel_stats == mock_image_channel_stats.return_value
    # call twice
    assert mock_image.channel_stats == mock_image_channel_stats.return_value
    mock_image_channel_stats.assert_called_once_with(image=mock_image)


def test_append_channels(mock_image: MultiChannelImage) -> None:
    extra_image_data = DataArray(
        data=np.arange(12).reshape(3, 2, 2),
        dims=("y", "x", "c"),
        coords={"c": ["Channel X", "Channel Y"]},
    )
    extra_image = MultiChannelImage(data=extra_image_data, is_foreground=mock_image.fg_mask)
    result = mock_image.append_channels(extra_image)
    assert result.channel_names == ["Channel A", "Channel B", "Channel X", "Channel Y"]
    assert result.retain_channels(coords=["Channel A", "Channel B"]).data_spatial.identical(mock_image.data_spatial)
    assert result.retain_channels(coords=["Channel X", "Channel Y"]).data_spatial.identical(extra_image.data_spatial)


def test_get_z_scaled(mock_image: MultiChannelImage) -> None:
    result = mock_image.get_z_scaled()
    np.testing.assert_almost_equal(
        np.array([[-1.46385011, -0.87831007], [-0.29277002, 0.29277002], [0.87831007, 1.46385011]]),
        result.data_spatial[:, :, 0].values,
    )
    np.testing.assert_almost_equal([[1.0, 1], [1, 1], [1, 1]], result.data_spatial[:, :, 1].values)


# def test_replace_bg_value(mock_data: DataArray, mock_image: MultiChannelImage) -> None:
#    mock_data[0, 0, :] = 0
#    mock_data[1, 0, 0] = np.nan
#
#    new_image = mock_image.replace_bg_value(42)
#    assert new_image.bg_value == 42
#    assert new_image.data_spatial[0, 0, 0] == 42
#    assert new_image.data_spatial[0, 0, 1] == 42
#    assert np.isnan(new_image.data_spatial[1, 0, 0])
#    assert new_image.data_spatial[1, 0, 1] == 5


def test_str(mock_image: MultiChannelImage) -> None:
    assert str(mock_image) == "MultiChannelImage(size_y=3, size_x=2, n_channels=2)"


def test_repr(mocker: MockerFixture, mock_image: MultiChannelImage) -> None:
    mocker.patch("xarray.DataArray.__repr__", return_value="DataArray")
    assert repr(mock_image) == "MultiChannelImage(data=DataArray)"


@pytest.mark.parametrize("bg_value", [0, 1, np.nan])
def test_compute_is_foreground(bg_value: float):
    a = bg_value - 1 if np.isfinite(bg_value) else 1
    array = DataArray(
        [
            [[bg_value, bg_value], [a, a]],
            [[3, bg_value], [bg_value, bg_value]],
            [[bg_value, bg_value], [bg_value, bg_value]],
        ],
        dims=("y", "x", "c"),
    )
    mask = MultiChannelImage._compute_is_foreground(array, bg_value=bg_value)
    xarray.testing.assert_equal(DataArray([[False, True], [True, False], [False, False]], dims=("y", "x")), mask)


@pytest.mark.parametrize(
    "input_coordinates",
    [
        np.array([[1, 2], [3, 4]]),
        xarray.DataArray([[1, 2], [3, 4]], dims=("i", "d"), coords={"d": ["x", "y"]}),
        xarray.DataArray([[2, 1], [4, 3]], dims=("i", "d"), coords={"d": ["y", "x"]}),
        xarray.DataArray([[1, 3], [2, 4]], dims=("d", "i"), coords={"d": ["x", "y"]}),
    ],
)
def test_validate_coordinates(input_coordinates):
    result = MultiChannelImage._validate_coordinates(input_coordinates)
    xarray.testing.assert_equal(result, xarray.DataArray([[1, 2], [3, 4]], dims=("i", "d"), coords={"d": ["x", "y"]}))


@pytest.mark.parametrize(
    "input_coordinates",
    [
        xarray.DataArray([[1, 2], [3, 4]], dims=("i", "d"), coords={"d": ["x", "z"]}),
    ],
)
def test_validate_coordinates_when_invalid(input_coordinates):
    with pytest.raises(ValueError):
        MultiChannelImage._validate_coordinates(input_coordinates)


def test_extract_flat_coordinates(mock_image_sparse):
    data_flat = xarray.DataArray(
        [[6.0, 8], [5, 5]], dims=("c", "i"), coords={"i": pd.MultiIndex.from_arrays(([0, 1], [1, 1]), names=("x", "y"))}
    )
    coords = MultiChannelImage._extract_flat_coordinates(data_flat)
    xarray.testing.assert_equal(
        coords,
        xarray.DataArray([[0, 1], [1, 1]], dims=("i", "d"), coords={"d": ["x", "y"]}),
    )


if __name__ == "__main__":
    pytest.main()
