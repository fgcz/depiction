import numpy as np
import pytest
import xarray.testing

from depiction.image.image_channel_stats import ImageChannelStats


@pytest.fixture
def mock_multi_channel_image(mocker):
    return mocker.Mock(name="mock_image", n_channels=3, channel_names=["channel1", "channel2", "channel3"], bg_value=0)


@pytest.fixture
def image_channel_stats(mock_multi_channel_image):
    return ImageChannelStats(mock_multi_channel_image)


def test_init(mock_multi_channel_image):
    stats = ImageChannelStats(mock_multi_channel_image)
    assert stats._image == mock_multi_channel_image


def test_five_number_summary(mocker, image_channel_stats, mock_multi_channel_image):
    mock_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    mocker.patch.object(
        image_channel_stats, "_get_channel_values", side_effect=[mock_data[0], mock_data[1], mock_data[2]]
    )
    result = image_channel_stats.five_number_summary

    expected = xarray.DataArray(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        dims=("c", "metric"),
        coords={"c": ["channel1", "channel2", "channel3"], "metric": ["min", "q1", "median", "q3", "max"]},
    )
    xarray.testing.assert_equal(result, expected)


def test_coefficient_of_variation(mocker, image_channel_stats, mock_multi_channel_image):
    mock_data = np.array([[1, 2, 3, 4, 5], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    mocker.patch.object(
        image_channel_stats, "_get_channel_values", side_effect=[mock_data[0], mock_data[1], mock_data[2]]
    )
    result = image_channel_stats.coefficient_of_variation
    xarray.testing.assert_allclose(
        result, xarray.DataArray([0.471404, 0.0, np.nan], coords={"c": ["channel1", "channel2", "channel3"]}, dims="c")
    )


def test_interquartile_range(mocker, image_channel_stats, mock_multi_channel_image):
    mock_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    mocker.patch.object(
        image_channel_stats, "_get_channel_values", side_effect=[mock_data[0], mock_data[1], mock_data[2]]
    )
    result = image_channel_stats.interquartile_range
    xarray.testing.assert_allclose(
        result, xarray.DataArray([2, 2, 2], coords={"c": ["channel1", "channel2", "channel3"]}, dims="c")
    )


def test_mean(mocker, image_channel_stats, mock_multi_channel_image):
    mock_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    mocker.patch.object(
        image_channel_stats, "_get_channel_values", side_effect=[mock_data[0], mock_data[1], mock_data[2]]
    )
    result = image_channel_stats.mean
    xarray.testing.assert_allclose(
        result, xarray.DataArray([3, 8, 13], coords={"c": ["channel1", "channel2", "channel3"]}, dims="c")
    )


def test_std(mocker, image_channel_stats, mock_multi_channel_image):
    mock_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    mocker.patch.object(
        image_channel_stats, "_get_channel_values", side_effect=[mock_data[0], mock_data[1], mock_data[2]]
    )
    result = image_channel_stats.std
    xarray.testing.assert_allclose(
        result,
        xarray.DataArray([1.414214, 1.414214, 1.414214], coords={"c": ["channel1", "channel2", "channel3"]}, dims="c"),
    )


def test_get_channel_values(image_channel_stats, mock_multi_channel_image):
    mock_data = np.array([1, 2, 3, 0, 5])
    mock_multi_channel_image.data_flat.isel.return_value.values = mock_data

    # Test without dropping missing values
    result = image_channel_stats._get_channel_values(0, drop_missing=False)
    np.testing.assert_array_equal(result, mock_data)

    # Test with dropping missing values (bg_value = 0)
    result = image_channel_stats._get_channel_values(0, drop_missing=True)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 5]))


def test_get_channel_values_with_nan(image_channel_stats, mock_multi_channel_image):
    mock_data = np.array([1, 2, 3, np.nan, 5])
    mock_multi_channel_image.data_flat.isel.return_value.values = mock_data
    mock_multi_channel_image.bg_value = np.nan

    # Test without dropping missing values
    result = image_channel_stats._get_channel_values(0, drop_missing=False)
    np.testing.assert_array_equal(result, mock_data)

    # Test with dropping missing values (bg_value = nan)
    result = image_channel_stats._get_channel_values(0, drop_missing=True)
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 5]))


def test_five_number_summary_when_empty_channel(mocker, image_channel_stats, mock_multi_channel_image):
    mock_data = np.array([])
    mocker.patch.object(image_channel_stats, "_get_channel_values", return_value=mock_data)
    five_number_summary = image_channel_stats.five_number_summary
    assert np.isnan(five_number_summary.sel(metric="min", c="channel1").item())
    assert np.isnan(five_number_summary.sel(metric="q1", c="channel1").item())
    assert np.isnan(five_number_summary.sel(metric="median", c="channel1").item())
    assert np.isnan(five_number_summary.sel(metric="q3", c="channel1").item())
    assert np.isnan(five_number_summary.sel(metric="max", c="channel1").item())


def test_interquartile_range_when_emtpy_channel(mocker, image_channel_stats, mock_multi_channel_image):
    mock_data = np.array([])
    mocker.patch.object(image_channel_stats, "_get_channel_values", return_value=mock_data)
    interquartile_range = image_channel_stats.interquartile_range
    xarray.testing.assert_equal(
        interquartile_range,
        xarray.DataArray([None, None, None], coords={"c": ["channel1", "channel2", "channel3"]}, dims="c"),
    )


if __name__ == "__main__":
    pytest.main()
