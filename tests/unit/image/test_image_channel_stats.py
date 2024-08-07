import numpy as np
import polars as pl
import pytest
import polars.testing
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

    expected = pl.DataFrame(
        {
            "c": ["channel1", "channel2", "channel3"],
            "min": [1, 6, 11],
            "q1": [2, 7, 12],
            "median": [3, 8, 13],
            "q3": [4, 9, 14],
            "max": [5, 10, 15],
        }
    )

    assert result.equals(expected)


def test_coefficient_of_variation(mocker, image_channel_stats, mock_multi_channel_image):
    mock_data = np.array([[1, 2, 3, 4, 5], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    mocker.patch.object(
        image_channel_stats, "_get_channel_values", side_effect=[mock_data[0], mock_data[1], mock_data[2]]
    )
    result = image_channel_stats.coefficient_of_variation
    expected = pl.DataFrame(
        {"c": ["channel1", "channel2", "channel3"], "cv": np.array([0.471404, 0.0, np.nan])}
    ).fill_nan(None)
    pl.testing.assert_frame_equal(result, expected, check_dtype=False, atol=1e-5)


def test_interquartile_range(mocker, image_channel_stats, mock_multi_channel_image):
    mock_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    mocker.patch.object(
        image_channel_stats, "_get_channel_values", side_effect=[mock_data[0], mock_data[1], mock_data[2]]
    )
    result = image_channel_stats.interquartile_range
    expected = pl.DataFrame({"c": ["channel1", "channel2", "channel3"], "iqr": [2, 2, 2]})
    assert result.equals(expected)


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


def test_empty_channel(mocker, image_channel_stats, mock_multi_channel_image):
    mock_data = np.array([])

    mocker.patch.object(image_channel_stats, "_get_channel_values", return_value=mock_data)
    five_number_summary = image_channel_stats.five_number_summary
    interquartile_range = image_channel_stats.interquartile_range

    assert five_number_summary["min"][0] is None
    assert five_number_summary["q1"][0] is None
    assert five_number_summary["median"][0] is None
    assert five_number_summary["q3"][0] is None
    assert five_number_summary["max"][0] is None
    assert interquartile_range["iqr"][0] is None


if __name__ == "__main__":
    pytest.main()
