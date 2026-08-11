from unittest.mock import MagicMock, call, ANY

import numpy as np
import pytest

from depiction.spectrum.peak_filtering import FilterNHighestIntensityPartitioned
from depiction.spectrum.peak_filtering.filter_n_highest_intensity_partitioned import (
    FilterNHighestIntensityPartitionedConfig,
)


@pytest.fixture()
def mock_config():
    return FilterNHighestIntensityPartitionedConfig(max_count=5, n_partitions=2)


@pytest.fixture()
def mock_spectrum_mz_arr(mocker):
    return mocker.MagicMock(name="mock_spectrum_mz_arr")


@pytest.fixture()
def mock_filter(mock_config) -> FilterNHighestIntensityPartitioned:
    return FilterNHighestIntensityPartitioned(config=mock_config)


@pytest.fixture()
def unmocked_filter() -> FilterNHighestIntensityPartitioned:
    return FilterNHighestIntensityPartitioned(
        config=FilterNHighestIntensityPartitionedConfig(max_count=4, n_partitions=2)
    )


@pytest.fixture()
def spectrum():
    """A spectrum whose highest peaks are, within each partition, not in m/z order."""
    return np.arange(100, 200, 10.0), np.array([10, 50, 90, 30, 70, 20, 80, 40, 60, 15.0])


def test_filter_index_peaks(mocker, mock_filter) -> None:
    mock_filter_n_highest_intensity = mocker.patch(
        "depiction.spectrum.peak_filtering.filter_n_highest_intensity_partitioned.FilterNHighestIntensity"
    )
    mock_filter_n_highest_intensity.return_value.filter_index_peaks.side_effect = [
        np.array([10, 20, 30]),
        np.array([200, 210]),
    ]
    spectrum_mz_arr = np.linspace(5, 210, 250)
    mock_spectrum_int_arr = MagicMock(name="mock_spectrum_int_arr")
    peak_idx_arr = np.array([10, 20, 30, 40, 190, 195, 197, 200, 210])

    peak_indices = mock_filter.filter_index_peaks(
        spectrum_mz_arr=spectrum_mz_arr,
        spectrum_int_arr=mock_spectrum_int_arr,
        peak_idx_arr=peak_idx_arr,
    )

    np.testing.assert_array_equal([10, 20, 30, 200, 210], peak_indices)
    assert mock_filter_n_highest_intensity.mock_calls == [
        call(max_count=2),
        call().filter_index_peaks(
            spectrum_mz_arr=ANY,
            spectrum_int_arr=mock_spectrum_int_arr,
            peak_idx_arr=ANY,
        ),
        call().filter_index_peaks(
            spectrum_mz_arr=ANY,
            spectrum_int_arr=mock_spectrum_int_arr,
            peak_idx_arr=ANY,
        ),
    ]

    np.testing.assert_array_equal(
        [10, 20, 30, 40],
        mock_filter_n_highest_intensity.return_value.filter_index_peaks.call_args_list[0][1]["peak_idx_arr"],
    )
    np.testing.assert_array_equal(
        [190, 195, 197, 200, 210],
        mock_filter_n_highest_intensity.return_value.filter_index_peaks.call_args_list[1][1]["peak_idx_arr"],
    )


def test_filter_index_peaks_returns_ascending_indices(unmocked_filter, spectrum) -> None:
    # deliberately does not patch FilterNHighestIntensity: the partitions are emitted in
    # ascending m/z, so the only order this can get wrong is the one the delegate returns
    spectrum_mz_arr, spectrum_int_arr = spectrum

    peak_indices = unmocked_filter.filter_index_peaks(
        spectrum_mz_arr=spectrum_mz_arr,
        spectrum_int_arr=spectrum_int_arr,
        peak_idx_arr=np.arange(len(spectrum_mz_arr)),
    )

    np.testing.assert_array_equal([2, 4, 6, 8], peak_indices)
    assert np.all(np.diff(spectrum_mz_arr[peak_indices]) > 0)


def test_filter_peaks_returns_ascending_mz(unmocked_filter, spectrum) -> None:
    # the same spectrum through the other method: both must select the same peaks in the same order
    spectrum_mz_arr, spectrum_int_arr = spectrum

    mz_arr, int_arr = unmocked_filter.filter_peaks(
        spectrum_mz_arr=spectrum_mz_arr,
        spectrum_int_arr=spectrum_int_arr,
        peak_mz_arr=spectrum_mz_arr,
        peak_int_arr=spectrum_int_arr,
    )

    np.testing.assert_array_equal([120.0, 140.0, 160.0, 180.0], mz_arr)
    np.testing.assert_array_equal([90.0, 70.0, 80.0, 60.0], int_arr)
    assert np.all(np.diff(mz_arr) > 0)


def test_filter_index_peaks_when_empty_input(mock_filter) -> None:
    mz_arr = np.array([])
    int_arr = np.array([])
    peak_idx_arr = np.array([])

    indices = mock_filter.filter_index_peaks(
        spectrum_mz_arr=mz_arr,
        spectrum_int_arr=int_arr,
        peak_idx_arr=peak_idx_arr,
    )

    np.testing.assert_array_equal(np.array([]), indices)
