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
