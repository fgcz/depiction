import pytest
from pytest_mock import MockerFixture

from depiction.spectrum.peak_filtering import ChainFilters


@pytest.fixture()
def mock_filter_1(mocker: MockerFixture):
    return mocker.MagicMock(name="mock_filter_1")


@pytest.fixture()
def mock_filter_2(mocker: MockerFixture):
    return mocker.MagicMock(name="mock_filter_2")


@pytest.fixture()
def chain_filters(mock_filter_1, mock_filter_2):
    return ChainFilters(filters=[mock_filter_1, mock_filter_2])


def test_filter_index_peaks(mocker, mock_filter_1, mock_filter_2, chain_filters) -> None:
    mock_filter_1.filter_index_peaks.return_value = [1, 2, 3]
    mock_filter_2.filter_index_peaks.return_value = [2, 3]
    mock_spectrum_mz_arr = mocker.MagicMock(name="mock_spectrum_mz_arr")
    mock_spectrum_int_arr = mocker.MagicMock(name="mock_spectrum_int_arr")
    peak_idx_arr = [1, 2, 3, 4]

    result_peak_idx_arr = chain_filters.filter_index_peaks(
        spectrum_mz_arr=mock_spectrum_mz_arr,
        spectrum_int_arr=mock_spectrum_int_arr,
        peak_idx_arr=peak_idx_arr,
    )

    assert result_peak_idx_arr == [2, 3]
    mock_filter_1.filter_index_peaks.assert_called_once_with(
        mock_spectrum_mz_arr,
        mock_spectrum_int_arr,
        peak_idx_arr,
    )
    mock_filter_2.filter_index_peaks.assert_called_once_with(
        mock_spectrum_mz_arr,
        mock_spectrum_int_arr,
        [1, 2, 3],
    )


def test_filter_peaks(mocker, mock_filter_1, mock_filter_2, chain_filters) -> None:
    mock_filter_1.filter_peaks.return_value = ([1, 2, 3], [4, 5, 6])
    mock_filter_2.filter_peaks.return_value = ([2, 3], [5, 6])
    mock_spectrum_mz_arr = mocker.MagicMock(name="mock_spectrum_mz_arr")
    mock_spectrum_int_arr = mocker.MagicMock(name="mock_spectrum_int_arr")
    peak_mz_arr = [1, 2, 3, 4]
    peak_int_arr = [4, 5, 6, 7]

    result_peak_mz_arr, result_peak_int_arr = chain_filters.filter_peaks(
        spectrum_mz_arr=mock_spectrum_mz_arr,
        spectrum_int_arr=mock_spectrum_int_arr,
        peak_mz_arr=peak_mz_arr,
        peak_int_arr=peak_int_arr,
    )

    assert result_peak_mz_arr == [2, 3]
    assert result_peak_int_arr == [5, 6]
    mock_filter_1.filter_peaks.assert_called_once_with(
        mock_spectrum_mz_arr,
        mock_spectrum_int_arr,
        peak_mz_arr,
        peak_int_arr,
    )
    mock_filter_2.filter_peaks.assert_called_once_with(
        mock_spectrum_mz_arr,
        mock_spectrum_int_arr,
        [1, 2, 3],
        [4, 5, 6],
    )
