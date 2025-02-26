import numpy as np
import pytest

pytest.importorskip("alphapept")

from depiction.spectrum.peak_filtering.filter_by_isotope_pattern import (
    FilterByIsotopePattern,
)


@pytest.fixture
def mock_agreement_threshold():
    return 0.8


@pytest.fixture
def mock_n_isotopic_peaks_max():
    return 5


@pytest.fixture
def mock_mass_distance_tolerance():
    return 0.2


@pytest.fixture
def mock_filter(mock_agreement_threshold, mock_n_isotopic_peaks_max, mock_mass_distance_tolerance):
    return FilterByIsotopePattern(
        agreement_threshold=mock_agreement_threshold,
        n_isotopic_peaks_min=3,
        n_isotopic_peaks_max=mock_n_isotopic_peaks_max,
        mass_distance_tolerance=mock_mass_distance_tolerance,
    )


def test_when_averagine(mock_filter):
    mz_pat = np.array(
        [
            200.11793518,
            201.12080382,
            202.12367246,
            203.1265411,
            208.1,
            210.2,
        ]
    )
    int_pat = np.array(
        [
            1.00000000e00,
            1.07631142e-01,
            1.14155451e-02,
            8.14533746e-04,
            4.49574778e-05,
            2.28225783e-06,
        ]
    )

    idx_peaks = mock_filter.filter_index_peaks(
        spectrum_mz_arr=mz_pat,
        spectrum_int_arr=int_pat,
        peak_idx_arr=np.arange(len(mz_pat), dtype=int),
    )
    np.testing.assert_array_equal([0, 1, 2, 3], idx_peaks)


def test_when_empty_input(mock_filter):
    mz_arr = np.array([])
    int_arr = np.array([])
    peak_idx_arr = np.array([])

    indices = mock_filter.filter_index_peaks(
        spectrum_mz_arr=mz_arr,
        spectrum_int_arr=int_arr,
        peak_idx_arr=peak_idx_arr,
    )

    np.testing.assert_array_equal(np.array([]), indices)
