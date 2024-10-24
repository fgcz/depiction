import numpy as np
import pytest

from depiction.calibration.methods.isotope_pattern_matcher import IsotopePatternMatcher


@pytest.fixture()
def mock_pattern_matcher() -> IsotopePatternMatcher:
    return IsotopePatternMatcher(cache_size=10)


def test_get_averagine_pattern_when_cache_empty(mock_pattern_matcher) -> None:
    mz_arr, int_arr = mock_pattern_matcher.get_averagine_pattern(100)
    assert len(mz_arr) == 5
    assert len(int_arr) == 5


# @unittest.skip
# def test_get_averagine_pattern_when_cache_not_empty_but_miss(self) -> NoReturn:
#    raise NotImplementedError


def test_get_averagine_pattern_when_cache_hit_exact(mocker, mock_pattern_matcher) -> None:
    method_compute_averagine_pattern = mocker.patch.object(IsotopePatternMatcher, "_compute_averagine_pattern")
    method_compute_averagine_pattern.return_value = [np.array([100, 110]), "data"]

    mz_arr_1, int_arr_1 = mock_pattern_matcher.get_averagine_pattern(100)
    mz_arr_2, int_arr_2 = mock_pattern_matcher.get_averagine_pattern(100)
    np.testing.assert_array_equal(np.array([100, 110]), mz_arr_1)
    assert int_arr_1 == "data"
    np.testing.assert_array_equal(np.array([100, 110]), mz_arr_2)
    assert int_arr_2 == "data"

    method_compute_averagine_pattern.assert_called_once_with(mass=100)


def test_get_averagine_pattern_when_cache_hit_tolerance(mocker, mock_pattern_matcher) -> None:
    method_compute_averagine_pattern = mocker.patch.object(IsotopePatternMatcher, "_compute_averagine_pattern")
    method_compute_averagine_pattern.return_value = [
        np.array([100.2, 110.2]),
        "data",
    ]

    mz_arr_1, int_arr_1 = mock_pattern_matcher.get_averagine_pattern(100.2)
    mz_arr_2, int_arr_2 = mock_pattern_matcher.get_averagine_pattern(100)
    np.testing.assert_array_equal(np.array([100.2, 110.2]), mz_arr_1)
    assert int_arr_1 == "data"
    np.testing.assert_array_equal(np.array([100, 110]), mz_arr_2)
    assert int_arr_2 == "data"

    method_compute_averagine_pattern.assert_called_once_with(mass=100.2)


# @unittest.skip
# def test_get_averagine_pattern_when_cache_evict(self) -> NoReturn:
#    raise NotImplementedError


def test_compute_spectra_agreement_when_mz_identical(mock_pattern_matcher) -> None:
    spectrum_1 = (np.array([10.0, 20, 30]), np.array([5.0, 6, 7]))
    spectrum_2 = (np.array([10, 20, 30]), np.array([10, 10, 10]))

    agreement_score, n_align = mock_pattern_matcher.compute_spectra_agreement(
        spectrum_1=spectrum_1,
        spectrum_2=spectrum_2,
        n_limit=5,
        distance_tolerance=0.1,
    )

    assert n_align == 3
    assert agreement_score == pytest.approx(0.99, abs=0.01)


def test_compute_spectra_agreement_when_mz_identical_start(mock_pattern_matcher) -> None:
    spectrum_1 = (
        np.array([100.1, 101.1, 102.1, 103.1, 104.1, 105.1]),
        np.array([10, 20, 10, 5, 5]),
    )
    spectrum_2 = (
        np.array([100.1, 101.1, 102.1, 103.1, 110, 120]),
        np.array([10, 20, 10, 5, 5]),
    )

    agreement_score, n_align = mock_pattern_matcher.compute_spectra_agreement(
        spectrum_1=spectrum_1,
        spectrum_2=spectrum_2,
        n_limit=5,
        distance_tolerance=0.1,
    )

    assert n_align == 4
    assert agreement_score == pytest.approx(1, abs=0.01)
