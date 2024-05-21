import unittest
from functools import cached_property
from unittest.mock import patch

import numpy as np

from ionplotter.calibration.isotope_pattern_matcher import IsotopePatternMatcher
from typing import NoReturn


class TestIsotopePatternMatcher(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_cache_size = 10

    @cached_property
    def mock_pattern_matcher(self) -> IsotopePatternMatcher:
        return IsotopePatternMatcher(cache_size=self.mock_cache_size)

    def test_get_averagine_pattern_when_cache_empty(self) -> None:
        mz_arr, int_arr = self.mock_pattern_matcher.get_averagine_pattern(100)
        self.assertEqual(5, len(mz_arr))
        self.assertEqual(5, len(int_arr))

    @unittest.skip
    def test_get_averagine_pattern_when_cache_not_empty_but_miss(self) -> NoReturn:
        raise NotImplementedError

    @patch.object(IsotopePatternMatcher, "_compute_averagine_pattern")
    def test_get_averagine_pattern_when_cache_hit_exact(self, method_compute_averagine_pattern) -> None:
        method_compute_averagine_pattern.return_value = [np.array([100, 110]), "data"]

        mz_arr_1, int_arr_1 = self.mock_pattern_matcher.get_averagine_pattern(100)
        mz_arr_2, int_arr_2 = self.mock_pattern_matcher.get_averagine_pattern(100)
        np.testing.assert_array_equal(np.array([100, 110]), mz_arr_1)
        self.assertEqual("data", int_arr_1)
        np.testing.assert_array_equal(np.array([100, 110]), mz_arr_2)
        self.assertEqual("data", int_arr_2)

        method_compute_averagine_pattern.assert_called_once_with(mass=100)

    @patch.object(IsotopePatternMatcher, "_compute_averagine_pattern")
    def test_get_averagine_pattern_when_cache_hit_tolerance(self, method_compute_averagine_pattern) -> None:
        method_compute_averagine_pattern.return_value = [
            np.array([100.2, 110.2]),
            "data",
        ]

        mz_arr_1, int_arr_1 = self.mock_pattern_matcher.get_averagine_pattern(100.2)
        mz_arr_2, int_arr_2 = self.mock_pattern_matcher.get_averagine_pattern(100)
        np.testing.assert_array_equal(np.array([100.2, 110.2]), mz_arr_1)
        self.assertEqual("data", int_arr_1)
        np.testing.assert_array_equal(np.array([100, 110]), mz_arr_2)
        self.assertEqual("data", int_arr_2)

        method_compute_averagine_pattern.assert_called_once_with(mass=100.2)

    @unittest.skip
    def test_get_averagine_pattern_when_cache_evict(self) -> NoReturn:
        raise NotImplementedError

    def test_compute_spectra_agreement_when_mz_identical(self) -> None:
        spectrum_1 = (np.array([10.0, 20, 30]), np.array([5.0, 6, 7]))
        spectrum_2 = (np.array([10, 20, 30]), np.array([10, 10, 10]))

        agreement_score, n_align = self.mock_pattern_matcher.compute_spectra_agreement(
            spectrum_1=spectrum_1,
            spectrum_2=spectrum_2,
            n_limit=5,
            distance_tolerance=0.1,
        )

        self.assertEqual(3, n_align)
        self.assertAlmostEqual(0.99, agreement_score, places=2)

    def test_compute_spectra_agreement_when_mz_identical_start(self) -> None:
        spectrum_1 = (
            np.array([100.1, 101.1, 102.1, 103.1, 104.1, 105.1]),
            np.array([10, 20, 10, 5, 5]),
        )
        spectrum_2 = (
            np.array([100.1, 101.1, 102.1, 103.1, 110, 120]),
            np.array([10, 20, 10, 5, 5]),
        )

        agreement_score, n_align = self.mock_pattern_matcher.compute_spectra_agreement(
            spectrum_1=spectrum_1,
            spectrum_2=spectrum_2,
            n_limit=5,
            distance_tolerance=0.1,
        )

        self.assertEqual(4, n_align)
        self.assertAlmostEqual(1, agreement_score, places=2)


if __name__ == "__main__":
    unittest.main()
