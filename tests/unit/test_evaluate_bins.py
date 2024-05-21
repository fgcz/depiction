import unittest
from functools import cached_property

import numpy as np

from ionplotter.spectrum.evaluate_bins import BinStatistic, EvaluateBins


class TestEvaluateBins(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_bin_edges = [1, 2, 3, 4]
        self.mock_statistic = BinStatistic.MEAN

    @cached_property
    def mock_evaluate_bins(self):
        return EvaluateBins(bin_edges=self.mock_bin_edges, statistic=self.mock_statistic)

    def test_evaluate(self) -> None:
        mz_values = np.array([0.5, 1.5, 1.5, 1.5, 3.5, 4.5])
        int_values = np.array([9, 2, 4, 6, 8, 7])
        binned_values = self.mock_evaluate_bins.evaluate(mz_values, int_values)
        np.testing.assert_array_equal(np.array([4, 0, 8]), binned_values)

    # TODO maybe recover the tests (but the method has been deleted)
    # def test_evaluate_all(self):
    #    self.mock_bin_edges = [110, 120, 140]
    #    mz_values = [
    #        np.array([119.5, 121.5, 127.1]),
    #        np.array([126.2, 130.2, 135.8, 173]),
    #    ]
    #    int_values = [np.array([1.0, 2.0, 4.0]), np.array([3.0, 6, 9, 64])]
    #    binned = self.mock_evaluate_bins.evaluate_all(mz_values, int_values, n_jobs=1)
    #    np.testing.assert_array_equal(np.array([[1.0, 3], [0, 6]]), binned)

    def test_mz_values(self) -> None:
        np.testing.assert_array_equal(np.array([1.5, 2.5, 3.5]), self.mock_evaluate_bins.mz_values)

    def test_bin_edges(self) -> None:
        np.testing.assert_array_equal(np.array([1, 2, 3, 4]), self.mock_evaluate_bins.bin_edges)

    def test_from_mz_values_when_center(self) -> None:
        mz_values = np.array([10.0, 160, 310, 460])
        eval_bins = EvaluateBins.from_mz_values(mz_values)
        np.testing.assert_array_equal(np.array([10.0, 160, 310, 460]), eval_bins.mz_values)

    def test_from_mz_values_when_not_center(self) -> None:
        mz_values = np.array([1, 10, 100, 1000], dtype=float)
        eval_bins = EvaluateBins.from_mz_values(mz_values)
        np.testing.assert_array_equal(np.array([1, 30.25, 302.5, 1000]), eval_bins.mz_values)

    # TODO as above
    # def test_is_aligned_when_true(self):
    #    mz_arr = np.array([1, 2, 3, 4])
    #    self.assertTrue(EvaluateBins.is_aligned([mz_arr, mz_arr, mz_arr]))

    # def test_is_aligned_when_false(self):
    #    mz_values = [np.array([1, 2, 3, 4]), np.array([1, 2, 4, 5])]
    #    self.assertFalse(EvaluateBins.is_aligned(mz_values))

    # def test_is_aligned_when_different_lengths(self):
    #    mz_values = [np.array([1, 2, 3, 4]), np.array([1, 2, 3])]
    #    self.assertFalse(EvaluateBins.is_aligned(mz_values))


if __name__ == "__main__":
    unittest.main()
