import unittest
from functools import cached_property
from unittest.mock import MagicMock

import numpy as np

from depiction.tools.experimental.adjust_median_shift import AdjustMedianShift
from typing import NoReturn


class TestAdjustMedianShift(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_peak_picker = MagicMock(name="mock_peak_picker")
        self.mock_ref_mz_arr = MagicMock(name="mock_ref_mz_arr")
        self.mock_parallel_config = MagicMock(name="mock_parallel_config")

    @cached_property
    def instance(self) -> AdjustMedianShift:
        return AdjustMedianShift(
            peak_picker=self.mock_peak_picker,
            ref_mz_arr=self.mock_ref_mz_arr,
            parallel_config=self.mock_parallel_config,
        )

    def test_compute_median_shift_ppm(self) -> None:
        self.mock_ref_mz_arr = np.array([100.0, 150.0, 200.0, 300, 400])
        mock_int_arr = np.ones(5)
        ppm_error_arr = np.array([200, 400, 200, 700, 700])
        mock_mz_arr = np.array(
            [ref_mz + ref_mz * ppm_error * 1e-6 for ref_mz, ppm_error in zip(self.mock_ref_mz_arr, ppm_error_arr)]
        )
        self.mock_peak_picker.pick_peaks_index.return_value = np.array([0, 1, 2])

        median_shift = self.instance.compute_median_shift_ppm(mz_arr=mock_mz_arr, int_arr=mock_int_arr)

        self.assertAlmostEqual(200.0, median_shift, places=7)

    @unittest.skip
    def test_compute_median_shifts(self) -> NoReturn:
        raise NotImplementedError

    @unittest.skip
    def test_smooth_median_shifts(self) -> NoReturn:
        raise NotImplementedError

    @unittest.skip
    def test_apply_correction(self) -> NoReturn:
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
