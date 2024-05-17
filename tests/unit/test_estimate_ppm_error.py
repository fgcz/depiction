import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ionmapper.estimate_ppm_error import EstimatePPMError
from ionmapper.parallel_ops import ReadSpectraParallel, ParallelConfig


class TestEstimatePPMError(unittest.TestCase):
    @patch.object(ReadSpectraParallel, "from_config")
    def test_estimate(self, mock_from_config) -> None:
        mock_read_file = MagicMock(name="mock_read_file", spec=[])
        mock_from_config.return_value.map_chunked.return_value = [
            (np.array([1000, np.nan, 1003, np.nan]), 1000, 1003),
            (np.array([1005, np.nan, np.nan]), 700, 1000),
        ]

        parallel_config = ParallelConfig(n_jobs=2, task_size=None)
        estimate_ppm = EstimatePPMError(parallel_config=parallel_config)
        result = estimate_ppm.estimate(mock_read_file)

        self.assertEqual(1003.0, result["ppm_median"])
        self.assertAlmostEqual(2.0548046676563256, result["ppm_std"])
        self.assertEqual(700, result["mz_min"])
        self.assertEqual(1003, result["mz_max"])

        mock_from_config.assert_called_once_with(config=parallel_config)

    def test_get_ppm_values(self) -> None:
        mock_reader = MagicMock(name="mock_reader")
        mock_reader.get_spectrum_mz.side_effect = [
            np.array([1000, 1001, 1002, 1003]),
            np.array([]),
            np.array([0]),
            np.array([1050, 1055]),
        ]
        result_ppm, result_min, result_max = EstimatePPMError._get_ppm_values(mock_reader, range(4))
        np.testing.assert_array_almost_equal(
            np.array([999.000999000999, np.nan, np.nan, 4761.904761904762]), result_ppm
        )
        self.assertEqual(1000, result_min)
        self.assertEqual(1055, result_max)

    def test_ppm_to_mz_values(self) -> None:
        ppm_error = 500
        mz_min = 100
        mz_max = 101
        values = EstimatePPMError.ppm_to_mz_values(ppm_error, mz_min, mz_max)
        expected_values = [
            100.0,
            100.05239513,
            100.10481646,
            100.157264,
            100.20973776,
            100.26223776,
            100.31476401,
            100.36731653,
            100.41989532,
            100.4725004,
            100.52513178,
            100.57778948,
            100.6304735,
            100.68318387,
            100.73592059,
            100.78868369,
            100.84147316,
            100.89428903,
            100.9471313,
            101.0,
        ]
        np.testing.assert_array_almost_equal(expected_values, values)


if __name__ == "__main__":
    unittest.main()
