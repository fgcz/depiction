import unittest
from functools import cached_property
from unittest.mock import MagicMock, patch

import numpy as np

from ionplotter.spectrum.evaluate_mean_spectrum import EvaluateMeanSpectrum
from ionplotter.parallel_ops import ReadSpectraParallel
from ionplotter.persistence import ImzmlModeEnum


class TestEvaluateMeanSpectrum(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_parallel_config = MagicMock(name="mock_parallel_config", n_jobs=2, task_size=None, verbose=True)
        self.mock_eval_bins = MagicMock(name="mock_eval_bins", spec=["evaluate"])

    @cached_property
    def mock_evaluate(self) -> EvaluateMeanSpectrum:
        return EvaluateMeanSpectrum(parallel_config=self.mock_parallel_config, eval_bins=self.mock_eval_bins)

    def test_evaluate_file_when_not_continuous_and_no_bins(self) -> None:
        mock_input_file = MagicMock(name="mock_input_file", imzml_mode=ImzmlModeEnum.PROCESSED)
        self.mock_eval_bins = None

        with self.assertRaises(ValueError) as error:
            self.mock_evaluate.evaluate_file(mock_input_file)

        self.assertIn("Input file", str(error.exception))

    @patch.object(EvaluateMeanSpectrum, "_get_spectra_sum")
    def test_evaluate_file_when_not_eval_bins(self, method_get_spectra_sum) -> None:
        mock_input_file = MagicMock(name="mock_input_file", imzml_mode=ImzmlModeEnum.CONTINUOUS, n_spectra=10)
        mock_mz_arr = MagicMock(name="mock_mz_arr", spec=[])
        mock_input_file.reader.return_value.__enter__.return_value.get_spectrum_mz.return_value = mock_mz_arr
        method_get_spectra_sum.return_value = np.array([4.0, 5, 6])
        self.mock_eval_bins = None

        mz_arr, int_arr = self.mock_evaluate.evaluate_file(mock_input_file)

        self.assertEqual(mock_mz_arr, mz_arr)
        np.testing.assert_array_equal(np.array([0.4, 0.5, 0.6]), int_arr)
        mock_input_file.reader.return_value.__enter__.return_value.get_spectrum_mz.assert_called_once_with(0)
        method_get_spectra_sum.assert_called_once_with(
            input_file=mock_input_file,
            parallel_config=self.mock_parallel_config,
            eval_bins=None,
        )

    @patch.object(EvaluateMeanSpectrum, "_get_spectra_sum")
    def test_evaluate_file_when_eval_bins(self, method_get_spectra_sum) -> None:
        mock_input_file = MagicMock(name="mock_input_file", imzml_mode=ImzmlModeEnum.CONTINUOUS, n_spectra=10)
        mock_mz_arr = MagicMock(name="mock_mz_arr", spec=[])
        self.mock_eval_bins.mz_values = mock_mz_arr
        method_get_spectra_sum.return_value = np.array([4.0, 5, 6])

        mz_arr, int_arr = self.mock_evaluate.evaluate_file(mock_input_file)

        self.assertEqual(mock_mz_arr, mz_arr)
        np.testing.assert_array_equal(np.array([0.4, 0.5, 0.6]), int_arr)
        mock_input_file.reader.return_value.__enter__.return_value.get_spectrum_mz.assert_not_called()
        method_get_spectra_sum.assert_called_once_with(
            input_file=mock_input_file,
            parallel_config=self.mock_parallel_config,
            eval_bins=self.mock_eval_bins,
        )

    @patch.object(ReadSpectraParallel, "from_config")
    def test_get_spectra_sum(self, mock_from_config) -> None:
        mock_parallelize = MagicMock(name="mock_parallelize", spec=["map_chunked"])
        mock_from_config.return_value = mock_parallelize
        mock_input_file = MagicMock(name="mock_input_file", spec=[""])
        mock_parallelize.map_chunked.return_value = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9]),
        ]

        total_sum = EvaluateMeanSpectrum._get_spectra_sum(
            mock_input_file,
            parallel_config=self.mock_parallel_config,
            eval_bins=self.mock_eval_bins,
        )
        np.testing.assert_array_equal(np.array([12, 15, 18]), total_sum)
        mock_from_config.assert_called_once_with(self.mock_parallel_config)

    def test_compute_chunk_sum_when_not_bin(self) -> None:
        mock_reader = MagicMock(name="mock_reader")
        spectra_ids = [5, 7, 10, 14]
        mock_reader.get_spectrum_int.side_effect = lambda x: np.array([x, x, 2.0 * x])
        chunk_sum = self.mock_evaluate._compute_chunk_sum(mock_reader, spectra_ids, eval_bins=None)
        np.testing.assert_array_equal(np.array([36.0, 36.0, 72.0]), chunk_sum)

    def test_compute_chunk_sum_when_evaluate_bins(self) -> None:
        mock_reader = MagicMock(name="mock_reader")
        spectra_ids = [5, 7, 10, 14]
        mock_reader.get_spectrum.side_effect = lambda x: (
            np.array([x, x, 2.0 * x]),
            np.array([1.0, 2.0, 1.0]),
        )
        mock_eval_bins = MagicMock(name="mock_eval_bins", spec=["evaluate"])
        mock_eval_bins.evaluate.side_effect = lambda x, y: y[:2]
        chunk_sum = self.mock_evaluate._compute_chunk_sum(mock_reader, spectra_ids, eval_bins=mock_eval_bins)
        np.testing.assert_array_equal(np.array([4.0, 8.0]), chunk_sum)


if __name__ == "__main__":
    unittest.main()
