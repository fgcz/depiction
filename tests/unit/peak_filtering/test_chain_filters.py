import unittest
from unittest.mock import MagicMock

from ionplotter.peak_filtering import ChainFilters


class TestChainFilters(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_filter_1 = MagicMock(name="mock_filter_1")
        self.mock_filter_2 = MagicMock(name="mock_filter_2")
        self.chain_filters = ChainFilters(
            filters=[
                self.mock_filter_1,
                self.mock_filter_2,
            ]
        )

    def test_filter_index_peaks(self) -> None:
        self.mock_filter_1.filter_index_peaks.return_value = [1, 2, 3]
        self.mock_filter_2.filter_index_peaks.return_value = [2, 3]
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        mock_spectrum_int_arr = MagicMock(name="mock_spectrum_int_arr")
        peak_idx_arr = [1, 2, 3, 4]

        result_peak_idx_arr = self.chain_filters.filter_index_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=mock_spectrum_int_arr,
            peak_idx_arr=peak_idx_arr,
        )

        self.assertEqual(result_peak_idx_arr, [2, 3])
        self.mock_filter_1.filter_index_peaks.assert_called_once_with(
            mock_spectrum_mz_arr,
            mock_spectrum_int_arr,
            peak_idx_arr,
        )
        self.mock_filter_2.filter_index_peaks.assert_called_once_with(
            mock_spectrum_mz_arr,
            mock_spectrum_int_arr,
            [1, 2, 3],
        )

    def test_filter_peaks(self) -> None:
        self.mock_filter_1.filter_peaks.return_value = ([1, 2, 3], [4, 5, 6])
        self.mock_filter_2.filter_peaks.return_value = ([2, 3], [5, 6])
        mock_spectrum_mz_arr = MagicMock(name="mock_spectrum_mz_arr")
        mock_spectrum_int_arr = MagicMock(name="mock_spectrum_int_arr")
        peak_mz_arr = [1, 2, 3, 4]
        peak_int_arr = [4, 5, 6, 7]

        result_peak_mz_arr, result_peak_int_arr = self.chain_filters.filter_peaks(
            spectrum_mz_arr=mock_spectrum_mz_arr,
            spectrum_int_arr=mock_spectrum_int_arr,
            peak_mz_arr=peak_mz_arr,
            peak_int_arr=peak_int_arr,
        )

        self.assertEqual(result_peak_mz_arr, [2, 3])
        self.assertEqual(result_peak_int_arr, [5, 6])
        self.mock_filter_1.filter_peaks.assert_called_once_with(
            mock_spectrum_mz_arr,
            mock_spectrum_int_arr,
            peak_mz_arr,
            peak_int_arr,
        )
        self.mock_filter_2.filter_peaks.assert_called_once_with(
            mock_spectrum_mz_arr,
            mock_spectrum_int_arr,
            [1, 2, 3],
            [4, 5, 6],
        )


if __name__ == "__main__":
    unittest.main()
