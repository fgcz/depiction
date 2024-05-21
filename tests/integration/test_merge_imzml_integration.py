import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from ionplotter.misc.integration_test_utils import IntegrationTestUtils
from ionplotter.persistence import ImzmlReadFile, ImzmlModeEnum, ImzmlWriteFile
from ionplotter.tools.merge_imzml import MergeImzml


class TestMergeImzmlIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.mock_input_file_1_path = os.path.join(self.tmp_dir.name, "input_1.imzML")
        self.mock_input_file_2_path = os.path.join(self.tmp_dir.name, "input_2.imzML")
        self.mock_output_file_path = os.path.join(self.tmp_dir.name, "output.imzML")

    def test_merge_continuous(self) -> None:
        IntegrationTestUtils.populate_test_file(
            path=self.mock_input_file_1_path,
            mz_arr_list=[[100, 200, 300], [100, 200, 300]],
            int_arr_list=[[1, 2, 3], [2, 2, 2]],
            imzml_mode=ImzmlModeEnum.CONTINUOUS,
            coordinates_list=[(0, 0), (0, 1)],
        )
        IntegrationTestUtils.populate_test_file(
            path=self.mock_input_file_2_path,
            mz_arr_list=[[100, 200, 300]],
            int_arr_list=[[4, 5, 6]],
            imzml_mode=ImzmlModeEnum.CONTINUOUS,
            coordinates_list=[(5, 1)],
        )
        file_1 = ImzmlReadFile(self.mock_input_file_1_path)
        file_2 = ImzmlReadFile(self.mock_input_file_2_path)
        output_file = ImzmlWriteFile(path=self.mock_output_file_path, imzml_mode=ImzmlModeEnum.CONTINUOUS)
        MergeImzml().merge(input_files=[file_1, file_2], output_file=output_file)

        with ImzmlReadFile(self.mock_output_file_path).reader() as reader:
            spectra = reader.get_spectra([0, 1, 2])
            coordinates = reader.coordinates
            self.assertEqual(ImzmlModeEnum.CONTINUOUS, reader.imzml_mode)

        np.testing.assert_array_equal(np.array([[100, 200, 300], [100, 200, 300], [100, 200, 300]]), spectra[0])
        np.testing.assert_array_equal(np.array([[1, 2, 3], [2, 2, 2], [4, 5, 6]]), spectra[1])
        np.testing.assert_array_equal(np.array([[0, 0, 1], [0, 1, 1], [5, 1, 1]]), coordinates)

    def test_merge_processed(self) -> None:
        IntegrationTestUtils.populate_test_file(
            path=self.mock_input_file_1_path,
            mz_arr_list=[[100, 200, 300], [200, 300]],
            int_arr_list=[[1, 2, 3], [2, 2]],
            imzml_mode=ImzmlModeEnum.PROCESSED,
            coordinates_list=[(0, 0), (0, 1)],
        )
        IntegrationTestUtils.populate_test_file(
            path=self.mock_input_file_2_path,
            mz_arr_list=[[100, 200, 300, 400]],
            int_arr_list=[[3, 3, 3, 3]],
            imzml_mode=ImzmlModeEnum.PROCESSED,
            coordinates_list=[(5, 1)],
        )
        file_1 = ImzmlReadFile(self.mock_input_file_1_path)
        file_2 = ImzmlReadFile(self.mock_input_file_2_path)
        output_file = ImzmlWriteFile(path=self.mock_output_file_path, imzml_mode=ImzmlModeEnum.PROCESSED)

        MergeImzml().merge(input_files=[file_1, file_2], output_file=output_file)

        with ImzmlReadFile(self.mock_output_file_path).reader() as reader:
            spectra = reader.get_spectra([0, 1, 2])
            coordinates = reader.coordinates
            self.assertEqual(ImzmlModeEnum.PROCESSED, reader.imzml_mode)

        self.assertEqual(3, len(spectra[0]))
        np.testing.assert_array_equal(np.array([100, 200, 300]), spectra[0][0])
        np.testing.assert_array_equal(np.array([200, 300]), spectra[0][1])
        np.testing.assert_array_equal(np.array([100, 200, 300, 400]), spectra[0][2])
        self.assertEqual(3, len(spectra[1]))
        np.testing.assert_array_equal(np.array([1, 2, 3]), spectra[1][0])
        np.testing.assert_array_equal(np.array([2, 2]), spectra[1][1])
        np.testing.assert_array_equal(np.array([3, 3, 3, 3]), spectra[1][2])

        np.testing.assert_array_equal(np.array([[0, 0, 1], [0, 1, 1], [5, 1, 1]]), coordinates)


if __name__ == "__main__":
    unittest.main()
