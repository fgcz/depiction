import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from depiction.calibration.deprecated.calibrate_image import CalibrateImage
from tests.integration.integration_test_utils import IntegrationTestUtils
from depiction.parallel_ops.parallel_config import ParallelConfig
from depiction.persistence import ImzmlModeEnum, ImzmlReadFile, ImzmlWriteFile


class TestCalibrateImageIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.mock_input_file_path = os.path.join(self.tmp_dir.name, "input.imzML")
        self.mock_output_file_path = os.path.join(self.tmp_dir.name, "output.imzML")

    def test_calibrate_image(self):
        mz_list = np.linspace(100, 110, 23)
        mz_list_outlier = mz_list.copy()
        mz_list_outlier[10] += 0.3
        mz_ref = mz_list[2::5]

        mz_arr_list = np.array([mz_list, mz_list + 0.1, mz_list - 0.2, mz_list_outlier])
        int_arr_list = np.full_like(mz_arr_list, 1.0)
        int_arr_list[..., 2::5] = 10.0

        IntegrationTestUtils.populate_test_file(
            path=self.mock_input_file_path,
            mz_arr_list=mz_arr_list,
            int_arr_list=int_arr_list,
            imzml_mode=ImzmlModeEnum.PROCESSED,
        )

        read_file = ImzmlReadFile(self.mock_input_file_path)
        write_file = ImzmlWriteFile(self.mock_output_file_path, imzml_mode=ImzmlModeEnum.PROCESSED)

        calibrate = CalibrateImage(
            reference_mz=mz_ref,
            model_type="linear",
            parallel_config=ParallelConfig(n_jobs=2, task_size=None),
            output_store=None,
        )
        calibrate.calibrate_image(read_file=read_file, write_file=write_file)

        # read the results
        with ImzmlReadFile(self.mock_output_file_path).reader() as reader:
            spectra = reader.get_spectra([0, 1, 2, 3])
            self.assertEqual(ImzmlModeEnum.PROCESSED, reader.imzml_mode)

        for i_spectrum in range(4):
            mz_arr = spectra[0][i_spectrum]
            int_arr = spectra[1][i_spectrum]

            mz_peak = mz_arr[int_arr == 10.0]
            # TODO ideally this would be as close to zero as possible:
            print(mz_peak - mz_ref)
            # but in practice it currently is not.
            # it needs to be checked why this is the case...

        print()
        # check the delta from the input
        print(spectra[0])
        print()
        print(spectra[1])
        delta = mz_arr_list - spectra[0]
        print(delta)
        self.assertFalse(True)


if __name__ == "__main__":
    unittest.main()
