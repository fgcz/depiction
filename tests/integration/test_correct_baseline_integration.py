import os
import unittest
from functools import cached_property
from tempfile import TemporaryDirectory

import numpy as np

from ionplotter.evaluate_baseline_correction import (
    EvaluateMWMVBaselineCorrection,
)
from ionplotter.misc.integration_test_utils import IntegrationTestUtils
from ionplotter.parallel_ops.parallel_config import ParallelConfig
from ionplotter.persistence import ImzmlModeEnum, ImzmlReadFile, ImzmlWriteFile
from ionplotter.tools.correct_baseline import CorrectBaseline


# TODO: move to a common test utilities file (and use it also in the other integration tests)


class TestCorrectBaselineIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.mock_input_file_path = os.path.join(self.tmp_dir.name, "input.imzML")
        self.mock_output_file_path = os.path.join(self.tmp_dir.name, "output.imzML")

        self.mock_baseline_spectrum = [10, 11, 10, 10, 250, 10, 10, 10, 11, 200, 10, 10]
        self.mock_clean_spectrum = [0, 1, 0, 0, 240, 0, 0, 0, 1, 190, 0, 0]
        self.mock_mz_arr = [(i + 1) * 100 for i in range(len(self.mock_baseline_spectrum))]

    @cached_property
    def mock_input_file(self) -> ImzmlReadFile:
        IntegrationTestUtils.populate_test_file(
            path=self.mock_input_file_path,
            mz_arr_list=[self.mock_mz_arr, self.mock_mz_arr, self.mock_mz_arr],
            int_arr_list=[
                self.mock_baseline_spectrum,
                np.zeros_like(self.mock_baseline_spectrum),
                self.mock_baseline_spectrum,
            ],
            imzml_mode=ImzmlModeEnum.CONTINUOUS,
        )
        return ImzmlReadFile(self.mock_input_file_path)

    def test_evaluate_file(self) -> None:
        correct_baseline = CorrectBaseline(
            parallel_config=ParallelConfig(n_jobs=2, task_size=2),
            baseline_correction=EvaluateMWMVBaselineCorrection(),
        )
        correct_baseline.evaluate_file(
            read_file=self.mock_input_file,
            write_file=ImzmlWriteFile(self.mock_output_file_path, imzml_mode=ImzmlModeEnum.CONTINUOUS),
        )

        with ImzmlReadFile(self.mock_output_file_path).reader() as reader:
            self.assertEqual(3, reader.n_spectra)
            spectra = reader.get_spectra([0, 1, 2])

        np.testing.assert_array_equal(np.array([self.mock_mz_arr, self.mock_mz_arr, self.mock_mz_arr]), spectra[0])
        np.testing.assert_array_almost_equal(
            np.array(
                [
                    self.mock_clean_spectrum,
                    np.zeros_like(self.mock_baseline_spectrum),
                    self.mock_clean_spectrum,
                ]
            ),
            spectra[1],
        )


if __name__ == "__main__":
    unittest.main()
