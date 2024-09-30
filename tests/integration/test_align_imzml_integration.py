import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from depiction.parallel_ops.parallel_config import ParallelConfig
from depiction.persistence import ImzmlModeEnum, ImzmlReadFile, ImzmlWriteFile
from depiction.tools.align_imzml import AlignImzml, AlignImzmlMethod


class TestAlignImzmlIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.mock_input_file_path = os.path.join(self.tmp_dir.name, "input.imzML")
        self.mock_output_file_path = os.path.join(self.tmp_dir.name, "output.imzML")
        self.mock_parallel_config = ParallelConfig(n_jobs=2)

    def mock_input_file(self, mz_arr_list: list[list[int]], imzml_mode: ImzmlModeEnum) -> ImzmlReadFile:
        write_file = ImzmlWriteFile(path=self.mock_input_file_path, imzml_mode=imzml_mode)
        with write_file.writer() as writer:
            for i, mz_arr in enumerate(mz_arr_list):
                writer.add_spectrum(mz_arr, np.repeat(0.5, len(mz_arr)), (i, 0))
        return ImzmlReadFile(self.mock_input_file_path)

    def test_already_aligned(self) -> None:
        input_file = self.mock_input_file(
            [[100, 200, 300], [100, 200, 300], [100, 200, 300]],
            imzml_mode=ImzmlModeEnum.CONTINUOUS,
        )
        print(str(input_file.get_reader()))
        align = AlignImzml(
            input_file=input_file,
            output_file_path=self.mock_output_file_path,
            method=AlignImzmlMethod.CARDINAL_ESTIMATE_PPM,
            parallel_config=self.mock_parallel_config,
        )
        aligned_file = align.evaluate()
        self.assertEqual(Path(self.mock_input_file_path), aligned_file.imzml_file)

    def test_perform_alignment(self) -> None:
        input_file = self.mock_input_file(
            [[100, 200, 300], [200, 250, 300, 310], [200, 250, 300], [200, 250, 300]],
            imzml_mode=ImzmlModeEnum.PROCESSED,
        )
        align = AlignImzml(
            input_file=input_file,
            output_file_path=self.mock_output_file_path,
            method=AlignImzmlMethod.CARDINAL_ESTIMATE_PPM,
            parallel_config=self.mock_parallel_config,
        )
        aligned_file = align.evaluate()
        self.assertEqual(Path(self.mock_output_file_path), aligned_file.imzml_file)

        with ImzmlReadFile(aligned_file.imzml_file).reader() as reader:
            self.assertEqual(4, reader.n_spectra)
            spectra = reader.get_spectra([0, 1, 2, 3])

        np.testing.assert_array_almost_equal(
            np.repeat(
                [
                    [
                        100.0,
                        128.38013568,
                        161.63443372,
                        202.37094882,
                        252.27317982,
                        310.0,
                    ]
                ],
                4,
                axis=0,
            ),
            spectra[0],
        )
        np.testing.assert_array_almost_equal(
            np.array(
                [
                    [0.5, 0.0, 0.0, 0.5, 0.0, 0.5],
                    [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
                    [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
                ]
            ),
            spectra[1],
        )


if __name__ == "__main__":
    unittest.main()
