import unittest
from functools import cached_property
from unittest.mock import patch, MagicMock

import numpy as np

from depiction.tools.subsample_imzml import SubsampleMode, SubsampleImzml


class TestSubsampleMode(unittest.TestCase):
    def test_randomized(self) -> None:
        samples = SubsampleMode.randomized.sample(np.arange(10), 5)
        self.assertEqual(5, len(samples))
        self.assertEqual(5, len(set(samples)))
        for sample in samples:
            self.assertTrue(0 <= sample < 10)

    def test_linspaced(self) -> None:
        np.testing.assert_array_equal(np.array([0, 3, 6, 9]), SubsampleMode.linspaced.sample(np.arange(10), 4))
        np.testing.assert_array_equal(np.array([0, 2, 4, 7, 9]), SubsampleMode.linspaced.sample(np.arange(10), 5))
        np.testing.assert_array_equal(np.array([0, 2, 4, 5, 7, 9]), SubsampleMode.linspaced.sample(np.arange(10), 6))


# TODO consider renaming this test class
class TestSubsampleImzml(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_ratio = 0.5
        self.mock_mode = SubsampleMode.randomized
        self.mock_seed = 0

    @cached_property
    def target(self) -> SubsampleImzml:
        return SubsampleImzml(
            ratio=self.mock_ratio,
            mode=self.mock_mode,
            seed=self.mock_seed,
        )

    @patch.object(SubsampleImzml, "determine_spectra_to_keep")
    @patch("builtins.open")
    @patch("json.dump")
    def test_dump_subsample_info(self, mock_json_dump, mock_open, method_determine_spectra_to_keep) -> None:
        mock_read_file = MagicMock(name="mock_read_file", imzml_file="dummy_input.imzML")
        output_path = "dummy_output.imzML"
        method_determine_spectra_to_keep.return_value = np.array([1, 50, 100])

        self.target.dump_subsample_info(read_file=mock_read_file, output_imzml=output_path)

        mock_open.assert_called_once_with("dummy_output.subsample_info.json", "w")
        mock_json_dump.assert_called_once_with(
            {
                "input_imzml": "dummy_input.imzML",
                "output_imzml": "dummy_output.imzML",
                "spectra_to_keep": [1, 50, 100],
                "ratio": 0.5,
                "mode": "randomized",
                "seed": 0,
            },
            mock_open.return_value.__enter__.return_value,
            indent=1,
        )
        method_determine_spectra_to_keep.assert_called_once_with(read_file=mock_read_file)


if __name__ == "__main__":
    unittest.main()
