import unittest
from functools import cached_property
from unittest.mock import MagicMock

import numpy as np

from ionplotter.parallel_ops import ReadSpectraParallel
from ionplotter.persistence import RamReadFile


class TestReadSpectraParallel(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_n_jobs = 2
        self.mock_task_size = None
        self.mock_verbose = 0

        self.mock_mz_list = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.mock_int_list = np.array([[500.0, 600.0, 700.0], [800.0, 900.0, 1000.0]])
        self.mock_coordinates = np.array([[0, 1, 2], [3, 4, 5]])

    @cached_property
    def mock_parallel(self) -> ReadSpectraParallel:
        return ReadSpectraParallel.from_params(
            n_jobs=self.mock_n_jobs,
            task_size=self.mock_task_size,
            verbose=self.mock_verbose,
        )

    @cached_property
    def mock_read_file(self) -> RamReadFile:
        return RamReadFile(
            mz_arr_list=self.mock_mz_list,
            int_arr_list=self.mock_int_list,
            coordinates=self.mock_coordinates,
        )

    def test_from_config(self) -> None:
        mock_config = MagicMock(name="mock_config")
        parallel = ReadSpectraParallel.from_config(config=mock_config)
        self.assertEqual(mock_config, parallel.config)

    def test_from_params(self) -> None:
        parallel = ReadSpectraParallel.from_params(
            n_jobs=self.mock_n_jobs,
            task_size=self.mock_task_size,
            verbose=self.mock_verbose,
        )
        self.assertEqual(self.mock_n_jobs, parallel.config.n_jobs)
        self.assertEqual(self.mock_task_size, parallel.config.task_size)
        self.assertEqual(self.mock_verbose, parallel.config.verbose)

    def test_map_chunked_default(self) -> None:
        def operation(reader, indices):
            return sum(reader.get_spectrum_mz(index).sum() for index in indices)

        results = self.mock_parallel.map_chunked(read_file=self.mock_read_file, operation=operation)
        self.assertListEqual([6.0, 15.0], results)

    def test_map_chunked_when_pass_task_index(self) -> None:
        def operation(reader, indices, task_index):
            return sum(reader.get_spectrum_mz(index).sum() for index in indices) + task_index

        results = self.mock_parallel.map_chunked(
            read_file=self.mock_read_file, operation=operation, pass_task_index=True
        )
        self.assertListEqual([6.0, 16.0], results)

    def test_map_chunked_when_spectra_indices(self) -> None:
        self.mock_mz_list = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.mock_int_list = np.array([[500.0, 600.0, 700.0], [800.0, 900.0, 1000.0], [1100.0, 1200.0, 1300.0]])
        self.mock_coordinates = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        def operation(reader, indices):
            return sum(reader.get_spectrum_mz(index).sum() for index in indices)

        results = self.mock_parallel.map_chunked(
            read_file=self.mock_read_file,
            operation=operation,
            spectra_indices=np.array([0, 2]),
        )
        self.assertListEqual([6.0, 24.0], results)

    def test_map_chunked_when_bind_args(self) -> None:
        def operation(reader, indices, *, my_arg):
            return sum(reader.get_spectrum_mz(index).sum() for index in indices) + my_arg

        results = self.mock_parallel.map_chunked(
            read_file=self.mock_read_file, operation=operation, bind_args={"my_arg": 10}
        )
        self.assertListEqual([16.0, 25.0], results)

    def test_map_chunked_when_reduce_concat(self) -> None:
        def operation(reader, indices):
            return reader.get_spectrum_mz(indices[0])

        results = self.mock_parallel.map_chunked(
            read_file=self.mock_read_file,
            operation=operation,
            reduce_fn=ReadSpectraParallel.reduce_concat,
        )

        self.assertListEqual([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], results)

    def test_reduce_concat(self) -> None:
        results = [[1, 2, 3], [4], [5, 6]]
        reduced = self.mock_parallel.reduce_concat(results)
        self.assertListEqual([1, 2, 3, 4, 5, 6], reduced)


if __name__ == "__main__":
    unittest.main()
