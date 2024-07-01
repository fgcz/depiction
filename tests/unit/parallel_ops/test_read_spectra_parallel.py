from unittest.mock import MagicMock

import numpy as np
import pytest

from depiction.parallel_ops import ReadSpectraParallel
from depiction.persistence import RamReadFile

mock_n_jobs = 2
mock_task_size = None
mock_verbose = 0
mock_mz_list = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
mock_int_list = np.array([[500.0, 600.0, 700.0], [800.0, 900.0, 1000.0]])
mock_coordinates = np.array([[0, 1, 2], [3, 4, 5]])


@pytest.fixture
def mock_parallel() -> ReadSpectraParallel:
    return ReadSpectraParallel.from_params(
        n_jobs=mock_n_jobs,
        task_size=mock_task_size,
        verbose=mock_verbose,
    )


@pytest.fixture
def mock_read_file() -> RamReadFile:
    return RamReadFile(
        mz_arr_list=mock_mz_list,
        int_arr_list=mock_int_list,
        coordinates=mock_coordinates,
    )


def test_from_config() -> None:
    mock_config = MagicMock(name="mock_config")
    parallel = ReadSpectraParallel.from_config(config=mock_config)
    assert parallel.config == mock_config


def test_from_params() -> None:
    parallel = ReadSpectraParallel.from_params(
        n_jobs=mock_n_jobs,
        task_size=mock_task_size,
        verbose=mock_verbose,
    )
    assert parallel.config.n_jobs == mock_n_jobs
    assert parallel.config.task_size == mock_task_size
    assert parallel.config.verbose == mock_verbose


def test_map_chunked_default(mock_parallel, mock_read_file) -> None:
    def operation(reader, indices):
        return sum(reader.get_spectrum_mz(index).sum() for index in indices)

    results = mock_parallel.map_chunked(read_file=mock_read_file, operation=operation)
    assert results == [6.0, 15.0]


def test_map_chunked_when_pass_task_index(mock_parallel, mock_read_file) -> None:
    def operation(reader, indices, task_index):
        return sum(reader.get_spectrum_mz(index).sum() for index in indices) + task_index

    results = mock_parallel.map_chunked(read_file=mock_read_file, operation=operation, pass_task_index=True)
    assert results == [6.0, 16.0]


def test_map_chunked_when_spectra_indices(mock_parallel) -> None:
    mock_mz_list = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    mock_int_list = np.array([[500.0, 600.0, 700.0], [800.0, 900.0, 1000.0], [1100.0, 1200.0, 1300.0]])
    mock_coordinates = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    mock_read_file = RamReadFile(
        mz_arr_list=mock_mz_list,
        int_arr_list=mock_int_list,
        coordinates=mock_coordinates,
    )

    def operation(reader, indices):
        return sum(reader.get_spectrum_mz(index).sum() for index in indices)

    results = mock_parallel.map_chunked(
        read_file=mock_read_file,
        operation=operation,
        spectra_indices=np.array([0, 2]),
    )
    assert results == [6.0, 24.0]


def test_map_chunked_when_bind_args(mock_parallel, mock_read_file) -> None:
    def operation(reader, indices, *, my_arg):
        return sum(reader.get_spectrum_mz(index).sum() for index in indices) + my_arg

    results = mock_parallel.map_chunked(read_file=mock_read_file, operation=operation, bind_args={"my_arg": 10})
    assert results == [16.0, 25.0]


def test_map_chunked_when_reduce_concat(mock_parallel, mock_read_file) -> None:
    def operation(reader, indices):
        return reader.get_spectrum_mz(indices[0])

    results = mock_parallel.map_chunked(
        read_file=mock_read_file,
        operation=operation,
        reduce_fn=ReadSpectraParallel.reduce_concat,
    )

    assert results == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_reduce_concat() -> None:
    results = [[1, 2, 3], [4], [5, 6]]
    reduced = ReadSpectraParallel.reduce_concat(results)
    assert reduced == [1, 2, 3, 4, 5, 6]


if __name__ == "__main__":
    pytest.main()
