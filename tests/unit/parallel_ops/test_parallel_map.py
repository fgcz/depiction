from unittest.mock import MagicMock

import pytest

from depiction.parallel_ops import ParallelConfig
from depiction.parallel_ops.parallel_map import ParallelMap


@pytest.fixture
def mock_config() -> ParallelConfig:
    return MagicMock(name="mock_config", spec=ParallelConfig, n_jobs=2)


@pytest.fixture
def mock_parallel(mock_config) -> ParallelMap:
    return ParallelMap(config=mock_config)


def test_config(mock_parallel, mock_config):
    assert mock_parallel.config == mock_config


def test_reduce_concat(mock_parallel):
    results = [[1, 2, 3], [4], [5, 6]]
    reduced = mock_parallel.reduce_concat(results)
    assert reduced == [1, 2, 3, 4, 5, 6]


def test_call_when_default():
    def mock_operation(x):
        return x * 2

    tasks = [1, 2, 3, 4, 5]
    mock_config = ParallelConfig(n_jobs=3, verbose=0, task_size=None)
    mock_parallel = ParallelMap(config=mock_config)
    result = mock_parallel(operation=mock_operation, tasks=tasks)

    assert result == [2, 4, 6, 8, 10]


def test_call_when_bind_kwargs():
    def mock_operation(x, y):
        return x * y

    tasks = [1, 2, 3, 4, 5]
    mock_config = ParallelConfig(n_jobs=3, verbose=0, task_size=None)
    mock_parallel = ParallelMap(config=mock_config)
    result = mock_parallel(operation=mock_operation, tasks=tasks, bind_kwargs={"y": 3})

    assert result == [3, 6, 9, 12, 15]


@pytest.mark.parametrize(
    "reduce_fn, expected_result", [(None, [[2, 4], [6, 8], [10]]), (ParallelMap.reduce_concat, [2, 4, 6, 8, 10])]
)
def test_call_when_reduce_fn(reduce_fn, expected_result):
    def mock_operation(x_list):
        return [x * 2 for x in x_list]

    tasks = [[1, 2], [3, 4], [5]]
    mock_config = ParallelConfig(n_jobs=3, verbose=0, task_size=None)
    mock_parallel = ParallelMap(config=mock_config)

    result = mock_parallel(operation=mock_operation, tasks=tasks, reduce_fn=reduce_fn)
    assert result == expected_result


def test_repr():
    mock_config = "1234"
    mock_parallel = ParallelMap(config=mock_config)
    assert repr(mock_parallel) == "ParallelMap(config='1234')"


if __name__ == "__main__":
    pytest.main()
