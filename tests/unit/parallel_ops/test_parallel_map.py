import unittest
from functools import cached_property
from unittest.mock import MagicMock

from ionplotter.parallel_ops import ParallelConfig
from ionplotter.parallel_ops.parallel_map import ParallelMap


class TestParallelMap(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_config = MagicMock(name="mock_config", spec=ParallelConfig)

    @cached_property
    def mock_parallel(self) -> ParallelMap:
        return ParallelMap(config=self.mock_config)

    def test_config(self) -> None:
        self.assertEqual(self.mock_config, self.mock_parallel.config)

    def test_reduce_concat(self) -> None:
        results = [[1, 2, 3], [4], [5, 6]]
        reduced = self.mock_parallel.reduce_concat(results)
        self.assertListEqual([1, 2, 3, 4, 5, 6], reduced)

    def test_call_when_default(self) -> None:
        def mock_operation(x):
            return x * 2

        tasks = [1, 2, 3, 4, 5]
        self.mock_config = ParallelConfig(n_jobs=3, verbose=0, task_size=None)
        result = self.mock_parallel(operation=mock_operation, tasks=tasks)

        self.assertListEqual([2, 4, 6, 8, 10], result)

    def test_call_when_bind_kwargs(self) -> None:
        def mock_operation(x, y):
            return x * y

        tasks = [1, 2, 3, 4, 5]
        self.mock_config = ParallelConfig(n_jobs=3, verbose=0, task_size=None)
        result = self.mock_parallel(operation=mock_operation, tasks=tasks, bind_kwargs={"y": 3})

        self.assertListEqual([3, 6, 9, 12, 15], result)

    def test_call_when_reduce_fn(self) -> None:
        def mock_operation(x_list):
            return [x * 2 for x in x_list]

        tasks = [[1, 2], [3, 4], [5]]
        self.mock_config = ParallelConfig(n_jobs=3, verbose=0, task_size=None)

        with self.subTest(reduce_fn="list"):
            result_list = self.mock_parallel(operation=mock_operation, tasks=tasks)
            self.assertListEqual([[2, 4], [6, 8], [10]], result_list)

        with self.subTest(reduce_fn="concat"):
            result_list = self.mock_parallel(operation=mock_operation, tasks=tasks, reduce_fn=ParallelMap.reduce_concat)
            self.assertListEqual([2, 4, 6, 8, 10], result_list)

    def test_repr(self) -> None:
        self.mock_config = "1234"
        self.assertEqual("ParallelMap(config='1234')", repr(self.mock_parallel))


if __name__ == "__main__":
    unittest.main()
