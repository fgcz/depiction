import unittest
from functools import cached_property
from unittest.mock import patch

from ionmapper.parallel_ops.parallel_config import ParallelConfig


class TestParallelConfig(unittest.TestCase):
    def setUp(self):
        self.mock_n_jobs = 4
        self.mock_task_size = None
        self.mock_verbose = 5

    @cached_property
    def mock_parallel_config(self):
        return ParallelConfig(
            n_jobs=self.mock_n_jobs,
            task_size=self.mock_task_size,
            verbose=self.mock_verbose,
        )

    def test_n_jobs(self):
        self.assertEqual(self.mock_n_jobs, self.mock_parallel_config.n_jobs)

    def test_task_size(self):
        self.assertEqual(self.mock_task_size, self.mock_parallel_config.task_size)

    def test_verbose(self):
        self.assertEqual(self.mock_verbose, self.mock_parallel_config.verbose)

    def test_get_splits_count_when_task_size(self):
        self.mock_task_size = 10
        self.assertEqual(5, self.mock_parallel_config.get_splits_count(50))
        self.assertEqual(1, self.mock_parallel_config.get_splits_count(5))

    def test_get_splits_count_when_no_task_size(self):
        self.mock_task_size = None
        self.mock_n_jobs = 4
        self.assertEqual(4, self.mock_parallel_config.get_splits_count(50))
        self.assertEqual(4, self.mock_parallel_config.get_splits_count(5))
        self.assertEqual(2, self.mock_parallel_config.get_splits_count(2))

    @patch.object(ParallelConfig, "get_splits_count")
    def test_get_task_splits_when_full_range(self, method_get_splits_count):
        method_get_splits_count.return_value = 3
        splits = self.mock_parallel_config.get_task_splits(n_items=10, item_indices=None)
        self.assertListEqual([[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]], splits)
        method_get_splits_count.assert_called_once_with(10)

    @patch.object(ParallelConfig, "get_splits_count")
    def test_get_task_splits_when_custom_selection(self, method_get_splits_count):
        method_get_splits_count.return_value = 3
        splits = self.mock_parallel_config.get_task_splits(n_items=None, item_indices=[0, 10, 20, 30, 50])
        self.assertListEqual([[0, 10], [20, 30], [50]], splits)
        method_get_splits_count.assert_called_once_with(5)

    def test_get_task_splits_expect_failure_when_both_parameters(self):
        with self.assertRaises(ValueError):
            self.mock_parallel_config.get_task_splits(n_items=10, item_indices=[0, 10, 20, 30, 50])

    def test_no_parallelism(self):
        parallel_config = ParallelConfig.no_parallelism()
        self.assertEqual(1, parallel_config.n_jobs)
        self.assertIsNone(parallel_config.task_size)
        self.assertEqual(1, parallel_config.verbose)


if __name__ == "__main__":
    unittest.main()
