from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class ParallelConfig:
    n_jobs: int
    task_size: int | None
    verbose: int = 1

    def get_splits_count(self, n_items: int) -> int:
        """
        Returns the number of necessary splits to process the given number of items.
        """
        if self.task_size:
            return max(n_items // self.task_size, 1)
        else:
            return min(self.n_jobs, n_items)

    def get_task_splits(self, n_items: int | None = None, item_indices: np.ndarray | None = None) -> list[list[int]]:
        """
        Returns a list of the indices of the spectra to process in each task.
        Either n_items or item_indices must be provided, but not both.
        :param n_items: the total number of items
        :param item_indices: restrict the operation to the given item's indices, if None, all items are processed
        """
        if (n_items is None) == (item_indices is None):
            raise ValueError("Either n_items or item_indices must be provided, but not both")

        if item_indices is None:
            item_indices = np.arange(n_items)

        return [list(s) for s in np.array_split(item_indices, self.get_splits_count(len(item_indices)))]

    @classmethod
    def no_parallelism(cls) -> ParallelConfig:
        """Returns a ParallelConfig object with no parallelism."""
        # Note: Maybe uses of this method could also hint at design issues in the code.
        return cls(n_jobs=1, task_size=None, verbose=1)
