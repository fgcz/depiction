from __future__ import annotations

from typing import (
    Any,
    Callable,
    TypeVar,
    TYPE_CHECKING,
    TypedDict,
)

import numpy as np

from depiction.parallel_ops.parallel_config import ParallelConfig
from depiction.parallel_ops.parallel_map import ParallelMap

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from depiction.persistence import ImzmlReadFile, ImzmlReader

    T = TypeVar("T")
    V = TypeVar("V")


class ReadSpectraParallel:
    """
    Enables parallelized processing of spectra in a given input file.
    """

    def __init__(self, config: ParallelConfig) -> None:
        self._config = config

    @classmethod
    def from_config(cls, config: ParallelConfig) -> ReadSpectraParallel:
        return cls(config=config)

    @classmethod
    def from_params(cls, n_jobs: int, task_size: int | None, verbose: int = 1) -> ReadSpectraParallel:
        """In general, try to use from_config and pass the configuration throughout the application as appropriate."""
        return cls(config=ParallelConfig(n_jobs=n_jobs, task_size=task_size, verbose=verbose))

    @property
    def config(self) -> ParallelConfig:
        return self._config

    # TODO
    reduce_concat = staticmethod(ParallelMap.reduce_concat)

    def map_chunked(
        self,
        read_file: ImzmlReadFile,
        operation: Callable[[ImzmlReader, list[int], ...], T] | Callable[[ImzmlReader, list[int], int, ...], T],
        spectra_indices: NDArray[int] | None = None,
        bind_args: dict[str, Any] | None = None,
        reduce_fn: Callable[[list[T]], V] = list,
        pass_task_index: bool = False,
    ) -> V:
        """Applies a function to chunks of spectra in the given file in parallel.
        :param read_file: the file to read the spectra from
        :param operation: the operation to apply to each chunk of spectra
            there are two possible signatures for the operation:
            - operation(reader: ImzmlReader, spectra_ids: list[int], **kwargs) -> T
            - operation(reader: ImzmlReader, spectra_ids: list[int], task_index: int, **kwargs) -> T
            where:
            - reader: the reader object to read the spectra from
            - spectra_ids: the indices of the spectra to process
            which one to use depends on the value of pass_task_index
        :param spectra_indices: the indices of the spectra to process, if None, all spectra are processed
        :param bind_args: additional keyword arguments to bind to the operation
        :param reduce_fn: the function to use to combine the results of each chunk, usually one of these is used:
            - list: simply returns a list with one element per chunk
            - self.reduce_concat: concatenates the results of each chunk into a single list
        :param pass_task_index: whether to pass the task index to the operation
        """
        parallel_map = ParallelMap(config=self._config)
        spectra_indices = spectra_indices if spectra_indices is not None else np.arange(read_file.n_spectra)
        task_splits = [
            (task, task_index) if pass_task_index else (task,)
            for task_index, task in enumerate(self._config.get_task_splits(item_indices=spectra_indices))
        ]

        def execute_task(args: list[Any], **kwargs: TypedDict[str, Any]) -> list[T]:
            with read_file.reader() as reader:
                return operation(reader, *args, **kwargs)

        return parallel_map(
            operation=execute_task,
            tasks=task_splits,
            bind_kwargs=bind_args,
            reduce_fn=reduce_fn,
        )
