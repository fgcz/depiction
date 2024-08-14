from __future__ import annotations

import functools
import operator
from typing import TypeVar, TYPE_CHECKING, Callable, Any

import multiprocess.pool
from loguru import logger

if TYPE_CHECKING:
    from depiction.parallel_ops import ParallelConfig

    S = TypeVar("S")
    T = TypeVar("T")
    U = TypeVar("U")


class ParallelMap:
    def __init__(self, config: ParallelConfig) -> None:
        self._config = config

    @classmethod
    def from_config(cls, config: ParallelConfig) -> ParallelMap:
        return cls(config=config)

    @property
    def config(self) -> ParallelConfig:
        return self._config

    @staticmethod
    def reduce_concat(results: list[list[T]]) -> list[T]:
        return functools.reduce(operator.iconcat, results, [])

    # TODO zip_args could be a useful feature here (remove comment if not needed)

    def __call__(
        self,
        operation: Callable[[S, ...], T],
        tasks: list[S],
        bind_kwargs: dict[str, Any] | None = None,
        reduce_fn: Callable[[list[T]], U] | None = None,
    ) -> list[T] | U:
        reduce_fn = reduce_fn if reduce_fn is not None else list
        operation_bound = self._bind(operation=operation, bind_kwargs=bind_kwargs)

        def wrapped_operation(tasks: list[S]) -> list[T]:
            try:
                return [operation_bound(task) for task in tasks]
            except Exception as e:
                logger.exception(f"Error in parallel operation: {e}")
                raise

        with multiprocess.pool.Pool(self.config.n_jobs) as pool:
            return reduce_fn(pool.map(wrapped_operation, tasks))

    def _bind(
        self,
        operation: Callable,
        bind_kwargs: dict[str, Any] | None,
    ) -> Callable:
        if bind_kwargs:
            return functools.partial(operation, **bind_kwargs)
        else:
            return operation

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={repr(self.config)})"
