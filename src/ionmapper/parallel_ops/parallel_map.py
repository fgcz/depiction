from __future__ import annotations

import functools
import operator

import joblib

from ionmapper.parallel_ops import ParallelConfig
from typing import TypeVar, TYPE_CHECKING, Callable, Any, Optional

if TYPE_CHECKING:
    S = TypeVar("S")
    T = TypeVar("T")
    U = TypeVar("U")


class ParallelMap:
    def __init__(self, config: ParallelConfig):
        self._config = config

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
        bind_kwargs: Optional[dict[str, Any]] = None,
        reduce_fn: Optional[Callable[[list[T]], U]] = None,
    ) -> list[T] | U:
        reduce_fn = reduce_fn if reduce_fn is not None else list
        joblib_parallel = joblib.Parallel(n_jobs=self.config.n_jobs, verbose=self.config.verbose)
        operation = self._bind(operation=operation, bind_kwargs=bind_kwargs)
        return reduce_fn(joblib_parallel(joblib.delayed(operation)(task) for task in tasks))

    def _bind(
        self,
        operation: Callable,
        bind_kwargs: Optional[dict[str, Any]],
    ) -> Callable:
        if bind_kwargs:
            return functools.partial(operation, **bind_kwargs)
        else:
            return operation

    def __repr__(self):
        return f"{self.__class__.__name__}(config={repr(self.config)})"
