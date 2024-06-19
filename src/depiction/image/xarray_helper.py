# TODO this is not the ideal place, but to avoid code duplication it's better to have a place for now
from __future__ import annotations

from typing import Callable

import numpy as np
import xarray
from xarray import DataArray


class XarrayHelper:

    @staticmethod
    def is_sparse(values: DataArray) -> bool:
        """Returns whether the DataArray is sparse."""
        return hasattr(values.data, "todense")

    @classmethod
    def ensure_dense(cls, values: DataArray, copy: bool = False) -> DataArray:
        """Returns a dense version of the DataArray, if it is sparse, and returns the original otherwise."""
        # TODO migrate to a specific path
        if cls.is_sparse(values):
            dense_data = values.data.todense()
            return DataArray(dense_data, dims=values.dims, coords=values.coords, attrs=values.attrs, name=values.name)
        elif copy:
            return values.copy()
        else:
            return values

    @staticmethod
    def apply_on_spatial_view(array: DataArray, fn: Callable[[DataArray], DataArray]) -> DataArray:
        if set(array.dims) == {"y", "x", "c"}:
            array = array.transpose("y", "x", "c")
            result = fn(array)
            return result.transpose("y", "x", "c")
        elif set(array.dims) == {"i", "c"}:
            # determine if i was indexing ["y", "x"] or ["x", "y"]
            index_order = XarrayHelper.get_index_order(array)

            # reset index
            original_coords = array.coords["i"]
            array = array.reset_index("i")

            # get 2d view
            array_flat = array.set_xindex(index_order)
            array_2d = array_flat.unstack("i").transpose("y", "x", "c")

            # call the function
            result = fn(array_2d)

            # keep only the elements that were present before
            array_present_before = (
                array_flat.notnull()
                .reduce(np.all, "c", keepdims=True)
                .unstack("i")
                .transpose("y", "x", "c")
                .astype(float)
                .where(lambda x: x)
            )
            if "c" in array_flat.coords:
                # TODO test case distinction
                array_present_before = array_present_before.assign_coords(c=["nan_before"])
            result = xarray.concat([result, array_present_before], dim="c")

            # make flat again
            result_flat = result.stack(i=index_order)
            # remove nan
            result = result_flat.drop_isel(c=-1).dropna("i", how="all")
            # TODO assigning the coords will be broken in the future, when "i" is a multi-index, however since in general
            #      it is not, this will require a case distinciton
            return result.assign_coords(i=original_coords)
        else:
            raise ValueError(f"Unsupported dims={set(array.dims)}")

    @staticmethod
    def get_index_order(array: DataArray) -> tuple[str, str]:
        index_order = tuple(array.coords["i"].coords)
        assert index_order in (("i", "y", "x"), ("i", "x", "y")), f"Unexpected index_order={index_order}"
        return index_order[1:]
