# TODO this is not the ideal place, but to avoid code duplication it's better to have a place for now
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
