import numpy as np
from xarray import DataArray


# TODO consider removing grid/flat terminology and only use dense/sparse
# TODO in many cases the logic can be abstracted, by simply allowing to pass a `sparse` object


class SparseRepresentation:
    """Utility to convert to and from sparse representation. Even when memory usage is not a concern, some operations
    can be simpler or require the sparse format, so this utility can be useful.

    The basic idea is that an image is either stored in a dense/grid format or a sparse/flat format.
    - Dense format: (y, x, c) for a 2D image with c channels.
    - Sparse format: (i, c) for a 2D image with c channels, plus (i, 2) for the coordinates.

    At the moment the functionality does not support z-axis, however it's built in a way that this functionality could
    easily be added on top by using DataArray which annotates the dimensions.
    """

    @classmethod
    def sparse_to_dense(cls, sparse_values: DataArray, coordinates: DataArray, bg_value: float) -> DataArray:
        """Converts the sparse image representation into a dense image representation.
        :param sparse_values: DataArray with "i" (index of value) and "c" (channel) dimensions
        :param coordinates: DataArray with "i" (index of value) and "d" (dimension) dimensions
        :param bg_value: the value to use for the background
        :return: DataArray with "y", "x", and "c" dimensions
        """
        n_channels = sparse_values.sizes["c"]
        sparse_values = sparse_values.transpose("i", "c").values
        coordinates = coordinates.transpose("i", "d").values

        coordinates_extent = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
        coordinates_shifted = coordinates - coordinates.min(axis=0)

        dtype = np.promote_types(sparse_values.dtype, np.obj2sctype(type(bg_value)))
        values_grid = np.full(
            (coordinates_extent[0], coordinates_extent[1], n_channels), fill_value=bg_value, dtype=dtype
        )
        for i_channel in range(n_channels):
            values_grid[tuple(coordinates_shifted.T) + (i_channel,)] = sparse_values[:, i_channel]

        return DataArray(values_grid, dims=("y", "x", "c"))

    @classmethod
    def dense_to_sparse(cls, grid_values: DataArray, bg_value: float | None) -> tuple[DataArray, DataArray]:
        """Converts the dense image representation into a sparse image representation.
        :param grid_values: DataArray with "y", "x", and "c" dimensions
        :param bg_value: the value to use for the background, or None to use all values
            (i.e. the result won't be actually sparse)
        :return: a tuple with two DataArrays, the first with "i" (index of value) and "c" (channel) dimensions,
            and the second with "i" (index of value) and "d" (dimension) dimensions
        """
        grid_values = grid_values.transpose("x", "y", "c").values
        if bg_value is None:
            # use all values
            sparse_values = grid_values.reshape(-1, grid_values.shape[-1])
            coordinates = np.array(list(np.ndindex(grid_values.shape[:-1])))
        else:
            # select the background
            if np.isnan(bg_value):
                mask_coords = np.where(~np.all(np.isnan(grid_values), axis=-1))
            else:
                mask_coords = np.where(~np.all(grid_values == bg_value, axis=-1))
            sparse_values = grid_values[mask_coords]
            coordinates = np.stack(mask_coords, axis=1)

        return DataArray(sparse_values, dims=("i", "c")), DataArray(
            coordinates, dims=("i", "d"), coords={"d": ["x", "y"]}
        )

    @classmethod
    def dense_to_sparse_coords(cls, grid_values: DataArray, coords: DataArray, is_shift_subtracted: bool) -> DataArray:
        if not is_shift_subtracted:
            coords = coords - coords.min(dim="i")
        grid_values = grid_values.transpose("y", "x", "c").values
        coords = coords.transpose("i", "d").values
        return DataArray(grid_values[tuple(coords.T)], dims=["i", "c"])
