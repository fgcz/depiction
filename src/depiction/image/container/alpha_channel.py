from __future__ import annotations

import xarray


class AlphaChannel:
    """Implements logic to stack an alpha channel on top of an arbitrary channel image and split it off again.

    The alpha channel is expected to be a boolean mask, where True indicates foreground and False background.
    When stacked together it will take the numeric value of 1 for True and 0 for False.

    Additionally, while the image is supposed to have shape (y, x, c), the alpha channel will be stacked as a single
    channel with shape (y, x).
    """

    def __init__(self, label: str) -> None:
        self._alpha_label = label

    def stack(self, data_array: xarray.DataArray, is_fg_array: xarray.DataArray) -> xarray.DataArray:
        """Stacks the alpha channel on top of the data array."""
        return xarray.concat(
            [data_array, is_fg_array.expand_dims("c", axis=-1).assign_coords(c=[self._alpha_label])], dim="c"
        )

    def split(self, combined: xarray.DataArray) -> tuple[xarray.DataArray, xarray.DataArray]:
        """Splits the alpha channel off the combined array and returns the data array and the alpha array."""
        data_array = combined.drop_sel(c=self._alpha_label)
        is_fg_array = combined.sel(c=self._alpha_label).drop_vars("c").astype(bool)
        return data_array, is_fg_array
