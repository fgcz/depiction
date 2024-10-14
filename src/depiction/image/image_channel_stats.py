from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import xarray

if TYPE_CHECKING:
    from depiction.image.multi_channel_image import MultiChannelImage


class ImageChannelStats:
    """Provides statistics for each channel of a multi-channel image."""

    def __init__(self, image: MultiChannelImage) -> None:
        self._image = image

    @cached_property
    def five_number_summary(self) -> xarray.DataArray:
        """Returns a DataArray with the five number summary for each channel,
        columns 'c', 'min', 'q1', 'median', 'q3', and 'max'."""
        data = np.zeros((self._image.n_channels, 5))
        for i_channel in range(self._image.n_channels):
            values = self._get_channel_values(i_channel=i_channel, drop_missing=True)
            if len(values) == 0:
                data[i_channel] = np.nan
                continue
            data[i_channel] = np.percentile(values, [0, 25, 50, 75, 100])

        return xarray.DataArray(
            data,
            dims=("c", "metric"),
            coords={"c": self._image.channel_names, "metric": ["min", "q1", "median", "q3", "max"]},
        ).fillna(None)

    @cached_property
    def coefficient_of_variation(self) -> xarray.DataArray:
        """Returns a DataFrame with the cv for each channel, columns 'c', and 'cv'."""
        return self._compute_scalar_metric(fn=lambda x: np.std(x) / np.mean(x), min_values=2)

    @cached_property
    def interquartile_range(self) -> xarray.DataArray:
        """Returns a DataArray with the iqr for each channel c, columns 'c', and 'iqr'."""
        return self._compute_scalar_metric(fn=lambda x: np.percentile(x, 75) - np.percentile(x, 25), min_values=2)

    @cached_property
    def mean(self) -> xarray.DataArray:
        """Returns a DataArray with the mean for each channel."""
        return self._compute_scalar_metric(fn=np.mean, min_values=1)

    @cached_property
    def std(self) -> xarray.DataArray:
        """Returns a DataArray with the standard deviation for each channel, columns 'c', and 'std'."""
        return self._compute_scalar_metric(fn=np.std, min_values=2)

    def _compute_scalar_metric(self, fn, min_values: int):
        data = np.zeros(self._image.n_channels)
        for i_channel in range(self._image.n_channels):
            values = self._get_channel_values(i_channel=i_channel, drop_missing=True)
            if min_values <= len(values):
                data[i_channel] = fn(values)
            else:
                data[i_channel] = np.nan
        return xarray.DataArray(data, dims="c", coords={"c": self._image.channel_names})

    def _get_channel_values(self, i_channel: int, drop_missing: bool) -> np.ndarray:
        """Returns the values of the given channel."""
        # TODO maybe caching data_flat would already make this faster, could be tested easily by temporarily adding the cache in the MultiChannelImage class
        data_channel = self._image.data_flat.isel(c=i_channel).values
        if drop_missing:
            data_channel = data_channel[self._image.fg_mask_flat]
        return data_channel
