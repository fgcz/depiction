from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from depiction.image.multi_channel_image import MultiChannelImage


class ImageChannelStats:
    """Provides statistics for each channel of a multi-channel image."""

    def __init__(self, image: MultiChannelImage) -> None:
        self._image = image

    @cached_property
    def five_number_summary(self) -> pl.DataFrame:
        """Returns a DataFrame with the five number summary for each channel,
        columns 'c', 'min', 'q1', 'median', 'q3', and 'max'."""
        data = np.zeros((self._image.n_channels, 5))
        for i_channel in range(self._image.n_channels):
            values = self._get_channel_values(i_channel=i_channel, drop_missing=True)
            if len(values) == 0:
                data[i_channel] = np.nan
                continue
            data[i_channel] = np.percentile(values, [0, 25, 50, 75, 100])
        return pl.DataFrame(
            {
                "c": self._image.channel_names,
                "min": data[:, 0],
                "q1": data[:, 1],
                "median": data[:, 2],
                "q3": data[:, 3],
                "max": data[:, 4],
            }
        ).fill_nan(None)

    @cached_property
    def coefficient_of_variation(self) -> pl.DataFrame:
        """Returns a DataFrame with the cv for each channel, columns 'c', and 'cv'."""
        data = np.zeros(self._image.n_channels)
        for i_channel in range(self._image.n_channels):
            values = self._get_channel_values(i_channel=i_channel, drop_missing=True)
            if len(values) == 0:
                data[i_channel] = np.nan
                continue
            data[i_channel] = np.std(values) / np.mean(values)
        return pl.DataFrame({"c": self._image.channel_names, "cv": data}).fill_nan(None)

    @cached_property
    def interquartile_range(self) -> pl.DataFrame:
        """Returns a DataFrame with the iqr for each channel c, columns 'c', and 'iqr'."""
        data = np.zeros(self._image.n_channels)
        for i_channel in range(self._image.n_channels):
            values = self._get_channel_values(i_channel=i_channel, drop_missing=True)
            if len(values) == 0:
                data[i_channel] = np.nan
                continue
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            data[i_channel] = q3 - q1
        return pl.DataFrame({"c": self._image.channel_names, "iqr": data}).fill_nan(None)

    @cached_property
    def mean(self) -> pl.DataFrame:
        """Returns a DataFrame with the mean for each channel, columns 'c', and 'mean'."""
        data = np.zeros(self._image.n_channels)
        for i_channel in range(self._image.n_channels):
            values = self._get_channel_values(i_channel=i_channel, drop_missing=True)
            if len(values) == 0:
                data[i_channel] = np.nan
                continue
            data[i_channel] = np.mean(values)
        return pl.DataFrame({"c": self._image.channel_names, "mean": data}).fill_nan(None)

    @cached_property
    def std(self) -> pl.DataFrame:
        """Returns a DataFrame with the standard deviation for each channel, columns 'c', and 'std'."""
        data = np.zeros(self._image.n_channels)
        for i_channel in range(self._image.n_channels):
            values = self._get_channel_values(i_channel=i_channel, drop_missing=True)
            if len(values) == 0:
                data[i_channel] = np.nan
                continue
            data[i_channel] = np.std(values)
        return pl.DataFrame({"c": self._image.channel_names, "std": data}).fill_nan(None)

    def _get_channel_values(self, i_channel: int, drop_missing: bool) -> np.ndarray:
        """Returns the values of the given channel."""
        data_channel = self._image.data_flat.isel(c=i_channel).values
        if drop_missing:
            bg = self._image.bg_value
            bg_mask = np.isnan(data_channel) if np.isnan(bg) else data_channel == bg
            data_channel = data_channel[~bg_mask]
        return data_channel
