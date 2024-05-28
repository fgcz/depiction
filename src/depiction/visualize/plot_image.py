from __future__ import annotations

import math
from typing import Callable, TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from depiction.image.multi_channel_image import MultiChannelImage
    from numpy.typing import NDArray


class PlotImage:
    def __init__(self, image: MultiChannelImage) -> None:
        self._image = image

    def plot_single_channel_image(
        self,
        channel: str,
        ax: plt.Axes | None = None,
        cmap: str = "mako",
        transform_int: Callable | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        vmin_fn: Callable | None = None,
        vmax_fn: Callable | None = None,
        interpolation: str | None = None,
        mask_background: bool = False,
    ) -> None:
        if ax is None:
            ax = plt.gca()

        ax.set_title(channel)
        ax.set_xticks([])
        ax.set_yticks([])

        channel_values_flat = self._image.get_channel_flat_array(name=channel)

        if vmin_fn is not None:
            vmin = vmin_fn(channel_values_flat.values)
        if vmax_fn is not None:
            vmax = vmax_fn(channel_values_flat.values)

        dense_values = self._image.get_channel_array(channel).values
        if transform_int is not None:
            dense_values = transform_int(dense_values)
        if mask_background:
            # TODO check if working
            # dense_values = np.ma.masked_invalid(dense_values)
            dense_values = np.ma.masked_where(
                dense_values == self._image.bg_value | (np.isnan(dense_values) & np.isnan(self._image.bg_value)),
                dense_values,
            )

        ax.imshow(
            dense_values,
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
        )

    def plot_channels_grid(
        self,
        n_per_row: int = 5,
        cmap: str = "mako",
        transform_int: Callable | None = None,
        single_im_width: float = 2.0,
        single_im_height: float | None = None,
        # TODO consider if this can be handled better (i.e. the parameter interface mainly)
        vmin: float | None = None,
        vmax: float | None = None,
        vmin_fn: Callable | None = None,
        vmax_fn: Callable | None = None,
        interpolation: str | None = None,
        mask_background: bool = False,
        axs: NDArray[plt.Axes] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        # determine plots layout
        n_channels = self._image.n_channels
        n_rows = math.ceil(n_channels / n_per_row)
        n_cols = min(n_channels, n_per_row)

        # determine the figure size
        im_width, im_height = self._image.dimensions
        aspect_ratio = im_width / im_height

        # set up the grid
        if single_im_height is None:
            single_im_height = single_im_width

        if axs is None:
            fig, axs = plt.subplots(
                n_rows,
                n_cols,
                figsize=(
                    n_cols * single_im_width,
                    n_rows * single_im_height * aspect_ratio,
                ),
                squeeze=False,
            )
        else:
            fig = axs.ravel()[0].get_figure()

        # create the plots
        # TODO logic:this is currently going to fail if there are multiple channels with the same image,
        #  an alternative would be to directly obtain the image data and then iterate over it... but would require
        #  background handling maybe
        for i_channel, channel in enumerate(self._image.channel_names):
            # Get the axis
            i_row = i_channel // n_per_row
            i_col = i_channel % n_per_row
            ax = axs[i_row, i_col]

            self.plot_single_channel_image(
                channel=channel,
                ax=ax,
                cmap=cmap,
                transform_int=transform_int,
                vmin=vmin,
                vmax=vmax,
                vmin_fn=vmin_fn,
                vmax_fn=vmax_fn,
                interpolation=interpolation,
                mask_background=mask_background,
            )

        # remove redundant axes
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                if i_row * n_per_row + i_col >= n_channels:
                    axs[i_row, i_col].set_axis_off()

        return fig, axs
