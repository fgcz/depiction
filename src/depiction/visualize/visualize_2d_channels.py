# TODO delete this file if not needed anymore
# from typing import Optional, Callable, Sequence
#
# from matplotlib import pyplot as plt
#
# from ion_mapper.image.sparse_image_2d import SparseImage2d
#
#
## TODO the configuration of intensity transformation needs to be more generic / standardized
#
# class Visualize2dChannels:
#    """Visualizes 2D image channels."""
#
#    def __init__(self, image: SparseImage2d):
#        self._image = image
#
#    def plot_channel(
#        self,
#        channel_index: int,
#        ax: plt.Axes,
#
#    ):
#
#    def plot_channel_old(
#        self,
#        channel_index: int,
#        ax: Optional[plt.Axes] = None,
#        cmap: str = "mako",
#        transform_int: Optional[Callable] = None,
#        vmin: Optional[float] = None,
#        vmax: Optional[float] = None,
#        vmin_fn: Optional[Callable] = None,
#        vmax_fn: Optional[Callable] = None,
#        interpolation: Optional[str] = None,
#        mask_background: bool = False,
#    ):
#        pass
#
#    def plot_channels(self, channel_indices: Sequence[int], axes: Sequence[plt.Axes]):
#        for channel_index, ax in zip(channel_indices, axes):
#            self.plot_channel(channel_index=channel_index, ax=ax)
#
#    def plot_channels_grid(
#        self,
#        n_per_row: int = 5,
#        cmap: str = "mako",
#        transform_int: Optional[Callable] = None,
#        single_im_width: float = 2.0,
#        single_im_height: Optional[float] = None,
#        # TODO consider if this can be handled better (i.e. the parameter interface mainly)
#        vmin: Optional[float] = None,
#        vmax: Optional[float] = None,
#        vmin_fn: Optional[Callable] = None,
#        vmax_fn: Optional[Callable] = None,
#        interpolation: Optional[str] = None,
#        mask_background: bool = False,
#        axs: Optional[NDArray[plt.Axes]] = None,
#    ) -> tuple[plt.Figure, plt.Axes]:
#        pass
#
