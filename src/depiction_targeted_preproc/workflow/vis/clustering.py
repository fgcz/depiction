from pathlib import Path
from typing import Annotated

import typer
from matplotlib import pyplot as plt
from typer import Option

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.visualize.plot_image import PlotImage


def vis_clustering(input_netcdf_path: Annotated[Path, Option()],
                   output_png_path: Annotated[Path, Option()]
                   ) -> None:
    source_image = MultiChannelImage.read_hdf5(input_netcdf_path)
    plot_image = PlotImage(source_image)
    fig = plt.figure(figsize=(10, 10))
    # categorical cmap
    cmap = "tab20"
    plot_image.plot_single_channel_image(channel=source_image.channel_names[0], cmap=cmap, ax=fig.add_subplot(111))
    fig.tight_layout()
    fig.savefig(output_png_path)


if __name__ == "__main__":
    typer.run(vis_clustering)
