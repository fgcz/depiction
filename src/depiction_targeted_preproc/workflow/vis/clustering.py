from pathlib import Path
from typing import Annotated

import typer
import xarray
from matplotlib import pyplot as plt
from typer import Option


def vis_clustering(input_netcdf_path: Annotated[Path, Option()], output_png_path: Annotated[Path, Option()]) -> None:
    source_image = xarray.open_dataarray(input_netcdf_path)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    source_image.plot(cmap="tab10", ax=ax)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_png_path)


if __name__ == "__main__":
    typer.run(vis_clustering)
