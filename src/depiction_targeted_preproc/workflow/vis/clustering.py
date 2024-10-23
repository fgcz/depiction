import cyclopts
import numpy as np
import xarray
from matplotlib import pyplot as plt
from pathlib import Path

app = cyclopts.App()


@app.default
def vis_clustering(input_netcdf_path: Path, output_png_path: Path) -> None:
    source_image = xarray.open_dataarray(input_netcdf_path)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    source_image.isel(c=0).plot.imshow(x="x", y="y", cmap="tab10", ax=ax, yincrease=False)
    ax.set_aspect("equal")

    # n_classes = len(set(source_image.values.ravel()))
    n_classes = len(np.unique(source_image.values))
    fig.suptitle(f"Clustering with {n_classes} classes")
    fig.tight_layout()
    fig.savefig(output_png_path)


if __name__ == "__main__":
    app()
