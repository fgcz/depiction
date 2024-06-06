from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import xarray
from sklearn.cluster import KMeans
from typer import Option

from depiction.image.multi_channel_image import MultiChannelImage


def cluster_kmeans(input_netcdf_path: Annotated[Path, Option()], output_netcdf_path: Annotated[Path, Option()]) -> None:
    image = MultiChannelImage.read_hdf5(input_netcdf_path)
    k = 10

    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(image.data_flat.transpose("i", "c").values)

    cluster_data = xarray.DataArray(
        clusters, dims=("i",), coords={"i": image.data_flat.coords["i"]}
    ).expand_dims("c")
    cluster_data.coords["c"] = ["cluster"]
    cluster_data.attrs["bg_value"] = np.nan
    cluster_image = MultiChannelImage(cluster_data.unstack("i"))
    cluster_image.write_hdf5(output_netcdf_path)


if __name__ == "__main__":
    typer.run(cluster_kmeans)
