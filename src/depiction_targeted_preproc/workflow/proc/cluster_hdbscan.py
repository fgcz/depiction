from pathlib import Path
from typing import Annotated

import math
import numpy as np
import typer
import xarray
from hdbscan.flat import (HDBSCAN_flat)
from loguru import logger
from sklearn.preprocessing import StandardScaler
from typer import Option

from depiction.image.multi_channel_image import MultiChannelImage
from depiction_targeted_preproc.workflow.proc.cluster_kmeans import retain_strongest_signals


def cluster_dbscan(input_netcdf_path: Annotated[Path, Option()], output_netcdf_path: Annotated[Path, Option()]) -> None:
    image = MultiChannelImage.read_hdf5(input_netcdf_path)
    # TODO make configurable
    n_clusters = 10
    n_features = 50

    reduced_data = retain_strongest_signals(image.data_flat.transpose("i", "c"), n_features)

    scaler = StandardScaler()
    scaler.fit(reduced_data.values)

    # kmeans = BisectingKMeans(n_clusters=n_clusters)
    # dbscan = (eps=0.3, min_samples=10)
    data_scaled = scaler.transform(reduced_data.values)

    try:
        clusterer = HDBSCAN_flat(data_scaled,
                                 n_clusters=10)
                                 #min_cluster_size=math.ceil(0.02 * data_scaled.shape[0]))
        clusters = clusterer.labels_
    except IndexError:
        logger.error("No clusters found")
        clusters = np.zeros(data_scaled.shape[0])

    cluster_data = xarray.DataArray(clusters, dims=("i",), coords={"i": image.data_flat.coords["i"]}).expand_dims("c")
    cluster_data.coords["c"] = ["cluster"]
    cluster_data.attrs["bg_value"] = np.nan
    cluster_image = MultiChannelImage(cluster_data.unstack("i"))
    cluster_image.write_hdf5(output_netcdf_path)


if __name__ == "__main__":
    typer.run(cluster_dbscan)
