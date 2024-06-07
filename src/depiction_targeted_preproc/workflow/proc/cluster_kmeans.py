from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import xarray
from loguru import logger
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.preprocessing import StandardScaler
from typer import Option

from depiction.image.multi_channel_image import MultiChannelImage


def retain_strongest_signals(data: xarray.DataArray, n_features: int) -> xarray.DataArray:
    """Retain the N strongest signals in the data."""
    # compute total signal strength per channel c
    total_signal_strength = xarray.apply_ufunc(np.fabs, data).sum(dim="i")
    # sort channels by total signal strength
    sorted_channels = total_signal_strength.argsort()
    # retain the N strongest signals
    strongest_channels = sorted_channels.coords["c"][-n_features:]
    logger.info(f"Retained channels: {strongest_channels}")
    return data.sel(c=strongest_channels)


def cluster_kmeans(input_netcdf_path: Annotated[Path, Option()], output_netcdf_path: Annotated[Path, Option()]) -> None:
    image = MultiChannelImage.read_hdf5(input_netcdf_path)
    # TODO make configurable
    n_clusters = 10
    n_features = 50

    reduced_data = retain_strongest_signals(image.data_flat.transpose("i", "c"), n_features)

    scaler = StandardScaler()
    scaler.fit(reduced_data.values)

    kmeans = BisectingKMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(scaler.transform(reduced_data.values))

    cluster_data = xarray.DataArray(clusters, dims=("i",), coords={"i": image.data_flat.coords["i"]}).expand_dims("c")
    cluster_data.coords["c"] = ["cluster"]
    cluster_data.attrs["bg_value"] = np.nan
    cluster_image = MultiChannelImage(cluster_data.unstack("i"))
    cluster_image.write_hdf5(output_netcdf_path)


if __name__ == "__main__":
    typer.run(cluster_kmeans)
