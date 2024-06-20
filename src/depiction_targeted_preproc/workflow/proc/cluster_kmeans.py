from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import xarray
from loguru import logger
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typer import Option
from xarray import DataArray

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


def retain_interesting_signals(data: xarray.DataArray, n_features: int) -> xarray.DataArray:
    model_nmf = NMF(n_components=5)
    model_nmf.fit(data.values)
    nmf_data = DataArray(model_nmf.components_, dims=["comp", "c"], coords={"c": data.coords["c"]})
    means = nmf_data.mean("comp")
    features = means.sortby(means, ascending=False)[:10].c.values
    logger.info(f"Retained channels: {features}")
    return data.sel(c=features)


def binarize_signals(data: xarray.DataArray) -> xarray.DataArray:
    signal_median = np.nanmedian(xarray.where(data > 0, data, np.nan), 0)
    return (data > signal_median).astype(int)


def find_num_clusters(data):
    # TODO this is from stackoverflow
    sil_score_max = -1  # this is the minimum possible score
    best_n_clusters = 1

    for n_clusters in range(2, 10):
        model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
        labels = model.fit_predict(data)
        sil_score = silhouette_score(data, labels)
        print("The average silhouette score for %i clusters is %0.2f" % (n_clusters, sil_score))
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters

    return best_n_clusters


def cluster_kmeans(input_netcdf_path: Annotated[Path, Option()], output_netcdf_path: Annotated[Path, Option()]) -> None:
    image = MultiChannelImage.read_hdf5(input_netcdf_path)
    # TODO make configurable
    # n_clusters = 5
    n_features = 50

    reduced_data = retain_interesting_signals(image.data_flat.transpose("i", "c"), n_features)
    reduced_data = binarize_signals(reduced_data)

    #scaler = StandardScaler()
    #scaler.fit(reduced_data.values)

    # n_clusters = find_num_clusters(scaler.transform(reduced_data.values))
    n_clusters = 7

    kmeans = BisectingKMeans(n_clusters=n_clusters)
    #clusters = kmeans.fit_predict(scaler.transform(reduced_data.values))
    clusters = kmeans.fit_predict(reduced_data.values)

    cluster_data = xarray.DataArray(clusters, dims=("i",), coords={"i": image.data_flat.coords["i"]}).expand_dims("c")
    cluster_data.coords["c"] = ["cluster"]
    cluster_data.attrs["bg_value"] = np.nan
    cluster_image = MultiChannelImage(cluster_data.unstack("i"))
    cluster_image.write_hdf5(output_netcdf_path)


if __name__ == "__main__":
    typer.run(cluster_kmeans)
