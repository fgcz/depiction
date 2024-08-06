from enum import Enum
from pathlib import Path

import cyclopts
import numpy as np
import xarray
from loguru import logger

from depiction.clustering.extrapolate import extrapolate_labels
from depiction.clustering.maxmin_sampling import maxmin_sampling
from depiction.clustering.metrics import cross_correlation
from depiction.clustering.stratified_grid import StratifiedGrid
from depiction.image.multi_channel_image import MultiChannelImage
from numpy.typing import NDArray
from sklearn.cluster import KMeans, BisectingKMeans


class MethodEnum(Enum):
    KMEANS = "kmeans"
    BISECTINGKMEANS = "bisectingkmeans"


app = cyclopts.App()


def retain_n_best_features(image_full: MultiChannelImage, n_best_features: int) -> MultiChannelImage:
    strongest_channels = image_full.channel_stats.interquartile_range.drop_nulls().sort("iqr").tail(n_best_features)
    logger.info(f"Retaining {n_best_features} best features: {strongest_channels['c'].to_numpy()}")
    return image_full.retain_channels(coords=strongest_channels["c"].to_numpy())


# TODO the feature selection should also be studied on a per-image basis to identify potential relevant differences


@app.default()
def clustering(
    input_hdf5: Path,
    output_hdf5: Path,
    method: MethodEnum,
    method_params: str,
    n_best_features: int = 30,
    n_samples_cluster: int = 10000,
    n_landmarks: int = 50,
) -> None:
    image_full = MultiChannelImage.read_hdf5(path=input_hdf5)
    image_full = image_full.drop_channels(coords=["image_index", "cluster"], allow_missing=True)
    rng = np.random.default_rng(42)

    # retain only the most informative features for the clustering
    image = retain_n_best_features(image_full, n_best_features=n_best_features)

    # sample a number of landmark features which will be used for correlation-based clustering
    # TODO it is possible that the landmarks are all from the same image in the concatenated case,
    #      which needs to be addressed somehow...
    landmark_indices = maxmin_sampling(image.data_flat.values.T, k=n_landmarks, rng=rng)
    landmark_features = image.data_flat.values.T[landmark_indices]

    # sample a large number of samples to cluster against the full image
    # TODO this could be improved a bit by making sure that the landmarks are never sampled here again
    grid = StratifiedGrid(cells_x=20, cells_y=20)
    sampled_features = grid.sample_points(array=image.data_flat, n_samples=n_samples_cluster, rng=rng).values.T

    # compute pairwise correlation between landmark features and sampled features
    correlation_features = cross_correlation(sampled_features, landmark_features)

    # perform the clustering on the sampled features
    sampled_labels = compute_labels(features=correlation_features, method=method, method_params=method_params)

    # extrapolate the labels to the full image
    full_labels = extrapolate_labels(
        sampled_features=sampled_features,
        sampled_labels=sampled_labels,
        full_features=image.data_flat.values.T,
    )
    label_image = MultiChannelImage.from_sparse(
        values=full_labels[:, np.newaxis], coordinates=image_full.coordinates_flat, channel_names=["cluster"]
    )

    # write the result of the operation
    output_image = MultiChannelImage(xarray.concat([label_image.data_spatial, image_full.data_spatial], dim="c"))
    output_image.write_hdf5(output_hdf5)


# def select_features_cv(image: MultiChannelImage, n_keep: int = 30) -> DataArray:
#    image_data = image.data_flat
#    cv_score = image_data.std("i") / image_data.mean("i")
#    return image.retain_channels(coords=cv_score.sortby("c")[-n_keep:].c.values)
#


def compute_labels(features: NDArray[float], method: MethodEnum, method_params: str) -> NDArray[int]:
    if method == MethodEnum.KMEANS:
        clu = KMeans(n_clusters=10).fit(features)
        return clu.labels_
    elif method == MethodEnum.BISECTINGKMEANS:
        clu = BisectingKMeans(n_clusters=10).fit(features)
        return clu.labels_
    else:
        raise ValueError(f"Method {method} not implemented")


if __name__ == "__main__":
    app()
