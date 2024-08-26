import json
from enum import Enum
from pathlib import Path

import cyclopts
import numpy as np
import xarray
from numpy.typing import NDArray
from sklearn.cluster import KMeans, BisectingKMeans, Birch

from depiction.clustering.extrapolate import extrapolate_labels
from depiction.clustering.maxmin_sampling import maxmin_sampling
from depiction.clustering.metrics import cross_correlation
from depiction.clustering.stratified_grid import StratifiedGrid
from depiction.image.feature_selection import FeatureSelectionIQR, retain_features
from depiction.image.image_normalization import ImageNormalization, ImageNormalizationVariant
from depiction.image.multi_channel_image import MultiChannelImage


class MethodEnum(Enum):
    KMEANS = "kmeans"
    BISECTINGKMEANS = "bisectingkmeans"
    BIRCH = "birch"


MethodParamsType = dict[str, bool | str | float | int | None]

app = cyclopts.App()


def get_landmark_indices(
    image_features: MultiChannelImage,
    image_index: MultiChannelImage,
    n_landmarks: int,
    rng: np.random.Generator,
    metric: str,
) -> NDArray[int]:
    image_joined = image_features.append_channels(image_index)
    image_joined_flat = image_joined.data_flat
    n_images = int(image_index.data_flat.values.max() + 1)
    indices = []
    for i_image in range(n_images):
        # determine the indices corresponding to i_image in the flat representation
        indices_image = np.where(image_joined_flat.sel(c="image_index").values == i_image)[0]

        # number of landmarks to retrieve for this particular image
        n_samples = n_landmarks // n_images if i_image != 0 else n_landmarks // n_images + n_landmarks % n_images

        # determine the landmark indices
        features_image = image_joined_flat.drop_sel(c="image_index").isel(i=indices_image)
        indices_image_landmarks = maxmin_sampling(features_image.values.T, k=n_samples, rng=rng, metric=metric)

        # revert these indices into the original space
        indices.extend(indices_image[indices_image_landmarks])
    return np.asarray(indices)


def compute_clustering(
    input_image: MultiChannelImage,
    method: MethodEnum,
    method_params: MethodParamsType,
    n_best_features: int = 30,
    n_samples_cluster: int = 10000,
    n_landmarks: int = 200,
    landmark_metric: str = "correlation",
) -> MultiChannelImage:
    rng = np.random.default_rng(42)

    # massage the input image
    assert "cluster" not in input_image.channel_names
    image_full_features = input_image.drop_channels(coords=["image_index"], allow_missing=True)
    image_full_features = ImageNormalization().normalize_image(
        image=image_full_features, variant=ImageNormalizationVariant.STD
    )
    if "image_index" in input_image.channel_names:
        image_full_image_index = input_image.retain_channels(coords=["image_index"])
    else:
        with xarray.set_options(keep_attrs=True):
            image_full_image_index = MultiChannelImage(
                xarray.zeros_like(input_image.data_spatial.isel(c=[0])).assign_coords(c=["image_index"])
            )

    # retain the most relevant features
    image_features = retain_features(
        feature_selection=FeatureSelectionIQR.model_validate({"n_features": n_best_features}), image=image_full_features
    )

    # sample a number of landmark features which will be used for correlation-based clustering
    # since we might have more than one image, we want to make sure that we sample a bit of each
    landmark_indices = get_landmark_indices(
        image_features=image_features,
        image_index=image_full_image_index,
        n_landmarks=n_landmarks,
        rng=rng,
        metric=landmark_metric,
    )

    landmark_features = image_features.data_flat.values.T[landmark_indices]

    # sample a large number of samples to cluster against the full image
    # TODO this could be improved a bit by making sure that the landmarks are never sampled here again
    grid = StratifiedGrid(cells_x=20, cells_y=20)
    sampled_features = grid.sample_points(array=image_features.data_flat, n_samples=n_samples_cluster, rng=rng).values.T

    # compute pairwise correlation between landmark features and sampled features
    correlation_features = cross_correlation(sampled_features, landmark_features)

    # perform the clustering on the sampled features
    sampled_labels = compute_labels(features=correlation_features, method=method, method_params=method_params)

    # extrapolate the labels to the full image
    full_labels = extrapolate_labels(
        sampled_features=sampled_features,
        sampled_labels=sampled_labels,
        full_features=image_features.data_flat.values.T,
    )
    label_image = MultiChannelImage.from_sparse(
        values=full_labels[:, np.newaxis],
        coordinates=image_full_features.coordinates_flat,
        channel_names=["cluster"],
        bg_value=np.nan,
    )

    # return the result of the operation
    return MultiChannelImage(xarray.concat([label_image.data_spatial, input_image.data_spatial], dim="c"))


@app.default()
def clustering(
    input_hdf5: Path,
    output_hdf5: Path,
    method: MethodEnum,
    method_params: str = "{}",
    n_best_features: int = 30,
    n_samples_cluster: int = 10000,
    n_landmarks: int = 200,
    landmark_metric: str = "correlation",
) -> None:
    image_full_combined = MultiChannelImage.read_hdf5(path=input_hdf5)
    output_image = compute_clustering(
        input_image=image_full_combined,
        method=method,
        method_params=json.loads(method_params) | {},
        n_best_features=n_best_features,
        n_samples_cluster=n_samples_cluster,
        n_landmarks=n_landmarks,
        landmark_metric=landmark_metric,
    )
    output_image.write_hdf5(output_hdf5)


def compute_labels(features: NDArray[float], method: MethodEnum, method_params: MethodParamsType) -> NDArray[int]:
    if method == MethodEnum.KMEANS:
        params = method_params | {"n_clusters": 10}
        clu = KMeans(**params).fit(features)
        return clu.labels_
    elif method == MethodEnum.BISECTINGKMEANS:
        clu = BisectingKMeans(n_clusters=10).fit(features)
        return clu.labels_
    elif method == MethodEnum.BIRCH:
        clu = Birch(n_clusters=10).fit(features)
        return clu.labels_
    else:
        raise ValueError(f"Method {method} not implemented")


if __name__ == "__main__":
    app()
