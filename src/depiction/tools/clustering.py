from enum import Enum
from pathlib import Path

import cyclopts
import numpy as np
import xarray
from depiction.clustering.extrapolate import extrapolate_labels
from depiction.clustering.stratified_grid import StratifiedGrid
from depiction.image.multi_channel_image import MultiChannelImage
from numpy.typing import NDArray
from sklearn.cluster import KMeans, BisectingKMeans
from xarray import DataArray


class MethodEnum(Enum):
    KMEANS = "kmeans"
    BISECTINGKMEANS = "bisectingkmeans"


app = cyclopts.App()


@app.default()
def clustering(
    input_hdf5: Path,
    output_hdf5: Path,
    method: MethodEnum,
    method_params: str,
) -> None:
    image = MultiChannelImage.read_hdf5(path=input_hdf5)
    n_samples = 5000
    grid = StratifiedGrid(cells_x=20, cells_y=20)
    rng = np.random.default_rng(42)
    if "featscv" in method_params:
        image = select_features_cv(image)
    sampled_features = grid.sample_points(array=image.data_flat, n_samples=n_samples, rng=rng)
    sampled_labels = compute_labels(features=sampled_features.T, method=method, method_params=method_params)
    full_labels = extrapolate_labels(
        sampled_features=sampled_features.values.T,
        sampled_labels=sampled_labels,
        full_features=image.data_flat.values.T,
    )
    label_image = MultiChannelImage.from_sparse(
        values=full_labels[:, np.newaxis], coordinates=image.coordinates_flat, channel_names=["cluster"]
    )

    output_image = MultiChannelImage(xarray.concat([label_image.data_spatial, image.data_spatial], dim="c"))
    output_image.write_hdf5(output_hdf5)


def select_features_cv(image: MultiChannelImage, n_keep: int = 30) -> DataArray:
    image_data = image.data_flat
    cv_score = image_data.std("i") / image_data.mean("i")
    return image.retain_channels(coords=cv_score.sortby("c")[-n_keep:].c.values)


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
