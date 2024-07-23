from enum import Enum

from depiction.clustering.extrapolate import extrapolate_labels
from numpy.typing import NDArray
from pathlib import Path

import cyclopts
import numpy as np
from depiction.clustering.stratified_grid import StratifiedGrid
from depiction.image.multi_channel_image import MultiChannelImage
from sklearn.cluster import KMeans


class MethodEnum(Enum):
    KMEANS = "kmeans"


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
    label_image.write_hdf5(output_hdf5)


def compute_labels(features: NDArray[float], method: MethodEnum, method_params: str) -> NDArray[int]:
    if method == MethodEnum.KMEANS:
        clu = KMeans(n_clusters=10).fit(features)
        return clu.labels_
    else:
        raise ValueError(f"Method {method} not implemented")


if __name__ == "__main__":
    app()
