import numpy as np
import pytest
import xarray

from depiction.clustering.remap_clusters import remap_cluster_labels, get_centroids
from depiction.image.multi_channel_image import MultiChannelImage


@pytest.fixture()
def label_image() -> MultiChannelImage:
    values = np.array([[0, 1, 0], [0, 0, 0], [0, 2, 2]]).reshape(3, 3, 1)
    data = xarray.DataArray(values, dims=("y", "x", "c"), coords={"c": ["cluster"]}, attrs={"bg_value": np.nan})
    return MultiChannelImage(data)


def test_get_centroids():
    n_clusters = 3
    expected_centroids = np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]])
    points = []
    labels = [0, 1, 2]
    for label, centroid in zip(labels, expected_centroids):
        points.append([*centroid, label])
        points.append([*(centroid - [0.1, 0.1]), label])
        points.append([*(centroid + [0.1, 0.1]), label])
    data_flat = xarray.DataArray(points, dims=("i", "c"), coords={"c": ["f1", "f2", "cluster"]})
    centroids = get_centroids(data_flat, n_clusters)
    np.testing.assert_allclose(centroids, expected_centroids)


def test_remap_cluster_labels(label_image):
    remapped = remap_cluster_labels(image=label_image, mapping={0: 1, 1: 0, 2: 2})
    expected = np.array([[1, 0, 1], [1, 1, 1], [1, 2, 2]]).reshape(3, 3, 1)
    assert np.allclose(remapped.data_spatial.values, expected)
    assert np.isnan(remapped.data_spatial.attrs["bg_value"])
    assert remapped.data_spatial.coords["c"].values == ["cluster"]
