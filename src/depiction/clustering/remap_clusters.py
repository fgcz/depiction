import numpy as np
import xarray
from loguru import logger
from scipy.optimize import linear_sum_assignment

from depiction.image.multi_channel_image import MultiChannelImage


def get_centroids(data_flat: xarray.DataArray, n_clusters: int) -> np.ndarray:
    """Returns the centroids of the clusters in the data array.
    :param data_flat: (n_samples, n_features) the data array with a cluster label
    :param n_clusters: the number of clusters
    """
    # TODO consider if n_clusters can be refactored away by infering the value
    n_classes = data_flat.sizes["c"] - 1
    centroids = np.zeros((n_clusters, n_classes))
    for i_cluster in range(n_clusters):
        centroids[i_cluster] = (
            data_flat.where(data_flat.sel(c="cluster") == i_cluster).drop_sel(c="cluster").mean(dim="i").values
        )
    return centroids


def compute_remapping(centroids_fixed: np.ndarray, centroids_moving: np.ndarray) -> dict[int, int]:
    """Computes a mapping of original to new cluster labels for the moving label image.
    This is done by solving the linear sum assignment problem with a cost matrix based on the correlation of the
    centroids.
    :param centroids_fixed: (n_clusters, n_features) the centroids of the fixed image
    :param centroids_moving: (n_clusters, n_features) the centroids of the moving image
    :return: a dictionary mapping the old cluster labels to the new ones
    """
    n_clusters = centroids_fixed.shape[0]
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost_matrix[i, j] = 1 - np.abs(np.corrcoef(centroids_fixed[i], centroids_moving[j])[0, 1])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # print info about the costs
    logger.debug(f"cost of original assignment: {np.diag(cost_matrix).sum():.4f}")
    logger.debug(f"cost of original assignment per label: {repr(np.diag(cost_matrix).round(4))}")
    logger.debug(f"cost of chosen assignment: {cost_matrix[row_ind, col_ind].sum():.4f}")
    logger.debug(f"cost of chosen assignment per label: {repr(cost_matrix[row_ind, col_ind].round(4))}")
    # compute the cost of the worst possible assignment (by inverse problem)
    worst_row_ind, worst_col_ind = linear_sum_assignment(-cost_matrix)
    worst_cost = cost_matrix[worst_row_ind, worst_col_ind].sum()
    logger.debug(f"cost of worst assignment: {worst_cost:.4f}")
    # return the mapping
    return dict(zip(col_ind, row_ind))


def remap_cluster_labels(
    image: MultiChannelImage, mapping: dict[int, int], cluster_channel: str = "cluster"
) -> MultiChannelImage:
    """Remaps the cluster labels in the image according to the given mapping.
    :param image: the image with cluster labels
    :param mapping: a dictionary mapping the old cluster labels to the new ones
    :param cluster_channel: the name of the channel with the cluster labels
    """
    with xarray.set_options(keep_attrs=True):
        relabeled = xarray.apply_ufunc(
            lambda v: mapping.get(v, np.nan), image.data_spatial.sel(c=[cluster_channel]), vectorize=True
        )
    img_relabeled = image.drop_channels(coords=[cluster_channel], allow_missing=False).append_channels(
        MultiChannelImage(relabeled)
    )
    return img_relabeled
