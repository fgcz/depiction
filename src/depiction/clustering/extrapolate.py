import numpy as np
from numpy.typing import NDArray


def extrapolate_labels(
    sampled_features: NDArray[float], sampled_labels: NDArray[int], full_features: NDArray[float]
) -> NDArray[int]:
    """Extrapolates cluster labels for a number of sampled features to the full set of features."""
    if sampled_features.shape[1] != full_features.shape[1]:
        raise ValueError(
            f"Number of features must be the same in sampled_features ({sampled_features.shape[1]}) and full_features ({full_features.shape[1]})"
        )
    n_full_samples = full_features.shape[0]
    cluster_centers = get_cluster_centers(features=sampled_features, labels=sampled_labels)
    full_labels = np.zeros(n_full_samples, dtype=int)
    for i_sample in range(n_full_samples):
        distances = np.linalg.norm(cluster_centers - full_features[i_sample], axis=1)
        full_labels[i_sample] = np.argmin(distances)
    return full_labels


def get_cluster_centers(features: NDArray[float], labels: NDArray[int]) -> NDArray[float]:
    """Returns the cluster centers for the given features and labels.
    This function assumes consecutive integers as labels.
    :param features: The features for each sample (n_samples, n_features).
    :param labels: The cluster labels for each sample (n_samples,).
    :return: The cluster centers (n_clusters, n_features) where n_clusters is the number of unique labels.
    """
    # normalize
    n_clusters = labels.max() - labels.min() + 1
    # shape: (n_samples,)
    labels = labels - labels.min()

    # compute cluster centers
    n_features = features.shape[1]
    cluster_centers = np.zeros((n_clusters, n_features))
    for i_cluster in range(n_clusters):
        cluster_features = features[labels == i_cluster, :]
        cluster_centers[i_cluster, :] = np.mean(cluster_features, axis=0)
    return cluster_centers
