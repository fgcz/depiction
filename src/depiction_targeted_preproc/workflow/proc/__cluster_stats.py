import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler

# TODO this file is problematic, because it copies unlicensed code from
#    https://github.com/zhaofangyuan98/SDMBench/blob/d60653469380f6bd0f574b732c0a20f8a984aafc/SDMBench/SDMBench.py#L108
# it would be best to reimplement it as anyhow it is also a bit slow


def compute_CHAOS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 1:  # Adjusted to <= 1 to handle clusters with one or fewer points
            continue
        n_location_cluster = len(location_cluster)
        results = [fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1

    return np.sum(dist_val) / len(clusterlabel)


def fx_1NN(i, location_in):
    location_in = np.array(location_in)
    if len(location_in) == 1:
        return 0  # If there's only one point in the cluster, distance to nearest neighbor is 0
    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    return np.min(dist_array)


# def compute_CHAOS(clusterlabel, location):
#    clusterlabel = np.array(clusterlabel)
#    location = np.array(location)
#    matched_location = StandardScaler().fit_transform(location)
#
#    clusterlabel_unique = np.unique(clusterlabel)
#    dist_val = np.zeros(len(clusterlabel_unique))
#    count = 0
#    for k in clusterlabel_unique:
#        location_cluster = matched_location[clusterlabel == k, :]
#        if len(location_cluster) <= 2:
#            continue
#        n_location_cluster = len(location_cluster)
#        results = [fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
#        dist_val[count] = np.sum(results)
#        count = count + 1
#
#    return np.sum(dist_val) / len(clusterlabel)
#
#
# def fx_1NN(i, location_in):
#    location_in = np.array(location_in)
#    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
#    dist_array[i] = np.inf
#    return np.min(dist_array)


def compute_PAS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel) for i in range(matched_location.shape[0])]
    return np.sum(results) / len(clusterlabel)


def fx_kNN(i, location_in, k, cluster_in):
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)

    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind] != cluster_in[i]) > (k / 2):
        return 1
    else:
        return 0
