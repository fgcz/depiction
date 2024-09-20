from __future__ import annotations

import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr


class AmapClustering:
    """Bindings to the R amap library which implements some K-Means variants which are otherwise hard to find.
    If this works nicely, we could consider reimplementing some of these methods in Python.
    """

    def corr_kmeans(
        self, data: np.ndarray, clusters: int, *, abs_correlation: bool = False, metric: str = None
    ) -> np.ndarray:
        """
        Call R's amap::Kmeans from Python using rpy2.

        Parameters:
        data (np.ndarray): The data matrix for clustering.
        clusters (int): Number of clusters.
        abs_correlation (bool): Whether to use absolute correlation as a distance metric.

        Returns:
        np.ndarray: The cluster assignments.
        """
        np_cv_rules = ro.default_converter + numpy2ri.converter
        amap = importr("amap")

        with np_cv_rules.context():
            method = metric or ("abscorrelation" if abs_correlation else "correlation")

            # Call the R function
            kmeans_result = amap.Kmeans(x=data, centers=clusters, method=method, nstart=10, iter_max=100)

            # Convert to numpy
            return np.array(kmeans_result["cluster"])
