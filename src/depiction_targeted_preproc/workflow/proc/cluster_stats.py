from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from loguru import logger
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typer import Option

from depiction.image.multi_channel_image import MultiChannelImage
from depiction_targeted_preproc.workflow.proc.__cluster_stats import compute_CHAOS, compute_PAS


def compute_silhouette(cluster_data, cluster_coords):
    # higher is better
    try:
        return silhouette_score(cluster_coords, cluster_data)
    except ValueError:
        return np.nan


def compute_davies_bouldin(cluster_data, cluster_coords):
    # higher is worse
    return davies_bouldin_score(cluster_coords, cluster_data)


def compute_calinski_harabasz(X, labels):
    try:
        return calinski_harabasz_score(X, labels)
    except ValueError:
        return np.nan


def compute_metrics(cluster_data: np.ndarray, cluster_coords: np.ndarray) -> dict[str, float]:
    chaos = compute_CHAOS(cluster_data, cluster_coords)
    logger.info(f"Computed CHAOS: {chaos}")

    pas = compute_PAS(cluster_data, cluster_coords)
    logger.info(f"Computed PAS: {pas}")

    silhouette = compute_silhouette(cluster_data, cluster_coords)
    logger.info(f"Computed silhouette: {silhouette}")

    davies_bouldin = compute_davies_bouldin(cluster_data, cluster_coords)
    logger.info(f"Computed Davies-Bouldin: {davies_bouldin}")

    calinski_harabasz = compute_calinski_harabasz(cluster_coords, cluster_data)
    logger.info(f"Computed Calinski-Harabasz: {calinski_harabasz}")

    return {
        "CHAOS": chaos,
        "PAS": pas,
        "Silhouette": silhouette,
        "Davies-Bouldin": davies_bouldin,
        "Calinski-Harabasz": calinski_harabasz,
    }


def cluster_stats(input_netcdf_path: Annotated[Path, Option()], output_csv_path: Annotated[Path, Option()]) -> None:
    cluster_image = MultiChannelImage.read_hdf5(input_netcdf_path)

    cluster_data = cluster_image.data_flat.values.ravel()
    cluster_coords = np.hstack(
        (
            cluster_image.data_flat.coords["x"].values.reshape(-1, 1),
            cluster_image.data_flat.coords["y"].values.reshape(-1, 1),
        )
    )

    metrics = compute_metrics(cluster_data, cluster_coords)

    metrics_df = pl.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    metrics_df.write_csv(output_csv_path)


if __name__ == "__main__":
    typer.run(cluster_stats)
