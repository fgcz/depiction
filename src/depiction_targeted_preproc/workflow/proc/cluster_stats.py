from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from loguru import logger
from typer import Option

from depiction.image.multi_channel_image import MultiChannelImage
from depiction_targeted_preproc.workflow.proc.__cluster_stats import compute_CHAOS, compute_PAS


def cluster_stats(input_netcdf_path: Annotated[Path, Option()], output_csv_path: Annotated[Path, Option()]) -> None:
    cluster_image = MultiChannelImage.read_hdf5(input_netcdf_path)

    cluster_data = cluster_image.data_flat.values.ravel()
    cluster_coords = np.hstack(
        (
            cluster_image.data_flat.coords["x"].values.reshape(-1, 1),
            cluster_image.data_flat.coords["y"].values.reshape(-1, 1),
        )
    )

    chaos = compute_CHAOS(cluster_data, cluster_coords)
    logger.info(f"Computed CHAOS: {chaos}")

    pas = compute_PAS(cluster_data, cluster_coords)
    logger.info(f"Computed PAS: {pas}")

    metrics_df = pl.DataFrame({"metric": ["CHAOS", "PAS"], "value": [chaos, pas]})
    metrics_df.write_csv(output_csv_path)


if __name__ == "__main__":
    typer.run(cluster_stats)
