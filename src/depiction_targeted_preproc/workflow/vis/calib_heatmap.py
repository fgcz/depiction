from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import typer

from depiction.image.multi_channel_image import MultiChannelImage


def vis_calib_heatmap(
    input_calib_data_path: Annotated[Path, typer.Option()],
    output_hdf5_path: Annotated[Path, typer.Option()],
) -> None:
    with h5py.File(input_calib_data_path, "r") as file:
        coef_processed = file["coef_processed"][:]
        coordinates_2d = file["coordinates_2d"][:]

    # compute the mean across channels
    shifts_mean = np.nanmean(coef_processed, axis=1, keepdims=True)
    shifts_map = MultiChannelImage.from_numpy_sparse(
        values=shifts_mean, coordinates=coordinates_2d, channel_names=["mean"]
    )

    # save the data
    shifts_map.write_hdf5(output_hdf5_path)


def main() -> None:
    typer.run(vis_calib_heatmap)


if __name__ == "__main__":
    main()