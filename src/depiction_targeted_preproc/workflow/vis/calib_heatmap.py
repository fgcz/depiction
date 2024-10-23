import cyclopts
import h5py
import numpy as np
from pathlib import Path

from depiction.image.multi_channel_image import MultiChannelImage

app = cyclopts.App()


@app.default
def vis_calib_heatmap(
    input_calib_data_path: Path,
    output_hdf5_path: Path,
) -> None:
    with h5py.File(input_calib_data_path, "r") as file:
        coef_processed = file["coef_processed"][:]
        coordinates_2d = file["coordinates_2d"][:]

    # compute the mean across channels
    shifts_mean = np.nanmean(coef_processed, axis=1, keepdims=True)
    shifts_map = MultiChannelImage.from_sparse(values=shifts_mean, coordinates=coordinates_2d, channel_names=["mean"])

    # save the data
    shifts_map.write_hdf5(output_hdf5_path)


if __name__ == "__main__":
    app()
