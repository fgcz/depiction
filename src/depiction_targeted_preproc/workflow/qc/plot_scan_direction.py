# TODO has been migrated/refactored in the new workflow folder
from pathlib import Path
from typing import Optional, Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from skimage.transform import resize_local_mean
from typer import Option

from depiction.persistence import ImzmlReadFile


def qc_plot_scan_direction(
    input_imzml_path: Annotated[Path, Option()],
    output_pdf: Annotated[Path, Option()],
) -> None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    imzml = ImzmlReadFile(input_imzml_path)
    visualize_coordinates_direction(
        coordinates=imzml.coordinates_2d,
        ax=ax,
        target_n_width=30,
        target_n_height=30,
        max_distance=2.0,
    )
    fig.savefig(output_pdf)


def visualize_coordinates_direction(
    coordinates: np.ndarray,
    ax: Optional[plt.Axes] = None,
    target_n_width: int = 30,
    target_n_height: int = 30,
    max_distance: float = 2.0,
):
    """
    Visualizes the scan direction based on the coordinates array.
    This approach is far from perfect having some important limitations to consider
    (and also be aware of it's discrepancies from matplotlib's streamplot).

    Limitations:
    - Uses means in the local neighborhood to determine the direction, i.e. there are cases where the image would
      be misleading (e.g. frequently alternating directions).
    - Borders are not handled correctly, since background is mixed in there.

    :param coordinates: (n, 2) or (n, 3) array of coordinates
    :param ax: matplotlib axis (or None if currently active axis should be used)
    :param target_n_width: number of points along the width of image to downsample the distance vectorfield to
    :param target_n_height: number of points along the height of image to downsample the distance vectorfield to
    :param max_distance: the maximum distance to consider for the vectorfield (before resizing)
    """
    ax = ax if ax is not None else plt.gca()

    coordinates = coordinates[:, :2] - coordinates.min(axis=0)[:2]
    domain = coordinates.max(axis=0)
    dir_vec = np.diff(coordinates, axis=0)
    dir_mat_2d = np.zeros((domain[0] + 1, domain[1] + 1, 2))
    dir_mat_2d[coordinates[:, 0], coordinates[:, 1], :] = np.concatenate([dir_vec, [[0, 0]]], axis=0)

    thresholded_dir_mat_2d = dir_mat_2d.copy()
    thresholded_dir_mat_2d[np.where(np.linalg.norm(dir_mat_2d, axis=2) > max_distance)] = 0

    thresholded_dir_img = resize_local_mean(thresholded_dir_mat_2d, (target_n_width, target_n_height), grid_mode=False)

    arrow_locations = np.argwhere(thresholded_dir_img.sum(axis=2) != 0)
    arrow_directions = thresholded_dir_img[arrow_locations[:, 0], arrow_locations[:, 1], :]

    ax.quiver(
        arrow_locations[:, 0],
        arrow_locations[:, 1],
        arrow_directions[:, 0],
        arrow_directions[:, 1],
        scale=1,
        units="xy",
    )
    ax.invert_yaxis()


if __name__ == "__main__":
    typer.run(qc_plot_scan_direction)
