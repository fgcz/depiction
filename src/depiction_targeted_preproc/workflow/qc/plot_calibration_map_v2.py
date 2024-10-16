from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import xarray
from matplotlib import pyplot as plt, colors
from typer import Option


def qc_plot_calibration_map_v2(
    input_mass_shifts: Annotated[Path, Option()],
    output_pdf: Annotated[Path, Option()],
) -> None:
    mass_shifts = xarray.open_dataarray(input_mass_shifts)

    fig, axs = plt.subplots(3, 1, figsize=(10, 20))

    # show the map
    vmax = np.percentile(np.abs(mass_shifts.isel(c=0).values.ravel()), 0.99)
    mass_shifts.isel(c=0).plot.imshow(x="x", y="y", ax=axs[0], cmap="RdBu_r", vmin=-vmax, vmax=+vmax, yincrease=False)
    axs[0].set_aspect("equal")
    test_mass = mass_shifts.coords["c"][0]
    axs[0].set_title(f"Computed shift for test mass {test_mass:.2f} (linear)")

    # show a more qualitative map
    mass_shifts.isel(c=0).plot.imshow(
        x="x",
        y="y",
        ax=axs[1],
        cmap="RdBu_r",
        yincrease=False,
        norm=colors.SymLogNorm(linthresh=0.001, vmin=-1, vmax=1),
    )
    # contour
    mass_shifts.isel(c=0).plot.contour(x="x", y="y", ax=axs[1], colors="black", yincrease=False, alpha=0.3)
    axs[1].set_aspect("equal")
    axs[1].set_title(f"Computed shift for test mass {test_mass:.2f} (symlog)")

    # show the histogram
    # TODO the clipping could be misleading (as it's not indicated)
    num_nans = np.sum(np.isnan(mass_shifts.isel(c=0)).values)
    if num_nans:
        raise ValueError("nans detected")
    mass_shifts.isel(c=0).clip(-0.5, 0.5).plot.hist(ax=axs[2], bins=100, color="gray")
    axs[2].set_title("Histogram of computed shifts")

    plt.savefig(output_pdf, bbox_inches="tight")


if __name__ == "__main__":
    typer.run(qc_plot_calibration_map_v2)
