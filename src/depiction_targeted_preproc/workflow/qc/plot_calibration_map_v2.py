from pathlib import Path
from typing import Annotated

import typer
import xarray
from matplotlib import pyplot as plt
from typer import Option


def qc_plot_calibration_map_v2(
    input_mass_shifts: Annotated[Path, Option()],
    output_pdf: Annotated[Path, Option()],
) -> None:
    mass_shifts = xarray.open_dataarray(input_mass_shifts)

    fig, axs = plt.subplots(2, 1, figsize=(10, 20))

    # show the map
    mass_shifts.isel(c=0).plot.imshow(x="x", y="y", ax=axs[0], cmap="coolwarm", vmin=-0.5, vmax=+0.5, yincrease=False)
    axs[0].set_aspect("equal")
    test_mass = mass_shifts.coords["c"][0]
    axs[0].set_title(f"Computed shift for test mass {test_mass:.2f}")

    # show the histogram
    # TODO the clipping could be misleading (as it's not indicated)
    mass_shifts.isel(c=0).clip(-0.5, 0.5).plot.hist(ax=axs[1], bins=100, color="gray")
    axs[1].set_title("Histogram of computed shifts")

    plt.savefig(output_pdf, bbox_inches="tight")


if __name__ == "__main__":
    typer.run(qc_plot_calibration_map_v2)
