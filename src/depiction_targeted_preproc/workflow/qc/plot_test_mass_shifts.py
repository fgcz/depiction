import cyclopts
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, colors
from pathlib import Path

from depiction.image import MultiChannelImage

app = cyclopts.App()


@app.default
def qc_plot_calibration_map_v2(
    input_mass_shifts: Path,
    output_pdf: Path,
) -> None:
    mass_shifts_img = MultiChannelImage.read_hdf5(input_mass_shifts)
    mass_shifts = mass_shifts_img.data_spatial

    fig, axs = plt.subplots(3, 1, figsize=(10, 20))

    # show the map
    vmax = np.percentile(np.abs(mass_shifts.isel(c=0).values.ravel()), 0.99)
    mass_shifts.isel(c=0).plot.imshow(x="x", y="y", ax=axs[0], cmap="RdBu_r", vmin=-vmax, vmax=+vmax, yincrease=False)
    axs[0].set_aspect("equal")
    test_mass_label = mass_shifts_img.channel_names[0]
    axs[0].set_title(f"Computed shift for test mass {test_mass_label} (linear)")

    # show a more qualitative map
    mass_shifts.isel(c=0).plot.imshow(
        x="x",
        y="y",
        ax=axs[1],
        cmap="RdBu_r",
        yincrease=False,
        norm=colors.SymLogNorm(linthresh=0.001, vmin=-1, vmax=1),
        interpolation="nearest",
    )
    # contour
    mass_shifts.isel(c=0).plot.contour(x="x", y="y", ax=axs[1], colors="black", yincrease=False, alpha=0.3)
    axs[1].set_aspect("equal")
    axs[1].set_title(f"Computed shift for test mass {test_mass_label} (symlog)")

    # show the histogram
    # TODO the clipping could be misleading (as it's not indicated)
    num_nans = np.sum(np.isnan(mass_shifts.isel(c=0)).values)
    if num_nans:
        raise ValueError("nans detected")
    mz_min, mz_max = -0.5, 0.5
    sns.histplot(
        mass_shifts_img.data_flat.isel(c=0).clip(mz_min, mz_max).values, bins=100, color="gray", ax=axs[2], kde=True
    )
    axs[2].set_xlabel(r"$\Delta \frac{m}{z}$")
    axs[2].set_title("Histogram of computed shifts")

    plt.savefig(output_pdf, bbox_inches="tight")


if __name__ == "__main__":
    app()
