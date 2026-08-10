import cyclopts
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, colors
from pathlib import Path

from depiction.image import MultiChannelImage

app = cyclopts.App()


@app.default
def qc_plot_test_mass_shifts(
    input_mass_shifts: Path,
    output_pdf: Path,
) -> None:
    mass_shifts_img = MultiChannelImage.read_hdf5(input_mass_shifts)
    mass_shifts = mass_shifts_img.data_spatial

    # one row per test mass, so a shift affecting only one end of the mass range stays visible
    fig, axs = plt.subplots(mass_shifts_img.n_channels, 3, figsize=(30, 10 * mass_shifts_img.n_channels), squeeze=False)

    # one scale for every row, otherwise each map autoscales to its own shift and a row with a
    # negligible shift looks identical to one with a large shift -- which is the comparison the
    # per-test-mass rows exist to make
    vmax = np.percentile(np.abs(mass_shifts.values), 99)

    for i_mass, test_mass_label in enumerate(mass_shifts_img.channel_names):
        shifts = mass_shifts.isel(c=i_mass)
        ax_linear, ax_symlog, ax_hist = axs[i_mass]

        # show the map
        shifts.plot.imshow(x="x", y="y", ax=ax_linear, cmap="RdBu_r", vmin=-vmax, vmax=+vmax, yincrease=False)
        ax_linear.set_aspect("equal")
        ax_linear.set_title(f"Computed shift for test mass {test_mass_label} (linear)")

        # show a more qualitative map
        shifts.plot.imshow(
            x="x",
            y="y",
            ax=ax_symlog,
            cmap="RdBu_r",
            yincrease=False,
            norm=colors.SymLogNorm(linthresh=0.001, vmin=-1, vmax=1),
            interpolation="nearest",
        )
        # contour
        shifts.plot.contour(x="x", y="y", ax=ax_symlog, colors="black", yincrease=False, alpha=0.3)
        ax_symlog.set_aspect("equal")
        ax_symlog.set_title(f"Computed shift for test mass {test_mass_label} (symlog)")

        # show the histogram
        # TODO the clipping could be misleading (as it's not indicated)
        num_nans = np.sum(np.isnan(shifts).values)
        if num_nans:
            raise ValueError(f"nans detected for test mass {test_mass_label}")
        mz_min, mz_max = -0.5, 0.5
        sns.histplot(
            mass_shifts_img.data_flat.isel(c=i_mass).clip(mz_min, mz_max).values,
            bins=100,
            color="gray",
            ax=ax_hist,
            kde=True,
        )
        ax_hist.set_xlabel(r"$\Delta \frac{m}{z}$")
        ax_hist.set_title(f"Histogram of computed shifts for test mass {test_mass_label}")

    plt.savefig(output_pdf, bbox_inches="tight")


if __name__ == "__main__":
    app()
