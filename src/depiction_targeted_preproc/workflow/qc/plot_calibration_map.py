import cyclopts
import numpy as np
import polars as pl
import xarray
from pathlib import Path

from depiction.calibration.models import LinearModel
from depiction.visualize.visualize_mass_shift_map import VisualizeMassShiftMap


# TODO maybe rename to mass_shift_map as well
def get_mass_groups(mass_min: float, mass_max: float, n_bins: int = 3, add_group_number: bool = True) -> pl.DataFrame:
    edges = np.linspace(mass_min, mass_max, n_bins + 1)
    mz_max = edges[1:]
    mz_max[-1] += 0.1
    mass_groups = pl.DataFrame({"mz_min": edges[:-1], "mz_max": mz_max, "group_index": range(n_bins)})

    if add_group_number:
        mass_groups = mass_groups.with_columns(
            mass_group=pl.format(
                "group_{} = [{}, {})", pl.col("group_index"), pl.col("mz_min").round(3), pl.col("mz_max").round(3)
            )
        )
    else:
        mass_groups = mass_groups.with_columns(
            mass_group=pl.format("[{}, {})", pl.col("mz_min").round(3), pl.col("mz_max").round(3))
        )
    return mass_groups.sort("mz_min")


app = cyclopts.App()


@app.default
def qc_plot_calibration_map(calib_data: Path, mass_list: Path, output_pdf: Path) -> None:
    model_coefs = xarray.open_dataarray(calib_data, group="model_coefs")
    coords_2d = np.stack([model_coefs.y, model_coefs.x], axis=-1)
    models = xarray.apply_ufunc(
        LinearModel, model_coefs, vectorize=True, input_core_dims=[["c"]], output_core_dims=[[]]
    ).values

    # determine mass groups
    mass_list_df = pl.read_csv(mass_list)
    mass_min, mass_max = mass_list_df["mass"].min(), mass_list_df["mass"].max()
    test_masses = np.array([mass_min, (mass_min + mass_max) / 2, mass_max])

    vis_shifts = VisualizeMassShiftMap(models=models, coordinates=coords_2d)
    # TODO fix same_scale
    fig, axs = vis_shifts.plot_test_mass_maps_and_histograms(
        test_masses=test_masses, same_scale=True, int_limits=(-0.25, 0.25)
    )
    fig.savefig(output_pdf)
    fig.clear()


if __name__ == "__main__":
    app()
