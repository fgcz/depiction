from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
import xarray as xr
from typer import Option

from depiction.tools.calibrate import get_calibration_instance
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters


def vis_test_mass_shifts(
    calib_hdf5_path: Annotated[Path, Option()],
    mass_list_path: Annotated[Path, Option()],
    config_path: Annotated[Path, Option()],
    output_hdf5_path: Annotated[Path, Option()],
) -> None:
    # load inputs
    model_coefs = xr.open_dataarray(calib_hdf5_path, group="model_coefs")
    config = PipelineParameters.parse_yaml(config_path)
    mass_list = pl.read_csv(mass_list_path)
    calibration = get_calibration_instance(config=config.calibration, mass_list=mass_list_path)

    # define test masses
    # to keep it simple for now only 1
    test_masses = np.array([(mass_list["mass"].max() + mass_list["mass"].min()) / 2])
    test_masses_int = np.ones_like(test_masses)

    # compute the shifts
    def compute_shifts(coef):
        result = calibration.apply_spectrum_model(
            spectrum_mz_arr=test_masses, spectrum_int_arr=test_masses_int, model_coef=xr.DataArray(coef, dims=["c"])
        )
        # result = test_mass - shift => shift = test_mass - result
        return xr.DataArray(test_masses - result[0], dims=["m"])

    shifts = xr.apply_ufunc(
        compute_shifts,
        model_coefs,
        input_core_dims=[["c"]],
        output_core_dims=[["m"]],
        vectorize=True,
    ).rename({"m": "c"})
    shifts = shifts.assign_coords(c=test_masses)

    shifts_2d = shifts.set_xindex(["x", "y"]).unstack("i")
    shifts_2d.attrs["bg_value"] = np.nan

    # save the result
    shifts_2d.to_netcdf(output_hdf5_path)


if __name__ == "__main__":
    typer.run(vis_test_mass_shifts)
